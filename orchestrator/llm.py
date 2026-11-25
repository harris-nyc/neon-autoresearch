"""Thin, identical-interface adapters. One call = (stable prefix, variable suffix) -> text + usage.

The prefix (program.md + task contract + baseline train.py) never changes within a run, so it is
cached: implicitly on Gemini (prefix caching, it just has to come first), explicitly on Anthropic
(cache_control on the system block).
"""
import os, random, re, time
from dataclasses import dataclass, field

MAX_RESPONSE_CHARS = 200_000


@dataclass
class LLMResult:
    text: str
    usage: dict = field(default_factory=dict)  # tokens_in, tokens_cached, tokens_cache_write, tokens_out_visible, tokens_thinking, tokens_out_total
    latency_s: float = 0.0       # successful call only
    retry_wait_s: float = 0.0    # time lost to 429/5xx retries (counts against wall-clock, logged separately)
    retries: int = 0
    thoughts: str = ""          # reasoning summaries the API returns (for the write-up; never fed back)


def _with_retries(fn, is_retryable, max_tries=10):
    waited, tries = 0.0, 0
    while True:
        t0 = time.time()
        try:
            out = fn()
            return out, time.time() - t0, waited, tries
        except Exception as e:  # noqa: BLE001
            if not is_retryable(e) or tries + 1 >= max_tries:
                raise
            tries += 1
            sleep = min(120, 2 ** tries) * (0.5 + random.random())
            print(f"[llm] retryable error ({type(e).__name__}: {str(e)[:120]}), sleeping {sleep:.0f}s", flush=True)
            time.sleep(sleep)
            waited += time.time() - t0


class Gemini:
    def __init__(self, model="gemini-4-neon", thinking_level=None):
        from google import genai
        from google.genai import types
        self.types = types
        self.model, self.thinking_level = model, thinking_level
        if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true"):
            # Vertex: set GOOGLE_CLOUD_PROJECT; use the global endpoint (regional costs +10%)
            self.client = genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"], location="global")
        else:
            self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def __call__(self, prefix, suffix):
        from google.genai import errors
        t = self.types
        # No temperature/top_p/top_k: Neon's migration notes say to leave them unset.
        # thinking_level=None -> model default (high for Neon). include_thoughts only returns
        # summaries of the reasoning so we can show it later; it doesn't change how much it thinks.
        tc = {"include_thoughts": True}
        if self.thinking_level:
            tc["thinking_level"] = self.thinking_level
        cfg = t.GenerateContentConfig(system_instruction=prefix, thinking_config=t.ThinkingConfig(**tc))
        call = lambda: self.client.models.generate_content(model=self.model, contents=suffix, config=cfg)
        retryable = lambda e: isinstance(e, errors.APIError) and (e.code == 429 or (e.code or 0) >= 500)
        resp, lat, waited, tries = _with_retries(call, retryable)
        u = resp.usage_metadata
        prompt = u.prompt_token_count or 0
        cached = u.cached_content_token_count or 0
        vis = u.candidates_token_count or 0
        think = u.thoughts_token_count or 0
        usage = {"tokens_in": prompt - cached, "tokens_cached": cached, "tokens_cache_write": 0,
                 "tokens_out_visible": vis, "tokens_thinking": think, "tokens_out_total": vis + think}
        thoughts = ""
        try:
            parts = resp.candidates[0].content.parts or []
            thoughts = "\n".join(p.text for p in parts if getattr(p, "thought", False) and p.text)
        except (AttributeError, IndexError, TypeError):
            pass
        return LLMResult(resp.text or "", usage, lat, waited, tries, thoughts)


class Anthropic:
    def __init__(self, model="claude-opus-5-5", effort=None, max_tokens=64000):
        import anthropic
        self.anthropic = anthropic
        self.client = anthropic.Anthropic()  # ANTHROPIC_API_KEY
        self.model, self.effort, self.max_tokens = model, effort, max_tokens

    def __call__(self, prefix, suffix):
        a = self.anthropic

        def call():
            # Adaptive thinking is always on for Opus 5.5. effort=None -> omit it and use the
            # model's own default (medium).
            extra = {"output_config": {"effort": self.effort}} if self.effort else {}
            with self.client.messages.stream(
                model=self.model, max_tokens=self.max_tokens,
                system=[{"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": suffix}], **extra,
            ) as s:
                return s.get_final_message()

        retryable = lambda e: isinstance(e, (a.RateLimitError, a.InternalServerError, a.APIConnectionError)) or (
            isinstance(e, a.APIStatusError) and e.status_code in (429, 500, 502, 503, 504, 529))
        msg, lat, waited, tries = _with_retries(call, retryable)
        u = msg.usage
        text = "".join(b.text for b in msg.content if b.type == "text")
        thoughts = "\n".join(getattr(b, "thinking", "") or "" for b in msg.content if b.type == "thinking")
        out = u.output_tokens or 0
        usage = {"tokens_in": u.input_tokens or 0, "tokens_cached": u.cache_read_input_tokens or 0,
                 "tokens_cache_write": u.cache_creation_input_tokens or 0,
                 "tokens_out_visible": None, "tokens_thinking": None,  # Anthropic bills thinking inside output_tokens
                 "tokens_out_total": out}
        return LLMResult(text, usage, lat, waited, tries, thoughts)


class Mock:
    """Offline stand-in for testing the loop. Mutates one hyperparameter of the incumbent;
    sometimes ships a bug so the crash-fix path gets exercised. Not a real model."""
    KNOBS = {"DEPTH": [1, 2, 3], "MODEL_DIM": [32, 64, 96], "MATRIX_LR": [0.001, 0.003, 0.006, 0.01],
             "EMBEDDING_LR": [0.001, 0.003, 0.01], "WEIGHT_DECAY": [0.0, 0.01, 0.1], "WARMDOWN_RATIO": [0.2, 0.5, 0.8],
             "DEVICE_BATCH_SIZE": [16, 32, 64]}

    def __init__(self, seed=0, bug_rate=0.15, **_):
        self.rng = random.Random(seed)
        self.bug_rate = bug_rate
        self.pending = None

    def __call__(self, prefix, suffix):
        time.sleep(0.2)
        inc = re.search(r"<incumbent_train_py>\n(.*?)\n</incumbent_train_py>", suffix, re.S).group(1)
        if "FIX REQUEST" in suffix and self.pending:
            hyp, code = self.pending
            self.pending = None
            return self._wrap(hyp + " (fixed)", code)
        knob = self.rng.choice(list(self.KNOBS))
        cur = re.search(rf"^{knob} = ([^#\n]+?)\s*(#.*)?$", inc, re.M).group(1)
        choices = [v for v in self.KNOBS[knob] if repr(v) != cur and str(v) != cur]
        val = self.rng.choice(choices)
        code = re.sub(rf"^{knob} = [^#\n]+?(\s*#.*)?$", lambda mm: f"{knob} = {val!r}" + (mm.group(1) or ""), inc, count=1, flags=re.M)
        hyp = f"set {knob} {cur} -> {val!r}"
        if self.rng.random() < self.bug_rate:
            self.pending = (hyp, code)
            code = code.replace("opt.step()", "opt.step(); undefined_name += 1", 1)
        return self._wrap(hyp, code)

    @staticmethod
    def _wrap(hyp, code):
        n = len(code) // 4
        return LLMResult(f"<hypothesis>{hyp}</hypothesis>\n<train_py>\n{code}\n</train_py>",
                         {"tokens_in": 500, "tokens_cached": 4000, "tokens_cache_write": 0,
                          "tokens_out_visible": n, "tokens_thinking": 0, "tokens_out_total": n}, 0.2, 0.0, 0)


def make(provider, model=None, thinking=None, seed=0):
    if provider == "gemini":
        return Gemini(model or "gemini-4-neon", thinking)
    if provider == "anthropic":
        return Anthropic(model or "claude-opus-5-5", thinking)
    if provider == "mock":
        return Mock(seed=seed)
    raise ValueError(provider)


def parse(text):
    """-> (hypothesis, train_py) or raises ValueError. Tag format instead of JSON so neither model
    has to JSON-escape a 300-line Python file (escaping errors would be harness noise, not research)."""
    if len(text) > MAX_RESPONSE_CHARS:
        raise ValueError(f"response too large ({len(text)} chars)")
    h = re.search(r"<hypothesis>(.*?)</hypothesis>", text, re.S)
    c = re.search(r"<train_py>\s*\n?(.*?)\n?</train_py>", text, re.S)
    if not h or not c:
        raise ValueError("missing <hypothesis> or <train_py> tags")
    code = c.group(1)
    code = re.sub(r"^```(?:python)?\s*\n", "", code.strip("\n"))
    code = re.sub(r"\n```\s*$", "", code)
    hyp = " ".join(h.group(1).split())[:300]
    return hyp, code + "\n"


# What "provider default" resolves to, for logs and the dashboard.
DEFAULT_REASONING = {"gemini-4-neon": "default (high)", "claude-opus-5-5": "default (medium)",
                     "claude-fable-5-2": "default (high)", "mock": "default"}
DEFAULT_MODEL = {"gemini": "gemini-4-neon", "anthropic": "claude-opus-5-5", "mock": "mock"}
