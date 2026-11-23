"""Static guard on every train.py before it runs. Karpathy's setup trusts the agent; we don't have to.

Rejects code that could read the eval data, fake the score, or reach outside the sandbox.
Anything rejected is reported back to the model like a crash (it gets the reason and can fix it).
The baseline train.py passes unchanged.
"""
import ast
import re

FORBIDDEN_SUBSTRINGS = {
    "parquet": "reading data files directly",
    "pyarrow": "reading data files directly",
    "DATA_DIR": "reading data files directly",
    "CACHE_DIR": "reading data files directly",
    "TOKENIZER_DIR": "reading tokenizer files directly",
    "shard_": "referencing data shards",
    "VAL_FILENAME": "referencing the eval shard",
    "EVAL_FILENAME": "referencing the eval shard",
    "TEST_": "referencing the held-out test shard",
    "_HELD_OUT": "referencing held-out data",
    "_document_batches": "using prepare's private data iterator",
    "AUTORESEARCH": "touching harness settings",
    "getenv": "reading environment variables",
    "environ.get": "reading environment variables",
    "environ.items": "reading environment variables",
    "environ.copy": "reading environment variables",
    "importlib": "dynamic imports",
    "sys.modules": "patching modules",
    "subprocess": "spawning processes",
    "socket": "network access",
    "requests": "network access",
    "urllib": "network access",
    "http.client": "network access",
    "setattr(": "patching objects",
    "__dict__": "patching objects",
    "builtins": "patching builtins",
    "exec(": "dynamic code execution",
    "eval(": "dynamic code execution",
}
ALLOWED_ENV_WRITES = {"PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "HF_HUB_DISABLE_PROGRESS_BARS",
                      "TOKENIZERS_PARALLELISM", "CUDA_LAUNCH_BLOCKING", "TORCH_LOGS"}


def check(code):
    """Return None if OK, else a short human-readable reason."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"
    # strip comments/strings-free view isn't needed: forbidden words in comments are rare and cheap to avoid
    for s, why in FORBIDDEN_SUBSTRINGS.items():
        if s in code:
            if s == "eval(" and re.search(r"(?<![\w.])eval\(", code) is None:
                continue  # model.eval() is fine
            if s == "exec(" and re.search(r"(?<![\w.])exec\(", code) is None:
                continue
            return f"forbidden: '{s}' ({why})"

    imports_prepare = False
    eval_calls = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "prepare":
            imports_prepare = True
            for a in node.names:
                if a.name == "*":
                    return "forbidden: 'from prepare import *'"
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "prepare":
                    imports_prepare = True
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else None
            if name == "evaluate_bpb":
                eval_calls += 1
            if name == "make_dataloader":
                split = node.args[3] if len(node.args) > 3 else next((k.value for k in node.keywords if k.arg == "split"), None)
                if not (isinstance(split, ast.Constant) and split.value == "train"):
                    return "forbidden: make_dataloader must use the literal split \"train\""
        # assignments to prepare.<anything> or to evaluate_bpb / make_dataloader names
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                for sub in ast.walk(t):
                    if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id == "prepare":
                        return "forbidden: assigning to prepare.<attr>"
                    if isinstance(sub, ast.Name) and sub.id in ("evaluate_bpb", "make_dataloader", "Tokenizer", "TIME_BUDGET", "MAX_SEQ_LEN"):
                        return f"forbidden: reassigning {sub.id}"
                    if (isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Attribute) and sub.value.attr == "environ"):
                        key = sub.slice.value if isinstance(sub.slice, ast.Constant) else None
                        if key not in ALLOWED_ENV_WRITES:
                            return f"forbidden: setting environment variable {key!r}"
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "environ" \
                and isinstance(node.ctx, ast.Load):
            return "forbidden: reading environment variables"
    if not imports_prepare:
        return "train.py must import from prepare"
    if eval_calls != 1:
        return f"train.py must call evaluate_bpb exactly once (found {eval_calls})"
    return None
