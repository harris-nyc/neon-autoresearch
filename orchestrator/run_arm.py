"""Main loop for one arm (one model). Resumable: re-run the same command to continue.

    CUDA_VISIBLE_DEVICES=0 python -m orchestrator.run_arm --arm neon --provider gemini      # Neon default (high)
    CUDA_VISIBLE_DEVICES=1 python -m orchestrator.run_arm --arm opus --provider anthropic   # Opus default (medium)
    AUTORESEARCH_SMOKE=1 python -m orchestrator.run_arm --arm test --provider mock --max-exps 10   # CPU smoke test

The orchestrator owns all state: the model never decides keep/discard, never sees labels,
never controls timing.
"""
import argparse, csv, datetime as dt, fcntl, json, os, subprocess, time

from . import llm, pricing, sandbox

ROOT = sandbox.ROOT
COLUMNS = ["exp_id", "arm", "model", "thinking_setting", "started_at", "hypothesis", "status",
           "val_bpb", "delta_vs_incumbent", "train_seconds", "peak_vram_mb",
           "sandbox_wall_s", "gpu_wait_s", "llm_latency_s", "llm_retry_wait_s", "retries", "fix_attempts",
           "tokens_in", "tokens_cached", "tokens_cache_write", "tokens_out_visible", "tokens_thinking",
           "cost_usd", "cum_cost_usd", "elapsed_s", "git_sha"]

CONTRACT = """## Harness facts
- Training budget: {budget}s (TIME_BUDGET in prepare.py). Whole process is killed at {kill}s.
- Noise floor: the baseline's val_bpb varies by {eps:.6f} (1 std) across seeds. A change is kept
  only if it lowers val_bpb by more than that: new < best - {eps:.6f}.
- Before running, train.py is checked: it must import from prepare, call evaluate_bpb exactly once,
  load training data only via make_dataloader(..., "train"), and not touch data files, the
  network, environment variables or prepare's internals. Violations are returned to you as errors.
"""


def git(arm_dir, *args):
    return subprocess.run(["git", "-C", arm_dir, *args], capture_output=True, text=True, check=True).stdout.strip()


def git_init(arm_dir):
    if not os.path.isdir(os.path.join(arm_dir, ".git")):
        subprocess.run(["git", "init", "-q", arm_dir], check=True)
        git(arm_dir, "config", "user.email", "orchestrator@recsys-autoresearch")
        git(arm_dir, "config", "user.name", "orchestrator")


def git_commit(arm_dir, code, message):
    with open(os.path.join(arm_dir, "train.py"), "w") as f:
        f.write(code)
    git(arm_dir, "add", "train.py")
    git(arm_dir, "commit", "-q", "--allow-empty", "-m", message)
    return git(arm_dir, "rev-parse", "--short", "HEAD")


def history_table(rows):
    lines = ["id | status | val_bpb | delta_vs_best (negative = better) | train_s | hypothesis"]
    for r in rows:
        nd = f"{float(r['val_bpb']):.6f}" if r["val_bpb"] not in ("", None) else "-"
        de = f"{float(r['delta_vs_incumbent']):+.6f}" if r["delta_vs_incumbent"] not in ("", None) else "-"
        ts = f"{float(r['train_seconds']):.0f}" if r["train_seconds"] not in ("", None) else "-"
        lines.append(f"{r['exp_id']} | {r['status']} | {nd} | {de} | {ts} | {r['hypothesis']}")
    return "\n".join(lines)


def build_suffix(rows, incumbent_code, best, last):
    parts = [f"## Current best val_bpb: {best:.6f} (lower is better)",
             "## History of all experiments so far\n" + history_table(rows)]
    if last:
        parts.append(f"## Last experiment ({last['exp_id']}): {last['status']}\n"
                     f"Last 60 lines of its log:\n```\n{last['log_tail']}\n```")
    parts.append(f"## Current best train.py (start from this)\n<incumbent_train_py>\n{incumbent_code.rstrip()}\n</incumbent_train_py>")
    parts.append("Propose your next experiment.")
    return "\n\n".join(parts)


def fix_suffix(base_suffix, broken_code, error_text):
    return (base_suffix + "\n\n## FIX REQUEST\nYour previous train.py failed before producing scores. "
            f"Error / last log lines:\n```\n{error_text}\n```\nYour train.py was:\n<broken_train_py>\n"
            f"{broken_code.rstrip()}\n</broken_train_py>\nReturn a corrected complete train.py (same hypothesis, "
            "same reply format).")


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t", extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def run_experiment(arm_dir, exp_id, code, mode):
    exp_dir = os.path.join(arm_dir, "exps", f"{exp_id:04d}")
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "train.py"), "w") as f:
        f.write(code)
    # One GPU shared by both arms: take turns. While one arm trains, the other is free to think.
    lock_path = os.path.join(ROOT, ".gpu.lock")
    t_wait = time.time()
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        waited = time.time() - t_wait
        try:
            res = sandbox.run(code, os.path.join(exp_dir, "work"), phase="search", mode=mode)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    res["gpu_wait_s"] = waited
    log = os.path.join(exp_dir, "work", "log.txt")
    if os.path.exists(log):
        os.replace(log, os.path.join(exp_dir, "log.txt"))
    subprocess.run(["rm", "-rf", os.path.join(exp_dir, "work")])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--provider", required=True, choices=["gemini", "anthropic", "mock"])
    ap.add_argument("--model")
    ap.add_argument("--thinking", default=None,
                    help="override gemini thinking_level / anthropic effort; omit to use each model's own default")
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--max-usd", type=float, default=40.0)
    ap.add_argument("--max-exps", type=int, default=10_000)
    ap.add_argument("--epsilon", type=float, help="default: runs/_preflight/noise.json")
    ap.add_argument("--sandbox", default=os.environ.get("RECSYS_SANDBOX", "local"))
    args = ap.parse_args()

    arm_dir = os.path.join(ROOT, "runs", args.arm)
    os.makedirs(arm_dir, exist_ok=True)
    git_init(arm_dir)
    log_path = os.path.join(arm_dir, "log.tsv")
    state_path = os.path.join(arm_dir, "state.json")

    if args.epsilon is None:
        noise = os.path.join(ROOT, "runs", "_preflight", "noise.json")
        if not os.path.exists(noise):
            raise SystemExit("run `python -m orchestrator.preflight` first (or pass --epsilon)")
        args.epsilon = json.load(open(noise))["epsilon"]
    if args.provider == "mock" and not sandbox.SMOKE:
        raise SystemExit("the mock provider is for smoke tests: set AUTORESEARCH_SMOKE=1")
    # per-arm compile caches (speeds up torch.compile across experiments; not visible to other arms)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(ROOT, ".compile_cache", args.arm, "inductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(ROOT, ".compile_cache", args.arm, "triton"))

    client = llm.make(args.provider, args.model, args.thinking, seed=sum(map(ord, args.arm)))
    model_name = args.model or llm.DEFAULT_MODEL[args.provider]
    baseline = open(sandbox.BASELINE).read()
    prefix = "\n\n".join([open(os.path.join(ROOT, "program.md")).read(),
                          CONTRACT.format(budget=sandbox.TIME_BUDGET, kill=sandbox.TIME_BUDGET + sandbox.GRACE_S, eps=args.epsilon),
                          "## prepare.py (fixed, read-only)\n```python\n" + open(sandbox.PREPARE).read() + "```",
                          "## Baseline train.py (experiment 0, Karpathy's)\n```python\n" + baseline + "```"])

    state = json.load(open(state_path)) if os.path.exists(state_path) else None
    common = {"arm": args.arm, "model": model_name,
              "thinking_setting": args.thinking or llm.DEFAULT_REASONING.get(model_name, "default")}

    if state is None:  # experiment 0: baseline, seed 0, sets the incumbent
        t0 = time.time()
        started = dt.datetime.now().isoformat(timespec="seconds")
        res = run_experiment(arm_dir, 0, baseline, args.sandbox)
        if res["val_bpb"] is None:
            raise SystemExit(f"baseline failed ({res['status']}):\n{res['log_tail']}")
        sha = git_commit(arm_dir, baseline, "baseline: Karpathy's train.py")
        state = {"incumbent_code": baseline, "best": res["val_bpb"], "next_id": 1, "cum_cost": 0.0,
                 "elapsed_s": time.time() - t0, "last": None}
        append_row(log_path, {**common, "exp_id": 0, "started_at": started, "hypothesis": "baseline (Karpathy's train.py)",
                              "status": "baseline", "val_bpb": res["val_bpb"], "delta_vs_incumbent": 0.0,
                              "train_seconds": res["train_seconds"], "peak_vram_mb": res["peak_vram_mb"],
                              "sandbox_wall_s": res["wall_s"], "cost_usd": 0.0, "cum_cost_usd": 0.0,
                              "elapsed_s": state["elapsed_s"], "git_sha": sha})
        json.dump(state, open(state_path, "w"))
        print(f"[{args.arm}] baseline val_bpb = {res['val_bpb']:.6f}", flush=True)

    while True:
        if state["elapsed_s"] >= args.hours * 3600:
            print(f"[{args.arm}] wall-clock budget reached"); break
        if state["cum_cost"] >= args.max_usd:
            print(f"[{args.arm}] $ cap reached"); break
        if state["next_id"] > args.max_exps:
            print(f"[{args.arm}] max experiments reached"); break

        exp_id = state["next_id"]
        t0 = time.time()
        started = dt.datetime.now().isoformat(timespec="seconds")
        rows = read_rows(log_path)
        suffix = build_suffix(rows, state["incumbent_code"], state["best"], state["last"])
        usage_tot = {k: 0 for k in ("tokens_in", "tokens_cached", "tokens_cache_write", "tokens_out_visible", "tokens_thinking", "tokens_out_total")}
        lat = wait = 0.0
        retries = fixes = 0
        cost = 0.0
        hyp, code, res = "(no valid response)", None, None
        exp_dir = os.path.join(arm_dir, "exps", f"{exp_id:04d}")
        os.makedirs(exp_dir, exist_ok=True)

        prompt = suffix
        for attempt in range(3):  # 1 proposal + up to 2 fix attempts
            try:
                r = client(prefix, prompt)
            except Exception as e:  # noqa: BLE001  non-retryable API failure: log it as a crash, move on
                res = {"status": "crash", "val_bpb": None, "log_tail": f"[harness] API error: {e}", "train_seconds": None, "wall_s": 0}
                break
            for k in usage_tot:
                usage_tot[k] += r.usage.get(k) or 0
            lat += r.latency_s; wait += r.retry_wait_s; retries += r.retries
            cost += pricing.cost(model_name, r.usage)
            with open(os.path.join(exp_dir, f"response_{attempt}.txt"), "w") as f:
                f.write(r.text)
            if r.thoughts:
                with open(os.path.join(exp_dir, f"reasoning_{attempt}.txt"), "w") as f:
                    f.write(r.thoughts)
            try:
                hyp_i, code = llm.parse(r.text)
                if attempt == 0:
                    hyp = hyp_i
            except ValueError as e:
                res = {"status": "crash", "val_bpb": None, "log_tail": f"[harness] invalid response: {e}", "train_seconds": None, "wall_s": 0}
                code = code or ""
                if attempt < 2:
                    fixes += 1
                    prompt = fix_suffix(suffix, r.text[:4000], str(e))
                    continue
                break
            res = run_experiment(arm_dir, exp_id, code, args.sandbox)
            if res["status"] in ("crash", "rejected") and attempt < 2:
                fixes += 1
                prompt = fix_suffix(suffix, code, res["log_tail"])
                continue
            break

        status = res["status"]
        delta = None
        sha = ""
        v = res.get("val_bpb")
        if v is not None:
            delta = v - state["best"]
            if v < state["best"] - args.epsilon:  # lower is better
                status = "keep"
                state["best"], state["incumbent_code"] = v, code
                sha = git_commit(arm_dir, code, f"exp {exp_id}: {hyp} ({v:.6f}, {delta:+.6f})")
            else:
                status = "discard"
        state["cum_cost"] += cost
        state["elapsed_s"] += time.time() - t0
        state["last"] = {"exp_id": exp_id, "status": status, "log_tail": res["log_tail"]}
        state["next_id"] = exp_id + 1
        append_row(log_path, {**common, "exp_id": exp_id, "started_at": started, "hypothesis": hyp, "status": status,
                              "val_bpb": "" if v is None else v,
                              "delta_vs_incumbent": "" if delta is None else delta,
                              "train_seconds": res.get("train_seconds") or "", "peak_vram_mb": res.get("peak_vram_mb") or "",
                              "sandbox_wall_s": res.get("wall_s", 0), "gpu_wait_s": round(res.get("gpu_wait_s", 0), 1),
                              "llm_latency_s": round(lat, 2), "llm_retry_wait_s": round(wait, 2), "retries": retries,
                              "fix_attempts": fixes, **{k: v for k, v in usage_tot.items() if k != "tokens_out_total"},
                              "cost_usd": round(cost, 5), "cum_cost_usd": round(state["cum_cost"], 5),
                              "elapsed_s": round(state["elapsed_s"], 1), "git_sha": sha})
        with open(os.path.join(arm_dir, "best_train.py"), "w") as f:
            f.write(state["incumbent_code"])
        json.dump(state, open(state_path, "w"))
        nd = "-" if v is None else f"{v:.6f}"
        print(f"[{args.arm}] exp {exp_id} {status:8s} {nd}  best {state['best']:.6f}  ${state['cum_cost']:.2f}  "
              f"{state['elapsed_s']/3600:.2f}h  | {hyp}", flush=True)


if __name__ == "__main__":
    main()
