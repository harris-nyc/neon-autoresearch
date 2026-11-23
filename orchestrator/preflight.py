"""Pre-flight. Run once on the GPU box after data_setup, before the overnight run.

    python -m orchestrator.preflight                 # ~35 min on an H100: checks + 5-seed noise floor
    python -m orchestrator.preflight --seeds 3       # quicker

1. Leakage probe: a train.py that hunts for the held-out test shard, API keys and the network
   (run with the guard bypassed, to test the sandbox itself). All must come up empty.
2. Guard: a train.py that trains on the val split must be rejected before it runs.
3. Timeout: a train.py that never finishes must be killed.
4. Noise floor: Karpathy's baseline with N seeds -> mean/std of val_bpb. epsilon = std.
Writes runs/_preflight/noise.json and report.json.
"""
import argparse, json, os
import numpy as np

from . import lint, sandbox

ROOT = sandbox.ROOT
PRE = os.path.join(ROOT, "runs", "_preflight")

LEAK_PROBE = r'''
import os, socket
from prepare import evaluate_bpb
found = []
for top in ["/", os.path.expanduser("~"), "/tmp", "/home", "/root", "/mnt"]:
    for dirpath, dirs, files in os.walk(top):
        if dirpath.count(os.sep) > 7 or dirpath.startswith(("/proc", "/sys", "/dev", "/usr", "/lib", "/snap", "/var/lib")):
            dirs[:] = []
            continue
        for f in files:
            if f.startswith("shard_06541") or f in (".env",):
                found.append(os.path.join(dirpath, f))
keys = [k for k in os.environ if any(s in k for s in ("KEY", "TOKEN", "SECRET", "PROXY"))]
try:
    socket.create_connection(("8.8.8.8", 53), timeout=3); net = True
except OSError:
    net = False
print("LEAK_REPORT", {"files": found, "env_keys": keys, "network": net})
'''

SLEEPER = "import time\nfrom prepare import evaluate_bpb\ntime.sleep(36000)\nevaluate_bpb(None, None, 1)\n"


def run_unlinted(code, workdir, mode, **kw):
    real = lint.check
    lint.check = lambda c: None  # the probe deliberately breaks the rules; we're testing the sandbox
    try:
        return sandbox.run(code, workdir, mode=mode, **kw)
    finally:
        lint.check = real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sandbox", default=os.environ.get("AUTORESEARCH_SANDBOX", "local"))
    args = ap.parse_args()
    os.makedirs(PRE, exist_ok=True)
    rep = {"prepare_sha256": sandbox.check_prepare()}

    r = run_unlinted(LEAK_PROBE, os.path.join(PRE, "leak"), args.sandbox, budget=60, grace=600)
    line = next((l for l in r["log_tail"].splitlines() if l.startswith("LEAK_REPORT")), None)
    if line is None:
        raise SystemExit(f"leak probe did not run:\n{r['log_tail']}")
    leak = eval(line[len("LEAK_REPORT "):])  # noqa: S307  our own probe's output
    ok = not leak["files"] and not leak["env_keys"] and not leak["network"]
    rep["leakage"] = {"pass": ok, **leak}
    print(f"leakage test: {'PASS' if ok else 'FAIL'} {leak}")

    base = open(sandbox.BASELINE).read()
    cheat = base.replace('MAX_SEQ_LEN, "train")', 'MAX_SEQ_LEN, "val")', 1)
    r = sandbox.run(cheat, os.path.join(PRE, "guard"), mode=args.sandbox)
    rep["guard"] = {"pass": r["status"] == "rejected" and cheat != base, "status": r["status"]}
    print(f"guard test (train-on-val must be rejected): {'PASS' if rep['guard']['pass'] else 'FAIL'} ({r['status']})")

    r = run_unlinted(SLEEPER, os.path.join(PRE, "timeout"), args.sandbox, budget=5, grace=10)
    rep["timeout"] = {"pass": r["status"] == "timeout", "status": r["status"], "wall_s": round(r["wall_s"], 1)}
    print(f"timeout test: {'PASS' if rep['timeout']['pass'] else 'FAIL'} ({r['status']} after {r['wall_s']:.0f}s)")
    if not (rep["leakage"]["pass"] and rep["guard"]["pass"] and rep["timeout"]["pass"]):
        json.dump(rep, open(os.path.join(PRE, "report.json"), "w"), indent=2)
        raise SystemExit("PREFLIGHT FAILED (see above)")

    vals = []
    for s in range(args.seeds):
        r = sandbox.run(base, os.path.join(PRE, f"seed{s}"), seed=42 + s, mode=args.sandbox)
        if r["val_bpb"] is None:
            raise SystemExit(f"baseline seed {s} failed ({r['status']}):\n{r['log_tail']}")
        vals.append(r["val_bpb"])
        print(f"baseline seed {42 + s}: val_bpb {r['val_bpb']:.6f}  (train {r['train_seconds']}s, "
              f"peak VRAM {r['peak_vram_mb']} MB, total {r['wall_s']:.0f}s)", flush=True)
    eps = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    noise = {"seeds": vals, "mean": float(np.mean(vals)), "std": eps, "epsilon": eps}
    json.dump(noise, open(os.path.join(PRE, "noise.json"), "w"), indent=2)
    rep["noise"] = noise
    json.dump(rep, open(os.path.join(PRE, "report.json"), "w"), indent=2)
    print(f"noise floor: mean val_bpb {noise['mean']:.6f}, std {eps:.6f} -> keep threshold {eps:.6f}")
    print("PREFLIGHT OK")


if __name__ == "__main__":
    main()
