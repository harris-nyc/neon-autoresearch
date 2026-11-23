"""Final test on the held-out shard. Run once, after both arms stop. Nothing else ever sees this shard.

    python -m orchestrator.final_test --arms neon opus --seeds 3
    python -m orchestrator.final_test --arms neon opus --seeds 3 --gpus 0 1    # split across both GPUs

For Karpathy's baseline and each arm's best train.py: train N seeds (5 min each), evaluate on test
shard 06541 -> test_bpb mean +- std. Also reports each arm's best val_bpb (val->test gap) and whether
the gain beats 2x the seed std. Writes runs/_final/results.json.
"""
import argparse, json, os, threading
import numpy as np

from . import sandbox

ROOT = sandbox.ROOT


def test_one(name, code, seeds, mode, gpu, out):
    vals = []
    for s in seeds:
        env_gpu = {"CUDA_VISIBLE_DEVICES": str(gpu)} if gpu is not None else {}
        old = {k: os.environ.get(k) for k in env_gpu}
        os.environ.update(env_gpu)
        r = sandbox.run(code, os.path.join(ROOT, "runs", "_final", name, f"seed{s}"), phase="final", seed=s, mode=mode)
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        print(f"{name} seed {s}: {r['status']} test_bpb {r['val_bpb']}", flush=True)
        vals.append(r["val_bpb"])
    ok = [v for v in vals if v is not None]
    out[name] = {"per_seed": vals, "failed_seeds": len(vals) - len(ok),
                 "test_bpb": {"mean": float(np.mean(ok)), "std": float(np.std(ok, ddof=1)) if len(ok) > 1 else 0.0} if ok else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--gpus", nargs="*", type=int, default=None, help="run jobs in parallel, one queue per GPU")
    ap.add_argument("--sandbox", default=os.environ.get("AUTORESEARCH_SANDBOX", "local"))
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") and args.gpus:
        raise SystemExit("unset CUDA_VISIBLE_DEVICES when using --gpus")
    jobs = [("baseline", open(sandbox.BASELINE).read())]
    for arm in args.arms:
        jobs.append((arm, open(os.path.join(ROOT, "runs", arm, "best_train.py")).read()))
    seeds = [42 + i for i in range(args.seeds)]
    results = {}
    if args.gpus:
        queues = {g: [] for g in args.gpus}
        for i, j in enumerate(jobs):
            queues[args.gpus[i % len(args.gpus)]].append(j)
        def worker(g):
            for name, code in queues[g]:
                test_one(name, code, seeds, args.sandbox, g, results)
        ts = [threading.Thread(target=worker, args=(g,)) for g in args.gpus]
        [t.start() for t in ts]; [t.join() for t in ts]
    else:
        for name, code in jobs:
            test_one(name, code, seeds, args.sandbox, None, results)

    base = results["baseline"]["test_bpb"]
    for arm in args.arms:
        r = results[arm]
        st = json.load(open(os.path.join(ROOT, "runs", arm, "state.json")))
        r["best_val_bpb"] = st["best"]
        r["experiments"] = st["next_id"] - 1
        if r["test_bpb"] and base:
            gain = base["mean"] - r["test_bpb"]["mean"]  # positive = better than baseline
            r["test_gain_vs_baseline"] = gain
            r["test_gain_pct"] = 100 * gain / base["mean"]
            r["beats_2x_seed_std"] = bool(gain > 2 * max(base["std"], r["test_bpb"]["std"]))
            r["val_test_gap"] = r["test_bpb"]["mean"] - st["best"]
    os.makedirs(os.path.join(ROOT, "runs", "_final"), exist_ok=True)
    json.dump(results, open(os.path.join(ROOT, "runs", "_final", "results.json"), "w"), indent=2)
    print(json.dumps({k: {kk: v.get(kk) for kk in ("test_bpb", "test_gain_pct", "beats_2x_seed_std")} for k, v in results.items()}, indent=1))


if __name__ == "__main__":
    main()
