"""Build the race dashboard: one self-contained HTML file (no network, data embedded).

    python -m analysis.build_dashboard --arms "neon:Gemini 4 Neon" "opus:Claude Opus 5.5"
    python -m analysis.build_dashboard --arms neon opus --watch 60    # rebuild every 60s during the run

Open runs/dashboard.html. Press Replay to animate the whole night (good for screen recording).
"""
import argparse, csv, json, os, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLORS = ["#3D8BF2", "#D9712F"]  # validated on #0D1117: lightness band, CVD dE >= 27, contrast >= 3:1. First arm = the star.
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_template.html")


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_arm(arm_id, label, color):
    rows = list(csv.DictReader(open(os.path.join(ROOT, "runs", arm_id, "log.tsv")), delimiter="\t"))
    out = []
    for r in rows:
        out.append({"id": int(r["exp_id"]), "t": (f(r["elapsed_s"]) or 0) / 3600, "status": r["status"],
                    "score": f(r["val_bpb"]), "delta": f(r["delta_vs_incumbent"]), "hyp": r["hypothesis"],
                    "cost": f(r["cost_usd"]) or 0, "cum": f(r["cum_cost_usd"]) or 0, "train_s": f(r["train_seconds"]),
                    "lat": f(r["llm_latency_s"]), "fix": int(f(r["fix_attempts"]) or 0),
                    "think": int(f(r["tokens_thinking"]) or 0), "out": int(f(r["tokens_out_visible"]) or 0)})
    model = rows[0]["model"] if rows else ""
    return {"id": arm_id, "label": label, "color": color, "model": model,
            "thinking": rows[0]["thinking_setting"] if rows else "", "rows": out}


def build(arm_specs, out_path, title, hours):
    arms = []
    for i, spec in enumerate(arm_specs):
        arm_id, _, label = spec.partition(":")
        arms.append(load_arm(arm_id, label or arm_id, COLORS[i % len(COLORS)]))
    noise_p = os.path.join(ROOT, "runs", "_preflight", "noise.json")
    noise = json.load(open(noise_p)) if os.path.exists(noise_p) else None
    final_p = os.path.join(ROOT, "runs", "_final", "results.json")
    final = json.load(open(final_p)) if os.path.exists(final_p) else None
    meta_p = os.path.join(ROOT, "data", "meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    test_run = meta.get("dataset") != "climbmix" or any(a["model"] == "mock" for a in arms)
    data = {"title": title, "hours": hours, "noise": noise, "final": final, "dataset": meta.get("dataset", "?"),
            "gpu": meta.get("gpu", "?"),
            "test_run": test_run, "arms": arms, "built": time.strftime("%Y-%m-%d %H:%M")}
    html = open(TEMPLATE).read().replace("/*__DATA__*/null", json.dumps(data))
    with open(out_path, "w") as fh:
        fh.write(html)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="arm_id or arm_id:Label")
    ap.add_argument("--title", default="I gave Gemini 4 Neon an H100 and went to sleep.")
    ap.add_argument("--hours", type=float, default=8.0, help="wall-clock budget per arm (x-axis end)")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "dashboard.html"))
    ap.add_argument("--watch", type=int, default=0, help="rebuild every N seconds")
    args = ap.parse_args()
    while True:
        print("wrote", build(args.arms, args.out, args.title, args.hours), flush=True)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
