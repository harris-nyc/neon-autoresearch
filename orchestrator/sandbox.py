"""Run one train.py: 5-minute training budget, no network, only its phase's data, read-only.

Modes
  local  (default, Linux): `unshare -rmn` -> new user + mount + network namespaces. Inside:
         the work dir and data/<phase> are bind-mounted onto fixed paths (data read-only), then
         data/ and runs/ are hidden behind empty tmpfs mounts. So during the search the test shard
         does not exist, and nothing from other experiments or the other model is visible.
  docker: `docker run --network none --gpus all` with only the work dir, data/<phase> (ro) and
         hf_cache (ro) mounted. Use if unshare isn't permitted on your box.
  none:  no isolation (debugging only, needs AUTORESEARCH_ALLOW_UNSAFE=1).

Every train.py is linted first (orchestrator/lint.py). The child gets a minimal environment:
no API keys, no proxies. The score is read from the JSON that prepare.evaluate_bpb writes.
"""
import hashlib, json, os, re, shutil, signal, subprocess, sys, time

from . import lint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = os.environ.get("AUTORESEARCH_SMOKE") == "1"
PREPARE = os.path.join(ROOT, "tests", "smoke", "prepare.py") if SMOKE else os.path.join(ROOT, "prepare.py")
PREPARE_SHA = os.path.join(os.path.dirname(PREPARE), "prepare.sha256") if SMOKE else os.path.join(ROOT, "orchestrator", "prepare.sha256")
BASELINE = os.path.join(ROOT, "tests", "smoke", "train.py") if SMOKE else os.path.join(ROOT, "train.py")
TIME_BUDGET = int(os.environ.get("AUTORESEARCH_SMOKE_BUDGET", "20")) if SMOKE else 300
GRACE_S = 60 if SMOKE else 300   # startup + compile + eval; total kill at 10 min, as in Karpathy's program.md
HF_CACHE = os.path.join(ROOT, "hf_cache")


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def check_prepare():
    want = open(PREPARE_SHA).read().strip()
    got = sha256(PREPARE)
    if got != want:
        raise RuntimeError(f"prepare.py hash mismatch ({got[:12]} != {want[:12]}); refusing to run")
    return got


def with_seed(code, seed):
    """Noise-floor / final-test seeds: Karpathy's train.py hard-codes manual_seed(42)."""
    return re.sub(r"manual_seed\(\s*42\s*\)", f"manual_seed({seed})", code) if seed is not None else code


def _env(work, data, hf=HF_CACHE, eval_shard=None):
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": work,
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "AUTORESEARCH_CACHE": data,
        "AUTORESEARCH_RESULT": os.path.join(work, "out", "result.json"),
        "HF_HOME": hf,
        "HF_HUB_OFFLINE": "1",
    }
    if eval_shard:
        env["AUTORESEARCH_EVAL_SHARD"] = eval_shard
    for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
              "AUTORESEARCH_SMOKE_BUDGET"):
        if k in os.environ:
            env[k] = os.environ[k]
    return env


def run(code, workdir, phase="search", seed=None, mode=None, budget=None, grace=None):
    """Returns dict(status=ok|crash|timeout|rejected, val_bpb, wall_s, train_seconds, log_tail)."""
    mode = mode or os.environ.get("AUTORESEARCH_SANDBOX", "local")
    budget = budget or TIME_BUDGET
    grace = GRACE_S if grace is None else grace
    out = {"status": "ok", "val_bpb": None, "wall_s": 0.0, "train_seconds": None, "peak_vram_mb": None, "log_tail": ""}
    try:
        check_prepare()
    except RuntimeError as e:
        return {**out, "status": "rejected", "log_tail": str(e)}
    problem = lint.check(code)
    if problem:
        return {**out, "status": "rejected", "log_tail": f"[harness] train.py rejected before running: {problem}"}

    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(os.path.join(workdir, "out"))
    shutil.copy(PREPARE, os.path.join(workdir, "prepare.py"))
    with open(os.path.join(workdir, "train.py"), "w") as f:
        f.write(with_seed(code, seed))
    data_dir = os.path.join(ROOT, "data", phase)
    eval_shard = open(os.path.join(ROOT, "data", f"{phase}.eval_shard")).read().strip()
    py = sys.executable

    if mode == "local":
        mw = os.path.join(ROOT, "orchestrator", ".mnt_work")
        md = os.path.join(ROOT, "orchestrator", ".mnt_data")
        os.makedirs(mw, exist_ok=True); os.makedirs(md, exist_ok=True)
        env = _env(mw, md, eval_shard=eval_shard)
        hide = [os.path.join(ROOT, d) for d in ("data", "runs") if os.path.isdir(os.path.join(ROOT, d))]
        steps = [f"mount --bind '{workdir}' '{mw}'",
                 f"mount --bind '{data_dir}' '{md}'",
                 f"mount -o remount,ro,bind '{md}'"]
        steps += [f"mount -t tmpfs none '{h}'" for h in hide]
        steps.append(f"cd '{mw}' && exec '{py}' -s -E train.py")
        cmd = ["unshare", "-rmn", "--fork", "sh", "-c", " && ".join(steps)]
    elif mode == "docker":
        image = os.environ.get("AUTORESEARCH_DOCKER_IMAGE", "neon-autoresearch:latest")
        cenv = _env("/work", "/data", "/hf_cache", eval_shard)
        envs = sum((["-e", f"{k}={v}"] for k, v in cenv.items() if k not in ("PATH", "LD_LIBRARY_PATH")), [])
        cmd = ["docker", "run", "--rm", "--network", "none", "--gpus", "all", "--user", f"{os.getuid()}:{os.getgid()}",
               "-v", f"{workdir}:/work", "-v", f"{data_dir}:/data:ro", "-v", f"{HF_CACHE}:/hf_cache:ro",
               "-w", "/work", *envs, image, "python", "-s", "train.py"]
        env = {"PATH": os.environ.get("PATH", "")}
    elif mode == "none":
        if os.environ.get("AUTORESEARCH_ALLOW_UNSAFE") != "1":
            raise RuntimeError("sandbox mode 'none' needs AUTORESEARCH_ALLOW_UNSAFE=1")
        env = _env(workdir, data_dir, eval_shard=eval_shard)
        cmd = ["sh", "-c", f"cd '{workdir}' && exec '{py}' -s -E train.py"]
    else:
        raise ValueError(mode)

    log_path = os.path.join(workdir, "log.txt")
    t0 = time.time()
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            if p.wait(timeout=budget + grace) != 0:
                out["status"] = "crash"
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL); p.wait()
            out["status"] = "timeout"
    out["wall_s"] = time.time() - t0
    text = open(log_path, errors="replace").read().replace("\r", "\n")
    res_path = os.path.join(workdir, "out", "result.json")
    if out["status"] == "ok":
        if os.path.exists(res_path):
            out["val_bpb"] = float(json.load(open(res_path))["val_bpb"])
            if out["val_bpb"] != out["val_bpb"]:  # NaN
                out["status"] = "crash"; text += "\n[harness] val_bpb is NaN\n"
        else:
            out["status"] = "crash"; text += "\n[harness] train.py finished without calling evaluate_bpb\n"
    if out["status"] == "timeout":
        text += f"\n[harness] killed after {out['wall_s']:.0f}s (limit {budget + grace}s)\n"
    for key, pat in (("train_seconds", r"^training_seconds:\s+([\d.]+)"), ("peak_vram_mb", r"^peak_vram_mb:\s+([\d.]+)")):
        m = re.search(pat, text, re.M)  # informational only; the score comes from result.json
        if m:
            out[key] = float(m.group(1))
    lines = [l for l in text.splitlines() if l.strip()]
    out["log_tail"] = "\n".join(lines[-60:])
    with open(log_path, "w") as f:
        f.write(text)
    return out
