"""One-time setup on the GPU box. Orchestrator-only.

    python -m orchestrator.data_setup            # real: Karpathy's data + tokenizer + FA3 kernel, ~5 min
    python -m orchestrator.data_setup --smoke    # tiny synthetic stand-in for CPU smoke tests

Layout
  data/cache/{data,tokenizer}   Karpathy's cache (10 train shards + val shard 06542 + test shard 06541)
  data/search/                  what the sandbox sees while the models search: train shards + VAL shard
  data/final/                   what the sandbox sees for the final test: train shards + TEST shard
  hf_cache/                     pre-downloaded flash-attention kernel (sandbox has no network)
The test shard is never visible during the search, and never used for training or the tokenizer.
"""
import argparse, json, os, shutil, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def link_tree(src_files, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for f in src_files:
        dst = os.path.join(dst_dir, os.path.basename(f))
        if not os.path.exists(dst):
            try:
                os.link(f, dst)
            except OSError:
                shutil.copy2(f, dst)


def build_phases(cache, eval_names):
    """eval_names: {"search": val file, "final": test file}"""
    shards = sorted(os.listdir(os.path.join(cache, "data")))
    held = set(eval_names.values())
    train = [os.path.join(cache, "data", s) for s in shards if s not in held and not s.endswith(".tmp")]
    tok = [os.path.join(cache, "tokenizer", f) for f in os.listdir(os.path.join(cache, "tokenizer"))]
    for phase, ev in eval_names.items():
        d = os.path.join(DATA, phase)
        shutil.rmtree(d, ignore_errors=True)
        link_tree(train + [os.path.join(cache, "data", ev)], os.path.join(d, "data"))
        link_tree(tok, os.path.join(d, "tokenizer"))
        with open(os.path.join(DATA, f"{phase}.eval_shard"), "w") as f:  # read by the sandbox, outside data/<phase>
            f.write(ev)
    return len(train)


def real(num_shards):
    cache = os.path.join(DATA, "cache")
    os.environ["AUTORESEARCH_CACHE"] = cache
    sys.path.insert(0, ROOT)
    import prepare  # Karpathy's (patched) prepare.py
    prepare.download_data(num_shards)
    if not prepare.download_single_shard(prepare.TEST_SHARD):
        raise SystemExit("failed to download the test shard")
    prepare.train_tokenizer()
    n = build_phases(cache, {"search": prepare.VAL_FILENAME, "final": prepare.TEST_FILENAME})

    # Flash-attention kernel: train.py fetches it from the Hugging Face Hub at import time.
    # Fetch it once here into hf_cache/; the sandbox then runs with HF_HUB_OFFLINE=1.
    os.environ["HF_HOME"] = os.path.join(ROOT, "hf_cache")
    import torch
    from kernels import get_kernel
    cap = torch.cuda.get_device_capability()
    repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
    get_kernel(repo)
    gpu = torch.cuda.get_device_name()
    meta = {"dataset": "climbmix", "train_shards": n, "val_shard": prepare.VAL_FILENAME,
            "test_shard": prepare.TEST_FILENAME, "gpu": gpu, "kernel_repo": repo}
    json.dump(meta, open(os.path.join(DATA, "meta.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))
    if cap != (9, 0):
        print("WARNING: not an H100. Karpathy's defaults assume an H100 80GB; results won't be comparable to his.")


def smoke():
    sys.path.insert(0, os.path.join(ROOT, "tests", "smoke"))
    import prepare as sp  # CPU stand-in with the same API
    cache = os.path.join(DATA, "cache")
    sp.make_synthetic(cache)
    build_phases(cache, {"search": sp.VAL_FILENAME, "final": sp.TEST_FILENAME})
    json.dump({"dataset": "synthetic", "gpu": "cpu"}, open(os.path.join(DATA, "meta.json"), "w"))
    print("smoke data ready")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--num-shards", type=int, default=10, help="training shards (Karpathy's default: 10)")
    a = ap.parse_args()
    smoke() if a.smoke else real(a.num_shards)
