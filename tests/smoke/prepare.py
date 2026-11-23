"""CPU STAND-IN for Karpathy's prepare.py, same public API, for testing the harness without a GPU.
Synthetic token data from a random Markov chain; "bytes" = 1 per token. Not used in real runs.
"""
import json, math, os
import numpy as np
import torch

MAX_SEQ_LEN = 64
TIME_BUDGET = int(os.environ.get("AUTORESEARCH_SMOKE_BUDGET", "20"))
EVAL_TOKENS = 64 * 64 * 4
CACHE_DIR = os.environ.get("AUTORESEARCH_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-smoke")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
VAL_FILENAME = "shard_06542.npy"
TEST_FILENAME = "shard_06541.npy"
EVAL_FILENAME = os.environ.get("AUTORESEARCH_EVAL_SHARD", VAL_FILENAME)
_HELD_OUT = {VAL_FILENAME, TEST_FILENAME}
VOCAB = 64


def make_synthetic(cache, seed=0):
    rng = np.random.default_rng(seed)
    nxt = rng.integers(1, VOCAB, size=(VOCAB, 3))
    def stream(n):
        out, cur = np.empty(n, np.int64), 1
        for i in range(n):
            out[i] = cur
            cur = int(nxt[cur, rng.integers(0, 3)]) if rng.random() < 0.85 else int(rng.integers(1, VOCAB))
        return out
    os.makedirs(os.path.join(cache, "data"), exist_ok=True)
    os.makedirs(os.path.join(cache, "tokenizer"), exist_ok=True)
    for name, n in (("shard_00000.npy", 200_000), ("shard_00001.npy", 200_000), (VAL_FILENAME, 20_000), (TEST_FILENAME, 20_000)):
        np.save(os.path.join(cache, "data", name), stream(n))
    json.dump({"vocab": VOCAB, "bos": 0}, open(os.path.join(cache, "tokenizer", "vocab.json"), "w"))


class Tokenizer:
    def __init__(self, meta):
        self.meta = meta

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        return cls(json.load(open(os.path.join(tokenizer_dir, "vocab.json"))))

    def get_vocab_size(self):
        return self.meta["vocab"]

    def get_bos_token_id(self):
        return self.meta["bos"]


def _tokens(split):
    names = sorted(os.listdir(DATA_DIR))
    if split == "train":
        names = [n for n in names if n not in _HELD_OUT]
    else:
        names = [EVAL_FILENAME]
    return np.concatenate([np.load(os.path.join(DATA_DIR, n)) for n in names])


def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    assert split in ["train", "val"]
    toks = torch.from_numpy(_tokens(split))
    g = torch.Generator().manual_seed(0)
    epoch = 1
    while True:
        idx = torch.randint(0, len(toks) - T - 1, (B,), generator=g)
        rows = torch.stack([toks[i:i + T + 1] for i in idx.tolist()])
        yield rows[:, :-1], rows[:, 1:], epoch


@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    toks = torch.from_numpy(_tokens("val"))
    T = MAX_SEQ_LEN
    n = EVAL_TOKENS // T
    rows = torch.stack([toks[i * T:i * T + T + 1] for i in range(n)])
    total_nats, total = 0.0, 0
    for b in range(0, n, batch_size):
        x, y = rows[b:b + batch_size, :-1], rows[b:b + batch_size, 1:]
        total_nats += model(x, y, reduction="none").float().sum().item()
        total += y.numel()
    bpb = total_nats / (math.log(2) * total)
    _out = os.environ.get("AUTORESEARCH_RESULT")
    if _out:
        with open(_out, "w") as f:
            json.dump({"val_bpb": bpb, "eval_shard": EVAL_FILENAME}, f)
    return bpb
