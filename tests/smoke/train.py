"""CPU STAND-IN for Karpathy's train.py (tiny GPT), same contract and knob names. Smoke tests only."""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

DEPTH = 2               # number of transformer layers
MODEL_DIM = 64          # width
MATRIX_LR = 0.003       # learning rate
EMBEDDING_LR = 0.003    # embedding learning rate
WEIGHT_DECAY = 0.0      # weight decay
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
DEVICE_BATCH_SIZE = 32  # per-device batch size


class GPT(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.wte = nn.Embedding(vocab, MODEL_DIM)
        self.wpe = nn.Embedding(MAX_SEQ_LEN, MODEL_DIM)
        layer = nn.TransformerEncoderLayer(MODEL_DIM, 4, 4 * MODEL_DIM, 0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, DEPTH, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(MODEL_DIM)
        self.head = nn.Linear(MODEL_DIM, vocab, bias=False)
        self.register_buffer("mask", torch.triu(torch.full((MAX_SEQ_LEN, MAX_SEQ_LEN), float("-inf")), 1))

    def forward(self, x, y, reduction="mean"):
        T = x.shape[1]
        h = self.wte(x) + self.wpe(torch.arange(T))
        h = self.ln(self.blocks(h, mask=self.mask[:T, :T]))
        logits = self.head(h)
        return F.cross_entropy(logits.view(-1, logits.shape[-1]), y.reshape(-1), reduction=reduction)


t_start = time.time()
torch.manual_seed(42)
tokenizer = Tokenizer.from_directory()
model = GPT(tokenizer.get_vocab_size())
emb = [model.wte.weight, model.wpe.weight]
other = [p for p in model.parameters() if all(p is not e for e in emb)]
opt = torch.optim.AdamW([{"params": emb, "lr": EMBEDDING_LR}, {"params": other, "lr": MATRIX_LR}], weight_decay=WEIGHT_DECAY)
base_lrs = [g["lr"] for g in opt.param_groups]
train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")

t0 = time.time()
step = 0
while True:
    x, y, _ = next(train_loader)
    progress = min((time.time() - t0) / TIME_BUDGET, 1.0)
    lrm = 1.0 if progress < 1 - WARMDOWN_RATIO else max(0.0, (1 - progress) / WARMDOWN_RATIO)
    for g, b in zip(opt.param_groups, base_lrs):
        g["lr"] = b * lrm
    loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    step += 1
    if time.time() - t0 >= TIME_BUDGET:
        break
training_seconds = time.time() - t0
model.eval()
val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)
print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {training_seconds:.1f}")
print(f"num_steps:        {step}")
print(f"depth:            {DEPTH}")
