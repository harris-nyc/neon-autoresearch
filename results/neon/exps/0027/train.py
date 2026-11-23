"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

torch._dynamo.config.cache_size_limit = 64
if hasattr(torch._dynamo.config, "recompile_limit"):
    torch._dynamo.config.recompile_limit = 64

from kernels import get_kernel
cap = torch.cuda.get_device_capability()
# varunneal's FA3 is Hopper only, use kernels-community on non-Hopper GPUs
repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
fa3 = get_kernel(repo).flash_attn_interface

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# Fused Chunked LM Head + Softcap + Cross-Entropy
# ---------------------------------------------------------------------------

@torch.compile(dynamic=False, fullgraph=True)
def _softcap_ce_chunk_fwd_bwd(raw_bf16, y_c, softcap, inv_n):
    raw = raw_bf16.float()
    t = torch.tanh(raw / softcap)
    logits = softcap * t
    lse = torch.logsumexp(logits, dim=-1, keepdim=True)
    nll = lse.squeeze(-1) - logits.gather(-1, y_c.unsqueeze(-1)).squeeze(-1)
    loss_sum = nll.sum()
    probs = torch.exp(logits - lse)
    neg_ones = torch.full((y_c.size(0), 1), -1.0, device=probs.device, dtype=probs.dtype)
    d_logits = probs.scatter_add(-1, y_c.unsqueeze(-1), neg_ones)
    d_raw = (d_logits * (1.0 - t * t) * inv_n).to(raw_bf16.dtype)
    return loss_sum, d_raw


@torch.compile(dynamic=False, fullgraph=True)
def _softcap_ce_chunk_eval(raw_bf16, y_c, softcap):
    logits = softcap * torch.tanh(raw_bf16.float() / softcap)
    return F.cross_entropy(logits, y_c, ignore_index=-1, reduction='none')


class FusedChunkedLMHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, weight, targets, softcap):
        B, T, D = h.shape
        N = B * T
        h_flat = h.reshape(N, D).to(torch.bfloat16)
        y_flat = targets.reshape(N)
        w_bf16 = weight.to(torch.bfloat16)
        d_h = torch.empty_like(h_flat)
        d_w = torch.zeros_like(weight, dtype=torch.float32)
        loss_sum = torch.zeros((), device=h.device, dtype=torch.float32)
        inv_n = 1.0 / N
        chunk_size = 16384
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            h_c = h_flat[start:end]
            y_c = y_flat[start:end]
            raw_c = F.linear(h_c, w_bf16)
            l_c, d_raw_c = _softcap_ce_chunk_fwd_bwd(raw_c, y_c, softcap, inv_n)
            loss_sum = loss_sum + l_c
            d_h[start:end] = d_raw_c @ w_bf16
            d_w.add_((d_raw_c.mT @ h_c).float())
        ctx.save_for_backward(d_h.view(B, T, D).to(h.dtype), d_w.to(weight.dtype))
        return loss_sum * inv_n

    @staticmethod
    def backward(ctx, grad_output):
        d_h, d_w = ctx.saved_tensors
        return d_h * grad_output.to(d_h.dtype), d_w * grad_output.to(d_w.dtype), None, None


@torch._dynamo.disable
def fused_chunked_lm_head(h, weight, targets, softcap, reduction):
    if reduction == 'mean' and h.requires_grad:
        return FusedChunkedLMHead.apply(h, weight, targets, softcap)
    B, T, D = h.shape
    N = B * T
    h_flat = h.reshape(N, D).to(torch.bfloat16)
    y_flat = targets.reshape(N)
    w_bf16 = weight.to(torch.bfloat16)
    chunk_size = 16384
    if reduction == 'none':
        out = torch.empty(N, device=h.device, dtype=torch.float32)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            raw_c = F.linear(h_flat[start:end], w_bf16)
            out[start:end] = _softcap_ce_chunk_eval(raw_c, y_flat[start:end], softcap)
        return out
    loss_sum = torch.zeros((), device=h.device, dtype=torch.float32)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        raw_c = F.linear(h_flat[start:end], w_bf16)
        loss_sum = loss_sum + _softcap_ce_chunk_eval(raw_c, y_flat[start:end], softcap).sum()
    return loss_sum / N

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
    bos_token_id: int = 0
    bigram_vocab_size: int = 1048576
    trigram_vocab_size: int = 2097152
    fourgram_vocab_size: int = 2097152
    skip_vocab_size: int = 1048576


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def hash64(key, c1, c2, mask):
    h = (key ^ (key >> 17)) * c1
    h = (h ^ (h >> 23)) * c2
    return (h ^ (h >> 19)) & mask


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Collision-optimal 1.61B 4-order Input+Output n-gram hash embeddings
        # Each head row stores [in_part | out_part] contiguously (2.44x less gather/scatter traffic)
        assert config.vocab_size <= (1 << 13)
        for sz in (config.bigram_vocab_size, config.trigram_vocab_size,
                   config.fourgram_vocab_size, config.skip_vocab_size):
            assert (sz & (sz - 1)) == 0
        # Bigram: 2 heads x 1M x 256 (128 in + 128 out) = 536.9M params
        self.bi_hdim = 128
        self.bigram_embeds = nn.ModuleList([
            nn.Embedding(config.bigram_vocab_size, 2 * self.bi_hdim),
            nn.Embedding(config.bigram_vocab_size, 2 * self.bi_hdim),
        ])
        # Trigram: 2 heads x 2M x 128 (64 in + 64 out) = 536.9M params
        self.tri_hdim = 64
        self.trigram_embeds = nn.ModuleList([
            nn.Embedding(config.trigram_vocab_size, 2 * self.tri_hdim),
            nn.Embedding(config.trigram_vocab_size, 2 * self.tri_hdim),
        ])
        # 4-gram: 2 heads x 2M x 64 (32 in + 32 out) = 268.4M params
        self.fg_hdim = 32
        self.fourgram_embeds = nn.ModuleList([
            nn.Embedding(config.fourgram_vocab_size, 2 * self.fg_hdim),
            nn.Embedding(config.fourgram_vocab_size, 2 * self.fg_hdim),
        ])
        # Skip-bigram (t-2, t): 2 heads x 1M x 128 (64 in + 64 out) = 268.4M params
        self.sk_hdim = 64
        self.skip_embeds = nn.ModuleList([
            nn.Embedding(config.skip_vocab_size, 2 * self.sk_hdim),
            nn.Embedding(config.skip_vocab_size, 2 * self.sk_hdim),
        ])
        # Per-head dims for the 8 independent heads: [128, 128, 64, 64, 32, 32, 64, 64] -> sum = 576
        self.head_dims = (
            self.bi_hdim, self.bi_hdim,
            self.tri_hdim, self.tri_hdim,
            self.fg_hdim, self.fg_hdim,
            self.sk_hdim, self.sk_hdim,
        )
        self.ngram_total_dim = sum(self.head_dims)  # 576
        self.out_groups_per_head = 4
        self.key_dim = 8
        # Muon-optimized 2D projections & nonlinear ReLU^2 output fusion MLP
        self.ngram_muon = nn.ModuleDict({
            "in_proj": nn.Linear(self.ngram_total_dim, config.n_embd, bias=False),
            "mid_proj": nn.Linear(self.ngram_total_dim, config.n_embd, bias=False),
            "out_lin": nn.Linear(self.ngram_total_dim, config.n_embd, bias=False),
            "out_fuse_fc": nn.Linear(2 * config.n_embd, 2 * config.n_embd, bias=False),
            "out_fuse_proj": nn.Linear(2 * config.n_embd, config.n_embd, bias=False),
        })
        # Strictly independent per-head key-match + context gating (mid after Block 1 & output before lm_head)
        self.ngram_gate = nn.ModuleDict({
            "mid_ctx": nn.Linear(config.n_embd, 8, bias=False),
            "mid_q": nn.Linear(config.n_embd, 8 * self.key_dim, bias=False),
            "mid_k": nn.ModuleList([nn.Linear(d, self.key_dim, bias=False) for d in self.head_dims]),
            "out_ctx": nn.Linear(config.n_embd, 8 * self.out_groups_per_head, bias=False),
            "out_q": nn.Linear(config.n_embd, 8 * self.out_groups_per_head * self.key_dim, bias=False),
            "out_k": nn.ModuleList([nn.Linear(d, self.out_groups_per_head * self.key_dim, bias=False) for d in self.head_dims]),
        })
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # N-gram embeddings init to zero (neutral start)
        for tbl_list in (self.bigram_embeds, self.trigram_embeds, self.fourgram_embeds, self.skip_embeds):
            for tbl in tbl_list:
                torch.nn.init.zeros_(tbl.weight)
        # Muon n-gram projections & output fusion MLP
        for name, lin in self.ngram_muon.items():
            if name == "out_fuse_proj":
                torch.nn.init.zeros_(lin.weight)
            elif name == "mid_proj":
                s_lin = 0.5 * (3**0.5 * lin.in_features**-0.5)
                torch.nn.init.uniform_(lin.weight, -s_lin, s_lin)
            else:
                s_lin = 3**0.5 * lin.in_features**-0.5
                torch.nn.init.uniform_(lin.weight, -s_lin, s_lin)
        # Independent per-head gating projections
        for name, mod in self.ngram_gate.items():
            if name in ("mid_ctx", "out_ctx"):
                torch.nn.init.zeros_(mod.weight)
            elif isinstance(mod, nn.ModuleList):
                for lin in mod:
                    s_lin = 3**0.5 * lin.in_features**-0.5
                    torch.nn.init.uniform_(lin.weight, -s_lin, s_lin)
            else:
                s_lin = 3**0.5 * mod.in_features**-0.5
                torch.nn.init.uniform_(mod.weight, -s_lin, s_lin)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)
        for tbl_list in (self.bigram_embeds, self.trigram_embeds, self.fourgram_embeds, self.skip_embeds):
            for tbl in tbl_list:
                tbl.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        ngram_embeds_numel = sum(
            tbl.weight.numel()
            for tbl_list in (self.bigram_embeds, self.trigram_embeds, self.fourgram_embeds, self.skip_embeds)
            for tbl in tbl_list
        )
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          ngram_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        bigram_embeds = sum(p.numel() for p in self.bigram_embeds.parameters())
        trigram_embeds = sum(p.numel() for p in self.trigram_embeds.parameters())
        fourgram_embeds = sum(p.numel() for p in self.fourgram_embeds.parameters())
        skip_embeds = sum(p.numel() for p in self.skip_embeds.parameters())
        ngram_dense = sum(p.numel() for p in self.ngram_muon.parameters()) + sum(p.numel() for p in self.ngram_gate.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = (wte + value_embeds + bigram_embeds + trigram_embeds + fourgram_embeds +
                 skip_embeds + ngram_dense + lm_head + transformer_matrices + scalars)
        return {
            'wte': wte, 'value_embeds': value_embeds,
            'bigram_embeds': bigram_embeds, 'trigram_embeds': trigram_embeds,
            'fourgram_embeds': fourgram_embeds, 'skip_embeds': skip_embeds,
            'ngram_dense': ngram_dense,
            'lm_head': lm_head, 'transformer_matrices': transformer_matrices,
            'scalars': scalars, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, ngram_lr=0.15,
                        matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters()) + list(self.ngram_muon.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        ngram_embeds_params = (
            list(self.bigram_embeds.parameters()) +
            list(self.trigram_embeds.parameters()) +
            list(self.fourgram_embeds.parameters()) +
            list(self.skip_embeds.parameters())
        )
        ngram_gate_params = list(self.ngram_gate.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(ngram_embeds_params) +
            len(ngram_gate_params) + len(resid_params) + len(x0_params))
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=ngram_embeds_params, lr=ngram_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=ngram_gate_params, lr=0.01 * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        # Exact bijective n-gram keys masked at BOS document boundaries
        is_bos = idx == self.config.bos_token_id
        prev_idx = F.pad(idx[:, :-1], (1, 0), value=0).masked_fill(is_bos, 0)
        prev_is_bos = F.pad(is_bos[:, :-1], (1, 0), value=True)
        mask2 = is_bos | prev_is_bos
        prev2_idx = F.pad(prev_idx[:, :-1], (1, 0), value=0).masked_fill(mask2, 0)
        prev2_is_bos = F.pad(prev_is_bos[:, :-1], (1, 0), value=True)
        mask3 = mask2 | prev2_is_bos
        prev3_idx = F.pad(prev2_idx[:, :-1], (1, 0), value=0).masked_fill(mask3, 0)

        key2 = (prev_idx << 13) | idx
        key3 = (prev2_idx << 26) | key2
        key4 = (prev3_idx << 39) | key3
        key_skip = (prev2_idx << 13) | idx

        # Two independent 64-bit hash functions per order
        C1A, C2A = 0x2545F4914F6CDD1D, 0x165667B19E3779F9
        C1B, C2B = 0x3C79AC492BA7B653, 0x1C69B3F74AC4AE35

        m2 = mask2.unsqueeze(-1)
        m3 = mask3.unsqueeze(-1)

        bmask = self.config.bigram_vocab_size - 1
        b0 = self.bigram_embeds[0](hash64(key2, C1A, C2A, bmask))
        b1 = self.bigram_embeds[1](hash64(key2, C1B, C2B, bmask))
        b0_in, b0_out = b0.chunk(2, dim=-1)
        b1_in, b1_out = b1.chunk(2, dim=-1)

        tmask = self.config.trigram_vocab_size - 1
        t0 = self.trigram_embeds[0](hash64(key3, C1A, C2A, tmask)).masked_fill(m2, 0)
        t1 = self.trigram_embeds[1](hash64(key3, C1B, C2B, tmask)).masked_fill(m2, 0)
        t0_in, t0_out = t0.chunk(2, dim=-1)
        t1_in, t1_out = t1.chunk(2, dim=-1)

        fmask = self.config.fourgram_vocab_size - 1
        f0 = self.fourgram_embeds[0](hash64(key4, C1A, C2A, fmask)).masked_fill(m3, 0)
        f1 = self.fourgram_embeds[1](hash64(key4, C1B, C2B, fmask)).masked_fill(m3, 0)
        f0_in, f0_out = f0.chunk(2, dim=-1)
        f1_in, f1_out = f1.chunk(2, dim=-1)

        smask = self.config.skip_vocab_size - 1
        s0 = self.skip_embeds[0](hash64(key_skip, C1A, C2A, smask)).masked_fill(m2, 0)
        s1 = self.skip_embeds[1](hash64(key_skip, C1B, C2B, smask)).masked_fill(m2, 0)
        s0_in, s0_out = s0.chunk(2, dim=-1)
        s1_in, s1_out = s1.chunk(2, dim=-1)

        out_heads = (b0_out, b1_out, t0_out, t1_out, f0_out, f1_out, s0_out, s1_out)

        ng_in = torch.cat([b0_in, b1_in, t0_in, t1_in, f0_in, f1_in, s0_in, s1_in], dim=-1)
        x = self.transformer.wte(idx) + self.ngram_muon["in_proj"](ng_in)
        x = norm(x)
        x0 = x

        K = self.key_dim
        inv_sqrt_k = K ** -0.5

        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
            if i == 1:
                # Mid-network Engram injection with strictly independent per-head key-match gating
                hm = norm(x)
                mid_ctx = self.ngram_gate["mid_ctx"](hm)
                mid_q = F.rms_norm(self.ngram_gate["mid_q"](hm).view(B, T, 8, K), (K,), eps=1e-4)
                mid_parts = []
                for idx_h, vh in enumerate(out_heads):
                    kh = self.ngram_gate["mid_k"][idx_h](vh)
                    kh = kh * torch.rsqrt(kh.square().mean(dim=-1, keepdim=True) + 1.0)
                    score = (mid_q[:, :, idx_h, :] * kh).sum(dim=-1) * inv_sqrt_k
                    g = 2.0 * torch.sigmoid(mid_ctx[:, :, idx_h] + score)
                    mid_parts.append(g.unsqueeze(-1) * vh)
                x = x + self.ngram_muon["mid_proj"](torch.cat(mid_parts, dim=-1))

        h = norm(x)

        # Strictly independent per-head 4-group key-match + context gating before projection
        G = self.out_groups_per_head
        out_ctx = self.ngram_gate["out_ctx"](h).view(B, T, 8, G)
        out_q = F.rms_norm(self.ngram_gate["out_q"](h).view(B, T, 8, G, K), (K,), eps=1e-4)
        gated_out_parts = []
        for idx_h, vh in enumerate(out_heads):
            kh = self.ngram_gate["out_k"][idx_h](vh).view(B, T, G, K)
            kh = kh * torch.rsqrt(kh.square().mean(dim=-1, keepdim=True) + 1.0)
            score = (out_q[:, :, idx_h, :, :] * kh).sum(dim=-1) * inv_sqrt_k
            g = 2.0 * torch.sigmoid(out_ctx[:, :, idx_h, :] + score)
            dh = self.head_dims[idx_h]
            gated_out_parts.append((g.unsqueeze(-1) * vh.view(B, T, G, dh // G)).reshape(B, T, dh))

        ng_out_512 = self.ngram_muon["out_lin"](torch.cat(gated_out_parts, dim=-1))

        # Muon-optimized nonlinear ReLU^2 output fusion MLP
        fuse_in = torch.cat([h, ng_out_512], dim=-1)
        fuse_delta = self.ngram_muon["out_fuse_proj"](F.relu(self.ngram_muon["out_fuse_fc"](fuse_in)).square())
        h_final = h + ng_out_512 + fuse_delta

        softcap = 15.0
        if targets is not None:
            return fused_chunked_lm_head(h_final, self.lm_head.weight, targets, softcap, reduction)

        logits = self.lm_head(h_final).float()
        return softcap * torch.tanh(logits / softcap)

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 64        # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128           # target head dimension for attention
WINDOW_PATTERN = "SSSL"  # sliding window pattern: L=full, S=half context
BIGRAM_VOCAB_SIZE = 1048576    # bigram hash table size per head (power of 2)
TRIGRAM_VOCAB_SIZE = 2097152   # trigram hash table size per head (power of 2)
FOURGRAM_VOCAB_SIZE = 2097152  # 4-gram hash table size per head (power of 2)
SKIP_VOCAB_SIZE = 1048576      # skip-bigram hash table size per head (power of 2)

# Optimization
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
EMBEDDING_LR = 0.45      # learning rate for token embeddings (Adam)
NGRAM_LR = 0.15          # learning rate for n-gram hash embeddings (Adam)
UNEMBEDDING_LR = 0.003   # learning rate for lm_head (Adam)
MATRIX_LR = 0.03         # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.4          # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2       # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0       # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5     # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0      # final LR as fraction of initial

# Model size
DEPTH = 8                # number of transformer layers
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
H100_BF16_PEAK_FLOPS = 989.5e12

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
bos_token_id = tokenizer.get_bos_token_id()
print(f"Vocab size: {vocab_size:,}")

def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN, bos_token_id=bos_token_id,
        bigram_vocab_size=BIGRAM_VOCAB_SIZE, trigram_vocab_size=TRIGRAM_VOCAB_SIZE,
        fourgram_vocab_size=FOURGRAM_VOCAB_SIZE, skip_vocab_size=SKIP_VOCAB_SIZE,
    )

config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    ngram_lr=NGRAM_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)

model = torch.compile(model, dynamic=False)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedules (all based on progress = training_time / TIME_BUDGET)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # Progress and schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Time's up — but only stop after warmup steps so we don't count compilation
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / H100_BF16_PEAK_FLOPS if total_training_time > 0 else 0
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
