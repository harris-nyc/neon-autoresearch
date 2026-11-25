# Gemini 4 Neon vs Claude Opus 5.5 and Claude Fable 5.2 on Karpathy's autoresearch

**Can a frontier model do ML research on its own, overnight?** I gave Gemini 4 Neon and Claude Opus 5.5 the same task: improve [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) GPT, one experiment at a time, on one NVIDIA H100, for 10 hours each, with no human in the loop. Every change was kept only if it beat the model's best so far by more than run-to-run noise. In the morning, each model's best version was retrained from scratch and scored on held-out data neither model ever saw.

**Update:** a second night ran Claude Fable 5.2 on the identical setup. See [Round 2](#round-2-gemini-4-neon-vs-claude-fable-52).

▶ **[Watch the 33-second video](media/neon_overnight.mp4)**

![Result](media/still_result.png)

## Results

**Held-out test** (each best `train.py` retrained from scratch, 3 seeds, scored on a data shard never visible during the night):

| | Test bits per byte (lower = better) | vs Karpathy's original |
|---|---:|---:|
| Karpathy's original `train.py` | 0.9940 ± 0.0011 | — |
| Claude Opus 5.5 | 0.9732 ± 0.0022 | 2.1% better |
| **Gemini 4 Neon** | **0.9573 ± 0.0009** | **3.7% better** |
| Claude Fable 5.2 (night 2, [Round 2](#round-2-gemini-4-neon-vs-claude-fable-52)) | 0.9484 ± 0.0006 | 4.6% better |

- **Neon improved the model 1.76× as much as Opus.** Both improvements are larger than 2× the seed-to-seed noise.
- **Neon's gains held up exactly on unseen data** (best validation 0.9573, test 0.9573). Opus's slipped slightly (0.9723 → 0.9732).
- **At equal effort, the gap is wider:** after 35 experiments each, Neon's validation improvement was ~2.1× Opus's.
- **In training-time terms:** Neon's 5-minute model matches Karpathy's original trained for ~11.4 minutes (**~2.3× speedup**); Opus's matches ~8 minutes (**~1.6×**). Estimated by training the unchanged original for 10, 20 and 30 minutes and interpolating on validation bits per byte (single seed per length).

![Race chart](media/race_chart.png)
![Speedup](media/speedup_chart.png)

| | Gemini 4 Neon | Claude Opus 5.5 |
|---|---:|---:|
| Experiments in 10 hours | 35 | 60 |
| Kept (beat best by > noise) | 4 | 6 |
| API cost | $35.07 | $27.46 |
| Median time per idea (model call) | ~8 min | ~1.5 min |
| Reasoning setting | default (`thinking_level=high`) | default (`effort=medium`) |
| Best model size | 1.66B params (mostly n-gram lookup tables) | 94M params |
| Best model peak GPU memory | 56 GB | 69 GB |

## How Neon found its biggest win

![One idea](media/still_one_idea.png)

At hour 0.7, Neon proposed a **bigram hash embedding**: a cheap lookup table indexed by a hash of each (previous token, current token) pair, added to the token embeddings. It's about 20 lines of code and costs almost no compute. It was the largest single improvement of the night for either model (−0.018 val_bpb). In its next turn, Neon's reasoning summary says:

> "That Experiment 3 result was fantastic … Now, my mind is racing, as I need to figure out why it worked and what we can push further. … why limit this to just bigrams?"

It then extended the idea to trigrams (kept), and later to a fused 4-order input+output n-gram memory (kept).

## Two research styles

**Neon took big, expensive swings.** Most of its 35 experiments were architectural: n-gram memories up to 4.3B parameters, a DeepSeek-style mixture of experts (0.12 bpb worse), product-key memory layers, custom fused cross-entropy kernels, document-masked attention. Four were kept, and the n-gram line of work produced nearly all of its gain. Late in the night its `train.py` grew to ~830 lines, and 6 responses were cut off mid-file and needed a repair turn.

**Opus ran a careful, systematic search.** Its 60 experiments were mostly sweeps (depth vs width, learning rates, warmdown, attention window patterns) plus ideas from the modded-nanogpt speedrun (attention output gates, value-embedding placement, Muon momentum warmdown). It repeatedly used a smart tactic: **combine two near-misses** that each fell just short of the noise threshold. Three of its six keeps came that way.

### Every kept change: Gemini 4 Neon

| Exp | Hour | val_bpb | Δ vs previous best | Hypothesis (model's own words, trimmed) |
|---:|---:|---:|---:|---|
| 0 | 0.1 | 0.9959 | +0.0000 | baseline (Karpathy's train.py) |
| 1 | 0.3 | 0.9871 | -0.0089 | Halve TOTAL_BATCH_SIZE from 2**19 (524K) to 2**18 (262K) with ~0.75x scaled LRs so the 50M model takes 2x as many optimizer steps in the 5-minute budget, moving closer to the critical batch size. |
| 3 | 0.7 | 0.9693 | -0.0178 | Add a 2-head bigram hash embedding table (2 x 131072 x n_embd/2, BOS-masked) summed into token embeddings before RMSNorm to give the 8K-vocab model high-capacity bigram lookup at near-zero FLOP cost. |
| 4 | 0.9 | 0.9604 | -0.0089 | Scale bigram hash tables 4x to 524,288 buckets per head and add 2-head BOS-masked trigram hash embeddings (2 x 524,288 x n_embd/2) summed into token embeddings to reduce bigram hash collisions and give the 8K-vocab model high-capa… |
| 16 | 4.0 | 0.9573 | -0.0031 | Upgrade n-gram memory to a hardware-fused 4-order Input+Output architecture (bigram, trigram, 1M-bucket 4-gram, skip-bigram) packing input and output vectors into the same embedding rows (halving random gathers/scatters) with exac… |
### Every kept change: Claude Opus 5.5

| Exp | Hour | val_bpb | Δ vs previous best | Hypothesis (model's own words, trimmed) |
|---:|---:|---:|---:|---|
| 0 | 0.2 | 0.9960 | +0.0000 | baseline (Karpathy's train.py) |
| 1 | 0.4 | 0.9870 | -0.0090 | Halve TOTAL_BATCH_SIZE to 2**18 (no grad accumulation): the ~50M model in a 5-min budget is likely step-limited, so twice as many optimizer steps at half the batch should lower val_bpb. |
| 6 | 1.3 | 0.9821 | -0.0049 | Combine DEPTH=10 with WARMDOWN_RATIO=0.7 (together -0.0013, a near-miss), and shrink the short sliding window from 1024 to 512 tokens. Attention is about a third of the FLOPs at 640 dim and 2048 context, so quarter-context windows… |
| 7 | 1.5 | 0.9799 | -0.0022 | Cut attention cost further: switch WINDOW_PATTERN to "SSSSL" (long-context layers at 4 and 9 only) and shrink the short window from 512 to 256 tokens. The last experiment showed this run is throughput-limited, and two full-context… |
| 25 | 4.6 | 0.9780 | -0.0019 | Combine the two best near-misses, which are independent: DEPTH=8 at 768 dim (exp 18, -0.0012) and "SSSL" with a 128-token short window (exp 24, -0.0006). The first gives wider matmuls with higher MFU and fewer layers. The second g… |
| 40 | 7.0 | 0.9755 | -0.0025 | Combine the two best near-misses. The first is the per-head attention output gate, y_h *= 2·sigmoid(W·x[:, :32]) with W initialised to zero, which scored -0.0017 in exp 39. The second is EMBEDDING_LR 0.9, which scored -0.0004 in e… |
| 50 | 8.5 | 0.9723 | -0.0031 | Move the value embeddings from alternating layers (1,3,5,7) to the first and last two layers (0,1,6,7), the placement used in modded-nanogpt, where token identity matters most. Pair this with the Muon momentum warmdown (0.95 to 0.… |
All experiments (including discarded ones), with each model's full response, the `train.py` it wrote, and the training log, are in `results/<model>/exps/`. `results/<model>/log.tsv` has one row per experiment.

## Round 2: Gemini 4 Neon vs Claude Fable 5.2

After the first post, people asked how Neon compares to Claude Fable 5.2. Fable ran on a second night with the identical harness, instructions, keep threshold (0.001881, from night 1's 5-seed noise floor) and H100 type. Box C was verified to match night 1's GPU before starting: 494.4M tokens in 5 minutes and baseline val_bpb 0.9958, both the same as night 1. Neon's results are reused from night 1.

▶ **[Watch the 33-second video](media/fable/neon_vs_fable.mp4)** · [interactive dashboard](results/dashboard_neon_vs_fable.html)

![Result](media/fable/still_result.png)

| | Test bits per byte (lower = better) | vs Karpathy's original |
|---|---:|---:|
| Karpathy's original `train.py` (night 2 retrain) | 0.9941 ± 0.0010 | — |
| Gemini 4 Neon (night 1) | 0.9573 ± 0.0009 | 3.7% better |
| **Claude Fable 5.2** | **0.9484 ± 0.0006** | **4.6% better** |

- **Fable won:** its improvement is ~1.24× Neon's, well beyond seed noise. Its validation and test scores also matched (0.9488 vs 0.9484).
- **Neon got 80% of Fable's improvement for half the API cost** ($35.07 vs $70.23, 35 experiments each).
- **Cost to reach a score favors Neon more.** Neon got below 0.961 validation bpb after $1.74 of API spend; Fable needed $13.17. Neon's best came at $12.00 and Fable's at $36.65. Neither improved after that point.
- **In training-time terms:** Fable's 5-minute model matches Karpathy's original trained ~15 minutes (**~3.0×**); Neon's ~11.4 minutes (~2.3×). Same speedup curve as night 1.
- **Fable hit its $70 budget cap at hour 9.1**, but its last improvement was at hour 5.4, so the cap didn't change the result.

| | Gemini 4 Neon | Claude Fable 5.2 |
|---|---:|---:|
| Experiments | 35 (10 hours) | 35 (9.1 hours, hit $70 cap) |
| Kept (beat best by > noise) | 4 | 9 |
| API cost | $35.07 | $70.23 |
| Median time per idea (model call) | ~8 min | ~5 min |
| Reasoning setting | default (`thinking_level=high`) | default (`effort=high`) |
| Best model size | 1.66B params (mostly n-gram lookup tables) | 1.69B params (mostly n-gram lookup tables) + 9 GB non-gradient memory |
| Best model peak GPU memory | 56 GB | 67 GB |

### Both models invented the same trick

![Same trick](media/fable/card_same.png)

Independently, both models added a **hashed bigram embedding**: a lookup table indexed by a hash of each (previous token, current token) pair, added to the token embeddings. Neon proposed it at minute 42 (experiment 3), Fable at 2h 11m (experiment 11). It was each model's single biggest win of the night (−0.018 and −0.015 val_bpb). Both then extended it to trigrams.

### Fable's 9 GB memory

![Memory](media/fable/card_memory.png)

Fable's biggest late win was a corpus-wide **follower memory**: three non-gradient hash tables that record, for every 3-, 5- and 8-token context in the training batches, the token that most recently followed it. Each batch is looked up before it is written, so the feature is always "held out" during training, and the tables are frozen before evaluation (`update_memory` is only called in the training loop), so test text is never stored. The 3 GB version took validation from 0.9569 to 0.9510; enlarging it to 9 GB reached 0.9488.

### Research styles and cost

![Styles](media/fable/card_style.png)
![Cost](media/fable/card_cost.png)

Neon took fewer, bigger swings (4 of 35 kept, including the largest single jump of either night). Fable stacked more, smaller wins (9 of 35): attention-window and depth changes, hashed bigram and trigram tables, a within-sequence "copy" feature, larger tables, then the follower memory.

Fable's thinking text isn't returned by the API by default, so only its written responses are logged (same as Opus).

### Every kept change: Claude Fable 5.2

| Exp | Hour | val_bpb | Δ vs previous best | Hypothesis (model's own words, trimmed) |
|---:|---:|---:|---:|---|
| 0 | 0.1 | 0.9960 | +0.0000 | baseline (Karpathy's train.py) |
| 1 | 0.3 | 0.9873 | -0.0088 | Halve TOTAL_BATCH_SIZE from 2**19 to 2**18 (grad accumulation 2 -> 1) so the fixed 5-minute budget yields about twice as many optimizer steps; at this short horizon the run is step-limited, so more updates should lower val_bpb mo… |
| 4 | 0.9 | 0.9850 | -0.0022 | Shrink the short attention window from 1/2 to 1/8 of context (1024 -> 256 tokens) in the six "S" layers, keeping the two full-context "L" layers: attention is a large share of compute at width 512, so cheaper local layers should… |
| 5 | 1.0 | 0.9816 | -0.0034 | Retry depth 9 at width 512 (ASPECT_RATIO 64 -> 56) now that short-window layers are cheap, with WINDOW_PATTERN "SSSSL" so there are still only two full-context layers (4 and 8) and the added layer is a 256-token one. Experiment 3… |
| 11 | 2.2 | 0.9664 | -0.0153 | Add a zero-initialised hashed bigram embedding (2**18 buckets indexed by a hash of the previous and current token) to the token embedding before the first norm. With an 8K vocab the previous token says a lot about the current sub… |
| 12 | 2.4 | 0.9631 | -0.0032 | Add a second zero-initialised hashed embedding for trigrams (2**19 buckets indexed by a hash of the two previous tokens and the current one), summed with the token and bigram embeddings before the first norm. The bigram table gav… |
| 13 | 2.6 | 0.9609 | -0.0023 | Add a causal "copy" feature: for each position, find the most recent earlier position in the same row with the same (prev, cur) bigram via a sort, and add a zero-initialised embedding of the token that followed it (reserved row w… |
| 16 | 3.4 | 0.9569 | -0.0039 | Enlarge the hashed n-gram tables 4x (bigram 2**18 -> 2**20 buckets, trigram 2**19 -> 2**21) and re-apply the near-miss schedule (WARMDOWN_RATIO 0.5 -> 0.75, FINAL_LR_FRAC 0.0 -> 0.05, about -0.001 in each of four runs). With far… |
| 20 | 4.5 | 0.9511 | -0.0059 | Add a corpus-wide "follower memory": non-gradient hash tables (int32, 2**28 buckets each, signature-checked) that store, for every 3-, 5- and 8-token context seen in training, the token that most recently followed it. Each batch… |
| 24 | 5.4 | 0.9488 | -0.0023 | Enlarge the follower-memory tables for the long contexts (order 5: 2**28 -> 2**30 buckets, order 8: 2**28 -> 2**30; order 3 stays 2**28, about +6GB VRAM). The last log shows fill of 0.69 and 0.77 from about 418M writes, so I esti… |

## Setup

- **Task:** Karpathy's autoresearch, unchanged: a ~50M-parameter GPT (`train.py`) trained for a fixed 5 minutes on one H100, scored by validation bits per byte on his ClimbMix data. The models may change anything in `train.py`.
- **Loop:** one model call per experiment. Each turn the model gets the instructions (Karpathy's `program.md` research rules, verbatim), `prepare.py`, the baseline, its current best `train.py`, the full history of its experiments, and the last run's log. It returns a one-line hypothesis and a complete new `train.py`. The harness runs it, scores it, and keeps or discards it. Crashes get up to 2 repair turns.
- **Keep rule:** kept only if val_bpb beats the model's best by more than the noise floor (0.0019 = 1 std of the baseline across 5 seeds). Karpathy's version keeps any improvement; this one is stricter so luck doesn't count as progress.
- **Fairness:** same instructions, same H100 (both models shared one GPU and took turns, so neither got a faster card), same 10-hour budget, $40 cap each, each model at its provider's default reasoning setting. Neither model saw the other's experiments.
- **Guards:** the held-out test shard doesn't exist inside the sandbox during the night. The sandbox has no network and no API keys. `prepare.py` (Karpathy's data and scoring code) is hash-checked, and the score is read from its `evaluate_bpb`, never from what `train.py` prints. Every `train.py` is statically checked before running: no reading data files, no training on the validation split, no environment variables, no patching `prepare`, exactly one call to `evaluate_bpb`. Pre-flight tests prove the sandbox and guard work (`results/preflight/`).

## Design decisions (and why)

- **A harness instead of a coding agent.** Karpathy runs an agent in a terminal. Here each model gets an identical, fixed context each turn and returns a file. That isolates research judgment from agent tooling (file browsing, shell use), makes the two models directly comparable, and lets the harness, not the model, own timing, scoring and keep/discard.
- **A noise-floor keep rule.** With ~100 tries, a "keep anything better" rule accepts lucky noise. Measuring seed variance first and requiring a margin above it means every kept change is likely real. The cost: small real gains get discarded (both models had several near-misses).
- **A held-out test shard, touched once.** Selecting the best of ~50 experiments on validation data overfits to it. Retraining each winner from scratch and scoring on unseen data checks for that. Neon's validation and test scores matched exactly; Opus's dropped by 0.001.
- **Measuring speedup, not just bits per byte.** 0.957 vs 0.973 means nothing to most people. Training the original for longer and finding where it matches each model gives an intuitive "Nx more efficient" number.
- **Verifying the hardware.** The first plan used two separate H100 boxes. Pre-flight showed one ran at half the throughput of the other (267M vs 494M tokens in 5 minutes, same card model), which would have decided the result on its own. That box was replaced by running both models on the verified box, taking turns.

## Limitations

- **One research run per model, one task.** The 3 seeds are on the final retrain, so each model's best `train.py` is measured reliably, but each model got only one overnight research run. Agent runs can vary a lot from run to run; several nights per model would be needed to rank models with confidence. This shows how these nights went, not a general ranking.
- **Fable ran on a different night** from Neon and Opus, on a different H100 box verified to match (same tokens per 5 minutes, same baseline score). In both nights two models shared one GPU, taking turns.
- **Fable's API budget cap ($70) ended its run at hour 9.1** instead of 10. Its last improvement was at hour 5.4.
- **Default settings differ:** Neon's default reasoning is high, Opus's is medium. That's how each ships, but Opus at high effort might do better (a natural follow-up).
- **Neon's model got much bigger in parameters** (1.66B, mostly sparse lookup tables), within the rules and the same 5-minute budget, using less GPU memory than Opus's version. Whether that's "the same model, improved" is a fair debate.
- **Speedup numbers are estimates:** one seed per training length, interpolated, on validation data.
- **Prompt caching rarely applied:** calls were ~10 minutes apart and the provider caches expire after ~5, so both costs are close to uncached prices.
- **Harness built with Claude**, which is also the comparison model. The instructions are Karpathy's, identical for both, and every prompt, response and file is published here.

## Reproduce

Needs one H100 80GB (or two), a paid Gemini API key and an Anthropic API key. About 12 hours, ~$60 in GPU time and ~$65 in API calls.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
uv sync
uv run python -m orchestrator.data_setup     # data, tokenizer, held-out test shard, offline attention kernel
uv run python -m orchestrator.preflight      # sandbox + guard + timeout checks, 5-seed noise floor
echo 'export GEMINI_API_KEY=...' > ~/.ai_keys && echo 'export ANTHROPIC_API_KEY=...' >> ~/.ai_keys
./scripts/start.sh neon && ./scripts/start.sh opus   # 10 hours, both share the GPU
# night 2: ./scripts/start.sh fable  (pass EPSILON=... to reuse night 1's threshold; default 0.001881)
./scripts/morning.sh                                 # held-out test, dashboard, replay video
```

Offline smoke test with a CPU stand-in and a mock model: `export AUTORESEARCH_SMOKE=1 AUTORESEARCH_SMOKE_BUDGET=8`, then `python -m orchestrator.data_setup --smoke`, `python -m orchestrator.preflight --seeds 3`, `python -m orchestrator.run_arm --arm test --provider mock --max-exps 8`.

## Repo layout

| Path | What |
|---|---|
| `train.py`, `prepare.py`, `program_karpathy_original.md` | Karpathy's autoresearch. `prepare.py` has 3 marked patches (configurable data path, held-out test shard, score written to a file). |
| `program.md` | The instructions both models got (Karpathy's research rules + the reply format). |
| `orchestrator/` | The loop, model adapters, sandbox, static guard, pre-flight, final test. |
| `analysis/` | Dashboard and replay video. |
| `scripts/` | One-command phases: setup, dry run, start, morning. |
| `results/` | Everything from both nights: every experiment (`neon/`, `opus/`, `fable/`), held-out tests (`final/`, `final_fable/`), pre-flight checks, speedup logs. Open `results/dashboard.html` or `results/dashboard_neon_vs_fable.html` for the interactive replays. |
| `media/` | Video, charts, stills (`media/fable/` for round 2). |

## License

MIT. `train.py`, `prepare.py` and `program_karpathy_original.md` are from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (MIT, © Andrej Karpathy).
