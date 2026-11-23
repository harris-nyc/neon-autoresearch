# autoresearch

This is an experiment to have the LLM do its own research.

(Adapted from Karpathy's program.md, kept verbatim in program_karpathy_original.md. The research
rules below are his. What changed: a harness runs the loop for you, so there are no git, shell or
results.tsv steps. You propose one experiment per turn by returning a new train.py.)

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation).

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already installed (the imports in the baseline train.py).
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric. Call it exactly once, at the end, as the baseline does.
- Touch the data or the evaluation outside prepare's public API: no reading data files directly, no `make_dataloader(..., "val")` for training, no network, no environment-variable reads, no modifying `prepare`'s attributes. The harness rejects files that do.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically. The GPU is one H100 80GB.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

## How the loop works here

- The harness trains your train.py for 5 minutes and reads val_bpb from `evaluate_bpb`.
- A change is **kept** if it lowers val_bpb by more than the noise floor (run-to-run variation of the baseline, given below). Otherwise it is discarded and you continue from the best version so far.
- If a run crashes or exceeds 10 minutes in total, it fails. On a crash you get the error and up to two chances to fix it.
- You see the full history of experiments each turn. Learn from it. If you feel stuck, think harder: re-read the code for new angles, combine previous near-misses, try more radical architectural changes.

## Reply format

Propose ONE experiment. Reply in exactly this format and nothing else:

<hypothesis>one line: what you're changing and why you expect it to lower val_bpb</hypothesis>
<train_py>
...complete contents of the new train.py...
</train_py>
