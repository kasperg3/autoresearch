# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, COCO data prep, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. RF-DETR model, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains COCO images and a split file. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with header row and baseline entry. The baseline results are already known from the output format section below. Do NOT re-run the baseline — just record it.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, and training constants (time budget, image size, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_l1` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_l1.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_l1 gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

**Experiment ideas** (CV-specific):
- Adjust `D_MODEL`, `N_ENC_LAYERS`, `N_DEC_LAYERS` for different model capacity
- Try different `LEARNING_RATE` and `BACKBONE_LR_SCALE` values
- Change `BATCH_SIZE` (careful with OOM)
- Try `PRETRAINED_BACKBONE = False` (random init backbone — probably worse but worth knowing)
- Adjust `WARMUP_RATIO`, `WARMDOWN_RATIO`, `FINAL_LR_FRAC`
- Tune `CLIP_GRAD_NORM`
- Modify the loss weights in `compute_loss` (e.g. the 5.0 L1 and 2.0 GIoU multipliers)

## Output format

Once the script finishes it prints a summary like this:

```
---
val_l1:           0.123456
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     12345.6
num_steps:        500
num_params_M:     43.5
d_model:          256
n_enc_layers:     6
n_dec_layers:     6
```

Note that the script is configured to always stop after 5 minutes. You can extract the key metric from the log file:

```
grep "^val_l1:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_l1	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_l1 achieved (e.g. 0.123456) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_l1	memory_gb	status	description
a1b2c3d	0.123456	12.0	keep	baseline
b2c3d4e	0.118200	12.1	keep	increase LR to 2e-4
c3d4e5f	0.130000	12.0	discard	remove pretrained backbone
d4e5f6g	0.000000	0.0	crash	batch size 16 (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_l1:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv
8. If val_l1 improved (lower), you "advance" the branch, keeping the git commit
9. If val_l1 is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

