# autoresearch

This is an experiment to have the LLM do its own research on computer vision.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, dataloader, and evaluation. Do not modify.
   - `train.py` — the file you modify. RF-DETR model, optimizer, and training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/coco/` contains the COCO dataset. If not, tell the human to run `uv run prepare.py` (downloads val2017 + annotations, ~1 GB). For the full training split (~18 GB) run `uv run prepare.py --train`.
5. **Initialize results.tsv**: Create `results.tsv` with a header row and a baseline entry. The baseline results are already known from the output format section below. Do NOT re-run the baseline — just record it.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture (backbone, transformer depth/width, number of queries), optimizer, hyperparameters, batch size, loss weights, LR schedule, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation harness (`evaluate_l1`), the dataloader, and the training constants.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_l1` function in `prepare.py` is the ground-truth metric.

**The goal is simple: get the lowest val_l1.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Experiment freely: change the architecture, optimizer, hyperparameters, batch size, model size. The only constraint is that the code runs without crashing within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_l1 gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome.

**Experiment ideas for CV:**
- Increase/decrease transformer depth (`N_ENC`, `N_DEC`) and width (`D_MODEL`, `N_HEAD`)
- Try more object queries (`NUM_QUERIES`) or fewer
- Change the LR balance between backbone and transformer (`LR_BACKBONE`, `LR_TRANSFORMER`)
- Freeze the backbone entirely for the first N steps then unfreeze
- Try a different LR schedule (e.g. longer warmup, stepped decay)
- Add GIoU loss alongside L1
- Increase batch size (reduces steps but bigger gradient signal)
- Try different loss weights (`LAMBDA_L1`, `LAMBDA_CE`, `BG_WEIGHT`)
- Try a lighter backbone (ResNet-18 or MobileNet) for more steps per second

## Output format

Once the script finishes it prints a summary like this:

```
---
val_l1:           0.350000
training_seconds: 300.1
total_seconds:    340.2
peak_vram_mb:     8192.0
num_steps:        750
num_params_M:     41.3
```

Extract the key metric with:

```bash
grep "^val_l1:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	val_l1	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_l1 achieved (e.g. 0.350000) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 8.0 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1`
5. Read out the results: `grep "^val_l1:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix. Give up after a few failed attempts.
7. Record the results in the tsv
8. If val_l1 improved (lower), advance the branch keeping the git commit
9. If val_l1 is equal or worse, git reset back to where you started

**Timeout**: Each experiment takes ~5 minutes. If a run exceeds 10 minutes, kill it and treat it as a failure.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. You are autonomous. Run until manually stopped.
