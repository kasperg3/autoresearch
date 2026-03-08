# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real **object detection** training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model.

The setup is built around an **RF-DETR** model (ResNet-50 backbone + DETR transformer encoder-decoder) trained on COCO 2017. The metric is **val_l1** — the mean L1 distance between matched predicted and ground-truth bounding boxes after Hungarian assignment. Lower is better.

The core idea: you are not touching Python files like a normal researcher. Instead, you program the `program.md` Markdown file that provides context to the AI agent and sets up your autonomous research org.

## How it works

The repo only has three files that matter:

- **`prepare.py`** — fixed constants, one-time data download (COCO 2017 val), dataloader, and evaluation (`evaluate_l1`). Not modified by the agent.
- **`train.py`** — the single file the agent edits. Contains the full RF-DETR model, optimizer, and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup), regardless of the details of your compute. The metric is **val_l1** (validation mean L1 box distance) — lower is better.

## Quick start

**Requirements:** A single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download COCO 2017 val data and annotations (one-time, ~1 GB)
uv run prepare.py

# Optionally also download train2017 for a larger training set (~18 GB extra)
# uv run prepare.py --train

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If the above commands all work, your setup is working and you can go into autonomous research mode.

## Running the agent

Spin up your Claude/Codex or whatever agent you want in this repo (disable all file-write permissions outside the repo), then prompt something like:

```
Hi, have a look at program.md and let's kick off a new experiment! Let's do the setup first.
```

The `program.md` file is essentially a super-lightweight "skill".

## Project structure

```
prepare.py      — constants, data download + runtime utilities (do not modify)
train.py        — RF-DETR model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your platform. This makes experiments directly comparable regardless of what the agent changes.
- **Self-contained.** No external dependencies beyond PyTorch, torchvision, scipy, and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## License

MIT
