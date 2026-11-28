# DUST

This is a minimal implementation of our research: <https://qlabs.sh/research/dust>

Train a transformer using forward evaluations and SGD, without backpropagation.
This minimal implementation preserves the paper's estimator and tuned defaults
while omitting execution optimizations used in the full experiments.

[`model.py`](model.py) contains the model, and [`dust.py`](dust.py) the algorithm and training loop. A simple
[backpropagation SGD baseline](baselines/backprop.py) uses the same model and data.

## Installation

Requires Linux, Python 3.10+, and an NVIDIA GPU with bfloat16 support.
The following installs PyTorch with CUDA 12.8; use a compatible NVIDIA driver.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

## Data

Prepare the 1M and 10M training sets:

```bash
python prepare_data.py
```

This streams FineWeb from Hugging Face and saves `data/train_1M.pt`,
`data/train_10M.pt`, and shared `data/val.pt` and `data/test.pt` files.
Use `--tokens 1M` or `--tokens 10M` to prepare only one training budget.
Each held-out split contains 544 sequences of 2,048 prediction tokens.

The fixed BPE-4096 tokenizer is included; retraining is optional.
Its training recipe and script are in [`tokenizer/`](tokenizer/README.md).

## DUST

Train with the default population of **16,384** on eight GPUs:

```bash
torchrun --standalone --nproc_per_node=8 dust.py --tokens 1M --output runs/1m
torchrun --standalone --nproc_per_node=8 dust.py --tokens 10M --output runs/10m
```

For a smaller, single-GPU run:

```bash
python dust.py --tokens 1M --population 256 --output runs/1m-p256
```

Supported populations are **256, 1024, 4096, and 16384**, using respectively
**1, 1, 4, and 8 GPUs** in the paper configurations. Each selects its tuned
per-layer draw allocation, credit decay and optimizer settings. Population counts direct-loss
draws across all GPUs; head and local attention draws are additional. Every GPU
processes the same batch and contributes different perturbations.

Both token budgets use an 8-layer, 512-wide transformer, sequence length 2,048,
and batch size 8. Add `--steps 2` for a short check using the selected dataset.

## Backpropagation baseline

Run SGD backpropagation on one GPU:

```bash
python baselines/backprop.py --tokens 1M --output runs/bp-1m
python baselines/backprop.py --tokens 10M --output runs/bp-10m
```

## Fixed versus adaptive allocation

Run a CPU toy campaign with the standard Python installation of PyTorch.

```bash
OMP_NUM_THREADS=1 python3 scripts/ab_experiment.py run --toy --steps 5 --seeds 42 --output runs/ab-toy
python3 scripts/ab_experiment.py summary --output runs/ab-toy
```

On Insomnia, stage the CUDA wheelhouse and prepare the FineWeb data from the repository root.

```bash
python3 scripts/stage_baseline_wheels.py .venv/wheelhouse-cu128
python3 prepare_data.py --tokens 1M
```

The Slurm setup job installs the staged wheels. The dry run requires PyTorch and prints the
matrix and Slurm commands. Use the existing CUDA environment on Insomnia or a local Python
environment with PyTorch.

```bash
.venv/baseline-cu128/bin/python scripts/launch_ab.py --dry-run --seeds 42 43 44 --output runs/ab-gpu
```

Submit the campaign from an Insomnia login node.

```bash
python3 scripts/launch_ab.py --seeds 42 43 44 --output runs/ab-gpu
```

The launch snapshots a clean commit, verifies the token files and tokenizer, and schedules one L40S
job per seed. Each seed runs the seven cases sequentially at 1M tokens, population 256, and 61 steps.
The case order rotates by seed. GPU campaign validation is pending.

Repeat the same run or launch command to resume. The runner skips verified cases and uses
`state.pt` to resume an interrupted case. It retains attempts that stop before their first
checkpoint and restarts those cases from their original seed. Keep the same source, data, seeds,
settings, and runtime when resuming. Run the summary after every GPU job completes.

```bash
.venv/baseline-cu128/bin/python runs/ab-gpu/source/scripts/ab_experiment.py summary --output runs/ab-gpu
```

The manifest records each requested and effective starting recipe. The stock fixed baseline,
adaptive variance, and adaptive agreement share a requested recipe. Each policy also gets a
measurement-only run at interval zero and a fixed control that uses the same adjusted starting
recipe. The draw count stays fixed. The FLOP estimates charge each step's live allocation.

`summary.json` reports validation loss, test loss at the validation-selected checkpoint,
allocation history, training time, and step-time percentiles after three warm-up steps.
It reports cumulative draws and analytic needed and dense executed FLOPs. The measurement-only
comparison uses the matched control's step and training times to estimate measurement overhead.
`allocation_seconds` includes allocation statistics aggregation and the allocation update inside
training time. The summary computes each difference per seed, then reports the mean and sample
standard deviation across seeds. Use toy results to verify the pipeline.

## Outputs

Each run saves its configuration, metrics, a validation-selected `best.pt`, and
`result.json` containing the selected checkpoint's test loss. Use a new output
directory for each run. Data, checkpoints, and logs are stored locally.

## License

Released under the [MIT License](LICENSE).
