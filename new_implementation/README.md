# new_implementation/ — DUST gradient-check, accounting & profiling

Our team's additions to the DUST codebase for the HPML project. Everything here is **new** or
**additive**. The original training path (`../dust.py`, `../model.py`, `../prepare_data.py`) is
untouched except for one in-place refactor noted below, so the run commands in the top-level
`../README.md` still work exactly as before.

## What DUST is (30 seconds)

DUST trains a transformer **without backpropagation**. Instead of a backward pass, it jiggles the
network's activations with small random noise, runs **forward passes only**, and keeps the noise that
lowered the loss. Averaging the reward-weighted noise over many draws (a "population") gives an
estimate of the gradient, which a normal optimizer then applies. One training step is therefore
thousands of tiny forward passes rather than one forward + one backward.

Two consequences drive everything in this folder:

1. The gradient is an **estimate**, so we need a way to check how good it is.
2. The step is **many tiny kernels**, so we need to measure performance carefully — it is likely
   limited by CPU kernel-launch overhead, not GPU math, which changes what "faster" even means.

## The files

- **`gradient_check.py`** — the **ruler**. Computes the true backprop gradient and scores how close
  DUST's noisy estimate is (direction via cosine, magnitude via relative norm), per module group, plus
  variance/bias over seeds and the paper's "quality improves with more draws" curve. It checks one
  step's gradient fidelity only; comparing training/held-out loss over a run is the experiment
  campaign, not this file. (Renamed from `diagnostics.py` — the old name implied a grab-bag; this does
  one thing.)
- **`accounting.py`** — the **cost ledger**, two halves. *Analytic:* counts the arithmetic of one
  update by formula, from the model's dimensions and the per-group draw counts — nothing runs — and
  reports it in "forward-pass equivalents" (how many clean forward passes' worth of work one update
  costs), so variants A/B/C/D are comparable on paper before any GPU time, and memory can be sized.
  *Measured:* times real steps the right way (CUDA events, warmup discarded, median/p10/p90) and flags
  launch-bound steps.
- **`roofline.py`** — the **roofline**: accounting's FLOPs paired with bytes moved, kernel launches and
  host syncs, per op and per phase, placed on an A100/H100 roofline to name each phase's regime
  (compute / bandwidth / launch) and the best-case MFU; plus `mfu()` and `amdahl()` for measured
  numbers. `python roofline.py --population 256 [--attention math] [--full-replay] [--json]`.
  `test_roofline.py` pins its matmul FLOPs to a real `DUST.estimate` under `FlopCounterMode`.
- **`profiling.py`** — the **torch.profiler harness**: the per-op / per-phase table and a Perfetto
  trace, plus a double-batch test that says whether the step is GPU-bound or launch-bound before we
  trust any savings number.
- **`checkpoint.py`** — **full-state save/resume**: weights, optimizer momentum, the perturbation RNG,
  the step counter, and the adaptive-allocator state, written atomically, one file per worker. (The
  stock code saves weights only, which cannot resume a run cleanly.)
- **`groups.py`** — the **shared vocabulary**: which parameters belong to which perturbation "group",
  which later layers a perturbation reaches, and how many draws each group gets. One source of truth
  so the three tools above never disagree. No torch dependency.
- **`run_metrics.py`** — the **training-time log**: the per-step record `dust.py` and
  `baselines/backprop.py` write to `metrics.jsonl` — cumulative analytic FLOPs and forward-equivalents
  (from the live Recipe, so a changing allocation is charged what it ran), cumulative draws, training
  seconds / GPU-hours with eval, probes and checkpoint writes **excluded**, honest total wall clock,
  step time and peak memory, and the draw allocation whenever it changes. `gradient_probe` scores the
  estimate against backprop on a held-out batch without disturbing the perturbation RNG.
- **`allocation.py`** — **adaptive allocation (method B)**: re-decides during training how many draws
  each part of the model gets, on a fixed budget (draws or FLOPs), from a measurement that costs no
  extra forward pass (`DUST.direct_errors(split=True)` compares two halves of the draws already run).
  Two policies: `variance` (draws in proportion to per-draw noise) and `agreement` (equalize the
  cosine between half-population estimates). `python dust.py --adaptive [--allocation-policy ...]`.
  No torch dependency.
- **`multiplexing.py`** — **structured multiplexing (method C)**: the sign codes (Hadamard columns
  chosen so that the baseline, each group's own curvature and every second-order cross term cancel;
  random signs for comparison) and the decoder. `DUST.multiplexed_errors` uses them to estimate a
  block's writers and hidden layer from shared forward passes: `python dust.py --multiplex hadamard`.
- **`multiplex_experiment.py`** — method C's comparison on one block: separate vs joint vs Hadamard vs
  Rademacher, scored against backprop over seeds. Runs on CPU in seconds.
- **`toy.py`** — a **toy setup** (4×64 model, tiny recipe, synthetic data with a learnable next-token
  rule) so the full training path runs on a CPU in seconds: `python dust.py --toy`,
  `python baselines/backprop.py --toy`. For checking plumbing, not for results.
- **`gpu_preflight.py`** — run this first on a GPU: a few minutes on the real model that exercise the
  CUDA paths the CPU tests cannot (bf16 suffix-vs-full-replay equivalence, timing per phase and peak
  memory in both modes, the gradient probe, checkpoint/resume with the CUDA generator, optionally the
  launch-bound test). Writes `preflight.json`; exits 1 on a failed check. See "Running the baseline".
- **`compare_runs.py`** — side-by-side table of finished runs (loss, FLOPs, step time, per-phase time,
  peak memory), the first run as the reference.
- **`test_upstream.py`** — checks the original public code out of git history (commit `b20f7c0`) and
  shows `DUST(suffix=False)` trains to bit-identical weights: `--full-replay` *is* the upstream code
  with our logging.
- **`test_*.py`** — CPU tests (pytest): `python -m pytest new_implementation/` from the repo root.
  Each checks against an independent truth (an uninterrupted run, finite differences, a known
  curve) on a tiny model, so they need no GPU and run in seconds.

## What changed in the original code

- **`../dust.py`** — the one in-place change. `DUST.step` was split into `estimate()` (the forward
  passes that produce a gradient, no weight change) + `apply()` (normalize, safety-check, optimizer
  step), with a thin `step()` wrapper that is bit-identical to the old fused step. Each phase of
  `estimate()` is tagged with `record_function('dust/<phase>')` so a profiler trace reads in DUST
  terms (clean / direct / embedding / hubs / head / local / scalars). This split is what lets
  `gradient_check.py` run the estimate without ever touching the weights.
- **Cached-suffix execution (Track 1, baseline A).** `../model.py` gains `GPT.forward_from(start, x,
  x0, idx, ...)` (the full `forward` is `forward_from(0, ...)` and can record the residual stream
  entering each block). `../dust.py` caches those streams in the clean pass, and a direct draw at block
  l reruns only blocks l onward; the token embedding still reruns everything. `--full-replay` restores
  the public code's whole-network replay for the upstream reference. `test_suffix.py` is the CPU
  equivalence gate (`python -m pytest new_implementation/test_suffix.py`).
- **Training loop → `train()`, with logging and checkpoint/resume.** `../dust.py`'s loop moved out of
  `main()` into `train(config, settings, training, validation, testing, output, ...)` — same steps in
  the same order (`test_training.py` checks it ends at the weights of a bare `DUST.step` loop). It now
  writes `run_metrics` records, saves `state.pt` every `--checkpoint-interval` steps, continues with
  `--resume` (refusing a different model/recipe/seed/world/data, and rewinding `metrics.jsonl` to the
  checkpoint), and probes gradient cosine every `--probe-interval` steps. `--toy` and `--device` allow
  CPU smoke runs. `elapsed_seconds` used to include evaluation time; `train_seconds` now excludes it.
- **`../baselines/backprop.py`** — same refactor into `train()` and the same `run_metrics` records (an
  update charged as 3 forwards), plus `--toy` / `--device`. No checkpoint/resume (not needed yet).

## Running the baseline (upstream vs cached suffix)

The baseline is the original public code; `--full-replay` reproduces it bit-for-bit (`test_upstream.py`)
while writing the same records as every other variant. Both runs consume the same noise, so their loss
curves agree up to bf16 rounding and the comparison is time, memory and FLOPs at equal quality.

```bash
python prepare_data.py --tokens 1M
python new_implementation/gpu_preflight.py --population 256 --launch-bound --output runs/preflight
python dust.py --tokens 1M --population 256 --full-replay --probe-interval 10 --checkpoint-interval 10 --output runs/1m-p256-upstream
python dust.py --tokens 1M --population 256               --probe-interval 10 --checkpoint-interval 10 --output runs/1m-p256-suffix
python new_implementation/compare_runs.py runs/1m-p256-upstream runs/1m-p256-suffix
```

Run them one after the other on an otherwise idle GPU. A run that dies continues with the same command
plus `--resume`. Analytic cost per update (forward-equivalents, full replay -> cached suffix):
population 256: 288 -> 244 (15.5% fewer FLOPs); population 1024: 1102 -> 924 (16.2%).

## How it wires together

`accounting` imports `groups`; `profiling` and `run_metrics` import `accounting`; `gradient_check`
imports `groups` and reaches back to `../dust.py` for one helper (`average_ranks`) through a small
`sys.path` shim. `allocation` imports `accounting` and `groups`; `multiplexing` only torch.
**`../dust.py` now imports from here** (`allocation`, `checkpoint`, `groups`, `multiplexing`, `run_metrics`; `toy` only under
`--toy`) by appending this folder to `sys.path`, so this folder can no longer be moved out without
breaking training. `run_metrics` imports `gradient_check` lazily to avoid an import cycle with `dust`.

## Where the rest lives

- `../tasks/PROJECT_PLAN.md` — the full project plan.
- `../tasks/TODO.md` — done vs. remaining, per track, in plain language.
- `../tasks/lessons/` — distilled lecture notes (how to profile and diagnose correctly) and the
  literature math verdicts for the two novel methods.
