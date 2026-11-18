# DUST project — analysis of results so far

Findings from runs. `TODO.md` holds only work items and points here. Raw evidence for each run is in
`results/`.

---

## Run 1 — GPU baseline (Patrick, 2026-11-13)

Slurm job 13891039. 1 × NVIDIA L40S (48 GB), torch 2.10.0+cu128, bf16 autocast. 1M tokens,
population 256, seed 42, batch 8 × 2,048, 61 steps per run, first 3 steps discarded. Full replay
then cached suffix, same GPU. Merged in PR #8; evidence and full report in `results/baseline-test/`.

| | full replay | cached suffix | change |
|---|---|---|---|
| median step time | 9.319 s | 8.061 s | −13.5% |
| training time | 568.3 s | 491.7 s | −76.6 s |
| analytic FLOPs / step | 3.567e14 | 3.016e14 | −15.5% |
| best val / test at best val | 6.0755 / 6.0605 | 6.0752 / 6.0601 | same within 0.006% |
| peak training memory | 6.63 GiB | 6.63 GiB | none |

### What it told us

- **Cached suffix (A) works on GPU.** −13.5% step time for −15.5% FLOPs.
- **Loss unchanged.** Max validation difference 1.3e-3 over 33 evaluations; not bit-identical under
  bf16 (bit-identical holds on CPU fp32 only).
- **No memory saving.** Peak 6.63 GiB in both modes.
- **Direct draws dominate.** Suffix step 8.06 s = direct 5.82 (72%), hubs 0.59, local 0.57,
  scalars 0.47, embedding 0.30, head 0.28, clean 0.03. Savings came only from direct (−15.3%) and
  hubs (−25.8%).
- **GPU-bound, not launch-bound.** Doubling batch: 8.04 → 17.58 s per step, ratio 2.19, verdict
  `gpu-bound` (`results/baseline-test/preflight/preflight.json`). FLOP cuts turn into time cuts.
- **All six preflight checks passed**, including resume (weights and generator state identical).
- Mean gradient-probe cosine vs backprop: 0.238 at population 256 (both modes).

### Caveats

- One seed, one repetition, full replay always ran first.
- The sbatch constraint allowed `h100|l40s|l40|A6000`; this run happened to land on an L40S.
- Which attention (SDPA) kernel ran is not measured yet (profiler pass in TODO). See the L40S
  roofline below for what the model implies.
- Not covered: backprop reference, W&B, B / C / D.

### Roofline on the L40S (added 2026-11-18)

`roofline.py` now has the L40S (NVIDIA datasheet, dense: 362 TFLOP/s bf16, 864 GB/s). Predictions
are lower bounds (best case); measured numbers from run 1.

| | predicted floor | measured | measured ÷ floor |
|---|---|---|---|
| cached suffix, `efficient` attention kernel | 5.23 s | 8.06 s | 1.54 |
| cached suffix, `math` attention kernel | 14.33 s | 8.06 s | 0.56 |
| full replay, `efficient` kernel | 5.97 s | 9.32 s | 1.56 |
| suffix saving | 12.3% | 13.5% | |

- **The `math` kernel did not run:** the step was faster than its floor. (The earlier "matches the
  math kernel" hint came from the A100 model and was wrong.) Most likely `efficient`; confirm with the profiler.
- **Bandwidth-bound:** the model puts 91% of the time in memory traffic (rotary, norms, elementwise
  ops), not matmuls. Weights are 0.6% of bytes.
- **MFU ≈ 12%:** 3.54e14 executed FLOPs / 8.06 s = 44 TFLOP/s of 362. The model's best case is 18.6%.

### FLOP counts (added 2026-11-18)

Run 1 logged one mixed attention count (3.016e14 FLOPs/step, suffix). The code now logs two:
- **needed** (`flops`): only the word pairs attention is allowed to use. 2.68e14 per suffix step.
- **executed** (`flops_executed`): what the GPU really computes, every pair, masked or not. 3.54e14.

So ~24% of each step's math is computed and thrown away by the mask. Suffix saving is 16.0% under
both counts. Use *needed* to compare methods, *executed* for MFU and timing.
- Measured is ~1.55× the floor in both modes, so the suffix saving is close to what the model predicts.
