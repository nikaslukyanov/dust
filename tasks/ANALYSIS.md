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

- **The kernel is not established by these timings.** The `math` model overpredicts the measured
  time, suggesting `efficient` as a hypothesis. This analytic roofline is not a calibrated hardware
  bound, so it cannot rule out a backend by itself; confirm with the profiler. The earlier A100
  comparison used different hardware and cannot identify the L40S kernel either.
- **Bandwidth-bound:** the model puts 91% of the time in memory traffic (rotary, norms, elementwise
  ops), not matmuls. Weights are 0.6% of bytes.
- **MFU ≈ 12%:** 3.54e14 executed FLOPs / 8.06 s = 44 TFLOP/s of 362. The model's best case is 18.6%.

### FLOP counts (added 2026-11-18)

Run 1 logged one mixed attention count (3.016e14 FLOPs/step, suffix). The code now logs two:
- **needed** (`flops`): only the word pairs attention is allowed to use. 2.68e14 per suffix step.
- **executed** (`flops_executed`): the dense-attention count used by the model and checked against
  PyTorch's FlopCounterMode. 3.54e14. This remains an analytic count, not hardware-counter evidence.

The two conventions imply ~24% nominal mask overhead in the dense count. Suffix saving is 16.0% under
both counts. Use *needed* to compare methods, *executed* for MFU and timing.
- Measured is ~1.55× the floor in both modes, so the suffix saving is close to what the model predicts.

---

## Run 2 — three-seed baseline and backprop reference (Patrick, 2026-11-18)

GPU jobs 13991787/13991788/13991789 completed with exit 0 on L40S GPUs. Each seed (42/43/44) ran
full replay, cached suffix, and backprop at 999,424 training tokens; the DUST execution order was
full-first / suffix-first / full-first. Source commit `0723bba`. Full report, raw records, hashes,
and Slurm provenance: [`results/baseline-three-seed/`](../results/baseline-three-seed/README.md).

| Metric, mean ± sample SD across seeds | Full replay | Cached suffix | Backprop SGD |
|---|---:|---:|---:|
| Per-run median step time, seconds | 9.273 ± 0.095 | 8.018 ± 0.077 | 0.0923 ± 0.0005 |
| Best validation loss | 6.08209 ± 0.01450 | 6.08133 ± 0.01468 | 5.95511 ± 0.00163 |
| Test loss at best validation | 6.06783 ± 0.01338 | 6.06719 ± 0.01364 | 5.94234 ± 0.00325 |
| Peak training memory, GiB | 6.6305 | 6.6305 | 4.8762 |

- Paired step-time reduction: **13.54% ± 0.13 percentage points**, range 13.39–13.65%. These are
  seed-level standard deviations, not confidence intervals. Step medians discard the first three
  updates; all nine runs completed 61 updates and selected the final checkpoint by validation loss.
- Largest absolute validation difference between the DUST modes: **0.003874**, over 99 paired
  evaluations. Selected-checkpoint test losses differ by at most **0.001517**. This supports similar
  observed learning behavior; it is not a formal equivalence test or evidence that suffix improves loss.
- Backprop is the quality and compute reference, and performed better on both here. This sweep
  establishes the suffix benefit within DUST; B/C/D still need their own experiments.
- All 18 GPU preflight checks passed. Setup passed 26 CPU tests, and the full audit suite passed
  **144 tests**. Raw histories, checkpoint selection, data/source hashes, cumulative counters, and
  nine distinct offline W&B artifacts were audited. No training rerun was necessary.
- Allocated GPU time: **1.47444 GPU-hours** across the three concurrent jobs; training updates alone:
  **0.88418 GPU-hours**. Each job took 29m16s–29m43s including preflight and evaluation.
- W&B logging works offline under `dust-hpml`; a team login and sync remain necessary for an online
  dashboard. The actual SDPA kernel still needs profiling. Hardware/model/population scaling and
  broader training conclusions remain outside this three-seed, 1M-token comparison.
