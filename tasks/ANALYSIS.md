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

---

## Run 3 — midpoint experiments: profiler, Track 3 GPU sweep, Track 2 (2026-11-21)

**Framing.** Cached suffix (A) is the reference, not a result of ours: it restores an optimization the
paper used and the public code left out. Our methods are judged as cache vs cache + B vs cache + C,
and both lost in this run. Every Track 2 run below uses cached suffix, and `separate/2` in the Track 3
table is plain DUST with cached suffix at the same forward passes as the coded methods.

Slurm jobs 14040948 (baseline rerun, `ins056`) and 14040949 (`scripts/gpu_experiments.sbatch`,
`ins038`), 1 × L40S each, torch 2.10.0+cu128, bf16 autocast, 1M tokens, population 256, seed 42,
commit `6444d3c`. One seed, one repetition. Evidence: `results/midpoint-seed42/` and
`results/baseline-seed42-rerun/` (checkpoints and the Perfetto trace stay on the cluster).

### Baseline rerun

Reproduces runs 1–2: cached suffix −13.4% step time (9.296 → 8.053 s), −16.0% FLOPs, best val
6.07518 vs 6.07512, peak memory 6.63 GiB in both, max validation difference 0.000814 over 33 evals.

### Profiler — the SDPA kernel is `efficient`

`profile_step.py`, one estimate after two warm-ups, both modes. Every attention call ran
`aten::_scaled_dot_product_efficient_attention` (CUTLASS `fmha_cutlassF_bf16_aligned_64x64`), not
flash (ruled out by the explicit mask) and not math — confirms the roofline's hypothesis.
Attention ≈ 1.4 s of the suffix estimate's GPU time; the rest is dominated by elementwise traffic
(`copy_`, `mul`, `pow`), as the bandwidth-bound roofline predicts. "Command Buffer Full" stalls
(~3.3 s) mean the CPU is waiting on a full GPU queue: GPU-bound, as the double-batch test said.
Profiler overhead inflates wall time; use the profile for shares and kernel identity only.

### Track 3 on GPU — C does not beat plain DUST at equal forward passes

`multiplex_experiment.py --real`, blocks 0/3/6, σ ∈ {0.05, 0.1, 0.2, 0.4, 0.8}, 16 passes, 16 seeds,
fp32 backprop reference. Cosine to backprop, writers / hidden, compute-matched comparison:

| | cache, separate/2 (same passes) | cache + hadamard | cache + rademacher |
|---|---|---|---|
| nrep 2, block 6, σ 0.2 | 0.294 / 0.161 | 0.288 / 0.166 | 0.222 / 0.146 |
| nrep 8, block 6, σ 0.2 | 0.374 / 0.209 | 0.288 / 0.166 | 0.222 / 0.146 |
| nrep 8, block 3, σ 0.2 | 0.275 / 0.170 | 0.212 / 0.125 | 0.171 / 0.114 |

- nrep 2: a tie, at every block and σ. nrep 8 (the shipped recipes from population 1024 up):
  Hadamard is clearly behind. Raising σ to 0.8 narrows the hidden-layer gap slightly but never
  closes it. Hadamard is also 4–10% slower per estimate than separate/2.
- Matches the CPU result and the direction-count derivation (Track 3). The bf16-readout escape
  hatch does not appear. **Verdict: C is not a speedup; report it as a negative result and a
  gradient-quality diagnostic. D (B + C) is dropped.**

### Track 2 on GPU — B is worse than fixed allocation with the default settings

| | cache (fixed) | cache + B variance | cache + B agreement |
|---|---:|---:|---:|
| best val | **6.07518** | 6.15693 | 6.30006 |
| test at best val | **6.06012** | 6.14268 | 6.28338 |
| FLOPs / step (needed) | 2.678e14 | 2.785e14 (+4.0%) | 2.163e14 (−19.2%) |
| median step | 8.00 s | 8.24 s | 6.56 s |
| mean probe cosine | 0.238 | 0.285 | 0.272 |

- Loss gaps (0.08, 0.22) are 6–15× the three-seed SD of A (~0.014): one seed is enough to call it.
- `variance` (default `--allocation-objective update`) moved almost every spare draw to the token
  embedding (phase time 0.30 → 4.94 s): its learning rate (1000) multiplies its weight, as the
  Track 2 notes warned. Embedding draws rerun the whole network, so FLOPs rose.
- `agreement` moved draws to the hubs (0.58 → 3.29 s) and spent 19% fewer FLOPs, at a worse loss.
- Probe cosine rose while loss worsened: the pooled cosine is dominated by the groups B favoured,
  while loss depends on the groups it starved. Pooled cosine is not a sufficient objective.
- Not tried yet: `--allocation-objective gradient`, a per-unit floor/cap, a longer recompute
  interval, longer runs. Defaults are untuned and the run is 61 steps.

---

## Run 4 — cache + C in training, and B retune round 1 (2026-11-21)

Jobs 14048112 (`track3train`) and 14048126 (`track2b`), 1 × L40S, seed 42, 1M tokens, population 256,
cached suffix in every run. Evidence: `results/midpoint-seed42/track3train/` and `track2b/`.

### cache vs cache + C (training)

`--multiplex` needs writer/hidden draws in multiples of the code length 4, so both runs use
`--direct-multiple 4` (110 → 120 draws per type). That rounding alone moved cache's test loss from
6.060 to 6.092: recipe-level changes of this size shift loss by ~0.03 at one seed.

| | cache (rounded) | cache + hadamard |
|---|---:|---:|
| test at best val | **6.0923** | 6.1251 (+0.033) |
| FLOPs / step | 2.797e14 | 1.721e14 (−38.5%) |
| median step | 8.41 s | 5.47 s (−35.0%) |
| peak memory | 6.63 GiB | 6.82 GiB |
| probe cosine | 0.238 | 0.227 |

- C trades a third of the step time for +0.5% loss. **Not yet a win:** at nrep 2 the gradient sweep
  showed Hadamard ≈ plain DUST with half the writer/hidden draws, so the decisive control is cache with
  those draws halved, at C's FLOPs. If that run matches C's loss, C adds nothing over using fewer draws.

### B retune round 1 (vs cache, test 6.0601)

| run | test at best val | FLOPs / step | step | where the draws went |
|---|---:|---:|---:|---|
| variance, objective gradient | 6.1083 (+0.048) | −8.7% | −5.8% | off the embedding (phase −59%) onto cheaper deep units |
| variance, budget flops | 6.1984 (+0.138) | 0.0% | −0.6% | still to the embedding (phase +1448%) |
| variance, gradient + flops | 6.1226 (+0.062) | 0.0% | +4.6% | off the embedding, onto shallow writers/hidden (direct +9%) |
| agreement, budget flops | 6.2203 (+0.160) | 0.0% | +3.1% | to the hubs (phase +651%) |
| measure-only (interval 0) | **6.0532 (−0.007)** | −2.3% | −0.2% | none after step 0 |

- **No setting beats cache.** The best, objective `gradient`, gives −8.7% FLOPs for +0.8% loss, a
  worse trade than C's.
- The learning-rate weighting was the embedding bug: `gradient` fixes it, the FLOP budget alone does not.
- Every adaptive run raised probe cosine (+2% to +20%) while loss got worse: minimizing pooled gradient
  noise is not aligned with loss here. The paper's hand-tuned recipe front-loads shallow writers
  (38 draws at block 0), which the noise-based policies undo.
- Measure-only: the split-half measurement does not hurt training, and its overhead is within noise
  (−0.2% step time). It is not an exact cache replay: the step-0 floor lifts units below two chunks,
  hence −2.3% FLOPs and a max validation difference of 0.0115.

### The missing control for both B and C

Both methods are judged against one cache point. What they need is cache's **draw-count curve**: cache
with writer/hidden draws scaled 0.5× and 0.75× (loss vs FLOPs). A method helps only if its point lies
below that curve. C at −38.5% FLOPs and B (`gradient`) at −8.7% may be no better than spending fewer
draws uniformly.

### The control (job 14048409): C loses to simply using fewer draws

Cache with the rounded recipe's writer/hidden draws halved (`--direct-multiple 4 --direct-scale 0.5`,
`[20,16,8,6,4,2,2,2]`) runs exactly C's forward passes:

| | cache (rounded) | cache, half draws | cache + hadamard |
|---|---:|---:|---:|
| test at best val | 6.0923 | **6.1056** (+0.013) | 6.1251 (+0.033) |
| FLOPs / step | 2.797e14 | 1.721e14 (−38.5%) | 1.721e14 (−38.5%) |
| median step | 8.41 s | **5.35 s** (−36.4%) | 5.47 s (−35.0%) |
| peak memory | 6.63 GiB | 6.63 GiB | 6.82 GiB |

- **At identical FLOPs, half-draw cache beats C** on loss (by 0.020), step time and memory. C's whole
  saving comes from using fewer forward passes, which plain DUST gets without coding. Consistent with
  the GPU gradient sweep and the direction-count derivation. **C is closed as a negative result.**
- **The draw-count curve is flat here:** halving the writer/hidden draws costs only +0.013 test loss
  for −38.5% FLOPs at 1M tokens / 61 steps. The paper's population-256 recipe has slack in this regime.
- **B is below this bar too:** its best setting (objective `gradient`) spent −8.7% FLOPs for +0.048,
  far worse than a uniform cut. Any future B result must beat the uniform-scaling curve, not one
  cache point.
