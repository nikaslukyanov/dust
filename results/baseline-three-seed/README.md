**Cached suffix reduced median training-step time by 13.54% ± 0.13 percentage points across three seeds.**
Mean test loss was 6.06783 for full replay and 6.06719 for cached suffix. Peak training memory was
6.6305 GiB for both. This confirms a timing benefit for baseline A at population 256; these runs do
not evaluate adaptive allocation, multiplexing, or their combination.

All nine runs completed 61 updates on November 18, 2026. Each used 999,424 training prediction tokens
(8 sequences × 2,048 tokens × 61 steps), the same FineWeb files, and the same 8-layer, 512-wide model
with vocabulary 4,096. Each held-out split contained 544 sequences. Training used bf16 autocast on
one NVIDIA L40S per run, Python 3.11.5, PyTorch 2.10.0+cu128, and CUDA 12.8.

| Method | Median step time, seconds | Best validation loss | Test loss at best validation | Training time, seconds | Peak training memory, GiB |
|---|---:|---:|---:|---:|---:|
| Full replay | 9.273 ± 0.095 | 6.08209 ± 0.01450 | 6.06783 ± 0.01338 | 565.85 ± 5.76 | 6.6305 |
| Cached suffix (A) | 8.018 ± 0.077 | 6.08133 ± 0.01468 | 6.06719 ± 0.01364 | 489.22 ± 4.62 | 6.6305 |
| Backprop SGD | 0.0923 ± 0.0005 | 5.95511 ± 0.00163 | 5.94234 ± 0.00325 | 5.94 ± 0.31 | 4.8762 |

Values are means ± sample standard deviations across seeds 42, 43, and 44, not confidence intervals.
For step time, each run first discards three warm-up updates and takes the conventional median of
the remaining 58. Training time includes all 61 updates and excludes evaluation, gradient probes,
checkpoint writes, and transfers outside the update timer. Backprop uses the shipped SGD reference
settings; no hyperparameter tuning was performed for this sweep. All nine runs selected step 61.

The speedup is computed within each seed before averaging:

| Seed | DUST execution order | Full replay, seconds/step | Cached suffix, seconds/step | Step-time reduction | Largest absolute validation-loss difference |
|---|---|---:|---:|---:|---:|
| 42 | Full, then suffix | 9.374817 | 8.095167 | 13.6499% | 0.000814 |
| 43 | Suffix, then full | 9.257455 | 8.017578 | 13.3933% | 0.003874 |
| 44 | Full, then suffix | 9.187279 | 7.941186 | 13.5632% | 0.001666 |

The largest validation difference across 99 paired evaluation points was 0.00387352. The largest
absolute difference in selected-checkpoint test loss was 0.00151657. These are small observed
differences, not a formal equivalence test or evidence of a quality advantage. GPU trajectories are
not bit-identical. Backprop achieved lower loss and much lower update cost at this token budget;
the suffix optimization improves DUST relative to its full-replay implementation.

Current analytic cost conventions are recorded separately below. They differ from the hybrid
attention count used by the earlier [single-seed run](../baseline-test/README.md).

| Method | Needed FLOPs/update | Dense executed FLOPs/update | Needed forward equivalents/update |
|---|---:|---:|---:|
| Full replay | 3.185898e14 | 4.209068e14 | 285.2508 |
| Cached suffix | 2.677622e14 | 3.535617e14 | 239.7421 |
| Backprop | 3.350628e12 | 4.329327e12 | 3.0000 |

Both DUST modes use the same 256 direct-loss draws and 1,616 total draws per update. FLOPs are
analytic estimates, not hardware-counter measurements; the backprop estimate assumes a backward
pass costs twice a forward. The SDPA kernel used in these jobs was not identified by a profiler.

**Audit and provenance.** Training used the unmodified snapshot of commit
`0723bbadf2e4e86d3731b862c3aa142b45750d17`. Summary code and its tests were added in
`615f65a`; they did not change the running training snapshot.

- Setup passed 26 CPU checks. The subsequent complete audit suite passed **144 tests**, including
  upstream training equivalence, suffix execution, accounting, checkpoint/resume, tracking, and
  the ten new sweep-summary tests. One profiler warning is retained in the audit log.
- All six GPU preflight checks passed on each of the three GPUs, including numerical equivalence
  and exact checkpoint/resume checks.
- The raw-record audit checked all 549 updates and 297 validation evaluations: finite metrics,
  complete step sequences, validation-best checkpoint selection, saved histories, training source
  hashes, token counts, cumulative timing/GPU-hours, analytic costs, and seven gradient probes per
  DUST run. The matched model/data/configuration checks also passed.
- Every run produced a distinct, nonempty offline W&B artifact under project `dust-hpml`.
  There is no online dashboard yet; a team login and an explicit sync are still needed.
- No training implementation failure was found and no GPU rerun was necessary.

| Seed | GPU job | Node | Slurm elapsed | Status |
|---|---:|---|---:|---|
| 42 | 13991787 | ins057 | 29m 43s | COMPLETED, 0:0 |
| 43 | 13991788 | ins057 | 29m 29s | COMPLETED, 0:0 |
| 44 | 13991789 | ins060 | 29m 16s | COMPLETED, 0:0 |

The three jobs ran concurrently, with separate GPUs for seeds 42 and 43 on the same node.
Each seed's three methods ran sequentially on its allocated GPU. Total allocated GPU time was
**1.47444 GPU-hours**, including preflight, evaluation, startup, and logging. Training updates alone
used **0.88418 GPU-hours**. Setup job 13991786 and audit job 13992283 used CPUs only.

These results cover one model size, one population, three seeds, one repetition per seed, and the
instrumented training path. Execution order alternated across seeds (two full-first, one
suffix-first). They do not establish behavior at other budgets or hardware configurations.

**Saved evidence and reproduction.** [summary.json](summary.json) contains the seed-level aggregates;
[audit.json](audit.json) contains the additional record checks and per-update costs.
[run_metadata.json](run_metadata.json) records source commits, Slurm accounting, the original run
directory, and SHA-256 hashes for 79 evidence files. Each `seed-*/` directory contains raw configs,
metrics, results, console logs, environment and preflight reports, and W&B run IDs. Model checkpoints
and W&B binary logs remain in `runs/baseline-gpu-20261118-172822/` on the cluster.

Recompute the summary from the committed evidence using only the Python standard library:

```bash
python3 scripts/check_baseline_results.py results/baseline-three-seed --sweep
git diff --exit-code -- results/baseline-three-seed/summary.json
```

The summary command refuses incomplete jobs, repeated seeds, and mismatched model/data/GPU or
optimizer configurations. It computes each seed's speedup separately rather than treating the
individual training steps as independent repetitions. Its red/green tests are reproducible with:

```bash
.venv/baseline-cu128/bin/python -m pytest -q scripts/test_check_baseline_results.py
```

The original launch command was:

```bash
WANDB_MODE=offline bash scripts/launch_baseline.sh \
  --seeds '42 43 44' --order alternate --wandb dust-hpml
```
