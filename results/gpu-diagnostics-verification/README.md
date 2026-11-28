# Frozen-model diagnostics verification

The diagnostic commands are implemented and CPU-verified at `4ef4ac0`. The initial real L40S
checks completed at `88c5bf0` and exposed a profiler-summary defect. The fix passes CPU checks
and replay against captured GPU events. **The corrected full GPU CLI runs are pending.**

Use the [reproduction commands](../../README.md#frozen-model-diagnostics) for the CPU toy modes,
Slurm dry runs, and L40S launches. Each diagnostic uses one frozen model and batch. It does not
train a model or establish a quality or speed advantage for adaptive allocation.

## Implementation and CPU checks

`scripts/gpu_diagnostics.py` wraps the existing profiling, accounting, and gradient utilities.
It saves manifests and raw results, keeps three timing scopes separate, checks frozen weights,
and reports unavailable populations without silently changing their recipes. The Slurm launcher
snapshots committed source. Existing outputs are preserved.

- [83 existing regression checks](tests/existing-regressions.log) passed before the additions.
  These cover profiling, gradients, accounting, roofline, A/B campaigns, and baseline summaries.
- [31 focused checks](tests/green-focused.log) passed after the corrections. These include the
  new diagnostics, existing profiler tests, A/B snapshot reuse, and A/B launch dry runs.
- [Four regressions failed before their fixes](tests/red.log). The profiler fixture reported
  CPU/device totals of `0/48` instead of the independently specified `60/12`.
- The statistical fixture rejected a [population-SD mutation](tests/statistics-mutation.log).
  Its known sample SD of `0.25` became `0.204124`, demonstrating that the check detects the
  wrong convention.
- Independent public-command checks verified [seed boundaries](cpu/alias.json), both
  [early](cpu/race-rev-parse.json) and [staging](cpu/race-archive.json) launch races,
  [A/B compatibility](cpu/ab.json), and [non-finite checkpoint rejection](cpu/nonfinite-result.json).
  Each race submitted once and preserved the winning metadata byte for byte.
- A real [CPU profile](cpu/profile-result.json) matched all seven phase durations and counts
  in its trace. Its [manifest](cpu/profile-manifest.json) matched an independent runtime probe.
  Earlier quality runs matched exactly after excluding timings; 82 seed-level aggregates
  were independently recomputed.

The [review findings](review-findings.md) record the defects and fixes. Full development logs,
CPU traces, temporary fixtures, and verification scripts remain under
`runs/gpu-diagnostics-development/` in the working checkout. Model checkpoint fixtures and
large traces are not committed here.

## Initial L40S evidence

| Check | Job | Source | Slurm result | Verification result |
|---|---:|---|---|---|
| Real-model profile | 14151783 | `88c5bf0` | Completed, `0:0` | Ordinary timing/backend evidence passed; phase CPU summary failed audit. |
| Real-model quality, population 256, seeds 0/1 | 14151784 | `88c5bf0` | Completed, `0:0` | Raw metrics and summaries passed. |
| CPU toy on a GPU allocation | 14151785 | `88c5bf0` | Completed, `0:0` | CUDA was hidden; CUDA timing values were null. |
| Profiler event probe | 14151798 | Diagnostic probe | Completed, `0:0` | Established the CPU/CUDA phase-name collision. |

The real-model checks used one NVIDIA L40S, PyTorch `2.10.0+cu128`, Python 3.11.5, bf16 DUST,
an FP32 chunked quality reference, and batch 8 by context 2,048. The model was initialized
with seed 42, with no trained checkpoint. Both commands used the same first eight FineWeb
training rows and the same model hash. See the raw [profile](initial/profile/manifest.json)
and [quality](initial/quality/manifest.json) manifests.

The script requested eight CPU threads and 64 GiB host memory. Slurm's memory-per-CPU policy
raised the task to 11 threads and allocated 12 hardware threads. The runtime manifests record
11, not eight. [Resource policy](initial/slurm-resource-policy.txt) and
[terminal accounting](initial/slurm.txt) retain the actual allocation.

The [trace audit](initial/trace-audit.json) found 1,160 memory-efficient SDPA calls and 1,160
associated CUTLASS bf16 attention kernels. PyTorch collapses nested duplicate operator nodes;
the raw trace's 2,124 outer-wrapper records include 964 nested duplicates. This observation
identifies the backend of this run and does not retroactively identify older baseline jobs.

The unprofiled pass discarded three warm-up estimates and retained five. Its median CUDA-event
time was 8.04191 seconds, range 7.99350 to 8.07684 seconds. It excludes the optimizer and data
loading. This is one diagnostic repetition on one initialized model, not a comparative benchmark.
The synchronized phase pass and profiler pass were separate. The original phase CPU summary
was wrong because a CUDA annotation overwrote each same-name CPU annotation. The
[corrected expression replay](probe-replay.json) selects the CPU durations and the associated
device kernel totals from the captured GPU events.

The [artifact audit](initial/artifact-audit.json) independently checked data/source identities,
unchanged model hashes, percentiles, quality summaries, FLOPs, and MFU arithmetic. The full
initial profile trace remains on Insomnia at
`/insomnia001/home/pts2125/hpml/runs/gpu-diagnostics-88c5bf0-20261128T063649Z/profile/diagnostics/trace.json`.
Its SHA-256 is `8070ade582144eb158dfeed6c60047c1e64e6f12c9a33b7fbccda9145b18aade`.

## Pending hardware confirmation and limitations

Corrected source `4ef4ac0` is staged at
`/insomnia001/home/pts2125/hpml/runs/gpu-diagnostics-4ef4ac0-20261128T064927Z`.
Profile job `14151824` and quality job `14151825` are pending as of 2026-11-28 01:56 EST.
The scheduler projected starts after 06:00 EST. Smaller resource requests and alternate
permitted queues offered no earlier slot. Each job has a 15-minute limit.

```bash
ssh insomnia 'squeue --start -j 14151824,14151825; sacct -X -j 14151824,14151825 -o JobID,State,ExitCode,Elapsed,NodeList'
bash runs/gpu-diagnostics-development/gpu-verification/collect-corrected.sh
```

The collection script remains in the development evidence directory. It refuses to call a
pending run complete. The existing cluster checkout remains clean on `patrick-next-steps`.

The cost formulas report approximately `2.67762e14` needed and `3.53562e14` dense-executed
FLOPs per estimate. Dividing those by the ordinary median and the L40S's dense bf16 peak gives
modeled MFU of 9.20% and 12.14%. These are analytic estimates, not complete hardware-counter
measurements. In particular, the trace exposed 32 local attention baseline recomputes omitted
by the existing formulas. A separate [accounting task](../../tasks/TODO.md) records that gap.

The full population sweep, paper comparison, larger-population resource limits, trained-checkpoint
GPU diagnostics, calibrated roofline, and multiplexing diagnostics remain unverified. Quality
timings have no discarded warm-up samples. Gradient seeds are repetitions on one model and
batch; they are not independent training runs. No adaptive-allocation performance claim follows
from these diagnostics.
