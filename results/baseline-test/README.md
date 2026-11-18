# DUST GPU baseline comparison

**Cached suffix reduced training time by 13.5%, with essentially unchanged loss and peak training memory.** It saved 76.61 seconds over the 61-step run, corresponding to a 1.156x training speedup, or about 15.6% higher throughput.

Slurm job **13891039** completed successfully with exit code **0** in **29 minutes 41 seconds** on November 13, 2026. Both execution modes completed all 61 steps and selected their best validation checkpoint at step 61.

## Experiment configuration

| Setting | Value |
|---|---|
| GPU | One NVIDIA L40S on node `ins039` |
| Software | Python 3.11.5; PyTorch 2.10.0+cu128; CUDA 12.8 |
| Data | Verified FineWeb sample-10BT subset; 999,424 training prediction tokens |
| Model | 8 layers, width 512, vocabulary 4,096, context length 2,048 |
| Batch size and steps | 8 sequences; 61 steps per run |
| Population and seed | 256; seed 42 |
| Execution order | Full replay, then cached suffix, sequentially on the same GPU |
| Probes and checkpoints | Every 10 steps and at the final step |
| Other variants | Adaptive allocation, multiplexing, and W&B disabled; no backprop SGD baseline |
| Source snapshot | `506f7a31ea3370156c96f1cf61e4b842eca853af` |
| GPU wall-time limit | 10 hours; a scheduler safety cap |

The saved configurations match except for execution mode. Setup passed 26 CPU tests. All six GPU preflight checks passed: environment, equivalence, timing, probe, training/resume, and doubled-batch performance. The preflight resume check reproduced identical weights and random-generator state.

The experiment snapshot included a two-line preflight fix that resolves bare `cuda` to an explicit device index before calling `torch.cuda.set_device()`. Training code was unchanged by this fix. Source provenance, dataset hashes, and verification details are recorded in [run_metadata.json](run_metadata.json).

## Loss and performance results

| Metric | Full replay | Cached suffix | Suffix difference |
|---|---:|---:|---:|
| Completed steps | 61 | 61 | Same workload |
| Best validation loss | 6.075500 | 6.075183 | -0.000317 |
| Test loss at best validation checkpoint | 6.060480 | 6.060117 | -0.000363 |
| Final training loss | 5.88333 | 5.88235 | -0.00098 |
| Median host step time | 9.319 s | 8.061 s | 13.50% less time |
| Median GPU step time | 9.322 s | 8.062 s | 13.5% less time |
| Reported training time | 568.29 s | 491.68 s | 76.61 s saved; 13.48% less time |
| Estimated FLOPs per step | 3.56717e14 | 3.01592e14 | 15.45% fewer |
| Forward-equivalents per step | 288.337 | 243.779 | 15.5% fewer |
| Peak training memory | 6.63046 GiB | 6.63046 GiB | Unchanged |

Across **33 evaluations**, the largest absolute validation-loss difference was **0.0012843**. Best validation and test losses differed by less than 0.006% of the full-replay loss. Mean probe cosine was 0.237931 for full replay and 0.237821 for cached suffix.

These measurements support similar learning behavior for this seed and workload. The small loss improvements do not establish a quality advantage, and the two training trajectories are not bit-identical.

Losses and host medians above use [summary.json](summary.json). GPU timing, phase timing, FLOPs, and final training loss use [comparison.txt](comparison.txt). Both summary methods exclude the first three steps. The JSON summary uses the conventional median, whereas the comparison script selects the upper middle observation; this explains their slightly different host medians. The timing object in `result.json` uses another percentile convention. These small aggregation differences do not change the conclusion.

FLOPs are analytic estimates. Memory fields are named `gb` in the source, but the code divides bytes by `1024**3`, so the table reports GiB. Training time measures the update block and excludes evaluation, probes, checkpoints, and data transfer performed outside that block.

## Sources of the time savings

| Phase median per step | Full replay | Cached suffix | Reduction |
|---|---:|---:|---:|
| Direct | 6.870400 s | 5.817910 s | 15.3%; about 1.0525 s |
| Hubs | 0.793008 s | 0.588261 s | 25.8%; about 0.2047 s |

All other phase medians changed by about 0.15% or less. The measurements point to reduced direct and hub replay work as the source of the savings. Separate phase medians need not sum exactly to the median total step time. Peak training memory did not decrease.

## Why the job finished in half an hour

The initial estimate of a few hours was too high. It was made before measuring the workload on the allocated GPU and should have been presented with greater uncertainty.

This 1M-token workload requires only **61 steps per run**: each step processes `8 * 2,048 = 16,384` prediction tokens, and `61 * 16,384 = 999,424`. At approximately 9.3 seconds per full-replay step and 8.1 seconds per suffix step, the measured runtime follows directly from the workload.

| Work within the successful GPU job | Time |
|---|---:|
| Full-replay training updates | 9m 28.29s |
| Cached-suffix training updates | 8m 11.68s |
| Six preflight checks | 4m 39.58s |
| Timed validation evaluation across both runs | 2m 27.94s |
| Gradient probes across both runs | 2m 04.21s |
| Timed resumable-checkpoint writes across both runs | 5.57s |
| Remaining startup and work outside these timers | Approximately 2m 43.73s |
| **Total Slurm job** | **29m 41s** |

The training updates total **17m 39.97s**. All other work totals approximately **12m 01.03s**. The remaining-time row is a residual computed from Slurm elapsed time and the recorded timers; it includes work such as imports, model initialization, data verification, logging, best-checkpoint handling, and final test evaluation. The preflight row contains its six checks, excluding its process startup and initial model construction. Values are rounded, so displayed rows can differ from the total by a few hundredths of a second.

After preflight supplied real measurements, the estimate was revised to **25-40 minutes**, which matched the outcome. The 10-hour Slurm request was a maximum limit. Earlier setup attempts and their failures were separate jobs. Neither training run was shortened, and no steps were skipped.

## Interpretation and next checks

This run supports retaining cached suffix for this configuration: it delivers the expected performance benefit without an observed loss or memory penalty.

The evidence covers one seed, one GPU model, one run per execution mode, a fixed execution order, and a short 1M-token budget. Timing includes the project's measurement instrumentation. It does not establish a confidence interval or behavior on longer runs or other hardware. Useful next checks are repeated matched runs across several seeds, alternating execution order, and a larger token budget to assess accumulated numerical drift.

## Reproduction and saved evidence

With the same verified dataset and dependencies available, run the modes sequentially using fresh output directories. The following Bash sequence stops if preflight or either run fails:

```bash
set -euo pipefail

python new_implementation/gpu_preflight.py --device cuda:0 --tokens 1M \
  --population 256 --launch-bound --output runs/reproduce-preflight

python dust.py --device cuda --tokens 1M --population 256 --seed 42 \
  --full-replay --probe-interval 10 --checkpoint-interval 10 \
  --output runs/reproduce-full-replay

python dust.py --device cuda --tokens 1M --population 256 --seed 42 \
  --probe-interval 10 --checkpoint-interval 10 \
  --output runs/reproduce-cached-suffix

python new_implementation/compare_runs.py \
  runs/reproduce-full-replay runs/reproduce-cached-suffix
```

The explicit `cuda:0` preflight argument also avoids the unindexed-device startup error in the original source snapshot.

The raw evidence is committed in this directory, so the reported measurements can be checked without cluster access. The following files are byte-for-byte copies from `runs/baseline-gpu-20261112-224500/`:

| Evidence | Committed files |
|---|---|
| Full replay | [config.json](full-replay/config.json), [metrics.jsonl](full-replay/metrics.jsonl), [result.json](full-replay/result.json), [console log](full-replay.log) |
| Cached suffix | [config.json](cached-suffix/config.json), [metrics.jsonl](cached-suffix/metrics.jsonl), [result.json](cached-suffix/result.json), [console log](cached-suffix.log) |
| Six GPU preflight checks | [preflight.json](preflight/preflight.json), [console log](preflight.log) |
| Successful GPU job | [job-13891039.log](job-13891039.log), [status.txt](status.txt) |
| Environment and setup | [gpu-environment.json](gpu-environment.json), [nvidia-smi.txt](nvidia-smi.txt), [environment.txt](environment.txt), [setup-13890838.log](setup-13890838.log) with the 26 passing CPU tests |
| Source adjustment | [preflight-device-index.patch](preflight-device-index.patch) |
| Saved aggregates | [summary.json](summary.json), [comparison.txt](comparison.txt) |

[slurm-accounting.txt](slurm-accounting.txt) is a fresh capture of `sacct` for job 13891039, including its completed state, exit code, allocation, and 1,781-second elapsed time. [run_metadata.json](run_metadata.json) records the capture command and timestamp, the original and committed artifact directories, SHA-256 hashes for every evidence file above, source and dataset hashes, and the timing accounting. Its `artifact_sha256` paths are relative to this directory. The previously recorded hashes of both runs and the preflight report are unchanged.

Both raw metric files contain all 61 steps plus step zero, finite losses and host timings, matching configurations apart from execution mode, and the same 33 validation evaluations. Dataset and tokenizer hashes were checked against the experiment manifest during the job. Model checkpoints (`best.pt` and `state.pt`) and dataset tensors remain on the cluster; they are not needed to recompute the reported aggregates from these logs.

To verify the evidence hashes and regenerate the numerical comparisons from a checkout, run the following from the repository root. These checks use only Python's standard library and do not require a GPU or training dependencies. The summary command rewrites `summary.json`; the subsequent diff confirms it matches the committed version.

```bash
set -euo pipefail

python3 - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path('results/baseline-test')
metadata = json.loads((root / 'run_metadata.json').read_text())
for name, expected in metadata['artifact_sha256'].items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f'Checksum mismatch: {name}')
print(f"Verified {len(metadata['artifact_sha256'])} evidence files")
PY

python3 scripts/check_baseline_results.py results/baseline-test
git diff --exit-code HEAD -- results/baseline-test/summary.json

python3 new_implementation/compare_runs.py \
  results/baseline-test/full-replay results/baseline-test/cached-suffix \
  | diff - results/baseline-test/comparison.txt
```
