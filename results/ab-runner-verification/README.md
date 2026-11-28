The A/B experiment runner was verified on CPU and on one NVIDIA L40S. These are functional checks, not an adaptive-allocation performance result. The full 61-step, three-seed GPU campaign remains unrun.

The tested implementation is `a7733056b5004bdd481ee48e5e3ab43b7b3c2967`. It uses the existing trainer for fixed allocation, both adaptive policies, and each policy's matched fixed/measurement-only controls. The manifest records requested and effective recipes; summaries validate artifacts and compute comparisons within each seed before aggregation.

CPU verification used PyTorch 2.10.0. An independent public-CLI campaign ran 21 cases across seeds 42, 43, and 44 for six updates each. Matched controls had identical weights, optimizer state, and RNG state. A process terminated after an adaptive checkpoint resumed to the same final state as its uninterrupted reference. Completed reruns preserved all run artifacts.

That campaign ran at `9e0058c` and exposed an unsealed-result recovery bug. The failing regression reported `Skipping verified` and created a receipt for an unreadable checkpoint. The correction requires an existing valid receipt to skip; otherwise it validates and resumes the checkpoint before publishing completion. At `a7733056`, an independent fresh-artifact run passed all 20 runner/summary tests, including rejection of missing or corrupted unsealed checkpoints. A separate 52-test runner, checkpoint, training, and allocation regression run passed on the corrected code.

The GPU checks exercised the runner APIs with shortened verification manifests; they did not execute the full GPU campaign. The default-CUDA GPU smoke, Slurm job `14150545` on `ins057`, completed all seven five-update cases using the real 8-layer, 512-wide model, population 256, batch 8, context 2,048, bf16 autocast, and PyTorch 2.10.0+cu128. Both overhead pairs had identical weights, per-step losses, and RNG states. Its additional strict optimizer-equality assertion failed, so the job exited `1:0` before the resume check.

The differences were confined to momentum buffers for `value_embeds.{1,3,5,7}.weight`: 39 differing entries for variance and 45 for agreement, out of 8,388,608 entries in each pair. The largest absolute difference was 1.1920928955078125e-7. This is consistent with the existing CUDA embedding-reduction nondeterminism documented in `new_implementation/gpu_preflight.py`; bit-exact optimizer equality under the default CUDA settings is not claimed. See [control-state-differences.json](control-state-differences.json).

A separate adaptive-variance resume check passed with verification-only deterministic CUDA settings: `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Slurm job `14150786` on `ins057` exited `0:0`. It terminated a real child process after checkpoint 2 and resumed through update 5. Model, optimizer, perturbation RNG, CPU RNG, CUDA RNG, and allocator state matched the uninterrupted reference exactly. These settings were confined to the verification helper; the production runner was unchanged. See [deterministic-resume.json](deterministic-resume.json).

The [evidence audit](evidence-audit.json) checked source and data identity, copied artifact checksums, complete step sequences, and recorded allocation changes. Full model checkpoints and helper scripts remain under `/insomnia001/home/pts2125/hpml/dust/runs/ab-runner-verification-20261128-004249-a773305` and its `-deterministic-resume` sibling on Insomnia.

The Slurm scripts and shell-safe dry-run commands were checked locally. On the Insomnia login node, the documented dry run requires the existing CUDA Python environment because the system Python lacks PyTorch.

```bash
OMP_NUM_THREADS=1 python3 scripts/ab_experiment.py run --toy --steps 5 --seeds 42 --output runs/ab-toy
python3 scripts/ab_experiment.py summary --output runs/ab-toy
.venv/baseline-cu128/bin/python scripts/launch_ab.py --dry-run --seeds 42 43 44 --output runs/ab-gpu
```
