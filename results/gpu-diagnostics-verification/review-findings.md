# Diagnostics review findings

The first implementation at `88c5bf0` passed its focused tests. Independent public-command checks
found the following defects before publication. These checks did not change production code.

| ID | Reproduction and cause | Correction | Status |
|---|---|---|---|
| D1 | CPU quality seeds `0` and `4294967296` produced identical metrics and zero sample SD. PyTorch's CPU generator aliases these seed values. | Restrict diagnostic perturbation seeds to distinct unsigned 32-bit integers. Keep the shared A/B validator unchanged. | Fixed at `4ef4ac0`; independent CLI checks passed. |
| D2 | Two concurrent identical launches submitted twice and overwrote the first job ID. The reused A/B snapshot helper intentionally accepts an identical existing launch. | Add an explicit exclusive mode to the snapshot helper, preserving its A/B default. Diagnostics require a fresh snapshot. | Fixed at `4ef4ac0`; both independent race checks passed. |
| D3 | A checkpoint with NaN in `lm_head.weight` produced NaN clean loss, but profiling exited zero and reported complete. Profile estimates were discarded without validation. | Check the separate phase estimate's loss and gradients outside measured intervals; reject non-finite diagnostic work. | Fixed at `4ef4ac0`; poisoned-checkpoint CLI check passed. |
| D4 | L40S profile phase CPU totals were zero although CPU trace spans were positive. CPU and CUDA annotations share names; the latter overwrote the former in a dictionary. | Select CPU profiler events for phase summaries. Keep CPU and device timing meanings explicit and match trace evidence. | Fixed at `4ef4ac0`; CPU trace and captured GPU-event replay passed. Corrected full GPU CLI run pending. |

Reproduction evidence is retained under `runs/gpu-diagnostics-development/cpu-verification/`
and `runs/gpu-diagnostics-development/gpu-verification/`. The initial L40S jobs were
`14151783`, `14151784`, and `14151785`. Their zero exits establish execution, not the absence
of these reporting defects. The final verification report will identify the corrected source.

The initial operator-count suspicion was not confirmed. Probe `14151798` showed that PyTorch
removes nested duplicate operator nodes before `key_averages()`. Raw trace counts therefore
need not equal the operator summary. This is separate from the confirmed phase-name collision.

The trace also exposed an existing analytic-accounting limitation. `DUST.local_errors` recomputes
an attention baseline for each kind and layer, adding 32 attention calls in this configuration.
The shared cost model omits those baseline terms. A separate TODO records the accounting audit;
this diagnostics goal preserves the existing formulas and labels their FLOPs and MFU as estimates.
