# DUST project — refined TODO

Updated 2026-11-11. Companion to `PROJECT_PLAN.md` (the full plan) and `lessons/lectures.md` (how we
measure performance). Each item says what it is and why it matters, not how it's wired.

Legend: `[x]` done · `[~]` in progress · `[ ]` todo · **(GPU)** needs a GPU to validate.

**Where the code lives:** our contribution is in `new_implementation/` (gradient_check, accounting,
profiling, checkpoint, groups — see `new_implementation/README.md`). The original DUST repo stays at
the root (`dust.py`, `model.py`, `prepare_data.py`) so the README's training commands keep working.
Only `dust.py` was modified in place (the estimate/apply split); everything else of ours is additive.

---

## Foundation — done

- [x] **`new_implementation/groups.py` — one shared vocabulary for which parts of the network get perturbed together**
      (the output projections, the hidden MLP, the attention "hubs", embeddings, the head, and the
      attention internals). It maps every weight to its group, says which later layers a given
      perturbation can reach, and counts how many noise draws each group gets. Diagnostics, the cost
      model, and the allocator all read from this one file so they can never disagree.
- [x] **`dust.py` — split one training step into "estimate" then "apply".** Estimate does the
      forward passes and produces a gradient without touching the weights; apply normalizes it, runs
      a safety check, and takes the optimizer step. Same numbers as before, but now the estimate half
      can be measured, profiled, and reused on its own.
- [x] **`dust.py` — label each phase of the estimate** (clean pass, direct draws, embedding, hubs,
      head, local, scalars) so the profiler timeline reads in those terms instead of a wall of raw ops.

## Measurement stack — done (grade the gradient, count the cost, time the run)

- [x] **`new_implementation/gradient_check.py` — score DUST's gradient against the truth.** (Renamed
      from `diagnostics.py`: it checks one thing — gradient fidelity — so the name now says so.)
      Computes the exact backprop gradient to compare against, runs the DUST estimate without changing
      weights, and reports how well they agree (direction and magnitude) per group. Repeats over seeds
      to separate bias from noise, and reproduces the paper's "estimate improves with more draws" curve.
      It does **not** track training or held-out loss over time — that is the experiment campaign
      (below), a separate and later measurement.
- [x] **`new_implementation/accounting.py` — count the cost on paper, and measure it on hardware.**
      Two halves. (1) *Analytic:* from the model's dimensions and the per-group draw counts it counts
      the arithmetic **by formula** — no GPU, nothing runs — and reports it as "forward-pass
      equivalents" (how many clean forward passes' worth of work one update costs), aware that a
      perturbation deep in the network re-runs fewer layers than one near the input. This is what makes
      variants A/B/C/D comparable before we spend any GPU time, and it sizes memory. (2) *Measured:*
      times real steps correctly (CUDA events, warmup discarded, median/p10/p90) and flags when a step
      is limited by CPU launch overhead, not GPU math. (The roofline — FLOPs vs bytes moved — lives
      in `roofline.py`; see Performance extensions.)
- [x] **`new_implementation/profiling.py` — the torch.profiler harness.** Produces the per-op and
      per-phase breakdown and a timeline trace, plus a quick "double the batch and see if the time
      doubles" test that says whether we're GPU-bound or launch-bound before trusting any savings number.
- [x] **`new_implementation/checkpoint.py` — resume a run exactly where it stopped.** Saves everything
      needed to continue bit-for-bit (weights, optimizer momentum, the noise generator's state, the
      allocator's state, the step count), writes it safely, one file per worker.
- [x] **Audited the whole stack against the profiling lecture** and fixed what was off: the timing
      method, warmup, reporting percentiles, a stability guard in the curve fit, the precision the
      reference gradient is computed at, and a caveat that FLOP savings may not show up in wall-clock.

## Tests for the measurement stack (CPU) — done, except CI and the GPU checks

The modules above were written without tests. Each test checks against an independent truth, and a
deliberately broken version of the code must fail it. Run: `python -m pytest new_implementation/`.

- [x] **`test_checkpoint.py`** — a run saved after 2 steps, rebuilt from different seeds, loaded and
      continued is bit-for-bit the 4-step run that never stopped (weights, momentum, noise generator);
      resuming without the generator or the momentum must diverge. Also: bf16 embeddings and the
      per-group learning rates round-trip, no `.tmp` is left behind, per-rank files, world mismatch.
- [x] **`test_gradient_check.py`** — `compare` on known answers; the backprop reference against a
      finite difference along each parameter's gradient (must also reject a half-size gradient);
      seeded, weight-preserving estimates; `fit_cosine_law` recovers a known law; on a tiny model the
      estimate's cosine to backprop rises with population (head 0.07 → 0.18 at 8× draws), and its
      scale matches the centering factor (head projection 0.506 vs the predicted 1/2).
      Fixed two bugs it found in `gradient_check.py`: one seed gave a NaN variance (now raises), and a
      fit with no positive ceiling returned a complex `c_max` (now NaN with a note).
- [x] **`test_groups.py`** — every parameter of a real GPT (tiny, base, odd depth) maps to a group; the
      value-embedding blocks match `model.has_ve`; population K = writers + hidden + hubs + embedding for
      every shipped recipe; whole-model attention-internal counts equal the draws `local_errors` samples.
      Fixed a bug it found: `group_draw_count`/`total_draws` returned *one block's* q/k/v/gate/ve count as
      the whole-model total (they are per block); now `count × blocks` (needs `n_layer`), plus a
      `population()` helper. `accounting` was unaffected (it already looped over blocks).
- [x] **`test_accounting.py`** — `analytic_cost`'s forward FLOPs (clean + direct + embedding + scalars)
      equal what PyTorch executed in a real `DUST.estimate` under `FlopCounterMode`, attention included
      (the CPU SDPA is swapped for an identical explicit-matmul attention), in suffix and full-replay
      modes; a wrong-suffix prediction fails. Also: window sizes and parameter bytes vs the model, the
      suffix saving, world split, memory scaling, nearest-rank percentiles, `RunMeters`, `measure`.
      **Open convention question:** on sliding-window blocks accounting counts only in-window keys, but
      `model.attention` passes a mask to SDPA, so the dense scores are computed — accounting is ~2–2.5%
      below executed FLOPs at SSSL (pinned by a test). Decide whether "FLOPs" means needed or executed.
- [x] **`test_profiling.py`** — `profile_estimate` labels every `dust/<phase>` in the table and the trace;
      `profile_training` stops after its schedule; `launch_bound_test` gives the right verdict for steps
      of known cost (constant → launch-bound, proportional → GPU-bound).
- [x] `gradient_check._cosine` now computes in float64: fp32 over the embedding group gave cos(g, g) =
      1.00016 and failed `test_compare_on_known_answers` on this machine.
- [ ] **CI** — run `pytest new_implementation/` on CPU on every push (GitHub Actions). Team decision.
- Note: the estimate's expected size is (1 − 1/n) of the gradient (centering includes the draw
  itself; n = nrep for direct sites, the batch for the head), and smaller still for σ-smoothed and
  attention-internal groups (measured 0.25–0.44 on the tiny model). Read `rel_norm` with that in mind.
- [x] **Training-time metrics wired in — `run_metrics.py`, `dust.train()`, `backprop.train()`.** Both
      loops write one record per step: cumulative analytic FLOPs / forward-equivalents / draws,
      `train_seconds` and GPU-hours (eval, probes, checkpoint writes excluded), honest `elapsed_seconds`,
      step time, peak memory (CUDA), the allocation when it changes; optional gradient-cosine probe vs
      backprop every N steps (`--probe-interval`, `--probe-seeds` ≥ 2 adds variance/bias). Fixes the old
      `elapsed_seconds`, which counted evaluation time as training time.
- [x] **Checkpoint/resume wired into `dust.py`** (`--checkpoint-interval`, `--resume`). Refuses a
      different model/recipe/seed/world/data; rewinds `metrics.jsonl` to the checkpoint; restores best
      val + best weights. Fixed `checkpoint.load_state` for CUDA generators (`set_state` needs a CPU
      tensor) — **unverified on GPU**. Backprop baseline has logging but no resume.
- [x] **`toy.py` + `--toy` / `--device`** — CPU smoke runs of both trainers (DUST: val 8.32 → ~2.1 in
      40 steps, ~15 s). **`test_training.py`**: train() = bare `DUST.step` loop; both methods learn; logged
      cost = steps × analytic cost (draws checked against `groups`); eval excluded from train time; the
      probe leaves training bit-identical; a crash past the checkpoint + resume = the uninterrupted run;
      resume refuses a changed seed/recipe; the CLIs run and resume. Resume and probe tests verified to
      fail when the rewind or the RNG restore is removed.
- [ ] **(GPU)** check the CUDA paths the CPU tests cannot: CUDA-event timing / peak memory in the records,
      generator restore on resume, multi-GPU (per-rank `state.pt`, rank-0 metrics), probe memory at batch 8.
- Note: the logged FLOPs depend on the FLOP convention (see "FLOP convention" below).

## Validation — todo (deferred until we have a GPU)

- [ ] **(GPU)** Confirm the new two-half step gives identical numbers to the old fused step under a
      matched seed.
- [ ] **(GPU)** Confirm our "quality improves with more draws" curve matches the paper's within noise.
      This is the cheapest convincing proof the estimator is faithful — the gate for baseline A.
- [ ] **(GPU)** Run the launch-bound test on real hardware, to settle GPU-bound vs launch-bound
      **before** we trust any predicted FLOP saving.

## Track 1 — only re-run the part of the network a perturbation affects (baseline A) — done on CPU, GPU gate open

A perturbation injected at some layer can only change the layers after it. Re-running the whole network
to score it is wasted work. This track caches everything up to the injection point and re-runs just the
tail.

- [x] **`model.py`** — add a way to run the model starting from a chosen layer using cached inputs, and
      rebuild the normal forward pass on top of it. (`GPT.forward_from`; `forward` is now
      `forward_from(0, ...)`, and can record the stream entering each block.)
- [x] **`dust.py`** — during the clean pass, cache each block's input; then score the direct draws by
      re-running only the layers after the injection point instead of the full network. On by default;
      `--full-replay` (`DUST(..., suffix=False)`) keeps the public code's behaviour as the upstream
      reference, with the same noise stream. Saved in `config.json` as `execution`.
- [x] **Gate, CPU half:** `new_implementation/test_suffix.py` — on a tiny fp32 model, suffix and full
      replay give identical gradient estimates under a matched seed (max relative difference 0.0), and
      block-execution counts prove draws at block l skip blocks before l. A deliberate off-by-one start
      block fails the check. Predicted saving from `accounting.analytic_cost`: 15.5–17.9% of update
      FLOPs across the shipped recipes (16.4% at 1M/16k).
- [ ] **(GPU) Gate, GPU half:** rerun the equivalence under bf16 autocast (expect small, not zero,
      differences: the replayed prefix runs at batch n·B vs B), then measure runtime and peak memory vs
      `--full-replay` (wall-clock may not move if we turn out to be launch-bound — see lessons L3.1).
- Note: this is the **only** paper optimization still missing from the public code — the other two
  shortcuts are already in it. It's also the groundwork that multiplexing (Track 3) builds on.

## Track 2 — spend noise draws where they help most (B, our idea) — built and CPU-tested; GPU comparison open

The paper hand-tunes how many draws each group gets and freezes it. Our idea is to decide it live, giving
more draws to the groups whose estimate is least reliable.

- [x] **Math verified.** The literature confirms the rule: draws in proportion to each group's noise
      minimizes total gradient error, and a group's importance already enters through its noise, so we
      don't weight it separately. We can measure that noise for free by splitting each step's draws in
      half and comparing. One trap to avoid (don't compare already-averaged numbers). Written up in
      `lessons/literature-findings.md`.
- [x] **`new_implementation/allocation.py`** — both policies the proposal names, on a fixed budget,
      with a floor per unit (two chunks per GPU), counts rounded to `world × nrep` by largest
      remainder, measurements EMA-smoothed, recomputed every N steps, each recompute moving only part
      of the way, and each allocation used from the step after the one that measured it.
  - **`variance`** (Neyman): draws ∝ per-draw noise. Units: each block's `writers`, each block's
    `hidden`, and `embedding`.
  - **`agreement`** (gradient agreement / the norm test): draws ∝ noise² ÷ signal², which brings every
    unit to the same cosine between its two half-population estimates. Scale-free, so it also
    reallocates the `hubs`.
- [x] **Free measurement hook** — `DUST.direct_errors(split=True)` sums the draw chunks it already
      runs into two halves; `split_statistics()` turns them into per-draw variance, squared gradient
      norm and the half-to-half cosine. No extra forward pass, the gradient estimate is bit-identical,
      one host sync per step and none inside the draw loops (lessons L3). Cost: two extra
      error-times-input products per measured layer per step (not yet in `roofline.py`).
- [x] **Integrated** into `dust.train()` (`allocation=...`; CLI `--adaptive`,
      `--allocation-policy/-interval/-budget/-objective`). Allocator state is in the checkpoint; resume
      refuses changed options. `metrics.jsonl` logs each new `allocation` plus the `allocation_noise`
      and `allocation_agreement` behind it; `result.json` totals the decision's own time as
      `allocation_seconds` (the proposal's "cost of recomputing allocations").
- [x] **`test_allocation.py`** (18 tests, CPU, ~30 s). Hand-solved allocations for both policies and a
      brute-force search (allocating by variance instead of noise must lose); measuring leaves the
      estimate and a whole training run bit-identical; the budget holds at every reallocation; logged
      FLOPs equal the sum of each step's cost under the allocation in force; resume equals an
      uninterrupted run. Against the real estimator, over seeds:
  - split-half variance ÷ variance over seeds = 1.01–1.02 pooled (0.92–1.06 per parameter);
  - split-half signal ÷ squared mean gradient = 0.87 pooled over 40 seeds (0.67–1.23 per parameter);
  - the measured half-to-half cosine matches the literature's ρ/(ρ+1) per parameter (e.g. 0.051 vs
    0.051, 0.028 vs 0.029, 0.080 vs 0.093).
- **What the toy shows about the signal** (4×64 model, 4–8 draws per unit): half-to-half cosines are
  0.01–0.08, i.e. every unit is deep in the noise. The variance signal is still tight per step; the
  agreement signal needs its EMA. Expect the same at population 256 on the real model.
- Toy training, 20 steps, one seed — **plumbing check, not a result**: fixed 1.975; `variance` 1.879;
  `agreement` 1.870 best val, at the same forward-equivalents within 1%. Decision time 1.6 ms over the
  run. Under `variance` with learning-rate weighting the embedding's noise is ~300× any block's (its
  rate is 1000), so it takes every spare draw.
- Decisions made while writing it — **team to confirm**:
  - **`variance` does not reallocate the `hubs`.** A hub's estimate is the target the attention
    internals are scored against, not a weight gradient, so its noise is in different units.
    `agreement` is scale-free and does include them. Head, attention internals and scalars are never
    moved (cheap, outside K).
  - **`variance` weighs noise by learning rate by default** (`--allocation-objective update`; the
    bibliography's weighted objective, K ∝ √w·σ): the token embedding's rate is 1000 against ~0.25 for
    the blocks. `gradient` weighs raw gradient noise instead. Which is better is an experiment.
  - **Budget is a draw count by default**, as the proposal says ("the same total budget; adaptive
    methods change only its distribution"). `--allocation-budget flops` fixes the analytic FLOPs
    instead (draws ∝ noise / √cost).
  - **Recipes with a unit under the floor are adjusted at step 0** (e.g. population 256, where the
    last blocks get 2 draws = one chunk): lifted to the floor, paid for by the rest. So B's starting
    allocation is not exactly A's there.
  - Defaults (interval 4, EMA 0.7, rate 0.5) are untuned guesses; the 1M run is only 61 steps. No
    dead-band beyond the damping.
- [ ] **(GPU)** multi-GPU path: the split statistics are summed over ranks with `all_reduce` so every
      rank picks the same allocation — untested.
- [ ] **Compare** B (both policies) against the fixed allocation at equal budget, then at smaller
      budgets to find what B needs to match A. Vary the recompute interval; report the overhead
      (measure-only run, `--allocation-interval 0`, against a fixed run).

## Track 3 — perturb several groups at once and untangle them (C) — built and CPU-tested; first result is negative

Instead of testing one group per forward pass, perturb several at once using coded patterns, then
separate out each group's individual effect afterward.

- [x] **Math verified.** The decode is exact to first order, but a second-order cross-term (curvature
      between groups) survives and biases the result when groups interact. The fix is a specific
      structured code, small noise, and only multiplexing groups in the same or neighboring layers. The
      catch: the groups cheapest to multiplex together are exactly the ones most prone to this bias.
      Written up in `lessons/literature-findings.md`.
- [x] **Codebook + decoder — `new_implementation/multiplexing.py`.** Sylvester Hadamard codes without
      the all-ones column (rejects the baseline and each group's own curvature), restricted to columns
      with an odd number of set bits — the literature's "sum-free" set — so that no second-order cross
      term reaches any group. G groups need sequences of M passes with M/2 ≥ G (2 groups: 4 passes).
      Random-sign (Rademacher) codes of the same length for the comparison. Decoder = code column
      correlated with the losses, centered over the sequence (DUST's reward centering, with the code
      sequence as the centering group). `test_multiplexing.py` proves it on controlled losses: a linear
      loss is recovered exactly; on a quadratic loss the sum-free code is still exact, while columns
      1, 2, 3 of H4 leak each pair's cross-curvature onto the third group at full strength, as the
      math predicts.
- [x] **Estimator + training — `DUST.multiplexed_errors`, `dust.py --multiplex hadamard|rademacher`.**
      Each block's writers and hidden layer (two groups that restart at the same block) share
      sign-coded passes: the same draws per group, half the forward passes for those groups.
      `accounting.analytic_cost(multiplex=True)` charges the shared passes once and still counts both
      groups' draws (the proposal's rule); it equals the FLOPs PyTorch executes, exactly. Metrics gain
      `direct_forwards`, the suffix-forward count.
- [x] **Constraint** — `multiplexed_errors` refuses groups that are not in the start block or the one
      after it.
- [x] **Experiment, CPU half — `multiplex_experiment.py`** (toy 4×64 model at init, block 0, cosine to
      backprop, mean over 16 seeds; P = 32 draws per group):

      | method | forward passes | cosine, nrep 2 (writers / hidden) | cosine, nrep 8 |
      |---|---|---|---|
      | separate (DUST as is) | 64 | 0.375 / 0.413 | 0.481 / 0.511 |
      | separate, half the draws | 32 | 0.284 / 0.308 | 0.365 / 0.391 |
      | joint, independent noise | 32 | 0.280 / 0.299 | 0.359 / 0.379 |
      | Hadamard | 32 | 0.313 / 0.317 | 0.313 / 0.317 |
      | Rademacher | 32 | 0.218 / 0.255 | 0.218 / 0.255 |

  - Exactly orthogonal codes beat random signs, as the literature says.
  - **At equal forward passes Hadamard does not beat plain DUST**: slightly ahead at nrep 2, behind at
    nrep 8 — and the shipped recipes use nrep 8–16 from population 1024 up.
  - **Why (our derivation, consistent with the table; team to check):** a code sequence holds one
    noise direction per group, so M passes test G directions however long the sequence is — 2
    directions per 4 passes here. Plain DUST tests one fresh direction per pass and loses one per
    chunk to centering: (nrep − 1)/nrep directions per pass, i.e. 1/2 at nrep 2 (a tie) and 7/8 at
    nrep 8. Gradient quality follows directions tested, so coding cannot win where the estimate is
    limited by direction count. This is the bibliography's MeZO caveat ("multiplexing re-allocates
    passes; it does not shrink the per-estimate dimension cost") showing up in numbers.
  - **Where C can still win:** where the reward *readout* is the limit rather than the number of
    directions — bf16 rounding on small loss changes (the paper raises σ to 0.4 in the deeper half for
    this reason). A sequence reads each direction M times and cancels curvature exactly, so it
    tolerates a larger σ. CPU runs are fp32 and cannot show this.
  - Toy training, 20 steps, one seed, untuned — plumbing check: C reaches 2.28 (Hadamard) / 2.14
    (Rademacher) best val against 1.98 for A, at 27% fewer forward-equivalents.
- [ ] **(GPU) Experiment, GPU half** — rerun `multiplex_experiment.py` on the real model under bf16,
      per block, with a σ sweep (the bias is second order in σ, the readout noise falls with σ), and
      time it. This decides whether C is worth a training campaign.
- [ ] **Higher code rate** — untested variants that test more directions per pass: 3 groups on all of
      H4's columns (rate 3/4, cross-curvature aliasing back); adding the block's hub as a third group;
      groups from the adjacent block.
- Not done: a learning-rate retune for C (Hadamard estimates are not shrunk by the (1 − 1/nrep)
  centering factor, so the same lr is a bigger step); `roofline.py` does not model the shared passes.

## Track D — combine B and C — todo

- [ ] Make adaptive allocation and multiplexing agree on sizes (draw counts vs code lengths), integrate
      them, and attribute the speedup to each piece against baselines A, B, and C.
      `dust.train` refuses `allocation` together with `multiplex` until this exists. What it needs:
      multiplexed writers + hidden draw from one count per block (one allocation unit, in multiples of
      the code length 4), and the split-half measurement taken over whole code sequences.

## Model sizes — todo

- [ ] Add the two smaller model configs the size study needs; check their parameter counts and where the
      value embeddings sit; train them with the paper's small-model recipe, not the main-run one.

## Measurement & performance refinements — todo (from the lectures; detail in `lessons/lectures.md`)

All lectures are now read in full. The measurement stack above is built and matches the course method;
these are the refinements the lectures call for, grouped by file.

- [x] **roofline + regimes — `new_implementation/roofline.py`.** Per-op FLOPs, dtype-aware bytes,
      kernel launches and host syncs for every phase of one update (the same phase names as the
      profiler labels), placed on an **A100** (or H100) roofline: compute / bandwidth / launch floors,
      the regime per phase, best-case MFU, plus `mfu()` and `amdahl()` for measured numbers. Models
      both SDPA kernels the explicit mask allows (`efficient`, `math`) and suffix vs full replay.
      `python new_implementation/roofline.py --population 256`. Checked on CPU
      (`test_roofline.py`): its matmul FLOPs match a real `DUST.estimate` under `FlopCounterMode`
      exactly, in both suffix and full-replay modes.
      **Prediction worth testing:** at K=256 the update is *bandwidth*-bound, not launch-bound —
      3.4 s GPU floor vs a 0.34 s launch floor, 67% of GPU time in unfused elementwise ops (rotary,
      relu², the fp32 logit cap every draw, norms, residual ops); weights are <1% of bytes, so they
      are not "re-streamed like decode". With the `math` attention kernel it is 8.2 s, dominated by
      the materialised [T, T] scores.
- [ ] **(GPU) roofline calibration.** Check which SDPA kernel runs; calibrate the per-launch latency
      from a trace; compare predicted vs measured per-phase time; split measured step time into kernel
      compute / HBM / launch / other and feed the measured fractions to `amdahl()`.
- [ ] **profiling — actually see the launch overhead.** Read the trace *gaps*, not just the per-op
      table (GPU-idle gap per step, mean-kernel-vs-gap "thin comb" test); add an nsys timeline with NVTX
      region labels; add a graph-break / compiled-region counter for the compiled path; run Nsight
      Compute on a kernel only once it's confirmed GPU-bound.
- [ ] **gradient_check — fair reference + a cost axis.** Reference under normal-grad FP32, DUST estimate
      under `inference_mode` (not the same memory profile — note it); fold a cost axis into the
      quality-vs-population curve; judge variants by steps/s **and** time-to-target, not step time alone.
- [ ] **the launch-bound fix (biggest lever), GPU-validate.** Vectorize the population into one batched
      forward; keep the inner loop sync-free and shapes static; then CUDA Graphs /
      `torch.compile(mode="reduce-overhead")` on the fixed-shape forward. First audit the hooks for
      `.item()` / branch-on-tensor / arbitrary Python — those break fusion **and** CUDA-Graph capture at
      once. This supersedes the old "scope decision": the course is built on exactly this, so treat it as
      first-class, not out of scope.
- [ ] **Masked-aware attention (skip masked positions), GPU-validate.** `model.attention` passes an
      explicit boolean mask to SDPA, which rules out the flash kernel: every block computes all T×T
      scores, including the causal future and the keys outside the sliding window. On the 8×512 model at
      2048 tokens that is ~32% more attention-inclusive update FLOPs than needed (1M/K=256 update:
      executed 3.54e14 vs needed 2.68e14). Likely one of the README's "omitted execution
      optimizations": `model.py`'s `(left, 0)` window tuples are flash-attn's `window_size` format, so the
      authors probably ran flash attention with a sliding window (inference, not stated in the paper).
      Plan: full-window blocks → SDPA `is_causal=True`, no mask (flash kernel, exact); sliding blocks →
      FlexAttention causal+window block mask (or the `flash-attn` FA2 package, `window_size=(w, 0)`; FA3
      is H100-only). `local_errors` calls the same function, so it benefits too. Behind a flag
      (`--attention masked|flash`) so the upstream reference stays reproducible; A/B/C/D all use the
      fast path. Gate: CPU test that outputs match the masked path within tolerance; GPU check within
      bf16 tolerance + timing. Report it as an engineering speedup, separately from B/C.
- [ ] **FLOP convention: log "needed" and "executed".** `accounting.attention_flops` is currently a
      hybrid — full-window blocks charged all T keys (incl. the causal-masked half), sliding blocks
      charged window+1 keys (ignoring causality) — 85% of executed on the base model, pinned by
      `test_accounting.py::test_sliding_windows_are_counted_as_windowed`. Replace with two counts:
      *needed* (causal + window; primary compute axis) and *executed* (what the kernel runs: dense
      today, ≈ needed + tile overhead once the masked-aware attention lands). Planned with the
      training-time metrics wiring.

## Experiment campaign — todo

- [ ] **Runner** (plan P4): sweep method × budget × seed with equal-budget learning-rate tuning and
      machine-readable output.
- [ ] **Matrix:** freeze the comparison set; run all four variants at the small scale, then the best at
      the larger scale; 3 seeds on the key comparisons; a backprop reference per model size.
- [ ] **Plots:** gradient quality vs compute, held-out loss vs wall/GPU-hours, how the allocation shifts
      over training, how far we can multiplex, and the best model at a fixed budget.

## Literature — done (3 subagents)

Full write-up in `lessons/literature-findings.md`; sources per folder in each `bibliography.md`.

- [x] **General (18 PDFs)** — the DUST estimator is mathematically sound: it's a known
      Gaussian-smoothing gradient estimate, unbiased with a small bias that shrinks as the noise shrinks.
      Its error falls as you add draws, scaled by the **width** of the layer, not the parameter count —
      which is why it beats weight-space methods. One correction to the paper's own wording: the
      signal-to-noise grows with the square root of (draws ÷ width), so you need at least about as many
      draws as a layer is wide before interference cancels. The centering baseline is essential.
- [x] **Adaptive allocation (11 PDFs)** — confirms our idea: draws proportional to each group's noise
      minimize total gradient error, and importance already rides inside that noise rather than needing a
      separate weight.
- [x] **Multiplexing (11 PDFs)** — confirms the decode is exact to first order; the surviving
      second-order bias is real, which is exactly why the structured codes, small noise, and
      same/adjacent-layer-only rule are needed.
- [x] Verdicts folded into Track 2 and Track 3 above.
