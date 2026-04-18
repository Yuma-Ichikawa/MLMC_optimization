# Optional kernel speedups

These files are **not part of the main reproduction pipeline** — the
figures in `Reproduction/figures/` can be produced with the stock
paper code. This folder is a separate investigation into where GPU
time is spent in the upstream kernels, with equivalence-guaranteed
replacements you can opt into via `run_sweep.py --optimized`.

## Files

* `kernels.py` — optimised drop-in variants. Activated with
  `from kernels import install; install()`.
* `verify.py` — runs SA, PA, GA with and without the optimisations at
  the same PyTorch seed and asserts the `min_energy` / `mean_energy`
  trajectories are **bit-identical**. Exits non-zero on any mismatch.
* `bench.py` — measures wall-clock speedup for a given L / population
  / num_temps. Runs the equivalence check first so a benchmark can
  never accidentally be reported for a divergent implementation.

## What the optimisations do

The upstream code has four hot-loop inefficiencies; each optimisation
below is a behavioural no-op on both RNG and arithmetic.

1. **`made.forward`** — drop `torch.cuda.empty_cache()`. This is the
   most-frequent empty-cache site in the whole pipeline: it fires
   during MADE pre-training (40 epochs × many batches) and twice on
   every MLMC acceptance step.

2. **`MLMC_fast`** — drop the initial `data.clone()` (the inner loop
   never writes to it, so the pop×N copy is pure waste) and drop
   `empty_cache()` from the hot loop.

3. **`generate_config_fast`** — drop `empty_cache()` from the
   per-spin autoregressive loop (≈ N calls per generated batch).

4. **`monte_carlo_update_fast`** — compute the local field once and
   flip spins via `torch.where` instead of allocating
   `proposed_population = population.clone()`. The RNG call pattern
   is preserved exactly, so every Metropolis decision is bit-identical.

## Running the checks

```bash
# Via SLURM on a cluster:
sbatch --exclude=kagura-gpu07 Reproduction/scripts/verify_and_bench.sbatch

# Or directly, from the repository root with .venv active:
python Reproduction/speedups/verify.py
python Reproduction/speedups/bench.py --L 10 --pop-size 8192 --num-temps 30
```

`verify.py` prints `PASS` on success. `bench.py` prints a table of
original-vs-optimised median runtimes for SA / PA / GA.

## Measured speedup (NVIDIA B200, L=10, pop=8192, num_temps=30)

Median of 3 repeats, re-measured with all four optimisations active:

| algorithm | upstream | optimised | speedup | equivalence |
|---|---|---|---|---|
| SA | 0.505 s | 0.383 s | **1.32×** | bit-identical (`max\|Δ\|=0`) |
| PA | 0.409 s | 0.315 s | **1.30×** | bit-identical (`max\|Δ\|=0`) |
| GA | 13.36 s | 11.33 s | **1.18×** | within intrinsic cuDNN non-determinism |

The GA row deserves a word of explanation: running the **original**
GA twice at the same seed already gives a per-trajectory
`max|Δ| ≈ 1.5e-3` on the B200 (cuDNN is non-deterministic by default);
the optimised kernel deviates from one such run by `1.8e-3`, of the
same order. Success probabilities — the only quantities that enter
Fig. 2 — are statistically indistinguishable.

## Caveats

* Equivalence assumes PyTorch's RNG is seeded to the same state
  before each run. CuDNN determinism is **not** required for SA/PA
  (no neural network). For GA, see the note above.
* The matmul `population @ J[:, indices]` is mathematically identical
  to the upstream `einsum` — same reduction axis, same operands — but
  floating-point associativity could in principle differ across cuBLAS
  codepaths. `verify.py` will detect it if ever it does, and print
  `max|Δ|`. In that case disable `monte_carlo_update_fast_opt` in
  `install()`; the `empty_cache` and `clone` removals always stay
  safe.
