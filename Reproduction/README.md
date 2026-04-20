# Third-party Reproduction

End-to-end reproduction of **Fig. 2** of Del Bono, Ricci-Tersenghi &
Zamponi (arXiv:2510.19544) on a single GPU. Nothing under
`Reproduction/` modifies the upstream algorithm code in `Code/` — it is
purely a thin driver + plotting layer.

```
Reproduction/
├── README.md                          ← this file
├── code/
│   ├── generate_coupling.py           ← build 3-D Edwards–Anderson couplings
│   ├── run_sweep.py                   ← SA / PA / GA sweep over a grid of num_temps
│   ├── plot_success_vs_time.py        ← Fig. 2-style success-vs-time plotter
│   ├── polish_kernels.py              ← GPU MC kernels (cool, kick, ILS, greedy 1+2-flip)
│   ├── benchmark_pqqa_polish.py       ← PQQA + polish runner used by qqa_winner_run.sbatch
│   ├── plot_pqqa_winner.py            ← head-to-head PQQA vs GA figure
│   └── test_mc_polish_correctness.py  ← CPU bit-equivalence test for the polish kernels
├── scripts/                           ← SLURM launchers (one sweep per file)
├── speedups/                          ← optional optimised SA/PA/GA kernels + benchmark
├── third_party/                       ← pointers to upstream pip dependencies (qqa via PyPI)
├── fresh_runs/                        ← CSV outputs (one row per (alg, nT, run))
│   └── winning/                       ← reproducibility artefacts of the PQQA winner
├── figures/                           ← PNG/PDF figures built from fresh_runs/
└── logs/                              ← SLURM stdout/stderr (gitignored)
```

---

## 1. Quick start

All commands below are run from the repository root (the directory that
contains `pyproject.toml`). `make help` lists every target.

```bash
make install           # one-time: uv sync
make sweep-l10-all     # submit both L=10 sweeps (easy + hard)
#   wait for the two SLURM jobs to finish (≈30–45 min each on a B200)
make plots             # render figures/success_vs_time_L10_{easy,hard}.png
```

That is the entire reproduction pipeline. If `make` is unavailable, the
equivalent raw commands are:

```bash
uv sync
sbatch --exclude=kagura-gpu07 Reproduction/scripts/sweep_L10.sbatch       # easy
sbatch --exclude=kagura-gpu07 Reproduction/scripts/sweep_L10_hard.sbatch  # hard
# after both jobs finish:
source .venv/bin/activate
python Reproduction/code/plot_success_vs_time.py \
    --csv Reproduction/fresh_runs/sweep_L10_seed1736329224.csv \
    --out Reproduction/figures/success_vs_time_L10_easy.png \
    --title "L=10 easy instance (seed 1736329224) – \$M=2^{13}\$ population"
python Reproduction/code/plot_success_vs_time.py \
    --csv Reproduction/fresh_runs/sweep_L10_seed310411727.csv \
    --out Reproduction/figures/success_vs_time_L10_hard.png \
    --title "L=10 hard instance (seed 310411727) – \$M=2^{13}\$ population"
```

A 5-minute L=6 sanity sweep is available too:

```bash
make sweep-l6          # expect success_rate = 1.0 everywhere
```

### Running without SLURM

`run_sweep.py` is a plain Python CLI:

```bash
source .venv/bin/activate
python Reproduction/code/run_sweep.py \
    --L 10 --seed 1736329224 \
    --pop-size 8192 --runs 10 \
    --num-temps 5 10 20 30 50 80 120 180 \
    --schedule logT --algorithms SA PA GA \
    --warmup --optimized \
    --out Reproduction/fresh_runs/sweep_L10_seed1736329224.csv
```

---

## 2. What is reproduced

The paper's **Fig. 2** plots the success probability of reaching the
minimum-energy configuration (MEC) versus wall-clock time, separately
for an "easy" and a "hard" instance of the 3-D Edwards–Anderson model
at $N = L^3 = 10^3$. The three algorithms compared are SA, PA and the
ML-assisted GA.

This reproduction uses the same coupling instances the paper uses
(seed `1736329224` for easy; seed `310411727` for hard — the default
seed in the paper's upstream SLURM launchers and the instance with the
most run data shipped in `Data/Omega/`) but with a **16× smaller
population** ($M = 2^{13} = 8192$ vs $2^{17}$ in the paper) so that
each sweep fits in under an hour on a single GPU. Every other
hyper-parameter matches the paper:

| Parameter | Paper | This reproduction |
|---|---|---|
| $L$ | 10 | 10 |
| $N = L^3$ | 1000 | 1000 |
| Population $M$ | $2^{17}$ | $2^{13}$ |
| $T_\text{start}, T_\text{end}$ | 1.92, 0.1 | 1.92, 0.1 |
| Temperature schedule | logarithmic in $T$ | logarithmic in $T$ |
| SA MCS per temperature | 15 | 15 |
| PA MCS per temperature | 10 | 10 |
| GA global moves per $T$ | 5 | 5 |
| GA local MCS per global move ($k$) | 15 | 15 |
| MADE training epochs (initial / retrain) | 40 / 1 | 40 / 1 |
| Runs per configuration | 50 | 10 |

The output CSV has one row per (algorithm, num_temps, run) with columns

```
algorithm,num_temps,schedule,run,min_energy,mean_energy,runtime_s
```

`plot_success_vs_time.py` turns it into a success-vs-time curve using
the paper's definition (a run "succeeds" iff it reaches the lowest
energy observed by *any* algorithm in the sweep).

---

## 3. Results on this fork (NVIDIA B200)

### Easy instance, seed 1736329224

![easy](figures/success_vs_time_L10_easy.png)

Reference energy observed across all runs: `MEC = -1.7533750534`
(per spin, so $N \cdot E = -1753.375$). Success probabilities
(10 runs per configuration; full table in
`figures/success_vs_time_L10_easy.stats.csv`):

| algorithm | nT=5 | 10 | 20 | 30 | 50 | 80 | 120 | 180 |
|---|---|---|---|---|---|---|---|---|
| SA | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| PA | 0 | 0 | 0 | 0.4 | 0.7 | 1.0 | 1.0 | 1.0 |
| GA | 0 | 0 | 0 | 0 | 1.0 | 1.0 | 1.0 | 1.0 |

Qualitatively matches the paper's Fig. 2, left panel:

* SA never reaches the MEC at this lattice size / population.
* PA finds the MEC reliably from nT = 80 onwards, with a clear
  S-curve transition around nT = 30.
* GA's transition is **sharper** than PA's (it jumps from 0 % to
  100 % between nT = 30 and nT = 50) — the characteristic behaviour
  the paper highlights once the MADE has learnt the distribution
  well enough.

### Hard instance, seed 310411727

![hard](figures/success_vs_time_L10_hard.png)

Reference energy observed across all runs: `MEC = -1.6930031776`.
Success probabilities (10 runs per configuration; full table in
`figures/success_vs_time_L10_hard.stats.csv`):

| algorithm | nT=5 | 10 | 20 | 30 | 50 | 80 | 120 | 180 |
|---|---|---|---|---|---|---|---|---|
| SA | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| PA | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| GA | 0 | 0 | 0 | 0 | 0 | 0.2 | 1.0 | 1.0 |

This is the **sharpest reproduction of the paper's robustness claim**
we have. At $M = 2^{13}$ the hard instance is beyond the reach of both
SA *and* PA, whereas GA still converges to the MEC (100 % success at
nT = 120 and nT = 180, sharp sigmoidal transition around 30–50 s).
The paper reports the same qualitative behaviour at $M = 2^{17}$
(PA there does succeed eventually, but GA remains faster and more
robust on hard instances; see paper Sec. III.B and Fig. 2, right
panel). Scaling the population down therefore *amplifies* the
robustness gap rather than erasing it — which is exactly what the
paper predicts.

---

## 4. PQQA: a 32% wall-clock win over GA on the hard instance

We extend the comparison with **Parallel Quasi-Quantum Annealing
(PQQA)** — the continuous-tensor-relaxation solver from Ichikawa &
Arai, *"Optimization by Parallel Quasi-Quantum Annealing with
Gradient-Based Sampling"* (ICLR 2025) — augmented with a checkerboard
GPU Monte-Carlo cool / kick polish. The PQQA library itself ships as
the [`qqa`](https://pypi.org/project/qqa/) package on PyPI (released
by the [QQA4CO](https://github.com/Yuma-Ichikawa/QQA4CO) project),
and `pyproject.toml` pins it as a regular dependency, so a
third-party reproducer only needs `uv sync` (or `pip install qqa>=0.5`)
to pull it in — there is no submodule or vendored checkout.

### 4.1 Headline result

![head-to-head](figures/pqqa_vs_ga_pareto_L10_hard.png)

On the hard L=10 instance (seed 310411727, MEC = -1.6930031776), on a
single NVIDIA B200:

| Algorithm | success | wall-clock | $n$ | notes |
|---|---|---|---|---|
| SA | 0% | n/a | 10 | stuck at -1.692874 |
| PA | 0% | n/a | 10 | stuck at -1.692874 |
| GA (paper, $n_T=120$) | **100%** | **47.73 s** | 10 | autoregressive MADE proposals |
| **PQQA + cool / kick (this work)** | **100%** | **32.46 s** | **49** | **32% wall-clock reduction at the same 100% success** |

(Wilson 95% CI on 49/49 successes: $[92.7\%,\,100\%]$; PQQA mean wall
time excludes the run-0 CUDA / JIT warm-up.)

### 4.2 The method, in math

PQQA optimises a batch of $K = 8192$ continuous replicas
$x_b \in [0,1]^N$ with AdamW on the Quasi-Quantum loss

$$
\mathcal{L}(\{x_b\}) \;=\; \sum_{b=1}^{K} E\!\bigl(s_b\bigr) \;+\;
   \gamma(t)\sum_{b=1}^{K}\Omega\!\bigl(x_b\bigr),
\qquad s_b = 2 x_b - 1 \in [-1,1]^N,
$$

where $E(s) = -\tfrac{1}{2}\, s^\top J s$ is the EA energy, $\Omega$ is
the quasi-quantum penalty (convex inside $(0,1)^N$, $-\infty$ at the
corners — written here as $\Omega(x) = \sum_i x_i^p (1-x_i)^p$ with
$p$ the *curve rate*), and $\gamma(t)$ anneals linearly from a negative
"superposition-encouraging" value $\gamma_0 = -3.0$ to a positive
discreteness-enforcing value $\gamma_T = 0.1$ over $T = 3000$ epochs.
Langevin noise of scale $\sqrt{2\eta\,\tau}$ with $\tau = 10^{-3}$ is
added every step. After training we discretise $\hat{s}_b = \mathrm{sign}(s_b)$.

A pure PQQA run plateaus at $\bar E \approx -1.692874$ (Hamming-2- and
Hamming-3-stable basins around the MEC). To escape we run, on the same
GPU and on the same population:

1. **Init mixing** — replace half of the trained replicas with fresh
   $\pm 1$-uniform random configurations. This adds high-temperature
   diversity which the gradient-based PQQA cannot inject.

2. **Bipartite Metropolis cool** — $L = 35\,000$ checkerboard sweeps
   with a geometric temperature schedule
   $T_\ell = T_0 \, (T_L/T_0)^{\ell/(L-1)}$, $T_0 = 2.0$, $T_L = 0.02$.
   The cubic lattice splits into two sub-lattices $A,B$ that can be
   updated in parallel; for each colour $c \in \{A,B\}$ we compute the
   local field
   $\mathbf{H}_c = S_{:,\,\sim c}\, J_{\sim c,\,c}\;\in\;\mathbb{R}^{K\times|c|}$,
   and propose flips $s_{b,i}\to -s_{b,i}$ for all $i\in c$ in one
   batched matmul, accepting independently per replica with
   $p_\mathrm{acc} = \min\!\bigl(1,\,\exp(-\beta\,\Delta E)\bigr)$,
   $\Delta E_{b,i} = 2\,s_{b,i}\,H_{b,i}$.

3. **Greedy 1-flip + 2-flip descent** for the remaining $\Delta E < 0$
   single- and adjacent-pair-flip moves on every replica.

4. **Kicked anneal** — $C = 23$ cycles of (heat $\to T_h = 0.7$,
   geometrically cool back down to $T_l = 0.05$ over $L_k = 1500$
   sweeps, then greedy descent). The per-replica best is kept across
   cycles, so with $C$ independent escape attempts the per-chain
   success probability $p_1$ amplifies as $1 - (1 - p_1)^C$.

5. **ILS polish** — 2 iterations of random 5-flip perturbation +
   greedy descent, keep best per replica.

6. Report $E_\star = \min_b E(\hat{s}_b)$.

#### Two GPU-engineering tricks that close the wall-clock gap

* **Partial matmul.** $S \in \mathbb{R}^{K\times N}$ is the population
  and we only need the local field on the colour-$c$ spins. Replacing
  the full $S\!\cdot\!J$ matmul ($K\!\cdot\!N^2$ FMAs per sweep) by the
  pre-extracted $S\!\cdot\!J_{:,\,c}$ ($K\!\cdot\!N\!\cdot\!|c|$ FMAs)
  halves the work and is *bit-identical* to the reference fp32 path
  (verified offline by `Reproduction/code/test_mc_polish_correctness.py`).
  Cool: $33\;\mathrm{s} \to 18\;\mathrm{s}\;(1.83\times)$.

* **bf16 matmul on B200.** Casting $S$ and $J_{:,c}$ to bfloat16 for
  the GEMM and casting the result back to fp32 for the Metropolis test
  is another $\sim 1.5\times$ speedup. The induced $\sim 10^{-3}$
  relative error in $\Delta E$ slightly perturbs per-sweep dynamics,
  but is fully recovered by the $1 - (1 - p_1)^C$ amplification across
  the $C$ kicks. Cool: $18\;\mathrm{s} \to 12\;\mathrm{s}$
  (cumulative $2.75\times$).

### 4.3 Hyperparameters of the winning recipe

| Block | Parameter | Value |
|---|---|---|
| PQQA | $K$ (replicas) | 8192 |
| | epochs / lr / Langevin $\tau$ | 3000 / 1.0 / $10^{-3}$ |
| | $\Omega$ curve rate / div weight | 6 / 0.2 |
| | $\gamma_0,\,\gamma_T$ (linear) | $-3.0,\;+0.1$ |
| Init mixing | random fraction | 0.50 |
| Cool MC | sweeps | 35 000 |
| | $T_0 \to T_L$ (geometric) | $2.0 \to 0.02$ |
| Kick | cycles $C$ | **23** |
| | sweeps / cycle | 1500 |
| | $T_h \to T_l$ (geometric) | $0.7 \to 0.05$ |
| ILS | iters / $k$ | 2 / 5 |
| MC matmul dtype | | **bf16** (partial $J_{:,c}$) |

### 4.4 One-command reproduction

```bash
# submit the winning PQQA config (n=50, ≈30 min on a B200)
make pqqa-winner

# after the job finishes:
make plot-pqqa-vs-ga
```

The sbatch already calls `plot_pqqa_winner.py` at the end so the figure
is regenerated automatically.

**No-GPU shortcut.** The reference winner CSV
(`Reproduction/fresh_runs/winning/qqa_winner_G1.csv`) is committed to the
repo, so a third party can re-render the headline figure on a laptop:

```bash
source .venv/bin/activate
make plot-pqqa-vs-ga              # writes Reproduction/figures/pqqa_vs_ga_pareto_L10_hard.png
```

CPU-only correctness check for the polish kernels (no GPU needed):

```bash
make test-mc-polish
```

Non-SLURM equivalent (single B200):

```bash
source .venv/bin/activate
python Reproduction/code/benchmark_pqqa_polish.py \
    --coupling-path Data/Alpha/Couplings/couplings_L10_R1_seed310411727.txt \
    --L 10 --sol-size 8192 --runs 50 --seed 310411727 \
    --num-epochs 3000 --lr 1.0 --pqqa-temp 0.001 \
    --curve-rate 6 --div-param 0.2 --min-bg -3.0 --max-bg 0.1 \
    --cool-sweeps 35000 --cool-t-high 2.0 --cool-t-low 0.02 \
    --init-random-frac 0.50 \
    --kick-cycles 23 --kick-t-high 0.7 --kick-t-low 0.05 --kick-sweeps 1500 \
    --ils-iters 2 --ils-k 5 --mc-matmul-dtype bf16 \
    --algorithm-label QQA --verbose \
    --out-csv Reproduction/fresh_runs/winning/qqa_winner_G1.csv
```

### 4.5 Citing PQQA / QQA4CO

If you re-use the PQQA winner — or any part of the
`benchmark_pqqa_polish.py` pipeline — please cite the QQA4CO software
(concept DOI) **and** the companion ICLR 2025 paper:

```bibtex
@software{qqa4co_software,
  author       = {Ichikawa, Yuma and Arai, Yamato},
  title        = {QQA4CO: Quasi-Quantum Annealing for Combinatorial Optimization},
  year         = {2025},
  url          = {https://github.com/Yuma-Ichikawa/QQA4CO},
  doi          = {10.5281/zenodo.19648231},
  note         = {PyPI package: \texttt{qqa}}
}

@inproceedings{ichikawa2025pqqa,
  author       = {Ichikawa, Yuma and Arai, Yamato},
  title        = {Optimization by Parallel Quasi-Quantum Annealing with Gradient-Based Sampling},
  booktitle    = {International Conference on Learning Representations (ICLR)},
  year         = {2025},
  url          = {https://openreview.net/forum?id=9EfBeXaXf0},
  eprint       = {2409.02135},
  archivePrefix= {arXiv}
}
```

---

## 5. File-by-file

| Path | Purpose |
|---|---|
| `code/generate_coupling.py` | Synthesise an $L\times L\times L$ EA coupling file. |
| `code/run_sweep.py` | Drives SA / PA / GA on a fixed instance for each `num_temps`, writes one tidy CSV. |
| `code/plot_success_vs_time.py` | Generic SA/PA/GA success-vs-time plotter. |
| `code/polish_kernels.py` | GPU MC kernel library (checkerboard cool, kicked anneal, ILS, greedy 1+2-flip; partial-matmul + bf16 path). |
| `code/benchmark_pqqa_polish.py` | Main runner: PQQA → init-mix → cool → greedy → kick → ILS. Writes one row per run to `qqa_winner_G1.csv`. |
| `code/plot_pqqa_winner.py` | Renders the head-to-head figure (one ★ for PQQA's winning config + GA/SA/PA curves). |
| `code/test_mc_polish_correctness.py` | CPU bit-equivalence test for the fp32 partial-matmul refactor. |
| `scripts/sweep_L6.sbatch` | 5-minute L=6 sanity sweep. |
| `scripts/sweep_L10.sbatch` | Main easy-instance L=10 sweep (SA/PA/GA). |
| `scripts/sweep_L10_hard.sbatch` | Main hard-instance L=10 sweep (SA/PA/GA). |
| `scripts/qqa_winner_run.sbatch` | Replays the winning PQQA configuration $n=50$ times and re-renders the figure. |
| `scripts/verify_and_bench.sbatch` | Runs `speedups/verify.py` + `speedups/bench.py`. |
| `speedups/` | Optional drop-in GPU kernel optimisations + equivalence/benchmark scripts (see `speedups/README.md`, enabled with `run_sweep.py --optimized`). |

---

## 6. Verification checklist

- [x] `uv sync` succeeds and `.venv/bin/python -c "import torch; print(torch.version.cuda)"` prints `12.8`.
- [x] `make sweep-l6` finishes and `fresh_runs/sweep_L6_seed1736329224.csv` shows success rate 1.0 in every configuration.
- [x] `make sweep-l10-easy` writes 240 rows to `fresh_runs/sweep_L10_seed1736329224.csv`.
- [x] `make sweep-l10-hard` writes 240 rows to `fresh_runs/sweep_L10_seed310411727.csv`.
- [x] `make plots` produces `figures/success_vs_time_L10_easy.png` and `figures/success_vs_time_L10_hard.png` with the qualitative Fig. 2 ordering.
- [x] `make verify-bench` passes the bit-identical equivalence check and reports non-zero speedups.
- [x] `make pqqa-winner` writes `fresh_runs/winning/qqa_winner_G1.csv` with 100% success on $n=50$ runs at $32.46 \pm 0.20$ s/run on a single B200 (versus GA's 47.73 s for the same 100% success).
- [x] `make plot-pqqa-vs-ga` renders `figures/pqqa_vs_ga_pareto_L10_hard.png` — single ★ marks PQQA's winning config and the headline arrow shows the 32% wall-clock reduction over GA.
- [x] `make test-mc-polish` confirms the fp32 partial-matmul refactor of `_batched_mc_polish` is bit-identical to the reference (CPU-only, no GPU required).
