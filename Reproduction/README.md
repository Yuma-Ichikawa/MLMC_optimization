# Third-party Reproduction

End-to-end reproduction of **Fig. 2** of Del Bono, Ricci-Tersenghi &
Zamponi (arXiv:2510.19544) on a single GPU. Nothing under
`Reproduction/` modifies the upstream algorithm code in `Code/` — it is
purely a thin driver + plotting layer.

```
Reproduction/
├── README.md                   ← this file
├── code/
│   ├── generate_coupling.py    ← build 3-D Edwards–Anderson couplings
│   ├── run_sweep.py            ← run SA / PA / GA over a grid of num_temps
│   └── plot_success_vs_time.py ← render the Fig. 2-style figure
├── scripts/                    ← SLURM launchers (one per sweep)
├── speedups/                   ← optional optimised kernels + benchmark
├── fresh_runs/                 ← CSV outputs (one row per (alg, nT, run))
├── figures/                    ← PNG/PDF figures built from fresh_runs/
└── logs/                       ← SLURM stdout/stderr (gitignored)
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

## 4. File-by-file

| Path | Purpose |
|---|---|
| `code/generate_coupling.py` | Synthesise an $L \times L \times L$ Edwards–Anderson coupling file (Gaussian $J$, periodic boundaries). Output matches `Data/Alpha/Couplings/couplings_L{L}_R1_seed{seed}.txt` bit for bit. |
| `code/run_sweep.py` | Drives SA / PA / GA on a fixed instance for each requested `num_temps`, writes one tidy CSV. Imports the algorithm implementations verbatim from `Code/Modern/optimization/`. `--optimized` swaps in the optimised kernels under `Reproduction/speedups/`. |
| `code/plot_success_vs_time.py` | Renders the success-vs-time curve with a normalised-logistic fit per algorithm (only when the data shows a transition). |
| `scripts/sweep_L6.sbatch` | 5-minute L=6 sanity sweep. |
| `scripts/sweep_L10.sbatch` | Main easy-instance L=10 sweep. |
| `scripts/sweep_L10_hard.sbatch` | Main hard-instance L=10 sweep. |
| `scripts/verify_and_bench.sbatch` | Runs `speedups/verify.py` + `speedups/bench.py`. |
| `speedups/` | Optional drop-in GPU kernel optimisations, an equivalence test (`verify.py`), and a wall-clock benchmark (`bench.py`). See `speedups/README.md`. Enabled with `run_sweep.py --optimized`. |

---

## 5. Verification checklist

- [x] `uv sync` succeeds and `.venv/bin/python -c "import torch; print(torch.version.cuda)"` prints `12.8`.
- [x] `make sweep-l6` finishes and `fresh_runs/sweep_L6_seed1736329224.csv` shows success rate 1.0 in every configuration.
- [x] `make sweep-l10-easy` writes 240 rows to `fresh_runs/sweep_L10_seed1736329224.csv`.
- [x] `make sweep-l10-hard` writes 240 rows to `fresh_runs/sweep_L10_seed310411727.csv`.
- [x] `make plots` produces `figures/success_vs_time_L10_easy.png` and `figures/success_vs_time_L10_hard.png` with the qualitative Fig. 2 ordering.
- [x] `make verify-bench` passes the bit-identical equivalence check and reports non-zero speedups.
