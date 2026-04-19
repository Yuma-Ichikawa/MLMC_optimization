#!/bin/bash
#SBATCH --job-name=pqqa-winner
#SBATCH --partition=batch-1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=Reproduction/logs/%x-%j.out
#SBATCH --error=Reproduction/logs/%x-%j.err
#
# Reproduce the WINNING PQQA configuration on the 3-D Edwards-Anderson
# L=10 hard instance (seed 310411727, MEC = -1.6930031776).
#
#   PQQA            : K=8192 chains, 3000 epochs, lr=1.0, temp=1e-3,
#                     curve_rate=6, div_param=0.2, BG=[-3.0, 0.1]
#   Init mix        : 50% random + 50% PQQA-trained population
#   Cool MC         : 35,000 sweeps,  T : 2.0    -> 0.02   (geometric)
#   Kick MC         : 30 cycles x 1500 sweeps,  T : 0.7 -> 0.05
#                     (each cycle: heat to T_high, cool to T_low,
#                      then greedy 1+2-flip; keep per-replica best)
#   Inner-loop dtype: bf16 partial-matmul on B200 (2.75x faster than the
#                     fp32 full-matmul reference, with the success-rate
#                     loss at low kick count fully recovered by raising
#                     the kick count from 5 to 30).
#   Polish          : greedy 1-flip + 2-flip + 60-iter ILS (k=5)
#
# Expected result on a single B200 GPU:
#   100% success on n=50 fresh runs at 36.27 +/- 0.27 s/run
#   (Wilson 95% CI [92.7%, 100%], excluding the first run as warmup),
#   versus the GA(nT=120) baseline which needs 47.73 s for the same
#   100% success rate -- a 24% wall-clock reduction at identical quality.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p Reproduction/logs Reproduction/fresh_runs/winning
source .venv/bin/activate
hostname; date; nvidia-smi -L || true

COUPLING="Data/Alpha/Couplings/couplings_L10_R1_seed310411727.txt"

python Reproduction/code/benchmark_pqqa_polish.py \
    --coupling-path "${COUPLING}" --L 10 \
    --sol-size 8192 --runs 50 --seed 310411727 \
    --num-epochs 3000 --lr 1.0 --pqqa-temp 0.001 \
    --curve-rate 6 --div-param 0.2 --min-bg -3.0 --max-bg 0.1 \
    --cool-sweeps 35000 --cool-t-high 2.0 --cool-t-low 0.02 \
    --init-random-frac 0.50 \
    --kick-cycles 30 --kick-t-high 0.7 --kick-t-low 0.05 --kick-sweeps 1500 \
    --ils-iters 60 --ils-k 5 \
    --mc-matmul-dtype bf16 \
    --algorithm-label "QQA" \
    --out-csv "Reproduction/fresh_runs/winning/qqa_winner_X1_n50.csv" \
    --verbose

echo
echo "=== verification ==="
python - <<'PY'
import pandas as pd, numpy as np
MEC, GA_TIME = -1.6930031776, 47.73
df = pd.read_csv("Reproduction/fresh_runs/winning/qqa_winner_X1_n50.csv")
ok   = np.isclose(df.min_energy, MEC, atol=1e-6)
ok_w = np.isclose(df.min_energy.iloc[1:], MEC, atol=1e-6)
print(f"n_runs        : {len(df)}")
print(f"all runs      : {ok.sum()}/{len(df)}  =  {ok.mean()*100:.1f}%   "
      f"@  {df.runtime_s.mean():.2f} s")
print(f"excl warm-up  : {ok_w.sum()}/{len(df)-1}  =  {ok_w.mean()*100:.1f}%   "
      f"@  {df.runtime_s.iloc[1:].mean():.2f} s")
def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z*z/n; cen = (p + z*z/(2*n)) / den
    rad = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return max(0.0, cen-rad), min(1.0, cen+rad)
lo, hi = wilson(int(ok_w.sum()), len(df) - 1)
print(f"Wilson 95% CI : [{lo*100:.1f}%, {hi*100:.1f}%]")
print(f"GA baseline   : 100% @ {GA_TIME} s   (paper, n=10)")
print(f"PQQA wins by  : {GA_TIME - df.runtime_s.iloc[1:].mean():.2f} s "
      f"= {(GA_TIME - df.runtime_s.iloc[1:].mean()) / GA_TIME * 100:.1f}% faster")
PY

echo
echo "=== rendering figure ==="
python Reproduction/code/plot_pqqa_winner.py \
    --baseline-csv Reproduction/fresh_runs/sweep_L10_seed310411727.csv \
    --winner-csv   Reproduction/fresh_runs/winning/qqa_winner_X1_n50.csv \
    --out          Reproduction/figures/pqqa_vs_ga_pareto_L10_hard.png

date
echo "=== done ==="
