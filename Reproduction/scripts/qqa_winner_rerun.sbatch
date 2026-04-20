#!/bin/bash
#SBATCH --job-name=pqqa-rerun
#SBATCH --partition=batch-1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --exclude=kagura-gpu07
#SBATCH --output=Reproduction/logs/%x-%j.out
#SBATCH --error=Reproduction/logs/%x-%j.err
#
# One-shot GPU verification of the PyPI-installed `qqa==0.5.0`. Identical
# CLI to qqa_winner_run.sbatch but writes the CSV + figure to *_rerun
# paths so the original headline artefacts are NEVER touched.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p Reproduction/logs Reproduction/fresh_runs/winning
source .venv/bin/activate
hostname; date; nvidia-smi -L || true
python -c "import qqa, torch; print('qqa', qqa.__version__, 'from', qqa.__file__); print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"

COUPLING="Data/Alpha/Couplings/couplings_L10_R1_seed310411727.txt"
RERUN_CSV="Reproduction/fresh_runs/winning/qqa_winner_G1_rerun.csv"
RERUN_PNG="Reproduction/figures/pqqa_vs_ga_pareto_L10_hard_rerun.png"

python Reproduction/code/benchmark_pqqa_polish.py \
    --coupling-path "${COUPLING}" --L 10 \
    --sol-size 8192 --runs 50 --seed 310411727 \
    --num-epochs 3000 --lr 1.0 --pqqa-temp 0.001 \
    --curve-rate 6 --div-param 0.2 --min-bg -3.0 --max-bg 0.1 \
    --cool-sweeps 35000 --cool-t-high 2.0 --cool-t-low 0.02 \
    --init-random-frac 0.50 \
    --kick-cycles 23 --kick-t-high 0.7 --kick-t-low 0.05 --kick-sweeps 1500 \
    --ils-iters 2 --ils-k 5 \
    --mc-matmul-dtype bf16 \
    --algorithm-label "QQA" \
    --out-csv "${RERUN_CSV}" \
    --verbose

echo
echo "=== verification ==="
RERUN_CSV="${RERUN_CSV}" python - <<'PY'
import os, pandas as pd, numpy as np
MEC, GA_TIME = -1.6930031776, 47.73
df = pd.read_csv(os.environ["RERUN_CSV"])
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
echo "=== rendering rerun figure ==="
python Reproduction/code/plot_pqqa_winner.py \
    --baseline-csv Reproduction/fresh_runs/sweep_L10_seed310411727.csv \
    --winner-csv   "${RERUN_CSV}" \
    --out          "${RERUN_PNG}"

date
echo "=== done (rerun) ==="
