#!/bin/bash
#SBATCH --job-name=mlmc-repro-L10h
#SBATCH --partition=batch-1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=Reproduction/logs/%x-%j.out
#SBATCH --error=Reproduction/logs/%x-%j.err
#
# Reproduction of Fig. 2 (right panel, hard instance): SA/PA/GA success-vs-time
# on the paper's reference "hard" L=10 instance (seed 310411727, the default
# seed in the paper's upstream scripts and the instance with the most run data
# shipped in Data/Omega/).
#
# Population size is reduced 16x (2^13 vs 2^17 in the paper) so the sweep fits
# in a few hours on one B200 while keeping L and schedule identical.
#
# Submit from the repo root:
#     sbatch --exclude=kagura-gpu07 Reproduction/scripts/sweep_L10_hard.sbatch
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
if [[ ! -f pyproject.toml || ! -d Code/Modern/optimization ]]; then
    echo "[FATAL] Expected repo root (got: ${REPO_ROOT})." >&2
    exit 1
fi
mkdir -p Reproduction/logs Reproduction/fresh_runs
# shellcheck disable=SC1091
source .venv/bin/activate

hostname; date; nvidia-smi -L || true
echo "=== Reproduction sweep (L=10, hard instance) ==="

OUT="Reproduction/fresh_runs/sweep_L10_seed310411727.csv"

python Reproduction/code/run_sweep.py \
    --L 10 \
    --seed 310411727 \
    --pop-size 8192 \
    --runs 10 \
    --num-temps 5 10 20 30 50 80 120 180 \
    --schedule logT \
    --algorithms SA PA GA \
    --optimized \
    --warmup \
    --out "${OUT}"

echo "=== done ==="
ls -la "${OUT}"
