#!/bin/bash
#SBATCH --job-name=mlmc-repro-L10
#SBATCH --partition=batch-1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=Reproduction/logs/%x-%j.out
#SBATCH --error=Reproduction/logs/%x-%j.err
#
# Reproduction of Fig. 2: SA/PA/GA success-vs-time on the paper's easy L=10
# instance (seed 1736329224). Population size reduced 16x (2^13 vs 2^17 in the
# paper) so the sweep fits in <1 h on one B200, while keeping L and schedule
# identical to the paper.
#
# Submit from the repo root:  sbatch --exclude=kagura-gpu07 Reproduction/scripts/sweep_L10.sbatch
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
echo "=== Reproduction sweep (L=10, easy instance) ==="

OUT="Reproduction/fresh_runs/sweep_L10_seed1736329224.csv"

python Reproduction/code/run_sweep.py \
    --L 10 \
    --seed 1736329224 \
    --pop-size 8192 \
    --runs 10 \
    --num-temps 5 10 20 30 50 80 120 180 \
    --schedule logT \
    --algorithms SA PA GA \
    --warmup \
    --out "${OUT}"

echo "=== done ==="
ls -la "${OUT}"
