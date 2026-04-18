#!/bin/bash
#SBATCH --job-name=mlmc-repro-L6
#SBATCH --partition=batch-1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=Reproduction/logs/%x-%j.out
#SBATCH --error=Reproduction/logs/%x-%j.err
#
# Small-scale reproduction of Fig. 2: SA/PA/GA success-vs-time on one instance.
# Submit from the repo root:  sbatch Reproduction/scripts/sweep_L6.sbatch
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
echo "=== Reproduction sweep (L=6) ==="

COUPLINGS="Data/Alpha/Couplings/couplings_L6_R1_seed1736329224.txt"
if [[ ! -f "${COUPLINGS}" ]]; then
    echo "[setup] synthesising ${COUPLINGS}"
    python Reproduction/code/generate_coupling.py --L 6 --seed 1736329224
fi

OUT="Reproduction/fresh_runs/sweep_L6_seed1736329224.csv"

python Reproduction/code/run_sweep.py \
    --L 6 \
    --seed 1736329224 \
    --pop-size 8192 \
    --runs 10 \
    --num-temps 10 20 30 50 \
    --schedule logT \
    --algorithms SA PA GA \
    --warmup \
    --out "${OUT}"

echo "=== done ==="
ls -la "${OUT}"
