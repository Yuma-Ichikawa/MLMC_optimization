#!/bin/bash
#SBATCH --job-name=mlmc-repro-verify
#SBATCH --partition=batch-1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=Reproduction/logs/%x-%j.out
#SBATCH --error=Reproduction/logs/%x-%j.err
#
# Speedup investigation:
#  1. verify.py  – drive SA/PA/GA at the same PyTorch seed with and without the
#                  optimised kernels in Reproduction/speedups/ and assert that
#                  every min_energy / mean_energy trajectory is bit-identical.
#  2. bench.py   – measure the wall-clock speedup on the paper's L=10 easy
#                  instance at pop_size=8192 for a light (num_temps=30) sweep.
#
# Submit from the repo root:
#     sbatch --exclude=kagura-gpu07 Reproduction/scripts/verify_and_bench.sbatch
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
if [[ ! -f pyproject.toml || ! -d Code/Modern/optimization ]]; then
    echo "[FATAL] Expected repo root (got: ${REPO_ROOT})." >&2
    exit 1
fi
mkdir -p Reproduction/logs
# shellcheck disable=SC1091
source .venv/bin/activate

hostname; date; nvidia-smi -L || true

echo "=== [1/2] equivalence verification ==="
python Reproduction/speedups/verify.py

echo "=== [2/2] wall-clock benchmark (L=10, pop=8192, nT=30) ==="
python Reproduction/speedups/bench.py \
    --L 10 --seed 1736329224 \
    --pop-size 8192 --num-temps 30 --repeats 3

echo "=== done ==="
