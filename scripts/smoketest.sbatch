#!/bin/bash
#SBATCH --job-name=mlmc-smoketest
#SBATCH --partition=batch-1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=scripts/logs/%x-%j.out
#SBATCH --error=scripts/logs/%x-%j.err
#
# Small end-to-end smoke test:
#   * Simulated Annealing
#   * Population Annealing
#   * Global Annealing (MLMC)
# All three are run with reduced pop_size / MC steps so the entire job
# completes in a few minutes on a single GPU.
#
# Usage (from the repository root):
#   sbatch scripts/smoketest.sbatch

set -euo pipefail

# SLURM copies the job script to a spool directory before running it, so we
# cannot rely on $BASH_SOURCE to locate the repo. Instead, require that
# sbatch was invoked from the repository root and use SLURM_SUBMIT_DIR,
# falling back to the current working directory for interactive use.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"

# Sanity check: make sure we landed in the right place.
if [[ ! -f pyproject.toml || ! -d Code/Modern/optimization ]]; then
    echo "[FATAL] Expected to run from the repository root (found: ${REPO_ROOT})." >&2
    echo "        Invoke as: (from repo root) 'sbatch scripts/smoketest.sbatch'" >&2
    exit 1
fi

mkdir -p scripts/logs

# Activate the uv-managed virtual environment.
if [[ ! -d .venv ]]; then
    echo "[FATAL] .venv not found. Run 'uv sync' from the repo root first." >&2
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "=== ENVIRONMENT ==="
hostname
date
nvidia-smi || true
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_built", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY

# The annealing entry-points use paths relative to themselves
# (e.g. ../../../Data/Alpha/...), so run them from their own directory.
cd "${REPO_ROOT}/Code/Modern/optimization"

echo ""
echo "=== [1/3] Simulated Annealing (small) ==="
python simulated_annealing.py --pop_size 1000 --L 10 --MCsteps 50 --Cv_factor 1.618

echo ""
echo "=== [2/3] Population Annealing (small) ==="
python population_annealing.py --pop_size 1000 --L 10 --MCsteps 50 --Cv_factor 1.618

echo ""
echo "=== [3/3] Global Annealing / MLMC (small) ==="
python global_annealing.py \
    --pop_size 500 --L 10 \
    --MLMCsteps 2 --MCsteps 5 \
    --num_epochs_start 5 --num_epochs_retrain 1 --batch_size 128 \
    --Cv_factor 1.618

echo ""
echo "=== DONE ==="
