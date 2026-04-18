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

# Resolve repo root from the location of this script so the job works no
# matter where sbatch was invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

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
