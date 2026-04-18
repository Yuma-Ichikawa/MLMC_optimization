#!/bin/bash
#SBATCH --job-name=mlmc-ga
#SBATCH --partition=batch-1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=scripts/logs/%x-%j.out
#SBATCH --error=scripts/logs/%x-%j.err
#
# Global Annealing (MLMC) at paper defaults.
# Override parameters via environment variables, e.g.:
#   POP_SIZE=20000 MLMC_STEPS=10 MC_STEPS=30 sbatch scripts/run_global_annealing.sbatch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p scripts/logs
# shellcheck disable=SC1091
source .venv/bin/activate

POP_SIZE="${POP_SIZE:-10000}"
L="${L:-10}"
SEED="${SEED:-310411727}"
T_START="${T_START:-1.92}"
T_END="${T_END:-0.1}"
CV_FACTOR="${CV_FACTOR:-1.618}"
MLMC_STEPS="${MLMC_STEPS:-5}"
MC_STEPS="${MC_STEPS:-15}"
EPOCHS_START="${EPOCHS_START:-40}"
EPOCHS_RETRAIN="${EPOCHS_RETRAIN:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SCHEDULE="${SCHEDULE:-Cv_beta}"

echo "=== Global Annealing (MLMC) ==="
printf 'POP_SIZE=%s L=%s SEED=%s T=[%s,%s] Cv=%s MLMC=%s MC=%s epochs=%s/%s bs=%s schedule=%s\n' \
    "${POP_SIZE}" "${L}" "${SEED}" "${T_START}" "${T_END}" "${CV_FACTOR}" \
    "${MLMC_STEPS}" "${MC_STEPS}" "${EPOCHS_START}" "${EPOCHS_RETRAIN}" "${BATCH_SIZE}" "${SCHEDULE}"
hostname; date; nvidia-smi -L || true

cd "${REPO_ROOT}/Code/Modern/optimization"
python global_annealing.py \
    --pop_size "${POP_SIZE}" \
    --L "${L}" \
    --seed "${SEED}" \
    --Tstart "${T_START}" \
    --Tend "${T_END}" \
    --Cv_factor "${CV_FACTOR}" \
    --MLMCsteps "${MLMC_STEPS}" \
    --MCsteps "${MC_STEPS}" \
    --num_epochs_start "${EPOCHS_START}" \
    --num_epochs_retrain "${EPOCHS_RETRAIN}" \
    --batch_size "${BATCH_SIZE}" \
    --schedule "${SCHEDULE}"
