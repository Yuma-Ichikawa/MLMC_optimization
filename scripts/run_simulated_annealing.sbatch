#!/bin/bash
#SBATCH --job-name=mlmc-sa
#SBATCH --partition=batch-1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=scripts/logs/%x-%j.out
#SBATCH --error=scripts/logs/%x-%j.err
#
# Simulated Annealing at paper defaults.
# Override any parameter via environment variables, e.g.:
#   POP_SIZE=20000 MC_STEPS=640 sbatch scripts/run_simulated_annealing.sbatch
#
# The output line format is:
#   <MCsteps> <num_temps> <schedule> <min_energy> <mean_energy@T=0.1> <runtime_s>

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
MC_STEPS="${MC_STEPS:-320}"
SCHEDULE="${SCHEDULE:-Cv_beta}"

echo "=== Simulated Annealing ==="
printf 'POP_SIZE=%s L=%s SEED=%s T=[%s,%s] Cv=%s MC=%s schedule=%s\n' \
    "${POP_SIZE}" "${L}" "${SEED}" "${T_START}" "${T_END}" "${CV_FACTOR}" "${MC_STEPS}" "${SCHEDULE}"
hostname; date; nvidia-smi -L || true

cd "${REPO_ROOT}/Code/Modern/optimization"
python simulated_annealing.py \
    --pop_size "${POP_SIZE}" \
    --L "${L}" \
    --seed "${SEED}" \
    --Tstart "${T_START}" \
    --Tend "${T_END}" \
    --Cv_factor "${CV_FACTOR}" \
    --MCsteps "${MC_STEPS}" \
    --schedule "${SCHEDULE}"
