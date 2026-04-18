#!/bin/bash
#SBATCH --job-name=mlmc-pa
#SBATCH --partition=batch-1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=scripts/logs/%x-%j.out
#SBATCH --error=scripts/logs/%x-%j.err
#
# Population Annealing at paper defaults.
# Override parameters via environment variables, e.g.:
#   POP_SIZE=20000 MC_STEPS=640 sbatch scripts/run_population_annealing.sbatch

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
if [[ ! -f pyproject.toml || ! -d Code/Modern/optimization ]]; then
    echo "[FATAL] Expected repo root (got: ${REPO_ROOT}). Run: sbatch scripts/run_population_annealing.sbatch" >&2
    exit 1
fi
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

echo "=== Population Annealing ==="
printf 'POP_SIZE=%s L=%s SEED=%s T=[%s,%s] Cv=%s MC=%s schedule=%s\n' \
    "${POP_SIZE}" "${L}" "${SEED}" "${T_START}" "${T_END}" "${CV_FACTOR}" "${MC_STEPS}" "${SCHEDULE}"
hostname; date; nvidia-smi -L || true

cd "${REPO_ROOT}/Code/Modern/optimization"
python population_annealing.py \
    --pop_size "${POP_SIZE}" \
    --L "${L}" \
    --seed "${SEED}" \
    --Tstart "${T_START}" \
    --Tend "${T_END}" \
    --Cv_factor "${CV_FACTOR}" \
    --MCsteps "${MC_STEPS}" \
    --schedule "${SCHEDULE}"
