#!/usr/bin/env bash
# Run terminal-representation training with absolute paths.
#
# Usage:
#   bash run_term.bash [GPU_ID] [CONFIG] [TASK] [SEED]
#
# Examples:
#   bash run_term.bash 3 mp_term metaworld_shelf-place 0
#   bash run_term.bash 3 mp1_term metaworld_shelf-place 0

set -euo pipefail

GPU_ID=${1:-3}
CONFIG=${2:-mp_term}
TASK=${3:-metaworld_shelf-place}
SEED=${4:-0}

PYTHON_BIN=${PYTHON_BIN:-/home/dbcloud/anaconda3/envs/MP1/bin/python}
LOGGING_MODE=${LOGGING_MODE:-online}
SAVE_CKPT=${SAVE_CKPT:-True}
ADDITION_INFO=${ADDITION_INFO:-0000}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${ROOT_DIR}/MP1"
EXP_NAME="${TASK}-${CONFIG}-${ADDITION_INFO}"
RUN_DIR="${PROJECT_DIR}/data/outputs/${EXP_NAME}_seed${SEED}"
ZARR_PATH="${PROJECT_DIR}/data/${TASK}_expert.zarr"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d "${ZARR_PATH}" ]]; then
  echo "Dataset not found: ${ZARR_PATH}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}"
cd "${PROJECT_DIR}"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "GPU=${GPU_ID} CONFIG=${CONFIG} TASK=${TASK} SEED=${SEED}"
echo "RUN_DIR=${RUN_DIR}"
echo "ZARR_PATH=${ZARR_PATH}"

exec "${PYTHON_BIN}" train.py \
  --config-name="${CONFIG}.yaml" \
  task="${TASK}" \
  task.dataset.zarr_path="${ZARR_PATH}" \
  hydra.run.dir="${RUN_DIR}" \
  training.debug=False \
  training.seed="${SEED}" \
  training.device="cuda:0" \
  exp_name="${EXP_NAME}" \
  logging.mode="${LOGGING_MODE}" \
  checkpoint.save_ckpt="${SAVE_CKPT}"
