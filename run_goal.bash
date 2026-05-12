#!/usr/bin/env bash
# Run second-stage MP goal-conditioned training with absolute paths.
#
# Usage:
#   bash run_goal.bash [GPU_ID] [CONFIG] [TASK] [SEED] [TERM_CONFIG] [GOAL_K] [TERM_SEED] [EXTRA_HYDRA_OVERRIDES...]
#
# Example:
#   bash run_goal.bash 3 mp_goal metaworld_shelf-place 0 mp_term 1 0

set -euo pipefail

GPU_ID=${1:-3}
CONFIG=${2:-mp_goal}
TASK=${3:-metaworld_shelf-place}
SEED=${4:-0}
TERM_CONFIG=${5:-mp_term}
GOAL_K=${6:-1}
TERM_SEED=${7:-0}
EXTRA_HYDRA_OVERRIDES=("${@:8}")

PYTHON_BIN=${PYTHON_BIN:-/home/dbcloud/anaconda3/envs/MP1/bin/python}
LOGGING_MODE=${LOGGING_MODE:-online}
SAVE_CKPT=${SAVE_CKPT:-True}
ADDITION_INFO=${ADDITION_INFO:-0000}
RUN_ADDITION_INFO=${RUN_ADDITION_INFO:-${ADDITION_INFO}}
TERM_ADDITION_INFO=${TERM_ADDITION_INFO:-${ADDITION_INFO}}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${ROOT_DIR}/MP1"
EXP_NAME="${TASK}-${CONFIG}-${RUN_ADDITION_INFO}"
RUN_DIR="${PROJECT_DIR}/data/outputs/${EXP_NAME}_seed${SEED}"
ZARR_PATH="${PROJECT_DIR}/data/${TASK}_expert.zarr"
GOAL_BANK_PATH="${PROJECT_DIR}/data/goal_bank/${TASK}/k_${GOAL_K}"
TERM_CKPT_PATH="${PROJECT_DIR}/data/outputs/${TASK}-${TERM_CONFIG}-${TERM_ADDITION_INFO}_seed${TERM_SEED}/checkpoints/latest.ckpt"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d "${ZARR_PATH}" ]]; then
  echo "Dataset not found: ${ZARR_PATH}" >&2
  exit 1
fi

if [[ ! -d "${GOAL_BANK_PATH}" ]]; then
  echo "Goal bank not found: ${GOAL_BANK_PATH}" >&2
  exit 1
fi

if [[ ! -f "${TERM_CKPT_PATH}" ]]; then
  echo "Terminal checkpoint not found: ${TERM_CKPT_PATH}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}"
cd "${PROJECT_DIR}"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "GPU=${GPU_ID} CONFIG=${CONFIG} TASK=${TASK} SEED=${SEED}"
echo "RUN_DIR=${RUN_DIR}"
echo "ZARR_PATH=${ZARR_PATH}"
echo "GOAL_BANK_PATH=${GOAL_BANK_PATH}"
echo "TERM_CKPT_PATH=${TERM_CKPT_PATH}"
echo "EXTRA_HYDRA_OVERRIDES=${EXTRA_HYDRA_OVERRIDES[*]:-}"

exec "${PYTHON_BIN}" train.py \
  --config-name="${CONFIG}.yaml" \
  task="${TASK}" \
  task.dataset.zarr_path="${ZARR_PATH}" \
  policy.goal_bank_path="${GOAL_BANK_PATH}" \
  policy.term_checkpoint_path="${TERM_CKPT_PATH}" \
  hydra.run.dir="${RUN_DIR}" \
  training.debug=False \
  training.seed="${SEED}" \
  training.device="cuda:0" \
  exp_name="${EXP_NAME}" \
  logging.mode="${LOGGING_MODE}" \
  checkpoint.save_ckpt="${SAVE_CKPT}" \
  "${EXTRA_HYDRA_OVERRIDES[@]}"
