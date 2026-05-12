#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/data1/sjy/MP1/MP1"
CONFIG_NAME="mp_terminal_goal_stage"
TASK_NAME="adroit_hammer"
GPU_ID="0"
SEED="0"
RUN_ROOT="${PROJECT_DIR}/data/outputs"
GOAL_BANK_ROOT="${PROJECT_DIR}/data/goal_bank_mp"
STAGE1_NUM="20"
STAGE3_NUM="10"
TERMINAL_WINDOW="8"
NUM_CLUSTERS="4"
GOAL_INDEX="0"
CKPT_PATH=""
PROTOTYPE_PATH=""

usage() {
  cat <<EOF
Usage:
  $0 stage1|stage2|stage3 [options]

Options:
  -g, --gpu ID              GPU id, default: ${GPU_ID}
  -t, --task NAME           Hydra task name, default: ${TASK_NAME}
  -c, --config NAME         Hydra config, default: ${CONFIG_NAME}
      --seed SEED           Training seed, default: ${SEED}
      --run-root DIR        Output root, default: ${RUN_ROOT}
      --goal-bank-root DIR  Goal-bank root, default: ${GOAL_BANK_ROOT}
      --stage1-num N        Stage1 episode count, default: ${STAGE1_NUM}
      --stage3-num N        Stage3 episode count, default: ${STAGE3_NUM}
      --terminal-window N   Terminal window, default: ${TERMINAL_WINDOW}
      --num-clusters K      Prototype cluster count, default: ${NUM_CLUSTERS}
      --goal-index N        Stage3 prototype index, default: ${GOAL_INDEX}
      --ckpt PATH           Stage1 ckpt or run dir. Required for stage2; optional for stage3 if auto found.
      --prototype-path PATH Stage3 prototype path. Defaults to goal-bank-root/task/k_K/prototypes.npy.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

STAGE="$1"
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    -g|--gpu) GPU_ID="$2"; shift 2 ;;
    -t|--task) TASK_NAME="$2"; shift 2 ;;
    -c|--config) CONFIG_NAME="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --goal-bank-root) GOAL_BANK_ROOT="$2"; shift 2 ;;
    --stage1-num) STAGE1_NUM="$2"; shift 2 ;;
    --stage3-num) STAGE3_NUM="$2"; shift 2 ;;
    --terminal-window) TERMINAL_WINDOW="$2"; shift 2 ;;
    --num-clusters) NUM_CLUSTERS="$2"; shift 2 ;;
    --goal-index) GOAL_INDEX="$2"; shift 2 ;;
    --ckpt) CKPT_PATH="$2"; shift 2 ;;
    --prototype-path) PROTOTYPE_PATH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
DEVICE="cuda:0"
RUN_NAME="${TASK_NAME}-${CONFIG_NAME}-stage${STAGE#stage}_seed${SEED}_k${NUM_CLUSTERS}"
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
DEFAULT_PROTOTYPE_PATH="${GOAL_BANK_ROOT}/${TASK_NAME}/k_${NUM_CLUSTERS}/prototypes.npy"
if [[ -z "${PROTOTYPE_PATH}" ]]; then
  PROTOTYPE_PATH="${DEFAULT_PROTOTYPE_PATH}"
fi

best_checkpoint_in_dir() {
  local dir="$1"
  local ckpt_dir="${dir}/checkpoints"
  if [[ -f "${ckpt_dir}/latest.ckpt" ]]; then
    echo "${ckpt_dir}/latest.ckpt"
    return 0
  fi
  if [[ -d "${ckpt_dir}" ]]; then
    local best
    best=$(ls -1 "${ckpt_dir}"/*.ckpt 2>/dev/null | sort -V | tail -n 1 || true)
    if [[ -n "${best}" ]]; then
      echo "${best}"
      return 0
    fi
  fi
  return 1
}

resolve_ckpt() {
  local input="$1"
  if [[ -f "${input}" ]]; then
    echo "${input}"
    return 0
  fi
  if [[ -d "${input}" ]]; then
    best_checkpoint_in_dir "${input}"
    return $?
  fi
  return 1
}

auto_stage1_ckpt() {
  if [[ -n "${CKPT_PATH}" ]]; then
    resolve_ckpt "${CKPT_PATH}"
    return $?
  fi
  local candidates
  candidates=$(ls -dt "${RUN_ROOT}/${TASK_NAME}-${CONFIG_NAME}-stage1_seed${SEED}"* 2>/dev/null || true)
  for dir in ${candidates}; do
    if best_checkpoint_in_dir "${dir}" >/dev/null; then
      best_checkpoint_in_dir "${dir}"
      return 0
    fi
  done
  return 1
}

COMMON_OVERRIDES=(
  "--config-name=${CONFIG_NAME}"
  "task=${TASK_NAME}"
  "training.device=${DEVICE}"
  "training.seed=${SEED}"
  "terminal_goal.stage1_num_train_episodes=${STAGE1_NUM}"
  "terminal_goal.stage3_num_train_episodes=${STAGE3_NUM}"
  "terminal_goal.terminal_window=${TERMINAL_WINDOW}"
  "terminal_goal.terminal_num_negatives=4"
  "terminal_goal.num_clusters=${NUM_CLUSTERS}"
  "terminal_goal.prototype_path=${PROTOTYPE_PATH}"
  "terminal_goal.goal_index=${GOAL_INDEX}"
  "hydra.run.dir=${RUN_DIR}"
  "hydra.sweep.dir=${RUN_DIR}"
  "multi_run.run_dir=${RUN_DIR}"
)

case "${STAGE}" in
  stage1|1)
    python "${PROJECT_DIR}/train.py" \
      "${COMMON_OVERRIDES[@]}" \
      "terminal_goal.stage=1" \
      "training.num_epochs=600" \
      "training.rollout_every=200" \
      "training.checkpoint_every=200"
    ;;
  stage2|2)
    STAGE1_CKPT=$(auto_stage1_ckpt) || {
      echo "Cannot find stage1 checkpoint. Pass --ckpt /path/to/stage1/run_or_ckpt"
      exit 1
    }
    python "${PROJECT_DIR}/scripts/build_mp_terminal_goal_bank.py" \
      --checkpoint "${STAGE1_CKPT}" \
      --task "${TASK_NAME}" \
      --config-name "${CONFIG_NAME}" \
      --ks "${NUM_CLUSTERS}" \
      --terminal-window "${TERMINAL_WINDOW}" \
      --output-root "${GOAL_BANK_ROOT}" \
      --device "${DEVICE}" \
      --max-episodes "${STAGE1_NUM}"
    ;;
  stage3|3)
    STAGE1_CKPT=$(auto_stage1_ckpt) || {
      echo "Cannot find stage1 checkpoint. Pass --ckpt /path/to/stage1/run_or_ckpt"
      exit 1
    }
    if [[ ! -f "${PROTOTYPE_PATH}" ]]; then
      echo "Prototype file not found: ${PROTOTYPE_PATH}"
      echo "Run stage2 first or pass --prototype-path."
      exit 1
    fi
    python "${PROJECT_DIR}/train.py" \
      "${COMMON_OVERRIDES[@]}" \
      "terminal_goal.stage=3" \
      "terminal_goal.terminal_encoder_ckpt=${STAGE1_CKPT}" \
      "terminal_goal.prototype_path=${PROTOTYPE_PATH}" \
      "training.num_epochs=3000" \
      "training.rollout_every=200" \
      "training.checkpoint_every=200"
    ;;
  *)
    echo "Unknown stage: ${STAGE}"
    usage
    exit 1
    ;;
esac
