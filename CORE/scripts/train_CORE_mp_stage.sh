#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_NAME="CORE_mp"
TASK_NAME="metaworld_box-close"
GPU_ID="0"
SEED="0"
RUN_ROOT="${PROJECT_DIR}/data/outputs"
GOAL_BANK_ROOT="${PROJECT_DIR}/data/goal_bank_CORE_mp"
STAGE1_NUM="20"
STAGE3_NUM="10"
STAGE1_EPOCHS="600"
TERMINAL_WINDOW="1"
NUM_CLUSTERS="1"
GOAL_SELECTION="common"
SOFT_NEAREST_TEMPERATURE="0.1"
GOAL_INDEX=""
CKPT_PATH=""
PROTOTYPE_PATH=""
COMMON_PROTOTYPE_PATH=""
CKPT_SELECT="best"

usage() {
  cat <<EOF
Usage:
  $0 stage1|stage2|stage3 [options]

Options:
  -g, --gpu ID                    GPU id, default: ${GPU_ID}
  -t, --task NAME                 Hydra task name, default: ${TASK_NAME}
  -c, --config NAME               Hydra config, default: ${CONFIG_NAME}
      --seed SEED                 Training seed, default: ${SEED}
      --run-root DIR              Output root, default: ${RUN_ROOT}
      --goal-bank-root DIR        Goal-bank root, default: ${GOAL_BANK_ROOT}
      --stage1-num N              Stage1/stage2 episode count, default: ${STAGE1_NUM}
      --stage3-num N              Stage3 episode count, default: ${STAGE3_NUM}
      --stage1-epochs N           Stage1 training epochs, default: ${STAGE1_EPOCHS}
      --terminal-window N         Terminal frame window for auxiliary samples. Default: ${TERMINAL_WINDOW}
      --num-clusters K            Output folder k_K, default: ${NUM_CLUSTERS}; no KMeans is run.
      --goal-selection MODE       Goal prototype selection mode. Default: ${GOAL_SELECTION}
      --goal-index N              Required only when --goal-selection fixed.
      --soft-temperature V        Soft nearest temperature, default: ${SOFT_NEAREST_TEMPERATURE}
      --ckpt PATH                 Stage1 ckpt or run dir. Required for stage2; optional for stage3 if auto found.
      --ckpt-select MODE          best | last/latest. Default: ${CKPT_SELECT}
      --prototype-path PATH       Defaults to goal-bank-root/task/k_K/prototypes.npy.
      --common-prototype-path PATH Defaults to goal-bank-root/task/k_K/common_prototype.npy.
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
    --stage1-epochs) STAGE1_EPOCHS="$2"; shift 2 ;;
    --terminal-window) TERMINAL_WINDOW="$2"; shift 2 ;;
    --num-clusters) NUM_CLUSTERS="$2"; shift 2 ;;
    --goal-selection) GOAL_SELECTION="$2"; shift 2 ;;
    --goal-index) GOAL_INDEX="$2"; shift 2 ;;
    --soft-temperature|--soft-nearest-temperature) SOFT_NEAREST_TEMPERATURE="$2"; shift 2 ;;
    --ckpt) CKPT_PATH="$2"; shift 2 ;;
    --ckpt-select) CKPT_SELECT="$2"; shift 2 ;;
    --prototype-path) PROTOTYPE_PATH="$2"; shift 2 ;;
    --common-prototype-path) COMMON_PROTOTYPE_PATH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

case "${GOAL_SELECTION}" in
  fixed|common|soft_nearest) ;;
  *) echo "--goal-selection must be fixed, common, or soft_nearest"; exit 1 ;;
esac
case "${CKPT_SELECT}" in
  best|last|latest) ;;
  *) echo "--ckpt-select must be best | last | latest"; exit 1 ;;
esac

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
DEVICE="cuda:0"
RUN_NAME="${TASK_NAME}-${CONFIG_NAME}-stage${STAGE#stage}_seed${SEED}_k${NUM_CLUSTERS}"
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
DEFAULT_PROTOTYPE_PATH="${GOAL_BANK_ROOT}/${TASK_NAME}/k_${NUM_CLUSTERS}/prototypes.npy"
DEFAULT_COMMON_PROTOTYPE_PATH="${GOAL_BANK_ROOT}/${TASK_NAME}/k_${NUM_CLUSTERS}/common_prototype.npy"
if [[ -z "${PROTOTYPE_PATH}" ]]; then
  PROTOTYPE_PATH="${DEFAULT_PROTOTYPE_PATH}"
fi
if [[ -z "${COMMON_PROTOTYPE_PATH}" ]]; then
  COMMON_PROTOTYPE_PATH="${DEFAULT_COMMON_PROTOTYPE_PATH}"
fi

best_checkpoint_in_dir() {
  local dir="$1"
  local ckpt_dir="${dir}/checkpoints"
  local topk_summary="${dir}/eval_results/topk_summary.json"

  if [[ "${CKPT_SELECT}" == "best" && -f "${topk_summary}" ]]; then
    local best_from_summary
    best_from_summary=$(python - "${topk_summary}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
payload = json.loads(summary_path.read_text())
items = payload.get("items") or []
if items:
    checkpoint = Path(str(items[0].get("checkpoint", "")))
    if checkpoint.is_file():
        print(checkpoint)
PY
)
    if [[ -n "${best_from_summary}" ]]; then
      echo "${best_from_summary}"
      return 0
    fi
  fi

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
  "terminal_goal.goal_selection=${GOAL_SELECTION}"
  "terminal_goal.soft_nearest_temperature=${SOFT_NEAREST_TEMPERATURE}"
  "terminal_goal.prototype_path=${PROTOTYPE_PATH}"
  "terminal_goal.common_prototype_path=${COMMON_PROTOTYPE_PATH}"
  "hydra.run.dir=${RUN_DIR}"
  "hydra.sweep.dir=${RUN_DIR}"
  "multi_run.run_dir=${RUN_DIR}"
)

if [[ -n "${GOAL_INDEX}" ]]; then
  COMMON_OVERRIDES+=("terminal_goal.goal_index=${GOAL_INDEX}")
fi

case "${STAGE}" in
  stage1|1)
    python "${PROJECT_DIR}/train.py" \
      "${COMMON_OVERRIDES[@]}" \
      "terminal_goal.stage=1" \
      "training.num_epochs=${STAGE1_EPOCHS}" \
      "training.rollout_every=200" \
      "training.checkpoint_every=200"
    ;;
  stage2|2)
    STAGE1_CKPT=$(auto_stage1_ckpt) || {
      echo "Cannot find stage1 checkpoint. Pass --ckpt /path/to/stage1/run_or_ckpt"
      exit 1
    }
    python "${PROJECT_DIR}/scripts/build_CORE_mp_bank.py" \
      --checkpoint "${STAGE1_CKPT}" \
      --task "${TASK_NAME}" \
      --config-name "${CONFIG_NAME}" \
      --k "${NUM_CLUSTERS}" \
      --output-root "${GOAL_BANK_ROOT}" \
      --device "${DEVICE}" \
      --max-episodes "${STAGE1_NUM}"
    ;;
  stage3|3)
    STAGE1_CKPT=$(auto_stage1_ckpt) || {
      echo "Cannot find stage1 checkpoint. Pass --ckpt /path/to/stage1/run_or_ckpt"
      exit 1
    }
    if [[ "${GOAL_SELECTION}" == "fixed" && -z "${GOAL_INDEX}" ]]; then
      echo "--goal-selection fixed requires --goal-index."
      exit 1
    fi
    if [[ "${GOAL_SELECTION}" == "common" && ! -f "${COMMON_PROTOTYPE_PATH}" ]]; then
      echo "Common prototype file not found: ${COMMON_PROTOTYPE_PATH}"
      echo "Run stage2 first or pass --common-prototype-path."
      exit 1
    fi
    if [[ "${GOAL_SELECTION}" != "common" && ! -f "${PROTOTYPE_PATH}" ]]; then
      echo "Prototype file not found: ${PROTOTYPE_PATH}"
      echo "Run stage2 first or pass --prototype-path."
      exit 1
    fi
    python "${PROJECT_DIR}/train.py" \
      "${COMMON_OVERRIDES[@]}" \
      "terminal_goal.stage=3" \
      "terminal_goal.terminal_encoder_ckpt='${STAGE1_CKPT}'" \
      "terminal_goal.prototype_path=${PROTOTYPE_PATH}" \
      "terminal_goal.common_prototype_path=${COMMON_PROTOTYPE_PATH}" \
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
