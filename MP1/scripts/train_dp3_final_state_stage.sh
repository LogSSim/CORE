#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash MP1/scripts/train_dp3_final_state_stage.sh [options] stage1 [hydra_overrides...]
  bash MP1/scripts/train_dp3_final_state_stage.sh [options] stage2 [stage1_run_dir_or_ckpt] [stage2_args...]
  bash MP1/scripts/train_dp3_final_state_stage.sh [options] stage3 [stage2_feature.npz] [hydra_overrides...]

Stage aliases:
  1, stage1    Train vanilla DP3 on the first 20 zarr demos.
  2, stage2    Encode the final frame of the first 20 zarr demos and cluster features.
  3, stage3    Train DP3 on the first 10 zarr demos with final-state conditions.

Common environment overrides:
  CONFIG_NAME=dp3_final_state_stage
  TASK_NAME=adroit_hammer
  GPU=0
  STAGE1_NUM=20
  STAGE3_NUM=10
  NUM_CLUSTERS=4
  FEATURE_PATH=/data1/sjy/MP1/MP1/data/final_state_features/${TASK_NAME}_stage2_final_state_k${NUM_CLUSTERS}.npz
  STAGE1_CKPT=/path/to/stage1/latest.ckpt

Options:
  -c, --config NAME         Hydra config name. Default: dp3_final_state_stage
  -t, --task NAME           Hydra task name, e.g. adroit_hammer, metaworld_shelf-place
  -g, --gpu ID              GPU id. Default: 0
  --stage1-num N            Number of zarr demos for stage1/stage2. Default: 20
  --stage3-num N            Number of zarr demos for stage3. Default: 10
  --num-clusters N          Stage2 k-means clusters. Default: 4
  --feature-path PATH       Stage2 output and stage3 input npz path.
  --ckpt PATH               Stage1 checkpoint/run dir. Stage2 defaults to the stage1 run dir.
  --seed N                  Training seed. Default: 0
  --run-root PATH           Output root. Default: /data1/sjy/MP1/MP1/data/outputs

Examples:
  bash MP1/scripts/train_dp3_final_state_stage.sh -g 0 -t adroit_hammer stage1

  bash MP1/scripts/train_dp3_final_state_stage.sh -g 0 -t adroit_hammer stage2 \
    data/outputs/.../checkpoints/latest.ckpt

  bash MP1/scripts/train_dp3_final_state_stage.sh -g 0 -t adroit_hammer stage3 \
    /data1/sjy/MP1/MP1/data/final_state_features/adroit_hammer_stage2_final_state.npz
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd -- "${PROJECT_DIR}/.." && pwd)"
TRAIN_PY="${PROJECT_DIR}/train.py"
STAGE2_PY="${PROJECT_DIR}/scripts/build_dp3_final_state_features.py"
cd "${WORKSPACE_DIR}"

CONFIG_NAME="${CONFIG_NAME:-dp3_final_state_stage}"
TASK_NAME="${TASK_NAME:-adroit_hammer}"
GPU="${GPU:-0}"
STAGE1_NUM="${STAGE1_NUM:-20}"
STAGE3_NUM="${STAGE3_NUM:-10}"
NUM_CLUSTERS="${NUM_CLUSTERS:-4}"
FEATURE_PATH="${FEATURE_PATH:-}"
STAGE1_CKPT="${STAGE1_CKPT:-}"
SEED="${SEED:-0}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/data/outputs}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)
      CONFIG_NAME="$2"
      shift 2
      ;;
    -t|--task)
      TASK_NAME="$2"
      shift 2
      ;;
    -g|--gpu)
      GPU="$2"
      shift 2
      ;;
    --stage1-num)
      STAGE1_NUM="$2"
      shift 2
      ;;
    --stage3-num)
      STAGE3_NUM="$2"
      shift 2
      ;;
    --num-clusters)
      NUM_CLUSTERS="$2"
      shift 2
      ;;
    --feature-path)
      FEATURE_PATH="$2"
      shift 2
      ;;
    --ckpt|--stage1-ckpt)
      STAGE1_CKPT="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --run-root)
      RUN_ROOT="$2"
      shift 2
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

STAGE="$1"
shift

if [[ -z "${FEATURE_PATH}" ]]; then
  FEATURE_PATH="${PROJECT_DIR}/data/final_state_features/${TASK_NAME}_stage2_final_state_k${NUM_CLUSTERS}.npz"
fi
DEVICE="cuda:${GPU}"
STAGE1_RUN_DIR="${RUN_ROOT}/${TASK_NAME}-${CONFIG_NAME}-stage1_seed${SEED}_k${NUM_CLUSTERS}"
STAGE3_RUN_DIR="${RUN_ROOT}/${TASK_NAME}-${CONFIG_NAME}-stage3_seed${SEED}_k${NUM_CLUSTERS}"

TRAIN_COMMON_OVERRIDES=(
  "task=${TASK_NAME}"
  "training.device=${DEVICE}"
  "training.seed=${SEED}"
  "final_state.stage1_num_train_episodes=${STAGE1_NUM}"
  "final_state.stage3_num_train_episodes=${STAGE3_NUM}"
  "final_state.feature_path=${FEATURE_PATH}"
)

best_checkpoint_in_dir() {
  local run_dir="$1"
  python - "$run_dir" <<'PY'
import glob
import os
import re
import sys

run_dir = sys.argv[1]
ckpt_dir = os.path.join(run_dir, "checkpoints")
pattern = os.path.join(ckpt_dir, "epoch=*-test_mean_score=*.ckpt")
best_path = ""
best_score = None
for path in glob.glob(pattern):
    match = re.search(r"test_mean_score=([-+0-9.eE]+)\.ckpt$", os.path.basename(path))
    if match is None:
        continue
    score = float(match.group(1))
    if best_score is None or score > best_score:
        best_score = score
        best_path = path
if best_path:
    print(best_path)
PY
}

resolve_checkpoint_path() {
  local ckpt="$1"
  if [[ -d "${ckpt}" ]]; then
    local best_ckpt
    best_ckpt="$(best_checkpoint_in_dir "${ckpt}")"
    if [[ -n "${best_ckpt}" && -f "${best_ckpt}" ]]; then
      ckpt="${best_ckpt}"
    elif [[ -f "${ckpt}/checkpoints/latest.ckpt" ]]; then
      ckpt="${ckpt}/checkpoints/latest.ckpt"
    elif [[ -f "${ckpt}/latest.ckpt" ]]; then
      ckpt="${ckpt}/latest.ckpt"
    else
      echo "Checkpoint directory does not contain checkpoints/latest.ckpt or latest.ckpt: ${ckpt}" >&2
      exit 2
    fi
  fi
  echo "${ckpt}"
}

take_checkpoint_arg() {
  local ckpt="${STAGE1_CKPT:-}"
  if [[ -z "${ckpt}" && $# -gt 0 && "$1" != -* ]]; then
    ckpt="$1"
    shift
  fi
  if [[ -z "${ckpt}" ]]; then
    ckpt="${STAGE1_RUN_DIR}"
  fi
  if [[ -z "${ckpt}" ]]; then
    echo "stage2 needs a stage1 checkpoint path." >&2
    exit 2
  fi
  ckpt="$(resolve_checkpoint_path "${ckpt}")"
  STAGE1_CKPT_RESOLVED="${ckpt}"
  REMAINING_ARGS=("$@")
}

take_stage3_args() {
  local maybe_feature=""
  STAGE3_CKPT_OVERRIDE=()
  if [[ $# -gt 0 && "$1" != -* && "$1" != *=* ]]; then
    maybe_feature="$1"
    shift
    if [[ "${maybe_feature}" == *.npz ]]; then
      FEATURE_PATH="${maybe_feature}"
    else
      # Backward compatibility: a run directory or ckpt can still be passed here.
      STAGE1_CKPT="${maybe_feature}"
    fi
  fi
  if [[ -n "${STAGE1_CKPT:-}" ]]; then
    STAGE1_CKPT_RESOLVED="$(resolve_checkpoint_path "${STAGE1_CKPT}")"
    STAGE3_CKPT_OVERRIDE=("final_state.stage1_checkpoint_path=${STAGE1_CKPT_RESOLVED}")
  fi
  REMAINING_ARGS=("$@")
}

case "${STAGE}" in
  1|stage1)
    python "${TRAIN_PY}" --config-name="${CONFIG_NAME}" \
      final_state.stage=1 \
      hydra.run.dir="${STAGE1_RUN_DIR}" \
      hydra.sweep.dir="${STAGE1_RUN_DIR}" \
      multi_run.run_dir="${STAGE1_RUN_DIR}" \
      exp_name="${TASK_NAME}-${CONFIG_NAME}-stage1_k${NUM_CLUSTERS}" \
      "${TRAIN_COMMON_OVERRIDES[@]}" \
      "$@"
    ;;

  2|stage2)
    take_checkpoint_arg "$@"
    python "${STAGE2_PY}" \
      --checkpoint "${STAGE1_CKPT_RESOLVED}" \
      --output "${FEATURE_PATH}" \
      --num-episodes "${STAGE1_NUM}" \
      --num-clusters "${NUM_CLUSTERS}" \
      --device "${DEVICE}" \
      --task "${TASK_NAME}" \
      "${REMAINING_ARGS[@]}"
    ;;

  3|stage3)
    take_stage3_args "$@"
    python "${TRAIN_PY}" --config-name="${CONFIG_NAME}" \
      final_state.stage=3 \
      hydra.run.dir="${STAGE3_RUN_DIR}" \
      hydra.sweep.dir="${STAGE3_RUN_DIR}" \
      multi_run.run_dir="${STAGE3_RUN_DIR}" \
      exp_name="${TASK_NAME}-${CONFIG_NAME}-stage3_k${NUM_CLUSTERS}" \
      "${TRAIN_COMMON_OVERRIDES[@]}" \
      final_state.feature_path="${FEATURE_PATH}" \
      final_state.num_clusters=${NUM_CLUSTERS} \
      "${STAGE3_CKPT_OVERRIDE[@]}" \
      training.num_epochs=3000 \
      training.rollout_every=200 \
      training.checkpoint_every=200 \
      "${REMAINING_ARGS[@]}"
    ;;

  -h|--help|help)
    usage
    ;;

  *)
    echo "Unknown stage: ${STAGE}" >&2
    usage
    exit 1
    ;;
esac
