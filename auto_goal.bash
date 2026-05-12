#!/usr/bin/env bash
# -------------------------------------------------------------------------
# auto_goal.bash GPU_ID CONFIG [TASKS] [SEEDS] [TERM_CONFIG] [GOAL_K] [TERM_SEED]
#
# Sequentially run goal-conditioned Stage 2 training, similar to auto_run.bash.
# TASKS and SEEDS accept comma-separated lists.
#
# Examples:
#   bash auto_goal.bash 3 mp_goal metaworld_shelf-place 0,1,2 mp_term 1 0
#   nohup bash auto_goal.bash 3 mp_goal metaworld_shelf-place 0,1,2 mp_term 1 0 \
#     > auto_goal_shelf-place.log 2>&1 &
# -------------------------------------------------------------------------

set -euo pipefail

show_usage() {
  cat <<EOF
use:
  $0 <GPU_ID> <CONFIG> [TASKS] [SEEDS] [TERM_CONFIG] [GOAL_K] [TERM_SEED]

examples:
  $0 3 mp_goal metaworld_shelf-place 0,1,2 mp_term 1 0
  $0 3 mp_goal "" 0,1,2 mp_term 1 0

notes:
  - Runs are sequential on one GPU, matching auto_run.bash behavior.
  - TERM_SEED should match the Stage 1 checkpoint used to build the goal bank.
  - By default, action seeds 0/1/2 share TERM_SEED=0:
    MP1/data/outputs/<TASK>-<TERM_CONFIG>-0000_seed0/checkpoints/latest.ckpt
  - The goal bank is shared by task/K:
    MP1/data/goal_bank/<TASK>/k_<GOAL_K>/
EOF
}

if (( $# < 2 )); then
  echo "parameters are not enough!"
  show_usage
  exit 1
fi

GPU_ID=$1
CONFIG=$2
TASKS_ARG=${3:-}
SEEDS_ARG=${4:-}
TERM_CONFIG=${5:-mp_term}
GOAL_K=${6:-1}
TERM_SEED=${7:-0}

TASKS_DEFAULT=(
  metaworld_shelf-place
)

SEEDS_DEFAULT=(0 1 2)

if [[ -n "${TASKS_ARG}" ]]; then
  IFS=',' read -ra TASKS <<< "${TASKS_ARG}"
else
  TASKS=("${TASKS_DEFAULT[@]}")
fi

if [[ -n "${SEEDS_ARG}" ]]; then
  IFS=',' read -ra SEEDS <<< "${SEEDS_ARG}"
else
  SEEDS=("${SEEDS_DEFAULT[@]}")
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for seed in "${SEEDS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo ">>> GPU=${GPU_ID} | config=${CONFIG} | task=${task} | seed=${seed} | term=${TERM_CONFIG} seed=${TERM_SEED} | K=${GOAL_K}"
    bash "${ROOT_DIR}/run_goal.bash" "${GPU_ID}" "${CONFIG}" "${task}" "${seed}" "${TERM_CONFIG}" "${GOAL_K}" "${TERM_SEED}"
  done
done
