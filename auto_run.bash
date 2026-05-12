#!/usr/bin/env bash
# -------------------------------------------------------------------------
# run_train.sh  GPU_ID CONFIG  [TASKS]  [SEEDS]  [RUN_TAG]
#
# - GPU_ID  : necessary
# - CONFIG  : necessary
# - TASKS   : option
# - SEEDS   : option 0,1,2
# - RUN_TAG : option, used in output dir name, default 0000
# -------------------------------------------------------------------------

set -euo pipefail      #

show_usage() {
  cat <<EOF
use:
  $0 <GPU_ID> <CONFIG> [TASKS] [SEEDS] [RUN_TAG]

example:
  $0 0 baseline metaworld_door-close 42
  $0 0 mp1_cm_dis metaworld_multitask_7 0 part1

  # explain:
  #   - run a task on GPU 0 
  #   - config file is baseline
  #   - random seed is 42
  #   - optional RUN_TAG avoids output directory collisions
EOF
}

# ----------- parameters checking-----------------------------------------------------
if (( $# < 2 )); then
  echo "❌ parameters are not enough!"
  show_usage
  exit 1
fi

GPU_ID=$1
CONFIG=$2
TASKS_ARG=${3:-}
SEEDS_ARG=${4:-}
RUN_TAG=${5:-${RUN_TAG:-0000}}

# ----------- default tasks and seeds ---------------------------------------------
TASKS_DEFAULT=(
  # metaworld_stick-pull
  # metaworld_stick-push
   # metaworld_basketball
   # metaworld_coffee-pull
  #  metaworld_push-wall
  # metaworld_dial-turn 
  # metaworld_lever-pull 
  # metaworld_reach-wall
)

TASKS_DEFAULT=(
  metaworld_multitask_part1
  metaworld_multitask_part2
  metaworld_multitask_part3
  metaworld_multitask_part4
)
SEEDS_DEFAULT=(0 1 2)
# SEEDS_DEFAULT=(1 2)
# ---------- tasks ------------------------------------------------------
if [[ -n $TASKS_ARG ]]; then
  IFS=',' read -ra TASKS <<< "$TASKS_ARG"   #
else
  TASKS=("${TASKS_DEFAULT[@]}")             #
fi

# ---------- seeds ------------------------------------------------------
if [[ -n $SEEDS_ARG ]]; then
  IFS=',' read -ra SEEDS <<< "$SEEDS_ARG"
else
  SEEDS=("${SEEDS_DEFAULT[@]}")             #
fi

# ----------- train loop -----------------------------------------------------
for seed in "${SEEDS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo "▶ GPU=$GPU_ID | config=$CONFIG | task=$task | seed=$seed | run_tag=$RUN_TAG"
    bash scripts/train_policy.sh "$CONFIG" "$task" "$RUN_TAG" "$seed" "$GPU_ID"
  done
done
