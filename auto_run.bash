#!/usr/bin/env bash
# -------------------------------------------------------------------------
# run_train.sh  GPU_ID CONFIG  [TASKS]  [SEEDS]
#
# - GPU_ID  : necessary
# - CONFIG  : necessary
# - TASKS   : option
# - SEEDS   : option 0,1,2
# -------------------------------------------------------------------------

set -euo pipefail      #

show_usage() {
  cat <<EOF
use:
  $0 <GPU_ID> <CONFIG> [TASKS] [SEEDS]

example:
  $0 0 baseline metaworld_door-close 42

  # explain:
  #   - run a task on GPU 0 
  #   - config file is baseline
  #   - random seed is 42
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
  
  metaworld_shelf-place
  metaworld_disassemble
  metaworld_pick-place-wall 
  metaworld_push
  metaworld_pick-place


  metaworld_hand-insert
  metaworld_assembly
  
  # metaworld_sweep
  # metaworld_soccer
  

  # metaworld_peg-insert-side
  # metaworld_hammer
  # metaworld_coffee-push
  # metaworld_box-close

  # metaworld_bin-picking
  # metaworld_peg-unplug-side
  # metaworld_window-open
  # metaworld_window-close
  
  # metaworld_reach
  # metaworld_handle-pull-side 
  # metaworld_plate-slide 
  # metaworld_plate-slide-back 
  
  # metaworld_plate-slide-back-side 
  # metaworld_plate-slide-side
  # metaworld_door-open
  # metaworld_door-unlock
  # metaworld_faucet-close

  # metaworld_faucet-open
  # metaworld_handle-press
  # metaworld_handle-pull
  # metaworld_button-press 
  # metaworld_button-press-wall 
  
  # metaworld_door-close 
  # adroit_hammer
  # adroit_door
  # adroit_pen

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
    echo "▶ GPU=$GPU_ID | config=$CONFIG | task=$task | seed=$seed"
    bash scripts/train_policy.sh "$CONFIG" "$task" 0000 "$seed" "$GPU_ID"
  done
done
