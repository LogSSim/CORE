#!/usr/bin/env bash
# -------------------------------------------------------------------------
# auto_3stage.bash
#
# Sequentially train stage1 -> stage2 -> stage3 for one of the six
# 3-stage pipelines (final_state / terminal_goal{,_v2}) on a single GPU.
#
# Usage:
#   bash auto_3stage.bash --method <NAME> --gpu <ID> [options]
#
# Methods (--method):
#   dp3_final     -> MP1/scripts/train_dp3_final_state_stage.sh
#   mp_final      -> MP1/scripts/train_mp_final_state_stage.sh
#   dp3_term      -> MP1/scripts/train_dp3_terminal_goal_stage.sh
#   mp_term       -> MP1/scripts/train_mp_terminal_goal_stage.sh
#   dp3_term_v2   -> MP1/scripts/train_dp3_terminal_goal_v2_stage.sh
#   mp_term_v2    -> MP1/scripts/train_mp_terminal_goal_v2_stage.sh
#
# Order: for each task, run all selected seeds for stage1, then a single
# stage2 (does not depend on seed), then all seeds for stage3.
#
# Examples:
#   bash auto_3stage.bash --method dp3_term_v2 --gpu 1 \
#       --tasks metaworld_shelf-place \
#       --seeds 0,1,2 --num-clusters 4 \
#       --terminal-window 2 --goal-selection common
#
#   nohup bash auto_3stage.bash --method mp_final --gpu 0 \
#       --tasks metaworld_shelf-place,metaworld_button-press \
#       --seeds 0,1,2 --num-clusters 4 \
#       > auto_3stage_mp_final.log 2>&1 &
# -------------------------------------------------------------------------

set -euo pipefail

show_usage() {
  cat <<'EOF'
Usage:
  bash auto_3stage.bash --method <NAME> --gpu <ID> [options]

Required:
  --method NAME            One of: dp3_final, mp_final, dp3_term, mp_term,
                           dp3_term_v2, mp_term_v2.
  --gpu, -g ID             GPU id (e.g. 0, 1).

Options (common):
  --tasks, -t LIST         Comma-separated tasks. Default: metaworld_shelf-place
  --seeds, -s LIST         Comma-separated seeds. Default: 0,1,2
  --stages LIST            Subset of 1,2,3 to run. Default: 1,2,3
  --num-clusters, -k K     KMeans cluster count. Default: 4
  --stage1-num N           Stage1/stage2 episode count. Default uses script default (20).
  --stage3-num N           Stage3 episode count. Default uses script default (10).

Options (terminal_goal only):
  --terminal-window N      Terminal window (frames at end of episode). Default: 2

Options (v2 only):
  --goal-selection MODE    fixed | common | soft_nearest. Default: common
  --goal-index I           Required when --goal-selection fixed.
  --soft-temperature T     Softmax temperature (for soft_nearest). Default uses script default.

Other:
  --dry-run                Print the underlying commands without executing.
  -h, --help               Show this help and exit.
EOF
}

if [[ $# -eq 0 ]]; then
  show_usage
  exit 1
fi

METHOD=""
GPU=""
TASKS_ARG=""
SEEDS_ARG=""
STAGES_ARG="1,2,3"
NUM_CLUSTERS="4"
TERMINAL_WINDOW="2"
GOAL_SELECTION="common"
GOAL_INDEX=""
SOFT_TEMP=""
STAGE1_NUM=""
STAGE3_NUM=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method)             METHOD="$2"; shift 2 ;;
    --gpu|-g)             GPU="$2"; shift 2 ;;
    --tasks|-t)           TASKS_ARG="$2"; shift 2 ;;
    --seeds|-s)           SEEDS_ARG="$2"; shift 2 ;;
    --stages)             STAGES_ARG="$2"; shift 2 ;;
    --num-clusters|-k)    NUM_CLUSTERS="$2"; shift 2 ;;
    --terminal-window)    TERMINAL_WINDOW="$2"; shift 2 ;;
    --goal-selection)     GOAL_SELECTION="$2"; shift 2 ;;
    --goal-index)         GOAL_INDEX="$2"; shift 2 ;;
    --soft-temperature|--soft-nearest-temperature)
                          SOFT_TEMP="$2"; shift 2 ;;
    --stage1-num)         STAGE1_NUM="$2"; shift 2 ;;
    --stage3-num)         STAGE3_NUM="$2"; shift 2 ;;
    --dry-run)            DRY_RUN=1; shift ;;
    -h|--help)            show_usage; exit 0 ;;
    *) echo "Unknown arg: $1"; show_usage; exit 1 ;;
  esac
done

if [[ -z "${METHOD}" || -z "${GPU}" ]]; then
  echo "❌ --method and --gpu are required."
  show_usage
  exit 1
fi

case "${METHOD}" in
  dp3_final)
    SCRIPT_REL="MP1/scripts/train_dp3_final_state_stage.sh"
    FAMILY="final"
    ;;
  mp_final)
    SCRIPT_REL="MP1/scripts/train_mp_final_state_stage.sh"
    FAMILY="final"
    ;;
  dp3_term)
    SCRIPT_REL="MP1/scripts/train_dp3_terminal_goal_stage.sh"
    FAMILY="term_v1"
    ;;
  mp_term)
    SCRIPT_REL="MP1/scripts/train_mp_terminal_goal_stage.sh"
    FAMILY="term_v1"
    ;;
  dp3_term_v2)
    SCRIPT_REL="MP1/scripts/train_dp3_terminal_goal_v2_stage.sh"
    FAMILY="term_v2"
    ;;
  mp_term_v2)
    SCRIPT_REL="MP1/scripts/train_mp_terminal_goal_v2_stage.sh"
    FAMILY="term_v2"
    ;;
  *) echo "Unknown --method: ${METHOD}"; show_usage; exit 1 ;;
esac

case "${FAMILY}" in
  term_v2)
    case "${GOAL_SELECTION}" in
      fixed|common|soft_nearest) ;;
      *) echo "--goal-selection must be fixed | common | soft_nearest"; exit 1 ;;
    esac
    if [[ "${GOAL_SELECTION}" == "fixed" && -z "${GOAL_INDEX}" ]]; then
      echo "❌ --goal-selection fixed requires --goal-index N."
      exit 1
    fi
    ;;
esac

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ABS="${ROOT_DIR}/${SCRIPT_REL}"
if [[ ! -f "${SCRIPT_ABS}" ]]; then
  echo "❌ underlying script not found: ${SCRIPT_ABS}"
  exit 1
fi

# --------- baked-in default task list (uncomment what you want to run) ----
# Edit this block directly when you want to run a fixed batch of tasks
# without passing --tasks on the CLI. Lines starting with '#' are skipped.
TASKS_DEFAULT=(
  # metaworld_disassemble
  # metaworld_stick-pull
  # metaworld_stick-push
  # metaworld_pick-place-wall

  # metaworld_shelf-place
  # metaworld_push
  # metaworld_pick-place
  # metaworld_hand-insert

  metaworld_assembly
  metaworld_sweep
  metaworld_soccer
  metaworld_push-wall

  # metaworld_peg-insert-side
  # metaworld_hammer
  # metaworld_coffee-push
  # metaworld_coffee-pull

  # metaworld_box-close
  # metaworld_bin-picking
  # metaworld_basketball
  # metaworld_peg-unplug-side

  # metaworld_window-open
  # metaworld_window-close
  # metaworld_reach-wall
  # metaworld_reach

  # metaworld_handle-pull-side
  # metaworld_lever-pull
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
  # metaworld_dial-turn
  # metaworld_door-close
)
SEEDS_DEFAULT=(0 1 2)
STAGES_DEFAULT=(1 2 3)

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
if [[ -n "${STAGES_ARG}" ]]; then
  IFS=',' read -ra STAGES <<< "${STAGES_ARG}"
else
  STAGES=("${STAGES_DEFAULT[@]}")
fi

stage_in_list() {
  local target="$1"
  for s in "${STAGES[@]}"; do
    if [[ "${s}" == "${target}" || "${s}" == "stage${target}" ]]; then
      return 0
    fi
  done
  return 1
}

# Build the option array shared by all stages of one underlying call.
build_opts() {
  local seed="$1"
  local opts=()
  opts+=(-g "${GPU}")
  if [[ -n "${seed}" ]]; then
    opts+=(--seed "${seed}")
  fi
  opts+=(--num-clusters "${NUM_CLUSTERS}")
  if [[ "${FAMILY}" == "term_v1" || "${FAMILY}" == "term_v2" ]]; then
    opts+=(--terminal-window "${TERMINAL_WINDOW}")
  fi
  if [[ "${FAMILY}" == "term_v2" ]]; then
    opts+=(--goal-selection "${GOAL_SELECTION}")
    if [[ -n "${GOAL_INDEX}" ]]; then
      opts+=(--goal-index "${GOAL_INDEX}")
    fi
    if [[ -n "${SOFT_TEMP}" ]]; then
      opts+=(--soft-temperature "${SOFT_TEMP}")
    fi
  fi
  if [[ -n "${STAGE1_NUM}" ]]; then
    opts+=(--stage1-num "${STAGE1_NUM}")
  fi
  if [[ -n "${STAGE3_NUM}" ]]; then
    opts+=(--stage3-num "${STAGE3_NUM}")
  fi
  printf '%s\0' "${opts[@]}"
}

# Run one stage through the underlying script. Honor each script's argument
# order convention (final_state: options first; terminal_goal: stage first).
run_one() {
  local stage="$1"
  local task="$2"
  local seed="${3:-}"
  local cmd=(bash "${SCRIPT_ABS}")
  local opts=()
  while IFS= read -r -d '' opt; do
    opts+=("${opt}")
  done < <(build_opts "${seed}")

  case "${FAMILY}" in
    final)
      cmd+=("${opts[@]}" -t "${task}" "stage${stage}")
      ;;
    term_v1|term_v2)
      cmd+=("stage${stage}" -t "${task}" "${opts[@]}")
      ;;
  esac
  echo
  echo ">>> [${METHOD}] task=${task} stage=${stage}${seed:+ seed=${seed}} K=${NUM_CLUSTERS}"
  printf '    %s ' "${cmd[@]}"; echo
  if (( DRY_RUN == 0 )); then
    "${cmd[@]}"
  fi
}

echo "============================================================"
echo " auto_3stage.bash"
echo "  method=${METHOD}  script=${SCRIPT_REL}"
echo "  gpu=${GPU}  K=${NUM_CLUSTERS}  stages=${STAGES_ARG}"
echo "  tasks=(${TASKS[*]})  seeds=(${SEEDS[*]})"
case "${FAMILY}" in
  term_v1) echo "  terminal_window=${TERMINAL_WINDOW}" ;;
  term_v2)
    echo "  terminal_window=${TERMINAL_WINDOW}"
    echo "  goal_selection=${GOAL_SELECTION} goal_index=${GOAL_INDEX:-<auto>} soft_temp=${SOFT_TEMP:-<default>}"
    ;;
esac
(( DRY_RUN == 1 )) && echo "  dry-run=1 (commands will be printed only)"
echo "============================================================"

for task in "${TASKS[@]}"; do
  if stage_in_list 1; then
    for seed in "${SEEDS[@]}"; do
      run_one 1 "${task}" "${seed}"
    done
  fi
  if stage_in_list 2; then
    # Stage2 only needs one stage1 ckpt; reuse the first seed for the auto-finder.
    run_one 2 "${task}" "${SEEDS[0]}"
  fi
  if stage_in_list 3; then
    for seed in "${SEEDS[@]}"; do
      run_one 3 "${task}" "${seed}"
    done
  fi
done

echo
echo "✅ auto_3stage.bash done."
