#!/usr/bin/env bash
# -------------------------------------------------------------------------
# auto_3stage.bash
#
# Sequentially train CORE stage1 -> stage2 -> stage3 on one GPU.
#
# Retained implementations:
#   CORE_mp  -> CORE/scripts/train_CORE_mp_stage.sh
#   CORE_dp3 -> CORE/scripts/train_CORE_dp3_stage.sh
#
# Retained task:
#   PART=metaworld_box-close
#
# Usage:
#   bash auto_3stage.bash --method CORE_mp --gpu 0
#   bash auto_3stage.bash --method CORE_dp3 --gpu 0 --seeds 0,1,2
# -------------------------------------------------------------------------

set -euo pipefail

show_usage() {
  cat <<'EOF'
Usage:
  bash auto_3stage.bash --method <NAME> --gpu <ID> [options]

Required:
  --method NAME            One of: CORE_mp, CORE_dp3.
  --gpu, -g ID             GPU id (e.g. 0, 1).

Fixed task:
  PART=metaworld_box-close

Options:
  --seeds, -s LIST         Comma-separated seeds. Default: 0,1,2
  --stages LIST            Subset of 1,2,3 to run. Default: 1,2,3
  --num-clusters, -k K     Prototype folder index k_K. Default: 4
  --stage1-num N           Stage1/stage2 episode count. Default: 50
  --stage3-num N           Stage3 episode count. Default: 10
  --stage1-epochs N        Stage1 training epochs. Default: 1000
  --terminal-window N      Terminal window. Default: 1
  --goal-selection MODE    fixed | common | soft_nearest. Default: common
  --goal-index I           Required when --goal-selection fixed.
  --soft-temperature T     Softmax temperature for soft_nearest.
  --ckpt-select MODE       Stage1 ckpt for stage2/3: best | last/latest. Default: last.
  --dry-run                Print the underlying commands without executing.
  -h, --help               Show this help and exit.
EOF
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

if [[ $# -eq 0 ]]; then
  show_usage
  exit 1
fi

METHOD=""
GPU=""
SEEDS_ARG="0,1,2"
STAGES_ARG="1,2,3"
NUM_CLUSTERS="4"
TERMINAL_WINDOW="1"
GOAL_SELECTION="common"
GOAL_INDEX=""
SOFT_TEMP=""
CKPT_SELECT="last"
STAGE1_NUM="50"
STAGE3_NUM="10"
STAGE1_EPOCHS="1000"
DRY_RUN=0

PART="metaworld_box-close"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method)             METHOD="$2"; shift 2 ;;
    --gpu|-g)             GPU="$2"; shift 2 ;;
    --seeds|-s)           SEEDS_ARG="$2"; shift 2 ;;
    --stages)             STAGES_ARG="$2"; shift 2 ;;
    --num-clusters|-k)    NUM_CLUSTERS="$2"; shift 2 ;;
    --terminal-window)    TERMINAL_WINDOW="$2"; shift 2 ;;
    --goal-selection)     GOAL_SELECTION="$2"; shift 2 ;;
    --goal-index)         GOAL_INDEX="$2"; shift 2 ;;
    --soft-temperature|--soft-nearest-temperature)
                          SOFT_TEMP="$2"; shift 2 ;;
    --ckpt-select)        CKPT_SELECT="$2"; shift 2 ;;
    --stage1-num)         STAGE1_NUM="$2"; shift 2 ;;
    --stage3-num)         STAGE3_NUM="$2"; shift 2 ;;
    --stage1-epochs)      STAGE1_EPOCHS="$2"; shift 2 ;;
    --dry-run)            DRY_RUN=1; shift ;;
    --tasks|-t)
      echo "--tasks is not supported. This CORE launcher is fixed to PART=${PART}." >&2
      exit 1
      ;;
    -h|--help)            show_usage; exit 0 ;;
    *) echo "Unknown arg: $1"; show_usage; exit 1 ;;
  esac
done

if [[ -z "${METHOD}" || -z "${GPU}" ]]; then
  echo "--method and --gpu are required." >&2
  show_usage
  exit 1
fi

case "${METHOD}" in
  CORE_mp)
    SCRIPT_REL="CORE/scripts/train_CORE_mp_stage.sh"
    BASE_METHOD="CORE_mp"
    ;;
  CORE_dp3)
    SCRIPT_REL="CORE/scripts/train_CORE_dp3_stage.sh"
    BASE_METHOD="CORE_dp3"
    ;;
  *)
    echo "Unknown --method: ${METHOD}. Use CORE_mp or CORE_dp3." >&2
    exit 1
    ;;
esac

case "${GOAL_SELECTION}" in
  fixed|common|soft_nearest) ;;
  *) echo "--goal-selection must be fixed | common | soft_nearest" >&2; exit 1 ;;
esac
if [[ "${GOAL_SELECTION}" == "fixed" && -z "${GOAL_INDEX}" ]]; then
  echo "--goal-selection fixed requires --goal-index N." >&2
  exit 1
fi

case "${CKPT_SELECT}" in
  best|last|latest) ;;
  *) echo "--ckpt-select must be best | last | latest" >&2; exit 1 ;;
esac

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ABS="${ROOT_DIR}/${SCRIPT_REL}"
if [[ ! -f "${SCRIPT_ABS}" ]]; then
  echo "Underlying script not found: ${SCRIPT_ABS}" >&2
  exit 1
fi

TASKS=("${PART}")
SEEDS=()
STAGES=()

IFS=',' read -ra SEED_TOKENS <<< "${SEEDS_ARG}"
for seed in "${SEED_TOKENS[@]}"; do
  seed="$(trim "${seed}")"
  [[ -z "${seed}" ]] && continue
  SEEDS+=("${seed}")
done

IFS=',' read -ra STAGE_TOKENS <<< "${STAGES_ARG}"
for stage in "${STAGE_TOKENS[@]}"; do
  stage="$(trim "${stage}")"
  [[ -z "${stage}" ]] && continue
  STAGES+=("${stage}")
done

if [[ "${#SEEDS[@]}" -eq 0 ]]; then
  echo "--seeds produced an empty seed list." >&2
  exit 1
fi
if [[ "${#STAGES[@]}" -eq 0 ]]; then
  echo "--stages produced an empty stage list." >&2
  exit 1
fi

for stage in "${STAGES[@]}"; do
  case "${stage}" in
    1|2|3|stage1|stage2|stage3) ;;
    *) echo "--stages entries must be 1,2,3 or stage1,stage2,stage3. Got: ${stage}" >&2; exit 1 ;;
  esac
done

stage_in_list() {
  local target="$1"
  local stage
  for stage in "${STAGES[@]}"; do
    if [[ "${stage}" == "${target}" || "${stage}" == "stage${target}" ]]; then
      return 0
    fi
  done
  return 1
}

build_opts() {
  local seed="$1"
  local opts=()

  opts+=(-g "${GPU}")
  opts+=(--seed "${seed}")
  opts+=(--num-clusters "${NUM_CLUSTERS}")
  opts+=(--terminal-window "${TERMINAL_WINDOW}")
  opts+=(--goal-selection "${GOAL_SELECTION}")
  opts+=(--ckpt-select "${CKPT_SELECT}")

  if [[ -n "${GOAL_INDEX}" ]]; then
    opts+=(--goal-index "${GOAL_INDEX}")
  fi
  if [[ -n "${SOFT_TEMP}" ]]; then
    opts+=(--soft-temperature "${SOFT_TEMP}")
  fi
  if [[ -n "${STAGE1_NUM}" ]]; then
    opts+=(--stage1-num "${STAGE1_NUM}")
  fi
  if [[ -n "${STAGE3_NUM}" ]]; then
    opts+=(--stage3-num "${STAGE3_NUM}")
  fi
  if [[ -n "${STAGE1_EPOCHS}" ]]; then
    opts+=(--stage1-epochs "${STAGE1_EPOCHS}")
  fi

  printf '%s\0' "${opts[@]}"
}

run_one() {
  local stage="$1"
  local task="$2"
  local seed="$3"
  local cmd=(bash "${SCRIPT_ABS}")
  local opts=()
  local opt

  while IFS= read -r -d '' opt; do
    opts+=("${opt}")
  done < <(build_opts "${seed}")

  cmd+=("stage${stage}" -t "${task}" "${opts[@]}")

  echo
  echo ">>> [${METHOD}] base=${BASE_METHOD} task=${task} stage=${stage} seed=${seed} K=${NUM_CLUSTERS}"
  printf '    %s ' "${cmd[@]}"
  echo
  if (( DRY_RUN == 0 )); then
    "${cmd[@]}"
  fi
}

echo "============================================================"
echo " auto_3stage.bash"
echo "  method=${METHOD}  base=${BASE_METHOD}  script=${SCRIPT_REL}"
echo "  PART=${PART}"
echo "  gpu=${GPU}  K=${NUM_CLUSTERS}  stages=${STAGES[*]}"
echo "  seeds=(${SEEDS[*]})"
echo "  stage1_num=${STAGE1_NUM} stage3_num=${STAGE3_NUM} stage1_epochs=${STAGE1_EPOCHS}"
echo "  terminal_window=${TERMINAL_WINDOW}"
echo "  ckpt_select=${CKPT_SELECT}"
echo "  goal_selection=${GOAL_SELECTION} goal_index=${GOAL_INDEX:-<auto>} soft_temp=${SOFT_TEMP:-<default>}"
(( DRY_RUN == 1 )) && echo "  dry-run=1 (commands will be printed only)"
echo "============================================================"

for task in "${TASKS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    if stage_in_list 1; then
      run_one 1 "${task}" "${seed}"
    fi
    if stage_in_list 2; then
      run_one 2 "${task}" "${seed}"
    fi
    if stage_in_list 3; then
      run_one 3 "${task}" "${seed}"
    fi
  done
done

echo
echo "auto_3stage.bash done."
