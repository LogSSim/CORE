#!/usr/bin/env bash
# -------------------------------------------------------------------------
# run_goal_k_sweep.bash GPU_ID CONFIG TASK SEED TERM_CONFIG TERM_SEED [KS] [GOAL_INDEX] [TOP_N]
#
# Sequentially run goal-conditioned training for K in 1,2,4 and save top-N
# evaluation averages for each K.
#
# Examples:
#   bash run_goal_k_sweep.bash 3 mp_goal_dis metaworld_shelf-place 0 mp_term_dis 0
#   nohup bash run_goal_k_sweep.bash 3 mp_goal_dis metaworld_shelf-place 0 mp_term_dis 0 \
#     > goal_k_sweep_shelf-place.log 2>&1 &
# -------------------------------------------------------------------------

set -euo pipefail

show_usage() {
  cat <<EOF
use:
  $0 <GPU_ID> <CONFIG> <TASK> <SEED> <TERM_CONFIG> <TERM_SEED> [KS] [GOAL_INDEX] [TOP_N]

examples:
  $0 3 mp_goal_dis metaworld_shelf-place 0 mp_term_dis 0
  $0 3 mp_goal metaworld_shelf-place 0 mp_term 0 1,2,4 0 5

defaults:
  KS=1,2,4
  GOAL_INDEX=0
  TOP_N=5

notes:
  - Each K writes to a separate output dir:
    MP1/data/outputs/<TASK>-<CONFIG>-0000_k<K>_seed<SEED>/
  - TERM_CONFIG/TERM_SEED should match the Stage 1 checkpoint used to build
    data/goal_bank/<TASK>/k_<K>/.
EOF
}

if (( $# < 6 )); then
  echo "parameters are not enough!"
  show_usage
  exit 1
fi

GPU_ID=$1
CONFIG=$2
TASK=$3
SEED=$4
TERM_CONFIG=$5
TERM_SEED=$6
KS_ARG=${7:-1,2,4}
GOAL_INDEX=${8:-0}
TOP_N=${9:-5}

BASE_ADDITION_INFO=${ADDITION_INFO:-0000}
TERM_ADDITION_INFO=${TERM_ADDITION_INFO:-0000}
PYTHON_BIN=${PYTHON_BIN:-/home/dbcloud/anaconda3/envs/MP1/bin/python}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${ROOT_DIR}/MP1"
SUMMARY_DIR="${PROJECT_DIR}/data/outputs/${TASK}-${CONFIG}-${BASE_ADDITION_INFO}_ksweep_seed${SEED}"
SUMMARY_PATH="${SUMMARY_DIR}/top${TOP_N}_summary.json"

IFS=',' read -ra KS <<< "${KS_ARG}"
mkdir -p "${SUMMARY_DIR}"

summarize_k() {
  local k="$1"
  local run_dir="$2"

  "${PYTHON_BIN}" - "$k" "$run_dir" "$SUMMARY_PATH" "$TOP_N" <<'PY'
import json
import sys
from pathlib import Path

k = str(sys.argv[1])
run_dir = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
top_n = int(sys.argv[4])
eval_dir = run_dir / "eval_results"

records = []
for path in sorted(eval_dir.glob("epoch_*.json")):
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    metrics = payload.get("metrics", {})
    if "test_mean_score" not in metrics:
        continue
    records.append({
        "epoch": int(payload.get("epoch", -1)),
        "global_step": int(payload.get("global_step", -1)),
        "test_mean_score": float(metrics["test_mean_score"]),
        "path": str(path),
    })

if not records:
    raise RuntimeError(f"No epoch_*.json with test_mean_score found in {eval_dir}")

top = sorted(records, key=lambda x: x["test_mean_score"], reverse=True)[:top_n]
top_mean = sum(item["test_mean_score"] for item in top) / len(top)
payload = {
    "k": int(k),
    "run_dir": str(run_dir),
    "top_n": top_n,
    "num_eval_points": len(records),
    "top_mean_score": top_mean,
    "top_mean_score_percent": 100.0 * top_mean,
    "top_items": top,
    "all_items": records,
}

per_run_path = eval_dir / f"top{top_n}_summary.json"
per_run_path.parent.mkdir(parents=True, exist_ok=True)
with per_run_path.open("w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)

if summary_path.exists():
    with summary_path.open("r", encoding="utf-8") as f:
        sweep = json.load(f)
else:
    sweep = {"items": []}

sweep["items"] = [item for item in sweep.get("items", []) if int(item.get("k", -1)) != int(k)]
sweep["items"].append(payload)
sweep["items"] = sorted(sweep["items"], key=lambda x: int(x["k"]))
sweep["best_by_top_mean"] = max(sweep["items"], key=lambda x: x["top_mean_score"])

summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8") as f:
    json.dump(sweep, f, indent=2, ensure_ascii=False)

print(f"[k_sweep] K={k} top{top_n}_mean={top_mean:.6f} saved {per_run_path}")
print(f"[k_sweep] sweep summary saved {summary_path}")
PY
}

for k in "${KS[@]}"; do
  RUN_ADDITION_INFO="${BASE_ADDITION_INFO}_k${k}"
  RUN_DIR="${PROJECT_DIR}/data/outputs/${TASK}-${CONFIG}-${RUN_ADDITION_INFO}_seed${SEED}"

  echo ">>> K=${k} | GPU=${GPU_ID} | config=${CONFIG} | task=${TASK} | seed=${SEED} | term=${TERM_CONFIG}/seed${TERM_SEED}"

  RUN_ADDITION_INFO="${RUN_ADDITION_INFO}" \
  TERM_ADDITION_INFO="${TERM_ADDITION_INFO}" \
  bash "${ROOT_DIR}/run_goal.bash" \
    "${GPU_ID}" "${CONFIG}" "${TASK}" "${SEED}" "${TERM_CONFIG}" "${k}" "${TERM_SEED}" \
    "policy.goal_index=${GOAL_INDEX}" \
    "checkpoint.topk.k=${TOP_N}"

  summarize_k "${k}" "${RUN_DIR}"
done

echo "All K sweep results saved to ${SUMMARY_PATH}"
