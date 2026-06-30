# CORE

## 📄 Paper

**CORE: Common Outcome Regularities from Action-Free Visual Demonstrations for Robot Manipulation**

Paper: [arXiv:2606.29517](https://arxiv.org/abs/2606.29517)

## 🧾 Abstract

Robot imitation learning often relies on costly robot demonstrations, while
abundant action-free visual demonstrations, such as human videos, are difficult
to use because they lack robot-executable actions and suffer from embodiment
gaps. We propose CORE, a policy learning framework that extracts Common Outcome
Regularities from visual demonstrations. Rather than transferring explicit
actions across embodiments, CORE exploits a key observation: although successful
trajectories for the same task can be diverse,
their terminal states often share stable object configurations, spatial
relations, and contact constraints. CORE first trains a terminal outcome encoder
with contrastive and auxiliary temporal objectives, then aggregates successful
terminal embeddings into visual goal prototypes, and finally injects these
prototypes as global goal conditions into robot policies. Compared with
language instructions, visual goal prototypes provide more concrete geometric
and physical constraints for task completion. Across Meta-World, RoboTwin 2.0,
and real-world manipulation, CORE improves the average success rate of the
corresponding policy backbones by up to +3.9, +11.1, and +17.0 percentage
points, respectively, and outperforms text-conditioned variants under the
evaluated settings.

## 🧭 Scope

This repository is a cleaned CORE training branch. It keeps only the
three-stage CORE pipeline for:

```bash
PART="metaworld_box-close"
```

Two variants are available:

- `CORE_mp`: CORE with the MP policy backbone.
- `CORE_dp3`: CORE with the DP3 policy backbone.

## 🧩 Pipeline

Both variants follow the same three-stage CORE pipeline:

1. Stage 1 trains a terminal outcome encoder with bidirectional contrastive
   learning and auxiliary temporal prediction losses.
2. Stage 2 encodes successful terminal frames and aggregates them into a shared
   visual goal prototype.
3. Stage 3 reloads the terminal encoder and shared prototype, then trains the
   policy with CORE goal conditioning.

## ⚙️ Installation

Follow [install.md](install.md) to set up the Python environment and simulation
dependencies.

The original development environment used Python 3.8, CUDA 11.8, PyTorch 2.2.1,
and MuJoCo 2.1.0.

## 📦 Data

Generate the fixed Meta-World task data:

```bash
bash scripts/gen_demonstration_metaworld.sh box-close
```

The expected dataset path is:

```text
CORE/data/metaworld_box-close_expert.zarr
```

## 🚀 Training

Run CORE with the MP backbone:

```bash
bash auto_3stage.bash --method CORE_mp --gpu 0
```

Run CORE with the DP3 backbone:

```bash
bash auto_3stage.bash --method CORE_dp3 --gpu 0
```

Run one seed:

```bash
bash auto_3stage.bash --method CORE_mp --gpu 0 --seeds 0
```

Dry run:

```bash
bash auto_3stage.bash --method CORE_mp --gpu 0 --seeds 0 --stages 1 --dry-run
```

## 🗂️ Main Files

```text
auto_3stage.bash
CORE/scripts/train_CORE_mp_stage.sh
CORE/scripts/train_CORE_dp3_stage.sh
CORE/scripts/build_CORE_mp_bank.py
CORE/scripts/build_CORE_dp3_bank.py
CORE/core/config/CORE_mp.yaml
CORE/core/config/CORE_dp3.yaml
CORE/core/policy/core_mp_policy.py
CORE/core/policy/core_dp3_policy.py
CORE/core/dataset/core_terminal_dataset.py
```

## 📁 Outputs

Training outputs:

```text
CORE/data/outputs/
```

Goal banks:

```text
CORE/data/goal_bank_CORE_mp/
CORE/data/goal_bank_CORE_dp3/
```

## ✅ Verification

```bash
python -m py_compile \
  CORE/scripts/build_CORE_mp_bank.py \
  CORE/scripts/build_CORE_dp3_bank.py \
  CORE/core/policy/core_auxiliary.py \
  CORE/core/policy/core_mp_policy.py \
  CORE/core/policy/core_dp3_policy.py \
  CORE/core/dataset/core_terminal_dataset.py
```

On Linux:

```bash
bash -n auto_3stage.bash
bash auto_3stage.bash --method CORE_mp --gpu 0 --seeds 0 --stages 1 --dry-run
```

## 🕶️ Anonymous Review

This repository is prepared for anonymous review. Author names, affiliations,
and identifying project paths are intentionally omitted.
