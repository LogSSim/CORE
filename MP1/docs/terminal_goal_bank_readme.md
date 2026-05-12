# Terminal Representation + Goal Prototype Bank

This document records the current two-stage workflow for terminal representation learning and goal prototype bank export.

## Method Summary

The method adds an explicit terminal-state representation on top of the original action policy. The goal is to learn a compact latent space where successful terminal observations are close to each other, non-terminal observations are separated, and the latent direction from the current observation to a terminal prototype can be used as goal information for action prediction.

### Terminal State Source

Terminal states are obtained from the demonstration episodes in the replay buffer. For each episode:

```text
episode_start = first frame index
episode_end = last frame index
terminal_start = max(episode_start, episode_end - terminal_window + 1)
```

The terminal window is:

```text
[terminal_start, episode_end]
```

This means the method does not need a separate goal image at test time. It learns terminal prototypes offline from successful demonstration endings, then uses those prototypes during policy training and evaluation.

### Stage 1: Terminal Representation Learning

Stage 1 trains a terminal encoder:

```text
term_module(point_cloud) -> feat, z
```

where:

```text
feat: unnormalized feature used by ttg_head and term_head
z: normalized projection used for contrastive learning and goal bank export
```

The input is point cloud only. When `use_pc_color=false`, only xyz is used:

```text
point_cloud[..., :3]
```

The dataset additionally samples:

```text
term_anchor_point_cloud: terminal-window frame
term_pos_point_cloud: another terminal-window frame
neg_point_clouds: non-terminal frames from the same episode
repr_point_cloud: frame used for terminal/ttg supervision
ttg_target: normalized time-to-goal
term_label: whether repr frame is inside terminal window
```

The auxiliary terminal losses are:

```text
loss_nce: pulls two terminal-window frames together and pushes non-terminal frames away
loss_ttg: predicts normalized time-to-goal
loss_term: predicts whether a frame is terminal
```

The Stage 1 training objective is:

```text
loss = base_action_loss
     + lambda_nce * loss_nce
     + lambda_ttg * loss_ttg
     + lambda_term * loss_term
```

For the dispersive variant, an extra UNet hidden-state dispersive loss is added:

```text
loss = loss + lambda_dis * dis_loss
```

The terminal encoder is saved inside the Stage 1 checkpoint as `term_module`.

### Stage 2: Goal Bank Construction

After Stage 1, the trained `term_module` is used offline to encode terminal windows from successful episodes:

```text
terminal point clouds -> term_module -> terminal embeddings
```

For each episode, embeddings inside the terminal window are averaged to get one episode-level terminal embedding. Then KMeans is applied with different K values:

```text
K = 1, 2, 4
```

For each K, the script saves:

```text
prototypes.npy: latent cluster centers
medoid_indices.npy: closest demonstration episode indices
medoid_point_clouds.npy: real terminal point clouds closest to each prototype
meta.json: metadata
```

The prototype latent is the goal used by the second-stage policy:

```text
z_goal = prototypes[goal_index]
```

### Stage 3: Goal-Conditioned Action Policy

During policy training and evaluation, the method does not require access to the demonstration final frame. It uses:

```text
current observation point cloud -> frozen term_module -> z_curr
stored prototype from goal bank -> z_goal
```

The direct-concatenation version forms:

```text
goal_cond = concat([z_curr, z_goal, z_goal - z_curr])
global_cond = concat([original_global_cond, goal_cond])
```

Here:

```text
z_curr: current terminal-space representation
z_goal: learned terminal prototype
z_goal - z_curr: latent goal direction / residual
```

The action model then predicts actions conditioned on both the original observation features and this terminal-goal feature.

The dispersive goal version keeps the same terminal-goal conditioning, but uses the dispersive UNet and adds a hidden-feature dispersive loss. The dispersive loss is applied to UNet internal features, not to `z_curr` or `z_goal`.

### Why This Works

Stage 1 makes the terminal latent space meaningful:

```text
terminal frames cluster together
non-terminal frames separate from terminal frames
time-to-goal information is encoded
terminal/non-terminal status is encoded
```

Stage 2 converts this learned terminal space into a small set of reusable goal prototypes.

Stage 3 uses the latent difference:

```text
z_goal - z_curr
```

as a compact representation of how the current observation differs from the learned terminal state. This gives the action policy an explicit terminal-state target without requiring a final-frame image or point cloud at test time.

## Current Status

Implemented:

- `mp1/dataset/metaworld_terminal_dataset.py`
- `mp1/model/vision/terminal_point_encoder.py`
- `mp1/policy/meanpolicy_term.py`
- `mp1/policy/meanpolicy_dis_term.py`
- `mp1/policy/meanpolicy_goal.py`
- `mp1/policy/meanpolicy_dis_goal.py`
- `mp1/policy/meanpolicy_goal_mod.py`
- `mp1/policy/meanpolicy_dis_goal_mod.py`
- `mp1/config/mp_term.yaml`
- `mp1/config/mp_term_dis.yaml`
- `mp1/config/mp1_term.yaml`
- `mp1/config/mp_goal.yaml`
- `mp1/config/mp_goal_dis.yaml`
- `mp1/config/mp_goal_mod.yaml`
- `mp1/config/mp_goal_dis_mod.yaml`
- `scripts/build_goal_bank.py`
- `run_term.bash`
- `run_goal.bash`

Not implemented yet:

- Medoid point-cloud goal mode.

The first stage does not change policy inference. `MeanpolicyTerm.predict_action` still inherits the original MP implementation.

## Stage 1: Train Terminal Representation

Use `mp_term` first. This aligns with the original MP policy and does not use the dispersive loss.

From repo root `/data1/sjy/MP1`:

```bash
bash run_term.bash 3 mp_term metaworld_shelf-place 0
```

Background mode:

```bash
mkdir -p MP1/data/outputs/metaworld_shelf-place-mp_term-0000_seed0
nohup bash run_term.bash 3 mp_term metaworld_shelf-place 0 \
  > MP1/data/outputs/metaworld_shelf-place-mp_term-0000_seed0/train_nohup.log 2>&1 &
```

Check log:

```bash
tail -f MP1/data/outputs/metaworld_shelf-place-mp_term-0000_seed0/train_nohup.log
```

Output directory:

```text
MP1/data/outputs/metaworld_shelf-place-mp_term-0000_seed0/
```

Important files:

```text
checkpoints/latest.ckpt
checkpoints/epoch=XXXX-test_mean_score=YYY.ckpt
eval_results/latest.json
eval_results/topk_summary.json
.hydra/config.yaml
train_nohup.log
logs.json.txt
```

The checkpoint contains the original action policy plus the learned `term_module`.

Dispersive Stage 1 variant:

```bash
bash run_term.bash 3 mp_term_dis metaworld_shelf-place 0
```

`mp_term_dis` uses `MeanpolicyDisTerm`: meanflow action loss + UNet hidden-state dispersive loss + terminal auxiliary losses.

## Stage 1 Losses

The total training loss is:

```text
total_loss = base_loss + lambda_nce * loss_nce + lambda_ttg * loss_ttg + lambda_term * loss_term
```

Where:

- `base_loss`: original MP action loss
- `loss_nce`: InfoNCE terminal representation loss
- `loss_ttg`: time-to-goal prediction loss
- `loss_term`: terminal-window binary classification loss

The terminal branch uses point clouds only in v1. `agent_pos` is still used by the original MP observation encoder.

## Build Goal Bank

After Stage 1 checkpoint is available, export the goal bank.

From `/data1/sjy/MP1/MP1`:

```bash
python scripts/build_goal_bank.py \
  --checkpoint data/outputs/metaworld_shelf-place-mp_term-0000_seed0/checkpoints/latest.ckpt \
  --task metaworld_shelf-place \
  --ks 1 2 4 \
  --use-ema
```

Or from `/data1/sjy/MP1`:

```bash
cd MP1
python scripts/build_goal_bank.py \
  --checkpoint data/outputs/metaworld_shelf-place-mp_term-0000_seed0/checkpoints/latest.ckpt \
  --task metaworld_shelf-place \
  --ks 1 2 4 \
  --use-ema
```

Expected output:

```text
data/goal_bank/metaworld_shelf-place/k_1/
data/goal_bank/metaworld_shelf-place/k_2/
data/goal_bank/metaworld_shelf-place/k_4/
```

For the dispersive Stage 1 variant, use the matching checkpoint:

```bash
python scripts/build_goal_bank.py \
  --checkpoint data/outputs/metaworld_shelf-place-mp_term_dis-0000_seed0/checkpoints/latest.ckpt \
  --task metaworld_shelf-place \
  --ks 1 2 4 \
  --use-ema
```

Do not mix a goal bank exported from `mp_term` with a `term_module` loaded from `mp_term_dis`; the latent spaces are not guaranteed to align.

Each `k_*` directory contains:

```text
prototypes.npy
medoid_indices.npy
medoid_point_clouds.npy
meta.json
```

Meaning:

- `prototypes.npy`: latent terminal prototypes from KMeans
- `medoid_indices.npy`: episode indices closest to each prototype
- `medoid_point_clouds.npy`: real terminal point clouds closest to each prototype
- `meta.json`: task, checkpoint, dataset, and medoid metadata

## Verify Goal Bank

Quick check:

```bash
python - <<'PY'
import numpy as np
from pathlib import Path

root = Path("data/goal_bank/metaworld_shelf-place")
for k_dir in sorted(root.glob("k_*")):
    proto = np.load(k_dir / "prototypes.npy")
    medoid_pc = np.load(k_dir / "medoid_point_clouds.npy")
    medoid_idx = np.load(k_dir / "medoid_indices.npy")
    print(k_dir, "prototypes", proto.shape, "medoid_pc", medoid_pc.shape, "indices", medoid_idx.shape)
PY
```

Expected shapes are roughly:

```text
prototypes: [K, term_proj_dim]
medoid_point_clouds: [K, N, C]
medoid_indices: [K]
```

## Stage 2: Goal-Conditioned MP Policy

After the goal bank exists, train a new MP-style policy that appends the fixed terminal prototype to the original MP `global_cond`.

From repo root `/data1/sjy/MP1`:

```bash
bash run_goal.bash 3 mp_goal metaworld_shelf-place 0 mp_term 1 0
```

Background mode:

```bash
mkdir -p MP1/data/outputs/metaworld_shelf-place-mp_goal-0000_seed0
nohup bash run_goal.bash 3 mp_goal metaworld_shelf-place 0 mp_term 1 0 \
  > MP1/data/outputs/metaworld_shelf-place-mp_goal-0000_seed0/train_nohup.log 2>&1 &
```

The command expects:

```text
MP1/data/outputs/metaworld_shelf-place-mp_term-0000_seed0/checkpoints/latest.ckpt
MP1/data/goal_bank/metaworld_shelf-place/k_1/prototypes.npy
```

Current goal conditioning design:

```text
current point cloud -> term_module -> z_curr
prototype latent from goal bank -> z_goal
goal_cond = concat([z_curr, z_goal, z_goal - z_curr])
extended_global_cond = concat([original_global_cond, goal_cond])
```

For shelf-place with `term_proj_dim=128`:

```text
original_global_cond: [B, 2 * obs_feature_dim]
goal_cond: [B, 384]
extended_global_cond: [B, 2 * obs_feature_dim + 384]
```

Important:

- Do not modify `meanpolicy.py` or `meanpolicy_dis.py`.
- Stage 2 currently uses `MeanpolicyGoal`, which is MP-aligned and does not include dispersive loss.
- `term_module` is loaded from the Stage 1 checkpoint and frozen by default.
- `predict_action` uses the same fixed prototype at test time.
- Action output shape remains the same as the original MP policy.

Dispersive Stage 2 variant:

```bash
bash run_goal.bash 3 mp_goal_dis metaworld_shelf-place 0 mp_term_dis 1 0
```

This uses `MeanpolicyDisGoal`: terminal goal features are appended to `global_cond`, and the dispersive loss is applied to the UNet hidden features returned by `conditional_unet1d_meanflow_dis.py`.

Modulation Stage 2 variants keep the original MP `global_cond` dimension and use the goal feature to produce `gamma/beta`:

```text
goal_feat = concat([z_curr, z_goal, z_goal - z_curr])
global_cond = old_global_cond * (1 + 0.1 * gamma) + 0.1 * beta
```

No dispersive loss:

```bash
bash run_goal.bash 3 mp_goal_mod metaworld_shelf-place 0 mp_term 1 0
```

With dispersive loss:

```bash
bash run_goal.bash 3 mp_goal_dis_mod metaworld_shelf-place 0 mp_term_dis 1 0
```

K sweep with the modulated dispersive policy:

```bash
bash run_goal_k_sweep.bash 3 mp_goal_dis_mod metaworld_shelf-place 0 mp_term_dis 0
```

Run K sweep for `K=1,2,4` and save top-5 averages:

```bash
bash run_goal_k_sweep.bash 3 mp_goal_dis metaworld_shelf-place 0 mp_term_dis 0
```

The sweep writes separate runs:

```text
data/outputs/metaworld_shelf-place-mp_goal_dis-0000_k1_seed0/
data/outputs/metaworld_shelf-place-mp_goal_dis-0000_k2_seed0/
data/outputs/metaworld_shelf-place-mp_goal_dis-0000_k4_seed0/
```

And saves the aggregate summary to:

```text
data/outputs/metaworld_shelf-place-mp_goal_dis-0000_ksweep_seed0/top5_summary.json
```

## Run Three Seeds

For Stage 2 shelf-place with action seeds `0,1,2`, use the same Stage 1 seed0 `term_module` that produced the current goal bank:

```bash
bash auto_goal.bash 3 mp_goal metaworld_shelf-place 0,1,2 mp_term 1 0
```

Background mode:

```bash
nohup bash auto_goal.bash 3 mp_goal metaworld_shelf-place 0,1,2 mp_term 1 0 \
  > auto_goal_shelf-place.log 2>&1 &
```

The last `0` is `TERM_SEED`. Keep it aligned with the checkpoint used when exporting `data/goal_bank/metaworld_shelf-place/k_1`.

## Common Pitfalls

If you are already inside `/data1/sjy/MP1/MP1`, do not prefix checkpoint paths with `MP1/`.

Correct:

```bash
--checkpoint data/outputs/metaworld_shelf-place-mp_term-0000_seed0/checkpoints/latest.ckpt
```

Wrong from inside `/data1/sjy/MP1/MP1`:

```bash
--checkpoint MP1/data/outputs/metaworld_shelf-place-mp_term-0000_seed0/checkpoints/latest.ckpt
```

If CUDA OOM occurs, check for duplicate runs:

```bash
ps -eo pid,ppid,stat,comm,args | rg "mp_term.yaml.*metaworld_shelf-place|run_term.bash"
nvidia-smi
```

Kill a duplicate run if needed:

```bash
kill <PID>
```

## Direct Concatenation Summary

This is the original goal-bank pipeline before goal modulation. It directly appends terminal goal features to the policy `global_cond`.

### Stage 1: Train Terminal Encoder

Use one of:

```bash
bash run_term.bash 3 mp_term metaworld_shelf-place 0
bash run_term.bash 3 mp_term_dis metaworld_shelf-place 0
```

`mp_term` trains:

```text
base action loss
+ terminal InfoNCE loss
+ time-to-goal loss
+ terminal classification loss
```

`mp_term_dis` additionally uses the UNet hidden-state dispersive loss.

The terminal branch is:

```text
point_cloud -> TerminalPointEncoder -> feat, z
```

where `z` is the normalized terminal latent used later by the goal bank.

### Stage 2: Build Goal Bank

After Stage 1 finishes, export prototypes:

```bash
cd /data1/sjy/MP1/MP1
python scripts/build_goal_bank.py \
  --checkpoint data/outputs/metaworld_shelf-place-mp_term_dis-0000_seed0/checkpoints/latest.ckpt \
  --task metaworld_shelf-place \
  --ks 1 2 4 \
  --use-ema
```

This saves:

```text
data/goal_bank/metaworld_shelf-place/k_1/prototypes.npy
data/goal_bank/metaworld_shelf-place/k_2/prototypes.npy
data/goal_bank/metaworld_shelf-place/k_4/prototypes.npy
```

Important: the goal bank and second-stage `term_module` must come from the same Stage 1 checkpoint family.

### Stage 3: Direct-Concatenation Goal Policy

Use one of:

```bash
bash run_goal.bash 3 mp_goal metaworld_shelf-place 0 mp_term 1 0
bash run_goal.bash 3 mp_goal_dis metaworld_shelf-place 0 mp_term_dis 1 0
```

The direct-concat policy computes:

```text
old_global_cond = obs_encoder(obs_{t-1:t})
z_curr = frozen_term_module(current_point_cloud)
z_goal = selected prototype from goal bank
goal_cond = concat([z_curr, z_goal, z_goal - z_curr])
global_cond = concat([old_global_cond, goal_cond])
```

With `term_proj_dim=128`, the extra goal condition is:

```text
goal_cond: [B, 384]
```

So the UNet condition becomes:

```text
old_global_cond_dim + 384
```

`mp_goal_dis` keeps the same direct concatenation, but uses the dispersive UNet and adds:

```text
loss = action_loss + lambda_dis * dis_loss
```

The dispersive loss is applied to UNet hidden features, not to `z_curr` or `z_goal`.

### K Sweep

Run direct-concat K sweep:

```bash
bash run_goal_k_sweep.bash 3 mp_goal_dis metaworld_shelf-place 0 mp_term_dis 0
```

This runs `K=1,2,4` with `goal_index=0` and saves:

```text
data/outputs/metaworld_shelf-place-mp_goal_dis-0000_ksweep_seed0/top5_summary.json
```

Use direct concatenation when you want the goal feature to enter the action model as an explicit extra condition. Use modulation when you want a smoother, lower-risk injection that preserves the original `global_cond` dimension.
