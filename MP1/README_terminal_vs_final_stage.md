# DP3 Final-State 与 Terminal-Goal 方法说明

这个文档说明当前项目里两类三阶段方法的区别：

- `final_state`：最终状态特征条件方法
- `terminal_goal`：终态表征学习 + goal prototype 条件方法

它们都是新建方法，不会影响原始 `dp3.yaml` 或其他已有 policy。

## 方法入口

Final-state 方法：

```bash
MP1/scripts/train_dp3_final_state_stage.sh
MP1/mp1/config/dp3_final_state_stage.yaml
MP1/mp1/policy/dp3_final_state_condition.py
```

Terminal-goal v2 方法，推荐使用这个版本：

```bash
MP1/scripts/train_dp3_terminal_goal_v2_stage.sh
MP1/mp1/config/dp3_terminal_goal_v2_stage.yaml
MP1/mp1/policy/dp3_terminal_goal_v2.py
```

旧版 terminal-goal 方法还保留在：

```bash
MP1/scripts/train_dp3_terminal_goal_stage.sh
MP1/mp1/config/dp3_terminal_goal_stage.yaml
MP1/mp1/policy/dp3_terminal_goal.py
```

旧版主要用于对照实验。因为旧版在 `K>1` 时默认使用 `prototype[0]`，容易让所有样本都条件到同一个 prototype，所以新实验建议用 v2。

## Final-State 是什么

Final-state 方法的核心思想是：直接把 demo 的最终状态点云编码成特征，并把这个特征作为 stage3 的额外条件。

三个阶段：

`stage1`：用 zarr 里的前 20 条 demo 训练普通 DP3，默认 600 epoch。这个阶段主要训练原 DP3 的点云视觉编码器。

`stage2`：读取 stage1 checkpoint，对前 20 条 demo 的每条 demo 取最后 1 帧点云，使用 stage1 的点云 encoder 编码，得到每条 demo 的 final feature。然后对 final feature 做 KMeans 聚类，保存 `.npz`。

`stage3`：用 zarr 里的前 10 条 demo 训练 DP3，condition 变成：

```text
原 DP3 observation condition
+ cluster_feature
+ 当前点云特征 - final_feature
```

默认输出：

```bash
/data1/sjy/MP1/MP1/data/final_state_features/${TASK_NAME}_stage2_final_state.npz
```

Final-state 更像是“显式使用最终帧特征”的方法。它使用的是每条 demo 的最终帧信息，因此训练阶段和推理阶段需要注意条件来源是否一致。

## Terminal-Goal 是什么

Terminal-goal 方法参考 RoboTwin 的 DP3 思路，不是直接把最终帧特征拼进去，而是先学习一个 terminal encoder，让它学会表示“终态区域”。

三个阶段：

`stage1`：训练 DP3，同时训练一个 `terminal_encoder`。除了 DP3 diffusion loss，还会加 terminal auxiliary losses：

- `InfoNCE`：让同一 episode 末尾窗口中的终态帧 embedding 更接近，让非终态帧远离。
- `TTG regression`：预测 time-to-go，帮助 encoder 理解离终点还有多远。
- `terminal classification`：判断当前帧是否属于终态窗口。

`stage2`：读取 stage1 checkpoint，用 `terminal_encoder` 对每条 demo 的最后 `terminal_window` 帧编码。默认 `terminal_window=8`，可以设成 2。每条 demo 的终态 embedding 是这几帧 embedding 的均值。然后做 KMeans，保存 goal bank。

v2 会保存：

```bash
prototypes.npy
common_prototype.npy
common_prototype_from_all.npy
cluster_sizes.npy
meta.json
```

默认路径：

```bash
/data1/sjy/MP1/MP1/data/goal_bank_v2/${TASK_NAME}/k_${K}/
```

`stage3`：训练 DP3 时不读取未来真实 final label，而是用当前观测点云编码得到 `z_curr`，再根据配置得到 `z_goal`，condition 变成：

```text
原 DP3 observation condition
+ z_curr
+ z_goal
+ z_goal - z_curr
```

Terminal-goal 更像是“学习一个终态语义空间”，然后用 prototype 或 common goal 作为目标条件。

## Terminal v2 的 Goal 选择

v2 新增了：

```yaml
goal_conditioning:
  selection: common
```

支持三种模式。

`common`：推荐默认模式。使用 `common_prototype.npy` 作为共有终态特征。训练和推理都使用同一个 common goal，不需要未来 demo 信息。

`soft_nearest`：根据当前 `z_curr` 到所有 `prototypes.npy` 的距离做 softmax 加权，得到：

```text
z_goal = weighted_sum(prototypes)
```

这个模式不会固定使用某一个 prototype，适合 `K>1` 的情况。

`fixed`：固定使用某个 prototype。必须显式指定 `--goal-index`，否则直接报错，避免静默使用 `prototype[0]`。

## Final 与 Terminal 的核心区别

| 项目 | Final-State | Terminal-Goal v2 |
| --- | --- | --- |
| stage1 训练目标 | 普通 DP3 | DP3 + terminal auxiliary losses |
| stage2 取点云 | 每条 demo 最后 1 帧 | 每条 demo 最后 `terminal_window` 帧 |
| stage2 输出 | `.npz` final/cluster features | `prototypes.npy` 和 `common_prototype.npy` |
| stage3 条件 | `obs_cond + cluster_feature + current-final delta` | `obs_cond + z_curr + z_goal + z_goal-z_curr` |
| 是否训练 terminal encoder | 否 | 是 |
| 是否推荐 K>1 | 可以，但依赖 final feature 聚类 | 推荐用 `common` 或 `soft_nearest` |
| train-test mismatch 风险 | 如果训练用 per-demo final、推理用均值，可能不一致 | v2 的 `common/soft_nearest` 训练推理逻辑一致 |

简单理解：

- Final-state：直接使用“最终帧是什么样”的特征。
- Terminal-goal：先学习“什么是终态”的 embedding 空间，再用 prototype/common goal 指导策略。

## 训练 Final-State

Stage1，默认前 20 条 demo，600 epoch：

```bash
bash MP1/scripts/train_dp3_final_state_stage.sh \
  -g 1 \
  -t metaworld_shelf-place \
  --seed 0 \
  stage1
```

Stage2，默认自动找同 task、同 seed 的 stage1 checkpoint：

```bash
bash MP1/scripts/train_dp3_final_state_stage.sh \
  -g 1 \
  -t metaworld_shelf-place \
  --seed 0 \
  stage2
```

Stage3，默认前 10 条 demo，3000 epoch：

```bash
bash MP1/scripts/train_dp3_final_state_stage.sh \
  -g 1 \
  -t metaworld_shelf-place \
  --seed 0 \
  stage3
```

如果要手动指定 stage2 特征：

```bash
bash MP1/scripts/train_dp3_final_state_stage.sh \
  -g 1 \
  -t metaworld_shelf-place \
  --seed 0 \
  stage3 \
  /data1/sjy/MP1/MP1/data/final_state_features/metaworld_shelf-place_stage2_final_state.npz
```

## 训练 Terminal-Goal v2

Stage1，默认前 20 条 demo，600 epoch：

```bash
bash MP1/scripts/train_dp3_terminal_goal_v2_stage.sh \
  stage1 \
  -g 1 \
  -t metaworld_shelf-place \
  --seed 0 \
  --terminal-window 2
```

Stage2，构建 goal bank：

```bash
bash MP1/scripts/train_dp3_terminal_goal_v2_stage.sh \
  stage2 \
  -g 1 \
  -t metaworld_shelf-place \
  --seed 0 \
  --terminal-window 2 \
  --num-clusters 4
```

Stage3，推荐使用 `common`：

```bash
bash MP1/scripts/train_dp3_terminal_goal_v2_stage.sh \
  stage3 \
  -g 1 \
  -t metaworld_shelf-place \
  --seed 0 \
  --terminal-window 2 \
  --num-clusters 4 \
  --goal-selection common
```

使用 `soft_nearest`：

```bash
bash MP1/scripts/train_dp3_terminal_goal_v2_stage.sh \
  stage3 \
  -g 1 \
  -t metaworld_shelf-place \
  --seed 0 \
  --terminal-window 2 \
  --num-clusters 4 \
  --goal-selection soft_nearest
```

使用固定 prototype：

```bash
bash MP1/scripts/train_dp3_terminal_goal_v2_stage.sh \
  stage3 \
  -g 1 \
  -t metaworld_shelf-place \
  --seed 0 \
  --terminal-window 2 \
  --num-clusters 4 \
  --goal-selection fixed \
  --goal-index 1
```

`fixed` 必须传 `--goal-index`，否则会报错。

## 常用参数

```bash
-g, --gpu ID                    GPU id
-t, --task NAME                 任务名，例如 metaworld_shelf-place
--seed N                        随机种子
--stage1-num N                  stage1/stage2 使用前 N 条 demo，默认 20
--stage3-num N                  stage3 使用前 N 条 demo，默认 10
--terminal-window N             terminal 方法 stage2 取最后 N 帧，默认 8
--num-clusters K                KMeans 聚类数，默认 4
--goal-selection MODE           common、soft_nearest、fixed
--goal-index N                  fixed 模式下必须提供
--ckpt PATH                     手动指定 stage1 checkpoint 或 run dir
```

## 建议怎么选

如果只是验证“最终状态特征能不能作为条件帮助 DP3”，先跑 `final_state`。

如果想更接近 RoboTwin，并且希望终态表示是通过辅助任务学出来的，跑 `terminal_goal_v2`。

如果 `K=1`，`common` 和固定单个 prototype 接近。

如果 `K>1`，建议优先：

```bash
--goal-selection common
```

或者：

```bash
--goal-selection soft_nearest
```

不建议再用旧版 `train_dp3_terminal_goal_stage.sh` 做主要实验，因为旧版默认 `goal_index=0`，在 `K>1` 时容易把所有样本都条件到第 0 个 prototype。

## 输出位置

训练输出：

```bash
/data1/sjy/MP1/MP1/data/outputs/
```

Final-state 特征：

```bash
/data1/sjy/MP1/MP1/data/final_state_features/
```

Terminal-goal v2 goal bank：

```bash
/data1/sjy/MP1/MP1/data/goal_bank_v2/
```

## 注意事项

脚本里 `--seed` 要放在 stage 后也可以用于 terminal v2；但 `final_state` 旧脚本是 `[options] stage` 格式，建议统一写成：

```bash
bash MP1/scripts/train_dp3_final_state_stage.sh -g 1 -t metaworld_shelf-place --seed 0 stage1
```

terminal v2 脚本是：

```bash
bash MP1/scripts/train_dp3_terminal_goal_v2_stage.sh stage1 -g 1 -t metaworld_shelf-place --seed 0
```

两个脚本参数顺序不完全一样，按上面的示例写最稳。
