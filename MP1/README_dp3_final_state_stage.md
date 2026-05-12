# DP3 Final-State 三阶段训练

这套实验是独立于原 `dp3.yaml` 的 DP3 最终状态条件训练流程。入口脚本是：

```bash
MP1/scripts/train_dp3_final_state_stage.sh
```

核心配置是：

```bash
MP1/mp1/config/dp3_final_state_stage.yaml
```

## 三个阶段

`stage1`：直接读取 zarr 里的前 20 条 demo，按原始 DP3 训练，默认训练 600 epoch。这个阶段主要把点云视觉编码器训练好。

`stage2`：读取 stage1 的 checkpoint，对前 20 条 demo 每条取最后 1 帧点云，使用 stage1 的编码器提取最终状态特征，然后做 k-means 聚类，保存 `.npz` 特征文件。

`stage3`：读取 zarr 里的前 10 条 demo 训练 DP3，condition 变成：

```text
原 DP3 observation condition + 聚类最终状态特征 + 当前点云特征与最终特征之差
```

## 默认路径

训练输出默认保存到：

```bash
/data1/sjy/MP1/MP1/data/outputs/
```

stage2 特征默认保存到：

```bash
/data1/sjy/MP1/MP1/data/final_state_features/${TASK_NAME}_stage2_final_state.npz
```

例如 `metaworld_shelf-place`：

```bash
/data1/sjy/MP1/MP1/data/final_state_features/metaworld_shelf-place_stage2_final_state.npz
```

## Stage1

示例：用 GPU 1 训练 `metaworld_shelf-place`。

```bash
bash MP1/scripts/train_dp3_final_state_stage.sh \
  -g 1 \
  -t metaworld_shelf-place \
  stage1
```

训练完成后，checkpoint 通常在：

```bash
/data1/sjy/MP1/MP1/data/outputs/<date>/<time>_train_dp3_final_state_stage_shelf-place/checkpoints/latest.ckpt
```

也可以把 run 目录直接给 stage2/stage3，脚本会自动找：

```bash
<run_dir>/checkpoints/latest.ckpt
```

## Stage2

传 stage1 的 run 目录：

```bash
bash MP1/scripts/train_dp3_final_state_stage.sh \
  -g 1 \
  -t metaworld_shelf-place \
  stage2 \
  /data1/sjy/MP1/MP1/data/outputs/<date>/<run_dir>/
```

或者直接传 checkpoint：

```bash
bash MP1/scripts/train_dp3_final_state_stage.sh \
  -g 1 \
  -t metaworld_shelf-place \
  stage2 \
  /data1/sjy/MP1/MP1/data/outputs/<date>/<run_dir>/checkpoints/latest.ckpt
```

如果想改聚类数：

```bash
bash MP1/scripts/train_dp3_final_state_stage.sh \
  -g 1 \
  -t metaworld_shelf-place \
  --num-clusters 4 \
  stage2 \
  /data1/sjy/MP1/MP1/data/outputs/<date>/<run_dir>/
```

## Stage3

stage3 默认读取 stage2 生成的 `.npz` 特征文件，不再强制要求 stage1 checkpoint：

```bash
bash MP1/scripts/train_dp3_final_state_stage.sh \
  -g 1 \
  -t metaworld_shelf-place \
  stage3
```

也可以显式指定 stage2 得到的聚类特征文件：

```bash
bash MP1/scripts/train_dp3_final_state_stage.sh \
  -g 1 \
  -t metaworld_shelf-place \
  stage3 \
  /data1/sjy/MP1/MP1/data/final_state_features/metaworld_shelf-place_stage2_final_state.npz
```

如果想用 stage1 checkpoint 初始化 stage3 的视觉编码器，可以额外传 `--ckpt`：

```bash
bash MP1/scripts/train_dp3_final_state_stage.sh \
  -g 1 \
  -t metaworld_shelf-place \
  --ckpt /data1/sjy/MP1/MP1/data/outputs/<date>/<run_dir>/ \
  stage3
```

## 常用参数

```bash
-g, --gpu ID              GPU id，例如 0、1
-t, --task NAME           任务名，例如 adroit_hammer、metaworld_shelf-place
-c, --config NAME         Hydra config，默认 dp3_final_state_stage
--stage1-num N            stage1/stage2 使用 zarr 前 N 条 demo，默认 20
--stage3-num N            stage3 使用 zarr 前 N 条 demo，默认 10
--num-clusters N          stage2 聚类数，默认 4
--feature-path PATH       stage2 输出和 stage3 输入的 npz 路径
--ckpt PATH               stage2 必需；stage3 可选，用于初始化视觉编码器
```

## 注意事项

stage2 需要 stage1 的 checkpoint 来生成最终状态特征。可以传 `latest.ckpt`，也可以传 run 目录。

stage3 主要需要 stage2 生成的 `.npz` 特征文件。默认路径是：

```bash
/data1/sjy/MP1/MP1/data/final_state_features/${TASK_NAME}_stage2_final_state.npz
```

stage3 的 stage1 checkpoint 是可选的，只用于初始化当前点云 encoder。

不要使用示例里的 `<date>`、`<run_dir>` 字符串原样运行，需要替换成真实目录名。

如果 checkpoint 保存时报 `KeyError: test_mean_score`，确认配置中：

```yaml
training:
  rollout_every: 200
  checkpoint_every: 200
```

这两个值需要对齐，否则保存 top-k checkpoint 时可能没有评估指标。

如果 zarr 或 feature 路径异常，优先检查是否在项目目录：

```bash
/data1/sjy/MP1/MP1/data/
```
