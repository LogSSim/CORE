"""Build stage-2 final-state features for DP3 stage-3 conditioning.

Example:
    python MP1/scripts/build_dp3_final_state_features.py \
        --checkpoint data/outputs/.../checkpoints/latest.ckpt \
        --output data/final_state_features/adroit_hammer_stage2.npz \
        --num-episodes 20
"""

import argparse
import os
import pathlib
import sys

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
ROOT_DIR = PROJECT_DIR.parents[1]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(PROJECT_DIR))
os.chdir(ROOT_DIR)

import dill
import hydra
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf
from termcolor import cprint


OmegaConf.register_new_resolver("eval", eval, replace=True)


def _resolve_data_path(path):
    if path is None or str(path).strip() == "":
        return None
    path = pathlib.Path(os.path.expanduser(str(path)))
    if path.is_absolute() and path.exists():
        return str(path)
    if path.exists():
        return str(path)
    candidate = PROJECT_DIR / path
    if candidate.exists():
        return str(candidate)
    candidate = pathlib.Path("MP1") / path
    if candidate.exists():
        return str(candidate)
    return str(path)


def _resolve_checkpoint_path(path):
    resolved = _resolve_data_path(path)
    if resolved is None:
        raise ValueError("checkpoint path is empty.")
    path = pathlib.Path(resolved)
    if path.is_dir():
        for candidate in (path / "checkpoints" / "latest.ckpt", path / "latest.ckpt"):
            if candidate.is_file():
                return str(candidate)
        raise FileNotFoundError(
            f"Checkpoint directory does not contain checkpoints/latest.ckpt or latest.ckpt: {path}"
        )
    return str(path)


def _resolve_output_path(path):
    path = pathlib.Path(os.path.expanduser(str(path)))
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def _non_empty(value):
    if value is None:
        return None
    value = str(value)
    if value.strip() == "":
        return None
    return value


def _task_zarr_path(task_name):
    task_cfg_path = PROJECT_DIR / "mp1" / "config" / "task" / f"{task_name}.yaml"
    if not task_cfg_path.is_file():
        raise FileNotFoundError(f"Task config not found: {task_cfg_path}")
    task_cfg = OmegaConf.load(task_cfg_path)
    return _non_empty(task_cfg.dataset.zarr_path)


def _episode_bounds(episode_ends, episode_idx):
    start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    end_exclusive = int(episode_ends[episode_idx])
    return start, end_exclusive


def _load_stage1_policy(checkpoint_path, device):
    payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload["state_dicts"]["model"], strict=True)
    policy.to(device)
    policy.eval()
    return policy, cfg


def _normalize_point_cloud(policy, point_cloud):
    normalizer = policy.normalizer["point_cloud"]
    point_cloud = normalizer.normalize(point_cloud)
    if not policy.use_pc_color:
        point_cloud = point_cloud[..., :3]
    return point_cloud


@torch.no_grad()
def _encode_final_point_clouds(policy, zarr_path, episode_indices, device, batch_size):
    root = zarr.open(zarr_path, mode="r")
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)

    final_point_clouds = []
    for episode_idx in episode_indices:
        start, end_exclusive = _episode_bounds(episode_ends, int(episode_idx))
        if end_exclusive <= start:
            raise RuntimeError(f"Episode {episode_idx} is empty.")
        final_point_clouds.append(root["data"]["point_cloud"][end_exclusive - 1].astype(np.float32))

    features = []
    for start in range(0, len(final_point_clouds), batch_size):
        pc_np = np.stack(final_point_clouds[start : start + batch_size], axis=0)
        pc = torch.from_numpy(pc_np).to(device=device)
        pc = _normalize_point_cloud(policy, pc)
        feat = policy.obs_encoder.extractor(pc)
        features.append(feat.detach().cpu())
    return torch.cat(features, dim=0)


def _kmeans(features, num_clusters, num_iters, seed):
    num_clusters = max(1, min(int(num_clusters), features.shape[0]))
    generator = torch.Generator(device=features.device)
    generator.manual_seed(int(seed))
    perm = torch.randperm(features.shape[0], generator=generator, device=features.device)
    centers = features[perm[:num_clusters]].clone()

    labels = torch.zeros(features.shape[0], dtype=torch.long, device=features.device)
    for _ in range(int(num_iters)):
        distances = torch.cdist(features, centers, p=2)
        labels = torch.argmin(distances, dim=1)
        next_centers = []
        for cluster_idx in range(num_clusters):
            mask = labels == cluster_idx
            if torch.any(mask):
                next_centers.append(features[mask].mean(dim=0))
            else:
                next_centers.append(centers[cluster_idx])
        centers = torch.stack(next_centers, dim=0)

    distances = torch.cdist(features, centers, p=2)
    labels = torch.argmin(distances, dim=1)
    return centers, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Stage-1 latest.ckpt path.")
    parser.add_argument("--output", required=True, help="Output .npz feature artifact.")
    parser.add_argument("--task", default=None, help="Override hydra task name from the checkpoint cfg.")
    parser.add_argument("--zarr-path", default=None, help="Override dataset zarr path.")
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Read the first N zarr episodes; default comes from final_state.stage1_num_train_episodes.",
    )
    parser.add_argument("--num-clusters", type=int, default=4)
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)
    policy, cfg = _load_stage1_policy(checkpoint_path, device)

    zarr_path = _non_empty(args.zarr_path)
    if zarr_path is None and args.task is not None:
        zarr_path = _task_zarr_path(args.task)
    if zarr_path is None:
        zarr_path = _non_empty(cfg.task.dataset.zarr_path)
    zarr_path = _resolve_data_path(zarr_path)
    if zarr_path is None or not pathlib.Path(zarr_path).exists():
        raise FileNotFoundError(
            "Could not resolve zarr_path. Pass --zarr-path explicitly, or check "
            f"task={args.task!r} and checkpoint cfg.task.dataset.zarr_path. "
            f"Resolved value: {zarr_path!r}"
        )

    num_episodes = args.num_episodes
    if num_episodes is None:
        num_episodes = int(getattr(cfg, "final_state", {}).get("stage1_num_train_episodes", 20))
    episode_indices = np.arange(int(num_episodes), dtype=np.int64)

    cprint(f"[stage2] checkpoint: {checkpoint_path}", "yellow")
    cprint(f"[stage2] zarr: {zarr_path}", "yellow")
    cprint(f"[stage2] episodes: first {num_episodes} zarr episodes", "yellow")

    features = _encode_final_point_clouds(
        policy=policy,
        zarr_path=zarr_path,
        episode_indices=episode_indices,
        device=device,
        batch_size=args.batch_size,
    )
    centers, labels = _kmeans(
        features=features.to(device),
        num_clusters=args.num_clusters,
        num_iters=args.num_iters,
        seed=args.seed,
    )
    centers = centers.detach().cpu()
    labels = labels.detach().cpu()
    cluster_features = centers[labels]

    output_path = _resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        episode_indices=episode_indices,
        final_features=features.numpy().astype(np.float32),
        cluster_centers=centers.numpy().astype(np.float32),
        cluster_labels=labels.numpy().astype(np.int64),
        cluster_features=cluster_features.numpy().astype(np.float32),
    )
    cprint(f"[stage2] saved {output_path}", "green")


if __name__ == "__main__":
    main()
