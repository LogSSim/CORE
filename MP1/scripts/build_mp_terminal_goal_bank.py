#!/usr/bin/env python3
"""Build goal prototype banks for MeanpolicyTerminalGoal stage 2 (v1)."""

import argparse
import json
import sys
from pathlib import Path

import dill
import numpy as np
import torch
import torch.nn.functional as F
import zarr
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.cluster import KMeans


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def _register_resolvers():
    OmegaConf.register_new_resolver("eval", eval, replace=True)


def _resolve_path(path):
    path = Path(str(path)).expanduser()
    if path.is_absolute():
        return path
    candidates = [PROJECT_DIR / path, PROJECT_DIR.parent / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return PROJECT_DIR / path


def _resolve_checkpoint_path(path):
    path = _resolve_path(path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    candidates = [path / "checkpoints" / "latest.ckpt", path / "latest.ckpt"]
    ckpt_dir = path / "checkpoints"
    if ckpt_dir.is_dir():
        candidates.extend(
            sorted(
                ckpt_dir.glob("*.ckpt"),
                key=lambda p: (p.name == "latest.ckpt", p.stat().st_mtime),
                reverse=True,
            )
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No .ckpt found under {path}")


def _compose_cfg(config_name, task_name):
    _register_resolvers()
    config_dir = str((PROJECT_DIR / "mp1" / "config").resolve())
    overrides = []
    if task_name:
        overrides.append(f"task={task_name}")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name=config_name, overrides=overrides)


def _load_policy(args):
    _register_resolvers()
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)
    payload = torch.load(checkpoint_path.open("rb"), pickle_module=dill, map_location="cpu")
    cfg = payload.get("cfg")
    if cfg is None:
        cfg = _compose_cfg(args.config_name, args.task)

    policy = instantiate(cfg.policy)
    state_dicts = payload.get("state_dicts", {})
    state_key = "ema_model" if args.use_ema and "ema_model" in state_dicts else "model"
    if state_key not in state_dicts:
        raise KeyError(f"Checkpoint has no '{state_key}' state_dict. Available: {list(state_dicts.keys())}")
    missing, unexpected = policy.load_state_dict(state_dicts[state_key], strict=False)
    if not hasattr(policy, "term_module"):
        raise RuntimeError("Loaded policy has no term_module. Use mp_terminal_goal_stage stage1 checkpoint.")
    if missing:
        print(f"[build_mp_terminal_goal_bank] missing keys: {missing}")
    if unexpected:
        print(f"[build_mp_terminal_goal_bank] unexpected keys: {unexpected}")
    policy.to(args.device)
    policy.eval()
    return policy, cfg, checkpoint_path


def _dataset_path_from_args(args, cfg):
    if args.zarr_path:
        return _resolve_path(args.zarr_path)
    return _resolve_path(cfg.task.dataset.zarr_path)


def _collect_episode_embeddings(policy, zarr_path, terminal_window, device, max_episodes=None):
    root = zarr.open(str(zarr_path), mode="r")
    point_cloud = root["data"]["point_cloud"]
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    embeddings = []
    medoid_point_clouds = []
    episode_indices = []
    metadata = []

    prev_end = 0
    with torch.no_grad():
        for episode_idx, end_exclusive in enumerate(episode_ends):
            start = int(prev_end)
            end = int(end_exclusive) - 1
            prev_end = int(end_exclusive)
            terminal_start = max(start, end - int(terminal_window) + 1)
            terminal_pc = point_cloud[terminal_start:end + 1].astype(np.float32)
            pc = torch.from_numpy(terminal_pc).to(device)
            z = policy.encode_terminal_z(pc)
            episode_z = F.normalize(z.mean(dim=0), dim=0)
            embeddings.append(episode_z.detach().cpu().numpy().astype(np.float32))
            medoid_point_clouds.append(point_cloud[end].astype(np.float32))
            episode_indices.append(int(episode_idx))
            metadata.append(
                {
                    "episode_idx": int(episode_idx),
                    "episode_start": start,
                    "episode_end": end,
                    "terminal_start": int(terminal_start),
                    "terminal_window_size": int(end - terminal_start + 1),
                }
            )
            if max_episodes is not None and len(embeddings) >= int(max_episodes):
                break
    if len(embeddings) == 0:
        raise RuntimeError(f"No episodes found in {zarr_path}")
    return (
        np.stack(embeddings, axis=0),
        np.stack(medoid_point_clouds, axis=0),
        np.asarray(episode_indices, dtype=np.int64),
        metadata,
    )


def _save_banks(args, embeddings, medoid_point_clouds, episode_indices, metadata, checkpoint_path, zarr_path):
    output_root = _resolve_path(args.output_root)
    task_name = args.task or args.task_name or "unknown_task"
    output_root.mkdir(parents=True, exist_ok=True)
    for k in args.ks:
        if int(k) > embeddings.shape[0]:
            raise ValueError(f"K={k} is larger than episode count {embeddings.shape[0]}")
        kmeans = KMeans(n_clusters=int(k), random_state=args.seed, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        centers = kmeans.cluster_centers_.astype(np.float32)
        centers = centers / np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8, None)

        medoid_local_indices = []
        for cluster_idx in range(int(k)):
            candidates = np.where(labels == cluster_idx)[0]
            dist = np.linalg.norm(embeddings[candidates] - centers[cluster_idx][None, :], axis=1)
            medoid_local_indices.append(int(candidates[np.argmin(dist)]))
        medoid_local_indices = np.asarray(medoid_local_indices, dtype=np.int64)

        out_dir = output_root / task_name / f"k_{int(k)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "prototypes.npy", centers)
        np.save(out_dir / "medoid_indices.npy", episode_indices[medoid_local_indices])
        np.save(out_dir / "medoid_point_clouds.npy", medoid_point_clouds[medoid_local_indices])
        with (out_dir / "meta.json").open("w") as f:
            json.dump(
                {
                    "task_name": task_name,
                    "k": int(k),
                    "checkpoint": str(checkpoint_path),
                    "zarr_path": str(zarr_path),
                    "terminal_window": int(args.terminal_window),
                    "embedding_dim": int(embeddings.shape[1]),
                    "num_episodes": int(embeddings.shape[0]),
                    "medoid_episode_indices": episode_indices[medoid_local_indices].tolist(),
                    "episode_metadata": [metadata[int(i)] for i in medoid_local_indices],
                },
                f,
                indent=2,
            )
        print(f"[build_mp_terminal_goal_bank] saved K={int(k)} goal bank to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build MP terminal-goal v1 prototype bank.")
    parser.add_argument("--checkpoint", required=True, help="Stage1 checkpoint or run directory.")
    parser.add_argument("--zarr-path", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--config-name", default="mp_terminal_goal_stage")
    parser.add_argument("--ks", nargs="+", type=int, default=[4])
    parser.add_argument("--terminal-window", type=int, default=8)
    parser.add_argument("--output-root", default="/data1/sjy/MP1/MP1/data/goal_bank_mp")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episodes", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    policy, cfg, checkpoint_path = _load_policy(args)
    if args.task_name is None and args.task is None and "task" in cfg and "name" in cfg.task:
        args.task_name = str(cfg.task.name)
    zarr_path = _dataset_path_from_args(args, cfg)
    embeddings, medoid_point_clouds, episode_indices, metadata = _collect_episode_embeddings(
        policy=policy,
        zarr_path=zarr_path,
        terminal_window=args.terminal_window,
        device=args.device,
        max_episodes=args.max_episodes,
    )
    _save_banks(
        args=args,
        embeddings=embeddings,
        medoid_point_clouds=medoid_point_clouds,
        episode_indices=episode_indices,
        metadata=metadata,
        checkpoint_path=checkpoint_path,
        zarr_path=zarr_path,
    )


if __name__ == "__main__":
    main()
