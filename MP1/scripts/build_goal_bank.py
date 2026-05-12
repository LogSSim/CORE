#!/usr/bin/env python3
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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolve_path(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _compose_cfg(config_name, task_name):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    config_dir = str((REPO_ROOT / "mp1" / "config").resolve())
    overrides = []
    if task_name:
        overrides.append(f"task={task_name}")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name=config_name, overrides=overrides)


def _load_checkpoint_policy(args):
    checkpoint_path = _resolve_path(args.checkpoint)
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
        raise RuntimeError(
            "Loaded policy has no term_module. Use a checkpoint trained with mp_term.yaml or mp1_term.yaml."
        )
    if missing:
        print(f"[build_goal_bank] Missing keys while loading policy: {missing}")
    if unexpected:
        print(f"[build_goal_bank] Unexpected keys while loading policy: {unexpected}")

    policy.to(args.device)
    policy.eval()
    return policy, cfg, checkpoint_path


def _dataset_path_from_args(args, cfg):
    if args.zarr_path is not None:
        return _resolve_path(args.zarr_path)
    zarr_path = cfg.task.dataset.zarr_path
    return _resolve_path(zarr_path)


def _episode_success_mask(root, episode_ends):
    n_episodes = len(episode_ends)
    total_steps = int(episode_ends[-1]) if n_episodes > 0 else 0
    candidates = ("success", "is_success", "episode_success", "successes")

    for key in candidates:
        arr = None
        if "meta" in root and key in root["meta"]:
            arr = np.asarray(root["meta"][key])
        elif "data" in root and key in root["data"]:
            arr = np.asarray(root["data"][key])
        if arr is None:
            continue

        flat = arr.reshape(-1)
        if flat.shape[0] == n_episodes:
            return flat.astype(bool)
        if flat.shape[0] == total_steps:
            return np.asarray([flat[int(end) - 1] for end in episode_ends], dtype=bool)

    return np.ones((n_episodes,), dtype=bool)


def _normalize_point_cloud(policy, point_cloud, device):
    pc = torch.from_numpy(point_cloud.astype(np.float32)).to(device)
    pc = policy.normalizer["point_cloud"].normalize(pc)
    if not policy.use_pc_color:
        pc = pc[..., :3]
    return pc


def _collect_episode_embeddings(policy, zarr_path, terminal_window, device, max_episodes=None):
    root = zarr.open(str(zarr_path), mode="r")
    point_cloud = root["data"]["point_cloud"]
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    success_mask = _episode_success_mask(root, episode_ends)

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
            if not bool(success_mask[episode_idx]):
                continue

            terminal_start = max(start, end - terminal_window + 1)
            terminal_pc = point_cloud[terminal_start:end + 1].astype(np.float32)
            n_terminal_pc = _normalize_point_cloud(policy, terminal_pc, device)
            _, z = policy.term_module.encode_proj(n_terminal_pc)

            # [W, D] terminal-window embeddings -> one episode-level prototype candidate.
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

            if max_episodes is not None and len(embeddings) >= max_episodes:
                break

    if len(embeddings) == 0:
        raise RuntimeError(f"No successful episodes found in {zarr_path}")

    return (
        np.stack(embeddings, axis=0),
        np.stack(medoid_point_clouds, axis=0),
        np.asarray(episode_indices, dtype=np.int64),
        metadata,
    )


def _save_goal_bank(args, embeddings, episode_point_clouds, episode_indices, metadata, checkpoint_path, zarr_path):
    output_root = _resolve_path(args.output_root)
    task_name = args.task or args.task_name
    if task_name is None:
        task_name = "unknown_task"

    for k in args.ks:
        if k > embeddings.shape[0]:
            raise ValueError(f"K={k} is larger than collected episode count {embeddings.shape[0]}")

        kmeans = KMeans(n_clusters=k, random_state=args.seed, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        centers = kmeans.cluster_centers_.astype(np.float32)
        norm = np.linalg.norm(centers, axis=1, keepdims=True)
        prototypes = centers / np.maximum(norm, 1e-8)

        medoid_local_indices = []
        for cluster_idx in range(k):
            candidates = np.where(labels == cluster_idx)[0]
            center = centers[cluster_idx]
            dist = np.linalg.norm(embeddings[candidates] - center[None, :], axis=1)
            medoid_local_indices.append(int(candidates[np.argmin(dist)]))

        medoid_local_indices = np.asarray(medoid_local_indices, dtype=np.int64)
        medoid_indices = episode_indices[medoid_local_indices]
        medoid_point_clouds = episode_point_clouds[medoid_local_indices]

        out_dir = output_root / task_name / f"k_{k}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "prototypes.npy", prototypes)
        np.save(out_dir / "medoid_indices.npy", medoid_indices)
        np.save(out_dir / "medoid_point_clouds.npy", medoid_point_clouds)

        meta = {
            "task_name": task_name,
            "k": int(k),
            "checkpoint": str(checkpoint_path),
            "zarr_path": str(zarr_path),
            "terminal_window": int(args.terminal_window),
            "embedding_dim": int(embeddings.shape[1]),
            "num_episodes": int(embeddings.shape[0]),
            "medoid_episode_indices": medoid_indices.tolist(),
            "episode_metadata": [metadata[int(i)] for i in medoid_local_indices],
        }
        with (out_dir / "meta.json").open("w") as f:
            json.dump(meta, f, indent=2)
        print(f"[build_goal_bank] saved K={k} goal bank to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build terminal goal prototype/medoid banks.")
    parser.add_argument("--checkpoint", required=True, help="Path to mp_term/mp1_term checkpoint.")
    parser.add_argument("--zarr-path", default=None, help="Optional dataset zarr path. Defaults to cfg task dataset.")
    parser.add_argument("--task", default=None, help="Hydra task override, e.g. metaworld_push.")
    parser.add_argument("--task-name", default=None, help="Output task folder name if --task is not provided.")
    parser.add_argument("--config-name", default="mp_term.yaml", help="Fallback config when checkpoint has no cfg.")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4], help="KMeans cluster counts.")
    parser.add_argument("--terminal-window", type=int, default=8, help="Terminal window size per episode.")
    parser.add_argument("--output-root", default="data/goal_bank", help="Output root directory.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-ema", action="store_true", help="Use ema_model state_dict when available.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional cap for quick tests.")
    return parser.parse_args()


def main():
    args = parse_args()
    policy, cfg, checkpoint_path = _load_checkpoint_policy(args)
    if args.task_name is None and args.task is None and "task" in cfg and "name" in cfg.task:
        args.task_name = str(cfg.task.name)
    zarr_path = _dataset_path_from_args(args, cfg)
    terminal_window = int(args.terminal_window)

    embeddings, episode_point_clouds, episode_indices, metadata = _collect_episode_embeddings(
        policy=policy,
        zarr_path=zarr_path,
        terminal_window=terminal_window,
        device=args.device,
        max_episodes=args.max_episodes,
    )
    _save_goal_bank(
        args=args,
        embeddings=embeddings,
        episode_point_clouds=episode_point_clouds,
        episode_indices=episode_indices,
        metadata=metadata,
        checkpoint_path=checkpoint_path,
        zarr_path=zarr_path,
    )


if __name__ == "__main__":
    main()
