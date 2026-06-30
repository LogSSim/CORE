#!/usr/bin/env python3
"""Build the shared terminal prototype for CORE_dp3 stage 2."""

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
    config_dir = str((PROJECT_DIR / "core" / "config").resolve())
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
    if not hasattr(policy, "terminal_encoder") or policy.terminal_encoder is None:
        raise RuntimeError(
            "Loaded policy has no terminal_encoder. Use CORE_dp3 stage1 checkpoint."
        )
    if missing:
        print(f"[build_CORE_dp3_bank] missing keys: {missing}")
    if unexpected:
        print(f"[build_CORE_dp3_bank] unexpected keys: {unexpected}")
    policy.to(args.device)
    policy.eval()
    return policy, cfg, checkpoint_path


def _dataset_path_from_args(args, cfg):
    if args.zarr_path:
        return _resolve_path(args.zarr_path)
    return _resolve_path(cfg.task.dataset.zarr_path)


def _collect_final_frame_embeddings(policy, zarr_path, device, max_episodes=None):
    root = zarr.open(str(zarr_path), mode="r")
    point_cloud = root["data"]["point_cloud"]
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    embeddings = []
    episode_indices = []
    metadata = []

    prev_end = 0
    with torch.no_grad():
        for episode_idx, end_exclusive in enumerate(episode_ends):
            start = int(prev_end)
            end = int(end_exclusive) - 1
            prev_end = int(end_exclusive)
            pc = torch.from_numpy(point_cloud[end].astype(np.float32)).to(device).unsqueeze(0)
            z = policy.encode_terminal_z(pc).squeeze(0)
            z = F.normalize(z, dim=0)
            embeddings.append(z.detach().cpu().numpy().astype(np.float32))
            episode_indices.append(int(episode_idx))
            metadata.append(
                {
                    "episode_idx": int(episode_idx),
                    "episode_start": start,
                    "episode_end": end,
                    "terminal_frame": end,
                }
            )
            if max_episodes is not None and len(embeddings) >= int(max_episodes):
                break
    if len(embeddings) == 0:
        raise RuntimeError(f"No episodes found in {zarr_path}")
    return np.stack(embeddings, axis=0), np.asarray(episode_indices, dtype=np.int64), metadata


def _normalized_mean(vectors):
    common = vectors.mean(axis=0).astype(np.float32)
    common = common / np.clip(np.linalg.norm(common), 1e-8, None)
    return common.astype(np.float32)


def _save_bank(args, embeddings, episode_indices, metadata, checkpoint_path, zarr_path):
    output_root = _resolve_path(args.output_root)
    task_name = args.task or args.task_name or "unknown_task"
    out_dir = output_root / task_name / f"k_{int(args.k)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    common = _normalized_mean(embeddings)
    np.save(out_dir / "common_prototype.npy", common)
    np.save(out_dir / "common_prototype_from_all.npy", common)
    np.save(out_dir / "prototypes.npy", common[None, :])
    np.save(out_dir / "episode_embeddings.npy", embeddings.astype(np.float32))
    np.save(out_dir / "episode_indices.npy", episode_indices.astype(np.int64))

    with (out_dir / "meta.json").open("w") as f:
        json.dump(
            {
                "task_name": task_name,
                "K": int(args.k),
                "k": int(args.k),
                "method": "mean_final_frame_embeddings",
                "checkpoint": str(checkpoint_path),
                "zarr_path": str(zarr_path),
                "common_prototype_path": str(out_dir / "common_prototype.npy"),
                "prototype_path": str(out_dir / "prototypes.npy"),
                "embedding_dim": int(embeddings.shape[1]),
                "num_episodes": int(embeddings.shape[0]),
                "episode_metadata": metadata,
            },
            f,
            indent=2,
        )
    print(f"[build_CORE_dp3_bank] saved mean goal bank to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build CORE-DP3 shared terminal prototype bank.")
    parser.add_argument("--checkpoint", required=True, help="Stage1 checkpoint or run directory.")
    parser.add_argument("--zarr-path", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--config-name", default="CORE_dp3")
    parser.add_argument("--k", type=int, default=1, help="Output folder suffix k_K; no KMeans is run.")
    parser.add_argument("--output-root", default="data/goal_bank_CORE_dp3")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    policy, cfg, checkpoint_path = _load_policy(args)
    if args.task_name is None and args.task is None and "task" in cfg and "name" in cfg.task:
        args.task_name = str(cfg.task.name)
    zarr_path = _dataset_path_from_args(args, cfg)
    embeddings, episode_indices, metadata = _collect_final_frame_embeddings(
        policy=policy,
        zarr_path=zarr_path,
        device=args.device,
        max_episodes=args.max_episodes,
    )
    _save_bank(
        args=args,
        embeddings=embeddings,
        episode_indices=episode_indices,
        metadata=metadata,
        checkpoint_path=checkpoint_path,
        zarr_path=zarr_path,
    )


if __name__ == "__main__":
    main()
