"""Standalone DP3 policy with terminal two-frame global conditioning.

This implementation is intentionally separate from the existing MP/MP1 policy
files.  It follows the original DP3 diffusion-action structure, and appends the
encoded features of two demonstration terminal frames to the global condition.
"""

from typing import Dict, Optional

import copy
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from termcolor import cprint

from mp1.common.model_util import print_params
from mp1.common.pytorch_util import dict_apply
from mp1.model.common.normalizer import LinearNormalizer
from mp1.model.mean.conditional_unet1d import ConditionalUnet1D
from mp1.model.mean.mask_generator import LowdimMaskGenerator
from mp1.model.vision.pointnet_extractor import MP1Encoder
from mp1.policy.base_policy import BasePolicy


def _as_plain_dict(cfg):
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return copy.deepcopy(cfg)
    try:
        return copy.deepcopy(dict(cfg))
    except Exception:
        return copy.deepcopy(cfg)


def _resolve_data_path(path):
    path = Path(os.path.expanduser(str(path)))
    if path.is_absolute() and path.exists():
        return str(path)
    if path.exists():
        return str(path)
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / path
    if candidate.exists():
        return str(candidate)
    return str(path)


def _episode_bounds(episode_ends: np.ndarray, episode_idx: int):
    start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    end_exclusive = int(episode_ends[episode_idx])
    return start, end_exclusive


def _select_terminal_episode_indices(
    n_episodes: int,
    stage: int,
    stage1_episode_start: int,
    stage1_episode_end: int,
    stage3_num_episodes: int,
    seed: int,
) -> np.ndarray:
    start = max(0, min(int(stage1_episode_start), int(n_episodes)))
    end = max(start, min(int(stage1_episode_end), int(n_episodes)))
    indices = np.arange(start, end, dtype=np.int64)
    if indices.size == 0:
        raise ValueError(
            f"No terminal-condition episodes selected from range "
            f"[{stage1_episode_start}, {stage1_episode_end}) for {n_episodes} episodes."
        )
    if int(stage) == 3:
        keep = min(int(stage3_num_episodes), indices.size)
        rng = np.random.default_rng(int(seed))
        indices = np.sort(rng.choice(indices, size=keep, replace=False)).astype(np.int64)
    elif int(stage) != 1:
        raise ValueError(f"stage must be 1 or 3, got {stage}")
    return indices


class DP3TerminalConditionPolicy(BasePolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        condition_type="film",
        use_down_condition=True,
        use_mid_condition=True,
        use_up_condition=True,
        encoder_output_dim=256,
        crop_shape=None,
        use_pc_color=False,
        pointnet_type="pointnet",
        pointcloud_encoder_cfg=None,
        # terminal two-frame condition
        terminal_frame_count=2,
        terminal_zarr_path=None,
        terminal_stage=3,
        terminal_stage1_episode_start=20,
        terminal_stage1_episode_end=50,
        terminal_stage3_num_episodes=10,
        terminal_seed=42,
        terminal_eval_mode="first",
        terminal_eval_index=0,
        **kwargs,
    ):
        super().__init__()
        if not obs_as_global_cond:
            raise ValueError("DP3TerminalConditionPolicy only supports obs_as_global_cond=True.")
        if int(terminal_frame_count) != 2:
            raise ValueError(
                "This policy is for terminal two-frame conditioning; "
                f"got terminal_frame_count={terminal_frame_count}."
            )

        self.condition_type = condition_type
        self.use_pc_color = bool(use_pc_color)
        self.pointnet_type = pointnet_type
        self.terminal_frame_count = int(terminal_frame_count)
        self.terminal_zarr_path = terminal_zarr_path
        self.terminal_stage = int(terminal_stage)
        self.terminal_stage1_episode_start = int(terminal_stage1_episode_start)
        self.terminal_stage1_episode_end = int(terminal_stage1_episode_end)
        self.terminal_stage3_num_episodes = int(terminal_stage3_num_episodes)
        self.terminal_seed = int(terminal_seed)
        self.terminal_eval_mode = str(terminal_eval_mode)
        self.terminal_eval_index = int(terminal_eval_index)

        action_shape = shape_meta["action"]["shape"]
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = int(action_shape[0])
        elif len(action_shape) == 2:
            action_dim = int(action_shape[0]) * int(action_shape[1])
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        obs_shape_meta = shape_meta["obs"]
        obs_dict = dict_apply(obs_shape_meta, lambda x: x["shape"])
        self.obs_keys = tuple(obs_dict.keys())

        obs_encoder = MP1Encoder(
            observation_space=obs_dict,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )

        obs_feature_dim = int(obs_encoder.output_shape())
        input_dim = action_dim
        if "cross_attention" in self.condition_type:
            global_cond_dim = obs_feature_dim
        else:
            global_cond_dim = obs_feature_dim * (int(n_obs_steps) + self.terminal_frame_count)

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )

        self.normalizer = LinearNormalizer()
        self.horizon = int(horizon)
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = int(n_action_steps)
        self.n_obs_steps = int(n_obs_steps)
        self.obs_as_global_cond = bool(obs_as_global_cond)
        self.kwargs = kwargs
        self.num_inference_steps = (
            int(num_inference_steps)
            if num_inference_steps is not None
            else int(noise_scheduler.config.num_train_timesteps)
        )

        self._init_terminal_bank_buffers()

        cprint("[DP3-Terminal] standalone DP3 + terminal two-frame condition", "yellow")
        cprint(
            f"[DP3-Terminal] stage={self.terminal_stage}, "
            f"stage1_range=[{self.terminal_stage1_episode_start}, {self.terminal_stage1_episode_end}), "
            f"stage3_num={self.terminal_stage3_num_episodes}",
            "yellow",
        )
        cprint(f"[DP3-Terminal] obs_feature_dim={self.obs_feature_dim}", "yellow")
        print_params(self)

    def _init_terminal_bank_buffers(self):
        point_cloud_bank = torch.empty(0)
        agent_pos_bank = torch.empty(0)
        episode_indices = torch.empty(0, dtype=torch.long)

        if self.terminal_zarr_path is not None:
            point_cloud_bank_np, agent_pos_bank_np, episode_indices_np = self._load_terminal_bank(
                self.terminal_zarr_path
            )
            point_cloud_bank = torch.from_numpy(point_cloud_bank_np.astype(np.float32))
            agent_pos_bank = torch.from_numpy(agent_pos_bank_np.astype(np.float32))
            episode_indices = torch.from_numpy(episode_indices_np.astype(np.int64))

        self.register_buffer("terminal_point_cloud_bank", point_cloud_bank, persistent=True)
        self.register_buffer("terminal_agent_pos_bank", agent_pos_bank, persistent=True)
        self.register_buffer("terminal_episode_indices", episode_indices, persistent=True)

    def _load_terminal_bank(self, zarr_path):
        import zarr

        path = _resolve_data_path(zarr_path)
        root = zarr.open(path, mode="r")
        episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
        episode_indices = _select_terminal_episode_indices(
            n_episodes=len(episode_ends),
            stage=self.terminal_stage,
            stage1_episode_start=self.terminal_stage1_episode_start,
            stage1_episode_end=self.terminal_stage1_episode_end,
            stage3_num_episodes=self.terminal_stage3_num_episodes,
            seed=self.terminal_seed,
        )

        pc_bank = []
        state_bank = []
        for ep_idx in episode_indices:
            start, end_exclusive = _episode_bounds(episode_ends, int(ep_idx))
            first_idx = max(start, end_exclusive - self.terminal_frame_count)
            frame_indices = np.arange(first_idx, end_exclusive, dtype=np.int64)
            if frame_indices.shape[0] < self.terminal_frame_count:
                pad = np.full(
                    (self.terminal_frame_count - frame_indices.shape[0],),
                    int(frame_indices[0]),
                    dtype=np.int64,
                )
                frame_indices = np.concatenate([pad, frame_indices], axis=0)
            pc_bank.append(root["data"]["point_cloud"][frame_indices].astype(np.float32))
            state_bank.append(root["data"]["state"][frame_indices].astype(np.float32))

        return (
            np.stack(pc_bank, axis=0),
            np.stack(state_bank, axis=0),
            episode_indices,
        )

    # ========= condition helpers =========
    def _normalize_obs(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs_dict = {k: obs_dict[k] for k in self.obs_keys if k in obs_dict}
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color and "point_cloud" in nobs:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]
        return nobs

    def _normalize_terminal_obs(self, terminal_obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        normalized = {}
        if "point_cloud" in terminal_obs:
            pc = self.normalizer["point_cloud"].normalize(terminal_obs["point_cloud"])
            if not self.use_pc_color:
                pc = pc[..., :3]
            normalized["point_cloud"] = pc
        if "agent_pos" in terminal_obs:
            normalized["agent_pos"] = self.normalizer["agent_pos"].normalize(terminal_obs["agent_pos"])
        return normalized

    def _terminal_obs_from_bank(self, batch_size: int, device, dtype) -> Dict[str, torch.Tensor]:
        if self.terminal_point_cloud_bank.numel() == 0:
            raise ValueError(
                "predict_action needs terminal_obs, or set policy.terminal_zarr_path so "
                "a terminal two-frame bank can be loaded."
            )

        pc_bank = self.terminal_point_cloud_bank.to(device=device, dtype=dtype)
        state_bank = self.terminal_agent_pos_bank.to(device=device, dtype=dtype)
        if self.terminal_eval_mode == "mean":
            pc = pc_bank.mean(dim=0, keepdim=True).expand(batch_size, -1, -1, -1)
            state = state_bank.mean(dim=0, keepdim=True).expand(batch_size, -1, -1)
        elif self.terminal_eval_mode == "first":
            idx = max(0, min(self.terminal_eval_index, pc_bank.shape[0] - 1))
            pc = pc_bank[idx:idx + 1].expand(batch_size, -1, -1, -1)
            state = state_bank[idx:idx + 1].expand(batch_size, -1, -1)
        else:
            raise ValueError(f"Unsupported terminal_eval_mode={self.terminal_eval_mode}")
        return {"point_cloud": pc, "agent_pos": state}

    def _build_global_cond(
        self,
        nobs: Dict[str, torch.Tensor],
        nterminal_obs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]

        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
        )
        obs_features = self.obs_encoder(this_nobs)

        terminal_flat = dict_apply(
            nterminal_obs,
            lambda x: x[:, : self.terminal_frame_count, ...].reshape(-1, *x.shape[2:]),
        )
        terminal_features = self.obs_encoder(terminal_flat)

        if "cross_attention" in self.condition_type:
            obs_tokens = obs_features.reshape(batch_size, self.n_obs_steps, -1)
            terminal_tokens = terminal_features.reshape(batch_size, self.terminal_frame_count, -1)
            return torch.cat([obs_tokens, terminal_tokens], dim=1)

        obs_features = obs_features.reshape(batch_size, -1)
        terminal_features = terminal_features.reshape(batch_size, -1)
        return torch.cat([obs_features, terminal_features], dim=-1)

    # ========= inference =========
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        **kwargs,
    ):
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )
        scheduler = self.noise_scheduler
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = self.model(
                sample=trajectory,
                timestep=t,
                local_cond=local_cond,
                global_cond=global_cond,
                training=False,
            )
            trajectory = scheduler.step(model_output, t, trajectory).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        terminal_obs = obs_dict.get("terminal_obs", None)
        current_obs = {k: v for k, v in obs_dict.items() if k != "terminal_obs"}
        nobs = self._normalize_obs(current_obs)

        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        device = self.device
        dtype = self.dtype
        if terminal_obs is None:
            terminal_obs = self._terminal_obs_from_bank(batch_size, device=device, dtype=dtype)
        nterminal_obs = self._normalize_terminal_obs(terminal_obs)

        global_cond = self._build_global_cond(nobs, nterminal_obs)
        cond_data = torch.zeros(
            size=(batch_size, self.horizon, self.action_dim),
            device=device,
            dtype=dtype,
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=None,
            global_cond=global_cond,
            training=False,
            **self.kwargs,
        )
        naction_pred = nsample[..., : self.action_dim]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        return {
            "action": action_pred[:, start:end],
            "action_pred": action_pred,
        }

    # ========= training =========
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        if "terminal_obs" not in batch:
            raise KeyError(
                "DP3TerminalConditionPolicy needs batch['terminal_obs']. "
                "Use DP3TerminalConditionDataset for training."
            )

        nobs = self._normalize_obs(batch["obs"])
        nterminal_obs = self._normalize_terminal_obs(batch["terminal_obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        trajectory = nactions
        cond_data = trajectory
        global_cond = self._build_global_cond(nobs, nterminal_obs)
        condition_mask = self.mask_generator(trajectory.shape)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        batch_size = trajectory.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        pred, _ = self.model(
            sample=noisy_trajectory,
            timestep=timesteps,
            local_cond=None,
            global_cond=global_cond,
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        elif pred_type == "v_prediction":
            # Compatibility with schedulers that expose alpha_t/sigma_t.
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t = self.noise_scheduler.alpha_t[timesteps].unsqueeze(-1).unsqueeze(-1)
            sigma_t = self.noise_scheduler.sigma_t[timesteps].unsqueeze(-1).unsqueeze(-1)
            target = alpha_t * noise - sigma_t * trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean").mean()

        loss_dict = {
            "bc_loss": loss.item(),
            "terminal_cond_episodes": float(max(1, self.terminal_episode_indices.numel())),
        }
        return loss, loss_dict
