"""Standalone DP3 policy for final-state cluster conditioning.

This follows the original DP3 diffusion policy and only changes the global
condition in stage 3:

    original observation condition
    + clustered final-state feature
    + (current point-cloud feature - final-state feature)
"""

from typing import Dict, Optional

import copy
import os
from pathlib import Path

import dill
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


def _clone_cfg(cfg):
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return copy.deepcopy(cfg)
    try:
        return copy.deepcopy(dict(cfg))
    except Exception:
        return copy.deepcopy(cfg)


def _resolve_data_path(path):
    if path is None:
        return None
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


def _resolve_checkpoint_path(path):
    path = Path(_resolve_data_path(path))
    if path.is_dir():
        for candidate in (path / "checkpoints" / "latest.ckpt", path / "latest.ckpt"):
            if candidate.is_file():
                return str(candidate)
        raise FileNotFoundError(
            f"Checkpoint directory does not contain checkpoints/latest.ckpt or latest.ckpt: {path}"
        )
    return str(path)


class DP3FinalStateConditionPolicy(BasePolicy):
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
        final_feature_dim=None,
        crop_shape=None,
        use_pc_color=False,
        pointnet_type="pointnet",
        pointcloud_encoder_cfg=None,
        policy_stage=1,
        final_state_feature_path=None,
        stage1_checkpoint_path=None,
        freeze_stage1_encoder=False,
        **kwargs,
    ):
        super().__init__()
        if not obs_as_global_cond:
            raise ValueError("DP3FinalStateConditionPolicy only supports obs_as_global_cond=True.")
        if "cross_attention" in condition_type:
            raise ValueError("Final-state conditioning currently supports FiLM/add conditions only.")

        self.condition_type = condition_type
        self.use_pc_color = bool(use_pc_color)
        self.pointnet_type = pointnet_type
        self.policy_stage = int(policy_stage)
        if self.policy_stage not in (1, 3):
            raise ValueError(f"policy_stage must be 1 or 3, got {self.policy_stage}")

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
            pointcloud_encoder_cfg=_clone_cfg(pointcloud_encoder_cfg),
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )

        obs_feature_dim = int(obs_encoder.output_shape())
        final_feature_dim = int(encoder_output_dim if final_feature_dim is None else final_feature_dim)
        global_cond_dim = obs_feature_dim * int(n_obs_steps)
        if self.policy_stage == 3:
            global_cond_dim += final_feature_dim * 2

        model = ConditionalUnet1D(
            input_dim=action_dim,
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
        self.final_feature_dim = final_feature_dim
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

        self._load_default_final_state_features(final_state_feature_path)
        if stage1_checkpoint_path is not None:
            self._load_stage1_encoder(stage1_checkpoint_path, freeze=freeze_stage1_encoder)

        cprint(
            f"[DP3-FinalState] stage={self.policy_stage}, "
            f"obs_feature_dim={self.obs_feature_dim}, final_feature_dim={self.final_feature_dim}",
            "yellow",
        )
        print_params(self)

    def _load_default_final_state_features(self, final_state_feature_path):
        default_final = torch.zeros(self.final_feature_dim, dtype=torch.float32)
        default_cluster = torch.zeros(self.final_feature_dim, dtype=torch.float32)
        path = _resolve_data_path(final_state_feature_path)
        if path is not None and Path(path).exists():
            artifact = np.load(path)
            final_features = artifact["final_features"].astype(np.float32)
            if "cluster_features" in artifact:
                cluster_features = artifact["cluster_features"].astype(np.float32)
            else:
                labels = artifact["cluster_labels"].astype(np.int64)
                centers = artifact["cluster_centers"].astype(np.float32)
                cluster_features = centers[labels]
            default_final = torch.from_numpy(final_features.mean(axis=0))
            default_cluster = torch.from_numpy(cluster_features.mean(axis=0))

        self.register_buffer("default_final_feature", default_final, persistent=True)
        self.register_buffer("default_cluster_feature", default_cluster, persistent=True)

    def _load_stage1_encoder(self, checkpoint_path, freeze=False):
        checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
        payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill, map_location="cpu")
        model_state = payload["state_dicts"]["model"]
        encoder_prefix = "obs_encoder."
        encoder_state = {
            key[len(encoder_prefix):]: value
            for key, value in model_state.items()
            if key.startswith(encoder_prefix)
        }
        missing, unexpected = self.obs_encoder.load_state_dict(encoder_state, strict=False)
        cprint(
            f"[DP3-FinalState] loaded stage1 encoder from {checkpoint_path}; "
            f"missing={len(missing)}, unexpected={len(unexpected)}",
            "yellow",
        )
        if freeze:
            for param in self.obs_encoder.parameters():
                param.requires_grad_(False)

    def _normalize_obs(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs_dict = {k: obs_dict[k] for k in self.obs_keys if k in obs_dict}
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color and "point_cloud" in nobs:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]
        return nobs

    def _encode_obs_condition(self, nobs: Dict[str, torch.Tensor]) -> torch.Tensor:
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
        )
        obs_features = self.obs_encoder(this_nobs)
        return obs_features.reshape(batch_size, -1)

    def _encode_current_point_feature(self, nobs: Dict[str, torch.Tensor]) -> torch.Tensor:
        point_cloud = nobs["point_cloud"][:, self.n_obs_steps - 1, ...]
        if not self.use_pc_color:
            point_cloud = point_cloud[..., :3]
        return self.obs_encoder.extractor(point_cloud)

    def _condition_from_batch(self, batch_size, device, dtype, condition=None):
        if condition is None:
            final_feature = self.default_final_feature.to(device=device, dtype=dtype)
            cluster_feature = self.default_cluster_feature.to(device=device, dtype=dtype)
            final_feature = final_feature.unsqueeze(0).expand(batch_size, -1)
            cluster_feature = cluster_feature.unsqueeze(0).expand(batch_size, -1)
            return final_feature, cluster_feature

        final_feature = condition["final_feature"].to(device=device, dtype=dtype)
        cluster_feature = condition["cluster_feature"].to(device=device, dtype=dtype)
        return final_feature, cluster_feature

    def _build_global_cond(
        self,
        nobs: Dict[str, torch.Tensor],
        final_state_condition: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        obs_cond = self._encode_obs_condition(nobs)
        if self.policy_stage == 1:
            return obs_cond

        batch_size = obs_cond.shape[0]
        device = obs_cond.device
        dtype = obs_cond.dtype
        final_feature, cluster_feature = self._condition_from_batch(
            batch_size,
            device,
            dtype,
            condition=final_state_condition,
        )
        current_feature = self._encode_current_point_feature(nobs)
        feature_delta = current_feature - final_feature
        return torch.cat([obs_cond, cluster_feature, feature_delta], dim=-1)

    def _model_forward(self, *args, **kwargs):
        pred = self.model(*args, **kwargs)
        if isinstance(pred, tuple):
            return pred[0]
        return pred

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
            model_output = self._model_forward(
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
        final_state_condition = obs_dict.get("final_state_condition", None)
        current_obs = {k: v for k, v in obs_dict.items() if k != "final_state_condition"}
        nobs = self._normalize_obs(current_obs)

        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        device = self.device
        dtype = self.dtype

        global_cond = self._build_global_cond(nobs, final_state_condition)
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
        if self.policy_stage == 3 and "final_state_condition" not in batch:
            raise KeyError(
                "stage=3 requires batch['final_state_condition']. "
                "Use DP3FinalStateDataset with final_state_feature_path."
            )

        nobs = self._normalize_obs(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        final_state_condition = batch.get("final_state_condition", None)

        trajectory = nactions
        cond_data = trajectory
        global_cond = self._build_global_cond(nobs, final_state_condition)
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

        pred = self._model_forward(
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

        loss_dict = {"bc_loss": loss.item()}
        if self.policy_stage == 3:
            loss_dict["final_delta_norm"] = (
                global_cond[:, -self.final_feature_dim :].detach().norm(dim=-1).mean().item()
            )
        return loss, loss_dict
