import sys
sys.path.append("MP1/mp1")

import copy
from functools import partial
from typing import Dict
import warnings

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from termcolor import cprint

from mp1.common.model_util import print_params
from mp1.common.pytorch_util import dict_apply
from mp1.model.common.normalizer import LinearNormalizer
from mp1.model.mean.conditional_unet1d_meanflow import ConditionalUnet1D
from mp1.model.mean.mask_generator import LowdimMaskGenerator
from mp1.model.vision.pointnet_extractor import MP1Encoder
from mp1.policy.base_policy import BasePolicy

warnings.filterwarnings("ignore")


def _obs_shape_dict(obs_shape_meta: dict) -> Dict:
    return {key: value["shape"] for key, value in obs_shape_meta.items()}


def _clone_cfg(cfg):
    if cfg is None:
        return None
    return copy.deepcopy(cfg)


def _set_cfg_value(cfg, key, value):
    if cfg is None:
        return
    try:
        setattr(cfg, key, value)
    except Exception:
        cfg[key] = value


class MeanpolicyTest2(BasePolicy):
    def __init__(
        self,
        shape_meta: dict,
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
        future_residual_steps=2,
        future_delta_encoder_output_dim=None,
        future_delta_include_state=True,
        future_residual_hidden_dim=256,
        future_residual_dropout=0.0,
        lambda_residual=0.1,
        detach_residual_target=True,
        detach_residual_for_action=False,
        **kwargs,
    ):
        super().__init__()

        if not obs_as_global_cond:
            raise ValueError("MeanpolicyTest2 only supports obs_as_global_cond=True.")
        if "cross_attention" in condition_type:
            raise NotImplementedError("MeanpolicyTest2 currently supports non-cross-attention conditioning only.")

        self.condition_type = condition_type
        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type

        action_shape = shape_meta["action"]["shape"]
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        obs_shape_meta = shape_meta["obs"]
        obs_dict = _obs_shape_dict(obs_shape_meta)
        self.obs_encoder = MP1Encoder(
            observation_space=obs_dict,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )

        self.obs_feature_dim = self.obs_encoder.output_shape()
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.num_inference_steps = num_inference_steps
        self.kwargs = kwargs

        self.future_residual_steps = int(future_residual_steps)
        self.required_obs_steps = self.n_obs_steps + self.future_residual_steps
        if self.future_residual_steps < 1:
            raise ValueError(f"future_residual_steps must be >= 1, got {self.future_residual_steps}")
        if self.horizon < self.required_obs_steps:
            raise ValueError(
                "MeanpolicyTest2 needs horizon >= n_obs_steps + future_residual_steps, "
                f"got horizon={self.horizon}, n_obs_steps={self.n_obs_steps}, "
                f"future_residual_steps={self.future_residual_steps}."
            )

        self.lambda_residual = float(lambda_residual)
        self.detach_residual_target = bool(detach_residual_target)
        self.detach_residual_for_action = bool(detach_residual_for_action)
        self.future_delta_include_state = bool(future_delta_include_state and "agent_pos" in obs_dict)

        future_delta_point_feature_dim = int(
            encoder_output_dim
            if future_delta_encoder_output_dim is None
            else future_delta_encoder_output_dim
        )
        delta_pointcloud_encoder_cfg = _clone_cfg(pointcloud_encoder_cfg)
        _set_cfg_value(delta_pointcloud_encoder_cfg, "out_channels", future_delta_point_feature_dim)
        delta_obs_dict = {"point_cloud": obs_dict["point_cloud"]}
        if self.future_delta_include_state:
            delta_obs_dict["agent_pos"] = obs_dict["agent_pos"]
        self.future_delta_encoder = MP1Encoder(
            observation_space=delta_obs_dict,
            img_crop_shape=crop_shape,
            out_channel=future_delta_point_feature_dim,
            pointcloud_encoder_cfg=delta_pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )
        self.future_delta_feature_dim = self.future_delta_encoder.output_shape()

        residual_input_dim = self.obs_feature_dim * self.n_obs_steps
        residual_output_dim = self.future_delta_feature_dim * self.future_residual_steps
        residual_hidden_dim = int(future_residual_hidden_dim)
        self.future_residual_predictor = nn.Sequential(
            nn.Linear(residual_input_dim, residual_hidden_dim),
            nn.LayerNorm(residual_hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(future_residual_dropout)),
            nn.Linear(residual_hidden_dim, residual_hidden_dim),
            nn.LayerNorm(residual_hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(future_residual_dropout)),
            nn.Linear(residual_hidden_dim, residual_output_dim),
        )
        nn.init.zeros_(self.future_residual_predictor[-1].weight)
        nn.init.zeros_(self.future_residual_predictor[-1].bias)

        global_cond_dim = (
            self.obs_feature_dim * self.n_obs_steps
            + self.future_delta_feature_dim * self.future_residual_steps
        )
        self.model = ConditionalUnet1D(
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

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()

        self.flow_ratio = float(kwargs.get("flow_ratio", 0.75))
        self.time_dist = kwargs.get("time_dist", ["lognorm", -0.4, 1.0])
        cfg_scale = float(kwargs.get("cfg_scale", 2.0))
        self.w = cfg_scale
        self.kappa = kwargs.get("kappa", None)
        init_alpha = kwargs.get("alpha", 0.0)
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

        cprint(f"[MeanpolicyTest2] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[MeanpolicyTest2] pointnet_type: {self.pointnet_type}", "yellow")
        cprint(
            "[MeanpolicyTest2] "
            f"obs_feature_dim={self.obs_feature_dim}, "
            f"future_delta_feature_dim={self.future_delta_feature_dim}, "
            f"future_delta_include_state={self.future_delta_include_state}, "
            f"future_residual_steps={self.future_residual_steps}, "
            f"global_cond_dim={global_cond_dim}, "
            f"lambda_residual={self.lambda_residual}, "
            f"detach_residual_for_action={self.detach_residual_for_action}",
            "yellow",
        )
        print_params(self)

    def _normalize_obs(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color and "point_cloud" in nobs:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]
        return nobs

    def _encode_obs_sequence(
        self,
        nobs: Dict[str, torch.Tensor],
        num_steps: int,
    ) -> torch.Tensor:
        value = next(iter(nobs.values()))
        if value.shape[1] < num_steps:
            raise ValueError(f"Need at least {num_steps} observation steps, got {value.shape[1]}")

        batch_size = value.shape[0]
        obs_window = dict_apply(
            nobs,
            lambda x: x[:, :num_steps, ...].reshape(-1, *x.shape[2:]),
        )
        obs_features = self.obs_encoder(obs_window)
        return obs_features.reshape(batch_size, num_steps, -1)

    def _predict_future_residuals(self, history_features: torch.Tensor) -> torch.Tensor:
        batch_size = history_features.shape[0]
        pred = self.future_residual_predictor(history_features.reshape(batch_size, -1))
        return pred.reshape(batch_size, self.future_residual_steps, self.future_delta_feature_dim)

    def _encode_delta_sequence(self, delta_obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        point_deltas = delta_obs["point_cloud"]
        if point_deltas.ndim != 4:
            raise ValueError(f"point_deltas must be [B, T, N, C], got {tuple(point_deltas.shape)}")

        batch_size, time_steps = point_deltas.shape[:2]
        flat_delta_obs = {}
        for key, value in delta_obs.items():
            if value.shape[:2] != (batch_size, time_steps):
                raise ValueError(
                    f"{key} delta must start with {(batch_size, time_steps)}, got {tuple(value.shape)}"
                )
            flat_delta_obs[key] = value.reshape(-1, *value.shape[2:])
        delta_features = self.future_delta_encoder(flat_delta_obs)
        return delta_features.reshape(batch_size, time_steps, -1)

    def _target_future_residuals(self, nobs: Dict[str, torch.Tensor]) -> torch.Tensor:
        point_cloud = nobs["point_cloud"]
        if point_cloud.shape[1] < self.required_obs_steps:
            raise ValueError(
                f"Need at least {self.required_obs_steps} point cloud steps, got {point_cloud.shape[1]}"
            )

        start = self.n_obs_steps - 1
        end = start + self.future_residual_steps
        prev_points = point_cloud[:, start:end, ...]
        next_points = point_cloud[:, start + 1:end + 1, ...]
        delta_obs = {"point_cloud": next_points - prev_points}
        if self.future_delta_include_state:
            agent_pos = nobs["agent_pos"]
            prev_state = agent_pos[:, start:end, ...]
            next_state = agent_pos[:, start + 1:end + 1, ...]
            delta_obs["agent_pos"] = next_state - prev_state
        target = self._encode_delta_sequence(delta_obs)
        if self.detach_residual_target:
            target = target.detach()
        return target

    def _build_global_condition(
        self,
        history_features: torch.Tensor,
        future_residuals: torch.Tensor,
    ) -> torch.Tensor:
        if self.detach_residual_for_action:
            future_residuals = future_residuals.detach()
        return torch.cat(
            [
                history_features.reshape(history_features.shape[0], -1),
                future_residuals.reshape(future_residuals.shape[0], -1),
            ],
            dim=-1,
        )

    def _model_forward(
        self,
        model: ConditionalUnet1D,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
        r: torch.Tensor,
        local_cond=None,
    ) -> torch.Tensor:
        original_horizon = sample.shape[1]
        num_upsamples = len(getattr(model, "up_modules", []))
        horizon_multiple = max(2 ** num_upsamples, 1)
        remainder = original_horizon % horizon_multiple
        if remainder != 0:
            pad_len = horizon_multiple - remainder
            sample_pad = sample[:, -1:, :].expand(-1, pad_len, -1)
            sample = torch.cat([sample, sample_pad], dim=1)
            if local_cond is not None:
                local_pad = local_cond[:, -1:, :].expand(-1, pad_len, -1)
                local_cond = torch.cat([local_cond, local_pad], dim=1)

        output = model(
            sample=sample,
            timestep=timestep,
            local_cond=local_cond,
            global_cond=global_cond,
            r=r,
        )
        if output.shape[1] < original_horizon:
            raise ValueError(
                f"Model output horizon {output.shape[1]} is shorter than input horizon {original_horizon}"
            )
        if output.shape[1] > original_horizon:
            output = output[:, :original_horizon, :]
        if output.shape[-1] != sample.shape[-1]:
            raise ValueError(
                f"Model output dim {output.shape[-1]} does not match input dim {sample.shape[-1]}"
            )
        return output

    def _meanflow_loss(
        self,
        x: torch.Tensor,
        global_cond: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ):
        t, r = self.sample_t_r(batch_size, device)
        t = t.to(dtype=x.dtype)
        r = r.to(dtype=x.dtype)
        t_ = rearrange(t, "b -> b 1 1")
        r_ = rearrange(r, "b -> b 1 1")

        e = torch.randn_like(x)
        z = (1 - t_) * x + t_ * e
        v = e - x

        if self.w is not None and self.kappa is None:
            with torch.no_grad():
                u_t = self._model_forward(
                    self.model,
                    sample=z,
                    timestep=t,
                    global_cond=global_cond,
                    r=t,
                )
            v_hat = self.w * v + (1 - self.w) * u_t
        elif self.w is not None and self.kappa > 0:
            uncond = torch.ones_like(global_cond)
            u_uncond = self._model_forward(
                self.model,
                sample=z,
                timestep=t,
                global_cond=uncond,
                r=t,
            )
            u_cond = self._model_forward(
                self.model,
                sample=z,
                timestep=t,
                global_cond=global_cond,
                r=t,
            )
            v_hat = self.w * v + (1 - self.w - self.kappa) * u_uncond + self.kappa * u_cond
        else:
            v_hat = v

        model_partial = partial(self._model_forward, self.model, global_cond=global_cond)
        u, dudt = torch.autograd.functional.jvp(
            lambda sample, time, ref: model_partial(sample=sample, timestep=time, r=time - ref),
            (z, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
            create_graph=True,
        )

        if self.alpha.detach().item() == 0.0:
            coef = t_ - r_
        else:
            t_safe = torch.clamp(t_, min=1e-5)
            coef = (t_safe - r_ * (r_ / t_safe) ** self.alpha) / (self.alpha + 1.0)

        u_tgt = v_hat - coef * dudt
        error = u - stopgrad(u_tgt)
        loss = adaptive_l2_loss(error)
        mse_val = (stopgrad(error) ** 2).mean()
        return loss, mse_val

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self._normalize_obs(obs_dict)
        history_features = self._encode_obs_sequence(nobs, self.n_obs_steps)
        pred_residuals = self._predict_future_residuals(history_features)
        global_cond = self._build_global_condition(history_features, pred_residuals)

        batch_size = history_features.shape[0]
        z = torch.randn(
            size=(batch_size, self.horizon, self.action_dim),
            dtype=global_cond.dtype,
            device=global_cond.device,
        )
        t = torch.ones((batch_size,), device=z.device, dtype=z.dtype)
        r = torch.zeros((batch_size,), device=z.device, dtype=z.dtype)

        self.model.eval()
        z = z - self._model_forward(
            self.model,
            sample=z,
            timestep=t,
            local_cond=None,
            global_cond=global_cond,
            r=r,
        )

        action_pred = self.normalizer["action"].unnormalize(z)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        return {
            "action": action,
            "action_pred": action_pred,
            "pred_future_residual": pred_residuals,
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        nobs = self._normalize_obs(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        batch_size = nactions.shape[0]
        device = nactions.device

        history_features = self._encode_obs_sequence(nobs, self.n_obs_steps)
        pred_residuals = self._predict_future_residuals(history_features)
        target_residuals = self._target_future_residuals(nobs)
        residual_loss = torch.nn.functional.mse_loss(pred_residuals, target_residuals)
        global_cond = self._build_global_condition(history_features, pred_residuals)

        action_loss, action_mse = self._meanflow_loss(
            x=nactions,
            global_cond=global_cond,
            batch_size=batch_size,
            device=device,
        )
        loss = action_loss + self.lambda_residual * residual_loss

        residual_mse = ((pred_residuals.detach() - target_residuals.detach()) ** 2).mean()
        loss_dict = {
            "bc_loss": loss.item(),
            "action_loss": action_loss.item(),
            "residual_loss": residual_loss.item(),
            "action_mse_val": action_mse.item(),
            "residual_mse_val": residual_mse.item(),
            "target_residual_norm": target_residuals.norm(dim=-1).mean().item(),
            "pred_residual_norm": pred_residuals.norm(dim=-1).mean().item(),
        }
        return loss, loss_dict

    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == "uniform":
            samples = np.random.rand(batch_size, 2).astype(np.float32)
        elif self.time_dist[0] == "lognorm":
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))
        else:
            raise ValueError(f"Unsupported time distribution {self.time_dist[0]}")

        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r


def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    delta_sq = torch.mean(error ** 2, dim=tuple(range(1, error.ndim)))
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    return (stopgrad(w) * delta_sq).mean()
