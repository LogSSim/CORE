import sys
sys.path.append('mp1')

from functools import partial
from typing import Dict

import numpy as np
import torch
from einops import rearrange
from termcolor import cprint

from mp1.common.model_util import print_params
from mp1.common.pytorch_util import dict_apply
from mp1.model.common.normalizer import LinearNormalizer
from mp1.model.mean.conditional_unet1d_meanflow_dis import ConditionalUnet1D
from mp1.model.mean.mask_generator import LowdimMaskGenerator
from mp1.model.vision.pointnet_extractor_temporal_conv import MP1TemporalConvEncoder
from mp1.policy.base_policy import BasePolicy


class Meanpolicy(BasePolicy):
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
        short_term_memory=None,
        **kwargs,
    ):
        super().__init__()

        self.condition_type = condition_type
        self.short_term_memory_cfg = short_term_memory or {}
        self.use_short_term_memory = bool(self.short_term_memory_cfg.get("enabled", False))
        self.num_memory_frames = int(self.short_term_memory_cfg.get("num_memory_frames", 1))

        action_shape = shape_meta["action"]["shape"]
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        obs_shape_meta = shape_meta["obs"]
        obs_dict = dict_apply(obs_shape_meta, lambda x: x["shape"])

        obs_encoder = MP1TemporalConvEncoder(
            observation_space=obs_dict,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
            short_term_memory=short_term_memory,
        )

        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(
            f"[MeanpolicyTemporalConv] use_pc_color: {self.use_pc_color}",
            "yellow",
        )
        cprint(
            f"[MeanpolicyTemporalConv] pointnet_type: {self.pointnet_type}",
            "yellow",
        )

        self.model = ConditionalUnet1D(
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
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if self.use_short_term_memory and self.n_obs_steps < 2:
            raise ValueError(
                "temporal short_term_memory requires n_obs_steps >= 2, "
                f"got n_obs_steps={self.n_obs_steps}"
            )

        self.num_inference_steps = num_inference_steps

        self.flow_ratio = 0.5
        self.time_dist = ["lognorm", -0.4, 1.0]
        cfg_scale = 2.0
        self.cfg_uncond = "u"
        self.w = cfg_scale
        print_params(self)

    def _encode_obs_features(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.obs_encoder(
            obs_dict,
            apply_short_term_memory=self.use_short_term_memory,
        )

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]

        value = next(iter(nobs.values()))
        batch_size, _ = value.shape[:2]
        horizon = self.horizon
        action_dim = self.action_dim
        obs_dim = self.obs_feature_dim
        obs_steps = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:, :obs_steps, ...])
            nobs_features = self._encode_obs_features(this_nobs)
            if "cross_attention" in self.condition_type:
                global_cond = nobs_features
            else:
                global_cond = nobs_features.reshape(batch_size, -1)
            cond_data = torch.zeros(
                size=(batch_size, horizon, action_dim),
                device=device,
                dtype=dtype,
            )
        else:
            this_nobs = dict_apply(nobs, lambda x: x[:, :obs_steps, ...])
            nobs_features = self._encode_obs_features(this_nobs)
            cond_data = torch.zeros(
                size=(batch_size, horizon, action_dim + obs_dim),
                device=device,
                dtype=dtype,
            )
            cond_data[:, :obs_steps, action_dim:] = nobs_features

        self.model.eval()
        z = torch.randn(
            size=cond_data.shape,
            dtype=cond_data.dtype,
            device=cond_data.device,
        )

        t = torch.ones((cond_data.shape[0],), device=cond_data.device)
        r = torch.zeros((cond_data.shape[0],), device=cond_data.device)

        z = z - self.model(
            sample=z,
            timestep=t,
            local_cond=local_cond,
            global_cond=global_cond,
            r=r,
            training=False,
        )

        naction_pred = z[..., :action_dim]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        start = obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        return {
            "action": action,
            "action_pred": action_pred,
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        if not self.use_pc_color:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        x = trajectory

        device = cond_data.device

        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda n: n[:, :self.n_obs_steps, ...])
            nobs_features = self._encode_obs_features(this_nobs)

            if "cross_attention" in self.condition_type:
                global_cond = nobs_features
            else:
                global_cond = nobs_features.reshape(batch_size, -1)
        else:
            this_nobs = nobs
            nobs_features = self._encode_obs_features(this_nobs)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        self.mask_generator(trajectory.shape)

        t, r = self.sample_t_r(batch_size, device)
        t_ = rearrange(t, "b -> b 1 1")
        r_ = rearrange(r, "b -> b 1 1")
        e = torch.randn_like(x)
        z = (1 - t_) * x + t_ * e
        v = e - x

        if self.w is not None:
            with torch.no_grad():
                u_t, _ = self.model(
                    sample=z,
                    timestep=t,
                    global_cond=global_cond,
                    r=t,
                )
            v_hat = self.w * v + (1 - self.w) * u_t
        else:
            v_hat = v

        model_partial = partial(self.model, global_cond=global_cond)
        pred, dudt = torch.autograd.functional.jvp(
            lambda z, t, r: model_partial(sample=z, timestep=t, r=r),
            (z, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
            create_graph=True,
        )

        u_tgt = v_hat - (t_ - r_) * dudt[0]

        error = pred[0] - stopgrad(u_tgt)
        meanflow_loss = adaptive_l2_loss(error)

        dis_loss = 0
        for feat in pred[1]:
            dis_loss += self.dispersive_loss(feat)

        loss = meanflow_loss + 0.5 * dis_loss
        mse_val = (stopgrad(error) ** 2).mean()

        loss_dict = {
            "bc_loss": loss.item(),
            "mse_val": mse_val.item(),
            "meanflow_loss": meanflow_loss.item(),
            "dis_loss": dis_loss.item(),
        }
        return loss, loss_dict

    def dispersive_loss(self, z, tau=1.0):
        dist_matrix = torch.cdist(z, z, p=2) ** 2
        dist_matrix = dist_matrix / torch.max(dist_matrix)
        exp_term = torch.exp(-dist_matrix / tau)
        mean_exp = torch.mean(exp_term)
        loss = torch.log(mean_exp)
        return loss

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
    loss = delta_sq
    return (stopgrad(w) * loss).mean()
