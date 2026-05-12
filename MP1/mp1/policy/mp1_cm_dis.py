from functools import partial
from typing import Dict, List

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from termcolor import cprint

from mp1.common.model_util import print_params
from mp1.common.pytorch_util import dict_apply
from mp1.model.common.causal_memory import CausalMemoryEncoder, MLPBlock
from mp1.model.common.normalizer import LinearNormalizer
from mp1.model.mean.conditional_unet1d_meanflow_dis import ConditionalUnet1D
from mp1.model.mean.mask_generator import LowdimMaskGenerator
from mp1.model.vision.pointnet_cm import MP1TemporalConvEncoder
from mp1.policy.base_policy import BasePolicy


class MP1_CM(BasePolicy):

    def __init__(
        self,
        shape_meta: dict,
        horizon,
        n_action_steps,
        n_obs_steps,
        dataset_obs_steps=None,
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
        rgb_encoder_cfg=None,
        short_term_memory=None,
        causal_memory=None,
        **kwargs,
    ):
        super().__init__()

        self.condition_type = condition_type
        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        self.kwargs = kwargs

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

        self.short_term_memory_cfg = short_term_memory or {}
        self.use_short_term_memory = bool(self.short_term_memory_cfg.get("enabled", False))

        self.obs_encoder = MP1TemporalConvEncoder(
            observation_space=obs_dict,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
            rgb_encoder_cfg=rgb_encoder_cfg,
            short_term_memory=short_term_memory,
        )
        self.causal_obs_encoder = copy.deepcopy(self.obs_encoder)

        self.obs_feature_dim = self.obs_encoder.output_shape()
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.num_inference_steps = num_inference_steps
        self._inference_action_history: List[torch.Tensor] = []

        if not obs_as_global_cond:
            raise ValueError("MP1_CM only supports obs_as_global_cond=True.")

        if "cross_attention" in self.condition_type:
            global_cond_dim = self.obs_feature_dim + 0
        else:
            global_cond_dim = self.obs_feature_dim * n_obs_steps

        cfg = causal_memory or {}
        self.dataset_obs_steps = int(dataset_obs_steps if dataset_obs_steps is not None else n_obs_steps)
        if self.dataset_obs_steps < n_obs_steps:
            raise ValueError(
                f"dataset_obs_steps must be >= n_obs_steps, got "
                f"{self.dataset_obs_steps} and {n_obs_steps}"
            )

        self.causal_history_steps = int(cfg.get("history_obs_steps", self.dataset_obs_steps))
        if self.causal_history_steps < 2:
            raise ValueError(f"causal_history_steps must be >= 2, got {self.causal_history_steps}")
        if self.causal_history_steps > self.dataset_obs_steps:
            raise ValueError(
                f"causal_history_steps must be <= dataset_obs_steps, got "
                f"{self.causal_history_steps} and {self.dataset_obs_steps}"
            )

        state_shape = shape_meta["obs"]["agent_pos"]["shape"]
        self.state_dim = int(state_shape[0])
        aligned_action_slice = cfg.get("aligned_action_slice")
        if aligned_action_slice is None:
            self.aligned_action_start = 0
            self.aligned_action_end = self.action_dim
        else:
            if len(aligned_action_slice) != 2:
                raise ValueError(
                    f"aligned_action_slice must have 2 entries, got {aligned_action_slice}"
                )
            self.aligned_action_start = int(aligned_action_slice[0])
            self.aligned_action_end = int(aligned_action_slice[1])

        if self.aligned_action_start < 0 or self.aligned_action_end > self.action_dim:
            raise ValueError(
                f"aligned_action_slice out of bounds for action_dim={self.action_dim}: "
                f"{self.aligned_action_start}:{self.aligned_action_end}"
            )
        if self.aligned_action_end <= self.aligned_action_start:
            raise ValueError(
                f"aligned_action_slice must satisfy end > start, got "
                f"{self.aligned_action_start}:{self.aligned_action_end}"
            )
        self.aligned_action_dim = self.aligned_action_end - self.aligned_action_start

        self.latent_source = str(cfg.get("latent_source", "causal_encoder"))
        if self.latent_source not in {"causal_encoder", "base_encoder_detach"}:
            raise ValueError(
                "latent_source must be one of ['causal_encoder', 'base_encoder_detach'], "
                f"got {self.latent_source}"
            )

        self.target_feature_source = str(cfg.get("target_feature_source", "base_detach"))
        if self.target_feature_source not in {"latent_detach", "base_detach"}:
            raise ValueError(
                "target_feature_source must be one of ['latent_detach', 'base_detach'], "
                f"got {self.target_feature_source}"
            )

        action_embed_dim = int(cfg.get("action_embed_dim", 64))
        token_dim = int(cfg.get("token_dim", 64))
        memory_dim = int(cfg.get("memory_dim", 64))
        hidden_dim = int(cfg.get("hidden_dim", 128))
        aggregator_layers = int(cfg.get("aggregator_layers", 1))
        dropout = float(cfg.get("dropout", 0.1))
        memory_cond_dim = int(cfg.get("memory_cond_dim", self.obs_feature_dim))

        self.causal_memory = CausalMemoryEncoder(
            obs_dim=self.obs_feature_dim,
            action_dim=self.aligned_action_dim,
            action_embed_dim=action_embed_dim,
            token_dim=token_dim,
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
            aggregator_layers=aggregator_layers,
            dropout=dropout,
        )
        self.causal_memory_dim = memory_dim
        self.memory_cond_dim = memory_cond_dim

        self.memory_proj = nn.Sequential(
            nn.Linear(self.causal_memory_dim, max(memory_cond_dim, self.causal_memory_dim)),
            nn.LayerNorm(max(memory_cond_dim, self.causal_memory_dim)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(memory_cond_dim, self.causal_memory_dim), self.memory_cond_dim),
            nn.LayerNorm(self.memory_cond_dim),
        )
        nn.init.zeros_(self.memory_proj[4].weight)
        nn.init.zeros_(self.memory_proj[4].bias)

        self.dynamics_head = MLPBlock(
            input_dim=self.causal_memory_dim,
            hidden_dim=hidden_dim,
            output_dim=self.obs_feature_dim,
            dropout=dropout,
        )

        learnable_memory_scale = bool(cfg.get("learnable_memory_scale", True))
        memory_scale_init = float(cfg.get("memory_condition_scale", 1.0))
        if learnable_memory_scale:
            self.memory_condition_scale = nn.Parameter(
                torch.tensor(memory_scale_init, dtype=torch.float32)
            )
            self.register_buffer("fixed_memory_condition_scale", None, persistent=False)
        else:
            self.register_buffer(
                "fixed_memory_condition_scale",
                torch.tensor(memory_scale_init, dtype=torch.float32),
                persistent=False,
            )
            self.memory_condition_scale = None

        self.lambda_dyn = float(cfg.get("lambda_dyn", 1.0))
        self.lambda_dis = float(cfg.get("lambda_dis", 0.5))
        self.dispersive_tau = float(cfg.get("dispersive_tau", 1.0))

        self.policy_use_short_term_memory = bool(
            cfg.get("policy_use_short_term_memory", self.use_short_term_memory)
        )
        self.causal_use_short_term_memory = bool(cfg.get("causal_use_short_term_memory", False))
        self.target_use_short_term_memory = bool(
            cfg.get("target_use_short_term_memory", self.policy_use_short_term_memory)
        )

        if self.use_short_term_memory:
            num_memory_frames = int(self.short_term_memory_cfg.get("num_memory_frames", 1))
            required_steps = num_memory_frames + 1
            if self.causal_use_short_term_memory and self.causal_history_steps < required_steps:
                raise ValueError(
                    "causal short_term_memory requires history_obs_steps >= num_memory_frames + 1, "
                    f"got history_obs_steps={self.causal_history_steps}, "
                    f"num_memory_frames={num_memory_frames}"
                )
            if (
                self.target_feature_source == "base_detach"
                and self.target_use_short_term_memory
                and self.causal_history_steps < required_steps
            ):
                raise ValueError(
                    "target short_term_memory requires history_obs_steps >= num_memory_frames + 1, "
                    f"got history_obs_steps={self.causal_history_steps}, "
                    f"num_memory_frames={num_memory_frames}"
                )

        if "cross_attention" in self.condition_type:
            global_cond_dim = self.obs_feature_dim + self.memory_cond_dim
        else:
            global_cond_dim = self.obs_feature_dim * n_obs_steps + self.memory_cond_dim

        self.model = ConditionalUnet1D(
            input_dim=self.action_dim,
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

        self.flow_ratio = float(kwargs.get("flow_ratio", 0.5))
        self.time_dist = kwargs.get("time_dist", ["lognorm", -0.4, 1.0])
        cfg_scale = float(kwargs.get("cfg_scale", 2.0))
        self.w = cfg_scale
        self.kappa = kwargs.get("kappa", None)
        init_alpha = kwargs.get("alpha", 0.0)
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

        cprint(f"[MP1_CM] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[MP1_CM] pointnet_type: {self.pointnet_type}", "yellow")
        cprint(
            "[MP1_CM] "
            f"dataset_obs_steps={self.dataset_obs_steps}, "
            f"causal_history_steps={self.causal_history_steps}, "
            f"aligned_action_slice={self.aligned_action_start}:{self.aligned_action_end}, "
            f"aligned_action_dim={self.aligned_action_dim}, "
            f"memory_dim={self.causal_memory_dim}, "
            f"memory_cond_dim={self.memory_cond_dim}",
            "yellow",
        )
        cprint(
            "[MP1_CM] routing: "
            f"latent_source={self.latent_source}, "
            f"target_feature_source={self.target_feature_source}, "
            f"policy_short_term={self.policy_use_short_term_memory}, "
            f"causal_short_term={self.causal_use_short_term_memory}, "
            f"target_short_term={self.target_use_short_term_memory}",
            "yellow",
        )
        cprint(
            "[MP1_CM] loss/config: "
            f"lambda_dyn={self.lambda_dyn}, "
            f"lambda_dis={self.lambda_dis}, "
            f"memory_condition_scale={self._memory_scale().detach().item():.4f}",
            "yellow",
        )

        print_params(self)

    def _memory_scale(self) -> torch.Tensor:
        if self.memory_condition_scale is not None:
            return self.memory_condition_scale
        return self.fixed_memory_condition_scale

    def _strip_unused_obs(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs = dict(obs_dict)
        if not self.use_pc_color and "point_cloud" in obs:
            obs["point_cloud"] = obs["point_cloud"][..., :3]
        return obs

    def _extract_available_obs_window(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        value = next(iter(obs_dict.values()))
        total_steps = value.shape[1]
        if total_steps < self.dataset_obs_steps:
            raise ValueError(
                f"Need at least {self.dataset_obs_steps} observation frames, got {total_steps}"
            )
        return dict_apply(obs_dict, lambda x: x[:, :self.dataset_obs_steps, ...])

    def _extract_policy_obs(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        value = next(iter(obs_dict.values()))
        total_steps = value.shape[1]
        if total_steps < self.n_obs_steps:
            raise ValueError(f"Need at least {self.n_obs_steps} observation frames, got {total_steps}")
        return dict_apply(obs_dict, lambda x: x[:, -self.n_obs_steps:, ...])

    def _extract_history_obs(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        value = next(iter(obs_dict.values()))
        total_steps = value.shape[1]
        if total_steps < self.causal_history_steps:
            raise ValueError(
                f"Need at least {self.causal_history_steps} observation frames, got {total_steps}"
            )
        return dict_apply(obs_dict, lambda x: x[:, -self.causal_history_steps:, ...])

    def reset(self):
        self._inference_action_history = []

    def _extract_aligned_actions_from_batch(self, nactions: torch.Tensor) -> torch.Tensor:
        required_steps = self.causal_history_steps - 1
        if nactions.ndim != 3:
            raise ValueError(f"nactions must be [B, T, A], got {tuple(nactions.shape)}")
        if nactions.shape[1] < required_steps:
            raise ValueError(
                f"Need at least {required_steps} action steps for causal memory, got {nactions.shape[1]}"
            )
        return nactions[:, :required_steps, self.aligned_action_start:self.aligned_action_end]

    def _get_inference_aligned_actions(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        required_steps = self.causal_history_steps - 1
        aligned_actions = torch.zeros(
            batch_size,
            required_steps,
            self.aligned_action_dim,
            device=device,
            dtype=dtype,
        )
        if required_steps == 0:
            return aligned_actions

        available_steps = min(required_steps, len(self._inference_action_history))
        if available_steps > 0:
            if self._inference_action_history[-1].shape[0] != batch_size:
                self._inference_action_history = []
                return aligned_actions
            history = torch.stack(self._inference_action_history[-available_steps:], dim=1)
            aligned_actions[:, -available_steps:, :] = history[
                :,
                :,
                self.aligned_action_start:self.aligned_action_end,
            ].to(device=device, dtype=dtype)

        return aligned_actions

    def _update_inference_action_history(self, executed_nactions: torch.Tensor):
        if executed_nactions.ndim != 3:
            raise ValueError(
                f"executed_nactions must be [B, T, A], got {tuple(executed_nactions.shape)}"
            )
        max_history = max(self.causal_history_steps - 1, 0)
        if executed_nactions.shape[1] == 0:
            return
        self._inference_action_history.append(executed_nactions[:, -1, :].detach())
        if max_history == 0:
            self._inference_action_history = []
        elif len(self._inference_action_history) > max_history:
            self._inference_action_history = self._inference_action_history[-max_history:]

    def _encode_policy_features(self, policy_obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.obs_encoder(
            policy_obs,
            apply_short_term_memory=self.policy_use_short_term_memory,
        )

    def _encode_latent_history(self, causal_obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.latent_source == "causal_encoder":
            return self.causal_obs_encoder(
                causal_obs,
                apply_short_term_memory=self.causal_use_short_term_memory,
            )

        with torch.no_grad():
            return self.obs_encoder(
                causal_obs,
                apply_short_term_memory=self.causal_use_short_term_memory,
            )

    def _encode_target_history(
        self,
        history_obs: Dict[str, torch.Tensor],
        policy_features: torch.Tensor,
        latent_history: torch.Tensor,
    ) -> torch.Tensor:
        if self.target_feature_source == "latent_detach":
            return latent_history.detach()

        if (
            self.causal_history_steps == self.n_obs_steps
            and self.target_use_short_term_memory == self.policy_use_short_term_memory
        ):
            return policy_features[:, -self.causal_history_steps:, :].detach()

        with torch.no_grad():
            return self.obs_encoder(
                history_obs,
                apply_short_term_memory=self.target_use_short_term_memory,
            )

    def _predict_next_latent(
        self,
        memory_states: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, time_steps = memory_states.shape[:2]
        z_pred_next = self.dynamics_head(memory_states.reshape(batch_size * time_steps, -1))
        return z_pred_next.reshape(batch_size, time_steps, -1)

    def _build_global_condition(
        self,
        nobs: Dict[str, torch.Tensor],
        aligned_actions: torch.Tensor,
        compute_auxiliary: bool = False,
    ) -> Dict[str, torch.Tensor]:
        available_obs = self._extract_available_obs_window(nobs)
        policy_obs = self._extract_policy_obs(available_obs)
        history_obs = self._extract_history_obs(available_obs)
        causal_obs = history_obs

        policy_features = self._encode_policy_features(policy_obs)
        latent_history = self._encode_latent_history(causal_obs)
        causal_outputs = self.causal_memory(latent_history, aligned_actions)

        memory_cond = self.memory_proj(causal_outputs["memory"])
        memory_cond_scaled = self._memory_scale() * memory_cond

        if "cross_attention" in self.condition_type:
            memory_tokens = memory_cond_scaled.unsqueeze(1).expand(-1, self.n_obs_steps, -1)
            global_cond = torch.cat([policy_features, memory_tokens], dim=-1)
        else:
            base_cond = policy_features.reshape(policy_features.shape[0], -1)
            global_cond = torch.cat([base_cond, memory_cond_scaled], dim=-1)

        causal_outputs["policy_features"] = policy_features
        causal_outputs["latent_history"] = latent_history
        causal_outputs["memory_cond"] = memory_cond
        causal_outputs["memory_cond_scaled"] = memory_cond_scaled
        causal_outputs["global_cond"] = global_cond

        if compute_auxiliary:
            target_history = self._encode_target_history(
                history_obs=history_obs,
                policy_features=policy_features,
                latent_history=latent_history,
            )
            z_pred_next = self._predict_next_latent(
                memory_states=causal_outputs["memory_states"],
            )
            target_z_next = target_history[:, 1:, :].detach()
            loss_dyn = F.mse_loss(z_pred_next, target_z_next)

            causal_outputs["target_history"] = target_history
            causal_outputs["target_z_next"] = target_z_next
            causal_outputs["z_pred_next"] = z_pred_next
            causal_outputs["loss_dyn"] = loss_dyn

        return causal_outputs

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        nobs = self._strip_unused_obs(nobs)

        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        aligned_actions = self._get_inference_aligned_actions(
            batch_size=batch_size,
            device=value.device,
            dtype=value.dtype,
        )

        cond_outputs = self._build_global_condition(
            nobs,
            aligned_actions=aligned_actions,
            compute_auxiliary=False,
        )
        global_cond = cond_outputs["global_cond"]

        z = torch.randn(
            size=(batch_size, self.horizon, self.action_dim),
            dtype=global_cond.dtype,
            device=global_cond.device,
        )

        t = torch.ones((batch_size,), device=z.device, dtype=z.dtype)
        r = torch.zeros((batch_size,), device=z.device, dtype=z.dtype)

        self.model.eval()
        z = z - self.model(
            sample=z,
            timestep=t,
            local_cond=None,
            global_cond=global_cond,
            r=r,
            training=False,
        )

        action_pred = self.normalizer["action"].unnormalize(z)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        self._update_inference_action_history(z[:, start:end, :])

        return {
            "action": action,
            "action_pred": action_pred,
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch["obs"])
        nobs = self._strip_unused_obs(nobs)
        nactions = self.normalizer["action"].normalize(batch["action"])

        batch_size = nactions.shape[0]
        aligned_actions = self._extract_aligned_actions_from_batch(nactions)
        causal_action_pad_mask = batch.get("causal_action_pad_mask", None)
        if causal_action_pad_mask is not None and aligned_actions.shape[1] > 0:
            causal_action_pad_mask = causal_action_pad_mask.to(
                device=aligned_actions.device,
                dtype=torch.bool,
            )
            aligned_actions = aligned_actions.clone()
            aligned_actions[causal_action_pad_mask, 0, :] = 0.0
        cond_outputs = self._build_global_condition(
            nobs,
            aligned_actions=aligned_actions,
            compute_auxiliary=True,
        )
        global_cond = cond_outputs["global_cond"]

        x = nactions
        t, r = self.sample_t_r(batch_size, x.device)
        t = t.to(dtype=x.dtype)
        r = r.to(dtype=x.dtype)
        t_ = rearrange(t, "b -> b 1 1")
        r_ = rearrange(r, "b -> b 1 1")

        e = torch.randn_like(x)
        z = (1 - t_) * x + t_ * e
        v = e - x

        if self.w is not None and self.kappa is None:
            with torch.no_grad():
                u_t = self.model(
                    sample=z,
                    timestep=t,
                    global_cond=global_cond,
                    r=t - r,
                    training=False,
                )
            v_hat = self.w * v + (1 - self.w) * u_t
        elif self.w is not None and self.kappa > 0:
            uncond = torch.ones_like(global_cond)
            u_uncond = self.model(
                sample=z,
                timestep=t,
                global_cond=uncond,
                r=t - r,
                training=False,
            )
            u_cond = self.model(
                sample=z,
                timestep=t,
                global_cond=global_cond,
                r=t - r,
                training=False,
            )
            v_hat = self.w * v + (1 - self.w - self.kappa) * u_uncond + self.kappa * u_cond
        else:
            v_hat = v

        model_partial = partial(self.model, global_cond=global_cond)
        pred, dudt = torch.autograd.functional.jvp(
            lambda sample, time, ref: model_partial(
                sample=sample,
                timestep=time,
                r=time - ref,
                training=True,
            ),
            (z, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
            create_graph=True,
        )

        if self.alpha.detach().item() == 0.0:
            coef = t_ - r_
        else:
            t_safe = torch.clamp(t_, min=1e-5)
            coef = (t_safe - r_ * (r_ / t_safe) ** self.alpha) / (self.alpha + 1.0)

        velocity_pred, dis_feats = pred
        velocity_dudt = dudt[0]

        u_tgt = v_hat - coef * velocity_dudt
        error = velocity_pred - stopgrad(u_tgt)
        loss_bc = adaptive_l2_loss(error)
        loss_dyn = cond_outputs["loss_dyn"]
        loss_dis = 0.0
        for feat in dis_feats:
            loss_dis = loss_dis + self.dispersive_loss(feat, tau=self.dispersive_tau)

        loss = loss_bc + self.lambda_dyn * loss_dyn + self.lambda_dis * loss_dis

        mse_val = (stopgrad(error) ** 2).mean()
        loss_dict = {
            "bc_loss": loss_bc.item(),
            "dyn_loss": loss_dyn.item(),
            "dis_loss": loss_dis.item(),
            "total_loss": loss.item(),
            "mse_val": mse_val.item(),
            "causal_memory_norm": cond_outputs["memory"].norm(dim=-1).mean().item(),
            "causal_token_norm": cond_outputs["causal_tokens"].norm(dim=-1).mean().item(),
            "latent_history_norm": cond_outputs["latent_history"].norm(dim=-1).mean().item(),
            "memory_cond_norm": cond_outputs["memory_cond"].norm(dim=-1).mean().item(),
            "memory_cond_scaled_norm": cond_outputs["memory_cond_scaled"].norm(dim=-1).mean().item(),
            "memory_scale": self._memory_scale().detach().item(),
            "target_latent_norm": cond_outputs["target_z_next"].norm(dim=-1).mean().item(),
            "pred_latent_norm": cond_outputs["z_pred_next"].norm(dim=-1).mean().item(),
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

    def dispersive_loss(self, z: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        dist_matrix = torch.cdist(z, z, p=2) ** 2
        dist_matrix = dist_matrix / torch.max(dist_matrix)
        exp_term = torch.exp(-dist_matrix / tau)
        mean_exp = torch.mean(exp_term)
        loss = torch.log(mean_exp)
        return loss


def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    delta_sq = torch.mean(error ** 2, dim=tuple(range(1, error.ndim)))
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    return (stopgrad(w) * delta_sq).mean()
