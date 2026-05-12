from functools import partial
from typing import Dict

import torch
import torch.nn.functional as F
from einops import rearrange
from termcolor import cprint

from mp1.common.model_util import print_params
from mp1.model.common.normalizer import LinearNormalizer
from mp1.model.mean.conditional_unet1d_test import ConditionalUnet1D
from mp1.policy.mp1_cm_1 import MP1_CM as BaseMP1CM
from mp1.policy.mp1_cm_1 import adaptive_l2_loss, stopgrad


class MP1_CM(BaseMP1CM):
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
        super().__init__(
            shape_meta=shape_meta,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            dataset_obs_steps=dataset_obs_steps,
            num_inference_steps=num_inference_steps,
            obs_as_global_cond=obs_as_global_cond,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
            encoder_output_dim=encoder_output_dim,
            crop_shape=crop_shape,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            rgb_encoder_cfg=rgb_encoder_cfg,
            short_term_memory=short_term_memory,
            causal_memory=causal_memory,
            **kwargs,
        )

        if "cross_attention" in self.condition_type:
            global_cond_dim = self.obs_feature_dim
        else:
            global_cond_dim = self.obs_feature_dim * self.n_obs_steps

        self.model = ConditionalUnet1D(
            input_dim=self.action_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            mid_cond_dim=self.memory_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        cprint(
            "[MP1_CM_TEST] routing: "
            "base obs -> global_cond, causal memory -> mid_cond only",
            "yellow",
        )
        cprint(
            "[MP1_CM_TEST] dims: "
            f"obs_feature_dim={self.obs_feature_dim}, "
            f"memory_cond_dim={self.memory_cond_dim}",
            "yellow",
        )
        print_params(self)

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
            global_cond = policy_features
        else:
            global_cond = policy_features.reshape(policy_features.shape[0], -1)

        causal_outputs["policy_features"] = policy_features
        causal_outputs["latent_history"] = latent_history
        causal_outputs["memory_cond"] = memory_cond
        causal_outputs["memory_cond_scaled"] = memory_cond_scaled
        causal_outputs["mid_cond"] = memory_cond_scaled
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
        mid_cond = cond_outputs["mid_cond"]

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
            mid_cond=mid_cond,
            r=r,
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
        mid_cond = cond_outputs["mid_cond"]

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
                    mid_cond=mid_cond,
                    r=t - r,
                )
            v_hat = self.w * v + (1 - self.w) * u_t
        elif self.w is not None and self.kappa > 0:
            uncond_global = torch.ones_like(global_cond)
            uncond_mid = torch.ones_like(mid_cond)
            u_uncond = self.model(
                sample=z,
                timestep=t,
                global_cond=uncond_global,
                mid_cond=uncond_mid,
                r=t - r,
            )
            u_cond = self.model(
                sample=z,
                timestep=t,
                global_cond=global_cond,
                mid_cond=mid_cond,
                r=t - r,
            )
            v_hat = self.w * v + (1 - self.w - self.kappa) * u_uncond + self.kappa * u_cond
        else:
            v_hat = v

        model_partial = partial(self.model, global_cond=global_cond, mid_cond=mid_cond)
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
        loss_bc = adaptive_l2_loss(error)
        loss_dyn = cond_outputs["loss_dyn"]
        loss = loss_bc + self.lambda_dyn * loss_dyn

        mse_val = (stopgrad(error) ** 2).mean()
        loss_dict = {
            "bc_loss": loss_bc.item(),
            "dyn_loss": loss_dyn.item(),
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
