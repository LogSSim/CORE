"""DP3 terminal-goal v2 with explicit prototype selection modes."""

from typing import Dict

import copy
import os
from pathlib import Path

import dill
import numpy as np
import torch
import torch.nn.functional as F
from einops import reduce
from termcolor import cprint

from mp1.common.model_util import print_params
from mp1.common.pytorch_util import dict_apply
from mp1.model.common.normalizer import LinearNormalizer
from mp1.model.mean.conditional_unet1d import ConditionalUnet1D
from mp1.model.mean.mask_generator import LowdimMaskGenerator
from mp1.model.vision.pointnet_extractor import MP1Encoder
from mp1.model.vision.terminal_point_encoder import TerminalPointEncoder
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


class DP3TerminalGoalV2Policy(BasePolicy):
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
        terminal_aux=None,
        goal_conditioning=None,
        **kwargs,
    ):
        super().__init__()
        self.condition_type = condition_type
        self.use_pc_color = bool(use_pc_color)
        self.pointnet_type = pointnet_type
        self.terminal_aux_cfg = terminal_aux or {}
        self.goal_conditioning_cfg = goal_conditioning or {}
        self.use_terminal_aux = bool(self.terminal_aux_cfg.get("enabled", False))
        self.use_goal_conditioning = bool(self.goal_conditioning_cfg.get("enabled", False))
        self.goal_mode = str(self.goal_conditioning_cfg.get("mode", "concat"))
        self.goal_selection = str(self.goal_conditioning_cfg.get("selection", "common"))
        self.soft_nearest_temperature = float(self.goal_conditioning_cfg.get("soft_nearest_temperature", 0.1))
        self.freeze_terminal_encoder = bool(self.goal_conditioning_cfg.get("freeze_terminal_encoder", True))
        self.terminal_projection_dim = int(
            self.terminal_aux_cfg.get(
                "projection_dim",
                self.goal_conditioning_cfg.get("projection_dim", 128),
            )
        )
        self.goal_feature_dim = self.terminal_projection_dim * 3

        if not obs_as_global_cond:
            raise ValueError("DP3TerminalGoalV2Policy requires obs_as_global_cond=True.")
        if self.use_goal_conditioning and "cross_attention" in condition_type:
            raise ValueError("goal_conditioning supports vector global conditioning only.")
        if self.use_goal_conditioning and self.goal_selection not in {"fixed", "common", "soft_nearest"}:
            raise ValueError(
                "goal_conditioning.selection must be one of fixed/common/soft_nearest, "
                f"got {self.goal_selection}"
            )
        if self.use_goal_conditioning and self.goal_selection == "fixed":
            if self.goal_conditioning_cfg.get("goal_index", None) is None:
                raise ValueError('goal_conditioning.selection="fixed" requires explicit goal_index.')

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

        self.obs_encoder = MP1Encoder(
            observation_space=obs_dict,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=_clone_cfg(pointcloud_encoder_cfg),
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )

        obs_feature_dim = int(self.obs_encoder.output_shape())
        global_cond_dim = obs_feature_dim * int(n_obs_steps)
        if self.use_goal_conditioning and self.goal_mode == "concat":
            global_cond_dim += self.goal_feature_dim

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
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )

        self.terminal_encoder = None
        if self.use_terminal_aux or self.use_goal_conditioning:
            self.terminal_encoder = TerminalPointEncoder(
                pointcloud_encoder_cfg=_clone_cfg(pointcloud_encoder_cfg),
                feat_dim=int(self.terminal_aux_cfg.get("feat_dim", encoder_output_dim)),
                proj_dim=self.terminal_projection_dim,
                use_pc_color=bool(self.terminal_aux_cfg.get("use_pc_color", use_pc_color)),
            )

        terminal_ckpt = self.goal_conditioning_cfg.get(
            "terminal_encoder_ckpt",
            self.terminal_aux_cfg.get("checkpoint_path", None),
        )
        if terminal_ckpt and self.terminal_encoder is not None:
            self.load_terminal_encoder_checkpoint(terminal_ckpt)
        if self.use_goal_conditioning and self.freeze_terminal_encoder and self.terminal_encoder is not None:
            self.terminal_encoder.requires_grad_(False)

        goal_prototypes = torch.empty(0, self.terminal_projection_dim, dtype=torch.float32)
        common_prototype = torch.empty(0, dtype=torch.float32)
        if self.use_goal_conditioning:
            if self.goal_selection in {"fixed", "soft_nearest"}:
                goal_prototypes = self._load_goal_prototypes(
                    self.goal_conditioning_cfg.get("prototype_path", None)
                    or self.goal_conditioning_cfg.get("bank_path", None)
                )
            if self.goal_selection == "common":
                common_prototype = self._load_common_prototype(
                    self.goal_conditioning_cfg.get("common_prototype_path", None)
                )
        self.register_buffer("goal_prototypes", goal_prototypes, persistent=False)
        self.register_buffer("common_prototype", common_prototype, persistent=False)

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
        cprint(
            f"[DP3-TerminalGoalV2] terminal_aux={self.use_terminal_aux}, "
            f"goal_conditioning={self.use_goal_conditioning}, selection={self.goal_selection}",
            "yellow",
        )
        print_params(self)

    def _resolve_path(self, path):
        p = Path(os.path.expanduser(str(path)))
        if not p.is_absolute():
            p = Path.cwd() / p
        if p.is_dir() and (p / "checkpoints" / "latest.ckpt").is_file():
            p = p / "checkpoints" / "latest.ckpt"
        return p

    def _load_goal_prototypes(self, prototype_path):
        if prototype_path is None:
            raise ValueError(
                'goal_conditioning.selection="fixed"/"soft_nearest" requires prototype_path.'
            )
        path = self._resolve_path(prototype_path)
        if path.is_dir():
            path = path / "prototypes.npy"
        prototypes = np.load(path).astype(np.float32)
        if prototypes.ndim == 1:
            prototypes = prototypes[None, :]
        norm = np.linalg.norm(prototypes, axis=-1, keepdims=True)
        prototypes = prototypes / np.clip(norm, 1e-8, None)
        return torch.from_numpy(prototypes)

    def _load_common_prototype(self, common_prototype_path):
        if common_prototype_path is None:
            raise ValueError('goal_conditioning.selection="common" requires common_prototype_path.')
        path = self._resolve_path(common_prototype_path)
        if path.is_dir():
            path = path / "common_prototype.npy"
        common = np.load(path).astype(np.float32).reshape(-1)
        common = common / max(float(np.linalg.norm(common)), 1e-8)
        return torch.from_numpy(common)

    def load_terminal_encoder_checkpoint(self, checkpoint_path):
        path = self._resolve_path(checkpoint_path)
        payload = torch.load(path.open("rb"), map_location="cpu", pickle_module=dill)
        state_dict = payload["state_dicts"]["model"] if "state_dicts" in payload else payload
        prefix = "terminal_encoder."
        terminal_state = {
            k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)
        }
        if len(terminal_state) == 0:
            terminal_state = state_dict
        missing, unexpected = self.terminal_encoder.load_state_dict(terminal_state, strict=False)
        cprint(
            f"[DP3-TerminalGoalV2] loaded terminal encoder {path}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}",
            "yellow",
        )

    def _normalize_obs(self, obs_dict):
        obs_dict = {k: obs_dict[k] for k in self.obs_keys if k in obs_dict}
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color and "point_cloud" in nobs:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]
        return nobs

    def _prepare_terminal_point_cloud(self, point_cloud):
        pc = self.normalizer["point_cloud"].normalize(point_cloud)
        if self.terminal_encoder is not None and not self.terminal_encoder.use_pc_color:
            pc = pc[..., :3]
        return pc

    def encode_terminal_z(self, point_cloud):
        pc = self._prepare_terminal_point_cloud(point_cloud)
        return self._terminal_forward(pc)["z"]

    def _terminal_forward(self, pc):
        leading_shape = pc.shape[:-2]
        flat_pc = pc.reshape(-1, *pc.shape[-2:])
        feat, z = self.terminal_encoder.encode_proj(flat_pc)
        ttg_pred = torch.sigmoid(self.terminal_encoder.ttg_head(feat)).squeeze(-1)
        term_logit = self.terminal_encoder.term_head(feat).squeeze(-1)
        return {
            "feat": feat.reshape(*leading_shape, -1),
            "z": z.reshape(*leading_shape, -1),
            "ttg_pred": ttg_pred.reshape(*leading_shape),
            "term_logit": term_logit.reshape(*leading_shape),
        }

    def _encode_obs_condition(self, nobs):
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
        )
        obs_features = self.obs_encoder(this_nobs)
        return obs_features.reshape(batch_size, -1)

    def _select_goal_prototype(self, z_curr):
        if self.goal_selection == "common":
            z_goal = self.common_prototype.to(device=z_curr.device, dtype=z_curr.dtype)
            return z_goal.unsqueeze(0).expand(z_curr.shape[0], -1)

        prototypes = self.goal_prototypes.to(device=z_curr.device, dtype=z_curr.dtype)
        if prototypes.numel() == 0:
            raise RuntimeError(f"No prototypes loaded for goal selection mode {self.goal_selection}.")

        if self.goal_selection == "fixed":
            goal_index = int(self.goal_conditioning_cfg["goal_index"])
            if goal_index < 0 or goal_index >= prototypes.shape[0]:
                raise IndexError(f"goal_index={goal_index} out of range.")
            return prototypes[goal_index].unsqueeze(0).expand(z_curr.shape[0], -1)

        if self.goal_selection == "soft_nearest":
            temperature = max(self.soft_nearest_temperature, 1e-6)
            dist = torch.cdist(z_curr, prototypes, p=2).pow(2)
            weights = torch.softmax(-dist / temperature, dim=-1)
            return weights @ prototypes

        raise ValueError(f"Unsupported goal selection mode: {self.goal_selection}")

    def _apply_goal_conditioning(self, global_cond, nobs):
        if not self.use_goal_conditioning:
            return global_cond
        current_pc = nobs["point_cloud"][:, self.n_obs_steps - 1, ...]
        if self.freeze_terminal_encoder:
            self.terminal_encoder.eval()
            with torch.no_grad():
                z_curr = self._terminal_forward(current_pc)["z"]
        else:
            z_curr = self._terminal_forward(current_pc)["z"]
        z_goal = self._select_goal_prototype(z_curr)
        goal_feat = torch.cat([z_curr, z_goal, z_goal - z_curr], dim=-1)
        return torch.cat([global_cond, goal_feat], dim=-1)

    def conditional_sample(self, condition_data, condition_mask, local_cond=None, global_cond=None, generator=None, **kwargs):
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
            )
            if isinstance(model_output, tuple):
                model_output = model_output[0]
            trajectory = scheduler.step(model_output, t, trajectory).prev_sample
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self._normalize_obs(obs_dict)
        batch_size = next(iter(nobs.values())).shape[0]
        global_cond = self._apply_goal_conditioning(self._encode_obs_condition(nobs), nobs)
        cond_data = torch.zeros(
            size=(batch_size, self.horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        nsample = self.conditional_sample(cond_data, cond_mask, global_cond=global_cond, **self.kwargs)
        action_pred = self.normalizer["action"].unnormalize(nsample[..., : self.action_dim])
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        return {"action": action_pred[:, start:end], "action_pred": action_pred}

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_terminal_aux_loss(self, batch):
        required = [
            "term_anchor_point_cloud",
            "term_pos_point_cloud",
            "neg_point_clouds",
            "repr_point_cloud",
            "ttg_target",
            "term_label",
        ]
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"terminal_aux requires dataset terminal samples. Missing: {missing}")

        temperature = float(self.terminal_aux_cfg.get("temperature", 0.1))
        lambda_nce = float(self.terminal_aux_cfg.get("lambda_nce", 1.0))
        lambda_ttg = float(self.terminal_aux_cfg.get("lambda_ttg", 0.3))
        lambda_term = float(self.terminal_aux_cfg.get("lambda_term", 0.1))

        anchor_out = self._terminal_forward(self._prepare_terminal_point_cloud(batch["term_anchor_point_cloud"]))
        pos_out = self._terminal_forward(self._prepare_terminal_point_cloud(batch["term_pos_point_cloud"]))
        neg_out = self._terminal_forward(self._prepare_terminal_point_cloud(batch["neg_point_clouds"]))
        repr_out = self._terminal_forward(self._prepare_terminal_point_cloud(batch["repr_point_cloud"]))

        z_a = anchor_out["z"]
        z_p = pos_out["z"]
        z_n = neg_out["z"]
        pos_logits = torch.sum(z_a * z_p, dim=-1, keepdim=True) / temperature
        neg_logits = torch.einsum("bd,bkd->bk", z_a, z_n) / temperature
        labels = torch.zeros(z_a.shape[0], dtype=torch.long, device=z_a.device)
        loss_nce_a = F.cross_entropy(torch.cat([pos_logits, neg_logits], dim=-1), labels)

        pos_logits_p = torch.sum(z_p * z_a, dim=-1, keepdim=True) / temperature
        neg_logits_p = torch.einsum("bd,bkd->bk", z_p, z_n) / temperature
        loss_nce_p = F.cross_entropy(torch.cat([pos_logits_p, neg_logits_p], dim=-1), labels)
        loss_nce = 0.5 * (loss_nce_a + loss_nce_p)

        ttg_target = batch["ttg_target"].float().reshape_as(repr_out["ttg_pred"])
        term_label = batch["term_label"].float().reshape_as(repr_out["term_logit"])
        loss_ttg = F.smooth_l1_loss(repr_out["ttg_pred"], ttg_target)
        loss_term = F.binary_cross_entropy_with_logits(repr_out["term_logit"], term_label)
        total = lambda_nce * loss_nce + lambda_ttg * loss_ttg + lambda_term * loss_term
        return total, {
            "loss_nce": loss_nce.item(),
            "loss_ttg": loss_ttg.item(),
            "loss_term": loss_term.item(),
            "terminal_aux_loss": total.item(),
        }

    def compute_loss(self, batch):
        nobs = self._normalize_obs(batch["obs"])
        trajectory = self.normalizer["action"].normalize(batch["action"])
        cond_data = trajectory
        global_cond = self._apply_goal_conditioning(self._encode_obs_condition(nobs), nobs)
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

        pred = self.model(sample=noisy_trajectory, timestep=timesteps, local_cond=None, global_cond=global_cond)
        if isinstance(pred, tuple):
            pred = pred[0]
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean").mean()
        loss_dict = {"bc_loss": loss.item()}
        if self.use_terminal_aux:
            aux_loss, aux_dict = self.compute_terminal_aux_loss(batch)
            loss = loss + aux_loss
            loss_dict.update(aux_dict)
            loss_dict["total_loss"] = loss.item()
        return loss, loss_dict
