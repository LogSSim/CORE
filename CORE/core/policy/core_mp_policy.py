"""CORE policy with the MP backbone.

Stage 1 learns terminal representations with bidirectional contrastive and
auxiliary temporal losses. Stage 3 injects the shared goal prototype built in
stage 2 into the MP policy backbone.
"""

import os
from pathlib import Path

import dill
import numpy as np
import torch
import torch.nn.functional as F
from termcolor import cprint

from core.policy.meanpolicy import Meanpolicy
from core.policy.core_auxiliary import TerminalAuxMixin

STAGE3_GOAL_FEAT_MODES = ("full", "curr_goal", "goal_only")


def goal_feature_dim_for_mode(term_proj_dim: int, mode: str) -> int:
    if mode == "full":
        return int(term_proj_dim) * 3
    if mode == "curr_goal":
        return int(term_proj_dim) * 2
    if mode == "goal_only":
        return int(term_proj_dim)
    raise ValueError(
        f"goal_feat_mode must be one of {STAGE3_GOAL_FEAT_MODES}, got {mode}"
    )


class COREMPPolicy(TerminalAuxMixin, Meanpolicy):
    def __init__(
        self,
        *args,
        terminal_aux=None,
        goal_conditioning=None,
        pointcloud_encoder_cfg=None,
        use_pc_color=False,
        encoder_output_dim=256,
        policy_stage=1,
        **kwargs,
    ):
        self.terminal_aux_cfg = terminal_aux or {}
        self.goal_conditioning_cfg = goal_conditioning or {}
        self.use_terminal_aux = bool(self.terminal_aux_cfg.get("enabled", False))
        self.use_goal_conditioning = bool(self.goal_conditioning_cfg.get("enabled", False))
        self.goal_selection = str(self.goal_conditioning_cfg.get("selection", "common"))
        self.soft_nearest_temperature = float(self.goal_conditioning_cfg.get("soft_nearest_temperature", 0.1))
        self.term_proj_dim = int(
            self.terminal_aux_cfg.get(
                "projection_dim",
                self.goal_conditioning_cfg.get("projection_dim", encoder_output_dim),
            )
        )
        self.goal_feat_mode = str(
            self.goal_conditioning_cfg.get("goal_feat_mode", "full")
        )
        if self.goal_feat_mode not in STAGE3_GOAL_FEAT_MODES:
            raise ValueError(
                f"goal_conditioning.goal_feat_mode must be one of {STAGE3_GOAL_FEAT_MODES}, "
                f"got {self.goal_feat_mode}"
            )
        self.goal_feature_dim = goal_feature_dim_for_mode(
            self.term_proj_dim, self.goal_feat_mode
        )
        if self.goal_feature_dim % 2 != 0:
            raise ValueError("COREMPPolicy requires an even goal_feature_dim.")
        if self.use_goal_conditioning and self.goal_selection not in {"fixed", "common", "soft_nearest"}:
            raise ValueError(
                "goal_conditioning.selection must be fixed/common/soft_nearest, "
                f"got {self.goal_selection}"
            )
        if self.use_goal_conditioning and self.goal_selection == "fixed":
            if self.goal_conditioning_cfg.get("goal_index", None) is None:
                raise ValueError('goal_conditioning.selection="fixed" requires explicit goal_index.')

        effective_stage = 3 if self.use_goal_conditioning else int(policy_stage)
        super().__init__(
            *args,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            encoder_output_dim=encoder_output_dim,
            policy_stage=effective_stage,
            final_feature_dim=self.goal_feature_dim // 2,
            final_state_feature_path=None,
            final_condition_mode="mean",
            **kwargs,
        )

        self.freeze_terminal_encoder = bool(self.goal_conditioning_cfg.get("freeze_terminal_encoder", True))
        self._init_terminal_aux(
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            terminal_window=int(self.terminal_aux_cfg.get("terminal_window", 8)),
            neg_count=int(self.terminal_aux_cfg.get("neg_count", 4)),
            lambda_nce=float(self.terminal_aux_cfg.get("lambda_nce", 1.0)),
            lambda_ttg=float(self.terminal_aux_cfg.get("lambda_ttg", 0.3)),
            lambda_term=float(self.terminal_aux_cfg.get("lambda_term", 0.1)),
            nce_tau=float(self.terminal_aux_cfg.get("temperature", 0.1)),
            term_feat_dim=int(self.terminal_aux_cfg.get("feat_dim", encoder_output_dim)),
            term_proj_dim=self.term_proj_dim,
            aux_loss_mode=str(self.terminal_aux_cfg.get("aux_loss_mode", "all")),
        )

        terminal_ckpt = self.goal_conditioning_cfg.get("terminal_encoder_ckpt", None)
        if terminal_ckpt:
            self.load_terminal_encoder_checkpoint(terminal_ckpt)
        if self.use_goal_conditioning and self.freeze_terminal_encoder:
            self.term_module.requires_grad_(False)

        goal_prototypes = torch.empty(0, self.term_proj_dim, dtype=torch.float32)
        common_prototype = torch.empty(0, dtype=torch.float32)
        if self.use_goal_conditioning:
            if self.goal_selection in {"fixed", "soft_nearest"}:
                goal_prototypes = self._load_goal_prototypes(self.goal_conditioning_cfg.get("prototype_path", None))
            if self.goal_selection == "common":
                common_prototype = self._load_common_prototype(
                    self.goal_conditioning_cfg.get("common_prototype_path", None)
                )
        self.register_buffer("goal_prototypes", goal_prototypes, persistent=False)
        self.register_buffer("common_prototype", common_prototype, persistent=False)
        cprint(
            f"[CORE-MP] terminal_aux={self.use_terminal_aux}, "
            f"aux_loss_mode={getattr(self, 'aux_loss_mode', 'all')}, "
            f"goal_conditioning={self.use_goal_conditioning}, "
            f"goal_feat_mode={self.goal_feat_mode}, selection={self.goal_selection}",
            "yellow",
        )

    def _resolve_path(self, path):
        p = Path(os.path.expanduser(str(path)))
        if not p.is_absolute():
            p = Path.cwd() / p
        if p.is_dir() and (p / "checkpoints" / "latest.ckpt").is_file():
            p = p / "checkpoints" / "latest.ckpt"
        return p

    def _load_goal_prototypes(self, prototype_path):
        if prototype_path is None:
            raise ValueError('selection="fixed"/"soft_nearest" requires prototype_path.')
        path = self._resolve_path(prototype_path)
        if path.is_dir():
            path = path / "prototypes.npy"
        prototypes = np.load(path).astype(np.float32)
        if prototypes.ndim == 1:
            prototypes = prototypes[None, :]
        prototypes = prototypes / np.clip(np.linalg.norm(prototypes, axis=-1, keepdims=True), 1e-8, None)
        return torch.from_numpy(prototypes)

    def _load_common_prototype(self, common_prototype_path):
        if common_prototype_path is None:
            raise ValueError('selection="common" requires common_prototype_path.')
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
        prefix = "term_module."
        terminal_state = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if len(terminal_state) == 0:
            prefix = "terminal_encoder."
            terminal_state = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if len(terminal_state) == 0:
            terminal_state = state_dict
        missing, unexpected = self.term_module.load_state_dict(terminal_state, strict=False)
        cprint(
            f"[CORE-MP] loaded terminal encoder {path}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}",
            "yellow",
        )

    def _terminal_z_from_normalized_pc(self, pc):
        if not self.use_pc_color:
            pc = pc[..., :3]
        leading_shape = pc.shape[:-2]
        flat_pc = pc.reshape(-1, *pc.shape[-2:])
        _, z = self.term_module.encode_proj(flat_pc)
        return z.reshape(*leading_shape, -1)

    def encode_terminal_z(self, point_cloud):
        pc = self._normalize_terminal_point_cloud(point_cloud)
        return self._terminal_z_from_normalized_pc(pc)

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

    def _apply_final_state_condition(self, global_cond, nobs, final_state_condition=None):
        if not self.use_goal_conditioning:
            return global_cond
        if global_cond.ndim != 2:
            raise ValueError(f"terminal-goal condition expects [B, D] global_cond, got {global_cond.shape}")
        current_pc = nobs["point_cloud"][:, self.n_obs_steps - 1, ...]
        if self.freeze_terminal_encoder:
            self.term_module.eval()
            with torch.no_grad():
                z_curr = self._terminal_z_from_normalized_pc(current_pc)
        else:
            z_curr = self._terminal_z_from_normalized_pc(current_pc)
        z_goal = self._select_goal_prototype(z_curr)
        goal_feat = self._build_goal_feature(z_curr, z_goal)
        return torch.cat([global_cond, goal_feat], dim=-1)

    def _build_goal_feature(self, z_curr: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        if self.goal_feat_mode == "full":
            return torch.cat([z_curr, z_goal, z_goal - z_curr], dim=-1)
        if self.goal_feat_mode == "curr_goal":
            return torch.cat([z_curr, z_goal], dim=-1)
        if self.goal_feat_mode == "goal_only":
            return z_goal
        raise ValueError(f"Unsupported goal_feat_mode: {self.goal_feat_mode}")

    def compute_loss(self, batch):
        base_loss, loss_dict = Meanpolicy.compute_loss(self, batch)
        if not self.use_terminal_aux:
            return base_loss, loss_dict
        aux_loss, aux_metrics = self._terminal_aux_loss(batch)
        total_loss = base_loss + aux_loss
        loss_dict = dict(loss_dict)
        loss_dict["base_loss"] = base_loss.item()
        loss_dict.update(aux_metrics)
        loss_dict["bc_loss"] = total_loss.item()
        return total_loss, loss_dict
