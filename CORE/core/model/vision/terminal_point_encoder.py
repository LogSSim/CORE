import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model.vision.pointnet_extractor import PointNetEncoderXYZ, PointNetEncoderXYZRGB


class _AttrDict(dict):

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _clone_cfg(pointcloud_encoder_cfg):
    if pointcloud_encoder_cfg is None:
        return _AttrDict(
            in_channels=3,
            out_channels=128,
            use_layernorm=True,
            final_norm="layernorm",
            normal_channel=False,
        )
    if isinstance(pointcloud_encoder_cfg, dict):
        return _AttrDict(copy.deepcopy(pointcloud_encoder_cfg))
    try:
        return _AttrDict(copy.deepcopy(dict(pointcloud_encoder_cfg)))
    except Exception:
        return copy.deepcopy(pointcloud_encoder_cfg)


class TerminalPointEncoder(nn.Module):
    """Point-cloud-only encoder for terminal representation learning."""

    def __init__(
        self,
        pointcloud_encoder_cfg=None,
        feat_dim=128,
        proj_dim=128,
        use_pc_color=False,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.proj_dim = int(proj_dim)
        self.use_pc_color = bool(use_pc_color)

        encoder_cfg = _clone_cfg(pointcloud_encoder_cfg)
        encoder_cfg.out_channels = self.feat_dim
        encoder_cfg.in_channels = 6 if self.use_pc_color else 3

        if self.use_pc_color:
            self.encoder = PointNetEncoderXYZRGB(**encoder_cfg)
        else:
            self.encoder = PointNetEncoderXYZ(**encoder_cfg)

        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.proj_dim),
        )
        self.ttg_head = nn.Linear(self.feat_dim, 1)
        self.term_head = nn.Linear(self.feat_dim, 1)

    def encode_feat(self, pc: torch.Tensor) -> torch.Tensor:
        """Encode point clouds.

        Args:
            pc: [B, N, C]. C may be 3 or 6; when use_pc_color=False only xyz is used.
        Returns:
            feat: [B, feat_dim]
        """
        if pc.ndim != 3:
            raise ValueError(f"TerminalPointEncoder expects [B, N, C], got {tuple(pc.shape)}")
        if not self.use_pc_color:
            pc = pc[..., :3]
        return self.encoder(pc)

    def encode_proj(self, pc: torch.Tensor):
        feat = self.encode_feat(pc)
        z = F.normalize(self.proj(feat), dim=-1)
        return feat, z
