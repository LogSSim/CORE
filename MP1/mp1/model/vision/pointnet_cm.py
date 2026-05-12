import copy
from typing import Dict

import torch
import torch.nn as nn

from mp1.model.vision.pointnet_extractor import (
    MP1Encoder,
    PointNetEncoderXYZ,
    PointNetEncoderXYZRGB,
)


class _AttrDict(dict):

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _clone_pointcloud_cfg(pointcloud_encoder_cfg):
    if pointcloud_encoder_cfg is None:
        return None
    if isinstance(pointcloud_encoder_cfg, dict):
        return _AttrDict(copy.deepcopy(pointcloud_encoder_cfg))
    try:
        return _AttrDict(copy.deepcopy(dict(pointcloud_encoder_cfg)))
    except Exception:
        return copy.deepcopy(pointcloud_encoder_cfg)


class _TemporalConvShortTermMemoryMixin:

    def _init_temporal_conv_memory(self, out_channels: int, short_term_memory=None):
        self.short_term_memory_cfg = short_term_memory or {}
        self.use_short_term_memory = bool(self.short_term_memory_cfg.get("enabled", False))
        self.num_memory_frames = int(self.short_term_memory_cfg.get("num_memory_frames", 1))
        kernel_size = int(self.short_term_memory_cfg.get("kernel_size", 3))
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")

        self.temporal_conv = None
        if self.use_short_term_memory:
            if self.num_memory_frames != 1:
                raise ValueError(
                    f"This simplified short-term memory expects num_memory_frames=1, got {self.num_memory_frames}"
                )

            residual_hidden_dim = int(
                self.short_term_memory_cfg.get(
                    "residual_hidden_dim",
                    min(128, out_channels),
                )
            )
            gate_hidden_dim = int(
                self.short_term_memory_cfg.get(
                    "gate_hidden_dim",
                    min(128, out_channels),
                )
            )

            self.delta_norm = nn.LayerNorm(out_channels)

            self.dynamic_encoder = nn.Sequential(
                nn.Linear(out_channels, residual_hidden_dim),
                nn.LayerNorm(residual_hidden_dim),
                nn.ReLU(),
                nn.Linear(residual_hidden_dim, out_channels),
            )

            self.gate_mlp = nn.Sequential(
                nn.Linear(out_channels * 4, gate_hidden_dim),
                nn.LayerNorm(gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, out_channels),
                nn.Sigmoid(),
            )

            self.fusion_mlp = nn.Sequential(
                nn.Linear(out_channels * 3, residual_hidden_dim),
                nn.LayerNorm(residual_hidden_dim),
                nn.ReLU(),
                nn.Linear(residual_hidden_dim, out_channels),
            )

            nn.init.zeros_(self.fusion_mlp[-1].weight)
            nn.init.zeros_(self.fusion_mlp[-1].bias)

            self.fuse_scale = nn.Parameter(torch.tensor(0.0))

    def _fuse_short_term_memory(self, frame_features: torch.Tensor) -> torch.Tensor:
        if frame_features.ndim != 3:
            raise ValueError(
                f"PointNetEncoderXYZ memory expects [B, T, D], got {tuple(frame_features.shape)}"
            )

        batch_size, time_steps, feat_dim = frame_features.shape
        if time_steps < 2:
            raise ValueError(f"Need at least 2 frames, got {time_steps}")

        prev_feat = frame_features[:, -2, :]     # [B, D]
        current_feat = frame_features[:, -1, :]  # [B, D]

        delta_feat = self.delta_norm(current_feat - prev_feat)  # [B, D]
        dynamic_feat = self.dynamic_encoder(delta_feat)         # [B, D]

        gate_in = torch.cat(
            [current_feat, prev_feat, delta_feat, delta_feat.abs()],
            dim=-1
        )
        gate = self.gate_mlp(gate_in)                           # [B, D]

        dyn_delta = self.fusion_mlp(
            torch.cat([current_feat, prev_feat, dynamic_feat], dim=-1)
        )                                                       # [B, D]

        enhanced_current = current_feat + torch.tanh(self.fuse_scale) * gate * dyn_delta

        out = frame_features.clone()
        out[:, -1, :] = enhanced_current
        return out


class TemporalConvPointNetEncoderXYZ(_TemporalConvShortTermMemoryMixin, PointNetEncoderXYZ):

    def __init__(self, *args, short_term_memory=None, out_channels=1024, **kwargs):
        super().__init__(*args, short_term_memory=None, out_channels=out_channels, **kwargs)
        self._init_temporal_conv_memory(
            out_channels=out_channels,
            short_term_memory=short_term_memory,
        )


class TemporalConvPointNetEncoderXYZRGB(_TemporalConvShortTermMemoryMixin, PointNetEncoderXYZRGB):

    def __init__(self, *args, short_term_memory=None, out_channels=1024, **kwargs):
        super().__init__(*args, short_term_memory=None, out_channels=out_channels, **kwargs)
        self._init_temporal_conv_memory(
            out_channels=out_channels,
            short_term_memory=short_term_memory,
        )


class MP1TemporalConvEncoder(MP1Encoder):

    def __init__(
        self,
        observation_space: Dict,
        img_crop_shape=None,
        out_channel=256,
        state_mlp_size=(64, 64),
        state_mlp_activation_fn=nn.ReLU,
        pointcloud_encoder_cfg=None,
        use_pc_color=False,
        pointnet_type="pointnet",
        rgb_encoder_cfg=None,
        short_term_memory=None,
    ):
        base_encoder_cfg = _clone_pointcloud_cfg(pointcloud_encoder_cfg)
        super().__init__(
            observation_space=observation_space,
            img_crop_shape=img_crop_shape,
            out_channel=out_channel,
            state_mlp_size=state_mlp_size,
            state_mlp_activation_fn=state_mlp_activation_fn,
            pointcloud_encoder_cfg=base_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
            rgb_encoder_cfg=rgb_encoder_cfg,
            short_term_memory=None,
        )

        if pointnet_type not in {"pointnet", "mlp"}:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")

        encoder_cfg = _clone_pointcloud_cfg(pointcloud_encoder_cfg)
        if use_pc_color:
            encoder_cfg.in_channels = 6
            self.extractor = TemporalConvPointNetEncoderXYZRGB(
                **encoder_cfg,
                short_term_memory=short_term_memory,
            )
        else:
            encoder_cfg.in_channels = 3
            self.extractor = TemporalConvPointNetEncoderXYZ(
                **encoder_cfg,
                short_term_memory=short_term_memory,
            )
