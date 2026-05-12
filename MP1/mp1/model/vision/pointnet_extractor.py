import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
import pdb


class _AttrDict(dict):

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _clone_cfg(cfg):
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return _AttrDict(copy.deepcopy(cfg))
    try:
        return _AttrDict(copy.deepcopy(dict(cfg)))
    except Exception:
        return copy.deepcopy(cfg)


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class SimpleRGBEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        output_dim: int = 64,
        hidden_dims=(32, 64, 128),
        kernel_size: int = 3,
        use_groupnorm: bool = False,
    ):
        super().__init__()
        layers = []
        prev_channels = in_channels
        padding = kernel_size // 2
        for hidden_dim in hidden_dims:
            layers.append(
                nn.Conv2d(
                    prev_channels,
                    hidden_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                )
            )
            if use_groupnorm:
                num_groups = min(8, hidden_dim)
                while hidden_dim % num_groups != 0 and num_groups > 1:
                    num_groups -= 1
                layers.append(nn.GroupNorm(num_groups, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_channels = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(prev_channels, output_dim),
        )

    def _to_nchw(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"SimpleRGBEncoder expects [B, C, H, W] or [B, H, W, C], got {x.shape}")
        if x.shape[1] in (1, 3, 4):
            return x
        if x.shape[-1] in (1, 3, 4):
            return x.permute(0, 3, 1, 2).contiguous()
        raise ValueError(f"Cannot infer image channel dimension from shape {x.shape}")

    def _normalize_image(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            return x.float() / 255.0

        x = x.float()
        if x.max() > 1.5:
            x = x / 255.0
        elif x.min() < -0.1:
            x = (x + 1.0) * 0.5
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_nchw(x)
        x = self._normalize_image(x)
        x = self.backbone(x)
        return self.projection(x)


class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1024,
        block_channels=(64, 128, 256, 512),
        use_layernorm: bool = False,
        final_norm: str = "none",
        use_projection: bool = True,
        short_term_memory=None,
        **kwargs,
    ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = list(block_channels)
        cprint("pointnet use_layernorm: {}".format(use_layernorm), "cyan")
        cprint("pointnet use_final_norm: {}".format(final_norm), "cyan")
        self.short_term_memory_cfg = short_term_memory or {}
        self.use_short_term_memory = bool(self.short_term_memory_cfg.get("enabled", False))
        self.num_memory_frames = int(self.short_term_memory_cfg.get("num_memory_frames", 1))

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )

        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(nn.Linear(block_channel[-1], out_channels),
                                                  nn.LayerNorm(out_channels))
        elif final_norm == "none":
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.dynamic_encoder = None
        self.gate_mlp = None
        self.fusion_mlp = None
        if self.use_short_term_memory:
            if self.num_memory_frames < 1:
                raise ValueError(f"num_memory_frames must be >= 1, got {self.num_memory_frames}")
            residual_hidden_dim = int(
                self.short_term_memory_cfg.get(
                    "residual_hidden_dim",
                    min(128, out_channels * self.num_memory_frames),
                )
            )
            gate_hidden_dim = int(
                self.short_term_memory_cfg.get(
                    "gate_hidden_dim",
                    min(128, out_channels),
                )
            )
            self.dynamic_encoder = nn.Sequential(
                nn.Linear(out_channels * self.num_memory_frames, residual_hidden_dim),
                nn.ReLU(),
                nn.Linear(residual_hidden_dim, out_channels),
            )
            self.gate_mlp = nn.Sequential(
                nn.Linear(out_channels * 2, gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, out_channels),
                nn.Sigmoid(),
            )
            self.fusion_mlp = nn.Sequential(
                nn.Linear(out_channels * 2, residual_hidden_dim),
                nn.ReLU(),
                nn.Linear(residual_hidden_dim, out_channels),
            )

    def _encode_single_frame(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x

    def _fuse_short_term_memory(self, frame_features: torch.Tensor) -> torch.Tensor:
        if frame_features.ndim != 3:
            raise ValueError(
                f"PointNetEncoderXYZRGB memory expects [B, T, D], got {tuple(frame_features.shape)}"
            )
        batch_size, time_steps, _ = frame_features.shape
        if time_steps < self.num_memory_frames + 1:
            raise ValueError(
                f"Need at least {self.num_memory_frames + 1} frames, got {time_steps}"
            )

        current_feat = frame_features[:, -1, :]
        memory_feats = frame_features[:, -1 - self.num_memory_frames:-1, :]
        residual_feats = current_feat.unsqueeze(1) - memory_feats
        dynamic_feat = self.dynamic_encoder(residual_feats.reshape(batch_size, -1))
        history_feat = memory_feats[:, -1, :]
        gate = self.gate_mlp(torch.cat([history_feat, dynamic_feat], dim=-1))
        fused_feat = gate * history_feat + (1.0 - gate) * dynamic_feat
        fused_feat = self.fusion_mlp(torch.cat([history_feat, fused_feat], dim=-1))

        fused_features = frame_features.clone()
        fused_features[:, -2, :] = fused_feat
        return fused_features

    def forward(self, x, apply_short_term_memory: bool = True):
        if x.ndim == 3:
            return self._encode_single_frame(x)
        if x.ndim != 4:
            raise ValueError(f"PointNetEncoderXYZRGB expects [B, N, C] or [B, T, N, C], got {x.shape}")

        batch_size, time_steps = x.shape[:2]
        frame_features = self._encode_single_frame(x.reshape(-1, *x.shape[2:])).reshape(batch_size, time_steps, -1)
        if self.use_short_term_memory and apply_short_term_memory:
            frame_features = self._fuse_short_term_memory(frame_features)
        return frame_features


class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud"""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1024,
        block_channels=(64, 128, 256),
        use_layernorm: bool = False,
        final_norm: str = "none",
        use_projection: bool = True,
        short_term_memory=None,
        **kwargs,
    ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = list(block_channels)
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), "cyan")
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), "cyan")

        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
        self.short_term_memory_cfg = short_term_memory or {}
        self.use_short_term_memory = bool(self.short_term_memory_cfg.get("enabled", False))
        self.num_memory_frames = int(self.short_term_memory_cfg.get("num_memory_frames", 1))

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]) if len(block_channel) > 3 else nn.Identity(),
            nn.LayerNorm(block_channel[3]) if use_layernorm and len(block_channel) > 3 else nn.Identity(),
            nn.ReLU() if len(block_channel) > 3 else nn.Identity(),
        )

        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(nn.Linear(block_channel[-1], out_channels),
                                                  nn.LayerNorm(out_channels))
        elif final_norm == "none":
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")

        self.dynamic_encoder = None
        self.gate_mlp = None
        self.fusion_mlp = None
        if self.use_short_term_memory:
            if self.num_memory_frames < 1:
                raise ValueError(f"num_memory_frames must be >= 1, got {self.num_memory_frames}")
            residual_hidden_dim = int(
                self.short_term_memory_cfg.get(
                    "residual_hidden_dim",
                    min(128, out_channels * self.num_memory_frames),
                )
            )
            gate_hidden_dim = int(
                self.short_term_memory_cfg.get(
                    "gate_hidden_dim",
                    min(128, out_channels),
                )
            )
            self.dynamic_encoder = nn.Sequential(
                nn.Linear(out_channels * self.num_memory_frames, residual_hidden_dim),
                nn.ReLU(),
                nn.Linear(residual_hidden_dim, out_channels),
            )
            self.gate_mlp = nn.Sequential(
                nn.Linear(out_channels * 2, gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, out_channels),
                nn.Sigmoid(),
            )
            self.fusion_mlp = nn.Sequential(
                nn.Linear(out_channels * 2, residual_hidden_dim),
                nn.ReLU(),
                nn.Linear(residual_hidden_dim, out_channels),
            )
            self.fuse_scale = nn.Parameter(torch.tensor(0.0))

        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)

    def _encode_single_frame(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x

    def _fuse_short_term_memory(self, frame_features: torch.Tensor) -> torch.Tensor:
        if frame_features.ndim != 3:
            raise ValueError(
                f"PointNetEncoderXYZ memory expects [B, T, D], got {tuple(frame_features.shape)}"
            )

        batch_size, time_steps, _ = frame_features.shape
        if time_steps < self.num_memory_frames + 1:
            raise ValueError(
                f"Need at least {self.num_memory_frames + 1} frames, got {time_steps}"
            )

        current_feat = frame_features[:, -1, :]
        memory_feats = frame_features[:, -1 - self.num_memory_frames:-1, :]
        residual_feats = current_feat.unsqueeze(1) - memory_feats
        dynamic_feat = self.dynamic_encoder(residual_feats.reshape(batch_size, -1))

        gate = self.gate_mlp(torch.cat([current_feat, dynamic_feat], dim=-1))
        fused_feat = gate * current_feat + (1.0 - gate) * dynamic_feat
        fused_feat = self.fusion_mlp(torch.cat([current_feat, fused_feat], dim=-1))

        fused_features = frame_features.clone()
        fused_features[:, -1, :] = fused_feat
        return fused_features

    def forward(self, x, apply_short_term_memory: bool = True):
        if x.ndim == 3:
            return self._encode_single_frame(x)
        if x.ndim != 4:
            raise ValueError(f"PointNetEncoderXYZ expects [B, N, C] or [B, T, N, C], got {x.shape}")

        batch_size, time_steps = x.shape[:2]
        frame_features = self._encode_single_frame(x.reshape(-1, *x.shape[2:])).reshape(batch_size, time_steps, -1)
        if self.use_short_term_memory and apply_short_term_memory:
            frame_features = self._fuse_short_term_memory(frame_features)
        return frame_features

    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()

    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()


class MP1Encoder(nn.Module):

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
        super().__init__()
        self.imagination_key = "imagin_robot"
        self.state_key = "agent_pos"
        self.point_cloud_key = "point_cloud"
        self.rgb_image_key = "image"
        self.n_output_channels = out_channel

        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.use_state = self.state_key in observation_space.keys()
        self.use_rgb_image = self.rgb_image_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key] if self.use_state else None
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
        if self.use_rgb_image:
            self.rgb_shape = observation_space[self.rgb_image_key]
        else:
            self.rgb_shape = None

        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")
        cprint(f"[DP3Encoder] rgb image shape: {self.rgb_shape}", "yellow")

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        self.rgb_encoder = None
        pointcloud_encoder_cfg = _clone_cfg(pointcloud_encoder_cfg)
        if pointnet_type in {"pointnet", "mlp"}:
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(
                    **pointcloud_encoder_cfg,
                    short_term_memory=short_term_memory,
                )
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ(
                    **pointcloud_encoder_cfg,
                    short_term_memory=short_term_memory,
                )
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")

        self.state_mlp = None
        if self.use_state:
            if len(state_mlp_size) == 0:
                raise RuntimeError("State mlp size is empty")
            if len(state_mlp_size) == 1:
                net_arch = []
            else:
                net_arch = state_mlp_size[:-1]
            output_dim = state_mlp_size[-1]
            self.n_output_channels += output_dim
            self.state_mlp = nn.Sequential(
                *create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn)
            )

        if self.use_rgb_image:
            rgb_encoder_cfg = _clone_cfg(rgb_encoder_cfg) or _AttrDict()
            rgb_output_dim = int(rgb_encoder_cfg.get("output_dim", out_channel))
            default_rgb_in_channels = 3
            if len(self.rgb_shape) == 3:
                if self.rgb_shape[0] in (1, 3, 4):
                    default_rgb_in_channels = int(self.rgb_shape[0])
                elif self.rgb_shape[-1] in (1, 3, 4):
                    default_rgb_in_channels = int(self.rgb_shape[-1])
            rgb_in_channels = int(rgb_encoder_cfg.get("in_channels", default_rgb_in_channels))
            hidden_dims = tuple(rgb_encoder_cfg.get("hidden_dims", (32, 64, 128)))
            kernel_size = int(rgb_encoder_cfg.get("kernel_size", 3))
            use_groupnorm = bool(rgb_encoder_cfg.get("use_groupnorm", False))
            self.rgb_encoder = SimpleRGBEncoder(
                in_channels=rgb_in_channels,
                output_dim=rgb_output_dim,
                hidden_dims=hidden_dims,
                kernel_size=kernel_size,
                use_groupnorm=use_groupnorm,
            )
            self.n_output_channels += rgb_output_dim

        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")

    def _encode_points(self, points: torch.Tensor, apply_short_term_memory: bool = True) -> torch.Tensor:
        if points.ndim == 3:
            return self.extractor(points)
        if points.ndim != 4:
            raise ValueError(f"Unsupported point cloud shape {points.shape}")

        return self.extractor(points, apply_short_term_memory=apply_short_term_memory)

    def _encode_state(self, state: torch.Tensor) -> torch.Tensor:
        if self.state_mlp is None:
            raise RuntimeError("State encoder is not initialized")
        if state.ndim == 2:
            return self.state_mlp(state)
        if state.ndim != 3:
            raise ValueError(f"Unsupported state shape {state.shape}")

        batch_size, time_steps = state.shape[:2]
        return self.state_mlp(state.reshape(-1, state.shape[-1])).reshape(batch_size, time_steps, -1)

    def _encode_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        if self.rgb_encoder is None:
            raise RuntimeError("RGB encoder is not initialized")
        if rgb.ndim == 4:
            return self.rgb_encoder(rgb)
        if rgb.ndim != 5:
            raise ValueError(f"Unsupported rgb shape {rgb.shape}")

        batch_size, time_steps = rgb.shape[:2]
        return self.rgb_encoder(rgb.reshape(-1, *rgb.shape[2:])).reshape(batch_size, time_steps, -1)

    def forward(self, observations: Dict, apply_short_term_memory: bool = True) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        if points.ndim not in (3, 4):
            raise ValueError(f"point cloud shape: {points.shape}, length should be 3 or 4")
        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]]  # align the last dim
            points = torch.concat([points, img_points], dim=-2)

        pn_feat = self._encode_points(points, apply_short_term_memory=apply_short_term_memory)
        feat_list = [pn_feat]

        if self.state_mlp is not None:
            state = observations[self.state_key]
            state_feat = self._encode_state(state)
            feat_list.append(state_feat)

        if self.rgb_encoder is not None:
            rgb = observations[self.rgb_image_key]
            rgb_feat = self._encode_rgb(rgb)
            feat_list.append(rgb_feat)

        final_feat = torch.cat(feat_list, dim=-1)
        return final_feat

    def output_shape(self):
        return self.n_output_channels
