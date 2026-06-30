from typing import Union
import logging

import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from core.model.mean.conv1d_components import (
    Conv1dBlock,
    Downsample1d,
    Upsample1d,
)
from core.model.mean.positional_embedding import SinusoidalPosEmb


logger = logging.getLogger(__name__)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cond_dim,
        kernel_size=3,
        n_groups=8,
        condition_type="film",
    ):
        super().__init__()

        if condition_type != "film":
            raise NotImplementedError(
                "conditional_unet1d_test.py is specialized for FiLM conditioning only, "
                f"got condition_type={condition_type}"
            )

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    n_groups=n_groups,
                ),
                Conv1dBlock(
                    out_channels,
                    out_channels,
                    kernel_size,
                    n_groups=n_groups,
                ),
            ]
        )

        self.condition_type = "film"
        cond_channels = out_channels * 2
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange("batch t -> batch t 1"),
        )
        self.out_channels = out_channels
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond=None):
        out = self.blocks[0](x)
        if cond is not None:
            embed = self.cond_encoder(cond)
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        mid_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=3,
        n_groups=8,
        var_fc_hidden_dim=512,
        condition_type="film",
        use_down_condition=True,
        use_mid_condition=True,
        use_up_condition=True,
    ):
        super().__init__()
        if condition_type != "film":
            raise NotImplementedError(
                "conditional_unet1d_test.py only supports condition_type='film', "
                f"got {condition_type}"
            )
        self.condition_type = condition_type

        self.use_down_condition = use_down_condition
        self.use_mid_condition = use_mid_condition
        self.use_up_condition = use_up_condition

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        hidden_dim = var_fc_hidden_dim
        self.var_est = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim
        mid_block_cond_dim = cond_dim
        if mid_cond_dim is not None:
            mid_block_cond_dim += mid_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList(
                [
                    ConditionalResidualBlock1D(
                        dim_in,
                        dim_out,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        condition_type=condition_type,
                    ),
                    ConditionalResidualBlock1D(
                        dim_in,
                        dim_out,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        condition_type=condition_type,
                    ),
                ]
            )

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=mid_block_cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    condition_type=condition_type,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=mid_block_cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    condition_type=condition_type,
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            condition_type=condition_type,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            condition_type=condition_type,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            condition_type=condition_type,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            condition_type=condition_type,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.diffusion_step_encoder_rs = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        local_cond=None,
        global_cond=None,
        mid_cond=None,
        r=None,
        **kwargs,
    ):
        sample = einops.rearrange(sample, "b h t -> b t h")

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0]).to(sample.device)
        timestep_embed = self.diffusion_step_encoder(timesteps)

        rs = r
        if not torch.is_tensor(rs):
            rs = torch.tensor([rs], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(rs) and len(rs.shape) == 0:
            rs = rs[None].to(sample.device)
        rs = rs.expand(sample.shape[0]).to(sample.device)
        rs_embed = self.diffusion_step_encoder_rs(rs)

        timestep_embed = timestep_embed + rs_embed

        if global_cond is not None:
            if global_cond.ndim != 2:
                raise ValueError(
                    "FiLM conditioning expects global_cond with shape [B, C], "
                    f"got {tuple(global_cond.shape)}"
                )
            global_feature = torch.cat([timestep_embed, global_cond], dim=-1)
        else:
            global_feature = timestep_embed

        if mid_cond is not None:
            if mid_cond.ndim != 2:
                raise ValueError(
                    "FiLM conditioning expects mid_cond with shape [B, C], "
                    f"got {tuple(mid_cond.shape)}"
                )
            mid_feature = torch.cat([global_feature, mid_cond], dim=-1)
        else:
            mid_feature = global_feature

        h_local = []
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, "b h t -> b t h")
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            if self.use_down_condition:
                x = resnet(x, global_feature)
                if idx == 0 and len(h_local) > 0:
                    x = x + h_local[0]
                x = resnet2(x, global_feature)
            else:
                x = resnet(x)
                if idx == 0 and len(h_local) > 0:
                    x = x + h_local[0]
                x = resnet2(x)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            if self.use_mid_condition:
                x = mid_module(x, mid_feature)
            else:
                x = mid_module(x)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            if self.use_up_condition:
                x = resnet(x, global_feature)
                if idx == len(self.up_modules) and len(h_local) > 0:
                    x = x + h_local[1]
                x = resnet2(x, global_feature)
            else:
                x = resnet(x)
                if idx == len(self.up_modules) and len(h_local) > 0:
                    x = x + h_local[1]
                x = resnet2(x)
            x = upsample(x)

        x = self.final_conv(x)
        velocity_pred = einops.rearrange(x, "b t h -> b h t")
        return velocity_pred
