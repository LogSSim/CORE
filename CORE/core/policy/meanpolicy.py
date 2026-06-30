import sys
sys.path.append('CORE/core')
from typing import Dict, Optional
import math
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint
import copy
import time
import dill
import numpy as np
from core.sde_lib import ConsistencyFM
from core.model.common.normalizer import LinearNormalizer
from core.policy.base_policy import BasePolicy
from core.model.mean.conditional_unet1d_meanflow import ConditionalUnet1D
from core.model.mean.mask_generator import LowdimMaskGenerator
from core.common.pytorch_util import dict_apply
from core.common.model_util import print_params
from core.model.vision.pointnet_extractor import COREPointNetEncoder
from functools import partial
import warnings
from einops import rearrange, reduce

warnings.filterwarnings("ignore")


def _resolve_data_path(path):
    if path is None:
        return None
    path = Path(os.path.expanduser(str(path)))
    if path.is_absolute() and path.exists():
        return str(path)
    if path.exists():
        return str(path)
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / path
    if candidate.exists():
        return str(candidate)
    return str(path)


def _resolve_checkpoint_path(path):
    path = Path(_resolve_data_path(path))
    if path.is_dir():
        for candidate in (path / "checkpoints" / "latest.ckpt", path / "latest.ckpt"):
            if candidate.is_file():
                return str(candidate)
        raise FileNotFoundError(
            f"Checkpoint directory does not contain checkpoints/latest.ckpt or latest.ckpt: {path}"
        )
    return str(path)

class Meanpolicy(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
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
            policy_stage=1,
            final_feature_dim=None,
            final_state_feature_path=None,
            final_condition_mode="mean",
            stage1_checkpoint_path=None,
            freeze_stage1_encoder=False,
            # parameters passed to step
            **kwargs):
        super().__init__()

        self.condition_type = condition_type
        self.policy_stage = int(policy_stage)
        if self.policy_stage not in (1, 3):
            raise ValueError(f"policy_stage must be 1 or 3, got {self.policy_stage}")
        if self.policy_stage == 3 and "cross_attention" in self.condition_type:
            raise ValueError("Meanpolicy final-state conditioning supports vector global conditioning only.")
        self.final_condition_mode = str(final_condition_mode)
        if self.final_condition_mode not in {"mean", "episode"}:
            raise ValueError("final_condition_mode must be 'mean' or 'episode'.")

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])


        obs_encoder = COREPointNetEncoder(observation_space=obs_dict,
                                                   img_crop_shape=crop_shape,
                                                out_channel=encoder_output_dim,
                                                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                                use_pc_color=use_pc_color,
                                                pointnet_type=pointnet_type,
                                                )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        final_feature_dim = int(encoder_output_dim if final_feature_dim is None else final_feature_dim)
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps
                if self.policy_stage == 3:
                    global_cond_dim += final_feature_dim * 2
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")



        model = ConditionalUnet1D(
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
        self.model = model

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.final_feature_dim = final_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        self.num_inference_steps = num_inference_steps

        self.flow_ratio=0.5
        self.time_dist=['lognorm', -0.4, 1.0]
        self.cfg_ratio=0.10
        cfg_scale=2.0
        # experimental
        self.cfg_uncond='u'
        self.w = cfg_scale
        self.kappa = None
        init_alpha = kwargs.get("alpha", 0.0)
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
        self._load_default_final_state_features(final_state_feature_path)
        if stage1_checkpoint_path is not None:
            self._load_stage1_encoder(stage1_checkpoint_path, freeze=freeze_stage1_encoder)
        print_params(self)
        

    def _load_default_final_state_features(self, final_state_feature_path):
        default_final = torch.zeros(self.final_feature_dim, dtype=torch.float32)
        default_cluster = torch.zeros(self.final_feature_dim, dtype=torch.float32)
        path = _resolve_data_path(final_state_feature_path)
        if path is not None and Path(path).exists():
            artifact = np.load(path)
            final_features = artifact["final_features"].astype(np.float32)
            if "cluster_features" in artifact:
                cluster_features = artifact["cluster_features"].astype(np.float32)
            else:
                labels = artifact["cluster_labels"].astype(np.int64)
                centers = artifact["cluster_centers"].astype(np.float32)
                cluster_features = centers[labels]
            default_final = torch.from_numpy(final_features.mean(axis=0))
            default_cluster = torch.from_numpy(cluster_features.mean(axis=0))

        self.register_buffer("default_final_feature", default_final, persistent=True)
        self.register_buffer("default_cluster_feature", default_cluster, persistent=True)

    def _load_stage1_encoder(self, checkpoint_path, freeze=False):
        checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
        payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill, map_location="cpu")
        model_state = payload["state_dicts"]["model"]
        encoder_prefix = "obs_encoder."
        encoder_state = {
            key[len(encoder_prefix):]: value
            for key, value in model_state.items()
            if key.startswith(encoder_prefix)
        }
        missing, unexpected = self.obs_encoder.load_state_dict(encoder_state, strict=False)
        cprint(
            f"[Meanpolicy-FinalState] loaded stage1 encoder from {checkpoint_path}; "
            f"missing={len(missing)}, unexpected={len(unexpected)}",
            "yellow",
        )
        if freeze:
            for param in self.obs_encoder.parameters():
                param.requires_grad_(False)

    def _encode_obs_condition(self, nobs: Dict[str, torch.Tensor]) -> torch.Tensor:
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
        )
        nobs_features = self.obs_encoder(this_nobs)
        if "cross_attention" in self.condition_type:
            return nobs_features.reshape(batch_size, self.n_obs_steps, -1)
        return nobs_features.reshape(batch_size, -1)

    def _encode_current_point_feature(self, nobs: Dict[str, torch.Tensor]) -> torch.Tensor:
        point_cloud = nobs["point_cloud"][:, self.n_obs_steps - 1, ...]
        if not self.use_pc_color:
            point_cloud = point_cloud[..., :3]
        return self.obs_encoder.extractor(point_cloud)

    def _condition_from_batch(self, batch_size, device, dtype, condition=None):
        use_episode_condition = self.final_condition_mode == "episode" and condition is not None
        if use_episode_condition:
            final_feature = condition["final_feature"].to(device=device, dtype=dtype)
            cluster_feature = condition["cluster_feature"].to(device=device, dtype=dtype)
            return final_feature, cluster_feature

        final_feature = self.default_final_feature.to(device=device, dtype=dtype)
        cluster_feature = self.default_cluster_feature.to(device=device, dtype=dtype)
        final_feature = final_feature.unsqueeze(0).expand(batch_size, -1)
        cluster_feature = cluster_feature.unsqueeze(0).expand(batch_size, -1)
        return final_feature, cluster_feature

    def _apply_final_state_condition(
        self,
        global_cond: torch.Tensor,
        nobs: Dict[str, torch.Tensor],
        final_state_condition: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if self.policy_stage == 1:
            return global_cond
        if global_cond.ndim != 2:
            raise ValueError(f"final-state condition expects [B, D] global_cond, got {global_cond.shape}")
        batch_size = global_cond.shape[0]
        final_feature, cluster_feature = self._condition_from_batch(
            batch_size=batch_size,
            device=global_cond.device,
            dtype=global_cond.dtype,
            condition=final_state_condition,
        )
        current_feature = self._encode_current_point_feature(nobs)
        feature_delta = current_feature - final_feature
        return torch.cat([global_cond, cluster_feature, feature_delta], dim=-1)


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        final_state_condition = obs_dict.get("final_state_condition", None)
        current_obs = {k: v for k, v in obs_dict.items() if k != "final_state_condition"}

        # normalize input
        nobs = self.normalizer.normalize(current_obs)
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_n_point_cloud = nobs['point_cloud']
        
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            global_cond = self._encode_obs_condition(nobs)
            global_cond = self._apply_final_state_condition(global_cond, nobs, final_state_condition)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        model = self.model
        model.eval()
        
        z = torch.randn(
            size=cond_data.shape, 
            dtype=cond_data.dtype,
            device=cond_data.device)

        t = torch.ones((cond_data.shape[0],), device=cond_data.device)
        r = torch.zeros((cond_data.shape[0],), device=cond_data.device)

        z = z - model(sample=z,
                    timestep=t, 
                    local_cond=local_cond, 
                    global_cond=global_cond, r=r)
           
        
        # unnormalize prediction
        naction_pred = z[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        # get prediction


        result = {
            'action': action,
            'action_pred': action_pred,
        }
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        final_state_condition = batch.get("final_state_condition", None)

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        x = trajectory
        
        device = cond_data.device
        
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            global_cond = self._encode_obs_condition(nobs)
            global_cond = self._apply_final_state_condition(global_cond, nobs, final_state_condition)
            # this_n_point_cloud = this_nobs['imagin_robot'].reshape(batch_size,-1, *this_nobs['imagin_robot'].shape[1:])
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()


        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)
        
        t, r = self.sample_t_r(batch_size, device)
        t_ = rearrange(t, "b -> b 1 1")
        r_ = rearrange(r, "b -> b 1 1")
        e = torch.randn_like(x)
        # x = normalize_to_neg1_1(x)
        z = (1 - t_) * x + t_ * e
        v = e - x

        if self.w is not None and self.kappa is None:
            with torch.no_grad():
                u_t = self.model(
                                sample=z, 
                                timestep=t, 
                                global_cond=global_cond, 
                                r=t)
            v_hat = self.w * v + (1 - self.w) * u_t
        
        elif self.w is not None and self.kappa > 0:
            uncond = torch.ones_like(global_cond)
            u_uncond = self.model(
                                sample=z, 
                                timestep=t, 
                                global_cond=uncond, 
                                r=t)
            u_cond = self.model(
                                sample=z, 
                                timestep=t, 
                                global_cond=global_cond, 
                                r=t)

            v_hat   = self.w * v + (1 - self.w - self.kappa) * u_uncond + self.kappa * u_cond 
               
        else:
            v_hat = v


        model_partial = partial(self.model, global_cond=global_cond)
        u, dudt = torch.autograd.functional.jvp(
            lambda z, t, r: model_partial(sample=z, timestep=t, r=t-r),
            # model,
            (z, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
            create_graph=True
        )

        # u_tgt = v_hat - (t_ - r_) * dudt
        if getattr(self, 'alpha', 0.0) == 0.0:
            coef = t_ - r_
        else:
            # clamp 防止 t_ 为 0 时产生除零错误
            t_safe = torch.clamp(t_, min=1e-5)
            coef = (t_safe - r_ * (r_ / t_safe)**self.alpha) / (self.alpha + 1.0)
        u_tgt = v_hat - coef * dudt
        error = u - stopgrad(u_tgt)
        loss = adaptive_l2_loss(error)
      

        mse_val = (stopgrad(error) ** 2).mean()

        loss_dict = {
                'bc_loss': loss.item(),
                'mse_val': mse_val.item()
            }
        
        return loss, loss_dict

    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)

        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))  # Apply sigmoid

        # Assign t = max, r = min, for each pair
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r

def normalize_to_neg1_1(x):
    return x * 2 - 1


def unnormalize_to_0_1(x):
    return (x + 1) * 0.5

def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (B, C, W, H)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    delta_sq = torch.mean(error ** 2, dim=tuple(range(1, error.ndim)))    
    # delta_sq = torch.sum(error ** 2, dim=tuple(range(1, error.ndim)))
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq
    return (stopgrad(w) * loss).mean()
