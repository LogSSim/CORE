from typing import Dict

import copy
import numpy as np
import torch

from core.common.pytorch_util import dict_apply
from core.common.replay_buffer import ReplayBuffer
from core.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from core.dataset.base_dataset import BaseDataset
from core.model.common.normalizer import LinearNormalizer


class AdroitGoalFinalDataset(BaseDataset):
    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        task_name=None,
    ):
        super().__init__()
        self.task_name = task_name
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=["state", "action", "point_cloud", "img"],
        )
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed,
        )

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.episode_ends = np.asarray(self.replay_buffer.episode_ends[:], dtype=np.int64)
        self.episode_last_indices = self.episode_ends - 1

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"][..., :],
            "point_cloud": self.replay_buffer["point_cloud"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _episode_idx_from_sample(self, idx: int) -> int:
        buffer_start_idx = int(self.sampler.indices[idx][0])
        return int(np.searchsorted(self.episode_ends, buffer_start_idx, side="right"))

    def _goal_obs_from_episode(self, episode_idx: int) -> Dict[str, np.ndarray]:
        goal_idx = int(self.episode_last_indices[episode_idx])
        return {
            "point_cloud": self.replay_buffer["point_cloud"][goal_idx].astype(np.float32),
            "image": np.asarray(self.replay_buffer["img"][goal_idx]),
        }

    def _sample_to_data(self, sample, goal_obs):
        agent_pos = sample["state"][:, :].astype(np.float32)
        point_cloud = sample["point_cloud"][:, :].astype(np.float32)

        data = {
            "obs": {
                "point_cloud": point_cloud,
                "agent_pos": agent_pos,
            },
            "goal_obs": goal_obs,
            "action": sample["action"].astype(np.float32),
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        episode_idx = self._episode_idx_from_sample(idx)
        goal_obs = self._goal_obs_from_episode(episode_idx)
        data = self._sample_to_data(sample, goal_obs)
        sample_start_idx = int(self.sampler.indices[idx][2])
        data["causal_action_pad_mask"] = np.array(sample_start_idx > 0, dtype=np.bool_)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
