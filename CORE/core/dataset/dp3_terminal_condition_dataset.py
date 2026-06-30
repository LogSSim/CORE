"""Standalone DP3 dataset with two terminal frames as extra condition.

This file is intentionally independent from the existing terminal-goal
policies.  It keeps the original DP3 imitation-learning samples unchanged and
only adds::

    data["terminal_obs"] = {
        "point_cloud": [2, N, C],
        "agent_pos":   [2, D],
    }

The terminal two-frame bank is selected from demonstration episode indices
[stage1_episode_start, stage1_episode_end).  For stage 3, only
stage3_num_episodes episodes are kept from that stage-1 range.
"""

from typing import Dict, Iterable, Optional

import copy
import os
from pathlib import Path
import numpy as np
import torch

from core.common.pytorch_util import dict_apply
from core.common.replay_buffer import ReplayBuffer
from core.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from core.dataset.base_dataset import BaseDataset
from core.model.common.normalizer import LinearNormalizer


def _resolve_data_path(path):
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


class DP3TerminalConditionDataset(BaseDataset):
    """Original DP3 sequence dataset plus a two-frame terminal condition.

    Args:
        stage: 1 uses all episodes in [stage1_episode_start, stage1_episode_end).
            3 uses only ``stage3_num_episodes`` episodes from that same range.
        stage1_episode_start/stage1_episode_end: demonstration episode range used
            as the terminal-condition source.  Defaults to 20..49.
        stage3_num_episodes: number of terminal-condition demonstrations used
            when ``stage == 3``. Defaults to 10.
        terminal_frame_count: number of final frames used as condition. Keep this
            at 2 for "终端两帧".
    """

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
        stage=3,
        stage1_episode_start=20,
        stage1_episode_end=50,
        stage3_num_episodes=10,
        terminal_frame_count=2,
        terminal_selection="cycle",
    ):
        super().__init__()
        self.task_name = task_name
        self.zarr_path = _resolve_data_path(zarr_path)
        self.seed = int(seed)
        self.stage = int(stage)
        self.stage1_episode_start = int(stage1_episode_start)
        self.stage1_episode_end = int(stage1_episode_end)
        self.stage3_num_episodes = int(stage3_num_episodes)
        self.terminal_frame_count = int(terminal_frame_count)
        self.terminal_selection = str(terminal_selection)

        if self.terminal_frame_count != 2:
            raise ValueError(
                "DP3TerminalConditionDataset is configured for terminal two frames; "
                f"got terminal_frame_count={self.terminal_frame_count}"
            )
        if self.stage not in (1, 3):
            raise ValueError(f"stage must be 1 or 3, got {self.stage}")
        if self.stage3_num_episodes < 1:
            raise ValueError("stage3_num_episodes must be >= 1")

        # Adroit zarr has img, Metaworld usually does not.  DP3 only needs these.
        self.replay_buffer = ReplayBuffer.copy_from_path(
            self.zarr_path,
            keys=["state", "action", "point_cloud"],
        )
        self.episode_ends = np.asarray(self.replay_buffer.episode_ends[:], dtype=np.int64)

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

        self.terminal_episode_indices = self._select_terminal_episode_indices()

    def _select_terminal_episode_indices(self) -> np.ndarray:
        n_episodes = int(self.replay_buffer.n_episodes)
        start = max(0, min(self.stage1_episode_start, n_episodes))
        end = max(start, min(self.stage1_episode_end, n_episodes))
        indices = np.arange(start, end, dtype=np.int64)
        if indices.size == 0:
            raise ValueError(
                f"No terminal-condition episodes selected from range "
                f"[{self.stage1_episode_start}, {self.stage1_episode_end}) "
                f"for dataset with {n_episodes} episodes."
            )

        if self.stage == 3:
            keep = min(self.stage3_num_episodes, indices.size)
            rng = np.random.default_rng(self.seed)
            # deterministic subset from the stage-1 range; sorted for reproducible logs.
            indices = np.sort(rng.choice(indices, size=keep, replace=False)).astype(np.int64)
        return indices

    def _episode_bounds(self, episode_idx: int):
        start = 0 if episode_idx == 0 else int(self.episode_ends[episode_idx - 1])
        end_exclusive = int(self.episode_ends[episode_idx])
        return start, end_exclusive

    def _terminal_obs_from_episode(self, episode_idx: int) -> Dict[str, np.ndarray]:
        start, end_exclusive = self._episode_bounds(int(episode_idx))
        if end_exclusive <= start:
            raise RuntimeError(f"Episode {episode_idx} is empty")

        first_idx = max(start, end_exclusive - self.terminal_frame_count)
        indices = np.arange(first_idx, end_exclusive, dtype=np.int64)
        if indices.shape[0] < self.terminal_frame_count:
            # Very short episode: repeat the first available frame on the left.
            pad = np.full(
                (self.terminal_frame_count - indices.shape[0],),
                int(indices[0]),
                dtype=np.int64,
            )
            indices = np.concatenate([pad, indices], axis=0)

        return {
            "point_cloud": self.replay_buffer["point_cloud"][indices].astype(np.float32),
            "agent_pos": self.replay_buffer["state"][indices].astype(np.float32),
        }

    def _terminal_episode_for_item(self, idx: int) -> int:
        if self.terminal_selection == "random":
            # Deterministic per-sample random choice to keep dataloader workers stable.
            rng = np.random.default_rng(self.seed + int(idx))
            return int(rng.choice(self.terminal_episode_indices))
        if self.terminal_selection != "cycle":
            raise ValueError(f"Unsupported terminal_selection={self.terminal_selection}")
        return int(self.terminal_episode_indices[int(idx) % len(self.terminal_episode_indices)])

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

    def _sample_to_data(self, sample, terminal_obs):
        return {
            "obs": {
                "point_cloud": sample["point_cloud"].astype(np.float32),
                "agent_pos": sample["state"].astype(np.float32),
            },
            "terminal_obs": terminal_obs,
            "action": sample["action"].astype(np.float32),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        terminal_episode_idx = self._terminal_episode_for_item(idx)
        terminal_obs = self._terminal_obs_from_episode(terminal_episode_idx)
        data = self._sample_to_data(sample, terminal_obs)
        sample_start_idx = int(self.sampler.indices[idx][2])
        data["causal_action_pad_mask"] = np.array(sample_start_idx > 0, dtype=np.bool_)
        data["terminal_episode_idx"] = np.array(terminal_episode_idx, dtype=np.int64)
        return dict_apply(data, torch.from_numpy)
