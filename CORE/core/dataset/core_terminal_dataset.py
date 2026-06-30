"""Dataset for CORE three-stage training."""

from typing import Dict

import copy
import os
from pathlib import Path

import numpy as np
import torch

from core.common.pytorch_util import dict_apply
from core.common.replay_buffer import ReplayBuffer
from core.common.sampler import SequenceSampler, get_val_mask
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


class CORETerminalDataset(BaseDataset):
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
        stage=1,
        stage1_num_train_episodes=20,
        stage3_num_train_episodes=10,
        return_terminal_samples=True,
        terminal_window=2,
        terminal_num_negatives=4,
    ):
        super().__init__()
        self.task_name = task_name
        self.zarr_path = _resolve_data_path(zarr_path)
        self.seed = int(seed)
        self.stage = int(stage)
        self.stage1_num_train_episodes = int(stage1_num_train_episodes)
        self.stage3_num_train_episodes = int(stage3_num_train_episodes)
        self.return_terminal_samples = bool(return_terminal_samples)
        self.terminal_window = int(terminal_window)
        self.terminal_num_negatives = int(terminal_num_negatives)

        self.replay_buffer = ReplayBuffer.copy_from_path(
            self.zarr_path,
            keys=["state", "action", "point_cloud"],
        )
        self.episode_ends = np.asarray(self.replay_buffer.episode_ends[:], dtype=np.int64)

        selected_mask = self._selected_episode_mask()
        val_mask = get_val_mask(self.replay_buffer.n_episodes, val_ratio, seed)
        train_mask = selected_mask & ~val_mask
        if max_train_episodes is not None and int(max_train_episodes) < int(train_mask.sum()):
            rng = np.random.default_rng(seed)
            train_indices = np.nonzero(train_mask)[0]
            keep = rng.choice(train_indices, size=int(max_train_episodes), replace=False)
            train_mask = np.zeros_like(train_mask)
            train_mask[keep] = True
        if not np.any(train_mask):
            raise ValueError("No training episodes selected for CORETerminalDataset.")

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.selected_mask = selected_mask
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        episode_indices = np.nonzero(self.train_mask)[0]
        self.positive_terminal_indices = np.asarray(
            [int(self.episode_ends[i]) - 1 for i in episode_indices],
            dtype=np.int64,
        )
        if len(self.positive_terminal_indices) == 0:
            raise ValueError("No terminal positive frames found for CORETerminalDataset.")

    def _selected_episode_mask(self):
        n_episodes = int(self.replay_buffer.n_episodes)
        num = self.stage1_num_train_episodes if self.stage == 1 else self.stage3_num_train_episodes
        end = max(0, min(int(num), n_episodes))
        mask = np.zeros(n_episodes, dtype=bool)
        mask[:end] = True
        return mask

    def _episode_bounds_for_sample(self, idx: int):
        buffer_start_idx, buffer_end_idx, _, _ = self.sampler.indices[idx]
        episode_idx = int(np.searchsorted(self.episode_ends, int(buffer_start_idx), side="right"))
        episode_start = 0 if episode_idx == 0 else int(self.episode_ends[episode_idx - 1])
        episode_end = int(self.episode_ends[episode_idx]) - 1
        repr_idx = min(max(int(buffer_end_idx) - 1, episode_start), episode_end)
        return episode_start, episode_end, repr_idx

    def _terminal_aux_data(self, idx: int) -> Dict[str, np.ndarray]:
        episode_start, episode_end, repr_idx = self._episode_bounds_for_sample(idx)
        anchor_idx = int(episode_end)
        pos_idx = int(np.random.choice(self.positive_terminal_indices))

        episode_len = max(1, episode_end - episode_start + 1)
        non_terminal_count = max(1, int(np.floor(0.9 * episode_len)))
        non_terminal_end = min(episode_end, episode_start + non_terminal_count - 1)
        neg_candidates = np.arange(episode_start, non_terminal_end + 1, dtype=np.int64)
        neg_indices = np.random.choice(
            neg_candidates,
            size=self.terminal_num_negatives,
            replace=True,
        ).astype(np.int64)

        point_cloud = self.replay_buffer["point_cloud"]
        norm_denom = max(1, episode_end - episode_start)
        ttg_target = np.clip((episode_end - repr_idx) / norm_denom, 0.0, 1.0)
        terminal_start = episode_end
        term_label = 1.0 if repr_idx >= terminal_start else 0.0
        return {
            "term_anchor_point_cloud": point_cloud[anchor_idx].astype(np.float32),
            "term_pos_point_cloud": point_cloud[pos_idx].astype(np.float32),
            "neg_point_clouds": point_cloud[neg_indices].astype(np.float32),
            "repr_point_cloud": point_cloud[repr_idx].astype(np.float32),
            "ttg_target": np.array(ttg_target, dtype=np.float32),
            "term_label": np.array(term_label, dtype=np.float32),
        }

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.selected_mask & ~self.train_mask,
        )
        val_set.train_mask = self.selected_mask & ~self.train_mask
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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = {
            "obs": {
                "point_cloud": sample["point_cloud"].astype(np.float32),
                "agent_pos": sample["state"].astype(np.float32),
            },
            "action": sample["action"].astype(np.float32),
        }
        if self.stage == 1 and self.return_terminal_samples:
            data.update(self._terminal_aux_data(idx))
        sample_start_idx = int(self.sampler.indices[idx][2])
        data["causal_action_pad_mask"] = np.array(sample_start_idx > 0, dtype=np.bool_)
        return dict_apply(data, torch.from_numpy)
