from typing import Dict

import numpy as np
import torch

from core.common.pytorch_util import dict_apply
from core.dataset.metaworld_dataset import MetaworldDataset


class MetaworldTerminalDataset(MetaworldDataset):
    """Metaworld dataset with extra terminal representation supervision fields.

    The parent sequence sampling is kept unchanged. This subclass only uses the
    sampled sequence index to locate the same episode and draw terminal/non-terminal
    point cloud frames for auxiliary losses.
    """

    def __init__(
        self,
        terminal_window=8,
        neg_count=4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.terminal_window = int(terminal_window)
        self.neg_count = int(neg_count)
        if self.terminal_window < 1:
            raise ValueError(f"terminal_window must be >= 1, got {self.terminal_window}")
        if self.neg_count < 1:
            raise ValueError(f"neg_count must be >= 1, got {self.neg_count}")

    def _episode_bounds_for_sample(self, idx: int):
        buffer_start_idx, buffer_end_idx, _, _ = self.sampler.indices[idx]
        episode_ends = np.asarray(self.replay_buffer.episode_ends[:], dtype=np.int64)
        episode_idx = int(np.searchsorted(episode_ends, int(buffer_start_idx), side="right"))
        episode_start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
        episode_end_exclusive = int(episode_ends[episode_idx])
        episode_end = episode_end_exclusive - 1

        # Use the last real frame touched by the sampled sequence, before any
        # pad_after frames are expanded by SequenceSampler.
        repr_idx = min(max(int(buffer_end_idx) - 1, episode_start), episode_end)
        return episode_start, episode_end, repr_idx

    def _sample_terminal_pair(self, terminal_start: int, episode_end: int):
        term_indices = np.arange(terminal_start, episode_end + 1, dtype=np.int64)
        if term_indices.shape[0] == 1:
            anchor_idx = pos_idx = int(term_indices[0])
        else:
            anchor_idx, pos_idx = np.random.choice(term_indices, size=2, replace=False)
            anchor_idx = int(anchor_idx)
            pos_idx = int(pos_idx)
        return anchor_idx, pos_idx

    def _sample_negative_indices(self, episode_start: int, terminal_start: int):
        non_terminal_end = terminal_start - 1
        if non_terminal_end >= episode_start:
            candidates = np.arange(episode_start, non_terminal_end + 1, dtype=np.int64)
            return np.random.choice(candidates, size=self.neg_count, replace=True).astype(np.int64)
        return np.full((self.neg_count,), episode_start, dtype=np.int64)

    def _terminal_aux_data(self, idx: int) -> Dict[str, np.ndarray]:
        episode_start, episode_end, repr_idx = self._episode_bounds_for_sample(idx)
        terminal_start = max(episode_start, episode_end - self.terminal_window + 1)

        anchor_idx, pos_idx = self._sample_terminal_pair(terminal_start, episode_end)
        neg_indices = self._sample_negative_indices(episode_start, terminal_start)

        point_cloud = self.replay_buffer["point_cloud"]
        episode_len = max(1, episode_end - episode_start)
        ttg_target = np.clip((episode_end - repr_idx) / episode_len, 0.0, 1.0)
        term_label = 1.0 if repr_idx >= terminal_start else 0.0

        return {
            "term_anchor_point_cloud": point_cloud[anchor_idx].astype(np.float32),
            "term_pos_point_cloud": point_cloud[pos_idx].astype(np.float32),
            "neg_point_clouds": point_cloud[neg_indices].astype(np.float32),
            "repr_point_cloud": point_cloud[repr_idx].astype(np.float32),
            "ttg_target": np.array(ttg_target, dtype=np.float32),
            "term_label": np.array(term_label, dtype=np.float32),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        data.update(self._terminal_aux_data(idx))

        sample_start_idx = int(self.sampler.indices[idx][2])
        data["causal_action_pad_mask"] = np.array(sample_start_idx > 0, dtype=np.bool_)
        return dict_apply(data, torch.from_numpy)
