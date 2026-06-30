"""DP3 dataset for three-stage final-state conditioning experiments.

Stage 1 reads the first 20 demonstrations from zarr as a normal DP3 imitation
dataset. Stage 3 reads the first 10 demonstrations and adds final-state features
built by the stage-2 feature extraction script.
"""

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


class DP3FinalStateDataset(BaseDataset):
    """Original DP3 sequence dataset plus optional final-state condition.

    Args:
        stage: 1 returns ordinary DP3 samples from the first 20 zarr episodes.
            3 additionally returns ``final_state_condition`` from the stage-2
            npz artifact and trains on the first 10 zarr episodes.
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
        stage=1,
        stage1_num_train_episodes=20,
        stage3_num_train_episodes=10,
        final_state_feature_path=None,
    ):
        super().__init__()
        self.task_name = task_name
        self.zarr_path = _resolve_data_path(zarr_path)
        self.seed = int(seed)
        self.stage = int(stage)
        self.stage1_num_train_episodes = int(stage1_num_train_episodes)
        self.stage3_num_train_episodes = int(stage3_num_train_episodes)
        self.final_state_feature_path = _resolve_data_path(final_state_feature_path)

        if self.stage not in (1, 3):
            raise ValueError(f"stage must be 1 or 3, got {self.stage}")

        self.replay_buffer = ReplayBuffer.copy_from_path(
            self.zarr_path,
            keys=["state", "action", "point_cloud"],
        )
        self.episode_ends = np.asarray(self.replay_buffer.episode_ends[:], dtype=np.int64)

        selected_mask = self._selected_episode_mask()
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = selected_mask & ~val_mask
        if max_train_episodes is not None and int(max_train_episodes) < int(train_mask.sum()):
            rng = np.random.default_rng(seed)
            train_indices = np.nonzero(train_mask)[0]
            keep = rng.choice(train_indices, size=int(max_train_episodes), replace=False)
            train_mask = np.zeros_like(train_mask)
            train_mask[keep] = True
        if not np.any(train_mask):
            raise ValueError(
                "No training episodes selected. Check stage1/stage3 episode counts and val_ratio."
            )

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

        self._feature_by_episode = {}
        if self.stage == 3:
            self._feature_by_episode = self._load_final_state_features()

    def _selected_episode_mask(self):
        n_episodes = int(self.replay_buffer.n_episodes)
        if self.stage == 1:
            num_episodes = self.stage1_num_train_episodes
        else:
            num_episodes = self.stage3_num_train_episodes
        end = max(0, min(int(num_episodes), n_episodes))
        mask = np.zeros(n_episodes, dtype=bool)
        mask[:end] = True
        if not np.any(mask):
            raise ValueError(
                f"No episodes selected from the first {num_episodes} zarr episodes "
                f"for dataset with {n_episodes} episodes."
            )
        return mask

    def _load_final_state_features(self):
        if self.final_state_feature_path is None:
            raise ValueError("stage=3 requires final_state_feature_path from stage 2.")

        artifact = np.load(self.final_state_feature_path)
        episode_indices = artifact["episode_indices"].astype(np.int64)
        final_features = artifact["final_features"].astype(np.float32)
        if "cluster_features" in artifact:
            cluster_features = artifact["cluster_features"].astype(np.float32)
        else:
            labels = artifact["cluster_labels"].astype(np.int64)
            centers = artifact["cluster_centers"].astype(np.float32)
            cluster_features = centers[labels]

        if final_features.shape != cluster_features.shape:
            raise ValueError(
                "final_features and cluster_features must have the same shape, got "
                f"{final_features.shape} and {cluster_features.shape}."
            )

        return {
            int(ep): {
                "final_feature": final_features[i],
                "cluster_feature": cluster_features[i],
            }
            for i, ep in enumerate(episode_indices)
        }

    def _episode_bounds(self, episode_idx: int):
        start = 0 if episode_idx == 0 else int(self.episode_ends[episode_idx - 1])
        end_exclusive = int(self.episode_ends[episode_idx])
        return start, end_exclusive

    def _episode_idx_for_item(self, idx: int) -> int:
        buffer_start_idx = int(self.sampler.indices[idx][0])
        return int(np.searchsorted(self.episode_ends, buffer_start_idx, side="right"))

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_mask = self.selected_mask & ~self.train_mask
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=val_mask,
        )
        val_set.train_mask = val_mask
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

    def _sample_to_data(self, sample):
        return {
            "obs": {
                "point_cloud": sample["point_cloud"].astype(np.float32),
                "agent_pos": sample["state"].astype(np.float32),
            },
            "action": sample["action"].astype(np.float32),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        sample_start_idx = int(self.sampler.indices[idx][2])
        data["causal_action_pad_mask"] = np.array(sample_start_idx > 0, dtype=np.bool_)

        episode_idx = self._episode_idx_for_item(idx)
        data["episode_idx"] = np.array(episode_idx, dtype=np.int64)
        if self.stage == 3:
            if episode_idx not in self._feature_by_episode:
                raise KeyError(
                    f"Episode {episode_idx} is missing from {self.final_state_feature_path}."
                )
            feature = self._feature_by_episode[episode_idx]
            data["final_state_condition"] = {
                "final_feature": feature["final_feature"],
                "cluster_feature": feature["cluster_feature"],
            }

        return dict_apply(data, torch.from_numpy)
