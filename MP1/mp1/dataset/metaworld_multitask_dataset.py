import bisect
import copy
from typing import Dict, List, Optional

import numpy as np
import torch

from mp1.common.pytorch_util import dict_apply
from mp1.dataset.base_dataset import BaseDataset
from mp1.dataset.metaworld_dataset import MetaworldDataset
from mp1.model.common.normalizer import LinearNormalizer


class MultiMetaworldDataset(BaseDataset):
    """Concatenate multiple MetaWorld datasets with the same observation/action shapes."""

    def __init__(
        self,
        tasks: List[Dict],
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes: Optional[int] = None,
    ):
        super().__init__()
        if len(tasks) == 0:
            raise ValueError("MultiMetaworldDataset requires at least one task.")

        self.tasks = [dict(task) for task in tasks]
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes

        self.datasets = [
            self._build_dataset(task_cfg=task_cfg, task_idx=task_idx)
            for task_idx, task_cfg in enumerate(self.tasks)
        ]
        self.task_names = [
            str(task_cfg.get("name", f"task_{idx}"))
            for idx, task_cfg in enumerate(self.tasks)
        ]
        self.cumulative_lengths = np.cumsum([len(dataset) for dataset in self.datasets]).tolist()

        if self.cumulative_lengths[-1] == 0:
            raise ValueError("MultiMetaworldDataset has no samples.")

        self._validate_shapes()

    def _build_dataset(self, task_cfg: Dict, task_idx: int) -> MetaworldDataset:
        if "zarr_path" not in task_cfg:
            raise ValueError(f"Task config at index {task_idx} is missing zarr_path.")
        return MetaworldDataset(
            zarr_path=task_cfg["zarr_path"],
            horizon=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            seed=int(task_cfg.get("seed", self.seed)),
            val_ratio=float(task_cfg.get("val_ratio", self.val_ratio)),
            max_train_episodes=task_cfg.get("max_train_episodes", self.max_train_episodes),
        )

    def _validate_shapes(self):
        first = self.datasets[0].replay_buffer
        expected = {
            "state": first["state"].shape[1:],
            "action": first["action"].shape[1:],
            "point_cloud": first["point_cloud"].shape[1:],
        }
        for task_name, dataset in zip(self.task_names, self.datasets):
            replay_buffer = dataset.replay_buffer
            for key, shape in expected.items():
                if replay_buffer[key].shape[1:] != shape:
                    raise ValueError(
                        f"Task {task_name} has incompatible {key} shape "
                        f"{replay_buffer[key].shape[1:]}; expected {shape}."
                    )

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.datasets = [dataset.get_validation_dataset() for dataset in self.datasets]
        val_set.cumulative_lengths = np.cumsum(
            [len(dataset) for dataset in val_set.datasets]
        ).tolist()
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self._concat_replay_key("action"),
            "agent_pos": self._concat_replay_key("state"),
            "point_cloud": self._concat_replay_key("point_cloud"),
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def _concat_replay_key(self, key: str) -> np.ndarray:
        return np.concatenate(
            [dataset.replay_buffer[key][...] for dataset in self.datasets],
            axis=0,
        )

    def __len__(self) -> int:
        return int(self.cumulative_lengths[-1])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        dataset_idx = bisect.bisect_right(self.cumulative_lengths, idx)
        prev_length = 0 if dataset_idx == 0 else self.cumulative_lengths[dataset_idx - 1]
        local_idx = idx - prev_length

        data = self.datasets[dataset_idx][local_idx]
        data["task_id"] = torch.tensor(dataset_idx, dtype=torch.long)
        return data
