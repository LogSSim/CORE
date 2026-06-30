import collections
import time
from typing import Dict, List

import numpy as np
import torch
import tqdm
from termcolor import cprint

from core.common.pytorch_util import dict_apply
from core.env import MetaWorldEnv
from core.env_runner.base_runner import BaseRunner
from core.gym_util.multistep_wrapper import MultiStepWrapper
from core.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper
from core.policy.base_policy import BasePolicy
import core.common.logger_util as logger_util


class MultiMetaworldRunner(BaseRunner):
    """Evaluate one policy on multiple MetaWorld tasks and report aggregate metrics."""

    def __init__(
        self,
        output_dir,
        task_names: List[str],
        eval_episodes=20,
        max_steps=1000,
        n_obs_steps=8,
        n_action_steps=8,
        fps=10,
        crf=22,
        render_size=84,
        tqdm_interval_sec=5.0,
        n_envs=None,
        n_train=None,
        n_test=None,
        device="cuda",
        use_point_crop=True,
        num_points=512,
    ):
        super().__init__(output_dir)
        if len(task_names) == 0:
            raise ValueError("MultiMetaworldRunner requires at least one task.")

        self.task_names = [str(task_name) for task_name in task_names]
        self.eval_episodes = eval_episodes
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.device = device
        self.use_point_crop = use_point_crop
        self.num_points = num_points

        self.envs = {
            task_name: self._make_env(task_name)
            for task_name in self.task_names
        }
        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)
        self.task_logger_util_test10 = {
            task_name: logger_util.LargestKRecorder(K=5)
            for task_name in self.task_names
        }

    def _make_env(self, task_name: str):
        return MultiStepWrapper(
            SimpleVideoRecordingWrapper(
                MetaWorldEnv(
                    task_name=task_name,
                    device=self.device,
                    use_point_crop=self.use_point_crop,
                    num_points=self.num_points,
                )
            ),
            n_obs_steps=self.n_obs_steps,
            n_action_steps=self.n_action_steps,
            max_episode_steps=self.max_steps,
            reward_agg_method="sum",
        )

    def _run_task(self, task_name: str, env, policy: BasePolicy) -> Dict:
        device = policy.device

        all_traj_rewards = []
        all_success_rates = []
        all_time = []

        desc = f"Eval in Metaworld {task_name} Pointcloud Env"
        for _ in tqdm.tqdm(
            range(self.eval_episodes),
            desc=desc,
            leave=False,
            mininterval=self.tqdm_interval_sec,
        ):
            obs = env.reset()
            policy.reset()

            done = False
            traj_reward = 0
            is_success = False
            actual_step_count = 0
            total_time = 0

            while not done:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(
                    np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device),
                )

                with torch.no_grad():
                    obs_dict_input = {
                        "point_cloud": obs_dict["point_cloud"].unsqueeze(0),
                        "agent_pos": obs_dict["agent_pos"].unsqueeze(0),
                    }
                    start_time = time.time()
                    action_dict = policy.predict_action(obs_dict_input)
                    total_time += time.time() - start_time

                np_action_dict = dict_apply(
                    action_dict,
                    lambda x: x.detach().to("cpu").numpy(),
                )
                action = np_action_dict["action"].squeeze(0)

                obs, reward, done, info = env.step(action)
                traj_reward += reward
                done = np.all(done)
                is_success = is_success or max(info["success"])
                actual_step_count += 1

            all_success_rates.append(is_success)
            all_traj_rewards.append(traj_reward)
            all_time.append(total_time / max(actual_step_count, 1))

        return {
            "mean_traj_rewards": float(np.mean(all_traj_rewards)),
            "mean_success_rates": float(np.mean(all_success_rates)),
            "mean_time": float(np.mean(all_time)),
            "test_mean_score": float(np.mean(all_success_rates)),
        }

    def run(self, policy: BasePolicy):
        per_task_logs = collections.OrderedDict()
        for task_name in self.task_names:
            task_log = self._run_task(task_name, self.envs[task_name], policy)
            per_task_logs[task_name] = task_log
            cprint(
                f"[{task_name}] test_mean_score: {task_log['test_mean_score'] * 100:.2f}",
                "green",
            )

        scores = np.array(
            [task_log["test_mean_score"] for task_log in per_task_logs.values()],
            dtype=np.float32,
        )
        rewards = np.array(
            [task_log["mean_traj_rewards"] for task_log in per_task_logs.values()],
            dtype=np.float32,
        )
        times = np.array(
            [task_log["mean_time"] for task_log in per_task_logs.values()],
            dtype=np.float32,
        )

        log_data = {
            "test_mean_score": float(np.mean(scores)),
            "mean_success_rates": float(np.mean(scores)),
            "mean_traj_rewards": float(np.mean(rewards)),
            "mean_time": float(np.mean(times)),
            "multi_task_min_score": float(np.min(scores)),
            "multi_task_max_score": float(np.max(scores)),
        }

        for task_name, task_log in per_task_logs.items():
            key_prefix = f"task_{task_name}"
            self.task_logger_util_test10[task_name].record(task_log["test_mean_score"])
            task_top5_avg = self.task_logger_util_test10[task_name].average_of_largest_K()
            log_data[f"{key_prefix}_test_mean_score"] = task_log["test_mean_score"]
            log_data[f"{key_prefix}_mean_success_rates"] = task_log["mean_success_rates"]
            log_data[f"{key_prefix}_mean_traj_rewards"] = task_log["mean_traj_rewards"]
            log_data[f"{key_prefix}_mean_time"] = task_log["mean_time"]
            log_data[f"{key_prefix}_SR_test_L5"] = task_top5_avg

        self.logger_util_test.record(log_data["test_mean_score"])
        self.logger_util_test10.record(log_data["test_mean_score"])
        log_data["SR_test_L3"] = self.logger_util_test.average_of_largest_K()
        log_data["SR_test_L5"] = self.logger_util_test10.average_of_largest_K()
        log_data["multi_task_taskwise_SR_test_L5"] = float(
            np.mean(
                [
                    self.task_logger_util_test10[task_name].average_of_largest_K()
                    for task_name in self.task_names
                ]
            )
        )

        cprint(f"multi_task_test_mean_score: {log_data['test_mean_score'] * 100:.2f}", "green")
        cprint(
            "multi_task_taskwise_SR_test_L5: "
            f"{log_data['multi_task_taskwise_SR_test_L5'] * 100:.2f}",
            "green",
        )
        return log_data
