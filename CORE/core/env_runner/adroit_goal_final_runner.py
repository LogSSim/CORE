import time

import numpy as np
import torch
import tqdm
from termcolor import cprint

import core.common.logger_util as logger_util
from core.common.pytorch_util import dict_apply
from core.common.replay_buffer import ReplayBuffer
from core.env import AdroitEnv
from core.env_runner.base_runner import BaseRunner
from core.gym_util.mjpc_diffusion_wrapper import MujocoPointcloudWrapperAdroit
from core.gym_util.multistep_wrapper import MultiStepWrapper
from core.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper
from core.policy.base_policy import BasePolicy


class AdroitGoalFinalRunner(BaseRunner):
    def __init__(
        self,
        output_dir,
        eval_episodes=20,
        max_steps=200,
        n_obs_steps=8,
        n_action_steps=8,
        fps=20,
        crf=22,
        render_size=84,
        tqdm_interval_sec=5.0,
        task_name=None,
        use_point_crop=True,
        goal_zarr_path=None,
        goal_selection_seed=42,
    ):
        super().__init__(output_dir)
        self.task_name = task_name

        def env_fn():
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    MujocoPointcloudWrapperAdroit(
                        env=AdroitEnv(env_name=task_name, use_point_cloud=True),
                        env_name="adroit_" + task_name,
                        use_point_crop=use_point_crop,
                    )
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method="sum",
            )

        self.eval_episodes = eval_episodes
        self.env = env_fn()
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

        if goal_zarr_path is None:
            raise ValueError("goal_zarr_path must be provided for AdroitGoalFinalRunner")
        self.goal_buffer = ReplayBuffer.copy_from_path(
            goal_zarr_path,
            keys=["point_cloud", "img"],
        )
        self.goal_episode_ends = np.asarray(self.goal_buffer.episode_ends[:], dtype=np.int64)
        self.goal_last_indices = self.goal_episode_ends - 1
        goal_rng = np.random.default_rng(goal_selection_seed)
        self.goal_episode_order = goal_rng.permutation(len(self.goal_last_indices))

    def _goal_obs_for_rollout(self, rollout_idx: int):
        if len(self.goal_episode_order) == 0:
            raise ValueError("Goal buffer is empty.")
        goal_episode_idx = int(self.goal_episode_order[rollout_idx % len(self.goal_episode_order)])
        goal_idx = int(self.goal_last_indices[goal_episode_idx])
        return {
            "point_cloud": self.goal_buffer["point_cloud"][goal_idx].astype(np.float32),
            "image": np.asarray(self.goal_buffer["img"][goal_idx]),
        }

    def run(self, policy: BasePolicy):
        device = policy.device
        env = self.env

        all_goal_achieved = []
        all_success_rates = []
        all_time = []

        for episode_idx in tqdm.tqdm(
            range(self.eval_episodes),
            desc=f"Eval in Adroit {self.task_name} Goal-Final Pointcloud Env",
            leave=False,
            mininterval=self.tqdm_interval_sec,
        ):
            obs = env.reset()
            policy.reset()
            goal_obs_np = self._goal_obs_for_rollout(episode_idx)
            goal_obs = dict_apply(goal_obs_np, lambda x: torch.from_numpy(x).to(device=device))

            done = False
            num_goal_achieved = 0
            actual_step_count = 0
            total_time = 0.0
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
                        "goal_obs": {
                            "point_cloud": goal_obs["point_cloud"].unsqueeze(0),
                            "image": goal_obs["image"].unsqueeze(0),
                        },
                    }
                    start_time = time.time()
                    action_dict = policy.predict_action(obs_dict_input)
                    end_time = time.time()
                    total_time += end_time - start_time

                np_action_dict = dict_apply(
                    action_dict,
                    lambda x: x.detach().to("cpu").numpy(),
                )
                action = np_action_dict["action"].squeeze(0)
                obs, reward, done, info = env.step(action)
                num_goal_achieved += np.sum(info["goal_achieved"])
                done = np.all(done)
                actual_step_count += 1

            all_success_rates.append(info["goal_achieved"])
            all_goal_achieved.append(num_goal_achieved)
            all_time.append(total_time / actual_step_count)

        log_data = dict()
        log_data["mean_n_goal_achieved"] = np.mean(all_goal_achieved)
        log_data["mean_success_rates"] = np.mean(all_success_rates)
        log_data["mean_time"] = np.mean(all_time)
        log_data["test_mean_score"] = np.mean(all_success_rates)

        cprint(f"test_mean_score: {np.mean(all_success_rates) * 100}", "green")
        cprint(f"test_mean_time: {np.mean(all_time) * 1000}", "red")

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data["SR_test_L3"] = self.logger_util_test.average_of_largest_K()
        log_data["SR_test_L5"] = self.logger_util_test10.average_of_largest_K()

        _ = env.reset()
        return log_data
