"""Miscellaneous functions for training."""

import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import Wrapper
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import (
    load_results,  # pyright: ignore[reportPrivateImportUsage]
    ts2xy,
)
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from rl_cuas.environment.environment import Environment


# Make Stable Baselines3 Env function
def make_sb3_env(
    env_class: Callable[..., Environment],
    wrapper_class: Callable[
        ...,
        Wrapper[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray],
    ]
    | None,
    num_envs: int,
    seed: int | None = None,
    allow_early_resets: bool = True,
    start_method: str | None = None,
    no_vec: bool = False,
    use_subprocess: bool = True,
    monitor_folder: str | None = None,
) -> VecEnv | Monitor[dict[str, np.ndarray], np.ndarray]:
    """
    Create a wrapped, monitored VecEnv.

    Parameters
    ----------
    env_class: Environment
        The environment class.
    wrapper_class: Wrapper
        The wrapper class.
    num_envs: int
        The number of environments.
    seed: int | None
        The seed.
    allow_early_resets: bool
        Whether to allow early resets.
    start_method: str | None
        The method used to start the subprocesses.
    no_vec: bool
        Whether to avoid usage of Vectorized Env or not.
    use_subprocess: bool
        Whether to use `SubprocVecEnv` or `DummyVecEnv`.
    monitor_folder: str
        The folder to save the monitor.

    Returns
    -------
    VecEnv:
        The vectorized environment.
    """
    if seed is None:
        seed = int(time.time())

    if monitor_folder is None:
        monitor_folder = tempfile.mkdtemp(prefix="invai_")

    def _make_sb3_env(
        rank: int,
    ) -> Callable[..., Monitor[dict[str, np.ndarray], np.ndarray]]:
        def _init() -> Monitor[dict[str, np.ndarray], np.ndarray]:
            env: (
                Environment
                | Wrapper[
                    dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray
                ]
            ) = env_class()
            if wrapper_class is not None:
                env = wrapper_class(env)

            # Create log dir
            monitor_dir = os.path.join(monitor_folder, str(rank))
            os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, monitor_dir, allow_early_resets=allow_early_resets)
            env.reset(seed=seed + rank)  # pyright: ignore[reportUnknownMemberType]
            return env

        set_random_seed(seed)
        return _init

    # If not wanting vectorized envs
    env: VecEnv | Monitor[dict[str, np.ndarray], np.ndarray]
    if no_vec and num_envs == 1:
        env = _make_sb3_env(0)()
    else:
        # When using one environment, no need to start subprocesses
        if num_envs == 1 or not use_subprocess:
            env = DummyVecEnv([_make_sb3_env(i) for i in range(num_envs)])
        else:
            env = SubprocVecEnv(
                [_make_sb3_env(i) for i in range(num_envs)], start_method=start_method
            )

    return env


# Linear scheduler for RL agent parameters
def linear_schedule(
    initial_value: float, final_value: float = 0.0
) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    Parameters
    ----------
    initial_value: float
        The initial value.
    final_value: float
        The final value.

    Returns
    -------
    Callable[[float], float]: The linear learning rate schedule.
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        if not initial_value > 0.0:
            raise ValueError(
                "linear_schedule work only with positive decreasing values"
            )

    def func(progress: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        Parameters
        ----------
        progress: float
            The progress.

        Returns
        -------
        float: The linear learning rate schedule.
        """
        return final_value + progress * (initial_value - final_value)

    return func


# AutoSave Callback
class AutoSave(BaseCallback):
    """
    Callback for saving a model, it is saved every ``check_freq`` steps.

    Parameters
    ----------
    check_freq: int
        The frequency of the check.
    save_path: str
        The path to the folder where the model will be saved.
    filename_prefix: str
        The filename prefix.
    verbose: int
        The verbosity level.
    save_best_model: bool
        Whether to save the best model.
    best_check_freq: int | None
        The frequency of the best check.
    monitor_folder: str | None
        The folder to save the monitor.
    num_best_envs: int
        The number of best environments.
    """

    def __init__(
        self,
        check_freq: int,
        num_envs: int,
        save_path: str,
        filename_prefix: str = "",
        starting_steps: int = 0,
        verbose: int = 1,
        save_best_model: bool = False,
        best_check_freq: int | None = None,
        monitor_folder: str | None = None,
        num_best_envs: int = 1,
    ):
        super().__init__(verbose)
        self.check_freq = int(check_freq / num_envs)
        self.num_envs = num_envs
        self.save_path_base = Path(save_path)
        self.filename = filename_prefix + "autosave_"
        self.filename_best = filename_prefix + "best"
        self.starting_steps = starting_steps
        self.best_mean_reward = -np.inf
        self.save_best_model = save_best_model
        self.num_best_envs = num_best_envs
        if save_best_model:
            if monitor_folder is None:
                raise ValueError(
                    "monitor_folder must be provided if save_best_model is True"
                )
            if best_check_freq is None:
                raise ValueError(
                    "best_check_freq must be provided if save_best_model is True"
                )
            self.monitor_folder = monitor_folder
            self.best_check_freq = int(best_check_freq / num_envs)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if self.verbose > 0:
                self.logger.info(f"Saving latest model to {self.save_path_base}")  # pyright: ignore[reportUnknownMemberType]
            # Save the agent
            self.model.save(
                self.save_path_base
                / (
                    self.filename
                    + str(self.starting_steps + self.n_calls * self.num_envs)
                )
            )

        if self.save_best_model:
            if self.n_calls % self.best_check_freq == 0:
                # Retrieve training reward
                mean_rewards: list[float] = []
                for env_idx in range(self.num_best_envs):
                    x, y = ts2xy(
                        load_results(os.path.join(self.monitor_folder, str(env_idx))),
                        "timesteps",
                    )
                    if len(x) > 0:
                        # Mean training reward over the last 100 episodes
                        mean_rewards.append(float(np.mean(y[-100:])))

                # New best model, you could save the agent here
                if len(mean_rewards) > 0:
                    mean_reward = float(np.mean(mean_rewards))
                    if self.verbose >= 1:
                        self.logger.info(f"Num timesteps: {self.num_timesteps}")  # pyright: ignore[reportUnknownMemberType]
                        self.logger.info(  # pyright: ignore[reportUnknownMemberType]
                            f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                        )
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        if self.verbose >= 1:
                            self.logger.info(  # pyright: ignore[reportUnknownMemberType]
                                f"Saving new best model to {self.save_path_base}"
                            )
                        self.model.save(self.save_path_base / self.filename_best)

        return True


# Initialize the starting steps number
class StartingSteps(BaseCallback):
    """
    Set the starting number of steps.

    Parameters
    ----------
    starting_steps: int
        The starting number of steps.
    """

    def __init__(self, starting_steps: int) -> None:
        super().__init__()
        self.starting_steps = starting_steps

    def _init_callback(self) -> None:
        self.model.num_timesteps = self.starting_steps

    def _on_step(self) -> bool:
        return True


# Custom Metrics
class CustomMetrics(BaseCallback):
    """
    Custom callback for logging values from the environment info dictionary.

    Automatically detects and logs all metrics with prefix 'custom_metrics/' in the info dict.

    Parameters
    ----------
    verbose: int
        The verbosity level.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.value_buffer: dict[str, list[Any]] = {}

    def _on_step(self) -> bool:
        """
        On step callback.

        Returns
        -------
        bool:
            True if the step is successful.
        """
        # Get info dict from locals
        infos = self.locals["infos"]

        # Collect all metrics from first info dict to initialize buffer if needed
        if len(infos) > 0:
            for key in infos[0].keys():
                if key.startswith("custom_metrics/") and key not in self.value_buffer:
                    self.value_buffer[key] = []

        # Aggregate values across all environments
        for key in self.value_buffer.keys():
            values: list[Any] = []
            for info in infos:
                if key in info:
                    values.append(info[key])

            if values:
                # Calculate mean if we have any values
                mean_value = np.mean(values)
                self.value_buffer[key].append(mean_value)

        # Log values every step
        for key in list(self.value_buffer.keys()):
            if self.value_buffer[key]:
                # Calculate mean over collected values
                mean_value = np.mean(self.value_buffer[key])
                # Log to tensorboard
                self.logger.record(
                    key, mean_value
                )  # key already includes "custom_metrics/" prefix
                # Clear buffer
                self.value_buffer[key] = []

        return True
