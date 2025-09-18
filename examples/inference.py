"""Inference script."""

import argparse
import logging
import statistics

from rl_cuas.control_policies.common import Agent
from rl_cuas.control_policies.deeprl import DeepRlAgent
from rl_cuas.control_policies.distance_based import DistanceBasedAgent
from rl_cuas.control_policies.random import RandomAgent
from rl_cuas.environment.environment import Environment
from rl_cuas.modifiers.metric_damage import CustomWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(
    n_episodes: int, seed: int, policy: str, no_render: bool = True
) -> tuple[list[float], list[float], list[float]]:
    """
    Run the inference script.

    Parameters
    ----------
    n_episodes: int
        The number of episodes to run.
    seed: int
        The seed to use for the environment.
    policy: str
        The policy to use.
    no_render: bool
        Whether to render the environment.

    Returns
    -------
    tuple[list[float], list[float], list[float]]
    """
    env: Environment | CustomWrapper
    env = Environment(render_mode="human" if not no_render else None)
    env = CustomWrapper(env)

    agent: Agent
    if policy == "random":
        agent = RandomAgent()
    elif policy == "deeprl":
        agent = DeepRlAgent(env)
    elif policy == "classic":
        agent = DistanceBasedAgent()
    else:
        raise Exception(f"Unrecognized policy type: {policy}")

    logger.info("==========================")
    obs, info = env.reset(seed=seed)
    env.render()
    cumulative_reward = [0.0]
    effectors_tracking: list[float] = []
    effectors_weapon_utilization: list[float] = []
    episode_counter = 0
    while episode_counter < n_episodes:
        env.render()
        action = agent.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        cumulative_reward[episode_counter] += reward
        if done:
            logger.info(
                f"Ep. # {episode_counter + 1} - Cumulative reward [%Damage]: {round(cumulative_reward[episode_counter], 2)}"
            )
            logger.info(
                "        - Effectors kinematic states:",
            )
            for key, value in info[
                "custom_eval_metrics/effectors_kinematic_states"
            ].items():
                logger.info(
                    f"             - {key}: {round(value, 2)}",
                )
            logger.info(
                "        - Effectors weapon states:",
            )
            for key, value in info[
                "custom_eval_metrics/effectors_weapon_states"
            ].items():
                logger.info(
                    f"             - {key}: {round(value, 2)}",
                )
            effectors_tracking.append(
                info["custom_eval_metrics/effectors_kinematic_states"]["TRACKING"]
            )
            effectors_weapon_utilization.append(
                info["custom_eval_metrics/effectors_weapon_states"]["UTILIZATION"]
            )
            env.render()
            obs, info = env.reset()
            episode_counter += 1
            cumulative_reward.append(0.0)

    env.close()  # type: ignore

    if n_episodes > 1:
        logger.info("==========================")
        logger.info(
            f"Cumulative reward [%Damage]: AVG = {round(statistics.mean(cumulative_reward[:-1]), 2)}, STD = {round(statistics.stdev(cumulative_reward[:-1]), 2)}"
        )

    return cumulative_reward[:-1], effectors_tracking, effectors_weapon_utilization


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy", type=str, default="deeprl", help="Type of control policy"
    )
    parser.add_argument(
        "--n_episodes", type=int, default=1, help="How many episodes to run"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--no_render", action="store_true", help="Disable rendering")
    opt = parser.parse_args()
    logger.info(opt)

    main(opt.n_episodes, opt.seed, opt.policy, opt.no_render)
