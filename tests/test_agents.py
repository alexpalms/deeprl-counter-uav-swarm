# tests/test_agents.py
"""Test the agents module."""

from __future__ import annotations

from rl_cuas.control_policies.deeprl import DeepRlAgent
from rl_cuas.control_policies.distance_based import DistanceBasedAgent
from rl_cuas.control_policies.random import RandomAgent
from rl_cuas.environment.environment import Environment


def test_random_agent() -> None:
    """Test the random agent."""
    env = Environment()
    agent = RandomAgent()

    obs, _ = env.reset()
    while True:
        action = agent.get_action(obs)
        assert env.action_space.contains(action)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    env.close()


def test_classic_agent() -> None:
    """Test the classic agent."""
    env = Environment()
    agent = DistanceBasedAgent()

    obs, _ = env.reset()
    while True:
        action = agent.get_action(obs)
        assert env.action_space.contains(action)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    env.close()


def test_deeprl_agent() -> None:
    """Test the deeprl agent."""
    env = Environment()
    agent = DeepRlAgent(env)

    obs, _ = env.reset()
    while True:
        action = agent.get_action(obs)
        assert env.action_space.contains(action)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    env.close()
