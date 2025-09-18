"""Command line interface."""

import argparse
import logging
import sys

from rl_cuas.evaluation.comparative_evaluation import comparative_evaluation
from rl_cuas.evaluation.evaluation import evaluation
from rl_cuas.train.training import training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_cli(config: str) -> None:
    """Train a new reinforcement learning agent.

    Parameters
    ----------
    config: str
        Path to the training configuration file.
    """
    training(config)


def evaluate_cli(policy: str, n_episodes: int, seed: int, no_render: bool) -> None:
    """Evaluate a reinforcement learning agent.

    Parameters
    ----------
    policy: str
        Type of control policy.
    n_episodes: int
        How many episodes to run.
    seed: int
        Seed.
    no_render: bool
        Disable rendering.
    """
    evaluation(n_episodes, seed, policy, no_render)


def compare_cli(n_episodes: int, seeds: list[int]) -> None:
    """Compare the performance of a reinforcement learning agent.

    Parameters
    ----------
    n_episodes: int
        How many episodes to run.
    seeds: list[int]
        Seeds.
    """
    comparative_evaluation(n_episodes, seeds)


def main() -> None:
    """Run the main function."""
    parser = argparse.ArgumentParser(description="RL-CUAS CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Train command
    train_parser = subparsers.add_parser(
        "train", help="Train a new control policy using deep reinforcement learning"
    )
    train_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training configuration file",
    )

    # Evaluate command
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate a control policy"
    )
    evaluate_parser.add_argument(
        "--policy", type=str, default="deeprl", help="Type of control policy"
    )
    evaluate_parser.add_argument(
        "--n_episodes", type=int, default=1, help="How many episodes to run"
    )
    evaluate_parser.add_argument("--seed", type=int, default=42, help="Seed")
    evaluate_parser.add_argument(
        "--no_render", action="store_true", help="Disable rendering"
    )

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare the performance of different policies: deeprl, classic, random",
    )
    compare_parser.add_argument(
        "--n_episodes",
        type=int,
        default=100,
        help="How many episodes to run",
    )
    compare_parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[10, 20, 30, 42, 50],
        help="Seeds",
    )

    args = parser.parse_args()
    logger.info(args)

    if args.command == "train":
        train_cli(args.config)
    elif args.command == "evaluate":
        evaluate_cli(args.policy, args.n_episodes, args.seed, args.no_render)
    elif args.command == "compare":
        compare_cli(args.n_episodes, args.seeds)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
