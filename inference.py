import argparse
from environment.environment import Environment
from modifiers.metric_damage import RewardWrapper
from control_policies.random import Agent as RandomAgent
from control_policies.distance_based import Agent as ClassicAgent
from control_policies.deeprl.deeprl import Agent as DeepRlAgent
import statistics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default="deeprl", help='Type of control policy')
    parser.add_argument('--n_episodes', type=int, default=1, help='How many episodes to run')
    parser.add_argument('--no_render', action='store_true', help='Disable rendering')
    opt = parser.parse_args()
    print(opt)

    if opt.n_episodes > 1:
        opt.no_render = True

    env = Environment(render_mode="human" if not opt.no_render else None)
    env = RewardWrapper(env)

    if opt.policy == "random":
        agent = RandomAgent()
    elif opt.policy == "deeprl":
        agent = DeepRlAgent()
    elif opt.policy == "classic":
        agent = ClassicAgent()
    else:
        raise Exception(f"Unrecognized policy type: {opt.policy}")

    print("==========================")
    obs, info = env.reset()
    env.render()
    cumulative_reward = [0.0]
    episode_counter = 0
    while episode_counter < opt.n_episodes:
        env.render()
        action = agent.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        cumulative_reward[episode_counter] += reward
        if done:
            print(f"Ep. # {episode_counter+1} - Cumulative reward [%Damage]: {round(cumulative_reward[episode_counter], 2)}")
            env.render()
            obs, info = env.reset()
            episode_counter += 1
            cumulative_reward.append(0.0)

    env.close()

    if opt.n_episodes > 1:
        print("==========================")
        print(f"Cumulative reward [%Damage]: AVG = {round(statistics.mean(cumulative_reward[:-1]), 2)}, STD = {round(statistics.stdev(cumulative_reward[:-1]), 2)}")