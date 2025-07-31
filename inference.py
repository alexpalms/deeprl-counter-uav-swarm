import argparse
from environment.environment import Environment
from modifiers.metric_damage import CustomWrapper
from control_policies.random import Agent as RandomAgent
from control_policies.distance_based import Agent as ClassicAgent
from control_policies.deeprl.deeprl import Agent as DeepRlAgent
import statistics

def main(n_episodes, seed, policy, no_render=True):
    if n_episodes > 1:
        no_render = True

    env = Environment(render_mode="human" if not no_render else None)
    env = CustomWrapper(env)

    if policy == "random":
        agent = RandomAgent()
    elif policy == "deeprl":
        agent = DeepRlAgent(env)
    elif policy == "classic":
        agent = ClassicAgent()
    else:
        raise Exception(f"Unrecognized policy type: {policy}")

    print("==========================")
    obs, info = env.reset(seed=seed)
    env.render()
    cumulative_reward = [0.0]
    effectors_tracking = []
    effectors_weapon_utilization = []
    episode_counter = 0
    while episode_counter < n_episodes:
        env.render()
        action = agent.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        cumulative_reward[episode_counter] += reward
        if done:
            print(f"Ep. # {episode_counter+1} - Cumulative reward [%Damage]: {round(cumulative_reward[episode_counter], 2)}")
            print(f"        - Effectors kinematic states: ", info["custom_eval_metrics/effectors_kinematic_states"])
            print(f"        - Effectors weapon states: ", info["custom_eval_metrics/effectors_weapon_states"])
            effectors_tracking.append(info["custom_eval_metrics/effectors_kinematic_states"]["TRACKING"])
            effectors_weapon_utilization.append(info["custom_eval_metrics/effectors_weapon_states"]["UTILIZATION"])
            env.render()
            obs, info = env.reset()
            episode_counter += 1
            cumulative_reward.append(0.0)

    env.close()

    if n_episodes > 1:
        print("==========================")
        print(f"Cumulative reward [%Damage]: AVG = {round(statistics.mean(cumulative_reward[:-1]), 2)}, STD = {round(statistics.stdev(cumulative_reward[:-1]), 2)}")

    return cumulative_reward[:-1], effectors_tracking, effectors_weapon_utilization


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default="deeprl", help='Type of control policy')
    parser.add_argument('--n_episodes', type=int, default=1, help='How many episodes to run')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument('--no_render', action='store_true', help='Disable rendering')
    opt = parser.parse_args()
    print(opt)

    main(opt.n_episodes, opt.seed, opt.policy, opt.no_render)