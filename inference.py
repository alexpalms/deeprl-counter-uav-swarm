import argparse
from environment.environment import Environment
from modifiers.metric_damage import RewardWrapper
from control_policies.random import Agent as RandomAgent
from control_policies.distance_based import Agent as ClassicAgent
from control_policies.deeprl.deeprl import Agent as DeepRlAgent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default="deeprl", help='Type of control policy')
    parser.add_argument('--n_episodes', type=int, default=1, help='How many episodes to run')
    parser.add_argument('--no_render', action='store_true', help='Disable rendering')
    opt = parser.parse_args()
    print(opt)

    if opt.n_episodes > 1:
        opt.no_render = True

    env = Environment(render_mode="human")
    env = RewardWrapper(env)

    if opt.policy == "random":
        agent = RandomAgent()
    elif opt.policy == "deeprl":
        agent = DeepRlAgent()
    elif opt.policy == "classic":
        agent = ClassicAgent()
    else:
        raise Exception(f"Unrecognized policy type: {opt.policy}")

    obs, info = env.reset()
    if not opt.no_render:
        env.render()
    cumulative_reward = 0
    episode_counter = 0
    while episode_counter < opt.n_episodes:
        if not opt.no_render:
            env.render()
        action = agent.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        cumulative_reward += reward
        if done:
            print("==========================")
            print(f"Cumulative reward [%Damage]: {cumulative_reward}")
            if not opt.no_render:
                frame = env.render()
            obs, info = env.reset()
            episode_counter += 1

    env.close()