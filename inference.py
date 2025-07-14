import cv2
from neodynamics.interface import EnvironmentClient, AgentClient
from modifiers.metric_damage import RewardWrapper

if __name__ == "__main__":
    agent = AgentClient("localhost:50052")
    env = EnvironmentClient("localhost:50051")
    env = RewardWrapper(env)
    obs, info = env.reset()
    frame = env.render()
    cv2.imshow("frame", frame[:, :, ::-1])
    cv2.waitKey(1)
    cumulative_reward = 0
    while True:
        frame = env.render()
        cv2.imshow("frame", frame[:, :, ::-1])
        cv2.waitKey(1)
        action = agent.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        cumulative_reward += reward
        if done:
            print("==========================")
            print(f"Cumulative reward [%Damage]: {cumulative_reward}")
            frame = env.render()
            cv2.imshow("frame", frame[:, :, ::-1])
            cv2.waitKey(1)
            obs, info = env.reset()
            break

    env.close()