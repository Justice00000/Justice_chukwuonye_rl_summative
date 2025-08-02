import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from environment.custom_env import CleanWaterAgentEnv

# Load the trained model
model_path = "model/ppo_clean_water_model"
model = PPO.load(model_path)

# Create environment
env = CleanWaterAgentEnv(render_mode="human")

# Number of episodes to evaluate
episodes = 10
rewards = []

for episode in range(episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    rewards.append(total_reward)
    print(f"Episode {episode + 1} finished with reward: {total_reward:.2f}")

env.close()

# Print average reward
average_reward = np.mean(rewards)
print(f"\nAverage reward over {episodes} episodes: {average_reward:.2f}")
print(f"Average reward over {episodes} episodes: {np.mean(rewards):.2f}")
print(f"Standard deviation: {np.std(rewards):.2f}")