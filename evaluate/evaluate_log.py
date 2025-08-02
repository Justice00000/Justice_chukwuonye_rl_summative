import csv
import os
import numpy as np
from environment.custom_env import CleanWaterAgentEnv
from stable_baselines3 import DQN

model = DQN.load("dqn_clean_water_fixed")
env = CleanWaterAgentEnv(render_mode="ansi")

episodes = 10
rewards = []

log_file = "evaluation_log.csv"
os.makedirs("logs", exist_ok=True)

with open(f"logs/{log_file}", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Episode", "Reward", "Steps"])

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        print(f"Episode {episode + 1} finished with reward: {total_reward:.2f}")
        print(env.render())  # Optional visual output
        writer.writerow([episode + 1, round(total_reward, 2), steps])
        rewards.append(total_reward)

    avg = np.mean(rewards)
    std = np.std(rewards)
    print(f"\nAverage reward over {episodes} episodes: {avg:.2f}")
    print(f"Standard deviation: {std:.2f}")