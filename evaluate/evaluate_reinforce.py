import torch
import numpy as np
from environment.custom_env import CleanWaterAgentEnv

# Define the same policy architecture used during training
class REINFORCEPolicy(torch.nn.Module):
    def __init__(self, obs_shape, n_actions):
        super(REINFORCEPolicy, self).__init__()
        self.fc1 = torch.nn.Linear(obs_shape[0] * obs_shape[1], 128)
        self.fc2 = torch.nn.Linear(128, n_actions)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

# Load environment
env = CleanWaterAgentEnv(render_mode="human")
obs_shape = env.observation_space.shape
n_actions = env.action_space.n

# Initialize and load policy
policy = REINFORCEPolicy(obs_shape, n_actions)
policy.load_state_dict(torch.load("model/reinforce_clean_water.pth"))
policy.eval()

# Evaluation loop
episodes = 10
rewards = []

for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        with torch.no_grad():
            probs = policy(obs)
            action = torch.argmax(probs).item()

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    rewards.append(total_reward)
    print(f"Episode {ep+1} finished with reward: {total_reward:.2f}")

env.close()

# Summary
print(f"\nAverage reward over {episodes} episodes: {np.mean(rewards):.2f}")
print(f"Standard deviation: {np.std(rewards):.2f}")