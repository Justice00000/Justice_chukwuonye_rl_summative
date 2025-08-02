import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from environment.custom_env import CleanWaterAgentEnv

class PolicyNetwork(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(obs_shape[0] * obs_shape[1], 128)
        self.fc2 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = x.float().view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        logits = torch.clamp(logits, -20, 20)  # prevent extreme values
        return F.softmax(logits, dim=-1)

def train():
    env = CleanWaterAgentEnv(render_mode=None)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    policy = PolicyNetwork(obs_shape, n_actions)
    optimizer = optim.Adam(policy.parameters(), lr=0.001)

    num_episodes = 1000
    gamma = 0.99

    for episode in range(num_episodes):
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        log_probs = []
        rewards = []

        done = False
        while not done:
            probs = policy(obs)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)

            obs_, reward, terminated, truncated, _ = env.step(action.item())
            obs = torch.tensor(obs_, dtype=torch.float32).unsqueeze(0)

            rewards.append(reward)
            done = terminated or truncated

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        std = returns.std()
        if std > 1e-6:
            returns = (returns - returns.mean()) / (std + 1e-8)
        else:
            returns = returns - returns.mean()

        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        policy_loss = torch.stack(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}, Total Reward: {sum(rewards):.2f}")

    os.makedirs("model", exist_ok=True)
    torch.save(policy.state_dict(), "model/reinforce_clean_water.pth")
    print("\n✅ REINFORCE training complete and model saved.")

if __name__ == "__main__":
    print("\n🚀 Training REINFORCE Agent...")
    train()