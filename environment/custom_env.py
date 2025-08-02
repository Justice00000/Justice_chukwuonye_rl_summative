import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class CleanWaterAgentEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, grid_size=5, num_risk_zones=3, num_obstacles=3, render_mode=None):
        super(CleanWaterAgentEnv, self).__init__()
        self.grid_size = grid_size
        self.num_risk_zones = num_risk_zones
        self.num_obstacles = num_obstacles
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(low=0, high=9, shape=(self.grid_size, self.grid_size), dtype=np.int8)

        self.agent_pos = (0, 0)
        self.grid = None
        self.visited = set()
        self.steps_taken = 0
        self.max_steps = grid_size * grid_size * 2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.steps_taken = 0
        self.visited = set()

        # Place agent
        self.agent_pos = (0, 0)

        # Place obstacles
        for _ in range(self.num_obstacles):
            while True:
                pos = tuple(np.random.randint(0, self.grid_size, size=2))
                if pos != self.agent_pos and self.grid[pos] == 0:
                    self.grid[pos] = 1  # Obstacle
                    break

        # Place risk zones
        for _ in range(self.num_risk_zones):
            while True:
                pos = tuple(np.random.randint(0, self.grid_size, size=2))
                if pos != self.agent_pos and self.grid[pos] == 0:
                    self.grid[pos] = 2  # Risk zone
                    break

        obs = self._get_obs()
        info = {
            "agent_pos": self.agent_pos,
            "steps_taken": self.steps_taken,
            "risk_zones_remaining": int(np.sum(self.grid == 2))
        }
        return obs, info

    def step(self, action):
        old_pos = self.agent_pos
        x, y = old_pos

        if action == 0 and x > 0: x -= 1      # Up
        elif action == 1 and x < self.grid_size - 1: x += 1  # Down
        elif action == 2 and y > 0: y -= 1    # Left
        elif action == 3 and y < self.grid_size - 1: y += 1  # Right

        new_pos = (x, y)
        self.steps_taken += 1
        reward = -0.2  # Small base penalty for any move
        terminated = False
        truncated = False

        if self.grid[new_pos] == 1:
            reward = -1.0  # Hit obstacle
            terminated = True
        elif self.grid[new_pos] == 2:
            reward = 3.0  # Cleared a risk zone
            self.grid[new_pos] = 0
        elif new_pos in self.visited:
            reward = -0.5  # Strong penalty for loops
        else:
            reward = 0.3  # Exploration reward

        self.visited.add(new_pos)
        self.agent_pos = new_pos

        if np.sum(self.grid == 2) == 0:
            reward += 5.0
            terminated = True

        if self.steps_taken >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {
            "agent_pos": self.agent_pos,
            "steps_taken": self.steps_taken,
            "risk_zones_remaining": int(np.sum(self.grid == 2))
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        obs = np.copy(self.grid)
        x, y = self.agent_pos
        obs[x, y] = 9  # Represent agent as 9
        return obs

    def render(self):
        if self.render_mode == "ansi":
            output = ""
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if (i, j) == self.agent_pos:
                        output += "A "
                    elif self.grid[i, j] == 0:
                        output += ". "
                    elif self.grid[i, j] == 1:
                        output += "X "
                    elif self.grid[i, j] == 2:
                        output += "R "
                output += "\n"
            return output
        else:
            print(self.render())

    def close(self):
        pass