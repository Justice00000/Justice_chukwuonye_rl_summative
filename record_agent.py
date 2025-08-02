import os
import pygame
import numpy as np
import imageio
from stable_baselines3 import DQN, PPO, A2C
from environment.custom_env import CleanWaterAgentEnv

# === CONFIGURATION ===
model_path = "dqn_clean_water_fixed"  # Change to PPO/A2C path if needed
model_type = "dqn"  # "dqn", "ppo", or "a2c"
save_path = "videos/trained_agent.gif"
num_episodes = 1
FPS = 4
CELL_SIZE = 80
MARGIN = 2

COLOR_MAP = {
    0: (255, 255, 255),  # Empty
    1: (0, 0, 0),        # Obstacle
    2: (255, 0, 0),      # Risk Zone
    9: (0, 255, 0),      # Agent
}

def draw_grid(grid):
    size = grid.shape[0]
    surface = pygame.Surface((size * (CELL_SIZE + MARGIN), size * (CELL_SIZE + MARGIN)))
    surface.fill((100, 100, 100))
    for i in range(size):
        for j in range(size):
            val = grid[i, j]
            color = COLOR_MAP.get(val, (200, 200, 200))
            rect = pygame.Rect(j * (CELL_SIZE + MARGIN), i * (CELL_SIZE + MARGIN), CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, color, rect)
    return surface

# Load the right model
if model_type == "dqn":
    model = DQN.load(model_path)
elif model_type == "ppo":
    model = PPO.load(model_path)
elif model_type == "a2c":
    model = A2C.load(model_path)
else:
    raise ValueError("Invalid model type")

env = CleanWaterAgentEnv(render_mode="human")
obs, _ = env.reset()

frames = []

pygame.init()
grid_size = env.grid_size
screen_width = grid_size * (CELL_SIZE + MARGIN)
screen_height = grid_size * (CELL_SIZE + MARGIN)
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Trained Agent Simulation")
clock = pygame.time.Clock()

done = False
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    surface = draw_grid(obs)
    screen.blit(surface, (0, 0))
    pygame.display.flip()

    # Save frame
    frame_rgb = pygame.surfarray.array3d(pygame.display.get_surface())
    frame_rgb = np.transpose(frame_rgb, (1, 0, 2))  # Convert to H x W x C
    frames.append(frame_rgb)

    # Get action from trained model
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    clock.tick(FPS)

pygame.quit()

# Save as GIF
os.makedirs("videos", exist_ok=True)
imageio.mimsave(save_path, frames, fps=FPS)
print(f"✅ Saved trained agent simulation to {save_path}")