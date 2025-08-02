import os
import pygame
import numpy as np
import imageio
from stable_baselines3 import DQN
from environment.custom_env import CleanWaterAgentEnv

CELL_SIZE = 80
MARGIN = 2
FPS = 4
EPISODES = 3
VIDEO_PATH = "videos/trained_agent_run.mp4"

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

def main():
    os.makedirs("videos", exist_ok=True)
    env = CleanWaterAgentEnv(render_mode="human")
    model = DQN.load("dqn_clean_water_fixed.zip")  # You can change to PPO or A2C

    pygame.init()
    screen = pygame.display.set_mode((env.grid_size * (CELL_SIZE + MARGIN), env.grid_size * (CELL_SIZE + MARGIN)))
    pygame.display.set_caption("Trained Agent Simulation")
    clock = pygame.time.Clock()

    writer = imageio.get_writer(VIDEO_PATH, fps=FPS)

    for ep in range(1, EPISODES + 1):
        obs, _ = env.reset()
        done = False
        while not done:
            frame_surface = draw_grid(obs)
            screen.blit(frame_surface, (0, 0))
            pygame.display.flip()

            # Save frame to video
            frame = pygame.surfarray.array3d(pygame.display.get_surface())
            frame = np.transpose(frame, (1, 0, 2))  # Convert to (H, W, C)
            writer.append_data(frame)

            # Predict action
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            clock.tick(FPS)

    writer.close()
    pygame.quit()
    print(f"🎥 Saved trained agent video to {VIDEO_PATH}")

if __name__ == "__main__":
    main()