import os
import numpy as np
import pygame
import imageio
from environment.custom_env import CleanWaterAgentEnv

# Constants
CELL_SIZE = 80
MARGIN = 2
FPS = 4
EPISODE_LENGTH = 30  # Number of steps to visualize

# Color mapping
COLOR_MAP = {
    0: (255, 255, 255),  # Empty
    1: (0, 0, 0),        # Obstacle
    2: (255, 0, 0),      # Risk zone
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

def record_random_episode(save_path="videos/random_agent.gif"):
    env = CleanWaterAgentEnv(render_mode="human")
    obs, _ = env.reset()
    frames = []

    pygame.init()
    grid_size = env.grid_size
    screen_width = grid_size * (CELL_SIZE + MARGIN)
    screen_height = grid_size * (CELL_SIZE + MARGIN)
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Random Agent")
    clock = pygame.time.Clock()

    for _ in range(EPISODE_LENGTH):
        # Render and capture
        frame_surface = draw_grid(obs)
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()

        # Convert Pygame surface to array
        frame_array = pygame.surfarray.array3d(pygame.display.get_surface())
        frame_array = np.transpose(frame_array, (1, 0, 2))  # Convert to H x W x C
        frames.append(frame_array)

        # Step randomly
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

        clock.tick(FPS)

    pygame.quit()

    # Save GIF using imageio
    imageio.mimsave(save_path, frames, fps=FPS)
    print(f"✅ Saved random agent simulation to {save_path}")

if __name__ == "__main__":
    os.makedirs("videos", exist_ok=True)
    record_random_episode()