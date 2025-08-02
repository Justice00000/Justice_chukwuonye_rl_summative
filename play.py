import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pygame
import numpy as np
from stable_baselines3 import DQN
from environment.custom_env import CleanWaterAgentEnv

CELL_SIZE = 80
MARGIN = 2
FPS = 4

COLOR_MAP = {
    0: (255, 255, 255),
    1: (0, 0, 0),  
    2: (255, 0, 0),
    9: (0, 255, 0),
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

def play_episodes(num_episodes=5):
    env = CleanWaterAgentEnv(render_mode="human")
    model = DQN.load("dqn_clean_water_fixed")

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        pygame.init()
        grid_size = env.grid_size
        screen_width = grid_size * (CELL_SIZE + MARGIN)
        screen_height = grid_size * (CELL_SIZE + MARGIN)

        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption(f"Episode {ep}")
        clock = pygame.time.Clock()

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            frame_surface = draw_grid(obs)
            screen.blit(frame_surface, (0, 0))
            pygame.display.flip()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            clock.tick(FPS)

        pygame.quit()
        print(f"Finished Episode {ep} | Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    play_episodes(num_episodes=3)