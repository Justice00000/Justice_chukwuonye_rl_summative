import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment.custom_env import CleanWaterAgentEnv

# Create and wrap the environment
env = DummyVecEnv([lambda: Monitor(CleanWaterAgentEnv())])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

# Create directories
ppo_log_dir = "./tensorboard_logs/ppo/"
a2c_log_dir = "./tensorboard_logs/a2c/"
os.makedirs(ppo_log_dir, exist_ok=True)
os.makedirs(a2c_log_dir, exist_ok=True)

# PPO Model
ppo_model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    gamma=0.99,
    n_steps=2048,
    ent_coef=0.0,
    tensorboard_log=ppo_log_dir
)
ppo_model.learn(total_timesteps=500_000, callback=ProgressBarCallback())
ppo_model.save("ppo_clean_water_model")

# A2C Model
a2c_model = A2C(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=7e-4,
    gamma=0.99,
    n_steps=5,
    gae_lambda=1.0,
    ent_coef=0.0,
    tensorboard_log=a2c_log_dir
)
a2c_model.learn(total_timesteps=500_000, callback=ProgressBarCallback())
a2c_model.save("a2c_clean_water_model")

# Save VecNormalize stats
env.save("pg_vec_normalize.pkl")

print("✅ PPO and A2C training complete.")