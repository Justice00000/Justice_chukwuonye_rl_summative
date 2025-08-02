import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from environment.custom_env import CleanWaterAgentEnv


# Function to create environment instance
def create_env():
    return CleanWaterAgentEnv()


# Vectorized environments
train_env = DummyVecEnv([lambda: Monitor(create_env())])
eval_env = DummyVecEnv([lambda: Monitor(create_env())])  # wrapped with Monitor too

# Logging and checkpoint directories
log_dir = "./tensorboard_logs/dqn_fixed/"
checkpoint_dir = "./checkpoints/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Evaluation callback to save best model and log results
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=checkpoint_dir,
    log_path=log_dir,
    eval_freq=10000,
    deterministic=True,
    render=False,
    n_eval_episodes=10,
    verbose=1
)

# Checkpoint callback to save periodically
checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path=checkpoint_dir,
    name_prefix="dqn_checkpoint"
)

# Define the DQN model
model = DQN(
    "MlpPolicy",
    train_env,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=100_000,
    batch_size=32,
    gamma=0.99,
    train_freq=(1, "episode"),
    target_update_interval=2000,
    exploration_fraction=0.3,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.02,
    tensorboard_log=log_dir,
    policy_kwargs=dict(net_arch=[256, 256])
)

print("🚀 Starting training with improved settings...")

# Begin training
model.learn(
    total_timesteps=1_000_000,
    callback=[eval_callback, checkpoint_callback],
    progress_bar=True
)

# Save final model
model.save("dqn_clean_water_fixed")
print("✅ Training complete! Model saved as 'dqn_clean_water_fixed'")
print("📊 To view progress, run: tensorboard --logdir=tensorboard_logs/dqn_fixed/")