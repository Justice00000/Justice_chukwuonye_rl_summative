import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO, A2C
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_PATHS = {
    'DQN': 'model/dqn_clean_water_fixed.zip',     
    'PPO': 'model/ppo_clean_water_model.zip',     
    'A2C': 'model/a2c_clean_water_model.zip',     
    'REINFORCE': 'model/reinforce_clean_water.pth'
}

from environment.custom_env import CleanWaterAgentEnv

# REINFORCE MODEL CLASSES (Multiple architectures to try)

class REINFORCEPolicy2Layer(nn.Module):
    """2-layer Policy network for REINFORCE (matches your saved model)"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(REINFORCEPolicy2Layer, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)  # Direct to actions
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x

class REINFORCEPolicy3Layer(nn.Module):
    """3-layer Policy network for REINFORCE"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(REINFORCEPolicy3Layer, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x

# IMPROVED EVALUATION FUNCTIONS

def evaluate_sb3_model(model, env, num_episodes=100, model_name="Model"):
    """Evaluate Stable Baselines3 model (DQN, PPO, A2C)"""
    episode_rewards = []
    episode_lengths = []
    
    print(f"Evaluating {model_name}...")
    
    for episode in range(num_episodes):
        # Handle both old and new Gym API
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result  # New Gym API
        else:
            obs = reset_result  # Old Gym API
        
        episode_reward = 0
        done = False
        step_count = 0
        max_steps = 200  # Prevent infinite episodes
        
        while not done and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            
            # Handle step result
            step_result = env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result  # Old Gym API
            else:
                obs, reward, terminated, truncated, info = step_result  # New Gym API
                done = terminated or truncated
            
            episode_reward += reward
            step_count += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        # Progress update
        if (episode + 1) % 20 == 0:
            recent_avg = np.mean(episode_rewards[-20:])
            recent_length = np.mean(episode_lengths[-20:])
            print(f"  Episode {episode + 1}/{num_episodes}: Recent avg = {recent_avg:.2f}, Avg length = {recent_length:.1f}")
    
    final_mean = np.mean(episode_rewards)
    final_std = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    print(f"  {model_name} Final: {final_mean:.2f} ± {final_std:.2f}, Avg length: {avg_length:.1f}")
    
    return episode_rewards, episode_lengths

def evaluate_reinforce_model(model_path, env, num_episodes=100):
    """Evaluate REINFORCE model with improved loading"""
    episode_rewards = []
    episode_lengths = []
    
    print("Evaluating REINFORCE...")
    
    # Create a dummy environment to get dimensions
    temp_env = env.__class__()
    temp_obs = temp_env.reset()
    if isinstance(temp_obs, tuple):
        temp_obs = temp_obs[0]
    
    # Get environment dimensions
    if hasattr(temp_env.observation_space, 'shape'):
        state_dim = np.prod(temp_env.observation_space.shape)
    else:
        state_dim = len(temp_obs) if hasattr(temp_obs, '__len__') else 1
    
    action_dim = temp_env.action_space.n if hasattr(temp_env.action_space, 'n') else 2
    temp_env.close()
    
    print(f"  Detected state_dim: {state_dim}, action_dim: {action_dim}")
    
    # Try to load REINFORCE model with multiple architectures
    model = None
    loading_success = False
    
    # First, analyze the saved model to understand its architecture
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"  Checkpoint type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"  Checkpoint keys: {list(checkpoint.keys())}")
            # Look for state dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            # It's likely the model itself
            try:
                model = checkpoint
                model.eval()
                loading_success = True
                print(f"  Loaded complete model directly")
            except:
                print(f"  Could not use checkpoint as complete model")
                return [], []
        
        if not loading_success:
            # Analyze state dict to determine architecture
            layer_info = {}
            for key, tensor in state_dict.items():
                layer_info[key] = tensor.shape
                print(f"    {key}: {tensor.shape}")
            
            # Determine architecture from layer shapes
            if 'fc2.weight' in layer_info:
                fc2_shape = layer_info['fc2.weight']
                if len(fc2_shape) == 2 and fc2_shape[1] == 128 and fc2_shape[0] == action_dim:
                    # 2-layer architecture: fc1 -> fc2 (outputs actions directly)
                    print(f"  Detected 2-layer architecture")
                    model = REINFORCEPolicy2Layer(state_dim, action_dim)
                elif 'fc3.weight' in layer_info:
                    # 3-layer architecture
                    print(f"  Detected 3-layer architecture")
                    model = REINFORCEPolicy3Layer(state_dim, action_dim)
                else:
                    print(f"  Unknown architecture, trying 2-layer")
                    model = REINFORCEPolicy2Layer(state_dim, action_dim)
            else:
                print(f"  Could not determine architecture, trying 2-layer")
                model = REINFORCEPolicy2Layer(state_dim, action_dim)
            
            # Try to load the state dict
            try:
                model.load_state_dict(state_dict)
                model.eval()
                loading_success = True
                print(f"  Successfully loaded state dict")
            except Exception as load_error:
                print(f"  Error loading state dict: {load_error}")
                
                # Try with different dimensions
                print(f"  Trying alternative dimensions...")
                for alt_state_dim in [25, 20, 16, 8, 4]:
                    for alt_action_dim in [4, 3, 2]:
                        try:
                            model = REINFORCEPolicy2Layer(alt_state_dim, alt_action_dim)
                            model.load_state_dict(state_dict)
                            model.eval()
                            loading_success = True
                            print(f"  Success with dimensions: state={alt_state_dim}, action={alt_action_dim}")
                            state_dim = alt_state_dim  # Update for observation processing
                            action_dim = alt_action_dim
                            break
                        except:
                            continue
                    if loading_success:
                        break
                
                if not loading_success:
                    print(f"  Failed to load REINFORCE model with any configuration")
                    return [], []
                
    except Exception as e:
        print(f"  Error loading REINFORCE model: {e}")
        return [], []
    
    # Evaluate the model
    for episode in range(num_episodes):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
            
        episode_reward = 0
        done = False
        step_count = 0
        max_steps = 200
        
        while not done and step_count < max_steps:
            try:
                # Prepare observation
                if isinstance(obs, (list, tuple)):
                    obs_array = np.array(obs, dtype=np.float32)
                elif isinstance(obs, np.ndarray):
                    obs_array = obs.astype(np.float32)
                else:
                    obs_array = np.array([obs], dtype=np.float32)
                
                # Flatten and ensure correct dimension
                obs_flat = obs_array.flatten()
                
                # Adjust observation size if needed
                if len(obs_flat) != state_dim:
                    if len(obs_flat) < state_dim:
                        # Pad with zeros
                        obs_flat = np.pad(obs_flat, (0, state_dim - len(obs_flat)))
                    else:
                        # Truncate
                        obs_flat = obs_flat[:state_dim]
                
                obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0)
                
                # Get action from model
                with torch.no_grad():
                    action_probs = model(obs_tensor)
                    action = torch.argmax(action_probs, dim=1).item()
                    
                    # Ensure action is valid
                    action = max(0, min(action, action_dim - 1))
                
                # Take step
                step_result = env.step(action)
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                else:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                
                episode_reward += reward
                step_count += 1
                
            except Exception as e:
                print(f"  Error during REINFORCE evaluation at episode {episode}, step {step_count}: {e}")
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        if (episode + 1) % 20 == 0:
            recent_avg = np.mean(episode_rewards[-20:])
            recent_length = np.mean(episode_lengths[-20:])
            print(f"  Episode {episode + 1}/{num_episodes}: Recent avg = {recent_avg:.2f}, Avg length = {recent_length:.1f}")
    
    if episode_rewards:
        final_mean = np.mean(episode_rewards)
        final_std = np.std(episode_rewards)
        avg_length = np.mean(episode_lengths)
        print(f"  REINFORCE Final: {final_mean:.2f} ± {final_std:.2f}, Avg length: {avg_length:.1f}")
    
    return episode_rewards, episode_lengths

# LOAD ALL MODELS AND EVALUATE

def load_and_evaluate_all_models(model_paths, env_class, num_episodes=100):
    """Load all saved models and evaluate them"""
    results = {}
    lengths = {}
    
    # Check if files exist
    print("Checking model files...")
    for alg, path in model_paths.items():
        if os.path.exists(path):
            print(f"  ✓ {alg}: {path}")
        else:
            print(f"  ✗ {alg}: {path} (FILE NOT FOUND)")
    
    print(f"\nStarting evaluation with {num_episodes} episodes per algorithm...\n")
    
    # Evaluate Stable Baselines3 models
    for alg_name in ['DQN', 'PPO', 'A2C']:
        if alg_name in model_paths and os.path.exists(model_paths[alg_name]):
            try:
                env = env_class()
                
                # Load the appropriate model
                if alg_name == 'DQN':
                    model = DQN.load(model_paths[alg_name])
                elif alg_name == 'PPO':
                    model = PPO.load(model_paths[alg_name])
                elif alg_name == 'A2C':
                    model = A2C.load(model_paths[alg_name])
                
                # Evaluate
                episode_rewards, episode_lengths = evaluate_sb3_model(model, env, num_episodes, alg_name)
                results[alg_name] = episode_rewards
                lengths[alg_name] = episode_lengths
                
                env.close()
                
            except Exception as e:
                print(f"Error loading {alg_name}: {e}")
    
    # Evaluate REINFORCE
    if 'REINFORCE' in model_paths and os.path.exists(model_paths['REINFORCE']):
        try:
            env = env_class()
            episode_rewards, episode_lengths = evaluate_reinforce_model(model_paths['REINFORCE'], env, num_episodes)
            if episode_rewards:  # Only add if evaluation succeeded
                results['REINFORCE'] = episode_rewards
                lengths['REINFORCE'] = episode_lengths
            env.close()
        except Exception as e:
            print(f"Error loading REINFORCE: {e}")
    
    return results, lengths

# ENHANCED PLOTTING FUNCTIONS

def plot_comprehensive_analysis(results, lengths, save_prefix='model_analysis'):
    """Create comprehensive analysis plots"""
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'DQN': 'blue', 'PPO': 'red', 'A2C': 'green', 'REINFORCE': 'orange'}
    
    # 1. Episode Rewards with Moving Average
    for alg, rewards in results.items():
        if len(rewards) > 0:
            episodes = range(1, len(rewards) + 1)
            
            # Raw rewards (light)
            ax1.plot(episodes, rewards, color=colors.get(alg, 'black'), alpha=0.3, linewidth=1)
            
            # Moving average
            window_size = max(5, len(rewards) // 10)
            if len(rewards) >= window_size:
                moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                ma_episodes = range(window_size, len(moving_avg) + window_size)
                ax1.plot(ma_episodes, moving_avg, 
                        color=colors.get(alg, 'black'), 
                        label=f'{alg} (μ={np.mean(rewards):.2f})',
                        linewidth=2.5)
    
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Episode Rewards Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative Rewards
    for alg, rewards in results.items():
        if len(rewards) > 0:
            cumulative_rewards = np.cumsum(rewards)
            episodes = range(1, len(cumulative_rewards) + 1)
            ax2.plot(episodes, cumulative_rewards, 
                    color=colors.get(alg, 'black'), 
                    label=f'{alg} (Final: {cumulative_rewards[-1]:.1f})',
                    linewidth=2.5)
    
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Cumulative Reward')
    ax2.set_title('Cumulative Rewards')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Reward Distribution (Box Plot)
    if results:
        reward_data = [rewards for rewards in results.values() if len(rewards) > 0]
        labels = [alg for alg, rewards in results.items() if len(rewards) > 0]
        
        box_colors = [colors.get(label, 'black') for label in labels]
        bp = ax3.boxplot(reward_data, tick_labels=labels, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax3.set_ylabel('Episode Reward')
    ax3.set_title('Reward Distribution')
    ax3.grid(True, alpha=0.3)
    
    # 4. Episode Lengths
    if lengths:
        for alg, length_data in lengths.items():
            if len(length_data) > 0:
                episodes = range(1, len(length_data) + 1)
                ax4.plot(episodes, length_data, 
                        color=colors.get(alg, 'black'), 
                        label=f'{alg} (μ={np.mean(length_data):.1f})',
                        linewidth=2, alpha=0.7)
    
    ax4.set_xlabel('Episodes')
    ax4.set_ylabel('Episode Length (Steps)')
    ax4.set_title('Episode Lengths Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f'{save_prefix}_comprehensive.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comprehensive analysis saved as: {save_path}")
    return fig

def print_detailed_summary(results, lengths):
    """Print detailed performance summary"""
    print("\n" + "="*80)
    print("DETAILED PERFORMANCE ANALYSIS")
    print("="*80)
    
    if not results:
        print("No results to analyze.")
        return
    
    # Calculate statistics
    stats = {}
    for alg, rewards in results.items():
        if len(rewards) > 0:
            stats[alg] = {
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'min': np.min(rewards),
                'max': np.max(rewards),
                'median': np.median(rewards),
                'q25': np.percentile(rewards, 25),
                'q75': np.percentile(rewards, 75),
                'episodes': len(rewards)
            }
            
            if alg in lengths and len(lengths[alg]) > 0:
                stats[alg]['avg_length'] = np.mean(lengths[alg])
                stats[alg]['length_std'] = np.std(lengths[alg])
    
    # Print summary table
    print(f"{'Algorithm':<12} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'Median':<8} {'Avg Len':<8}")
    print("-" * 80)
    
    for alg, stat in stats.items():
        print(f"{alg:<12} {stat['mean']:<8.2f} {stat['std']:<8.2f} {stat['min']:<8.2f} "
              f"{stat['max']:<8.2f} {stat['median']:<8.2f} {stat.get('avg_length', 0):<8.1f}")
    
    # Identify best performer
    if stats:
        best_alg = max(stats.keys(), key=lambda x: stats[x]['mean'])
        print(f"\nBest performing algorithm: {best_alg} with mean reward {stats[best_alg]['mean']:.2f}")
        
        # Performance analysis
        print(f"\nPerformance Analysis:")
        for alg, stat in stats.items():
            consistency = "High" if stat['std'] < stat['mean'] * 0.5 else "Low"
            trend = "Positive" if stat['mean'] > 0 else "Negative"
            print(f"  {alg}: {trend} performance, {consistency} consistency")
    
    print("="*80)

# MAIN EXECUTION

if __name__ == "__main__":
    print("CleanWaterAgentEnv - Model Performance Evaluation")
    print("="*50)
    
    # Load and evaluate all models
    results, lengths = load_and_evaluate_all_models(MODEL_PATHS, CleanWaterAgentEnv, num_episodes=100)
    
    if not results:
        print("No models were successfully loaded. Please check your paths and try again.")
    else:
        print(f"\nSuccessfully evaluated {len(results)} algorithms.")
        
        # Generate comprehensive analysis
        print("\nGenerating comprehensive analysis...")
        fig = plot_comprehensive_analysis(results, lengths, 'clean_water_model_analysis')
        
        # Print detailed summary
        print_detailed_summary(results, lengths)
        
        # Additional insights
        if len(results) > 1:
            print(f"\nKey Insights:")
            all_rewards = [reward for rewards in results.values() for reward in rewards]
            overall_mean = np.mean(all_rewards)
            print(f"  - Overall mean reward across all algorithms: {overall_mean:.2f}")
            
            positive_performers = [alg for alg, rewards in results.items() if np.mean(rewards) > 0]
            if positive_performers:
                print(f"  - Algorithms with positive mean rewards: {', '.join(positive_performers)}")
            else:
                print(f"  - No algorithms achieved positive mean rewards")
                print(f"  - Consider reviewing environment rewards and training procedures")