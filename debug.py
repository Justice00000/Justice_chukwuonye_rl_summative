# Debug REINFORCE model loading
import torch
import numpy as np
from environment.custom_env import CleanWaterAgentEnv

def debug_reinforce_model():
    """Debug what's inside your REINFORCE model and environment"""
    
    # Load the environment to check observation space
    env = CleanWaterAgentEnv()
    print("Environment Analysis:")
    print(f"  Observation Space: {env.observation_space}")
    print(f"  Action Space: {env.action_space}")
    
    # Get a sample observation
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]  # Handle new Gym API
    
    print(f"  Sample observation type: {type(obs)}")
    print(f"  Sample observation shape: {np.array(obs).shape}")
    print(f"  Sample observation: {obs}")
    
    # Load and inspect the REINFORCE model
    print("\nREINFORCE Model Analysis:")
    try:
        model = torch.load('model/reinforce_clean_water.pth', map_location='cpu')
        print(f"  Model type: {type(model)}")
        print(f"  Model attributes: {dir(model)}")
        
        if hasattr(model, 'state_dict'):
            state_dict = model.state_dict()
            print(f"  Model layers:")
            for name, param in state_dict.items():
                print(f"    {name}: {param.shape}")
        
        # Try to determine input size from the first layer
        if hasattr(model, 'policy'):
            print(f"  Policy network: {model.policy}")
        elif hasattr(model, 'parameters'):
            params = list(model.parameters())
            if params:
                first_layer = params[0]
                print(f"  First layer shape: {first_layer.shape}")
                print(f"  Expected input size: {first_layer.shape[1]}")
        
    except Exception as e:
        print(f"  Error loading model: {e}")
    
    return env, obs

def create_reinforce_compatible_input(obs, expected_size):
    """Convert observation to the format expected by REINFORCE model"""
    
    # Convert to numpy array
    if isinstance(obs, (list, tuple)):
        obs_array = np.array(obs, dtype=np.float32)
    elif isinstance(obs, np.ndarray):
        obs_array = obs.astype(np.float32)
    else:
        obs_array = np.array([obs], dtype=np.float32)
    
    # Flatten the observation
    obs_flat = obs_array.flatten()
    
    print(f"Original obs shape: {obs_array.shape}")
    print(f"Flattened obs size: {len(obs_flat)}")
    print(f"Expected input size: {expected_size}")
    
    # Pad or trim to match expected size
    if len(obs_flat) < expected_size:
        # Pad with zeros
        padded = np.zeros(expected_size, dtype=np.float32)
        padded[:len(obs_flat)] = obs_flat
        obs_flat = padded
        print(f"Padded to size: {len(obs_flat)}")
    elif len(obs_flat) > expected_size:
        # Trim to expected size
        obs_flat = obs_flat[:expected_size]
        print(f"Trimmed to size: {len(obs_flat)}")
    
    # Convert to torch tensor
    obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0)
    print(f"Final tensor shape: {obs_tensor.shape}")
    
    return obs_tensor

def test_reinforce_model():
    """Test REINFORCE model with corrected input"""
    
    env, sample_obs = debug_reinforce_model()
    
    try:
        model = torch.load('model/reinforce_clean_water.pth', map_location='cpu')
        
        # Try to determine expected input size
        if hasattr(model, 'parameters'):
            params = list(model.parameters())
            if params:
                expected_input_size = params[0].shape[1]
                print(f"\nTesting with expected input size: {expected_input_size}")
                
                # Create compatible input
                obs_tensor = create_reinforce_compatible_input(sample_obs, expected_input_size)
                
                # Test forward pass
                with torch.no_grad():
                    if hasattr(model, 'policy'):
                        output = model.policy(obs_tensor)
                    elif hasattr(model, '__call__'):
                        output = model(obs_tensor)
                    else:
                        output = model.forward(obs_tensor)
                    
                    print(f"Model output shape: {output.shape}")
                    print(f"Model output: {output}")
                    
                    # Get action
                    action = torch.argmax(output, dim=1).item()
                    print(f"Selected action: {action}")
                    
                    return True, expected_input_size
                    
    except Exception as e:
        print(f"Error testing model: {e}")
        return False, None
    
    return False, None

# Run the debug
if __name__ == "__main__":
    success, input_size = test_reinforce_model()
    
    if success:
        print(f"\n✅ REINFORCE model can be loaded with input size: {input_size}")
        print("You can now update the evaluation function with this input size.")
    else:
        print("\n❌ REINFORCE model needs further debugging.")
        print("Please share the output above so I can help fix the loading issue.")