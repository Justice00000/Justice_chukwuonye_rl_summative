import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from environment.custom_env import CleanWaterAgentEnv

def quick_test_current_env():
    """Quick test to see if current environment can ever succeed"""
    
    print("🧪 QUICK TEST: Can current environment succeed?")
    print("="*50)
    
    env = CleanWaterAgentEnv()
    
    # Test 10 different scenarios
    max_positive_reward = -float('inf')
    successful_deliveries = 0
    
    for test in range(10):
        obs, _ = env.reset()
        
        print(f"\nTest {test + 1}:")
        print("Grid:")
        for i, row in enumerate(obs):
            print(f"  {row}")
        
        # Find all zone positions
        zones = []
        for i in range(obs.shape[0]):
            for j in range(obs.shape[1]):
                if obs[i, j] in [1, 2]:
                    zones.append((i, j))
        
        print(f"Zones found: {zones}")
        
        # Try to reach each zone and deliver
        for zone_row, zone_col in zones:
            # Reset to try reaching this zone
            env.reset()
            terminated = False
            truncated = False
            
            # Navigate to zone (simple pathfinding)
            current_pos = [0, 0]  # Start position
            
            # Move down to correct row
            while current_pos[0] < zone_row and not (terminated or truncated):
                obs, reward, terminated, truncated, info = env.step(1)  # DOWN
                current_pos = info['agent_pos']
                max_positive_reward = max(max_positive_reward, reward)
                if terminated or truncated:
                    break
            
            # Move right to correct column
            while current_pos[1] < zone_col and not (terminated or truncated):
                obs, reward, terminated, truncated, info = env.step(3)  # RIGHT
                current_pos = info['agent_pos']
                max_positive_reward = max(max_positive_reward, reward)
                if terminated or truncated:
                    break
            
            # Move up if needed
            while current_pos[0] > zone_row and not (terminated or truncated):
                obs, reward, terminated, truncated, info = env.step(0)  # UP
                current_pos = info['agent_pos']
                max_positive_reward = max(max_positive_reward, reward)
                if terminated or truncated:
                    break
            
            # Move left if needed  
            while current_pos[1] > zone_col and not (terminated or truncated):
                obs, reward, terminated, truncated, info = env.step(2)  # LEFT
                current_pos = info['agent_pos']
                max_positive_reward = max(max_positive_reward, reward)
                if terminated or truncated:
                    break
            
            # Now try DELIVER at this position
            if not (terminated or truncated):
                print(f"  Trying DELIVER at zone position ({zone_row}, {zone_col})")
                print(f"  Agent actually at: {current_pos}")
                
                obs, reward, terminated, truncated, info = env.step(4)  # DELIVER
                max_positive_reward = max(max_positive_reward, reward)
                
                print(f"    DELIVER reward: {reward}")
                print(f"    Zones served after: {info['served_zones']}")
                
                if reward > 0:
                    successful_deliveries += 1
                    print(f"    🎯 SUCCESS! Positive reward achieved!")
                    break  # Found success, move to next test
            
            if successful_deliveries > 0:
                break  # Found success, move to next test
    
    print(f"\n📊 RESULTS:")
    print(f"  Highest reward achieved: {max_positive_reward}")
    print(f"  Successful deliveries: {successful_deliveries}")
    print(f"  Success rate: {successful_deliveries/10*100:.1f}%")
    
    if successful_deliveries > 0:
        print("✅ Environment CAN succeed! Problem is with training/exploration.")
        print("🔧 Recommended fixes:")
        print("   1. Increase exploration in training")
        print("   2. Use curriculum learning")
        print("   3. Add reward shaping")
    else:
        print("❌ Environment CANNOT succeed with current reward structure!")
        print("🔧 CRITICAL: You must fix the environment reward logic first!")
        print("   1. Check DELIVER action implementation")
        print("   2. Make sure positive rewards are possible")
        print("   3. Fix zone detection logic")
    
    return successful_deliveries > 0

def test_reward_shaping():
    """Test a simple reward shaping approach"""
    
    print("\n🎯 TESTING REWARD SHAPING APPROACH")
    print("="*50)
    
    env = CleanWaterAgentEnv()
    obs, _ = env.reset()
    
    # Manual reward shaping example
    def get_shaped_reward(action, old_pos, new_pos, base_reward, zones):
        """Add reward shaping to guide agent"""
        
        # Start with base reward
        shaped_reward = base_reward
        
        # Add distance-based reward
        old_min_dist = min([abs(old_pos[0]-z[0]) + abs(old_pos[1]-z[1]) for z in zones])
        new_min_dist = min([abs(new_pos[0]-z[0]) + abs(new_pos[1]-z[1]) for z in zones])
        
        if action in [0,1,2,3]:  # Movement
            if new_min_dist < old_min_dist:
                shaped_reward += 1.0  # Getting closer bonus
            elif new_min_dist > old_min_dist:
                shaped_reward -= 0.5  # Getting farther penalty
        
        # For DELIVER, if at zone position, give huge bonus
        if action == 4:
            for zone in zones:
                if new_pos[0] == zone[0] and new_pos[1] == zone[1]:
                    shaped_reward += 100.0  # Override the -10 penalty!
                    break
        
        return shaped_reward
    
    # Find zones
    zones = []
    for i in range(obs.shape[0]):
        for j in range(obs.shape[1]):
            if obs[i, j] in [1, 2]:
                zones.append((i, j))
    
    print(f"Testing reward shaping with zones: {zones}")
    
    # Test the shaping
    if zones:
        target_zone = zones[0]
        print(f"Moving toward zone at {target_zone}")
        
        total_shaped_reward = 0
        
        # Move toward first zone
        while True:
            old_pos = [0, 0]  # Simplified for demo
            
            # Move down toward zone
            obs, base_reward, terminated, truncated, info = env.step(1)
            new_pos = info['agent_pos']
            
            shaped_reward = get_shaped_reward(1, old_pos, new_pos, base_reward, zones)
            total_shaped_reward += shaped_reward
            
            print(f"  Move: base_reward={base_reward:.2f}, shaped_reward={shaped_reward:.2f}")
            
            if new_pos[0] >= target_zone[0]:
                break
        
        # Try DELIVER at target
        old_pos = new_pos.copy()
        obs, base_reward, terminated, truncated, info = env.step(4)
        new_pos = info['agent_pos']
        
        shaped_reward = get_shaped_reward(4, old_pos, new_pos, base_reward, zones)
        total_shaped_reward += shaped_reward
        
        print(f"  DELIVER: base_reward={base_reward:.2f}, shaped_reward={shaped_reward:.2f}")
        print(f"  Total shaped reward: {total_shaped_reward:.2f}")
        
        if shaped_reward > 0:
            print("✅ Reward shaping makes DELIVER attractive!")
        else:
            print("❌ Even reward shaping can't fix this environment")

if __name__ == "__main__":
    try:
        can_succeed = quick_test_current_env()
        
        if not can_succeed:
            test_reward_shaping()
            print("\n🚨 CRITICAL ACTION NEEDED:")
            print("Your environment's reward structure is fundamentally broken.")
            print("The DELIVER action never gives positive rewards.")
            print("You MUST fix the environment before training will work.")
        else:
            print("\n✅ Environment can work! Focus on improving training.")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()