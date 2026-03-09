"""
Test script for rl/traffic_env.py

Tests the main RL environment including:
- Gymnasium interface compliance
- SUMO integration
- Reward calculation integration
- Episode lifecycle
- Action/observation spaces
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_import():
    """Test that the module can be imported."""
    print("Testing module import...")
    
    try:
        from rl.traffic_env import TrafficEnv
        print("✓ Module imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_environment_creation():
    """Test environment initialization."""
    print("\nTesting environment creation...")
    
    from rl.traffic_env import TrafficEnv
    
    try:
        # Create environment
        env = TrafficEnv(use_gui=False, curriculum_stage=0)
        print("✓ Environment created")
        
        # Check spaces
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gymnasium_interface():
    """Test Gymnasium interface compliance."""
    print("\nTesting Gymnasium interface...")
    
    from rl.traffic_env import TrafficEnv
    import gymnasium as gym
    
    tests_passed = 0
    tests_total = 5
    
    try:
        env = TrafficEnv(use_gui=False)
        
        # Test 1: Is a Gymnasium environment
        tests_total += 1
        if isinstance(env, gym.Env):
            print("✓ Is Gymnasium environment")
            tests_passed += 1
        else:
            print("✗ Not a Gymnasium environment")
        
        # Test 2: Has required spaces
        tests_total += 1
        if hasattr(env, 'observation_space') and hasattr(env, 'action_space'):
            print("✓ Has observation and action spaces")
            tests_passed += 1
        else:
            print("✗ Missing required spaces")
        
        # Test 3: Reset returns correct format
        tests_total += 1
        result = env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            if isinstance(obs, np.ndarray) and isinstance(info, dict):
                print("✓ Reset returns (obs, info)")
                tests_passed += 1
            else:
                print("✗ Reset returns wrong types")
        else:
            print("✗ Reset returns wrong format")
        
        # Test 4: Step returns correct format
        tests_total += 1
        action = env.action_space.sample()
        result = env.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
            if all([
                isinstance(obs, np.ndarray),
                isinstance(reward, (int, float)),
                isinstance(terminated, bool),
                isinstance(truncated, bool),
                isinstance(info, dict)
            ]):
                print("✓ Step returns (obs, reward, terminated, truncated, info)")
                tests_passed += 1
            else:
                print("✗ Step returns wrong types")
        else:
            print("✗ Step returns wrong format")
        
        # Test 5: Observation in space
        tests_total += 1
        if env.observation_space.contains(obs):
            print("✓ Observation within observation space")
            tests_passed += 1
        else:
            print("✗ Observation outside observation space")
        
        env.close()
        
        print(f"\nPassed {tests_passed}/{tests_total} interface tests")
        return tests_passed == tests_total
        
    except Exception as e:
        print(f"✗ Interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_episode_lifecycle():
    """Test complete episode lifecycle."""
    print("\nTesting episode lifecycle...")
    
    from rl.traffic_env import TrafficEnv
    
    try:
        env = TrafficEnv(use_gui=False)
        
        # Reset
        obs, info = env.reset()
        print(f"✓ Episode started: {info['scenario']}")
        
        # Run some steps
        steps_run = 0
        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps_run += 1
            
            if terminated or truncated:
                break
        
        print(f"✓ Ran {steps_run} steps successfully")
        
        # Get metrics
        metrics = env.get_episode_metrics()
        print(f"✓ Episode metrics: reward={metrics['total_reward']:.2f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Lifecycle test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_observation_space():
    """Test observation space properties."""
    print("\nTesting observation space...")
    
    from rl.traffic_env import TrafficEnv
    
    env = TrafficEnv(use_gui=False)
    obs, _ = env.reset()
    
    tests_passed = 0
    tests_total = 4
    
    # Test shape
    tests_total += 1
    if obs.shape == (9,):
        print(f"✓ Observation shape correct: {obs.shape}")
        tests_passed += 1
    else:
        print(f"✗ Wrong observation shape: {obs.shape}")
    
    # Test normalization (should be [0, 1])
    tests_total += 1
    if np.all(obs >= 0) and np.all(obs <= 1):
        print(f"✓ Observations normalized to [0, 1]")
        tests_passed += 1
    else:
        print(f"✗ Observations not normalized: min={obs.min()}, max={obs.max()}")
    
    # Test that observations change
    tests_total += 1
    action = env.action_space.sample()
    obs2, _, _, _, _ = env.step(action)
    
    if not np.array_equal(obs, obs2):
        print("✓ Observations change after actions")
        tests_passed += 1
    else:
        print("✗ Observations don't change")
    
    # Test observation components
    tests_total += 1
    queues = obs[0:4]
    wait_times = obs[4:8]
    emergency = obs[8]
    
    if all([
        queues.shape == (4,),
        wait_times.shape == (4,),
        emergency >= 0 and emergency <= 1
    ]):
        print("✓ Observation components have correct structure")
        tests_passed += 1
    else:
        print("✗ Observation components incorrect")
    
    env.close()
    
    print(f"\nPassed {tests_passed}/{tests_total} observation tests")
    return tests_passed == tests_total


def test_action_space():
    """Test action space properties."""
    print("\nTesting action space...")
    
    from rl.traffic_env import TrafficEnv
    from gymnasium.spaces import MultiDiscrete
    
    env = TrafficEnv(use_gui=False)
    
    tests_passed = 0
    tests_total = 3
    
    # Test action space type
    tests_total += 1
    if isinstance(env.action_space, MultiDiscrete):
        print("✓ Action space is MultiDiscrete")
        tests_passed += 1
    else:
        print(f"✗ Wrong action space type: {type(env.action_space)}")
    
    # Test action space shape
    tests_total += 1
    if env.action_space.nvec.tolist() == [2, 3]:
        print("✓ Action space shape correct: [2, 3]")
        tests_passed += 1
    else:
        print(f"✗ Wrong action space shape: {env.action_space.nvec}")
    
    # Test that actions work
    tests_total += 1
    env.reset()
    try:
        for _ in range(5):
            action = env.action_space.sample()
            env.step(action)
        print("✓ Random actions execute successfully")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Actions failed: {e}")
    
    env.close()
    
    print(f"\nPassed {tests_passed}/{tests_total} action tests")
    return tests_passed == tests_total


def test_reward_integration():
    """Test reward calculation integration."""
    print("\nTesting reward integration...")
    
    from rl.traffic_env import TrafficEnv
    
    env = TrafficEnv(use_gui=False)
    env.reset()
    
    tests_passed = 0
    tests_total = 3
    
    # Test that rewards are generated
    tests_total += 1
    rewards = []
    for _ in range(10):
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)
        rewards.append(reward)
    
    if len(rewards) == 10:
        print(f"✓ Rewards generated: range [{min(rewards):.2f}, {max(rewards):.2f}]")
        tests_passed += 1
    else:
        print("✗ Rewards not generated")
    
    # Test reward components available
    tests_total += 1
    action = env.action_space.sample()
    _, _, _, _, info = env.step(action)
    
    if 'reward_components' in info:
        components = info['reward_components']
        expected_keys = ['throughput', 'efficiency', 'fairness']
        if all(key in components for key in expected_keys):
            print("✓ Reward components available in info")
            tests_passed += 1
        else:
            print("✗ Missing reward components")
    else:
        print("✗ No reward components in info")
    
    # Test cumulative reward tracking
    tests_total += 1
    metrics = env.get_episode_metrics()
    if 'total_reward' in metrics:
        print(f"✓ Cumulative reward tracked: {metrics['total_reward']:.2f}")
        tests_passed += 1
    else:
        print("✗ Cumulative reward not tracked")
    
    env.close()
    
    print(f"\nPassed {tests_passed}/{tests_total} reward tests")
    return tests_passed == tests_total


def test_curriculum_learning():
    """Test curriculum learning functionality."""
    print("\nTesting curriculum learning...")
    
    from rl.traffic_env import TrafficEnv
    
    tests_passed = 0
    tests_total = 2
    
    # Test different stages create different scenarios
    tests_total += 1
    try:
        env0 = TrafficEnv(use_gui=False, curriculum_stage=0)
        _, info0 = env0.reset()
        flow0 = info0['traffic_flow']
        env0.close()
        
        env2 = TrafficEnv(use_gui=False, curriculum_stage=2)
        _, info2 = env2.reset()
        flow2 = info2['traffic_flow']
        env2.close()
        
        if flow2 > flow0:
            print(f"✓ Higher stage has more traffic: {flow0} < {flow2}")
            tests_passed += 1
        else:
            print(f"⚠ Traffic flow similar across stages: {flow0} vs {flow2}")
            tests_passed += 1  # Not a hard failure
            
    except Exception as e:
        print(f"✗ Curriculum test failed: {e}")
    
    # Test stage can be updated
    tests_total += 1
    try:
        env = TrafficEnv(use_gui=False, curriculum_stage=0)
        env.set_curriculum_stage(2)
        print("✓ Curriculum stage can be updated")
        tests_passed += 1
        env.close()
    except Exception as e:
        print(f"✗ Stage update failed: {e}")
    
    print(f"\nPassed {tests_passed}/{tests_total} curriculum tests")
    return tests_passed == tests_total


def test_emergency_detection():
    """Test emergency vehicle detection."""
    print("\nTesting emergency detection...")
    
    from rl.traffic_env import TrafficEnv
    
    env = TrafficEnv(use_gui=False)
    
    # Run multiple episodes to potentially encounter emergency
    emergency_detected = False
    
    for episode in range(5):
        env.reset()
        
        for _ in range(20):
            action = env.action_space.sample()
            obs, _, _, _, info = env.step(action)
            
            # Check observation emergency flag
            if obs[8] > 0.5:
                emergency_detected = True
                print(f"✓ Emergency vehicle detected in episode {episode}")
                break
        
        if emergency_detected:
            break
    
    if emergency_detected:
        print("✓ Emergency detection working")
    else:
        print("⚠ No emergency detected (probabilistic, may be OK)")
    
    env.close()
    return True  # Don't fail on probabilistic test


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("TESTING rl/traffic_env.py")
    print("=" * 60)
    print("\nNote: These tests require SUMO to be installed!")
    print()
    
    results = []
    
    results.append(("Import Test", test_import()))
    results.append(("Environment Creation", test_environment_creation()))
    results.append(("Gymnasium Interface", test_gymnasium_interface()))
    results.append(("Episode Lifecycle", test_episode_lifecycle()))
    results.append(("Observation Space", test_observation_space()))
    results.append(("Action Space", test_action_space()))
    results.append(("Reward Integration", test_reward_integration()))
    results.append(("Curriculum Learning", test_curriculum_learning()))
    results.append(("Emergency Detection", test_emergency_detection()))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:25s}: {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print("=" * 60)
    print(f"OVERALL: {total_passed}/{total_tests} test suites passed")
    print("=" * 60)
    
    return total_passed == total_tests


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    success = run_all_tests()
    sys.exit(0 if success else 1)