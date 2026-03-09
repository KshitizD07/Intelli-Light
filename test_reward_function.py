"""
Test script for rl/reward_function.py

Tests the reward calculation logic including:
- Multi-objective reward components
- Weight tuning
- Curriculum scaling
- Edge cases
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
        from rl.reward_function import RewardCalculator, calculate_simple_reward
        print("✓ Module imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_initialization():
    """Test RewardCalculator initialization."""
    print("\nTesting initialization...")
    
    from rl.reward_function import RewardCalculator
    
    try:
        # Default initialization
        calc = RewardCalculator()
        print("✓ Default initialization successful")
        
        # Custom weights
        calc2 = RewardCalculator(
            wait_time_weight=-2.0,
            throughput_weight=1.0,
            fairness_weight=-0.5
        )
        print("✓ Custom weights initialization successful")
        
        # Curriculum stage
        calc3 = RewardCalculator(curriculum_stage=2)
        print("✓ Curriculum stage initialization successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False


def test_reward_calculation():
    """Test basic reward calculation."""
    print("\nTesting reward calculation...")
    
    from rl.reward_function import RewardCalculator
    
    calc = RewardCalculator()
    
    tests_passed = 0
    tests_total = 3
    
    # Test 1: Light traffic scenario
    tests_total += 1
    queues = np.array([2, 3, 2, 1], dtype=np.float32)
    wait_times = np.array([5, 8, 6, 4], dtype=np.float32)
    arrived = 10
    
    reward = calc.calculate_reward(queues, wait_times, arrived, current_step=10)
    
    if isinstance(reward, float):
        print(f"✓ Light traffic reward: {reward:.2f}")
        tests_passed += 1
    else:
        print(f"✗ Invalid reward type: {type(reward)}")
    
    # Test 2: Heavy traffic scenario
    tests_total += 1
    queues = np.array([15, 12, 18, 14], dtype=np.float32)
    wait_times = np.array([45, 50, 60, 55], dtype=np.float32)
    arrived = 15
    
    reward2 = calc.calculate_reward(queues, wait_times, arrived, current_step=20)
    
    if reward2 < reward:  # Heavy traffic should have worse reward
        print(f"✓ Heavy traffic penalty: {reward2:.2f} < {reward:.2f}")
        tests_passed += 1
    else:
        print(f"✗ Heavy traffic reward too high")
    
    # Test 3: Emergency bonus
    tests_total += 1
    queues = np.array([5, 5, 5, 5], dtype=np.float32)
    wait_times = np.array([10, 10, 10, 10], dtype=np.float32)
    arrived = 20
    
    reward_normal = calc.calculate_reward(
        queues, wait_times, arrived, current_step=30,
        emergency_active=False
    )
    
    calc.reset()  # Reset for fair comparison
    
    reward_emergency = calc.calculate_reward(
        queues, wait_times, arrived, current_step=30,
        emergency_active=True, action_direction=0
    )
    
    if reward_emergency > reward_normal:
        print(f"✓ Emergency bonus applied: {reward_emergency:.2f} > {reward_normal:.2f}")
        tests_passed += 1
    else:
        print(f"✗ Emergency bonus not working")
    
    print(f"\nPassed {tests_passed}/{tests_total} reward calculation tests")
    return tests_passed == tests_total


def test_component_breakdown():
    """Test individual reward components."""
    print("\nTesting reward component breakdown...")
    
    from rl.reward_function import RewardCalculator
    
    calc = RewardCalculator()
    
    queues = np.array([5, 6, 4, 5], dtype=np.float32)
    wait_times = np.array([15, 18, 12, 14], dtype=np.float32)
    arrived = 8
    
    reward = calc.calculate_reward(queues, wait_times, arrived, current_step=10)
    components = calc.get_component_breakdown()
    
    expected_components = [
        'throughput', 'efficiency', 'fairness',
        'wait_penalty', 'queue_penalty', 'starvation', 'emergency_bonus'
    ]
    
    tests_passed = 0
    tests_total = len(expected_components)
    
    for comp in expected_components:
        if comp in components:
            print(f"✓ {comp}: {components[comp]:.3f}")
            tests_passed += 1
        else:
            print(f"✗ Missing component: {comp}")
    
    # Check that components sum approximately to total reward
    component_sum = sum(components.values())
    curriculum_mult = 1.0  # Stage 0
    expected_total = component_sum * curriculum_mult
    
    if abs(reward - expected_total) < 0.1:
        print(f"✓ Components sum to total reward")
        tests_passed += 1
        tests_total += 1
    else:
        print(f"✗ Component sum mismatch: {component_sum:.2f} vs {reward:.2f}")
        tests_total += 1
    
    print(f"\nPassed {tests_passed}/{tests_total} component tests")
    return tests_passed == tests_total


def test_fairness_metric():
    """Test that fairness rewards balanced queues."""
    print("\nTesting fairness metric...")
    
    from rl.reward_function import RewardCalculator
    
    calc = RewardCalculator(fairness_weight=-1.0)
    
    # Balanced queues
    balanced_queues = np.array([5, 5, 5, 5], dtype=np.float32)
    wait_times = np.array([10, 10, 10, 10], dtype=np.float32)
    
    reward_balanced = calc.calculate_reward(
        balanced_queues, wait_times, arrived_count=10, current_step=10
    )
    fairness_balanced = calc.get_component_breakdown()['fairness']
    
    calc.reset()
    
    # Unbalanced queues
    unbalanced_queues = np.array([15, 2, 18, 1], dtype=np.float32)
    
    reward_unbalanced = calc.calculate_reward(
        unbalanced_queues, wait_times, arrived_count=10, current_step=20
    )
    fairness_unbalanced = calc.get_component_breakdown()['fairness']
    
    if fairness_balanced > fairness_unbalanced:
        print(f"✓ Fairness metric working")
        print(f"  Balanced: {fairness_balanced:.3f}")
        print(f"  Unbalanced: {fairness_unbalanced:.3f}")
        return True
    else:
        print(f"✗ Fairness metric not working correctly")
        return False


def test_starvation_penalty():
    """Test starvation penalty for ignored directions."""
    print("\nTesting starvation penalty...")
    
    from rl.reward_function import RewardCalculator
    from configs.parameters import SignalTiming
    
    calc = RewardCalculator()
    
    queues = np.array([5, 5, 5, 5], dtype=np.float32)
    wait_times = np.array([10, 10, 10, 10], dtype=np.float32)
    
    # Serve EW repeatedly, ignore NS
    for step in range(SignalTiming.MAX_WAIT + 20):
        reward = calc.calculate_reward(
            queues, wait_times, arrived_count=step,
            current_step=step, action_direction=0  # Always EW
        )
    
    starvation = calc.get_component_breakdown()['starvation']
    
    if starvation < -10:  # Should have significant penalty
        print(f"✓ Starvation penalty applied: {starvation:.2f}")
        return True
    else:
        print(f"✗ Starvation penalty too small: {starvation:.2f}")
        return False


def test_reset():
    """Test reset functionality."""
    print("\nTesting reset...")
    
    from rl.reward_function import RewardCalculator
    
    calc = RewardCalculator()
    
    # Calculate some rewards
    queues = np.array([5, 5, 5, 5], dtype=np.float32)
    wait_times = np.array([10, 10, 10, 10], dtype=np.float32)
    
    for i in range(10):
        calc.calculate_reward(queues, wait_times, arrived_count=i*5, current_step=i)
    
    # Reset
    calc.reset()
    
    # Check that state is cleared
    if calc.previous_arrived_count == 0:
        print("✓ Reset cleared previous_arrived_count")
        
        # Check that next throughput calculation is correct
        reward = calc.calculate_reward(queues, wait_times, arrived_count=10, current_step=0)
        throughput = calc.get_component_breakdown()['throughput']
        
        if throughput > 0:  # Should detect 10 new arrivals
            print("✓ Reset allows fresh throughput calculation")
            return True
        else:
            print("✗ Throughput not calculated after reset")
            return False
    else:
        print("✗ Reset didn't clear state")
        return False


def test_weight_updates():
    """Test dynamic weight updates."""
    print("\nTesting weight updates...")
    
    from rl.reward_function import RewardCalculator
    
    calc = RewardCalculator(throughput_weight=1.0)
    
    queues = np.array([5, 5, 5, 5], dtype=np.float32)
    wait_times = np.array([10, 10, 10, 10], dtype=np.float32)
    
    # Calculate with initial weights
    reward1 = calc.calculate_reward(queues, wait_times, arrived_count=10, current_step=10)
    throughput1 = calc.get_component_breakdown()['throughput']
    
    # Update weights
    calc.update_weights(throughput_weight=2.0)
    calc.reset()
    
    # Calculate with new weights
    reward2 = calc.calculate_reward(queues, wait_times, arrived_count=10, current_step=10)
    throughput2 = calc.get_component_breakdown()['throughput']
    
    if abs(throughput2 - 2 * throughput1) < 0.1:
        print(f"✓ Weight update working")
        print(f"  Before: {throughput1:.3f}")
        print(f"  After: {throughput2:.3f}")
        return True
    else:
        print(f"✗ Weight update not working correctly")
        return False


def test_curriculum_scaling():
    """Test curriculum learning scaling."""
    print("\nTesting curriculum scaling...")
    
    from rl.reward_function import RewardCalculator
    
    queues = np.array([5, 5, 5, 5], dtype=np.float32)
    wait_times = np.array([10, 10, 10, 10], dtype=np.float32)
    
    rewards = []
    
    for stage in [0, 1, 2]:
        calc = RewardCalculator(curriculum_stage=stage)
        reward = calc.calculate_reward(queues, wait_times, arrived_count=10, current_step=10)
        rewards.append(reward)
        print(f"  Stage {stage}: {reward:.3f}")
    
    # Rewards should increase with stage (same traffic gets higher reward)
    if rewards[2] > rewards[1] > rewards[0]:
        print("✓ Curriculum scaling working")
        return True
    else:
        print("✗ Curriculum scaling not working")
        return False


def test_simple_reward_function():
    """Test convenience function."""
    print("\nTesting simple reward function...")
    
    from rl.reward_function import calculate_simple_reward
    
    try:
        wait_times = np.array([10, 15, 12, 14], dtype=np.float32)
        throughput = 8
        
        reward = calculate_simple_reward(wait_times, throughput)
        
        if isinstance(reward, float):
            print(f"✓ Simple reward function works: {reward:.2f}")
            return True
        else:
            print(f"✗ Invalid return type")
            return False
            
    except Exception as e:
        print(f"✗ Simple reward function failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("TESTING rl/reward_function.py")
    print("=" * 60)
    
    results = []
    
    results.append(("Import Test", test_import()))
    results.append(("Initialization Test", test_initialization()))
    results.append(("Reward Calculation Test", test_reward_calculation()))
    results.append(("Component Breakdown Test", test_component_breakdown()))
    results.append(("Fairness Metric Test", test_fairness_metric()))
    results.append(("Starvation Penalty Test", test_starvation_penalty()))
    results.append(("Reset Test", test_reset()))
    results.append(("Weight Updates Test", test_weight_updates()))
    results.append(("Curriculum Scaling Test", test_curriculum_scaling()))
    results.append(("Simple Function Test", test_simple_reward_function()))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30s}: {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print("=" * 60)
    print(f"OVERALL: {total_passed}/{total_tests} test suites passed")
    print("=" * 60)
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)