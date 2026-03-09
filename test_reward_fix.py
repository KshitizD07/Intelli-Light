"""
Test script to validate reward function fixes.
Run this AFTER making each fix to verify it works correctly.
"""

from rl.reward_function import RewardCalculator
import numpy as np

print("=" * 60)
print("TESTING REWARD FUNCTION FIXES")
print("=" * 60)

calc = RewardCalculator()

# ============================================================================
# TEST 1: FAIRNESS REWARD FIX
# ============================================================================
print("\n" + "=" * 60)
print("TEST 1: Fairness Reward (Should penalize imbalance)")
print("=" * 60)

# Scenario A: Perfectly balanced traffic
print("\nScenario A: Balanced Traffic")
balanced_queues = np.array([5.0, 5.0, 5.0, 5.0])
balanced_waits = np.array([10.0, 10.0, 10.0, 10.0])

reward_balanced = calc.calculate_reward(
    queues=balanced_queues,
    wait_times=balanced_waits,
    arrived_count=10,
    current_step=10,
    action_direction=0
)

components_balanced = calc.get_component_breakdown()
print(f"  Queues: {balanced_queues}")
print(f"  Wait times: {balanced_waits}")
print(f"  Fairness component: {components_balanced['fairness']:.3f}")
print(f"  Total reward: {reward_balanced:.2f}")

# Scenario B: Highly imbalanced traffic
print("\nScenario B: Imbalanced Traffic")
calc.reset()

imbalanced_queues = np.array([20.0, 2.0, 18.0, 1.0])
imbalanced_waits = np.array([60.0, 5.0, 55.0, 3.0])

reward_imbalanced = calc.calculate_reward(
    queues=imbalanced_queues,
    wait_times=imbalanced_waits,
    arrived_count=10,
    current_step=10,
    action_direction=0
)

components_imbalanced = calc.get_component_breakdown()
print(f"  Queues: {imbalanced_queues}")
print(f"  Wait times: {imbalanced_waits}")
print(f"  Fairness component: {components_imbalanced['fairness']:.3f}")
print(f"  Total reward: {reward_imbalanced:.2f}")

# Validation
print("\n" + "-" * 60)
print("VALIDATION:")
fairness_balanced = components_balanced['fairness']
fairness_imbalanced = components_imbalanced['fairness']

print(f"  Balanced fairness: {fairness_balanced:.3f}")
print(f"  Imbalanced fairness: {fairness_imbalanced:.3f}")

if abs(fairness_balanced) < 1.0 and fairness_imbalanced < -2.0:
    print("  ✅ PASS: Balanced has low penalty, imbalanced has high penalty")
    test1_pass = True
else:
    print("  ❌ FAIL: Fairness penalty not working correctly!")
    print(f"     Expected: balanced ≈ 0, imbalanced < -2")
    print(f"     Got: balanced = {fairness_balanced:.3f}, imbalanced = {fairness_imbalanced:.3f}")
    test1_pass = False

# ============================================================================
# TEST 2: REWARD BALANCE (After weight changes)
# ============================================================================
print("\n" + "=" * 60)
print("TEST 2: Reward Component Balance")
print("=" * 60)

calc.reset()

# Typical traffic scenario
typical_queues = np.array([8.0, 10.0, 7.0, 9.0])
typical_waits = np.array([25.0, 30.0, 20.0, 28.0])
throughput = 5

reward_typical = calc.calculate_reward(
    queues=typical_queues,
    wait_times=typical_waits,
    arrived_count=throughput,
    current_step=50,
    action_direction=0
)

components = calc.get_component_breakdown()

print("\nReward Component Breakdown:")
for name, value in components.items():
    print(f"  {name:20s}: {value:8.2f}")

print(f"\nTotal Reward: {reward_typical:.2f}")

# Validation
print("\n" + "-" * 60)
print("VALIDATION:")

throughput_magnitude = abs(components['throughput'])
wait_magnitude = abs(components['wait_penalty'])
queue_magnitude = abs(components['queue_penalty'])

print(f"  Throughput magnitude: {throughput_magnitude:.2f}")
print(f"  Wait penalty magnitude: {wait_magnitude:.2f}")
print(f"  Queue penalty magnitude: {queue_magnitude:.2f}")

# Check if penalties can compete with throughput
if wait_magnitude > 0.3 * throughput_magnitude and queue_magnitude > 0:
    print("  ✅ PASS: Penalties are significant relative to throughput")
    test2_pass = True
else:
    print("  ⚠️ WARNING: Throughput might dominate other components")
    test2_pass = False

# ============================================================================
# TEST 3: REWARD RANGE (No explosions)
# ============================================================================
print("\n" + "=" * 60)
print("TEST 3: Reward Stability (No Explosions)")
print("=" * 60)

calc.reset()

# Worst-case scenario: Heavy congestion, long waits
worst_queues = np.array([50.0, 45.0, 48.0, 50.0])
worst_waits = np.array([200.0, 180.0, 190.0, 200.0])
worst_throughput = 2  # Very low throughput

# Simulate long wait for starvation
for step in range(150):
    reward_worst = calc.calculate_reward(
        queues=worst_queues,
        wait_times=worst_waits,
        arrived_count=worst_throughput + step,
        current_step=step,
        action_direction=0  # Only serve EW, ignore NS
    )

components_worst = calc.get_component_breakdown()

print("\nWorst-Case Scenario (Heavy congestion, long waits):")
print(f"  Queues: {worst_queues}")
print(f"  Wait times: {worst_waits}")
print(f"  Throughput: {worst_throughput}")
print(f"\nReward Components:")
for name, value in components_worst.items():
    print(f"  {name:20s}: {value:8.2f}")

print(f"\nTotal Reward: {reward_worst:.2f}")

# Validation
print("\n" + "-" * 60)
print("VALIDATION:")

if -10000 < reward_worst < 1000:
    print(f"  ✅ PASS: Reward in reasonable range ({reward_worst:.2f})")
    test3_pass = True
else:
    print(f"  ❌ FAIL: Reward explosion detected! ({reward_worst:.2f})")
    test3_pass = False

# Check individual components
max_component = max(abs(v) for v in components_worst.values())
if max_component < 1000:
    print(f"  ✅ PASS: No single component exploded (max: {max_component:.2f})")
else:
    print(f"  ⚠️ WARNING: Large component detected (max: {max_component:.2f})")
    for name, value in components_worst.items():
        if abs(value) > 500:
            print(f"      {name}: {value:.2f}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("FINAL TEST SUMMARY")
print("=" * 60)

tests = {
    "Fairness Logic Fix": test1_pass,
    "Reward Balance": test2_pass,
    "Reward Stability": test3_pass,
}

passed = sum(tests.values())
total = len(tests)

for test_name, result in tests.items():
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"{test_name:30s}: {status}")

print("=" * 60)
if passed == total:
    print(f"🎉 ALL TESTS PASSED ({passed}/{total})")
    print("=" * 60)
    print("\nReward function is ready for training!")
else:
    print(f"⚠️  SOME TESTS FAILED ({passed}/{total})")
    print("=" * 60)
    print("\nPlease fix the failing tests before training.")

print()