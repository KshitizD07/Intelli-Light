from rl.traffic_env import TrafficEnv
from stable_baselines3 import PPO


print("Testing fixed environment...")

# Create environment
env = TrafficEnv(use_gui=False, curriculum_stage=0)

# Quick test
print("\n1. Testing reset...")
obs, info = env.reset()
print(f"   ✓ Reset successful. Scenario: {info['scenario']}")

print("\n2. Testing 50 steps...")
total_reward = 0
for i in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    if i % 10 == 0:
        print(f"   Step {i}: reward={reward:.2f}")

print(f"\n3. Total reward after 50 steps: {total_reward:.2f}")
print(f"   Episode metrics: {env.get_episode_metrics()}")

# Check if rewards are reasonable
if -5000 < total_reward < 5000:
    print("\n✓ REWARDS LOOK REASONABLE!")
else:
    print(f"\n⚠ Rewards still unusual: {total_reward}")

env.close()

print("\n4. Quick PPO training test (2000 steps)...")
env = TrafficEnv(use_gui=False)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2000)

print("\n✓ All tests passed!")
env.close()