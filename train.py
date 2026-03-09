from rl.traffic_env import TrafficEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class SIHMetricsCallback(BaseCallback):
    """
    Wiretaps the environment's 'info' dictionary to send real-world 
    traffic metrics directly to TensorBoard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Check if the environment passed 'infos' during this step
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                # If your environment outputs 'total_wait', log it!
                if 'total_wait' in info:
                    self.logger.record("sih_metrics/total_wait_time", info['total_wait'])
                
                # If your environment outputs 'total_throughput', log it!
                if 'total_throughput' in info:
                    self.logger.record("sih_metrics/throughput", info['total_throughput'])
        return True

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

# --- CHANGES MADE BELOW THIS LINE ---

# 1. Added tensorboard_log argument to save the dashboard data
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./sih_tensorboard_logs/")

# 2. Instantiated your callback
metrics_callback = SIHMetricsCallback()

# 3. Added the callback to the learn function
model.learn(total_timesteps=2000, callback=metrics_callback)

print("\n✓ All tests passed!")
env.close()