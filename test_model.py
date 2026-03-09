from rl.traffic_env import TrafficEnv
from stable_baselines3 import PPO
import numpy as np

# Load the trained model
model = PPO.load("C:/Users/kshit/cs/project/Intelli-Light/models/checkpoints/intellilight_final.zip")

# Create environment (with GUI to watch)
env = TrafficEnv(use_gui=True, curriculum_stage=0)

# Run one episode
obs, info = env.reset()
total_reward = 0
steps = 0

print("Testing trained model...")
print(f"Scenario: {info['scenario']}")

# for step in range(1800):  # Full episode
    # Get action from trained model
while True:
    action, _ = model.predict(obs, deterministic=True)
    
    # Take step
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1
    
    if steps % 100 == 0:
        print(f"Step {steps}: Reward so far = {total_reward:.2f}")
    
    if terminated or truncated:
        print(f"Episode ended at step {steps}")
        break

print(f"\n{'='*60}")
print(f"EPISODE COMPLETE")
print(f"{'='*60}")
print(f"Total steps: {steps}")
print(f"Total reward: {total_reward:.2f}")
print(f"Throughput: {info['throughput']}")
print(f"Final queues: {info['queues']}")
print(f"{'='*60}")

env.close()