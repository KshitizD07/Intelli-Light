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
print(f"Episode length config: {env.episode_length}")  # ← FIXED: Removed invalid accesses

for step in range(1800):  # Full episode
    # Get action from trained model
    action, _ = model.predict(obs, deterministic=True)
    
    # Take step
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1
    
    # Print progress every 5 steps OR if ending
    if step % 5 == 0 or truncated or terminated:  # ← FIXED: Use step not steps
        print(f"Step {step}: "
              f"sim_step={info['simulation_step']}, "  # ← Now available!
              f"reward={total_reward:.2f}, "
              f"throughput={info['throughput']}, "
              f"queues={info['queues']}")
    
    if terminated or truncated:
        print(f"\nEpisode ended at step {step}")
        print(f"Reason: {'terminated' if terminated else 'truncated'}")
        break

print(f"\n{'='*60}")
print(f"EPISODE COMPLETE")
print(f"{'='*60}")
print(f"Total action steps: {steps}")
print(f"Total simulation steps: {info['simulation_step']}")
print(f"Total reward: {total_reward:.2f}")
print(f"Throughput: {info['throughput']}")
print(f"Final queues: {info['queues']}")
print(f"{'='*60}")

env.close()