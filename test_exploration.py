#!/usr/bin/env python3
"""
Test script to verify epsilon-greedy exploration is working
"""
import numpy as np

# Simulate exploration for 100 episodes
total_episodes = 100
action_space_n = 91

print("Testing Epsilon-Greedy Exploration")
print("=" * 60)
print(f"Total episodes: {total_episodes}")
print(f"Action space: {action_space_n} actions")
print()

episode_actions = []

for episode in range(total_episodes):
    # Calculate epsilon (same formula as training script)
    epsilon = max(0.0, 0.5 * (1.0 - episode / (total_episodes * 0.5)))
    
    # Simulate 252 steps per episode
    episode_action_counts = {}
    random_actions = 0
    
    for step in range(252):
        use_random_action = (np.random.random() < epsilon)
        
        if use_random_action:
            # Random exploration - exclude action 0 (HOLD)
            action = np.random.randint(1, action_space_n)
            random_actions += 1
        else:
            # Simulated policy (would normally come from agent)
            # For testing, assume policy always picks action 0
            action = 0
        
        episode_action_counts[action] = episode_action_counts.get(action, 0) + 1
    
    unique_actions = len(episode_action_counts)
    trades = sum(count for action, count in episode_action_counts.items() if action != 0)
    
    if episode < 10 or episode % 10 == 0:
        print(f"Episode {episode:3d}: epsilon={epsilon:.3f}, random_actions={random_actions:3d}, unique_actions={unique_actions:2d}, trades={trades:3d}")
    
    episode_actions.append({
        'episode': episode,
        'epsilon': epsilon,
        'random_actions': random_actions,
        'unique_actions': unique_actions,
        'trades': trades
    })

print()
print("Summary:")
print("=" * 60)

# First 10 episodes
first_10 = episode_actions[:10]
avg_trades_first_10 = np.mean([e['trades'] for e in first_10])
print(f"First 10 episodes: avg trades = {avg_trades_first_10:.1f}")

# Last 10 episodes
last_10 = episode_actions[-10:]
avg_trades_last_10 = np.mean([e['trades'] for e in last_10])
print(f"Last 10 episodes: avg trades = {avg_trades_last_10:.1f}")

# Overall
avg_trades_all = np.mean([e['trades'] for e in episode_actions])
print(f"All episodes: avg trades = {avg_trades_all:.1f}")

print()
print("Expected behavior:")
print("- First 10 episodes should have ~126 trades (50% of 252 steps)")
print("- Last 10 episodes should have ~0 trades (epsilon=0, policy picks HOLD)")
print("- Overall should have ~63 trades per episode")

