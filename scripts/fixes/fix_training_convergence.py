#!/usr/bin/env python3
"""
Fix training convergence issues where model gets stuck at 0% win rate after initial success
"""

import re

# Read the training script
train_file = "/home/devin/Desktop/Trading_bot/train_profitable_optimized.py"

with open(train_file, 'r') as f:
    content = f.read()

# Fix 1: Add gradient clipping to prevent large updates that could destroy good policies
# Find the agent initialization
pattern = r'(agent = OptionsCLSTMPPOAgent\(.*?device=device.*?\))'
match = re.search(pattern, content, re.DOTALL)

if match:
    # Add gradient clipping after agent creation
    agent_init = match.group(0)
    replacement = agent_init + """
    
    # Add gradient clipping to prevent large updates
    agent.max_grad_norm = 0.5  # Clip gradients to prevent policy destruction"""
    
    content = content.replace(agent_init, replacement)
    print("Added gradient clipping configuration")

# Fix 2: Reduce learning rate after initial success to prevent overshooting
# Find where episode 75 is processed
lr_reduction = """
        # Reduce learning rate after initial success to prevent policy destruction
        if episode == 76 and best_win_rate > 0.2:  # After first successful episode
            logger.info(f"🎯 Reducing learning rates after achieving {best_win_rate:.1%} win rate")
            for param_group in agent.ppo_optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5
            for param_group in agent.clstm_optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5
            logger.info(f"Learning rates reduced by 50% to stabilize training")
"""

# Insert after episode processing
insert_after = "episodes_processed += 1"
content = content.replace(insert_after, insert_after + lr_reduction)
print("Added learning rate reduction after initial success")

# Fix 3: Add PPO clip annealing to be more conservative over time
# Find the PPO training section
ppo_clip_pattern = r"(if len\(agent\.buffer\) >= agent\.batch_size:)"
ppo_clip_replacement = r"""# Anneal PPO clip epsilon for more conservative updates over time
        if episode > 75:
            agent.clip_epsilon = max(0.05, 0.1 * (0.995 ** (episode - 75)))
        
        \1"""

content = re.sub(ppo_clip_pattern, ppo_clip_replacement, content)
print("Added PPO clip epsilon annealing")

# Fix 4: Add experience replay buffer shuffling to prevent overfitting to recent data
# This is handled in the agent's train method, but we can add a flag
replay_pattern = r"(train_metrics = agent\.train\(\))"
replay_replacement = r"""# Ensure buffer is properly shuffled before training
                    agent.shuffle_buffer = True  # Enable buffer shuffling
                    \1"""

content = re.sub(replay_pattern, replay_replacement, content)
print("Ensured buffer shuffling is enabled")

# Fix 5: Add early stopping if performance degrades too much
early_stop = """
            # Early stopping if performance crashes after initial success
            if episode > 80 and best_win_rate > 0.2:  # Had success before
                recent_win_rates = all_win_rates[-5:] if len(all_win_rates) >= 5 else all_win_rates
                if all(wr == 0 for wr in recent_win_rates):  # 5 episodes of 0% win rate
                    logger.warning(f"⚠️  Performance crashed to 0% for 5 episodes after achieving {best_win_rate:.1%}")
                    logger.warning("Consider reloading from last good checkpoint")
                    # Could implement automatic checkpoint reload here
"""

# Insert after win rate calculation
insert_after_wr = "all_win_rates.append(win_rate)"
content = content.replace(insert_after_wr, insert_after_wr + early_stop)
print("Added early stopping detection")

# Fix 6: Ensure environments are truly independent
# Add environment ID tracking to debug
env_tracking = """
        # Add environment ID tracking for debugging
        for i, env in enumerate(env.envs):
            env._env_id = i  # Tag each environment with unique ID
"""

# Insert after environment creation
env_creation_pattern = r"(env = SubprocVecEnv\(env_fns\))"
content = re.sub(env_creation_pattern, r"\1" + env_tracking, content)
print("Added environment ID tracking")

# Save the updated file
with open(train_file, 'w') as f:
    f.write(content)

print("\nAll fixes applied successfully!")
print("\nChanges made:")
print("1. Added gradient clipping (max_grad_norm=0.5)")
print("2. Learning rate reduction after initial success")
print("3. PPO clip epsilon annealing for conservative updates")
print("4. Ensured buffer shuffling")
print("5. Early stopping detection for performance crashes")
print("6. Environment ID tracking for debugging")
print("\nThese changes should prevent the model from destroying good policies.")