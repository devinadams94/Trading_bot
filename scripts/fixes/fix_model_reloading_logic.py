#!/usr/bin/env python3
"""
Fix the model reloading logic to properly check performance decline
based on average win rates over a window of episodes
"""

import re

# Read the training script
train_file = "/home/devin/Desktop/Trading_bot/train_profitable_optimized.py"

with open(train_file, 'r') as f:
    content = f.read()

print("Fixing model reloading logic...")

# Find and replace the performance decline detection logic
old_logic = r"""# Declining performance - adjust learning but maintain model continuity
                if episodes_without_improvement >= stagnation_threshold and best_win_rate > 0:
                    if recent_avg < best_win_rate \* 0\.8:  # Performance dropped by 20% - more severe threshold
                        logger\.info\(f"\\n⚠️  SEVERE PERFORMANCE DECLINE DETECTED!"\)
                        logger\.info\(f"   Recent avg: \{recent_avg:\.2%\} vs Best: \{best_win_rate:\.2%\}"\)
                        logger\.info\(f"   Episodes without improvement: \{episodes_without_improvement\}"\)
                        
                        # Option to reload best model \(only in extreme cases\)
                        reload_best_model = recent_avg < best_win_rate \* 0\.7  # 30% drop"""

new_logic = """# Declining performance - adjust learning but maintain model continuity
                if episodes_without_improvement >= stagnation_threshold and best_avg_win_rate > 0:
                    # Calculate actual performance drop percentage
                    performance_drop = (best_avg_win_rate - recent_avg) / best_avg_win_rate
                    
                    if performance_drop >= 0.20:  # 20% or more drop from best average
                        logger.info(f"\\n⚠️  PERFORMANCE DECLINE DETECTED!")
                        logger.info(f"   Recent 50-ep avg: {recent_avg:.2%}")
                        logger.info(f"   Best 50-ep avg: {best_avg_win_rate:.2%}")
                        logger.info(f"   Performance drop: {performance_drop:.1%}")
                        logger.info(f"   Episodes without improvement: {episodes_without_improvement}")
                        
                        # Reload model only on severe decline (30% or more drop)
                        reload_best_model = performance_drop >= 0.30"""

# Apply the fix
content = re.sub(old_logic, new_logic, content, flags=re.DOTALL)

# Also need to ensure best_avg_win_rate is properly tracked
# Find the initialization section
init_pattern = r"(best_win_rate = 0\.0\n)"
init_replacement = r"best_win_rate = 0.0\n    best_avg_win_rate = 0.0  # Track best average win rate\n"

# Check if best_avg_win_rate is already initialized
if "best_avg_win_rate = 0.0" not in content:
    content = re.sub(init_pattern, init_replacement, content)

# Update the section that tracks best average win rate to ensure it's properly set
# Find where best_avg_win_rate is updated
avg_wr_pattern = r"(if current_avg_wr > best_avg_win_rate \* 1\.02:.*?best_avg_win_rate = current_avg_wr)"
avg_wr_replacement = r"\1\n                # Also update for decline detection\n                if best_avg_win_rate == 0.0:\n                    best_avg_win_rate = current_avg_wr"

content = re.sub(avg_wr_pattern, avg_wr_replacement, content, flags=re.DOTALL)

# Fix the second occurrence where average model is saved
second_pattern = r"(is_new_best_avg = current_avg_win_rate > best_avg_win_rate and len\(all_win_rates\) >= save_interval)"
second_replacement = r"# Check if this is a new best average (and initialize if needed)\n                if best_avg_win_rate == 0.0 and len(all_win_rates) >= save_interval:\n                    best_avg_win_rate = current_avg_win_rate\n                is_new_best_avg = current_avg_win_rate > best_avg_win_rate and len(all_win_rates) >= save_interval"

content = re.sub(second_pattern, second_replacement, content)

# Add better logging for the moderate decline case
moderate_decline_pattern = r"else:\n                            # Just reduce learning rates without reloading\n                            logger\.info\(f\"📉 Reducing learning rates without model reload\.\.\.\"\)"

moderate_decline_replacement = """else:
                            # Just reduce learning rates without reloading
                            logger.info(f"📉 Moderate decline ({performance_drop:.1%}) - reducing learning rates without reload...")"""

content = re.sub(moderate_decline_pattern, moderate_decline_replacement, content)

# Save the fixed file
with open(train_file, 'w') as f:
    f.write(content)

print("✅ Model reloading logic fixed!")
print("\nKey changes:")
print("1. Now uses best_avg_win_rate (50-episode average) instead of best single episode")
print("2. Calculates actual performance drop percentage")
print("3. Shows clear percentage drop in logs")
print("4. Triggers reload at 30% drop from best average (not 30% of best)")
print("\nThresholds:")
print("- Moderate decline (≥20% drop): Reduce learning rate, boost exploration")
print("- Severe decline (≥30% drop): Reload best model, reduce LR, major exploration boost")