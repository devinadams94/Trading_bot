#!/usr/bin/env python3
"""
Fix the IndexError in performance history tracking
"""

import re

# Read the training script
train_file = "/home/devin/Desktop/Trading_bot/train_profitable_optimized.py"

with open(train_file, 'r') as f:
    content = f.read()

print("Fixing IndexError in performance history...")

# Fix the problematic line that calculates episodes since best
old_line = r"logger\.info\(f\"   • Episodes since best: \{episode - performance_history\['episode'\]\[performance_history\['win_rate'\]\.index\(max\(performance_history\['win_rate'\]\)\)\]\}\"\)"

new_line = '''# Calculate episodes since best safely
                    if performance_history['win_rate']:
                        max_wr = max(performance_history['win_rate'])
                        max_wr_idx = performance_history['win_rate'].index(max_wr)
                        if max_wr_idx < len(performance_history['episode']):
                            best_episode = performance_history['episode'][max_wr_idx]
                            episodes_since_best = episode - best_episode
                        else:
                            episodes_since_best = "N/A"
                    else:
                        episodes_since_best = "N/A"
                    logger.info(f"   • Episodes since best: {episodes_since_best}")'''

# Replace the problematic line
content = re.sub(old_line, new_line, content)

# Also add safety check at the beginning of performance tracking
# Find where performance_history is updated
pattern = r"(# Update performance history\n\s+performance_history\['episode'\]\.append\(episode\))"
replacement = r"""# Update performance history
            # Ensure lists are initialized
            if 'episode' not in performance_history:
                performance_history['episode'] = []
            if 'win_rate' not in performance_history:
                performance_history['win_rate'] = []
            if 'avg_return' not in performance_history:
                performance_history['avg_return'] = []
            
            performance_history['episode'].append(episode)"""

content = re.sub(pattern, replacement, content)

# Save the fixed file
with open(train_file, 'w') as f:
    f.write(content)

print("✅ IndexError fixed!")
print("The script now safely handles:")
print("- Empty performance history")
print("- Mismatched list lengths")
print("- Missing data points")