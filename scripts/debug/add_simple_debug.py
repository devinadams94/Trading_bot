#!/usr/bin/env python3
"""Add simple debug logging to understand 0% win rate issue"""

import os

# Read the training script
train_file = "/home/devin/Desktop/Trading_bot/train_profitable_optimized.py"

with open(train_file, 'r') as f:
    lines = f.readlines()

# Find key locations to add debug logging
new_lines = []
in_balanced_env = False
in_step_method = False
in_close_position = False

for i, line in enumerate(lines):
    # Track which class/method we're in
    if 'class BalancedEnvironment' in line:
        in_balanced_env = True
    elif 'class ' in line and in_balanced_env:
        in_balanced_env = False
        
    if in_balanced_env and 'def step(' in line:
        in_step_method = True
    elif in_balanced_env and line.strip().startswith('def ') and 'step' not in line:
        in_step_method = False
        
    # Add debug logging after closing positions
    if in_step_method and 'self.total_pnl += pnl' in line:
        new_lines.append(line)
        # Add debug logging with proper indentation
        indent = ' ' * 12  # Match the indentation of surrounding code
        new_lines.append(f'{indent}# Debug: Log trade result\n')
        new_lines.append(f'{indent}logger.info(f"[DEBUG] Trade closed: pnl=${{pnl:.2f}}, total_pnl=${{self.total_pnl:.2f}}, "\n')
        new_lines.append(f'{indent}          f"wins={{self.winning_trades}}, losses={{self.losing_trades}}")\n')
        continue
        
    # Add debug at start of step to track actions
    if in_step_method and 'action_name = self.actions[action]' in line:
        new_lines.append(line)
        indent = ' ' * 8  # Match indentation
        new_lines.append(f'{indent}# Debug: Log action taken\n')
        new_lines.append(f'{indent}if self.current_step % 10 == 0:  # Log every 10 steps\n')
        new_lines.append(f'{indent}    logger.info(f"[DEBUG] Step {{self.current_step}}: action={{action_name}}, "\n')
        new_lines.append(f'{indent}              f"capital=${{self.capital:.2f}}, positions={{len(self.positions)}}")\n')
        continue
        
    # Add debug for position updates
    if in_step_method and 'if should_exit:' in line:
        new_lines.append(line)
        indent = ' ' * 16  # Match indentation  
        new_lines.append(f'{indent}# Debug: Log exit reason\n')
        new_lines.append(f'{indent}exit_reason = "stop_loss" if pnl_pct <= dynamic_stop_loss else "take_profit" if pnl_pct >= dynamic_take_profit else "time"\n')
        new_lines.append(f'{indent}logger.info(f"[DEBUG] Position exit: {{exit_reason}}, pnl_pct={{pnl_pct:.2%}}")\n')
        continue
        
    # Add episode summary in reset
    if in_balanced_env and 'def reset(' in line:
        new_lines.append(line)
        # Find the next line to get proper indentation
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            indent = len(next_line) - len(next_line.lstrip())
            indent_str = ' ' * indent
            new_lines.append(f'{indent_str}# Debug: Log episode reset\n')
            new_lines.append(f'{indent_str}if hasattr(self, "winning_trades"):\n')
            new_lines.append(f'{indent_str}    logger.info(f"[DEBUG] Episode summary: wins={{self.winning_trades}}, "\n')
            new_lines.append(f'{indent_str}              f"losses={{self.losing_trades}}, total_pnl=${{getattr(self, "total_pnl", 0):.2f}}")\n')
        continue
        
    new_lines.append(line)

# Write the updated file
with open(train_file, 'w') as f:
    f.writelines(new_lines)

print("Added simple debug logging successfully")