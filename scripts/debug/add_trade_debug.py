#!/usr/bin/env python3
"""Add more detailed trade debugging"""

# Read the training script
train_file = "/home/devin/Desktop/Trading_bot/train_profitable_optimized.py"

with open(train_file, 'r') as f:
    content = f.read()

# Add debug logging to the key places where trades are closed
# 1. Add logging when positions are closed
debug_position_close = '''
            # DEBUG: Log position close details
            logger.info(f"[DEBUG-CLOSE] Closing position {i}: entry=${pos['entry_price']:.2f}, "
                      f"exit=${current_value/100:.2f}, pnl=${pnl:.2f}, "
                      f"age={position_age}, pnl_pct={pnl_pct:.2%}")
'''

# Find where positions are closed
import re
close_pattern = r'(positions_to_close\.append\(i\))'
content = re.sub(close_pattern, debug_position_close + '\n                ' + r'\1', content)

# 2. Add logging when trades are opened
debug_trade_open = '''
                # DEBUG: Log trade opening
                logger.info(f"[DEBUG-OPEN] Opening {action_name}: strike=${strike:.2f}, "
                          f"premium=${option['ask']:.2f}, contracts={contracts}, "
                          f"total_cost=${total_cost:.2f}, capital_remaining=${self.capital:.2f}")
'''

# Find where trades are opened
open_pattern = r'(self\.capital -= total_cost)'
content = re.sub(open_pattern, debug_trade_open + '\n                ' + r'\1', content)

# 3. Add logging every N steps to track progress
debug_step_progress = '''
        # DEBUG: Log progress every 25 steps
        if self.current_step % 25 == 0:
            logger.info(f"[DEBUG-PROGRESS] Step {self.current_step}: capital=${self.capital:.2f}, "
                      f"positions={len(self.positions)}, wins={self.winning_trades}, "
                      f"losses={self.losing_trades}, total_pnl=${self.total_pnl:.2f}")
'''

# Find the beginning of step method after action execution
step_start_pattern = r'(action_name = self\.actions\[action\]\n)'
content = re.sub(step_start_pattern, r'\1' + debug_step_progress + '\n', content)

# Save the updated file
with open(train_file, 'w') as f:
    f.write(content)

print("Added detailed trade debugging")