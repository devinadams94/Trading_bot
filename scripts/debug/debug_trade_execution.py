#!/usr/bin/env python3
"""
Add detailed trade execution debugging to understand why all trades result in losses
"""

import re

# Read the training script
train_file = "/home/devin/Desktop/Trading_bot/train_profitable_optimized.py"

with open(train_file, 'r') as f:
    content = f.read()

# Add detailed trade execution logging in the BalancedEnvironment
# Find the _execute_trade method
trade_execution_pattern = r'(def _execute_trade\(self, action_name: str, current_price: float\):)'

debug_code = r'''\1
        """Execute a trade with comprehensive debugging"""
        # DEBUG: Log every trade attempt after episode 75
        if hasattr(self, '_episode_count') and self._episode_count >= 75:
            logger.info(f"[DEBUG-TRADE-ATTEMPT] Ep{self._episode_count} Step{self.current_step}: {action_name} @ price ${current_price:.2f}")'''

content = re.sub(trade_execution_pattern, debug_code, content)

# Add debugging to position updates
position_update_pattern = r'(def _update_positions_realistic\(self\):)'

position_debug = r'''\1
        """Update positions with realistic P&L tracking and debugging"""
        # DEBUG: Log position states after episode 75
        if hasattr(self, '_episode_count') and self._episode_count >= 75 and self.current_step % 10 == 0:
            logger.info(f"[DEBUG-POSITIONS] Ep{self._episode_count} Step{self.current_step}: {len(self.positions)} positions")
            for i, pos in enumerate(self.positions[:3]):  # First 3 positions
                entry_price = pos.get('entry_price', 0)
                current_price = self._prices[self.current_step]
                price_change = (current_price - pos['entry_underlying']) / pos['entry_underlying']
                logger.info(f"  Pos{i}: {pos['option_type']} entered@${entry_price:.2f}, underlying ${pos['entry_underlying']:.2f}→${current_price:.2f} ({price_change:.1%})")'''

content = re.sub(position_update_pattern, position_debug, content)

# Add episode counter to BalancedEnvironment reset
reset_pattern = r'(def reset\(self\):.*?obs = super\(\)\.reset\(\))'

reset_debug = r'''\1
        
        # Track episode count for debugging
        if not hasattr(self, '_episode_count'):
            self._episode_count = 0
        self._episode_count += 1'''

content = re.sub(reset_pattern, reset_debug, content, flags=re.DOTALL)

# Add debugging to understand why trades immediately lose
# Find where position P&L is calculated
pnl_calc_pattern = r'(# Calculate P&L.*?pnl_pct = pnl / entry_cost)'

pnl_debug = r'''\1
            
            # DEBUG: Log P&L calculation details after episode 75
            if hasattr(self, '_episode_count') and self._episode_count >= 75 and self.current_step % 10 == 0:
                logger.info(f"[DEBUG-PNL] Pos {i}: entry_cost=${entry_cost:.2f}, current_value=${current_value:.2f}, pnl=${pnl:.2f} ({pnl_pct:.1%})")
                logger.info(f"  Price movement: {price_change:.3%}, Option change: {option_price_change:.3%}, Delta: {delta}")'''

content = re.sub(pnl_calc_pattern, pnl_debug, content, flags=re.DOTALL)

# Add check for data continuity
data_check = '''
        # DEBUG: Check if data is jumping between episodes
        if hasattr(self, '_episode_count') and self._episode_count >= 75 and self._episode_count <= 77:
            if hasattr(self, '_last_price'):
                price_jump = abs((current_price - self._last_price) / self._last_price)
                if price_jump > 0.1:  # More than 10% jump
                    logger.warning(f"[DEBUG-DATA-JUMP] Large price jump: ${self._last_price:.2f} → ${current_price:.2f} ({price_jump:.1%})")
            self._last_price = current_price
'''

# Insert after getting current price in step method
step_pattern = r'(current_price = self\._prices\[self\.current_step\])'
content = re.sub(step_pattern, r'\1' + data_check, content)

# Save the updated file
with open(train_file, 'w') as f:
    f.write(content)

print("Added comprehensive trade execution debugging")
print("\nDebug points added:")
print("1. Trade attempt logging - shows every trade attempted")
print("2. Position state logging - shows position details every 10 steps")
print("3. P&L calculation logging - shows how losses are calculated")
print("4. Data continuity check - detects large price jumps between steps")
print("\nThis will help identify:")
print("- If trades are being executed properly")
print("- If position tracking is working")
print("- If P&L calculations are correct")
print("- If data is continuous or jumping")