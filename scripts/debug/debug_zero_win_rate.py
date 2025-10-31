#!/usr/bin/env python3
"""
Add comprehensive debugging to understand why win rate is 0% after first episode
"""

import re

# Read the training script
train_file = "/home/devin/Desktop/Trading_bot/train_profitable_optimized.py"

with open(train_file, 'r') as f:
    content = f.read()

# 1. Add detailed logging in the reward calculation
reward_debug = '''
        # DEBUG: Log reward details for every trade close
        if hasattr(self, '_debug_mode') and self._debug_mode:
            logger.info(f"[DEBUG-REWARD] Step {self.current_step}: action={action_name}, step_pnl={step_pnl:.2f}, reward={reward:.2f}")
'''

# Find the step method reward calculation
reward_pattern = r'(reward = base_reward \+ reward_adjustment)'
if re.search(reward_pattern, content):
    content = re.sub(reward_pattern, r'\1' + reward_debug, content)

# 2. Add debugging to track unique trades vs repeated trades
unique_trade_debug = '''
        # DEBUG: Track unique trades to detect repetition
        if not hasattr(self, '_trade_history'):
            self._trade_history = []
            self._unique_trades = set()
            
        trade_signature = f"{self.current_step}_{current_price:.2f}_{action_name}"
        self._trade_history.append(trade_signature)
        self._unique_trades.add(trade_signature)
        
        if len(self._trade_history) > 10 and len(self._unique_trades) < 3:
            logger.warning(f"[DEBUG-REPETITION] Only {len(self._unique_trades)} unique trades in last 10 attempts!")
'''

# Insert after action execution in step method
action_exec_pattern = r'(# Execute action\s*\n\s*reward = 0)'
content = re.sub(action_exec_pattern, r'\1' + unique_trade_debug, content)

# 3. Add debugging to position exit logic
exit_debug = '''
            # DEBUG: Log why positions are exiting
            if hasattr(self, '_debug_mode') and self._debug_mode and should_exit:
                exit_reason = "unknown"
                if pnl_pct <= dynamic_stop_loss:
                    exit_reason = f"stop_loss (pnl={pnl_pct:.2%} <= {dynamic_stop_loss:.2%})"
                elif pnl_pct >= dynamic_take_profit:
                    exit_reason = f"take_profit (pnl={pnl_pct:.2%} >= {dynamic_take_profit:.2%})"
                elif position_age > 15:
                    exit_reason = f"time_exit (age={position_age})"
                logger.info(f"[DEBUG-EXIT] Position {i} exiting: {exit_reason}, final_pnl={pnl:.2f}")
'''

# Find position exit logic
exit_pattern = r'(if should_exit:\s*\n\s*positions_to_close\.append\(i\))'
content = re.sub(exit_pattern, exit_debug + r'\n            \1', content)

# 4. Add trade entry debugging
entry_debug = '''
                # DEBUG: Log trade entry details
                if hasattr(self, '_debug_mode') and self._debug_mode:
                    logger.info(f"[DEBUG-ENTRY] Opening {action_name} at ${option['ask']:.2f}, "
                              f"contracts={contracts}, cost=${total_cost:.2f}, "
                              f"underlying=${current_price:.2f}, moneyness={moneyness:.2f}")
'''

# Find trade entry logic
entry_pattern = r'(self\.capital -= total_cost)'
content = re.sub(entry_pattern, r'\1' + entry_debug, content)

# 5. Enable debug mode for specific episodes
debug_mode_enable = '''
        # Enable debug mode for problem episodes
        self._debug_mode = self._episode_count <= 5 or self._episode_count % 10 == 0
        if self._debug_mode:
            logger.info(f"[DEBUG-MODE] Enabled for episode {self._episode_count}")
'''

# Insert in reset method after episode count increment
reset_pattern = r'(self\._episode_count \+= 1)'
content = re.sub(reset_pattern, r'\1' + debug_mode_enable, content)

# 6. Add P&L calculation debugging
pnl_calc_debug = '''
            # DEBUG: Log detailed P&L calculation
            if hasattr(self, '_debug_mode') and self._debug_mode and i < 3:  # First 3 positions
                logger.info(f"[DEBUG-PNL-CALC] Pos {i}: price_change={price_change:.3%}, "
                          f"delta={delta:.2f}, option_change={option_price_change:.3%}, "
                          f"entry=${entry_cost:.2f}, current=${current_value:.2f}, "
                          f"pnl=${pnl:.2f} ({pnl_pct:.2%})")
'''

# Find P&L calculation section
pnl_pattern = r'(pnl_pct = pnl / entry_cost)'
content = re.sub(pnl_pattern, r'\1' + pnl_calc_debug, content)

# 7. Add win/loss tracking debug
win_loss_debug = '''
                    # DEBUG: Track wins and losses
                    if hasattr(self, '_debug_mode') and self._debug_mode:
                        logger.info(f"[DEBUG-TRADE-RESULT] {'WIN' if pnl > 0 else 'LOSS'}: pnl=${pnl:.2f}, "
                                  f"total_wins={self.winning_trades}, total_losses={self.losing_trades}")
'''

# Find where wins/losses are counted
win_pattern = r'(if pnl > 0:\s*\n\s*self\.winning_trades \+= 1)'
content = re.sub(win_pattern, r'\1' + win_loss_debug, content, flags=re.MULTILINE)

# 8. Add episode summary debug
episode_summary = '''
            # DEBUG: Episode summary for problem episodes
            if hasattr(self, '_debug_mode') and self._debug_mode:
                logger.info(f"[DEBUG-EPISODE-SUMMARY] Episode {self._episode_count}: "
                          f"wins={self.winning_trades}, losses={self.losing_trades}, "
                          f"positions={len(self.positions)}, capital=${self.capital:.2f}")
'''

# Insert at end of step method when done
done_pattern = r'(if self\.done:\s*\n\s*return)'
content = re.sub(done_pattern, episode_summary + r'\n        \1', content)

# Save the updated file
with open(train_file, 'w') as f:
    f.write(content)

print("Added comprehensive debugging for zero win rate issue")
print("\nDebug features added:")
print("1. Reward calculation logging")
print("2. Trade repetition detection") 
print("3. Position exit reason tracking")
print("4. Trade entry details")
print("5. Debug mode for specific episodes")
print("6. P&L calculation breakdown")
print("7. Win/loss tracking")
print("8. Episode summaries")
print("\nRun training and look for [DEBUG-*] messages to understand the issue.")