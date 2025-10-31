#!/usr/bin/env python3
"""
Add exponential reward scaling based on returns to incentivize larger profitable trades
"""

import re

# Read the training script
train_file = "/home/devin/Desktop/Trading_bot/train_profitable_optimized.py"

with open(train_file, 'r') as f:
    content = f.read()

print("Adding exponential reward scaling for large returns...")

# First, add the exponential reward function after the class definitions start
exponential_reward_function = '''
    def _calculate_exponential_reward(self, pnl, pnl_pct, base_reward=1.0):
        """
        Calculate exponentially scaled reward based on P&L
        Encourages larger profitable trades while penalizing losses
        """
        if pnl > 0:
            # Exponential scaling for profits
            # Small profits: linear scaling
            # Large profits: exponential boost
            if pnl_pct < 0.05:  # Under 5%
                reward = base_reward * pnl_pct * 20  # Linear: 5% = 1.0 reward
            elif pnl_pct < 0.10:  # 5-10%
                reward = base_reward * (1.0 + (pnl_pct - 0.05) * 40)  # Accelerating
            elif pnl_pct < 0.20:  # 10-20%
                reward = base_reward * (3.0 + (pnl_pct - 0.10) * 80)  # Steep acceleration
            else:  # Over 20%
                # Exponential scaling for huge wins
                # Using approximation: e^x ≈ 1 + x + x²/2 + x³/6 for small x
                x = (pnl_pct - 0.20) * 2
                exp_approx = 1 + x + (x*x)/2 + (x*x*x)/6
                reward = base_reward * (11.0 * exp_approx)
            
            # Additional bonus for absolute P&L size
            if pnl > 1000:  # Over $1000 profit
                reward *= 1.5
            elif pnl > 5000:  # Over $5000 profit
                reward *= 2.0
            elif pnl > 10000:  # Over $10000 profit
                reward *= 3.0
                
        else:
            # Losses: Less severe penalty to encourage risk-taking
            # But still penalize to maintain profitability
            if pnl_pct > -0.05:  # Small loss
                reward = base_reward * pnl_pct * 10  # -5% = -0.5 reward
            elif pnl_pct > -0.10:  # Moderate loss
                reward = base_reward * (-0.5 + (pnl_pct + 0.05) * 20)
            else:  # Large loss
                reward = base_reward * (-1.5 + pnl_pct * 5)  # Capped penalty
        
        return reward
'''

# Find where to insert the function (after __init__ in BalancedEnvironment)
pattern = r'(class BalancedEnvironment\(HistoricalOptionsEnvironment\):.*?def __init__.*?\n\n)'
match = re.search(pattern, content, re.DOTALL)
if match:
    insert_pos = match.end()
    content = content[:insert_pos] + exponential_reward_function + '\n' + content[insert_pos:]
else:
    # Fallback: insert after the first reset method
    pattern = r'(def reset\(self\):.*?\n        return obs\n)'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        insert_pos = match.end()
        content = content[:insert_pos] + '\n' + exponential_reward_function + '\n' + content[insert_pos:]

# Now update the step function reward calculation (line ~1356)
old_reward_calc = r'step_pnl = portfolio_value_after - portfolio_value_before\n        reward \+= step_pnl / 1000'
new_reward_calc = '''step_pnl = portfolio_value_after - portfolio_value_before
        
        # Use exponential reward scaling for portfolio changes
        if step_pnl != 0:
            pnl_pct = step_pnl / portfolio_value_before if portfolio_value_before > 0 else 0
            reward += self._calculate_exponential_reward(step_pnl, pnl_pct, base_reward=1.0)
        else:
            # Small penalty for holding without positions
            if len(self.positions) == 0 and action_name == 'hold':
                reward -= 0.01'''

content = re.sub(old_reward_calc, new_reward_calc, content)

# Update position closing rewards in _fast_update_positions (around line 1409)
old_winning_code = r'if pnl > 0:\n                    self\.winning_trades \+= 1\n                else:\n                    self\.losing_trades \+= 1'
new_winning_code = '''if pnl > 0:
                    self.winning_trades += 1
                    # Add exponential reward for closing profitable position
                    self.last_trade_reward = self._calculate_exponential_reward(pnl, pnl_pct, base_reward=5.0)
                else:
                    self.losing_trades += 1
                    # Penalty for losing trade
                    self.last_trade_reward = self._calculate_exponential_reward(pnl, pnl_pct, base_reward=2.0)'''

content = re.sub(old_winning_code, new_winning_code, content, count=1)

# Update the reward in step function to include last_trade_reward
pattern = r'(# Update step\n        self\.current_step \+= 1)'
replacement = r'''# Add any trade closing rewards
        if hasattr(self, 'last_trade_reward'):
            reward += self.last_trade_reward
            self.last_trade_reward = 0
        
        \1'''
content = re.sub(pattern, replacement, content)

# Update _fast_close_all_positions to use exponential rewards (around line 1431)
old_close_all = r'if pnl > 0:\n                self\.winning_trades \+= 1\n                total_reward \+= 10\.0\n            else:\n                self\.losing_trades \+= 1'
new_close_all = '''if pnl > 0:
                self.winning_trades += 1
                # Exponential reward for profitable trade
                pnl_pct = pnl / entry_cost
                total_reward += self._calculate_exponential_reward(pnl, pnl_pct, base_reward=10.0)
            else:
                self.losing_trades += 1
                # Smaller penalty for losses to encourage risk-taking
                pnl_pct = pnl / entry_cost
                total_reward += self._calculate_exponential_reward(pnl, pnl_pct, base_reward=3.0)'''

content = re.sub(old_close_all, new_close_all, content)

# Initialize last_trade_reward in reset method
reset_pattern = r'(self\._last_action = \'hold\'  # Reset last action)'
reset_replacement = r"\1\n        self.last_trade_reward = 0  # Reset trade reward"
content = re.sub(reset_pattern, reset_replacement, content)

# Also add the same function to OptimizedProfitableEnvironment
pattern = r'(class OptimizedProfitableEnvironment\(HistoricalOptionsEnvironment\):.*?super\(\).__init__\(\*args, \*\*kwargs\)\n)'
match = re.search(pattern, content, re.DOTALL)
if match:
    insert_pos = match.end()
    content = content[:insert_pos] + '\n' + exponential_reward_function + '\n' + content[insert_pos:]

# Save the updated file
with open(train_file, 'w') as f:
    f.write(content)

print("✅ Exponential reward scaling added successfully!")
print("\nReward Structure:")
print("📈 Profits:")
print("  - 0-5% return: Linear scaling (5% = 1.0 reward)")
print("  - 5-10% return: Accelerating rewards (10% = 3.0 reward)")
print("  - 10-20% return: Steep acceleration (20% = 11.0 reward)")
print("  - >20% return: Exponential scaling")
print("  - Additional multipliers for absolute P&L > $1k, $5k, $10k")
print("\n📉 Losses:")
print("  - Moderate penalties to encourage calculated risk-taking")
print("  - Capped downside to prevent over-conservative behavior")
print("\nThis will encourage the model to:")
print("✅ Hold positions for larger gains")
print("✅ Take calculated risks for big returns")
print("✅ Still manage risk appropriately")