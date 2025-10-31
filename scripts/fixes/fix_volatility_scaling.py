#!/usr/bin/env python3
"""
Fix the volatility scaling issue in the options trading environment.
"""

import os

# Read the current environment file
env_file = "/home/devin/Desktop/Trading_bot/src/options_trading_env.py"

with open(env_file, 'r') as f:
    content = f.read()

# Fix 1: Scale the volatility properly in _update_positions
# Find the line with historical_volatility usage
old_line = "            price_change = np.random.normal(mean_return, historical_volatility)"
new_line = "            # Scale volatility from annual to daily\n" + \
           "            daily_volatility = historical_volatility / np.sqrt(252)  # 252 trading days\n" + \
           "            daily_mean_return = mean_return / 252  # Daily drift\n" + \
           "            price_change = np.random.normal(daily_mean_return, daily_volatility)"

content = content.replace(old_line, new_line)

# Fix 2: Update the mean return comment to be clearer
old_comment = "        self.mean_return = 0.01  # Reduce drift to make it more challenging"
new_comment = "        self.mean_return = 0.01  # 1% annual drift"

content = content.replace(old_comment, new_comment)

# Fix 3: Update default mean return in _update_positions
old_default = "            mean_return = getattr(self, 'mean_return', 0.0176)  # Default 1.76% drift"
new_default = "            mean_return = getattr(self, 'mean_return', 0.01)  # Default 1% annual drift"

content = content.replace(old_default, new_default)

# Save the fixed file
with open(env_file, 'w') as f:
    f.write(content)

print("Fixed volatility scaling in options_trading_env.py")
print("\nChanges made:")
print("1. Scaled annual volatility to daily volatility (divided by sqrt(252))")
print("2. Scaled annual mean return to daily (divided by 252)")
print("3. Updated comments for clarity")
print("\nThis should prevent immediate stop-loss triggers and allow for realistic trading.")