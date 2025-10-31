#!/usr/bin/env python3
"""Debug position sizing"""

from src.options_trading_env import OptionsTradingEnvironment

env = OptionsTradingEnvironment(initial_capital=100000)
env.reset()

# Try to open a position
print("Opening position...")
obs, reward, done, info = env.step(1)  # Buy call

if env.positions:
    pos = env.positions[0]
    print(f"\nContract details:")
    print(f"  Ask price: ${pos.contract.ask:.2f}")
    print(f"  Strike: ${pos.contract.strike:.2f}")
    print(f"  Contract value: ${pos.contract.ask * 100:.2f}")
    
    print(f"\nPosition sizing calculation:")
    print(f"  Capital available: ${env.capital:.2f}")
    print(f"  Target value (95% of capital): ${env.capital * 0.95:.2f}")
    print(f"  Contract cost: ${pos.contract.ask * 100:.2f}")
    print(f"  Max contracts from formula: {int((env.capital * 0.95) / (pos.contract.ask * 100))}")
    print(f"  Actual contracts: {pos.quantity}")
    print(f"  Total position value: ${pos.quantity * pos.contract.ask * 100:.2f}")
else:
    print("No position opened")

# The issue is that option prices are very low (e.g. $0.44) 
# So even 200 contracts only uses ~$8,800 of capital
# We need to either:
# 1. Increase max contracts beyond 200
# 2. Use more expensive options
# 3. Trade multiple positions at once