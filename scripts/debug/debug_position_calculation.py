#!/usr/bin/env python3
"""Debug the position calculation issue"""

from src.options_trading_env import OptionsTradingEnvironment

env = OptionsTradingEnvironment(initial_capital=100000)
env.reset()

# Get the contract that would be created
contract = env._create_simulated_option('buy_call')

print("=== Position Calculation Debug ===")
print(f"Capital: ${env.capital:,.2f}")
print(f"Target (95%): ${env.capital * 0.95:,.2f}")
print(f"\nContract details:")
print(f"  Ask price: ${contract.ask:.2f}")
print(f"  Contract value: ${contract.ask * 100:.2f}")
print(f"  Commission: ${env.commission:.2f}")
print(f"  Cost per contract: ${(contract.ask * 100) + env.commission:.2f}")

# Manual calculation
target_value = env.capital * 0.95
cost_per_contract = (contract.ask * 100) + env.commission
ideal_size = int(target_value / cost_per_contract)

print(f"\nCalculation:")
print(f"  ideal_size = int({target_value:.2f} / {cost_per_contract:.2f})")
print(f"  ideal_size = int({target_value / cost_per_contract:.2f})")
print(f"  ideal_size = {ideal_size}")

# Now actually execute the trade
obs, reward, done, info = env.step(1)

if env.positions:
    pos = env.positions[0]
    actual_cost = pos.quantity * pos.contract.ask * 100 + pos.quantity * env.commission
    print(f"\nActual position:")
    print(f"  Contracts: {pos.quantity}")
    print(f"  Total cost: ${actual_cost:,.2f}")
    print(f"  Capital used: {(actual_cost / 100000) * 100:.1f}%")
    
# The issue might be that we're resetting capital after opening position
print(f"\nRemaining capital after trade: ${env.capital:,.2f}")