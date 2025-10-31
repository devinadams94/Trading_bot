#!/usr/bin/env python3
"""Debug why we can't use 95% of capital"""

from src.options_trading_env import OptionsTradingEnvironment

# Create environment
env = OptionsTradingEnvironment(initial_capital=100000)
env.reset()

# Create a test contract
contract = env._create_simulated_option('buy_call')

print("=== Capital Usage Analysis ===")
print(f"Initial capital: ${env.capital:,.2f}")
print(f"Target usage (95%): ${env.capital * 0.95:,.2f}")
print(f"\nOption details:")
print(f"  Ask price: ${contract.ask:.2f}")
print(f"  Contract value: ${contract.ask * 100:.2f}")
print(f"  Commission per contract: ${env.commission:.2f}")

# Calculate ideal position size
target_value = env.capital * 0.95
ideal_contracts = int(target_value / (contract.ask * 100))

print(f"\nPosition sizing:")
print(f"  Ideal contracts (no commission): {ideal_contracts}")
print(f"  Position value: ${ideal_contracts * contract.ask * 100:,.2f}")
print(f"  Total commission: ${ideal_contracts * env.commission:,.2f}")
print(f"  Total cost: ${ideal_contracts * contract.ask * 100 + ideal_contracts * env.commission:,.2f}")

# Account for commission in calculation
contracts_with_commission = int(target_value / (contract.ask * 100 + env.commission))
print(f"\nAdjusted for commission:")
print(f"  Contracts: {contracts_with_commission}")
print(f"  Position value: ${contracts_with_commission * contract.ask * 100:,.2f}")
print(f"  Commission: ${contracts_with_commission * env.commission:,.2f}")
print(f"  Total cost: ${contracts_with_commission * (contract.ask * 100 + env.commission):,.2f}")
print(f"  Capital usage: {(contracts_with_commission * (contract.ask * 100 + env.commission) / env.capital) * 100:.1f}%")

print(f"\nThe issue is that we're not accounting for commission in the position sizing calculation!")
print(f"Commission adds ${env.commission:.2f} per contract, which reduces the number of contracts we can buy.")