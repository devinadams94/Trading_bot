#!/usr/bin/env python3
"""
Diagnose why trades immediately show negative rewards.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.options_trading_env import OptionsTradingEnvironment
import numpy as np

def test_trade_execution():
    """Test a single trade to see the reward calculation."""
    print("=== Testing Trade Execution and Reward Calculation ===\n")
    
    # Create environment with verbose logging
    env = OptionsTradingEnvironment(
        initial_capital=100000,
        commission=0.65,
    )
    
    # Reset environment
    obs = env.reset()
    print(f"Initial state:")
    print(f"  Capital: ${env.capital:,.2f}")
    print(f"  Portfolio Value: ${env._calculate_portfolio_value():,.2f}")
    print(f"  Last Portfolio Value: ${getattr(env, 'last_portfolio_value', 'Not set')}")
    print()
    
    # Execute a buy call action
    action = 1  # buy_call
    print(f"Executing action: {env.action_mapping[action]}")
    
    # Get initial portfolio value
    initial_portfolio_value = env._calculate_portfolio_value()
    
    # Execute step
    obs, reward, done, info = env.step(action)
    
    print(f"\nAfter trade:")
    print(f"  Capital: ${env.capital:,.2f}")
    print(f"  Portfolio Value: ${env._calculate_portfolio_value():,.2f}")
    print(f"  Number of positions: {len(env.positions)}")
    
    if env.positions:
        pos = env.positions[0]
        print(f"\nPosition details:")
        print(f"  Contract: {pos.contract.option_type} @ ${pos.contract.strike}")
        print(f"  Quantity: {pos.quantity}")
        print(f"  Entry price: ${pos.entry_price:.2f}")
        print(f"  Current bid: ${pos.contract.bid:.2f}")
        print(f"  Current ask: ${pos.contract.ask:.2f}")
        print(f"  Mid price: ${(pos.contract.bid + pos.contract.ask) / 2:.2f}")
        print(f"  Position value: ${pos.current_value:,.2f}")
        print(f"  Position P&L: ${pos.pnl:.2f}")
    
    print(f"\nReward calculation breakdown:")
    print(f"  Initial portfolio: ${initial_portfolio_value:,.2f}")
    print(f"  Current portfolio: ${env._calculate_portfolio_value():,.2f}")
    print(f"  Portfolio change: ${env._calculate_portfolio_value() - initial_portfolio_value:,.2f}")
    print(f"  Reward: {reward:.4f}")
    
    # Let's manually calculate what the reward should be
    if hasattr(env, 'last_portfolio_value'):
        step_pnl = env._calculate_portfolio_value() - env.last_portfolio_value
        base_reward = step_pnl / 1000.0
        print(f"\nManual calculation:")
        print(f"  Last portfolio value: ${env.last_portfolio_value:,.2f}")
        print(f"  Step P&L: ${step_pnl:.2f}")
        print(f"  Base reward: {base_reward:.4f}")
        
        # Check if there's commission impact
        if env.positions:
            pos = env.positions[0]
            commission_cost = env.commission * pos.quantity
            print(f"\nCommission analysis:")
            print(f"  Commission per contract: ${env.commission}")
            print(f"  Total commission: ${commission_cost:.2f}")
            print(f"  Entry cost (ask price): ${pos.entry_price * pos.quantity * 100:.2f}")
            print(f"  Total cost with commission: ${pos.entry_price * pos.quantity * 100 + commission_cost:.2f}")

def test_bid_ask_spread():
    """Test the impact of bid-ask spread on immediate losses."""
    print("\n\n=== Testing Bid-Ask Spread Impact ===\n")
    
    env = OptionsTradingEnvironment()
    env.reset()
    
    # Create a simulated option
    option = env._create_simulated_option('buy_call')
    
    print(f"Option pricing:")
    print(f"  Bid: ${option.bid:.2f}")
    print(f"  Ask: ${option.ask:.2f}")
    print(f"  Spread: ${option.ask - option.bid:.2f}")
    print(f"  Mid price: ${(option.bid + option.ask) / 2:.2f}")
    
    # Calculate immediate loss from spread
    immediate_loss = option.ask - (option.bid + option.ask) / 2
    print(f"\nImmediate loss from spread: ${immediate_loss:.2f} per contract")
    print(f"For 100 contracts: ${immediate_loss * 100 * 100:.2f}")
    
    # Calculate percentage loss
    pct_loss = immediate_loss / option.ask * 100
    print(f"Percentage loss: {pct_loss:.2f}%")

if __name__ == "__main__":
    test_trade_execution()
    test_bid_ask_spread()