#!/usr/bin/env python3
"""
Diagnose the stop-loss trigger issue.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.options_trading_env import OptionsTradingEnvironment, OptionsPosition, OptionContract
from datetime import datetime, timedelta
import numpy as np

def test_stop_loss_trigger():
    """Test if stop loss is triggered immediately."""
    
    # Create a test contract
    contract = OptionContract(
        symbol='TEST',
        strike=100,
        expiration=datetime.now() + timedelta(days=30),
        option_type='call',
        bid=9.95,
        ask=10.05,
        last_price=10.00,
        volume=1000,
        open_interest=5000,
        implied_volatility=0.20,
        delta=0.5,
        gamma=0.01,
        theta=-0.05,
        vega=0.10,
        rho=0.02
    )
    
    # Create a position (buying at ask)
    position = OptionsPosition(
        contract=contract,
        quantity=10,
        entry_price=contract.ask,  # Buy at ask = 10.05
        entry_time=datetime.now(),
        position_type='long'
    )
    
    print("Position details:")
    print(f"  Entry price (ask): ${position.entry_price:.2f}")
    print(f"  Current bid: ${contract.bid:.2f}")
    print(f"  Current ask: ${contract.ask:.2f}")
    print(f"  Mid price: ${(contract.bid + contract.ask) / 2:.2f}")
    print(f"  Position P&L: ${position.pnl:.2f}")
    
    # Calculate stop loss threshold
    stop_loss_threshold = -position.contract.ask * position.quantity * 100 * 0.10
    print(f"\nStop loss calculation:")
    print(f"  Stop loss threshold: ${stop_loss_threshold:.2f}")
    print(f"  Current P&L: ${position.pnl:.2f}")
    print(f"  Will trigger stop loss: {position.pnl < stop_loss_threshold}")
    
    # Let's see what P&L would be with different spreads
    print("\nP&L with different bid-ask spreads:")
    for spread in [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]:
        test_bid = 10.00 - spread/2
        test_ask = 10.00 + spread/2
        contract.bid = test_bid
        contract.ask = test_ask
        
        # Recalculate P&L
        mid_price = (contract.bid + contract.ask) / 2
        pnl = position.quantity * (mid_price - position.entry_price) * 100
        
        print(f"  Spread ${spread:.2f}: Bid=${test_bid:.2f}, Ask=${test_ask:.2f}, "
              f"Mid=${mid_price:.2f}, P&L=${pnl:.2f}")

def test_realistic_scenario():
    """Test with realistic market parameters."""
    print("\n\n=== Testing Realistic Trading Scenario ===\n")
    
    env = OptionsTradingEnvironment()
    
    # Get a simulated option
    option = env._create_simulated_option('buy_call')
    
    # Calculate the immediate P&L
    mid_price = (option.bid + option.ask) / 2
    immediate_pnl_pct = (mid_price - option.ask) / option.ask * 100
    
    print(f"Simulated option:")
    print(f"  Bid: ${option.bid:.2f}")
    print(f"  Ask: ${option.ask:.2f}")
    print(f"  Spread: ${option.ask - option.bid:.2f}")
    print(f"  Spread %: {(option.ask - option.bid) / option.ask * 100:.2f}%")
    print(f"  Immediate P&L %: {immediate_pnl_pct:.2f}%")
    
    # Check against 10% stop loss
    print(f"\nStop loss check:")
    print(f"  Stop loss threshold: -10%")
    print(f"  Immediate loss: {immediate_pnl_pct:.2f}%")
    print(f"  Will trigger stop loss: {immediate_pnl_pct < -10}")

if __name__ == "__main__":
    test_stop_loss_trigger()
    test_realistic_scenario()