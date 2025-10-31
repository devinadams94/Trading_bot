#!/usr/bin/env python3
"""
Debug script to understand why training shows 0% win rate and repeated trades
"""

import numpy as np
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.historical_options_data import HistoricalOptionsEnvironment
from train_profitable_optimized import UltraFastEnvironment, BalancedEnvironment

def test_environment_reset():
    """Test if environments properly randomize on reset"""
    print("=== Testing Environment Reset Randomization ===\n")
    
    # Create dummy data
    import pandas as pd
    dummy_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1h'),
        'underlying_price': np.random.uniform(95, 105, 1000),
        'strike': [100] * 1000,
        'option_type': ['call'] * 500 + ['put'] * 500,
        'bid': np.random.uniform(1, 5, 1000),
        'ask': np.random.uniform(1.1, 5.1, 1000),
        'volume': np.random.randint(100, 10000, 1000),
        'moneyness': np.random.uniform(0.9, 1.1, 1000)
    })
    
    # Test BalancedEnvironment
    env = BalancedEnvironment(
        historical_data={'TEST': dummy_data},
        symbols=['TEST']
    )
    
    starting_positions = []
    for i in range(5):
        env.reset()
        starting_positions.append(env.current_step)
        print(f"Reset {i+1}: Starting at step {env.current_step}")
    
    if len(set(starting_positions)) == 1:
        print("\n❌ PROBLEM: All resets start at the same position!")
    else:
        print("\n✅ Good: Resets start at different positions")
    
    return starting_positions

def test_action_diversity():
    """Test if actions are diverse or stuck in a pattern"""
    print("\n=== Testing Action Diversity ===\n")
    
    # Simulate action selection with current entropy
    n_actions = 11
    n_steps = 50
    
    # Test with low entropy (old setting)
    print("With low entropy (0.005):")
    logits = torch.randn(n_steps, n_actions)
    temperature = 1.0
    scaled_logits = logits / temperature
    
    dist = torch.distributions.Categorical(logits=scaled_logits)
    actions = dist.sample().numpy()
    unique_actions = len(np.unique(actions))
    print(f"Unique actions: {unique_actions}/{n_actions}")
    print(f"Action distribution: {np.bincount(actions, minlength=n_actions)}")
    
    # Test with higher entropy and noise
    print("\nWith higher entropy (0.02) + noise:")
    noise = torch.randn_like(logits) * 0.1
    scaled_logits = (logits + noise) / temperature
    
    dist = torch.distributions.Categorical(logits=scaled_logits)
    actions = dist.sample().numpy()
    unique_actions = len(np.unique(actions))
    print(f"Unique actions: {unique_actions}/{n_actions}")
    print(f"Action distribution: {np.bincount(actions, minlength=n_actions)}")

def test_reward_calculation():
    """Test if rewards are being calculated properly"""
    print("\n=== Testing Reward Calculation ===\n")
    
    # Simulate position P&L with new pricing model
    underlying_prices = [100, 101, 99, 102, 98]
    strike = 100
    entry_price = 2.0
    
    for i, current_price in enumerate(underlying_prices):
        price_change = (current_price - underlying_prices[0]) / underlying_prices[0]
        moneyness = current_price / strike
        
        # Old linear model
        old_option_change = price_change * 1.5 if price_change > 0 else price_change * 0.8
        old_value = entry_price * (1 + old_option_change)
        old_pnl_pct = old_option_change
        
        # New delta model
        if moneyness > 1.02:
            delta = 0.7
        elif moneyness > 0.98:
            delta = 0.5
        else:
            delta = 0.3
        
        new_option_change = price_change * delta
        new_option_change = max(new_option_change, -0.15)  # Cap losses
        new_value = entry_price * (1 + new_option_change)
        new_pnl_pct = new_option_change
        
        print(f"Step {i}: Price ${underlying_prices[0]} → ${current_price}")
        print(f"  Old model: {old_pnl_pct:.1%} P&L → {'STOP LOSS' if old_pnl_pct <= -0.05 else 'HOLD'}")
        print(f"  New model: {new_pnl_pct:.1%} P&L → {'STOP LOSS' if new_pnl_pct <= -0.05 else 'HOLD'}")

def check_data_loading():
    """Check how data is being loaded and distributed"""
    print("\n=== Checking Data Loading ===\n")
    
    # This would require actual data files
    # For now, just explain the expected behavior
    print("Expected data flow:")
    print("1. Historical data loaded into memory")
    print("2. Each environment gets full dataset")
    print("3. On reset, starting position randomized")
    print("4. Episodes should see different data slices")
    
    print("\nPotential issues:")
    print("- Data might be truncated or limited")
    print("- Symbols might have insufficient data")
    print("- Episode length might exceed available data")

def main():
    """Run all debug tests"""
    print("Debugging Training Issues\n")
    
    # Test 1: Environment reset
    starting_positions = test_environment_reset()
    
    # Test 2: Action diversity
    test_action_diversity()
    
    # Test 3: Reward calculation
    test_reward_calculation()
    
    # Test 4: Data loading
    check_data_loading()
    
    print("\n=== Diagnostic Summary ===")
    print("\nMost likely causes of repeated trades with 0% win rate:")
    print("1. ❓ Environments might be sharing state (not independent)")
    print("2. ❓ Model weights might not be updating properly")
    print("3. ❓ Rewards might still be triggering immediate exits")
    print("4. ❓ Data slices might be too similar")
    print("5. ❓ Buffer might be corrupted or not clearing between episodes")

if __name__ == "__main__":
    main()