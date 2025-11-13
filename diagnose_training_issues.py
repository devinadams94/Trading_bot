#!/usr/bin/env python3
"""
Diagnostic script to identify why the model is losing money
"""

import os
import sys
import torch
import numpy as np
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

from src.historical_options_data import OptimizedHistoricalOptionsDataLoader
from src.working_options_env import WorkingOptionsEnvironment

def diagnose_environment():
    """Run diagnostic tests on the environment"""
    
    print("=" * 80)
    print("🔍 DIAGNOSING TRAINING ISSUES")
    print("=" * 80)
    print()
    
    # Initialize data loader
    api_key = os.getenv('MASSIVE_API_KEY')
    if not api_key:
        print("❌ ERROR: MASSIVE_API_KEY not found in .env file")
        return
    
    print("📊 Loading market data...")
    data_loader = OptimizedHistoricalOptionsDataLoader(
        api_key=api_key,
        api_secret=None,
        base_url=None,
        data_url=None
    )
    
    # Load data for SPY only (quick test)
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=90)  # 90 days
    
    import asyncio
    asyncio.run(data_loader.load_data(
        symbols=['SPY'],
        start_date=start_date,
        end_date=end_date,
        timeframe='1Hour'
    ))
    
    print("✅ Data loaded")
    print()
    
    # Initialize environment
    print("🏗️  Initializing environment...")
    env = WorkingOptionsEnvironment(
        data_loader=data_loader,
        symbols=['SPY'],
        initial_capital=100000,
        max_positions=5,
        episode_length=200,
        lookback_window=30,
        include_technical_indicators=True,
        include_market_microstructure=True,
        use_realistic_costs=True,
        enable_slippage=True,
        slippage_model='volume_based'
    )
    
    print("✅ Environment initialized")
    print()
    
    # Run diagnostic episodes
    print("=" * 80)
    print("🧪 RUNNING DIAGNOSTIC EPISODES")
    print("=" * 80)
    print()
    
    num_episodes = 5
    strategies = {
        'random': 'Random actions',
        'hold_only': 'Hold only (action 0)',
        'buy_calls_only': 'Buy calls only (actions 1-15)',
        'buy_puts_only': 'Buy puts only (actions 16-30)',
        'close_frequently': 'Buy and close frequently'
    }
    
    results = {}
    
    for strategy_name, strategy_desc in strategies.items():
        print(f"\n📈 Testing strategy: {strategy_desc}")
        print("-" * 80)
        
        episode_returns = []
        episode_trades = []
        episode_rewards = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            done = False
            total_reward = 0
            trades = 0
            
            while not done:
                # Choose action based on strategy
                if strategy_name == 'random':
                    action = np.random.randint(0, 31)
                elif strategy_name == 'hold_only':
                    action = 0
                elif strategy_name == 'buy_calls_only':
                    action = np.random.randint(1, 16)
                elif strategy_name == 'buy_puts_only':
                    action = np.random.randint(16, 31)
                elif strategy_name == 'close_frequently':
                    # Alternate between buying and closing
                    if len(env.positions) > 0:
                        action = 0  # Close positions (hold triggers auto-close logic)
                    else:
                        action = np.random.randint(1, 31)  # Buy something
                
                obs, reward, done, info = env.step(action)
                total_reward += reward
                
                if info.get('trade_executed', False):
                    trades += 1
            
            # Get final portfolio value
            final_value = info.get('portfolio_value', env.initial_capital)
            portfolio_return = (final_value - env.initial_capital) / env.initial_capital
            
            episode_returns.append(portfolio_return)
            episode_trades.append(trades)
            episode_rewards.append(total_reward)
        
        # Calculate statistics
        avg_return = np.mean(episode_returns)
        avg_trades = np.mean(episode_trades)
        avg_reward = np.mean(episode_rewards)
        
        results[strategy_name] = {
            'avg_return': avg_return,
            'avg_trades': avg_trades,
            'avg_reward': avg_reward,
            'returns': episode_returns
        }
        
        print(f"   Avg Return: {avg_return:+.4f} ({avg_return*100:+.2f}%)")
        print(f"   Avg Trades: {avg_trades:.1f}")
        print(f"   Avg Reward: {avg_reward:+.6f}")
        print(f"   Returns: {[f'{r*100:+.2f}%' for r in episode_returns]}")
    
    print()
    print("=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print()
    
    # Find best strategy
    best_strategy = max(results.items(), key=lambda x: x[1]['avg_return'])
    worst_strategy = min(results.items(), key=lambda x: x[1]['avg_return'])
    
    print(f"🏆 Best Strategy: {strategies[best_strategy[0]]}")
    print(f"   Avg Return: {best_strategy[1]['avg_return']*100:+.2f}%")
    print()
    
    print(f"❌ Worst Strategy: {strategies[worst_strategy[0]]}")
    print(f"   Avg Return: {worst_strategy[1]['avg_return']*100:+.2f}%")
    print()
    
    # Analyze reward scaling
    print("=" * 80)
    print("🔍 REWARD SCALING ANALYSIS")
    print("=" * 80)
    print()
    
    # Check if rewards are too small
    all_rewards = [r['avg_reward'] for r in results.values()]
    max_reward = max(all_rewards)
    min_reward = min(all_rewards)
    
    print(f"Max avg reward: {max_reward:+.6f}")
    print(f"Min avg reward: {min_reward:+.6f}")
    print(f"Reward range: {max_reward - min_reward:.6f}")
    print()
    
    if abs(max_reward) < 0.01 and abs(min_reward) < 0.01:
        print("⚠️  WARNING: Rewards are very small (< 0.01)")
        print("   This may cause slow learning or numerical instability")
        print("   Consider increasing reward scaling factor")
    
    # Analyze transaction costs
    print()
    print("=" * 80)
    print("💰 TRANSACTION COST ANALYSIS")
    print("=" * 80)
    print()
    
    # Run one episode and track costs
    obs = env.reset()
    done = False
    total_costs = 0
    total_raw_return = 0
    step_count = 0
    
    while not done and step_count < 50:  # Limit to 50 steps
        action = np.random.randint(1, 31)  # Random buy action
        obs, reward, done, info = env.step(action)
        
        if 'transaction_costs' in info:
            total_costs += info['transaction_costs']
        if 'raw_return' in info:
            total_raw_return += info['raw_return']
        
        step_count += 1
    
    print(f"Total transaction costs (50 steps): ${total_costs:.2f}")
    print(f"Total raw return (50 steps): ${total_raw_return:.2f}")
    print(f"Net return (50 steps): ${total_raw_return - total_costs:.2f}")
    print(f"Cost as % of return: {abs(total_costs / max(abs(total_raw_return), 1)) * 100:.2f}%")
    print()
    
    if total_costs > abs(total_raw_return):
        print("⚠️  WARNING: Transaction costs exceed returns!")
        print("   The model may be overtrading")
        print("   Consider:")
        print("   1. Reducing transaction costs")
        print("   2. Penalizing excessive trading")
        print("   3. Rewarding holding profitable positions")
    
    print()
    print("=" * 80)
    print("✅ DIAGNOSIS COMPLETE")
    print("=" * 80)
    print()
    
    # Recommendations
    print("📋 RECOMMENDATIONS:")
    print()
    
    if best_strategy[1]['avg_return'] < 0:
        print("❌ All strategies are losing money!")
        print()
        print("Possible issues:")
        print("1. Transaction costs are too high")
        print("2. Reward function is not aligned with profitability")
        print("3. Environment dynamics are too difficult")
        print("4. Data quality issues")
        print()
        print("Suggested fixes:")
        print("1. Run with --no-realistic-costs to test without transaction costs")
        print("2. Increase reward scaling (currently 1e-4)")
        print("3. Add reward shaping for profitable trades")
        print("4. Reduce episode length to learn faster")
    else:
        print(f"✅ {strategies[best_strategy[0]]} is profitable!")
        print(f"   The model should be able to learn this strategy")
        print()
        print("If the model is still losing money:")
        print("1. Increase exploration (entropy coefficient)")
        print("2. Reduce learning rate for more stable learning")
        print("3. Increase batch size for better gradient estimates")
        print("4. Add reward shaping to guide the model")

if __name__ == '__main__':
    diagnose_environment()

