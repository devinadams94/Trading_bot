#!/usr/bin/env python3
"""Quick validation script that works with partially trained or untrained models"""

import os
import torch
import numpy as np
import json
from datetime import datetime

from src.options_trading_env import OptionsTradingEnvironment
from src.options_data_collector import OptionsDataSimulator
from src.options_clstm_ppo import OptionsCLSTMPPOAgent

def quick_validate():
    """Run quick validation on a model (or random agent if no model exists)"""
    
    print("=" * 60)
    print("QUICK VALIDATION - Options Trading Bot")
    print("=" * 60)
    
    # Initialize environment
    env = OptionsTradingEnvironment(initial_capital=100000)
    
    # Initialize agent
    agent = OptionsCLSTMPPOAgent(
        observation_space=env.observation_space,
        action_space=env.action_space.n
    )
    
    # Try to load latest checkpoint
    checkpoint_dir = "checkpoints/options_clstm_ppo"
    model_loaded = False
    
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        if checkpoints:
            latest = sorted(checkpoints)[-1]
            model_path = os.path.join(checkpoint_dir, latest)
            try:
                agent.load(model_path)
                print(f"\n✅ Loaded model: {model_path}")
                model_loaded = True
            except:
                print(f"\n⚠️  Could not load model, using random agent")
    
    if not model_loaded:
        print("\n⚠️  No trained model found, using random agent for demonstration")
    
    # Data simulator
    simulator = OptionsDataSimulator()
    
    # Run validation episodes
    num_episodes = 5
    results = {
        'episodes': [],
        'total_trades': 0,
        'actions_taken': {},
        'model_loaded': model_loaded
    }
    
    print(f"\nRunning {num_episodes} validation episodes...")
    print("-" * 60)
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_trades = []
        
        symbol = ['SPY', 'TSLA', 'NVDA'][episode % 3]
        stock_price = np.random.uniform(100, 500)
        
        print(f"\nEpisode {episode + 1}: {symbol} at ${stock_price:.2f}")
        
        for step in range(50):  # Shorter episodes for quick validation
            # Generate options
            options_chain = simulator.simulate_options_chain(
                symbol=symbol,
                stock_price=stock_price,
                num_strikes=20,
                num_expirations=4
            )
            
            # Update observation
            if 'options_chain' in obs:
                sorted_options = sorted(options_chain, key=lambda x: x.volume, reverse=True)[:20]
                options_features = []
                for opt in sorted_options:
                    features = [
                        opt.strike, opt.bid, opt.ask, opt.last_price, opt.volume,
                        opt.open_interest, opt.implied_volatility, opt.delta,
                        opt.gamma, opt.theta, opt.vega, opt.rho,
                        1.0 if opt.option_type == 'call' else 0.0,
                        30, (opt.bid + opt.ask) / 2
                    ]
                    options_features.append(features)
                while len(options_features) < 20:
                    options_features.append([0] * 15)
                obs['options_chain'] = np.array(options_features[:20], dtype=np.float32)
            
            # Get action
            action, _ = agent.act(obs, deterministic=True)
            action_name = env.action_mapping[action]
            
            # Step
            next_obs, reward, done, info = env.step(action)
            
            # Track trades
            if action_name not in ['hold', 'close_all_positions']:
                episode_trades.append(action_name)
                results['total_trades'] += 1
            
            # Track action distribution
            results['actions_taken'][action_name] = results['actions_taken'].get(action_name, 0) + 1
            
            episode_reward += reward
            obs = next_obs
            
            # Simulate price movement
            stock_price *= np.random.uniform(0.995, 1.005)
            
            if done:
                break
        
        # Calculate metrics
        final_value = env._calculate_portfolio_value()
        episode_return = (final_value - env.initial_capital) / env.initial_capital
        
        print(f"  Return: {episode_return:.2%}")
        print(f"  Final Value: ${final_value:,.2f}")
        print(f"  Trades: {len(episode_trades)}")
        if episode_trades:
            print(f"  Actions: {', '.join(set(episode_trades))}")
        
        results['episodes'].append({
            'symbol': symbol,
            'return': float(episode_return),
            'final_value': float(final_value),
            'trades': len(episode_trades),
            'reward': float(episode_reward)
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    avg_return = np.mean([e['return'] for e in results['episodes']])
    total_trades = sum([e['trades'] for e in results['episodes']])
    
    print(f"Average Return: {avg_return:.2%}")
    print(f"Total Trades: {total_trades}")
    print(f"Model Status: {'Trained' if model_loaded else 'Random Agent'}")
    
    print("\nAction Distribution:")
    for action, count in sorted(results['actions_taken'].items(), key=lambda x: x[1], reverse=True):
        pct = count / sum(results['actions_taken'].values()) * 100
        print(f"  {action}: {count} ({pct:.1f}%)")
    
    # Assessment
    print("\n" + "=" * 60)
    print("QUICK ASSESSMENT")
    print("=" * 60)
    
    if not model_loaded:
        print("❓ Using random agent - train the model for real results")
        print("   Run: ./venv/bin/python simple_train.py --episodes 100")
    elif avg_return > 0.01:
        print("✅ Model shows positive returns!")
    elif avg_return > -0.01:
        print("⚠️  Model is roughly break-even")
    else:
        print("❌ Model shows negative returns - needs more training")
    
    if total_trades == 0:
        print("❌ No trades executed - model may be too conservative")
    elif total_trades > 20:
        print("⚠️  High trade frequency - check if overtrading")
    else:
        print("✅ Reasonable trading frequency")
    
    # Save results
    with open('quick_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to quick_validation_results.json")
    print("\nFor full validation, run:")
    print("  ./venv/bin/python advanced_validation.py --model MODEL_PATH")

if __name__ == "__main__":
    quick_validate()