#!/usr/bin/env python3
"""Complete training and validation pipeline for options trading bot"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Train and validate options trading bot')
    parser.add_argument('--mode', choices=['quick', 'full', 'validate'], default='quick',
                       help='Training mode: quick (100 episodes), full (1000 episodes), or validate only')
    parser.add_argument('--model', type=str, help='Path to saved model for validation')
    args = parser.parse_args()
    
    if args.mode == 'validate' and not args.model:
        print("Error: --model path required for validation mode")
        sys.exit(1)
    
    # Training parameters based on mode
    episodes = 100 if args.mode == 'quick' else 1000
    
    print("=" * 80)
    print(f"OPTIONS TRADING BOT - TRAINING & VALIDATION")
    print("=" * 80)
    
    if args.mode != 'validate':
        # STEP 1: TRAINING
        print("\n📚 STEP 1: TRAINING PHASE")
        print("-" * 40)
        
        # Train with popular volatile symbols
        symbols = ['SPY', 'QQQ', 'TSLA', 'NVDA', 'AMD', 'PLTR', 'META', 'COIN']
        print(f"Training on symbols: {', '.join(symbols)}")
        print(f"Episodes: {episodes}")
        
        # Create training command - use GPU-optimized training
        train_cmd = f"./venv/bin/python train_gpu_optimized.py --episodes {episodes} --batch-size 64"
        
        print(f"\nExecuting: {train_cmd}")
        print("This may take a while...\n")
        
        # Execute training
        os.system(train_cmd)
        
        # Find the latest checkpoint
        checkpoint_dir = "checkpoints/options_clstm_ppo"
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            if checkpoints:
                # Sort by modification time
                checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
                latest_model = os.path.join(checkpoint_dir, checkpoints[-1])
                print(f"\n✅ Training complete! Model saved: {latest_model}")
            else:
                print("\n❌ No checkpoint found after training")
                sys.exit(1)
        else:
            print("\n❌ Checkpoint directory not found")
            sys.exit(1)
    else:
        latest_model = args.model
    
    # STEP 2: VALIDATION
    print("\n🔍 STEP 2: VALIDATION PHASE")
    print("-" * 40)
    
    # Create validation script
    validation_script = """
import numpy as np
import torch
from src.options_trading_env import OptionsTradingEnvironment
from src.options_data_collector import OptionsDataSimulator
from src.options_clstm_ppo import OptionsCLSTMPPOAgent
import json

# Load environment and agent
env = OptionsTradingEnvironment(initial_capital=100000)
agent = OptionsCLSTMPPOAgent(
    observation_space=env.observation_space,
    action_space=env.action_space.n
)

# Load trained model
model_path = '""" + latest_model + """'
agent.load(model_path)
print(f"Loaded model: {model_path}")

# Validation parameters
num_episodes = 20
simulator = OptionsDataSimulator()

# Metrics tracking
results = {
    'episodes': [],
    'total_trades': 0,
    'winning_trades': 0,
    'losing_trades': 0,
    'actions_taken': {},
    'final_values': [],
    'max_drawdowns': [],
    'sharpe_ratios': []
}

print("\\nRunning validation episodes...")
print("-" * 60)

for episode in range(num_episodes):
    obs = env.reset()
    episode_rewards = []
    episode_actions = []
    portfolio_values = [env.initial_capital]
    
    # Test on different symbols
    test_symbol = ['SPY', 'TSLA', 'NVDA', 'AMD'][episode % 4]
    stock_price = np.random.uniform(100, 500)
    
    print(f"\\nEpisode {episode + 1}/{num_episodes} - Testing {test_symbol} at ${stock_price:.2f}")
    
    for step in range(100):  # 100 steps per episode
        # Generate options chain
        options_chain = simulator.simulate_options_chain(
            symbol=test_symbol,
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
        
        # Get action from trained agent
        action, _ = agent.act(obs, deterministic=True)
        action_name = env.action_mapping[action]
        
        # Step environment
        next_obs, reward, done, info = env.step(action)
        
        # Track metrics
        episode_rewards.append(reward)
        episode_actions.append(action_name)
        portfolio_values.append(env._calculate_portfolio_value())
        
        # Update state
        obs = next_obs
        
        # Simulate price movement
        stock_price *= np.random.uniform(0.99, 1.01)
        
        if done:
            break
    
    # Calculate episode metrics
    total_reward = sum(episode_rewards)
    final_value = portfolio_values[-1]
    
    # Count trades
    trade_actions = [a for a in episode_actions if a not in ['hold', 'close_all_positions']]
    if trade_actions:
        results['total_trades'] += len(trade_actions)
        if final_value > env.initial_capital:
            results['winning_trades'] += 1
        else:
            results['losing_trades'] += 1
    
    # Track action distribution
    for action in episode_actions:
        results['actions_taken'][action] = results['actions_taken'].get(action, 0) + 1
    
    # Calculate max drawdown
    portfolio_array = np.array(portfolio_values)
    running_max = np.maximum.accumulate(portfolio_array)
    drawdowns = (portfolio_array - running_max) / running_max
    max_drawdown = abs(drawdowns.min())
    
    # Calculate Sharpe ratio (simplified)
    if len(episode_rewards) > 1:
        returns = np.diff(portfolio_array) / portfolio_array[:-1]
        sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-8)
    else:
        sharpe = 0
    
    # Store results
    results['episodes'].append({
        'total_reward': total_reward,
        'final_value': final_value,
        'return': (final_value - env.initial_capital) / env.initial_capital,
        'trades': len(trade_actions),
        'max_drawdown': max_drawdown
    })
    results['final_values'].append(final_value)
    results['max_drawdowns'].append(max_drawdown)
    results['sharpe_ratios'].append(sharpe)
    
    print(f"  Return: {(final_value - env.initial_capital) / env.initial_capital:.2%}")
    print(f"  Trades: {len(trade_actions)}")
    print(f"  Max Drawdown: {max_drawdown:.2%}")
    print(f"  Actions: {', '.join(set(trade_actions))}")

# Calculate summary statistics
avg_return = np.mean([(e['final_value'] - env.initial_capital) / env.initial_capital for e in results['episodes']])
avg_sharpe = np.mean([s for s in results['sharpe_ratios'] if s != 0])
avg_drawdown = np.mean(results['max_drawdowns'])
win_rate = results['winning_trades'] / max(1, results['total_trades'])

print("\\n" + "=" * 60)
print("VALIDATION RESULTS SUMMARY")
print("=" * 60)
print(f"Episodes Run: {num_episodes}")
print(f"Average Return: {avg_return:.2%}")
print(f"Average Sharpe Ratio: {avg_sharpe:.2f}")
print(f"Average Max Drawdown: {avg_drawdown:.2%}")
print(f"Total Trades: {results['total_trades']}")
print(f"Win Rate: {win_rate:.2%}")
print(f"\\nAction Distribution:")
for action, count in sorted(results['actions_taken'].items(), key=lambda x: x[1], reverse=True):
    print(f"  {action}: {count} ({count/sum(results['actions_taken'].values()):.1%})")

# Save results
with open('validation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\\nDetailed results saved to validation_results.json")

# Performance assessment
print("\\n" + "=" * 60)
print("PERFORMANCE ASSESSMENT")
print("=" * 60)

if avg_return > 0.05:
    print("✅ EXCELLENT: Model shows strong positive returns")
elif avg_return > 0.02:
    print("✅ GOOD: Model shows positive returns")
elif avg_return > 0:
    print("⚠️  ACCEPTABLE: Model shows small positive returns")
else:
    print("❌ POOR: Model shows negative returns")

if avg_sharpe > 1.5:
    print("✅ EXCELLENT: Strong risk-adjusted returns")
elif avg_sharpe > 1.0:
    print("✅ GOOD: Decent risk-adjusted returns")
elif avg_sharpe > 0.5:
    print("⚠️  ACCEPTABLE: Moderate risk-adjusted returns")
else:
    print("❌ POOR: Low risk-adjusted returns")

if avg_drawdown < 0.1:
    print("✅ EXCELLENT: Low drawdown risk")
elif avg_drawdown < 0.2:
    print("✅ GOOD: Acceptable drawdown")
elif avg_drawdown < 0.3:
    print("⚠️  ACCEPTABLE: Moderate drawdown")
else:
    print("❌ POOR: High drawdown risk")

if win_rate > 0.6:
    print("✅ EXCELLENT: High win rate")
elif win_rate > 0.5:
    print("✅ GOOD: Positive win rate")
elif win_rate > 0.4:
    print("⚠️  ACCEPTABLE: Below average win rate")
else:
    print("❌ POOR: Low win rate")
"""
    
    # Save and run validation script
    with open('run_validation.py', 'w') as f:
        f.write(validation_script)
    
    print("Running validation on trained model...")
    os.system("./venv/bin/python run_validation.py")
    
    # STEP 3: VISUALIZATION
    print("\n📊 STEP 3: CREATING PERFORMANCE PLOTS")
    print("-" * 40)
    
    # Create visualization script
    viz_script = """
import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('validation_results.json', 'r') as f:
    results = json.load(f)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Options Trading Bot Performance Validation', fontsize=16)

# 1. Portfolio Value Over Episodes
ax = axes[0, 0]
final_values = results['final_values']
episodes = range(1, len(final_values) + 1)
ax.plot(episodes, final_values, 'b-', linewidth=2)
ax.axhline(y=100000, color='r', linestyle='--', label='Initial Capital')
ax.set_xlabel('Episode')
ax.set_ylabel('Portfolio Value ($)')
ax.set_title('Portfolio Value Across Validation Episodes')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Returns Distribution
ax = axes[0, 1]
returns = [(v - 100000) / 100000 * 100 for v in final_values]
ax.hist(returns, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
ax.axvline(x=0, color='r', linestyle='--', label='Break-even')
ax.set_xlabel('Return (%)')
ax.set_ylabel('Frequency')
ax.set_title('Returns Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Action Distribution
ax = axes[1, 0]
actions = list(results['actions_taken'].keys())
counts = list(results['actions_taken'].values())
colors = plt.cm.Set3(range(len(actions)))
ax.bar(actions, counts, color=colors)
ax.set_xlabel('Action')
ax.set_ylabel('Count')
ax.set_title('Trading Actions Distribution')
ax.tick_params(axis='x', rotation=45)
for i, v in enumerate(counts):
    ax.text(i, v + 0.5, str(v), ha='center')

# 4. Drawdown Analysis
ax = axes[1, 1]
drawdowns = [d * 100 for d in results['max_drawdowns']]
ax.plot(episodes, drawdowns, 'r-', linewidth=2, marker='o')
ax.fill_between(episodes, 0, drawdowns, alpha=0.3, color='red')
ax.set_xlabel('Episode')
ax.set_ylabel('Max Drawdown (%)')
ax.set_title('Maximum Drawdown per Episode')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('validation_performance.png', dpi=300, bbox_inches='tight')
print("Performance plots saved to validation_performance.png")
plt.show()
"""
    
    with open('create_plots.py', 'w') as f:
        f.write(viz_script)
    
    os.system("./venv/bin/python create_plots.py")
    
    # Cleanup
    os.remove('run_validation.py')
    os.remove('create_plots.py')
    
    print("\n" + "=" * 80)
    print("✅ TRAINING AND VALIDATION COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review validation_results.json for detailed metrics")
    print("2. Check validation_performance.png for visual analysis")
    print("3. If results are good, run live simulation:")
    print(f"   ./venv/bin/python main_options_clstm_ppo.py --mode simulation")
    print("4. For more training:")
    print(f"   ./venv/bin/python train_options_clstm_ppo.py --simulated --episodes 1000")

if __name__ == '__main__':
    main()