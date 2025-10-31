#!/usr/bin/env python3
"""Advanced validation and backtesting for the options trading bot"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import argparse
from typing import Dict, List, Tuple

def backtest_strategy(model_path: str, test_days: int = 30) -> Dict:
    """Run a comprehensive backtest of the trained model"""
    
    from src.options_trading_env import OptionsTradingEnvironment
    from src.options_data_collector import OptionsDataSimulator
    from src.options_clstm_ppo import OptionsCLSTMPPOAgent
    
    print(f"Loading model from: {model_path}")
    
    # Initialize components
    env = OptionsTradingEnvironment(
        initial_capital=100000,
        commission=0.65
    )
    
    agent = OptionsCLSTMPPOAgent(
        observation_space=env.observation_space,
        action_space=env.action_space.n
    )
    agent.load(model_path)
    
    simulator = OptionsDataSimulator()
    
    # Backtest parameters
    symbols = ['SPY', 'QQQ', 'TSLA', 'NVDA', 'AMD', 'PLTR', 'META', 'COIN']
    results = {
        'daily_returns': [],
        'trades': [],
        'positions': [],
        'portfolio_values': [],
        'timestamps': []
    }
    
    print(f"\nRunning {test_days}-day backtest on {len(symbols)} symbols...")
    print("-" * 60)
    
    # Simulate each trading day
    for day in range(test_days):
        daily_trades = []
        daily_pnl = 0
        
        # Morning portfolio value
        morning_value = env._calculate_portfolio_value()
        
        # Trade each symbol
        for symbol in symbols:
            # Reset environment for each symbol
            obs = env.reset()
            
            # Simulate intraday price movement
            base_price = np.random.uniform(100, 500)
            
            for hour in range(7):  # 7 trading hours
                # Generate realistic price movement
                price_change = np.random.normal(0, 0.002)  # 0.2% std dev
                current_price = base_price * (1 + price_change)
                
                # Generate options chain
                options_chain = simulator.simulate_options_chain(
                    symbol=symbol,
                    stock_price=current_price,
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
                
                # Get trading decision
                action, info = agent.act(obs, deterministic=True)
                action_name = env.action_mapping[action]
                
                # Execute trade
                if action_name not in ['hold', 'close_all_positions']:
                    # Record trade
                    trade = {
                        'timestamp': datetime.now() + timedelta(days=day, hours=hour),
                        'symbol': symbol,
                        'action': action_name,
                        'price': current_price,
                        'confidence': info.get('confidence', 0)
                    }
                    daily_trades.append(trade)
                    results['trades'].append(trade)
                
                # Step environment
                next_obs, reward, done, _ = env.step(action)
                daily_pnl += reward
                
                obs = next_obs
                base_price = current_price
        
        # End of day calculations
        evening_value = env._calculate_portfolio_value()
        daily_return = (evening_value - morning_value) / morning_value
        
        results['daily_returns'].append(daily_return)
        results['portfolio_values'].append(evening_value)
        results['timestamps'].append(datetime.now() + timedelta(days=day))
        
        if day % 5 == 0:
            print(f"Day {day+1}: Portfolio ${evening_value:,.2f}, "
                  f"Daily return: {daily_return:.2%}, "
                  f"Trades: {len(daily_trades)}")
    
    return analyze_backtest_results(results)

def analyze_backtest_results(results: Dict) -> Dict:
    """Analyze backtest results and calculate performance metrics"""
    
    # Convert to pandas for easier analysis
    returns = pd.Series(results['daily_returns'])
    portfolio = pd.Series(results['portfolio_values'])
    
    # Calculate metrics
    metrics = {
        'total_return': (portfolio.iloc[-1] - 100000) / 100000,
        'annualized_return': returns.mean() * 252,
        'volatility': returns.std() * np.sqrt(252),
        'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
        'max_drawdown': calculate_max_drawdown(portfolio),
        'win_rate': len(returns[returns > 0]) / len(returns),
        'average_win': returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0,
        'average_loss': returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0,
        'total_trades': len(results['trades']),
        'calmar_ratio': returns.mean() * 252 / calculate_max_drawdown(portfolio)
    }
    
    # Risk metrics
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean()
    metrics['var_95'] = var_95
    metrics['cvar_95'] = cvar_95
    
    return metrics

def calculate_max_drawdown(portfolio_values: pd.Series) -> float:
    """Calculate maximum drawdown from portfolio values"""
    running_max = portfolio_values.expanding().max()
    drawdown = (portfolio_values - running_max) / running_max
    return abs(drawdown.min())

def stress_test_model(model_path: str) -> Dict:
    """Run stress tests on the model under various market conditions"""
    
    from src.options_trading_env import OptionsTradingEnvironment
    from src.options_data_collector import OptionsDataSimulator
    from src.options_clstm_ppo import OptionsCLSTMPPOAgent
    
    print(f"\nRunning stress tests...")
    
    # Load model
    env = OptionsTradingEnvironment(initial_capital=100000)
    agent = OptionsCLSTMPPOAgent(
        observation_space=env.observation_space,
        action_space=env.action_space.n
    )
    agent.load(model_path)
    
    simulator = OptionsDataSimulator()
    
    # Test scenarios
    scenarios = {
        'bull_market': {'trend': 0.001, 'volatility': 0.01},
        'bear_market': {'trend': -0.001, 'volatility': 0.01},
        'high_volatility': {'trend': 0, 'volatility': 0.03},
        'flash_crash': {'trend': -0.01, 'volatility': 0.05},
        'steady_growth': {'trend': 0.0005, 'volatility': 0.005}
    }
    
    results = {}
    
    for scenario_name, params in scenarios.items():
        print(f"\nTesting {scenario_name}...")
        
        obs = env.reset()
        scenario_returns = []
        
        # Run 100 steps
        base_price = 450  # SPY price
        
        for step in range(100):
            # Simulate price with scenario parameters
            price_change = np.random.normal(params['trend'], params['volatility'])
            base_price *= (1 + price_change)
            
            # Generate options
            options_chain = simulator.simulate_options_chain(
                symbol='SPY',
                stock_price=base_price,
                num_strikes=20,
                num_expirations=4
            )
            
            # Update observation (simplified)
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
            
            # Step
            next_obs, reward, done, _ = env.step(action)
            scenario_returns.append(reward)
            
            obs = next_obs
            
            if done:
                break
        
        # Calculate scenario metrics
        total_return = sum(scenario_returns)
        avg_return = np.mean(scenario_returns)
        volatility = np.std(scenario_returns)
        
        results[scenario_name] = {
            'total_return': total_return,
            'average_return': avg_return,
            'volatility': volatility,
            'final_portfolio': env._calculate_portfolio_value()
        }
        
        print(f"  Total return: {total_return:.2f}")
        print(f"  Final portfolio: ${env._calculate_portfolio_value():,.2f}")
    
    return results

def create_validation_report(model_path: str, output_dir: str = 'validation_reports'):
    """Create a comprehensive validation report"""
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE MODEL VALIDATION REPORT")
    print("=" * 80)
    
    # 1. Run backtest
    print("\n1. Running 30-day backtest...")
    backtest_results = backtest_strategy(model_path, test_days=30)
    
    # 2. Run stress tests
    print("\n2. Running stress tests...")
    stress_results = stress_test_model(model_path)
    
    # 3. Create visualizations
    print("\n3. Creating performance visualizations...")
    
    fig = plt.figure(figsize=(20, 12))
    
    # Performance metrics summary
    ax1 = plt.subplot(2, 3, 1)
    metrics = list(backtest_results.keys())[:6]
    values = [backtest_results[m] for m in metrics]
    colors = ['green' if v > 0 else 'red' for v in values]
    ax1.bar(range(len(metrics)), values, color=colors, alpha=0.7)
    ax1.set_xticks(range(len(metrics)))
    ax1.set_xticklabels(metrics, rotation=45)
    ax1.set_title('Key Performance Metrics')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Stress test results
    ax2 = plt.subplot(2, 3, 2)
    scenarios = list(stress_results.keys())
    returns = [stress_results[s]['total_return'] for s in scenarios]
    ax2.bar(scenarios, returns, color='skyblue', alpha=0.7)
    ax2.set_xlabel('Market Scenario')
    ax2.set_ylabel('Total Return')
    ax2.set_title('Stress Test Performance')
    ax2.tick_params(axis='x', rotation=45)
    
    # Returns distribution
    ax3 = plt.subplot(2, 3, 3)
    if 'daily_returns' in backtest_results:
        ax3.hist(backtest_results['daily_returns'], bins=20, color='purple', alpha=0.7)
        ax3.axvline(x=0, color='red', linestyle='--')
        ax3.set_xlabel('Daily Return')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Returns Distribution')
    
    # Risk-return scatter
    ax4 = plt.subplot(2, 3, 4)
    for scenario, results in stress_results.items():
        ax4.scatter(results['volatility'], results['average_return'], 
                   label=scenario, s=100)
    ax4.set_xlabel('Volatility')
    ax4.set_ylabel('Average Return')
    ax4.set_title('Risk-Return Profile')
    ax4.legend()
    
    # Performance summary text
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    summary_text = f"""
    Model Performance Summary
    ========================
    
    Total Return: {backtest_results.get('total_return', 0):.2%}
    Sharpe Ratio: {backtest_results.get('sharpe_ratio', 0):.2f}
    Max Drawdown: {backtest_results.get('max_drawdown', 0):.2%}
    Win Rate: {backtest_results.get('win_rate', 0):.2%}
    Total Trades: {backtest_results.get('total_trades', 0)}
    
    Risk Metrics:
    - VaR (95%): {backtest_results.get('var_95', 0):.2%}
    - CVaR (95%): {backtest_results.get('cvar_95', 0):.2%}
    - Calmar Ratio: {backtest_results.get('calmar_ratio', 0):.2f}
    """
    ax5.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
    
    # Save plot
    plt.tight_layout()
    report_path = os.path.join(output_dir, f'validation_report_{timestamp}.png')
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    print(f"\nValidation report saved to: {report_path}")
    
    # Save detailed results
    detailed_results = {
        'timestamp': timestamp,
        'model_path': model_path,
        'backtest_results': backtest_results,
        'stress_test_results': {k: {kk: float(vv) if isinstance(vv, np.number) else vv 
                                    for kk, vv in v.items()} 
                               for k, v in stress_results.items()}
    }
    
    json_path = os.path.join(output_dir, f'validation_results_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"Detailed results saved to: {json_path}")
    
    # Performance assessment
    print("\n" + "=" * 60)
    print("PERFORMANCE ASSESSMENT")
    print("=" * 60)
    
    total_return = backtest_results.get('total_return', 0)
    sharpe = backtest_results.get('sharpe_ratio', 0)
    max_dd = backtest_results.get('max_drawdown', 0)
    win_rate = backtest_results.get('win_rate', 0)
    
    score = 0
    
    if total_return > 0.1:
        print("✅ EXCELLENT: Strong returns (>10%)")
        score += 25
    elif total_return > 0.05:
        print("✅ GOOD: Positive returns (>5%)")
        score += 20
    elif total_return > 0:
        print("⚠️  FAIR: Small positive returns")
        score += 10
    else:
        print("❌ POOR: Negative returns")
    
    if sharpe > 2:
        print("✅ EXCELLENT: Outstanding risk-adjusted returns")
        score += 25
    elif sharpe > 1:
        print("✅ GOOD: Strong risk-adjusted returns")
        score += 20
    elif sharpe > 0.5:
        print("⚠️  FAIR: Acceptable risk-adjusted returns")
        score += 10
    else:
        print("❌ POOR: Low risk-adjusted returns")
    
    if max_dd < 0.1:
        print("✅ EXCELLENT: Very low drawdown")
        score += 25
    elif max_dd < 0.2:
        print("✅ GOOD: Acceptable drawdown")
        score += 20
    elif max_dd < 0.3:
        print("⚠️  FAIR: Moderate drawdown")
        score += 10
    else:
        print("❌ POOR: High drawdown risk")
    
    if win_rate > 0.6:
        print("✅ EXCELLENT: High win rate")
        score += 25
    elif win_rate > 0.5:
        print("✅ GOOD: Positive win rate")
        score += 20
    elif win_rate > 0.45:
        print("⚠️  FAIR: Below average win rate")
        score += 10
    else:
        print("❌ POOR: Low win rate")
    
    print(f"\nOVERALL SCORE: {score}/100")
    
    if score >= 80:
        print("🌟 Model is performing EXCELLENTLY and ready for paper trading!")
    elif score >= 60:
        print("✅ Model is performing WELL and can be used with caution")
    elif score >= 40:
        print("⚠️  Model needs MORE TRAINING before deployment")
    else:
        print("❌ Model is NOT READY - significant improvements needed")
    
    return detailed_results

def main():
    parser = argparse.ArgumentParser(description='Advanced validation for options trading bot')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--quick', action='store_true', help='Run quick validation only')
    parser.add_argument('--output-dir', type=str, default='validation_reports', 
                       help='Directory for validation reports')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    if args.quick:
        # Quick validation
        print("Running quick validation...")
        metrics = backtest_strategy(args.model, test_days=7)
        print("\nQuick Validation Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    else:
        # Full validation
        create_validation_report(args.model, args.output_dir)

if __name__ == '__main__':
    main()