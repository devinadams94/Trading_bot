#!/usr/bin/env python3
"""Profit-focused training using REAL historical options data"""

import torch
import numpy as np
import logging
from tqdm import tqdm
import os
import sys
import asyncio
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.options_clstm_ppo import OptionsCLSTMPPOAgent
from src.historical_options_data import HistoricalOptionsDataLoader, HistoricalOptionsEnvironment
from config.config_loader import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def train_on_real_data(num_episodes=1000, symbols=None):
    """Train the bot on REAL historical options data for maximum profitability"""
    
    # Load configuration
    config = load_config('config/config_real_data.yaml')
    
    # Default symbols if not provided
    if symbols is None:
        symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMD']
    
    logger.info(f"Loading real historical data for: {symbols}")
    logger.info(f"Training for {num_episodes} episodes")
    
    # Initialize data collector with Alpaca credentials
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not api_secret:
        logger.error("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        return
    
    data_loader = HistoricalOptionsDataLoader(api_key, api_secret)
    
    # Collect historical data
    logger.info("Loading real options data from Alpaca...")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    try:
        # Load historical data for all symbols
        historical_data = await data_loader.load_historical_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            use_cache=True  # Use cached data to speed up loading
        )
        
        # Log what we loaded
        for symbol, df in historical_data.items():
            if not df.empty:
                logger.info(f"  Loaded {len(df)} data points for {symbol}")
            else:
                logger.warning(f"  No data found for {symbol}")
                
    except Exception as e:
        logger.error(f"Error loading historical data: {e}")
        historical_data = {}
    
    if not historical_data or all(df.empty for df in historical_data.values()):
        logger.error("No historical data loaded. Using simulated environment.")
        # Fall back to simulated data
        from src.options_trading_env import OptionsTradingEnvironment
        env = OptionsTradingEnvironment(initial_capital=100000)
    else:
        total_data_points = sum(len(df) for df in historical_data.values() if not df.empty)
        logger.info(f"Creating environment with {total_data_points} total data points")
        
        # Create environment with real historical data
        env = HistoricalOptionsEnvironment(
            historical_data=historical_data,
            data_loader=data_loader,
            symbols=symbols,
            initial_capital=100000,
            max_positions=5
        )
    
    # Create agent with profit-optimized settings
    agent = OptionsCLSTMPPOAgent(
        observation_space=env.observation_space,
        action_space=11,
        learning_rate_actor_critic=5e-5,  # Conservative learning
        learning_rate_clstm=1e-4,
        gamma=0.99,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        batch_size=64,
        n_epochs=5
    )
    
    # Training metrics
    all_profits = []
    all_win_rates = []
    best_avg_profit = -float('inf')
    checkpoint_dir = "checkpoints/real_data_profitable"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger.info("\nStarting profit-focused training on REAL DATA...")
    logger.info("="*60)
    
    # Progress bar
    pbar = tqdm(range(num_episodes), desc="Training on Real Data")
    
    for episode in pbar:
        obs = env.reset()
        episode_return = 0
        initial_value = env._calculate_portfolio_value()
        
        # Run episode with real data
        for step in range(100):  # 100 steps per episode
            # Get action
            action, info = agent.act(obs, deterministic=False)
            
            # Step through REAL historical data
            next_obs, reward, done, env_info = env.step(action)
            
            # Calculate actual profit from real price movements
            current_value = env._calculate_portfolio_value()
            actual_profit = current_value - initial_value
            profit_pct = actual_profit / initial_value
            
            # Profit-focused reward based on REAL P&L
            profit_reward = actual_profit / 1000  # Scale down
            
            # Amplify rewards for profitable trades on real data
            if actual_profit > 0:
                profit_reward *= (1 + 10 * profit_pct)  # Big multiplier
                if hasattr(env, 'consecutive_profits'):
                    env.consecutive_profits += 1
                    profit_reward += env.consecutive_profits * 0.5
                else:
                    env.consecutive_profits = 1
            else:
                profit_reward *= 2  # Penalize losses
                env.consecutive_profits = 0
                
                # Extra penalty for large losses
                if profit_pct < -0.02:  # More than 2% loss
                    profit_reward -= 5
            
            # Bonus for closing profitable positions
            if len(env.closed_positions) > 0:
                last_trade = env.closed_positions[-1]
                if last_trade['pnl'] > 0:
                    profit_reward += 2
            
            # Store transition with modified reward
            agent.store_transition(obs, action, profit_reward, next_obs, done, info)
            
            obs = next_obs
            episode_return += reward
            
            if done:
                break
        
        # Calculate episode metrics with REAL data results
        final_value = env._calculate_portfolio_value()
        episode_profit = final_value - initial_value
        episode_profit_pct = episode_profit / initial_value
        
        # Calculate win rate from real trades
        total_trades = len(env.closed_positions)
        winning_trades = sum(1 for p in env.closed_positions if p['pnl'] > 0)
        win_rate = winning_trades / max(1, total_trades)
        
        all_profits.append(episode_profit_pct)
        all_win_rates.append(win_rate)
        
        # Train agent on real data experiences
        if len(agent.buffer) >= agent.batch_size:
            agent.train()
        
        # Update progress with real results
        recent_profits = all_profits[-50:] if len(all_profits) >= 50 else all_profits
        recent_win_rates = all_win_rates[-50:] if len(all_win_rates) >= 50 else all_win_rates
        
        avg_profit = np.mean(recent_profits)
        avg_win_rate = np.mean(recent_win_rates)
        
        pbar.set_postfix({
            'Avg P&L': f'{avg_profit:.2%}',
            'Win Rate': f'{avg_win_rate:.2%}',
            'Last P&L': f'{episode_profit_pct:.2%}',
            'Symbol': env_info.get('symbol', 'N/A')
        })
        
        # Log detailed progress every 50 episodes
        if episode % 50 == 0 and episode > 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"Episode {episode}/{num_episodes} - REAL DATA RESULTS")
            logger.info(f"  50-episode avg profit: {avg_profit:.2%}")
            logger.info(f"  50-episode win rate: {avg_win_rate:.2%}")
            logger.info(f"  Last episode P&L: ${episode_profit:,.2f} ({episode_profit_pct:.2%})")
            logger.info(f"  Total trades analyzed: {sum(len(p) for p in all_profits)}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"episode_{episode}.pt")
            agent.save(checkpoint_path)
            logger.info(f"  Checkpoint saved: {checkpoint_path}")
            
            # Save best model
            if avg_profit > best_avg_profit and avg_profit > 0:
                best_avg_profit = avg_profit
                best_path = os.path.join(checkpoint_dir, f"best_profit_{avg_profit:.4f}.pt")
                agent.save(best_path)
                logger.info(f"  💰 New best model! Avg profit: {avg_profit:.2%}")
        
        # Early stopping if we achieve consistent profitability
        if episode > 100 and avg_profit > 0.02 and avg_win_rate > 0.6:
            logger.info(f"\n🎉 Target profitability achieved on REAL DATA!")
            logger.info(f"Average profit: {avg_profit:.2%}, Win rate: {avg_win_rate:.2%}")
            final_path = os.path.join(checkpoint_dir, "final_profitable_real_data.pt")
            agent.save(final_path)
            break
    
    # Final results
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE - REAL DATA RESULTS")
    logger.info(f"Episodes trained: {episode + 1}")
    logger.info(f"Final 100-episode average profit: {np.mean(all_profits[-100:]):.2%}")
    logger.info(f"Final 100-episode win rate: {np.mean(all_win_rates[-100:]):.2%}")
    logger.info(f"Best average profit achieved: {best_avg_profit:.2%}")
    logger.info(f"\nBest model saved in: {checkpoint_dir}/")
    
    return agent


def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(description='Train options bot on real data for profitability')
    parser.add_argument('--episodes', type=int, default=1000, 
                        help='Number of episodes to train (default: 1000)')
    parser.add_argument('--symbols', nargs='+', 
                        default=['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMD'],
                        help='Symbols to train on (default: SPY QQQ AAPL TSLA NVDA AMD)')
    parser.add_argument('--test', action='store_true',
                        help='Run a quick test with 50 episodes')
    
    args = parser.parse_args()
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        logger.info("Using CPU (training will be slower)")
    
    # Set number of episodes
    num_episodes = 50 if args.test else args.episodes
    
    # Run training
    asyncio.run(train_on_real_data(num_episodes=num_episodes, symbols=args.symbols))


if __name__ == "__main__":
    main()