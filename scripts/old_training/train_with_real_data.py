#!/usr/bin/env python3
"""Simplified real data training that works with the existing codebase"""

import os
import sys
import torch
import numpy as np
import logging
from tqdm import tqdm
import asyncio
from datetime import datetime, timedelta
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.options_clstm_ppo import OptionsCLSTMPPOAgent
from src.historical_options_data import HistoricalOptionsDataLoader, HistoricalOptionsEnvironment
from src.options_trading_env import OptionsTradingEnvironment
from config.config_loader import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main(num_episodes=1000, use_real_data=True, symbols=None):
    """Main training function"""
    
    # Default symbols
    if symbols is None:
        symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA']
    
    logger.info(f"Training configuration:")
    logger.info(f"  Episodes: {num_episodes}")
    logger.info(f"  Data source: {'Real (Alpaca)' if use_real_data else 'Simulated'}")
    logger.info(f"  Symbols: {', '.join(symbols)}")
    
    # Create environment
    if use_real_data:
        # Check API credentials
        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_SECRET_KEY')
        
        if not api_key or not api_secret:
            logger.warning("Alpaca credentials not found. Falling back to simulated data.")
            env = OptionsTradingEnvironment(initial_capital=100000, max_positions=5)
        else:
            try:
                # Load real historical data
                logger.info("Loading historical options data...")
                data_loader = HistoricalOptionsDataLoader(api_key, api_secret)
                
                # Load last 30 days
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                
                historical_data = await data_loader.load_historical_data(
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=True
                )
                
                # Check if we got data
                if historical_data and any(not df.empty for df in historical_data.values()):
                    logger.info(f"Loaded data for {len([s for s, df in historical_data.items() if not df.empty])} symbols")
                    env = HistoricalOptionsEnvironment(
                        historical_data=historical_data,
                        data_loader=data_loader,
                        symbols=symbols,
                        initial_capital=100000,
                        max_positions=5
                    )
                else:
                    logger.warning("No historical data found. Using simulated environment.")
                    env = OptionsTradingEnvironment(initial_capital=100000, max_positions=5)
                    
            except Exception as e:
                logger.error(f"Error loading historical data: {e}")
                logger.info("Falling back to simulated environment")
                env = OptionsTradingEnvironment(initial_capital=100000, max_positions=5)
    else:
        env = OptionsTradingEnvironment(initial_capital=100000, max_positions=5)
    
    # Create agent with profit-focused settings
    agent = OptionsCLSTMPPOAgent(
        observation_space=env.observation_space,
        action_space=11,
        learning_rate_actor_critic=5e-5,
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
    
    # Create checkpoint directory
    checkpoint_dir = "checkpoints/profitable_models"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger.info("\nStarting profit-focused training...")
    logger.info("="*60)
    
    # Progress bar
    pbar = tqdm(range(num_episodes), desc="Training")
    
    for episode in pbar:
        obs = env.reset()
        
        # Skip if no observation (no data)
        if obs is None:
            continue
            
        initial_value = env._calculate_portfolio_value()
        episode_pnl = 0
        trades_won = 0
        trades_total = 0
        
        # Run episode
        for step in range(100):
            # Get action
            action, info = agent.act(obs, deterministic=False)
            
            # Step environment
            next_obs, reward, done, env_info = env.step(action)
            
            # Calculate P&L
            current_value = env._calculate_portfolio_value()
            episode_pnl = current_value - initial_value
            profit_pct = episode_pnl / initial_value
            
            # Profit-focused reward
            profit_reward = episode_pnl / 1000  # Scale
            
            if episode_pnl > 0:
                profit_reward *= (1 + 10 * profit_pct)  # Amplify profits
            else:
                profit_reward *= 2  # Penalize losses
            
            # Store transition
            agent.store_transition(obs, action, profit_reward, next_obs, done, info)
            
            obs = next_obs
            
            if done or next_obs is None:
                break
        
        # Calculate episode metrics
        final_value = env._calculate_portfolio_value()
        episode_profit = final_value - initial_value
        episode_profit_pct = episode_profit / initial_value
        
        # Track win rate
        if hasattr(env, 'closed_positions'):
            new_trades = len(env.closed_positions) - trades_total
            if new_trades > 0:
                trades_total = len(env.closed_positions)
                trades_won = sum(1 for p in env.closed_positions if p['pnl'] > 0)
        
        win_rate = trades_won / max(1, trades_total)
        
        all_profits.append(episode_profit_pct)
        all_win_rates.append(win_rate)
        
        # Train agent
        if len(agent.buffer) >= agent.batch_size:
            agent.train()
        
        # Update progress
        avg_profit = np.mean(all_profits[-50:]) if len(all_profits) > 50 else episode_profit_pct
        avg_win_rate = np.mean(all_win_rates[-50:]) if len(all_win_rates) > 50 else win_rate
        
        pbar.set_postfix({
            'Avg P&L': f'{avg_profit:.2%}',
            'Win Rate': f'{avg_win_rate:.2%}',
            'Last P&L': f'{episode_profit_pct:.2%}'
        })
        
        # Save best model
        if episode % 10 == 0 and episode > 0:
            if avg_profit > best_avg_profit and avg_profit > 0:
                best_avg_profit = avg_profit
                best_path = os.path.join(checkpoint_dir, f"best_profit_{avg_profit:.4f}.pt")
                agent.save(best_path)
                logger.info(f"\n💰 New best model! Avg profit: {avg_profit:.2%}")
        
        # Log progress
        if episode % 50 == 0 and episode > 0:
            logger.info(f"\nEpisode {episode}/{num_episodes}")
            logger.info(f"  50-episode avg profit: {avg_profit:.2%}")
            logger.info(f"  50-episode win rate: {avg_win_rate:.2%}")
            logger.info(f"  Last episode P&L: ${episode_profit:,.2f} ({episode_profit_pct:.2%})")
        
        # Early stopping
        if episode > 100 and avg_profit > 0.02 and avg_win_rate > 0.6:
            logger.info(f"\n🎉 Target profitability achieved!")
            logger.info(f"Average profit: {avg_profit:.2%}, Win rate: {avg_win_rate:.2%}")
            final_path = os.path.join(checkpoint_dir, "final_profitable.pt")
            agent.save(final_path)
            break
    
    # Final results
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Final 100-episode average profit: {np.mean(all_profits[-100:]):.2%}")
    logger.info(f"Final 100-episode win rate: {np.mean(all_win_rates[-100:]):.2%}")
    logger.info(f"Best average profit: {best_avg_profit:.2%}")
    logger.info(f"Models saved in: {checkpoint_dir}/")
    
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--simulated', action='store_true', help='Use simulated data')
    parser.add_argument('--symbols', nargs='+', default=['SPY', 'QQQ', 'AAPL', 'TSLA'])
    
    args = parser.parse_args()
    
    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    # Run training
    asyncio.run(main(
        num_episodes=args.episodes,
        use_real_data=not args.simulated,
        symbols=args.symbols
    ))