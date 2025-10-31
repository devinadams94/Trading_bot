#!/usr/bin/env python3
"""Diagnose why win rate is always 0%"""

import os
import sys
import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.historical_options_data import HistoricalOptionsDataLoader
from train_profitable_optimized import OptimizedProfitableEnvironment
from config.symbols_loader import SymbolsConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_environment():
    """Test the environment to see if trades can be executed"""
    
    # Load some data
    symbols_config = SymbolsConfig()
    symbols = symbols_config.get_training_recommendations()[:3]  # Just 3 symbols
    
    data_loader = HistoricalOptionsDataLoader(
        api_key=os.getenv('ALPACA_API_KEY', 'dummy'),
        api_secret=os.getenv('ALPACA_SECRET_KEY', 'dummy'),
        base_url='https://paper-api.alpaca.markets'
    )
    
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=10)  # Just 10 days
    
    logger.info(f"Loading data for {symbols} from {start_date} to {end_date}")
    historical_data = await data_loader.load_historical_options_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        use_cache=True
    )
    
    # Create environment
    env = OptimizedProfitableEnvironment(
        historical_data=historical_data,
        data_loader=data_loader,
        symbols=symbols,
        initial_capital=100000,
        max_positions=2,
        commission=0.65,
        episode_length=100
    )
    
    # Run a simple episode with specific actions
    obs = env.reset()
    logger.info(f"\nStarting episode with capital: ${env.capital}")
    
    action_sequence = [
        (0, 'hold'),
        (1, 'buy_call'),
        (0, 'hold'),
        (0, 'hold'),
        (1, 'buy_call'),
        (0, 'hold'),
        (10, 'close_all_positions'),
        (2, 'buy_put'),
        (0, 'hold'),
        (10, 'close_all_positions')
    ]
    
    for step, (action, action_name) in enumerate(action_sequence):
        if env.done:
            break
            
        logger.info(f"\nStep {step}: Action = {action_name}")
        
        # Check state before action
        logger.info(f"  Before: Positions={len(env.positions)}, Capital=${env.capital:.0f}")
        
        # Take action
        obs, reward, done, info = env.step(action)
        
        # Check state after action
        logger.info(f"  After: Positions={len(env.positions)}, Capital=${env.capital:.0f}, Reward={reward:.2f}")
        logger.info(f"  Trades: W={env.winning_trades}, L={env.losing_trades}")
        
        # Show position details if any
        if env.positions:
            for i, pos in enumerate(env.positions):
                logger.info(f"    Position {i}: {pos['option_type']} {pos['strike']} @ ${pos['entry_price']:.2f}")
    
    # Final stats
    logger.info(f"\nEpisode complete:")
    logger.info(f"  Final capital: ${env.capital:.0f}")
    logger.info(f"  Total trades: {env.winning_trades + env.losing_trades}")
    logger.info(f"  Winning trades: {env.winning_trades}")
    logger.info(f"  Losing trades: {env.losing_trades}")
    logger.info(f"  Win rate: {env.winning_trades / max(1, env.winning_trades + env.losing_trades):.1%}")
    
    # Check if options data exists
    if hasattr(env, 'training_data') and env.training_data is not None:
        logger.info(f"\nData check:")
        logger.info(f"  Total rows: {len(env.training_data)}")
        logger.info(f"  Unique timestamps: {len(env.training_data['timestamp'].unique())}")
        logger.info(f"  Sample option prices:")
        sample = env.training_data.head(5)
        for idx, row in sample.iterrows():
            logger.info(f"    {row['option_type']} {row['strike']}: bid=${row['bid']:.2f}, ask=${row['ask']:.2f}")

if __name__ == "__main__":
    asyncio.run(test_environment())