#!/usr/bin/env python3
"""Debug training issues"""

import asyncio
import os
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv

from src.historical_options_data import HistoricalOptionsDataLoader, HistoricalOptionsEnvironment
from src.options_trading_env import OptionsTradingEnvironment
from src.options_clstm_ppo import OptionsCLSTMPPOAgent

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def debug_issues():
    # Load some data
    loader = HistoricalOptionsDataLoader(
        api_key=os.getenv('ALPACA_API_KEY'),
        api_secret=os.getenv('ALPACA_SECRET_KEY')
    )
    
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=5)
    
    data = await loader.load_historical_options_data(
        symbols=['SPY'],
        start_date=start_date,
        end_date=end_date
    )
    
    # Create environments
    hist_env = HistoricalOptionsEnvironment(
        historical_data=data,
        initial_capital=100000
    )
    
    sim_env = OptionsTradingEnvironment(
        initial_capital=100000
    )
    
    logger.info("\n=== DEBUGGING TRAINING ISSUES ===\n")
    
    # 1. Check data quality
    logger.info("1. DATA QUALITY CHECK:")
    if 'SPY' in data:
        df = data['SPY']
        logger.info(f"   - Total records: {len(df)}")
        logger.info(f"   - Unique timestamps: {df['timestamp'].nunique()}")
        logger.info(f"   - Price range: ${df['underlying_price'].min():.2f} - ${df['underlying_price'].max():.2f}")
        logger.info(f"   - Average option price: ${df['last'].mean():.2f}")
    
    # 2. Test simulated environment rewards
    logger.info("\n2. SIMULATED ENVIRONMENT TEST:")
    sim_env.reset()
    total_reward = 0
    for i in range(10):
        action = i % 11  # Try different actions
        obs, reward, done, info = sim_env.step(action)
        total_reward += reward
        if reward != 0:
            logger.info(f"   Step {i}: Action {action} -> Reward {reward:.2f}, Portfolio: ${info['portfolio_value']:.2f}")
    logger.info(f"   Total reward from 10 steps: {total_reward:.2f}")
    
    # 3. Check position sizes
    logger.info("\n3. POSITION SIZE CHECK:")
    sim_env.reset()
    sim_env.step(1)  # Buy call
    if sim_env.positions:
        pos = sim_env.positions[0]
        logger.info(f"   - Position size: {pos.quantity} contracts")
        logger.info(f"   - Position value: ${pos.current_value:.2f}")
        logger.info(f"   - As % of capital: {pos.current_value/100000*100:.2f}%")
    
    # 4. Test option price movements
    logger.info("\n4. OPTION PRICE MOVEMENT TEST:")
    sim_env.reset()
    sim_env.step(1)  # Buy call
    initial_value = sim_env.positions[0].current_value if sim_env.positions else 0
    
    # Step 10 times and track P&L
    for i in range(10):
        obs, reward, done, info = sim_env.step(0)  # Hold
        if sim_env.positions:
            current_value = sim_env.positions[0].current_value
            price_change = (current_value - initial_value) / initial_value * 100
            logger.info(f"   Step {i+1}: Value ${current_value:.2f} ({price_change:+.1f}%), Reward: {reward:.2f}")
    
    # 5. Calculate expected returns
    logger.info("\n5. EXPECTED RETURNS ANALYSIS:")
    logger.info(f"   - Current avg reward per episode: $15.80")
    logger.info(f"   - Current return per episode: 0.016%")
    logger.info(f"   - Target return per episode: 0.5-2%")
    logger.info(f"   - Required avg reward: $500-2000")
    logger.info(f"   - Reward multiplier needed: {500/15.8:.0f}x - {2000/15.8:.0f}x")

if __name__ == "__main__":
    asyncio.run(debug_issues())