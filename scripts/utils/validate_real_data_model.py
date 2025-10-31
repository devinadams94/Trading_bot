#!/usr/bin/env python3
"""
Validate a model trained on real options data
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime, timedelta
import asyncio
import logging
from dotenv import load_dotenv

from src.historical_options_data import HistoricalOptionsDataLoader, HistoricalOptionsEnvironment
from src.options_clstm_ppo import OptionsCLSTMPPOAgent

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def validate_model(model_path: str = 'checkpoints/options_real_data/real_data_final.pt'):
    """Validate a trained model on recent options data"""
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return
    
    # Load API credentials
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not api_secret:
        logger.error("Alpaca API credentials not found in environment")
        return
    
    # Create data loader
    data_loader = HistoricalOptionsDataLoader(
        api_key=api_key,
        api_secret=api_secret
    )
    
    # Load recent data for validation
    logger.info("Loading recent options data for validation...")
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=7)
    
    historical_data = await data_loader.load_historical_options_data(
        symbols=['SPY'],
        start_date=start_date,
        end_date=end_date,
        use_cache=True
    )
    
    if not historical_data or 'SPY' not in historical_data:
        logger.error("Failed to load validation data")
        return
    
    # Create environment
    env = HistoricalOptionsEnvironment(
        historical_data=historical_data,
        initial_capital=100000,
        commission=0.65
    )
    
    # Initialize agent
    agent = OptionsCLSTMPPOAgent(
        observation_space=env.observation_space,
        action_space=11
    )
    
    # Load trained model
    logger.info(f"Loading model from {model_path}")
    agent.load(model_path)
    
    # Run validation episodes
    logger.info("Running validation episodes...")
    total_rewards = []
    actions_taken = []
    
    for episode in range(5):
        obs = env.reset()
        episode_reward = 0
        episode_actions = []
        done = False
        steps = 0
        
        while not done and steps < 100:
            # Get action from trained model
            action, _ = agent.act(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_actions.append(action)
            steps += 1
        
        total_rewards.append(episode_reward)
        actions_taken.extend(episode_actions)
        
        logger.info(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")
    
    # Print validation summary
    logger.info("\n=== Validation Summary ===")
    logger.info(f"Average reward: {np.mean(total_rewards):.2f}")
    logger.info(f"Reward std: {np.std(total_rewards):.2f}")
    logger.info(f"Action distribution: {np.bincount(actions_taken, minlength=11)}")
    
    # Check if model is making diverse decisions
    unique_actions = len(set(actions_taken))
    if unique_actions < 3:
        logger.warning("Model is using very few action types - may need more training")
    else:
        logger.info(f"Model is using {unique_actions} different action types - good diversity")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Validate trained options model')
    parser.add_argument('--model-path', type=str, 
                       default='checkpoints/options_real_data/real_data_final.pt',
                       help='Path to trained model')
    
    args = parser.parse_args()
    
    asyncio.run(validate_model(args.model_path))


if __name__ == '__main__':
    main()