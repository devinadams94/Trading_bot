#!/usr/bin/env python3
"""
Quick training script for Algorithm 2 that uses simulated data
This allows training without waiting for historical data downloads
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_ppo_lstm import PPOLSTMTrainer
from src.options_trading_env import OptionsTradingEnvironment
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Quick PPO-LSTM training with simulated data')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to train (default: 100)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use: cuda or cpu (default: cpu)')
    parser.add_argument('--update-interval', type=int, default=64,
                        help='Update interval T (default: 64)')
    
    args = parser.parse_args()
    
    logger.info("Starting PPO-LSTM training with simulated data")
    
    # Create simulated environment
    env = OptionsTradingEnvironment(
        initial_capital=100000,
        commission=0.65,
        max_positions=5
    )
    
    # Create trainer
    trainer = PPOLSTMTrainer(
        env=env,
        learning_rate_actor=3e-4,
        learning_rate_critic=1e-3,
        gamma=0.99,
        epsilon=0.2,
        update_interval=args.update_interval,
        device=args.device
    )
    
    # Train
    logger.info(f"Training for {args.episodes} episodes...")
    trainer.train(num_episodes=args.episodes, checkpoint_interval=50)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()