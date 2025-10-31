#!/usr/bin/env python3
"""Simplified training script for options trading bot"""

import os
import sys
import torch
import numpy as np
from loguru import logger

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.options_trading_env import OptionsTradingEnvironment
from src.options_data_collector import OptionsDataSimulator
from src.options_clstm_ppo import OptionsCLSTMPPOAgent

def train_simple(episodes=100):
    """Simple training without pre-training complications"""
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        logger.warning("No GPU detected - training will be slower on CPU")
    
    # Initialize environment
    env = OptionsTradingEnvironment(
        initial_capital=100000,
        commission=0.65
    )
    
    # Initialize agent (will auto-detect GPU)
    agent = OptionsCLSTMPPOAgent(
        observation_space=env.observation_space,
        action_space=env.action_space.n,
        device=None  # Auto-detect best device
    )
    
    # Data simulator
    simulator = OptionsDataSimulator()
    
    # Training loop
    logger.info(f"Starting simple training for {episodes} episodes")
    
    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        # Get a random symbol
        symbol = np.random.choice(['SPY', 'QQQ', 'TSLA', 'NVDA', 'AMD'])
        stock_price = np.random.uniform(100, 500)
        
        while not done and steps < 200:
            # Generate options chain
            options_chain = simulator.simulate_options_chain(
                symbol=symbol,
                stock_price=stock_price,
                num_strikes=20,
                num_expirations=4
            )
            
            # Update observation with options chain
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
            action, act_info = agent.act(obs)
            
            # Step environment
            next_obs, reward, done, env_info = env.step(action)
            
            # Store experience
            agent.store_transition(obs, action, reward, next_obs, done, act_info)
            
            # Update state
            obs = next_obs
            episode_reward += reward
            steps += 1
            
            # Simulate price movement
            stock_price *= np.random.uniform(0.99, 1.01)
        
        # Train at end of episode
        if len(agent.buffer) >= agent.batch_size:
            train_metrics = agent.train()
            
            if episode % 10 == 0:
                logger.info(f"Episode {episode}: Reward={episode_reward:.2f}, Steps={steps}")
                if train_metrics:
                    logger.info(f"  Loss={train_metrics.get('total_loss', 0):.4f}")
        
        # Save checkpoint periodically
        if (episode + 1) % 50 == 0:
            checkpoint_dir = "checkpoints/options_clstm_ppo"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"clstm_ppo_episode_{episode+1}.pt")
            agent.save(checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join("checkpoints/options_clstm_ppo", "clstm_ppo_final.pt")
    agent.save(final_path)
    logger.info(f"Training complete! Final model saved: {final_path}")
    
    return final_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    args = parser.parse_args()
    
    train_simple(args.episodes)