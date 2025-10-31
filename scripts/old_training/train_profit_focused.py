#!/usr/bin/env python3
"""Simplified profit-focused training that actually works"""

import torch
import numpy as np
import logging
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.options_clstm_ppo import OptionsCLSTMPPOAgent
from src.options_trading_env import OptionsTradingEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_for_profit(num_episodes=300):
    """Train the bot to actually make money"""
    
    # Create environment
    env = OptionsTradingEnvironment(
        initial_capital=100000,
        max_positions=5,
        commission=0.65
    )
    
    # Create agent with conservative settings
    agent = OptionsCLSTMPPOAgent(
        observation_space=env.observation_space,
        action_space=11,
        learning_rate_actor_critic=5e-5,  # Very low learning rate
        learning_rate_clstm=1e-4,
        gamma=0.99,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        batch_size=64,
        n_epochs=5  # Fewer epochs to prevent overfitting
    )
    
    # Metrics
    all_profits = []
    all_win_rates = []
    best_avg_profit = -float('inf')
    
    logger.info("Starting profit-focused training...")
    logger.info("Goal: Achieve consistent positive returns")
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        obs = env.reset()
        episode_return = 0
        initial_value = env._calculate_portfolio_value()
        
        # Run episode
        for step in range(100):  # 100 steps per episode
            # Get action
            action, info = agent.act(obs, deterministic=False)
            
            # Step environment
            next_obs, reward, done, env_info = env.step(action)
            
            # Calculate actual profit
            current_value = env._calculate_portfolio_value()
            actual_profit = current_value - initial_value
            profit_pct = actual_profit / initial_value
            
            # Create profit-focused reward
            # Simple: positive for profit, negative for loss
            profit_reward = actual_profit / 1000  # Scale down
            
            # Add bonuses for good behavior
            if len(env.closed_positions) > 0:
                last_trade = env.closed_positions[-1]
                if last_trade['pnl'] > 0:
                    profit_reward += 2  # Bonus for profitable trades
            
            # Penalize holding when losing
            if actual_profit < -1000 and action == 0:  # Losing $1k and holding
                profit_reward -= 5
            
            # Store with modified reward
            agent.store_transition(obs, action, profit_reward, next_obs, done, info)
            
            obs = next_obs
            episode_return += reward
            
            if done:
                break
        
        # Calculate episode metrics
        final_value = env._calculate_portfolio_value()
        episode_profit = final_value - initial_value
        episode_profit_pct = episode_profit / initial_value
        
        # Calculate win rate
        total_trades = len(env.closed_positions)
        winning_trades = sum(1 for p in env.closed_positions if p['pnl'] > 0)
        win_rate = winning_trades / max(1, total_trades)
        
        all_profits.append(episode_profit_pct)
        all_win_rates.append(win_rate)
        
        # Train agent
        if len(agent.buffer) >= agent.batch_size:
            agent.train()
        
        # Log progress every 10 episodes
        if episode % 10 == 0:
            recent_profits = all_profits[-20:] if len(all_profits) >= 20 else all_profits
            recent_win_rates = all_win_rates[-20:] if len(all_win_rates) >= 20 else all_win_rates
            
            avg_profit = np.mean(recent_profits)
            avg_win_rate = np.mean(recent_win_rates)
            
            logger.info(f"\nEpisode {episode}")
            logger.info(f"  Recent avg profit: {avg_profit:.2%}")
            logger.info(f"  Recent win rate: {avg_win_rate:.2%}")
            logger.info(f"  Last episode P&L: ${episode_profit:,.2f} ({episode_profit_pct:.2%})")
            
            # Save if improving
            if avg_profit > best_avg_profit and avg_profit > 0:
                best_avg_profit = avg_profit
                os.makedirs("checkpoints", exist_ok=True)
                agent.save(f"checkpoints/profitable_{avg_profit:.4f}.pt")
                logger.info(f"  💰 New best model saved! Avg profit: {avg_profit:.2%}")
        
        # Early stopping if we achieve good performance
        if episode > 50:
            recent_50 = all_profits[-50:]
            if np.mean(recent_50) > 0.01 and np.std(recent_50) < 0.02:
                logger.info("\n🎉 Achieved stable positive returns!")
                agent.save("checkpoints/final_profitable_model.pt")
                break
    
    # Final results
    logger.info("\n" + "="*50)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Final 100-episode average profit: {np.mean(all_profits[-100:]):.2%}")
    logger.info(f"Final 100-episode win rate: {np.mean(all_win_rates[-100:]):.2%}")
    logger.info(f"Best average profit achieved: {best_avg_profit:.2%}")
    
    return agent


def test_profitability(agent, num_episodes=10):
    """Test the trained agent's profitability"""
    
    env = OptionsTradingEnvironment(initial_capital=100000)
    
    test_profits = []
    
    logger.info("\nTesting profitability...")
    
    for episode in range(num_episodes):
        obs = env.reset()
        initial_value = env._calculate_portfolio_value()
        
        for _ in range(100):
            action, _ = agent.act(obs, deterministic=True)  # Use deterministic actions
            obs, _, done, _ = env.step(action)
            if done:
                break
        
        final_value = env._calculate_portfolio_value()
        profit = final_value - initial_value
        profit_pct = profit / initial_value
        test_profits.append(profit_pct)
        
        logger.info(f"Test episode {episode + 1}: ${profit:,.2f} ({profit_pct:.2%})")
    
    logger.info(f"\nTest average profit: {np.mean(test_profits):.2%}")
    logger.info(f"Profitable episodes: {sum(1 for p in test_profits if p > 0)}/{num_episodes}")


if __name__ == "__main__":
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        # Enable mixed precision for faster training
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        logger.info("Using CPU (training will be slower)")
    
    # Train the agent
    agent = train_for_profit(num_episodes=300)
    
    # Test profitability
    test_profitability(agent, num_episodes=10)