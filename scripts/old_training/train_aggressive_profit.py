#!/usr/bin/env python3
"""Aggressive training strategy to force profitability"""

import os
import sys
import torch
import numpy as np
import logging
from tqdm import tqdm
from collections import deque

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.options_clstm_ppo import OptionsCLSTMPPOAgent
from src.options_trading_env import OptionsTradingEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProfitOnlyEnvironment(OptionsTradingEnvironment):
    """Environment that heavily rewards profits and punishes losses"""
    
    def _calculate_reward(self, trade_result):
        """Override reward to be extremely profit-focused"""
        portfolio_value = self._calculate_portfolio_value()
        
        if not hasattr(self, 'last_portfolio_value'):
            self.last_portfolio_value = self.initial_capital
        
        # Calculate P&L
        step_pnl = portfolio_value - self.last_portfolio_value
        self.last_portfolio_value = portfolio_value
        
        # AGGRESSIVE REWARD SHAPING
        if step_pnl > 0:
            # Massive reward for any profit
            reward = step_pnl / 10.0  # Scale down but keep positive
            
            # Extra bonus for larger profits
            profit_pct = step_pnl / self.initial_capital
            if profit_pct > 0.01:  # More than 1%
                reward *= 10
            if profit_pct > 0.02:  # More than 2%
                reward *= 10
        else:
            # Huge penalty for any loss
            reward = step_pnl / 1.0  # Full penalty
            
            # Extra penalty for large losses
            loss_pct = abs(step_pnl) / self.initial_capital
            if loss_pct > 0.01:  # More than 1% loss
                reward *= 10
        
        # Penalty for being underwater
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital
        if total_return < 0:
            reward -= abs(total_return) * 100  # Big penalty for negative total return
        
        return reward


def train_aggressive(num_episodes=3000):
    """Train with aggressive profit focus"""
    
    logger.info("AGGRESSIVE PROFIT TRAINING")
    logger.info("="*60)
    logger.info("Strategy:")
    logger.info("- Massive rewards for ANY profit")
    logger.info("- Huge penalties for ANY loss")
    logger.info("- Force close losing positions")
    logger.info("- Only trade when confident")
    logger.info("="*60)
    
    # Create profit-focused environment
    env = ProfitOnlyEnvironment(
        initial_capital=100000,
        max_positions=2,  # Even fewer positions
        commission=0.65
    )
    
    # Create agent with very conservative learning
    agent = OptionsCLSTMPPOAgent(
        observation_space=env.observation_space,
        action_space=11,
        learning_rate_actor_critic=5e-6,  # Ultra low learning rate
        learning_rate_clstm=1e-5,
        gamma=0.99,
        clip_epsilon=0.1,
        entropy_coef=0.001,  # Very low exploration
        batch_size=128,
        n_epochs=3  # Fewer epochs to prevent overfitting
    )
    
    # Metrics
    episode_returns = deque(maxlen=100)
    episode_win_rates = deque(maxlen=100)
    positive_episodes = 0
    total_episodes = 0
    
    # Best tracking
    best_avg_return = -10.0
    best_positive_rate = 0.0
    
    # Checkpoint directory
    checkpoint_dir = "checkpoints/aggressive_profit"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Action tracking
    action_counts = np.zeros(11)
    profitable_actions = np.zeros(11)
    
    pbar = tqdm(range(num_episodes), desc="Aggressive Training")
    
    for episode in pbar:
        obs = env.reset()
        episode_pnl = 0
        episode_actions = []
        trades_executed = 0
        
        for step in range(50):  # Shorter episodes
            # Get action
            action, info = agent.act(obs, deterministic=False)
            
            # Track action
            episode_actions.append(action)
            action_counts[action] += 1
            
            # Override bad actions when losing
            current_value = env._calculate_portfolio_value()
            if current_value < env.initial_capital:
                # If we have positions and we're losing, consider closing
                if len(env.positions) > 0:
                    if action != 10 and np.random.random() < 0.7:  # 70% chance
                        action = 10  # Force close
                # Don't open new positions when underwater
                elif action in [1, 2, 3, 4]:  # Buy/sell actions
                    action = 0  # Force hold
            
            # Step
            next_obs, reward, done, info = env.step(action)
            
            # Extra reward shaping
            if action in [1, 2, 3, 4]:
                trades_executed += 1
            
            # Store transition
            agent.store_transition(obs, action, reward, next_obs, done, info)
            
            obs = next_obs
            if done:
                break
        
        # Episode complete
        final_value = env._calculate_portfolio_value()
        episode_return = (final_value - env.initial_capital) / env.initial_capital
        episode_pnl = final_value - env.initial_capital
        
        # Track if profitable
        if episode_pnl > 0:
            positive_episodes += 1
            # Track which actions led to profit
            for act in episode_actions:
                profitable_actions[act] += 1
        
        total_episodes += 1
        
        # Win rate
        total_trades = env.winning_trades + env.losing_trades
        win_rate = env.winning_trades / max(1, total_trades) if total_trades > 0 else 0
        
        episode_returns.append(episode_return)
        episode_win_rates.append(win_rate)
        
        # Train
        if len(agent.buffer) >= agent.batch_size:
            agent.train()
        
        # Metrics
        avg_return = np.mean(episode_returns)
        avg_win_rate = np.mean(episode_win_rates)
        positive_rate = positive_episodes / total_episodes
        
        pbar.set_postfix({
            'Avg': f'{avg_return:.2%}',
            'WR': f'{avg_win_rate:.2%}',
            'Positive': f'{positive_rate:.2%}',
            'Last': f'{episode_return:.2%}'
        })
        
        # Save progress
        if episode % 50 == 0 and episode > 0:
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"ep{episode}.pt")
            agent.save(checkpoint_path)
            
            # Save best models
            if avg_return > best_avg_return:
                best_avg_return = avg_return
                agent.save(os.path.join(checkpoint_dir, "best_return.pt"))
                logger.info(f"\n🏆 New best avg return: {avg_return:.2%}")
            
            if positive_rate > best_positive_rate:
                best_positive_rate = positive_rate
                agent.save(os.path.join(checkpoint_dir, "best_positive_rate.pt"))
                logger.info(f"\n📈 New best positive rate: {positive_rate:.2%}")
            
            # Log action analysis
            if episode % 200 == 0:
                logger.info(f"\nAction Analysis (Episode {episode}):")
                for i in range(11):
                    if action_counts[i] > 0:
                        profit_rate = profitable_actions[i] / action_counts[i]
                        logger.info(f"  Action {i}: {action_counts[i]} times, {profit_rate:.2%} profitable")
        
        # Early stopping if we achieve profitability
        if episode > 500 and avg_return > 0.02 and positive_rate > 0.6:
            logger.info(f"\n🎯 TARGET ACHIEVED! Stopping early.")
            agent.save(os.path.join(checkpoint_dir, "target_achieved.pt"))
            break
    
    # Final save
    agent.save(os.path.join(checkpoint_dir, "final_model.pt"))
    
    # Results
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Episodes trained: {episode + 1}")
    logger.info(f"Final avg return: {np.mean(list(episode_returns)):.2%}")
    logger.info(f"Final win rate: {np.mean(list(episode_win_rates)):.2%}")
    logger.info(f"Positive episode rate: {positive_rate:.2%}")
    logger.info(f"Best avg return: {best_avg_return:.2%}")
    logger.info(f"Best positive rate: {best_positive_rate:.2%}")
    logger.info(f"\nModels saved in: {checkpoint_dir}/")
    
    # Final action analysis
    logger.info("\nFinal Action Analysis:")
    action_names = ['hold', 'buy_call', 'buy_put', 'sell_call', 'sell_put', 
                    'bull_spread', 'bear_spread', 'iron_condor', 'straddle', 
                    'strangle', 'close_all']
    for i in range(11):
        if action_counts[i] > 0:
            profit_rate = profitable_actions[i] / action_counts[i]
            logger.info(f"  {action_names[i]}: {profit_rate:.2%} profitable ({int(action_counts[i])} times)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=3000)
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        torch.backends.cuda.matmul.allow_tf32 = True
    
    train_aggressive(num_episodes=args.episodes)