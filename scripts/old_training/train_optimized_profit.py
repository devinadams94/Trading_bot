#!/usr/bin/env python3
"""Optimized training script focused on profitability"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import logging
from tqdm import tqdm
import asyncio
from torch.utils.data import DataLoader, Dataset
import torch.cuda.amp as amp

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.options_clstm_ppo import OptionsCLSTMPPOAgent
from src.options_trading_env import OptionsTradingEnvironment
from src.historical_options_data import HistoricalOptionsEnvironment
from config.config_loader import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizedTrainer:
    def __init__(self, config_path: str = 'config/config_real_data.yaml'):
        self.config = load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Enable mixed precision training for faster GPU performance
        self.scaler = amp.GradScaler()
        
        # Training parameters optimized for profit
        self.num_episodes = self.config.get('num_episodes', 500)  # Reduced to prevent overfitting
        self.episode_length = self.config.get('episode_length', 100)
        self.batch_size = 128  # Increased batch size for better GPU utilization
        self.num_workers = 4  # Parallel data loading
        
        # Profit-focused parameters
        self.min_profit_threshold = 0.02  # 2% profit target per episode
        self.max_loss_threshold = -0.01  # 1% max loss tolerance
        
        logger.info(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    
    async def train(self):
        """Main training loop optimized for profitability"""
        
        # Create environment with profit-focused settings
        env = OptionsTradingEnvironment(
            initial_capital=100000,
            max_positions=3,  # Limit positions to reduce risk
            commission=0.65
        )
        
        # Initialize agent with optimized settings
        agent = OptionsCLSTMPPOAgent(
            observation_space=env.observation_space,
            action_space=11,
            learning_rate_actor_critic=1e-4,  # Lower LR for stability
            learning_rate_clstm=5e-4,
            gamma=0.95,  # Lower gamma for shorter-term focus
            clip_epsilon=0.1,  # Tighter clipping for stability
            entropy_coef=0.001,  # Lower entropy for more deterministic actions
            batch_size=self.batch_size
        )
        
        # Training metrics
        episode_profits = []
        win_rates = []
        best_profit = -float('inf')
        consecutive_losses = 0
        
        # Progress bar
        pbar = tqdm(range(self.num_episodes), desc="Training for Profit")
        
        for episode in pbar:
            obs = env.reset()
            episode_pnl = 0
            trades_won = 0
            trades_total = 0
            
            for step in range(self.episode_length):
                # Get action with profit bias
                action, info = agent.act(obs, deterministic=False)
                
                # Bias towards profitable actions based on current P&L
                if episode_pnl < 0 and action == 0:  # If losing, don't just hold
                    action = np.random.choice([10, 1, 2])  # Close or open new position
                
                # Execute action
                next_obs, reward, done, env_info = env.step(action)
                
                # Track P&L
                if 'total_pnl' in env_info:
                    episode_pnl = env_info['total_pnl']
                
                # Count trades
                if len(env.closed_positions) > trades_total:
                    trades_total = len(env.closed_positions)
                    if env.closed_positions[-1]['pnl'] > 0:
                        trades_won += 1
                
                # Store transition with modified reward
                profit_reward = reward
                if episode_pnl > self.min_profit_threshold * env.initial_capital:
                    profit_reward += 10  # Big bonus for reaching profit target
                elif episode_pnl < self.max_loss_threshold * env.initial_capital:
                    profit_reward -= 20  # Big penalty for excessive loss
                
                agent.store_transition(obs, action, profit_reward, next_obs, done, info)
                
                obs = next_obs
                if done:
                    break
            
            # Calculate episode metrics
            episode_profit_pct = episode_pnl / env.initial_capital
            episode_profits.append(episode_profit_pct)
            win_rate = trades_won / max(1, trades_total)
            win_rates.append(win_rate)
            
            # Train agent every episode for faster learning
            if len(agent.buffer) >= agent.batch_size:
                with amp.autocast():
                    train_metrics = agent.train()
            
            # Update progress bar
            avg_profit = np.mean(episode_profits[-50:]) if len(episode_profits) > 50 else episode_profit_pct
            avg_win_rate = np.mean(win_rates[-50:]) if len(win_rates) > 50 else win_rate
            
            pbar.set_postfix({
                'Avg Profit': f'{avg_profit:.2%}',
                'Win Rate': f'{avg_win_rate:.2%}',
                'Episode P&L': f'{episode_profit_pct:.2%}'
            })
            
            # Early stopping for consistent profits
            if avg_profit > 0.02 and avg_win_rate > 0.6:  # 2% profit with 60% win rate
                logger.info(f"Target profitability reached! Avg profit: {avg_profit:.2%}, Win rate: {avg_win_rate:.2%}")
                best_model_path = "checkpoints/profitable_model.pt"
                agent.save(best_model_path)
                break
            
            # Stop if losing consistently
            if episode_profit_pct < 0:
                consecutive_losses += 1
                if consecutive_losses > 30:
                    logger.warning("Too many consecutive losses. Adjusting strategy...")
                    # Reset some parameters
                    agent.clip_epsilon *= 0.9  # Reduce exploration
                    consecutive_losses = 0
            else:
                consecutive_losses = 0
            
            # Save best profitable model
            if avg_profit > best_profit and avg_profit > 0:
                best_profit = avg_profit
                agent.save(f"checkpoints/best_profit_{avg_profit:.4f}.pt")
            
            # Checkpoint every 50 episodes
            if episode % 50 == 0 and episode > 0:
                agent.save(f"checkpoints/checkpoint_ep{episode}.pt")
        
        logger.info(f"Training complete. Final avg profit: {np.mean(episode_profits[-100:]):.2%}")
        logger.info(f"Final win rate: {np.mean(win_rates[-100:]):.2%}")


class FastExperienceReplay(Dataset):
    """Fast experience replay buffer for GPU training"""
    
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def __getitem__(self, idx):
        return self.buffer[idx]
    
    def __len__(self):
        return len(self.buffer)


def optimize_gpu_settings():
    """Optimize GPU settings for faster training"""
    if torch.cuda.is_available():
        # Enable TF32 on Ampere GPUs for faster training
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cudNN autotuner
        torch.backends.cudnn.benchmark = True
        
        # Set optimal number of threads
        torch.set_num_threads(min(8, os.cpu_count()))
        
        logger.info("GPU optimizations enabled")


if __name__ == "__main__":
    # Optimize GPU settings
    optimize_gpu_settings()
    
    # Create trainer and run
    trainer = OptimizedTrainer()
    asyncio.run(trainer.train())