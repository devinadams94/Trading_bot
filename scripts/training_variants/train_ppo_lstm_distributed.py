#!/usr/bin/env python3
"""
Distributed implementation of Algorithm 2: PPO with LSTM
Supports multi-GPU training with data parallelism and distributed environments
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import os
import sys
from dotenv import load_dotenv
import argparse
import asyncio
from datetime import datetime, timedelta
import json
import socket
import time

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.options_clstm_ppo import OptionsCLSTMPPONetwork
from src.historical_options_data import HistoricalOptionsDataLoader, HistoricalOptionsEnvironment
from src.options_trading_env import OptionsTradingEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_distributed(rank: int, world_size: int, master_port: int = 12355):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    
    # Set NCCL environment variables for better stability
    os.environ['NCCL_TIMEOUT'] = '3600'
    os.environ['NCCL_BLOCKING_WAIT'] = '1'
    os.environ['NCCL_DEBUG'] = 'INFO'
    
    # Initialize the process group
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://localhost:{master_port}',
        world_size=world_size,
        rank=rank
    )
    
    # Set device
    torch.cuda.set_device(rank)
    

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()


class DistributedRolloutBuffer:
    """Distributed rollout buffer that syncs across GPUs"""
    
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.local_buffer = {
            'features': [],
            'actions': [],
            'advantages': [],
            'returns': [],
            'old_log_probs': []
        }
        
    def add(self, features, action, advantage, returns, old_log_prob):
        """Add transition to local buffer"""
        self.local_buffer['features'].append(features)
        self.local_buffer['actions'].append(action)
        self.local_buffer['advantages'].append(advantage)
        self.local_buffer['returns'].append(returns)
        self.local_buffer['old_log_probs'].append(old_log_prob)
        
    def get_all_gathered(self, device):
        """Gather buffers from all GPUs"""
        gathered_buffers = {key: [] for key in self.local_buffer.keys()}
        
        for key in self.local_buffer.keys():
            if not self.local_buffer[key]:
                continue
                
            if key == 'features':
                # Handle tensor features
                local_tensor = torch.stack(self.local_buffer[key])
                gathered = [torch.zeros_like(local_tensor) for _ in range(self.world_size)]
                dist.all_gather(gathered, local_tensor)
                gathered_buffers[key] = torch.cat(gathered, dim=0)
            else:
                # Handle scalar values
                local_tensor = torch.tensor(self.local_buffer[key], device=device)
                gathered = [torch.zeros_like(local_tensor) for _ in range(self.world_size)]
                dist.all_gather(gathered, local_tensor)
                gathered_buffers[key] = torch.cat(gathered, dim=0)
                
        return gathered_buffers
    
    def clear(self):
        """Clear local buffer"""
        for key in self.local_buffer:
            self.local_buffer[key].clear()


class DistributedPPOLSTMTrainer:
    """Distributed PPO with LSTM trainer for multi-GPU training"""
    
    def __init__(
        self,
        env,
        rank: int,
        world_size: int,
        learning_rate_actor: float = 3e-4,
        learning_rate_critic: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        update_interval: int = 128,
        checkpoint_dir: str = 'checkpoints/ppo_lstm_distributed'
    ):
        self.env = env
        self.rank = rank
        self.world_size = world_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_interval = update_interval // world_size  # Divide by number of GPUs
        self.device = torch.device(f'cuda:{rank}')
        self.checkpoint_dir = checkpoint_dir
        
        # Create model and move to GPU
        self.network = OptionsCLSTMPPONetwork(
            observation_space=env.observation_space,
            action_dim=11
        ).to(self.device)
        
        # Wrap model with DDP
        self.network = DDP(
            self.network, 
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True
        )
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(
            self.network.module.actor.parameters(), 
            lr=learning_rate_actor
        )
        self.critic_optimizer = optim.Adam(
            self.network.module.critic.parameters(), 
            lr=learning_rate_critic
        )
        
        # Distributed rollout buffer
        self.replay_buffer = DistributedRolloutBuffer(rank, world_size)
        
        self.t = 0  # Local step counter
        self.global_t = 0  # Global step counter
        self.start_episode = 0
        
        # Best model tracking (only on rank 0)
        if rank == 0:
            self.best_avg_return = float('-inf')
            self.best_win_rate = 0.0
            self.best_combined_score = float('-inf')
            self.episode_returns = []
            self.episode_win_rates = []
            
    def sync_networks(self):
        """Synchronize network parameters across all GPUs"""
        for param in self.network.parameters():
            dist.broadcast(param.data, src=0)
            
    def process_state_with_lstm(self, state: Dict[str, np.ndarray]) -> torch.Tensor:
        """Process state with LSTM to obtain feature vector"""
        if state is None:
            return torch.zeros(1, self.network.module.clstm_encoder.hidden_dim).to(self.device)
            
        features = []
        for key in ['price_history', 'technical_indicators', 'options_chain', 
                   'portfolio_state', 'greeks_summary']:
            if key in state:
                tensor = torch.tensor(state[key], dtype=torch.float32).to(self.device)
                features.append(tensor.flatten())
        
        if not features:
            return torch.zeros(1, self.network.module.clstm_encoder.hidden_dim).to(self.device)
            
        combined = torch.cat(features, dim=0).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            ft = self.network.module.clstm_encoder(combined)
            
        return ft.squeeze(0)
    
    def train_episode(self):
        """Train one episode on this GPU"""
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]
            
        episode_reward = 0
        episode_steps = 0
        episode_return = 0
        winning_trades = 0
        losing_trades = 0
        total_trades = 0
        
        while True:
            if state is None:
                logger.warning(f"GPU {self.rank}: Received None state, ending episode")
                break
                
            # Process state
            ft = self.process_state_with_lstm(state)
            
            # Get value estimate
            with torch.no_grad():
                vt = self.network.module.critic(ft.unsqueeze(0)).item()
            
            # Sample action
            with torch.no_grad():
                action_logits = self.network.module.actor(ft.unsqueeze(0))
                dist = Categorical(logits=action_logits)
                at = dist.sample()
                log_prob = dist.log_prob(at).item()
            
            # Execute action
            step_result = self.env.step(at.item())
            if len(step_result) == 4:
                next_state, rt, done, info = step_result
                truncated = False
            else:
                next_state, rt, done, truncated, info = step_result
            episode_reward += rt
            
            # Track metrics
            if info:
                if 'episode_return' in info:
                    episode_return = info['episode_return']
                elif 'portfolio_value' in info and hasattr(self.env, 'initial_capital'):
                    episode_return = info['portfolio_value'] - self.env.initial_capital
                elif 'capital' in info and hasattr(self.env, 'initial_capital'):
                    episode_return = info['capital'] - self.env.initial_capital
                
                if 'trade_result' in info and isinstance(info['trade_result'], dict):
                    trade_result = info['trade_result']
                    if trade_result.get('success') and 'close' in str(trade_result.get('message', '')).lower():
                        total_trades += 1
                        if 'pnl' in trade_result:
                            if trade_result['pnl'] > 0:
                                winning_trades += 1
                            else:
                                losing_trades += 1
                
                if hasattr(self.env, 'winning_trades'):
                    winning_trades = self.env.winning_trades
                if hasattr(self.env, 'losing_trades'):
                    losing_trades = self.env.losing_trades
                    total_trades = winning_trades + losing_trades
                    
                open_positions = 0
                if 'positions' in info:
                    if isinstance(info['positions'], int):
                        open_positions = info['positions']
                    elif isinstance(info['positions'], list):
                        open_positions = len(info['positions'])
            
            # Get next value
            if not done and not truncated and next_state is not None:
                ft_next = self.process_state_with_lstm(next_state)
                with torch.no_grad():
                    vt_next = self.network.module.critic(ft_next.unsqueeze(0)).item()
            else:
                vt_next = 0
            
            # Compute advantage
            At = rt + self.gamma * vt_next - vt
            
            # Add to buffer
            self.replay_buffer.add(ft, at.item(), At, rt + self.gamma * vt_next, log_prob)
            
            # Update networks when buffer is full
            self.t += 1
            if self.t % self.update_interval == 0:
                self.update_networks()
                self.replay_buffer.clear()
            
            state = next_state
            episode_steps += 1
            
            if done or truncated:
                if hasattr(self.env, 'capital') and hasattr(self.env, 'initial_capital'):
                    episode_return = self.env.capital - self.env.initial_capital
                elif hasattr(self.env, 'get_portfolio_value'):
                    episode_return = self.env.get_portfolio_value() - self.env.initial_capital
                break
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        open_positions = len(self.env.positions) if hasattr(self.env, 'positions') else 0
        
        return episode_reward, episode_steps, episode_return, win_rate, total_trades, open_positions
    
    def update_networks(self):
        """Update networks using gathered data from all GPUs"""
        # Gather buffers from all GPUs
        gathered_buffers = self.replay_buffer.get_all_gathered(self.device)
        
        if not gathered_buffers['features']:
            return
            
        # Prepare tensors
        features = gathered_buffers['features']
        actions = gathered_buffers['actions'].long()
        advantages = gathered_buffers['advantages'].float()
        returns = gathered_buffers['returns'].float()
        old_log_probs = gathered_buffers['old_log_probs'].float()
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update critic
        values = self.network.module.critic(features).squeeze()
        critic_loss = F.mse_loss(values, returns)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.module.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update actor
        action_logits = self.network.module.actor(features)
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.module.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        if self.rank == 0:
            logger.info(f"Updated networks - Critic Loss: {critic_loss:.4f}, Actor Loss: {actor_loss:.4f}")
    
    def train(self, num_episodes: int = 1000, checkpoint_interval: int = 100):
        """Main training loop"""
        episodes_per_gpu = num_episodes // self.world_size
        extra_episodes = num_episodes % self.world_size
        
        # Rank 0 gets any extra episodes
        if self.rank == 0:
            episodes_per_gpu += extra_episodes
            
        total_episodes = self.start_episode + num_episodes
        
        if self.rank == 0:
            logger.info(f"Training from episode {self.start_episode + 1} to {total_episodes}")
            logger.info(f"Episodes per GPU: {episodes_per_gpu} (GPU 0: {episodes_per_gpu})")
        
        for local_episode in range(episodes_per_gpu):
            # Calculate global episode number
            global_episode = self.start_episode + (local_episode * self.world_size) + self.rank + 1
            
            # Train episode
            episode_reward, steps, episode_return, win_rate, trades, open_positions = self.train_episode()
            
            # Gather metrics from all GPUs
            metrics = torch.tensor([episode_reward, episode_return, win_rate, trades, steps], device=self.device)
            gathered_metrics = [torch.zeros_like(metrics) for _ in range(self.world_size)]
            dist.all_gather(gathered_metrics, metrics)
            
            # Process and log on rank 0
            if self.rank == 0:
                # Average metrics across GPUs
                all_metrics = torch.stack(gathered_metrics).mean(dim=0)
                avg_reward = all_metrics[0].item()
                avg_return = all_metrics[1].item()
                avg_win_rate = all_metrics[2].item()
                avg_trades = int(all_metrics[3].item())
                avg_steps = int(all_metrics[4].item())
                
                # Store metrics
                self.episode_returns.append(avg_return)
                self.episode_win_rates.append(avg_win_rate)
                
                # Calculate rolling averages
                recent_returns = self.episode_returns[-100:] if len(self.episode_returns) > 100 else self.episode_returns
                recent_win_rates = self.episode_win_rates[-100:] if len(self.episode_win_rates) > 100 else self.episode_win_rates
                roll_avg_return = np.mean(recent_returns) if recent_returns else 0
                roll_avg_win_rate = np.mean(recent_win_rates) if recent_win_rates else 0
                
                # Display results
                if avg_trades == 0 and open_positions > 0:
                    logger.info(f"Episode {global_episode}/{total_episodes} - "
                               f"Reward: {avg_reward:.2f}, "
                               f"Return: ${avg_return:.2f} (unrealized from {open_positions} open positions), "
                               f"Win Rate: {avg_win_rate:.1%} ({avg_trades} closed trades), "
                               f"Steps: {avg_steps} | GPUs: {self.world_size}")
                else:
                    logger.info(f"Episode {global_episode}/{total_episodes} - "
                               f"Reward: {avg_reward:.2f}, "
                               f"Return: ${avg_return:.2f}, "
                               f"Win Rate: {avg_win_rate:.1%} ({avg_trades} closed, {open_positions} open), "
                               f"Steps: {avg_steps} | GPUs: {self.world_size}")
                
                # Check for best model
                if len(self.episode_returns) >= 10:
                    combined_score = 0.7 * roll_avg_return + 0.3 * (roll_avg_win_rate * 1000)
                    
                    if combined_score > self.best_combined_score:
                        self.best_combined_score = combined_score
                        self.best_avg_return = roll_avg_return
                        self.best_win_rate = roll_avg_win_rate
                        
                        best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                        self.save_checkpoint(best_path, episode=global_episode, is_best=True)
                        logger.info(f"  🏆 New best model! Avg Return: ${roll_avg_return:.2f}, Avg Win Rate: {roll_avg_win_rate:.1%}")
                
                # Display rolling averages
                if global_episode % 10 == 0:
                    logger.info(f"  → 100-Episode Averages - Return: ${roll_avg_return:.2f}, Win Rate: {roll_avg_win_rate:.1%}")
                    logger.info(f"  → Best so far - Return: ${self.best_avg_return:.2f}, Win Rate: {self.best_win_rate:.1%}")
                
                # Save checkpoints
                if global_episode % checkpoint_interval == 0:
                    checkpoint_path = os.path.join(self.checkpoint_dir, f'ep{global_episode}.pt')
                    self.save_checkpoint(checkpoint_path, episode=global_episode)
                
                # Always save latest
                latest_path = os.path.join(self.checkpoint_dir, 'latest.pt')
                self.save_checkpoint(latest_path, episode=global_episode)
            
            # Synchronize GPUs
            dist.barrier()
    
    def save_checkpoint(self, path: str, episode: int = None, is_best: bool = False):
        """Save model checkpoint (only on rank 0)"""
        if self.rank == 0:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            checkpoint_data = {
                'network_state_dict': self.network.module.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                't': self.global_t,
                'episode': episode if episode is not None else self.start_episode,
                'episode_returns': self.episode_returns,
                'episode_win_rates': self.episode_win_rates,
                'best_avg_return': self.best_avg_return,
                'best_win_rate': self.best_win_rate,
                'best_combined_score': self.best_combined_score,
                'world_size': self.world_size
            }
            torch.save(checkpoint_data, path)
            logger.info(f"{'📌 Best model saved' if is_best else 'Saved checkpoint'} to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        if os.path.exists(path):
            if self.rank == 0:
                logger.info(f"Loading checkpoint from {path}")
            
            # Handle PyTorch 2.6+ security changes
            try:
                checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            except:
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            
            self.network.module.load_state_dict(checkpoint['network_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            
            if self.rank == 0:
                self.global_t = checkpoint.get('t', 0)
                self.start_episode = checkpoint.get('episode', 0)
                self.episode_returns = checkpoint.get('episode_returns', [])
                self.episode_win_rates = checkpoint.get('episode_win_rates', [])
                self.best_avg_return = checkpoint.get('best_avg_return', float('-inf'))
                self.best_win_rate = checkpoint.get('best_win_rate', 0.0)
                self.best_combined_score = checkpoint.get('best_combined_score', float('-inf'))
                
                saved_world_size = checkpoint.get('world_size', 1)
                if saved_world_size != self.world_size:
                    logger.warning(f"Checkpoint was saved with {saved_world_size} GPUs, now using {self.world_size}")
                
                logger.info(f"✅ Resumed from episode {self.start_episode} (step {self.global_t})")
                if self.episode_returns:
                    logger.info(f"  Previous avg return: ${np.mean(self.episode_returns[-100:]):.2f}")
                if self.episode_win_rates:
                    logger.info(f"  Previous avg win rate: {np.mean(self.episode_win_rates[-100:]):.1%}")
            
            # Sync loaded state across GPUs
            self.sync_networks()
            dist.barrier()
            
            return True
        return False


def run_distributed_training(rank: int, world_size: int, args):
    """Run training on a specific GPU"""
    try:
        # Setup distributed training
        setup_distributed(rank, world_size, args.master_port)
        
        # Set up logging for this rank
        logging.basicConfig(
            level=logging.INFO,
            format=f'[GPU {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
        
        # Only rank 0 loads data and creates environments
        if rank == 0:
            logger.info("Loading data on primary GPU...")
            
            # Get API keys
            api_key = os.getenv('ALPACA_API_KEY')
            api_secret = os.getenv('ALPACA_SECRET_KEY')
            
            if not api_key or not api_secret:
                logger.error("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env file")
                sys.exit(1)
            
            # Create data loader
            data_loader = HistoricalOptionsDataLoader(
                api_key=api_key,
                api_secret=api_secret
            )
            
            # Determine date range
            if args.end_date:
                end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
            else:
                end_date = datetime.now() - timedelta(days=1)
                
            if args.start_date:
                start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
            else:
                start_date = end_date - timedelta(days=30)
            
            # Load historical data
            historical_data = {}
            if not args.no_data_load:
                logger.info("Loading historical options data...")
                
                # Import the async loading function
                from train_ppo_lstm import load_historical_data
                
                historical_data = asyncio.run(load_historical_data(
                    data_loader, args.symbols, start_date, end_date
                ))
                
                if not historical_data:
                    logger.warning("No historical data loaded. Will use simulated data.")
            else:
                logger.info("Skipping historical data load (--no-data-load specified)")
        
        # Synchronize all GPUs
        dist.barrier()
        
        # Create environment on each GPU
        if rank == 0 and (historical_data or not args.no_data_load):
            # Rank 0 uses historical environment
            env = HistoricalOptionsEnvironment(
                historical_data=historical_data,
                data_loader=data_loader,
                symbols=args.symbols,
                initial_capital=100000,
                max_positions=5,
                commission=0.65
            )
            logger.info(f"GPU {rank}: Using HistoricalOptionsEnvironment with {len(historical_data)} symbols")
        else:
            # Other ranks use simulated environment
            env = OptionsTradingEnvironment(
                initial_capital=100000,
                commission=0.65,
                max_positions=5
            )
            logger.info(f"GPU {rank}: Using simulated OptionsTradingEnvironment")
        
        # Create distributed trainer
        trainer = DistributedPPOLSTMTrainer(
            env=env,
            rank=rank,
            world_size=world_size,
            learning_rate_actor=args.learning_rate_actor,
            learning_rate_critic=args.learning_rate_critic,
            gamma=args.gamma,
            epsilon=args.epsilon,
            update_interval=args.update_interval,
            checkpoint_dir=args.checkpoint_dir
        )
        
        # Handle checkpoint resumption (only rank 0 handles this)
        if rank == 0:
            resumed = False
            
            # Find latest checkpoint
            from train_ppo_lstm import PPOLSTMTrainer
            temp_trainer = PPOLSTMTrainer(env, checkpoint_dir=args.checkpoint_dir)
            
            if not args.no_auto_resume or args.resume or args.resume_from:
                if args.resume_from:
                    resumed = trainer.load_checkpoint(args.resume_from)
                    if not resumed:
                        logger.error(f"Failed to load checkpoint from {args.resume_from}")
                        cleanup_distributed()
                        sys.exit(1)
                else:
                    latest_checkpoint = temp_trainer.find_latest_checkpoint()
                    if latest_checkpoint:
                        logger.info(f"Found checkpoint: {latest_checkpoint}")
                        resumed = trainer.load_checkpoint(latest_checkpoint)
                    elif args.resume:
                        logger.error("--resume specified but no checkpoint found")
                        cleanup_distributed()
                        sys.exit(1)
                    else:
                        logger.info("No checkpoint found. Starting fresh training.")
            
            # Log training parameters
            logger.info(f"Distributed Training Algorithm 2: PPO with LSTM on {world_size} GPUs")
            logger.info(f"Episodes: {args.episodes}")
            logger.info(f"Actor LR: {args.learning_rate_actor}")
            logger.info(f"Critic LR: {args.learning_rate_critic}")
            logger.info(f"Gamma: {args.gamma}")
            logger.info(f"Epsilon: {args.epsilon}")
            logger.info(f"Update Interval: {args.update_interval} (per GPU: {args.update_interval // world_size})")
            logger.info(f"Symbols: {args.symbols}")
            logger.info(f"Checkpoint Directory: {args.checkpoint_dir}")
            if resumed:
                logger.info(f"✨ Resumed training from episode {trainer.start_episode}")
        
        # Synchronize all GPUs before training
        dist.barrier()
        
        # Train
        trainer.train(num_episodes=args.episodes, checkpoint_interval=args.checkpoint_interval)
        
        # Cleanup
        cleanup_distributed()
        
    except Exception as e:
        logger.error(f"GPU {rank} encountered error: {e}")
        import traceback
        traceback.print_exc()
        cleanup_distributed()
        raise


def main():
    """Main function to run distributed training"""
    parser = argparse.ArgumentParser(description='Distributed PPO with LSTM training')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes to train (default: 1000)')
    parser.add_argument('--learning-rate-actor', type=float, default=3e-4,
                        help='Actor learning rate αθ (default: 3e-4)')
    parser.add_argument('--learning-rate-critic', type=float, default=1e-3,
                        help='Critic learning rate αV (default: 1e-3)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor γ (default: 0.99)')
    parser.add_argument('--epsilon', type=float, default=0.2,
                        help='PPO clipping range ε (default: 0.2)')
    parser.add_argument('--update-interval', type=int, default=128,
                        help='Update interval T (default: 128)')
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                        help='Save checkpoint every N episodes (default: 100)')
    parser.add_argument('--symbols', nargs='+', default=['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT'],
                        help='Symbols to trade (default: SPY QQQ IWM AAPL MSFT)')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date for historical data (YYYY-MM-DD). Default: 30 days ago')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date for historical data (YYYY-MM-DD). Default: yesterday')
    parser.add_argument('--no-data-load', action='store_true',
                        help='Skip historical data loading (use simulated data)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')
    parser.add_argument('--no-auto-resume', action='store_true',
                        help='Disable automatic resumption from latest checkpoint')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume training from specific checkpoint file')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/ppo_lstm_distributed',
                        help='Directory for saving checkpoints')
    parser.add_argument('--num-gpus', type=int, default=None,
                        help='Number of GPUs to use (default: all available)')
    parser.add_argument('--master-port', type=int, default=12355,
                        help='Master port for distributed training (default: 12355)')
    
    args = parser.parse_args()
    
    # Determine number of GPUs
    if args.num_gpus is None:
        world_size = torch.cuda.device_count()
    else:
        world_size = min(args.num_gpus, torch.cuda.device_count())
    
    if world_size < 1:
        print("No GPUs available. Please run on a machine with GPUs.")
        sys.exit(1)
    
    print(f"Starting distributed training on {world_size} GPUs...")
    
    # Spawn processes for distributed training
    mp.spawn(
        run_distributed_training,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()