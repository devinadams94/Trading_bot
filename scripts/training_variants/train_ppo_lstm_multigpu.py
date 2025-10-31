#!/usr/bin/env python3
"""
Simplified multi-GPU training for PPO with LSTM
Uses DataParallel instead of DistributedDataParallel for easier setup
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import logging
from typing import Dict, List, Tuple
import os
import sys
from dotenv import load_dotenv
import argparse
import asyncio
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.options_clstm_ppo import OptionsCLSTMPPONetwork
from src.historical_options_data import HistoricalOptionsDataLoader, HistoricalOptionsEnvironment
from src.options_trading_env import OptionsTradingEnvironment
from train_ppo_lstm import load_historical_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiGPUPPOLSTMTrainer:
    """Multi-GPU PPO with LSTM trainer using DataParallel"""
    
    def __init__(
        self,
        envs: List,  # List of environments (one per GPU)
        learning_rate_actor: float = 3e-4,
        learning_rate_critic: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        update_interval: int = 128,
        device_ids: List[int] = None,
        checkpoint_dir: str = 'checkpoints/ppo_lstm_multigpu'
    ):
        self.envs = envs
        self.num_envs = len(envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_interval = update_interval // self.num_envs  # Distribute across envs
        self.checkpoint_dir = checkpoint_dir
        
        # Set up devices
        if device_ids is None:
            self.device_ids = list(range(torch.cuda.device_count()))
        else:
            self.device_ids = device_ids
        
        self.primary_device = torch.device(f'cuda:{self.device_ids[0]}')
        
        logger.info(f"Using {len(self.device_ids)} GPUs: {self.device_ids}")
        
        # Create model on primary device
        self.network = OptionsCLSTMPPONetwork(
            observation_space=envs[0].observation_space,
            action_dim=11
        ).to(self.primary_device)
        
        # Use DataParallel if multiple GPUs
        if len(self.device_ids) > 1:
            self.network = nn.DataParallel(self.network, device_ids=self.device_ids)
            self.base_network = self.network.module
        else:
            self.base_network = self.network
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(
            self.base_network.actor.parameters(), 
            lr=learning_rate_actor
        )
        self.critic_optimizer = optim.Adam(
            self.base_network.critic.parameters(), 
            lr=learning_rate_critic
        )
        
        # Replay buffers (one per environment)
        self.replay_buffers = [
            {
                'features': [],
                'actions': [],
                'advantages': [],
                'returns': [],
                'old_log_probs': []
            } for _ in range(self.num_envs)
        ]
        
        self.t = 0
        self.start_episode = 0
        
        # Metrics tracking
        self.best_avg_return = float('-inf')
        self.best_win_rate = 0.0
        self.best_combined_score = float('-inf')
        self.episode_returns = []
        self.episode_win_rates = []
        
        # Thread pool for parallel environment stepping
        self.executor = ThreadPoolExecutor(max_workers=self.num_envs)
        
    def process_state_with_lstm(self, state: Dict[str, np.ndarray], device: torch.device) -> torch.Tensor:
        """Process state with LSTM to obtain feature vector"""
        if state is None:
            return torch.zeros(1, self.base_network.clstm_encoder.hidden_dim).to(device)
            
        features = []
        for key in ['price_history', 'technical_indicators', 'options_chain', 
                   'portfolio_state', 'greeks_summary']:
            if key in state:
                tensor = torch.tensor(state[key], dtype=torch.float32).to(device)
                features.append(tensor.flatten())
        
        if not features:
            return torch.zeros(1, self.base_network.clstm_encoder.hidden_dim).to(device)
            
        combined = torch.cat(features, dim=0).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            # Always process on primary device
            combined_primary = combined.to(self.primary_device)
            ft = self.base_network.clstm_encoder(combined_primary)
            
        return ft.squeeze(0)
    
    def train_episode_on_env(self, env_idx: int):
        """Train one episode on a specific environment"""
        env = self.envs[env_idx]
        device = torch.device(f'cuda:{self.device_ids[env_idx % len(self.device_ids)]}')
        
        state = env.reset()
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
                logger.warning(f"Env {env_idx}: Received None state, ending episode")
                break
                
            # Process state
            ft = self.process_state_with_lstm(state, device)
            
            # Move to primary device for model inference
            ft_primary = ft.to(self.primary_device)
            
            # Get value estimate
            with torch.no_grad():
                vt = self.base_network.critic(ft_primary.unsqueeze(0)).item()
            
            # Sample action
            with torch.no_grad():
                action_logits = self.base_network.actor(ft_primary.unsqueeze(0))
                dist = Categorical(logits=action_logits)
                at = dist.sample()
                log_prob = dist.log_prob(at).item()
            
            # Execute action
            step_result = env.step(at.item())
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
                elif 'portfolio_value' in info and hasattr(env, 'initial_capital'):
                    episode_return = info['portfolio_value'] - env.initial_capital
                elif 'capital' in info and hasattr(env, 'initial_capital'):
                    episode_return = info['capital'] - env.initial_capital
                
                if 'trade_result' in info and isinstance(info['trade_result'], dict):
                    trade_result = info['trade_result']
                    if trade_result.get('success') and 'close' in str(trade_result.get('message', '')).lower():
                        total_trades += 1
                        if 'pnl' in trade_result:
                            if trade_result['pnl'] > 0:
                                winning_trades += 1
                            else:
                                losing_trades += 1
                
                if hasattr(env, 'winning_trades'):
                    winning_trades = env.winning_trades
                if hasattr(env, 'losing_trades'):
                    losing_trades = env.losing_trades
                    total_trades = winning_trades + losing_trades
                    
                open_positions = 0
                if 'positions' in info:
                    if isinstance(info['positions'], int):
                        open_positions = info['positions']
                    elif isinstance(info['positions'], list):
                        open_positions = len(info['positions'])
            
            # Get next value
            if not done and not truncated and next_state is not None:
                ft_next = self.process_state_with_lstm(next_state, device)
                ft_next_primary = ft_next.to(self.primary_device)
                with torch.no_grad():
                    vt_next = self.base_network.critic(ft_next_primary.unsqueeze(0)).item()
            else:
                vt_next = 0
            
            # Compute advantage
            At = rt + self.gamma * vt_next - vt
            
            # Add to buffer (store on primary device)
            self.replay_buffers[env_idx]['features'].append(ft_primary)
            self.replay_buffers[env_idx]['actions'].append(at.item())
            self.replay_buffers[env_idx]['advantages'].append(At)
            self.replay_buffers[env_idx]['returns'].append(rt + self.gamma * vt_next)
            self.replay_buffers[env_idx]['old_log_probs'].append(log_prob)
            
            state = next_state
            episode_steps += 1
            
            if done or truncated:
                if hasattr(env, 'capital') and hasattr(env, 'initial_capital'):
                    episode_return = env.capital - env.initial_capital
                elif hasattr(env, 'get_portfolio_value'):
                    episode_return = env.get_portfolio_value() - env.initial_capital
                break
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        open_positions = len(env.positions) if hasattr(env, 'positions') else 0
        
        return episode_reward, episode_steps, episode_return, win_rate, total_trades, open_positions
    
    def train_episodes_parallel(self):
        """Train episodes in parallel across environments"""
        # Submit tasks to thread pool
        futures = []
        for env_idx in range(self.num_envs):
            future = self.executor.submit(self.train_episode_on_env, env_idx)
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            result = future.result()
            results.append(result)
            
        return results
    
    def update_networks(self):
        """Update networks using combined buffers from all environments"""
        # Combine all buffers
        all_features = []
        all_actions = []
        all_advantages = []
        all_returns = []
        all_old_log_probs = []
        
        for buffer in self.replay_buffers:
            if buffer['features']:
                all_features.extend(buffer['features'])
                all_actions.extend(buffer['actions'])
                all_advantages.extend(buffer['advantages'])
                all_returns.extend(buffer['returns'])
                all_old_log_probs.extend(buffer['old_log_probs'])
        
        if not all_features:
            return
            
        # Convert to tensors
        features = torch.stack(all_features)
        actions = torch.tensor(all_actions, dtype=torch.long, device=self.primary_device)
        advantages = torch.tensor(all_advantages, dtype=torch.float32, device=self.primary_device)
        returns = torch.tensor(all_returns, dtype=torch.float32, device=self.primary_device)
        old_log_probs = torch.tensor(all_old_log_probs, dtype=torch.float32, device=self.primary_device)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update critic
        values = self.base_network.critic(features).squeeze()
        critic_loss = F.mse_loss(values, returns)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.base_network.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update actor
        action_logits = self.base_network.actor(features)
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.base_network.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        logger.info(f"Updated networks - Critic Loss: {critic_loss:.4f}, Actor Loss: {actor_loss:.4f}")
        
        # Clear buffers
        for buffer in self.replay_buffers:
            for key in buffer:
                buffer[key].clear()
    
    def train(self, num_episodes: int = 1000, checkpoint_interval: int = 100):
        """Main training loop"""
        total_episodes = self.start_episode + num_episodes
        logger.info(f"Training from episode {self.start_episode + 1} to {total_episodes}")
        logger.info(f"Using {self.num_envs} parallel environments on {len(self.device_ids)} GPUs")
        
        episode_idx = 0
        while episode_idx < num_episodes:
            # Train episodes in parallel
            results = self.train_episodes_parallel()
            
            # Process results
            for env_idx, (reward, steps, return_, win_rate, trades, open_pos) in enumerate(results):
                actual_episode = self.start_episode + episode_idx + env_idx + 1
                
                # Store metrics
                self.episode_returns.append(return_)
                self.episode_win_rates.append(win_rate)
                
                # Display results
                if trades == 0 and open_pos > 0:
                    logger.info(f"Episode {actual_episode}/{total_episodes} (Env {env_idx}) - "
                               f"Reward: {reward:.2f}, "
                               f"Return: ${return_:.2f} (unrealized from {open_pos} open positions), "
                               f"Win Rate: {win_rate:.1%} ({trades} closed trades), "
                               f"Steps: {steps}")
                else:
                    logger.info(f"Episode {actual_episode}/{total_episodes} (Env {env_idx}) - "
                               f"Reward: {reward:.2f}, "
                               f"Return: ${return_:.2f}, "
                               f"Win Rate: {win_rate:.1%} ({trades} closed, {open_pos} open), "
                               f"Steps: {steps}")
                
                # Update step counter
                self.t += steps
            
            # Update networks if needed
            if self.t >= self.update_interval:
                self.update_networks()
                self.t = 0
            
            # Calculate rolling averages
            recent_returns = self.episode_returns[-100:] if len(self.episode_returns) > 100 else self.episode_returns
            recent_win_rates = self.episode_win_rates[-100:] if len(self.episode_win_rates) > 100 else self.episode_win_rates
            avg_return = np.mean(recent_returns) if recent_returns else 0
            avg_win_rate = np.mean(recent_win_rates) if recent_win_rates else 0
            
            # Check for best model
            if len(self.episode_returns) >= 10:
                combined_score = 0.7 * avg_return + 0.3 * (avg_win_rate * 1000)
                
                if combined_score > self.best_combined_score:
                    self.best_combined_score = combined_score
                    self.best_avg_return = avg_return
                    self.best_win_rate = avg_win_rate
                    
                    best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                    self.save_checkpoint(best_path, episode=actual_episode, is_best=True)
                    logger.info(f"  🏆 New best model! Avg Return: ${avg_return:.2f}, Avg Win Rate: {avg_win_rate:.1%}")
            
            # Display rolling averages
            actual_episode = self.start_episode + episode_idx + self.num_envs
            if actual_episode % 10 == 0:
                logger.info(f"  → 100-Episode Averages - Return: ${avg_return:.2f}, Win Rate: {avg_win_rate:.1%}")
                logger.info(f"  → Best so far - Return: ${self.best_avg_return:.2f}, Win Rate: {self.best_win_rate:.1%}")
            
            # Save checkpoints
            if actual_episode % checkpoint_interval == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f'ep{actual_episode}.pt')
                self.save_checkpoint(checkpoint_path, episode=actual_episode)
            
            # Always save latest
            latest_path = os.path.join(self.checkpoint_dir, 'latest.pt')
            self.save_checkpoint(latest_path, episode=actual_episode)
            
            episode_idx += self.num_envs
    
    def save_checkpoint(self, path: str, episode: int = None, is_best: bool = False):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint_data = {
            'network_state_dict': self.base_network.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            't': self.t,
            'episode': episode if episode is not None else self.start_episode,
            'episode_returns': self.episode_returns,
            'episode_win_rates': self.episode_win_rates,
            'best_avg_return': self.best_avg_return,
            'best_win_rate': self.best_win_rate,
            'best_combined_score': self.best_combined_score,
            'num_gpus': len(self.device_ids)
        }
        torch.save(checkpoint_data, path)
        logger.info(f"{'📌 Best model saved' if is_best else 'Saved checkpoint'} to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        if os.path.exists(path):
            logger.info(f"Loading checkpoint from {path}")
            # Handle PyTorch 2.6+ security changes
            try:
                checkpoint = torch.load(path, map_location=self.primary_device, weights_only=True)
            except:
                checkpoint = torch.load(path, map_location=self.primary_device, weights_only=False)
            
            self.base_network.load_state_dict(checkpoint['network_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.t = checkpoint.get('t', 0)
            self.start_episode = checkpoint.get('episode', 0)
            self.episode_returns = checkpoint.get('episode_returns', [])
            self.episode_win_rates = checkpoint.get('episode_win_rates', [])
            self.best_avg_return = checkpoint.get('best_avg_return', float('-inf'))
            self.best_win_rate = checkpoint.get('best_win_rate', 0.0)
            self.best_combined_score = checkpoint.get('best_combined_score', float('-inf'))
            
            saved_num_gpus = checkpoint.get('num_gpus', 1)
            if saved_num_gpus != len(self.device_ids):
                logger.warning(f"Checkpoint was saved with {saved_num_gpus} GPUs, now using {len(self.device_ids)}")
            
            logger.info(f"✅ Resumed from episode {self.start_episode} (step {self.t})")
            if self.episode_returns:
                logger.info(f"  Previous avg return: ${np.mean(self.episode_returns[-100:]):.2f}")
            if self.episode_win_rates:
                logger.info(f"  Previous avg win rate: {np.mean(self.episode_win_rates[-100:]):.1%}")
            
            return True
        return False


def main():
    """Main function to run multi-GPU training"""
    parser = argparse.ArgumentParser(description='Multi-GPU PPO with LSTM training')
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
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/ppo_lstm_multigpu',
                        help='Directory for saving checkpoints')
    parser.add_argument('--num-gpus', type=int, default=None,
                        help='Number of GPUs to use (default: all available)')
    
    args = parser.parse_args()
    
    # Determine number of GPUs
    if args.num_gpus is None:
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = min(args.num_gpus, torch.cuda.device_count())
    
    if num_gpus < 1:
        print("No GPUs available. Please run on a machine with GPUs.")
        sys.exit(1)
    
    print(f"Using {num_gpus} GPUs for training")
    device_ids = list(range(num_gpus))
    
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
        historical_data = asyncio.run(load_historical_data(
            data_loader, args.symbols, start_date, end_date
        ))
        
        if not historical_data:
            logger.warning("No historical data loaded. Will use simulated data.")
    else:
        logger.info("Skipping historical data load (--no-data-load specified)")
    
    # Create environments (one per GPU)
    envs = []
    for i in range(num_gpus):
        if i == 0 and (historical_data or not args.no_data_load):
            # First environment uses historical data
            env = HistoricalOptionsEnvironment(
                historical_data=historical_data,
                data_loader=data_loader,
                symbols=args.symbols,
                initial_capital=100000,
                max_positions=5,
                commission=0.65
            )
            logger.info(f"Environment 0: Using HistoricalOptionsEnvironment with {len(historical_data)} symbols")
        else:
            # Other environments use simulated data
            env = OptionsTradingEnvironment(
                initial_capital=100000,
                commission=0.65,
                max_positions=5
            )
            logger.info(f"Environment {i}: Using simulated OptionsTradingEnvironment")
        envs.append(env)
    
    # Create trainer
    trainer = MultiGPUPPOLSTMTrainer(
        envs=envs,
        learning_rate_actor=args.learning_rate_actor,
        learning_rate_critic=args.learning_rate_critic,
        gamma=args.gamma,
        epsilon=args.epsilon,
        update_interval=args.update_interval,
        device_ids=device_ids,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Handle checkpoint resumption
    resumed = False
    
    # Find latest checkpoint using the single-GPU trainer's method
    from train_ppo_lstm import PPOLSTMTrainer
    temp_trainer = PPOLSTMTrainer(envs[0], checkpoint_dir=args.checkpoint_dir)
    
    if not args.no_auto_resume or args.resume or args.resume_from:
        if args.resume_from:
            resumed = trainer.load_checkpoint(args.resume_from)
            if not resumed:
                logger.error(f"Failed to load checkpoint from {args.resume_from}")
                sys.exit(1)
        else:
            latest_checkpoint = temp_trainer.find_latest_checkpoint()
            if latest_checkpoint:
                logger.info(f"Found checkpoint: {latest_checkpoint}")
                resumed = trainer.load_checkpoint(latest_checkpoint)
            elif args.resume:
                logger.error("--resume specified but no checkpoint found")
                sys.exit(1)
            else:
                logger.info("No checkpoint found. Starting fresh training.")
    
    # Log training parameters
    logger.info(f"Multi-GPU Training: PPO with LSTM on {num_gpus} GPUs")
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"Actor LR: {args.learning_rate_actor}")
    logger.info(f"Critic LR: {args.learning_rate_critic}")
    logger.info(f"Gamma: {args.gamma}")
    logger.info(f"Epsilon: {args.epsilon}")
    logger.info(f"Update Interval: {args.update_interval}")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Checkpoint Directory: {args.checkpoint_dir}")
    if resumed:
        logger.info(f"✨ Resumed training from episode {trainer.start_episode}")
    
    # Train
    trainer.train(num_episodes=args.episodes, checkpoint_interval=args.checkpoint_interval)


if __name__ == "__main__":
    main()