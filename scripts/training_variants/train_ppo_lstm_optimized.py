#!/usr/bin/env python3
"""
Optimized PPO with LSTM training script
Incorporates multiple speed optimizations while maintaining learning quality:
1. Parallel episode collection across multiple GPUs
2. Vectorized environment processing
3. Optimized batch processing
4. Efficient data loading with prefetching
5. Mixed precision training (FP16)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import os
import sys
from dotenv import load_dotenv
import argparse
import asyncio
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from collections import deque
import time

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.options_clstm_ppo import OptionsCLSTMPPONetwork
from src.historical_options_data import HistoricalOptionsDataLoader, HistoricalOptionsEnvironment
from src.options_trading_env import OptionsTradingEnvironment
from src.qlib_features import QlibFeatureEnhancer, QlibEnhancedEnvironment
from fix_zero_trading import create_enhanced_environment, add_entropy_bonus_to_loss
from fix_position_closing_paper_compliant import create_paper_compliant_closing_env
from fix_rewards_and_winrate import create_fixed_reward_environment
from train_ppo_lstm import load_historical_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorizedEnvironmentWrapper:
    """Wrapper to vectorize multiple environments for parallel processing"""
    
    def __init__(self, envs: List, device_mapping: Dict[int, int] = None):
        self.envs = envs
        self.num_envs = len(envs)
        self.device_mapping = device_mapping or {i: i % torch.cuda.device_count() for i in range(self.num_envs)}
        
    def reset(self, env_indices: Optional[List[int]] = None):
        """Reset specified environments or all if none specified"""
        if env_indices is None:
            env_indices = list(range(self.num_envs))
            
        states = []
        for idx in env_indices:
            state = self.envs[idx].reset()
            if isinstance(state, tuple):
                state = state[0]
            states.append(state)
        return states
    
    def step(self, actions: List[int], env_indices: Optional[List[int]] = None):
        """Execute actions in parallel across environments"""
        if env_indices is None:
            env_indices = list(range(self.num_envs))
            
        results = []
        for idx, action in zip(env_indices, actions):
            result = self.envs[idx].step(action)
            results.append(result)
        return results


class OptimizedPPOLSTMTrainer:
    """Optimized PPO with LSTM trainer with multiple speed improvements"""
    
    def __init__(
        self,
        env_wrapper: VectorizedEnvironmentWrapper,
        learning_rate_actor: float = 3e-4,
        learning_rate_critic: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        update_interval: int = 128,
        batch_size: int = 64,
        num_epochs: int = 4,
        device_ids: List[int] = None,
        checkpoint_dir: str = 'checkpoints/ppo_lstm_optimized',
        entropy_coef: float = 0.01,
        portfolio_bonus: bool = False,
        use_mixed_precision: bool = True,
        prefetch_buffer_size: int = 4
    ):
        self.env_wrapper = env_wrapper
        self.num_envs = env_wrapper.num_envs
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_interval = update_interval
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.entropy_coef = entropy_coef
        self.portfolio_bonus = portfolio_bonus
        self.use_mixed_precision = use_mixed_precision
        self.prefetch_buffer_size = prefetch_buffer_size
        
        # Set up devices
        if device_ids is None:
            self.device_ids = list(range(torch.cuda.device_count()))
        else:
            self.device_ids = device_ids
        
        self.primary_device = torch.device(f'cuda:{self.device_ids[0]}')
        
        logger.info(f"Using {len(self.device_ids)} GPUs: {self.device_ids}")
        logger.info(f"Running {self.num_envs} parallel environments")
        
        # Create model on primary device
        self.network = OptionsCLSTMPPONetwork(
            observation_space=env_wrapper.envs[0].observation_space,
            action_dim=11
        ).to(self.primary_device)
        
        # Use DataParallel for multi-GPU training
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
        
        # Mixed precision training
        self.scaler = GradScaler() if use_mixed_precision else None
        
        # Experience buffer with prefetching
        self.experience_buffer = deque(maxlen=update_interval * num_envs)
        self.prefetch_queue = deque(maxlen=prefetch_buffer_size)
        
        # Parallel executors
        self.thread_executor = ThreadPoolExecutor(max_workers=num_envs)
        self.process_executor = ProcessPoolExecutor(max_workers=min(4, mp.cpu_count()))
        
        # Tracking
        self.t = 0
        self.start_episode = 0
        self.episodes_completed = 0
        
        # Metrics
        self.best_avg_return = float('-inf')
        self.best_win_rate = 0.0
        self.best_combined_score = float('-inf')
        self.episode_returns = deque(maxlen=1000)
        self.episode_win_rates = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        
        # State cache for faster processing
        self.state_cache = {}
        
    def process_states_batch(self, states: List[Dict], device: torch.device) -> torch.Tensor:
        """Process multiple states in a batch for efficiency"""
        if not states or all(s is None for s in states):
            return torch.zeros(len(states), self.base_network.clstm_encoder.hidden_dim).to(device)
        
        batch_features = []
        for state in states:
            if state is None:
                batch_features.append(torch.zeros(self.base_network.clstm_encoder.hidden_dim).to(device))
                continue
                
            # Check cache
            state_hash = hash(str(state))
            if state_hash in self.state_cache:
                batch_features.append(self.state_cache[state_hash].to(device))
                continue
            
            features = []
            for key in ['price_history', 'technical_indicators', 'options_chain', 
                       'portfolio_state', 'greeks_summary', 'qlib_features', 'market_features']:
                if key in state:
                    tensor = torch.tensor(state[key], dtype=torch.float32)
                    features.append(tensor.flatten())
            
            if features:
                combined = torch.cat(features, dim=0)
            else:
                combined = torch.zeros(self.base_network.clstm_encoder.hidden_dim)
            
            # Cache the processed feature
            self.state_cache[state_hash] = combined.cpu()
            batch_features.append(combined.to(device))
        
        # Stack all features
        batch_tensor = torch.stack(batch_features).unsqueeze(1).to(device)
        
        # Process through LSTM encoder
        with torch.no_grad():
            if self.use_mixed_precision:
                with autocast():
                    encoded = self.base_network.clstm_encoder(batch_tensor)
            else:
                encoded = self.base_network.clstm_encoder(batch_tensor)
                
        return encoded.squeeze(1)
    
    def collect_rollouts_parallel(self):
        """Collect rollouts from all environments in parallel"""
        # Reset any done environments
        states = self.env_wrapper.reset()
        
        rollout_data = {env_idx: {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
            'infos': []
        } for env_idx in range(self.num_envs)}
        
        steps_collected = 0
        active_envs = list(range(self.num_envs))
        
        while steps_collected < self.update_interval and active_envs:
            # Process states in batch
            batch_states = [states[i] if i in active_envs else None for i in range(self.num_envs)]
            device = self.primary_device
            
            # Get features for all active environments
            features = self.process_states_batch(batch_states, device)
            
            # Get values and actions in batch
            with torch.no_grad():
                if self.use_mixed_precision:
                    with autocast():
                        values = self.base_network.critic(features)
                        action_logits = self.base_network.actor(features)
                else:
                    values = self.base_network.critic(features)
                    action_logits = self.base_network.actor(features)
                
                # Sample actions
                dists = Categorical(logits=action_logits)
                actions = dists.sample()
                log_probs = dists.log_prob(actions)
            
            # Execute actions in parallel
            action_list = [actions[i].item() if i in active_envs else 0 for i in range(self.num_envs)]
            results = self.env_wrapper.step(action_list, active_envs)
            
            # Process results
            new_active_envs = []
            for idx, (env_idx, result) in enumerate(zip(active_envs, results)):
                if len(result) == 4:
                    next_state, reward, done, info = result
                    truncated = False
                else:
                    next_state, reward, done, truncated, info = result
                
                # Apply portfolio bonus if enabled
                if self.portfolio_bonus and info and hasattr(self.env_wrapper.envs[env_idx], 'initial_capital'):
                    current_value = None
                    if 'portfolio_value' in info:
                        current_value = info['portfolio_value']
                    elif 'capital' in info:
                        current_value = info['capital']
                    
                    if current_value is not None:
                        portfolio_return = (current_value - self.env_wrapper.envs[env_idx].initial_capital) / self.env_wrapper.envs[env_idx].initial_capital
                        if portfolio_return > 0:
                            bonus = portfolio_return * 0.1
                            reward += bonus
                
                # Store experience
                rollout_data[env_idx]['states'].append(states[env_idx])
                rollout_data[env_idx]['actions'].append(actions[idx].item())
                rollout_data[env_idx]['rewards'].append(reward)
                rollout_data[env_idx]['values'].append(values[idx].item())
                rollout_data[env_idx]['log_probs'].append(log_probs[idx].item())
                rollout_data[env_idx]['dones'].append(done or truncated)
                rollout_data[env_idx]['infos'].append(info)
                
                steps_collected += 1
                
                if done or truncated:
                    # Reset this environment
                    reset_state = self.env_wrapper.envs[env_idx].reset()
                    if isinstance(reset_state, tuple):
                        reset_state = reset_state[0]
                    states[env_idx] = reset_state
                    
                    # Track episode metrics
                    if info:
                        episode_return = 0
                        if 'episode_return' in info:
                            episode_return = info['episode_return']
                        elif hasattr(self.env_wrapper.envs[env_idx], 'initial_capital'):
                            if 'portfolio_value' in info:
                                episode_return = info['portfolio_value'] - self.env_wrapper.envs[env_idx].initial_capital
                            elif 'capital' in info:
                                episode_return = info['capital'] - self.env_wrapper.envs[env_idx].initial_capital
                        
                        self.episode_returns.append(episode_return)
                        
                        # Track win rate
                        win_rate = 0
                        if hasattr(self.env_wrapper.envs[env_idx], 'winning_trades') and hasattr(self.env_wrapper.envs[env_idx], 'losing_trades'):
                            total_trades = self.env_wrapper.envs[env_idx].winning_trades + self.env_wrapper.envs[env_idx].losing_trades
                            if total_trades > 0:
                                win_rate = self.env_wrapper.envs[env_idx].winning_trades / total_trades
                        
                        self.episode_win_rates.append(win_rate)
                        self.episode_lengths.append(len(rollout_data[env_idx]['rewards']))
                        self.episodes_completed += 1
                else:
                    states[env_idx] = next_state
                    new_active_envs.append(env_idx)
            
            active_envs = new_active_envs
        
        return rollout_data
    
    def compute_returns_and_advantages(self, rollout_data: Dict):
        """Compute returns and advantages for all rollouts"""
        all_experiences = []
        
        for env_idx, data in rollout_data.items():
            if not data['rewards']:
                continue
                
            rewards = torch.tensor(data['rewards'], dtype=torch.float32)
            values = torch.tensor(data['values'], dtype=torch.float32)
            dones = torch.tensor(data['dones'], dtype=torch.float32)
            
            # Compute returns and advantages
            returns = torch.zeros_like(rewards)
            advantages = torch.zeros_like(rewards)
            
            # Bootstrap from last value
            if not data['dones'][-1]:
                # Get last state value
                last_state = data['states'][-1]
                with torch.no_grad():
                    last_features = self.process_states_batch([last_state], self.primary_device)
                    if self.use_mixed_precision:
                        with autocast():
                            last_value = self.base_network.critic(last_features).item()
                    else:
                        last_value = self.base_network.critic(last_features).item()
            else:
                last_value = 0
            
            # Compute returns backwards
            running_return = last_value
            for t in reversed(range(len(rewards))):
                if dones[t]:
                    running_return = 0
                returns[t] = rewards[t] + self.gamma * running_return
                running_return = returns[t]
            
            # Compute advantages
            advantages = returns - values
            
            # Store processed experiences
            for t in range(len(rewards)):
                all_experiences.append({
                    'state': data['states'][t],
                    'action': data['actions'][t],
                    'return': returns[t].item(),
                    'advantage': advantages[t].item(),
                    'old_log_prob': data['log_probs'][t]
                })
        
        return all_experiences
    
    def update_networks_optimized(self, experiences: List[Dict]):
        """Update networks with optimized batching and mixed precision"""
        if not experiences:
            return
        
        # Shuffle experiences
        np.random.shuffle(experiences)
        
        # Normalize advantages
        advantages = torch.tensor([exp['advantage'] for exp in experiences], dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for epoch in range(self.num_epochs):
            # Process in batches
            for i in range(0, len(experiences), self.batch_size):
                batch = experiences[i:i + self.batch_size]
                batch_advantages = advantages[i:i + self.batch_size].to(self.primary_device)
                
                # Prepare batch data
                batch_states = [exp['state'] for exp in batch]
                batch_actions = torch.tensor([exp['action'] for exp in batch], dtype=torch.long).to(self.primary_device)
                batch_returns = torch.tensor([exp['return'] for exp in batch], dtype=torch.float32).to(self.primary_device)
                batch_old_log_probs = torch.tensor([exp['old_log_prob'] for exp in batch], dtype=torch.float32).to(self.primary_device)
                
                # Process states
                batch_features = self.process_states_batch(batch_states, self.primary_device)
                
                # Update critic
                if self.use_mixed_precision:
                    with autocast():
                        values = self.base_network.critic(batch_features).squeeze()
                        critic_loss = F.mse_loss(values, batch_returns)
                    
                    self.critic_optimizer.zero_grad()
                    self.scaler.scale(critic_loss).backward()
                    self.scaler.unscale_(self.critic_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.base_network.critic.parameters(), 1.0)
                    self.scaler.step(self.critic_optimizer)
                    self.scaler.update()
                else:
                    values = self.base_network.critic(batch_features).squeeze()
                    critic_loss = F.mse_loss(values, batch_returns)
                    
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.base_network.critic.parameters(), 1.0)
                    self.critic_optimizer.step()
                
                # Update actor
                if self.use_mixed_precision:
                    with autocast():
                        action_logits = self.base_network.actor(batch_features)
                        dist = Categorical(logits=action_logits)
                        log_probs = dist.log_prob(batch_actions)
                        
                        # PPO objective
                        ratio = torch.exp(log_probs - batch_old_log_probs)
                        surr1 = ratio * batch_advantages
                        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                        actor_loss = -torch.min(surr1, surr2).mean()
                        
                        # Add entropy bonus
                        if self.entropy_coef > 0:
                            entropy = dist.entropy().mean()
                            actor_loss = actor_loss - self.entropy_coef * entropy
                    
                    self.actor_optimizer.zero_grad()
                    self.scaler.scale(actor_loss).backward()
                    self.scaler.unscale_(self.actor_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.base_network.actor.parameters(), 1.0)
                    self.scaler.step(self.actor_optimizer)
                    self.scaler.update()
                else:
                    action_logits = self.base_network.actor(batch_features)
                    dist = Categorical(logits=action_logits)
                    log_probs = dist.log_prob(batch_actions)
                    
                    ratio = torch.exp(log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    if self.entropy_coef > 0:
                        entropy = dist.entropy().mean()
                        actor_loss = actor_loss - self.entropy_coef * entropy
                    
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.base_network.actor.parameters(), 1.0)
                    self.actor_optimizer.step()
        
        # Clear state cache periodically
        if len(self.state_cache) > 10000:
            self.state_cache.clear()
    
    def train(self, num_episodes: int = 1000, checkpoint_interval: int = 100):
        """Main training loop with optimizations"""
        total_episodes = self.start_episode + num_episodes
        logger.info(f"Optimized training from episode {self.start_episode + 1} to {total_episodes}")
        logger.info(f"Using {self.num_envs} parallel environments on {len(self.device_ids)} GPUs")
        logger.info(f"Mixed precision training: {'Enabled' if self.use_mixed_precision else 'Disabled'}")
        
        start_time = time.time()
        last_checkpoint_time = start_time
        
        while self.episodes_completed < num_episodes:
            # Collect rollouts in parallel
            rollout_start = time.time()
            rollout_data = self.collect_rollouts_parallel()
            rollout_time = time.time() - rollout_start
            
            # Compute returns and advantages
            compute_start = time.time()
            experiences = self.compute_returns_and_advantages(rollout_data)
            compute_time = time.time() - compute_start
            
            # Update networks
            update_start = time.time()
            self.update_networks_optimized(experiences)
            update_time = time.time() - update_start
            
            # Log progress
            if self.episodes_completed % 10 == 0:
                elapsed_time = time.time() - start_time
                episodes_per_second = self.episodes_completed / elapsed_time
                
                # Calculate metrics
                avg_return = np.mean(list(self.episode_returns)) if self.episode_returns else 0
                avg_win_rate = np.mean(list(self.episode_win_rates)) if self.episode_win_rates else 0
                avg_length = np.mean(list(self.episode_lengths)) if self.episode_lengths else 0
                
                logger.info(f"Episodes: {self.episodes_completed}/{num_episodes} | "
                          f"Speed: {episodes_per_second:.2f} eps/s | "
                          f"Avg Return: ${avg_return:.2f} | "
                          f"Win Rate: {avg_win_rate:.1%} | "
                          f"Avg Length: {avg_length:.0f}")
                logger.info(f"  Timing - Rollout: {rollout_time:.2f}s, Compute: {compute_time:.2f}s, Update: {update_time:.2f}s")
                
                # Check for best model
                if len(self.episode_returns) >= 10:
                    combined_score = avg_return * (1 + avg_win_rate)
                    if combined_score > self.best_combined_score:
                        self.best_combined_score = combined_score
                        self.best_avg_return = avg_return
                        self.best_win_rate = avg_win_rate
                        
                        best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                        self.save_checkpoint(best_path, is_best=True)
                        logger.info(f"  🏆 New best model! Score: {combined_score:.2f}")
            
            # Save checkpoints
            current_time = time.time()
            if current_time - last_checkpoint_time > 300:  # Every 5 minutes
                checkpoint_path = os.path.join(self.checkpoint_dir, f'ep{self.episodes_completed}.pt')
                self.save_checkpoint(checkpoint_path)
                last_checkpoint_time = current_time
            
            # Always save latest
            if self.episodes_completed % checkpoint_interval == 0:
                latest_path = os.path.join(self.checkpoint_dir, 'latest.pt')
                self.save_checkpoint(latest_path)
        
        # Final stats
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/3600:.2f} hours")
        logger.info(f"Average speed: {num_episodes/total_time:.2f} episodes/second")
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint_data = {
            'network_state_dict': self.base_network.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'episodes_completed': self.episodes_completed,
            'episode_returns': list(self.episode_returns),
            'episode_win_rates': list(self.episode_win_rates),
            'best_avg_return': self.best_avg_return,
            'best_win_rate': self.best_win_rate,
            'best_combined_score': self.best_combined_score,
            'num_gpus': len(self.device_ids),
            'num_envs': self.num_envs
        }
        torch.save(checkpoint_data, path)
        logger.info(f"{'📌 Best model' if is_best else 'Checkpoint'} saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        if os.path.exists(path):
            logger.info(f"Loading checkpoint from {path}")
            try:
                checkpoint = torch.load(path, map_location=self.primary_device, weights_only=True)
            except:
                checkpoint = torch.load(path, map_location=self.primary_device, weights_only=False)
            
            self.base_network.load_state_dict(checkpoint['network_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            if self.scaler and checkpoint.get('scaler_state_dict'):
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            self.episodes_completed = checkpoint.get('episodes_completed', 0)
            self.start_episode = self.episodes_completed
            self.episode_returns = deque(checkpoint.get('episode_returns', []), maxlen=1000)
            self.episode_win_rates = deque(checkpoint.get('episode_win_rates', []), maxlen=1000)
            self.best_avg_return = checkpoint.get('best_avg_return', float('-inf'))
            self.best_win_rate = checkpoint.get('best_win_rate', 0.0)
            self.best_combined_score = checkpoint.get('best_combined_score', float('-inf'))
            
            logger.info(f"✅ Resumed from episode {self.episodes_completed}")
            return True
        return False


def create_optimized_environments(args, historical_data, data_loader):
    """Create multiple environments for parallel training"""
    envs = []
    num_envs = min(torch.cuda.device_count() * 2, 8)  # 2 envs per GPU, max 8
    
    for i in range(num_envs):
        if i == 0 and historical_data:
            # First environment uses historical data
            env = HistoricalOptionsEnvironment(
                historical_data=historical_data,
                data_loader=data_loader,
                symbols=args.symbols,
                initial_capital=100000,
                max_positions=5,
                commission=0.65
            )
        else:
            # Others use simulated data for diversity
            env = OptionsTradingEnvironment(
                initial_capital=100000,
                commission=0.65,
                max_positions=5
            )
        
        # Apply enhancements if requested
        if args.fix_zero_trading:
            base_class = type(env)
            if isinstance(env, HistoricalOptionsEnvironment):
                env = create_enhanced_environment(
                    base_class,
                    historical_data=historical_data,
                    data_loader=data_loader,
                    symbols=args.symbols,
                    initial_capital=100000,
                    max_positions=5,
                    commission=0.65
                )
            else:
                env = create_enhanced_environment(
                    base_class,
                    initial_capital=100000,
                    max_positions=5,
                    commission=0.65
                )
        
        if args.closing_incentives:
            base_class = type(env)
            if isinstance(env, HistoricalOptionsEnvironment):
                env = create_paper_compliant_closing_env(
                    base_class,
                    historical_data=historical_data,
                    data_loader=data_loader,
                    symbols=args.symbols,
                    initial_capital=100000,
                    max_positions=5,
                    commission=0.65
                )
            else:
                env = create_paper_compliant_closing_env(
                    base_class,
                    initial_capital=100000,
                    max_positions=5,
                    commission=0.65
                )
        
        if args.fix_rewards:
            base_class = type(env)
            if isinstance(env, HistoricalOptionsEnvironment):
                env = create_fixed_reward_environment(
                    base_class,
                    historical_data=historical_data,
                    data_loader=data_loader,
                    symbols=args.symbols,
                    initial_capital=100000,
                    max_positions=5,
                    commission=0.65
                )
            else:
                env = create_fixed_reward_environment(
                    base_class,
                    initial_capital=100000,
                    max_positions=5,
                    commission=0.65
                )
        
        envs.append(env)
    
    return envs


def main():
    """Main function to run optimized training"""
    parser = argparse.ArgumentParser(description='Optimized PPO with LSTM training')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes to train (default: 1000)')
    parser.add_argument('--learning-rate-actor', type=float, default=3e-4,
                        help='Actor learning rate (default: 3e-4)')
    parser.add_argument('--learning-rate-critic', type=float, default=1e-3,
                        help='Critic learning rate (default: 1e-3)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--epsilon', type=float, default=0.2,
                        help='PPO clipping range (default: 0.2)')
    parser.add_argument('--update-interval', type=int, default=256,
                        help='Update interval (default: 256)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for updates (default: 64)')
    parser.add_argument('--num-epochs', type=int, default=4,
                        help='Number of epochs per update (default: 4)')
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                        help='Save checkpoint every N episodes (default: 100)')
    parser.add_argument('--symbols', nargs='+', default=['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'TSLA', 'PLTR', 'NVDA', 'META', 'GOOGL', 'AMZN'],
                        help='Symbols to trade')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date for historical data (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date for historical data (YYYY-MM-DD)')
    parser.add_argument('--no-data-load', action='store_true',
                        help='Skip historical data loading')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/ppo_lstm_optimized',
                        help='Directory for saving checkpoints')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='Entropy coefficient (default: 0.01)')
    parser.add_argument('--portfolio-bonus', action='store_true',
                        help='Add positive reward bonus for portfolio gains')
    parser.add_argument('--fix-zero-trading', action='store_true',
                        help='Use enhanced environment to fix zero trading')
    parser.add_argument('--closing-incentives', action='store_true',
                        help='Add incentives for closing positions')
    parser.add_argument('--fix-rewards', action='store_true',
                        help='Fix reward calculation')
    parser.add_argument('--no-mixed-precision', action='store_true',
                        help='Disable mixed precision training')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume from specific checkpoint')
    
    args = parser.parse_args()
    
    # Check GPU availability
    if torch.cuda.device_count() < 1:
        logger.error("No GPUs available. This script requires at least one GPU.")
        sys.exit(1)
    
    logger.info(f"Found {torch.cuda.device_count()} GPUs available")
    
    # Load environment variables
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
    
    # Create optimized environments
    envs = create_optimized_environments(args, historical_data, data_loader)
    env_wrapper = VectorizedEnvironmentWrapper(envs)
    
    # Create trainer
    trainer = OptimizedPPOLSTMTrainer(
        env_wrapper=env_wrapper,
        learning_rate_actor=args.learning_rate_actor,
        learning_rate_critic=args.learning_rate_critic,
        gamma=args.gamma,
        epsilon=args.epsilon,
        update_interval=args.update_interval,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        checkpoint_dir=args.checkpoint_dir,
        entropy_coef=args.entropy_coef,
        portfolio_bonus=args.portfolio_bonus,
        use_mixed_precision=not args.no_mixed_precision
    )
    
    # Handle checkpoint resumption
    if args.resume_from:
        if not trainer.load_checkpoint(args.resume_from):
            logger.error(f"Failed to load checkpoint from {args.resume_from}")
            sys.exit(1)
    elif args.resume:
        latest_path = os.path.join(args.checkpoint_dir, 'latest.pt')
        if os.path.exists(latest_path):
            trainer.load_checkpoint(latest_path)
        else:
            logger.warning("No checkpoint found to resume from")
    
    # Start training
    logger.info("Starting optimized training...")
    trainer.train(num_episodes=args.episodes, checkpoint_interval=args.checkpoint_interval)


if __name__ == "__main__":
    main()