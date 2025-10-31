#!/usr/bin/env python3
"""
PPO Training with Winning Episode Focus
Implements the specified PPO algorithm with LSTM processing and winning episode filtering
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import logging
from collections import deque
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
import pickle

# Import our modules
from src.options_clstm_ppo import OptionsCLSTMPPOAgent
from src.historical_options_data import HistoricalOptionsDataLoader, HistoricalOptionsEnvironment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WinningEpisodeBuffer:
    """Buffer that stores only winning episodes for training"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.winning_episodes = deque(maxlen=1000)  # Store metadata of winning episodes
        
    def add_episode(self, episode_data: List[Dict], total_return: float, win_rate: float):
        """Add episode if it's a winning episode"""
        if total_return > 0 and win_rate > 0.5:  # Criteria for winning episode
            self.buffer.extend(episode_data)
            self.winning_episodes.append({
                'return': total_return,
                'win_rate': win_rate,
                'length': len(episode_data),
                'timestamp': datetime.now()
            })
            logger.info(f"Added winning episode: Return=${total_return:.2f}, Win Rate={win_rate:.2%}")
    
    def sample(self, batch_size: int) -> List[Dict]:
        """Sample from winning episodes only"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


class PPOTrainerWithWinners:
    """PPO Trainer that focuses on winning episodes"""
    
    def __init__(
        self,
        env: gym.Env,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        update_interval: int = 2048,
        device: str = 'cuda'
    ):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_interval = update_interval
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize networks
        obs_space = env.observation_space
        action_space = env.action_space.n if hasattr(env.action_space, 'n') else 11
        
        # Create agent with CLSTM
        self.agent = OptionsCLSTMPPOAgent(
            observation_space=obs_space,
            action_space=action_space,
            learning_rate_actor_critic=actor_lr,
            learning_rate_clstm=critic_lr,
            gamma=gamma,
            clip_epsilon=epsilon,
            device=self.device
        )
        
        # Separate optimizers as specified
        self.actor_optimizer = optim.Adam(self.agent.network.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.agent.network.critic.parameters(), lr=critic_lr)
        
        # Replay buffer for PPO
        self.replay_buffer = []
        
        # Winning episode buffer
        self.winning_buffer = WinningEpisodeBuffer()
        
        # Best model tracking
        self.best_win_rate = 0.0
        self.best_model_state = None
        
    def process_state_with_lstm(self, state: Dict[str, np.ndarray]) -> torch.Tensor:
        """Process state with LSTM to obtain feature vector ft"""
        # Convert state to tensor
        state_tensor = self.agent._observation_to_tensor(state)
        
        # Process through CLSTM encoder
        with torch.no_grad():
            # Get LSTM features
            features = self.agent.network.clstm_encoder(
                self._prepare_lstm_input(state_tensor)
            )
        
        return features
    
    def _prepare_lstm_input(self, state_tensor: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Prepare input for LSTM by concatenating all features"""
        features = []
        for key in ['price_history', 'technical_indicators', 'options_chain', 
                   'portfolio_state', 'greeks_summary']:
            if key in state_tensor:
                features.append(state_tensor[key].flatten())
        
        return torch.cat(features, dim=-1).unsqueeze(0).unsqueeze(0)  # Add batch and seq dims
    
    def compute_advantage(self, reward: float, value: float, next_value: float) -> float:
        """Compute advantage estimate At = rt + γV(st+1) - V(st)"""
        return reward + self.gamma * next_value - value
    
    def update_networks(self):
        """Update actor and critic networks using PPO objective"""
        if len(self.replay_buffer) < self.update_interval:
            return
        
        # Convert replay buffer to tensors
        states = torch.stack([item['features'] for item in self.replay_buffer])
        actions = torch.tensor([item['action'] for item in self.replay_buffer], device=self.device)
        advantages = torch.tensor([item['advantage'] for item in self.replay_buffer], device=self.device)
        old_log_probs = torch.tensor([item['log_prob'] for item in self.replay_buffer], device=self.device)
        returns = torch.tensor([item['return'] for item in self.replay_buffer], device=self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update epochs
        for _ in range(10):  # K epochs
            # Get current policy
            action_logits = self.agent.network.actor(states)
            dist = Categorical(logits=action_logits)
            log_probs = dist.log_prob(actions)
            
            # Compute ratio
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.network.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            # Update critic
            values = self.agent.network.critic(states).squeeze()
            critic_loss = F.mse_loss(values, returns)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.network.critic.parameters(), 0.5)
            self.critic_optimizer.step()
        
        # Clear replay buffer
        self.replay_buffer.clear()
    
    def train_episode(self) -> Tuple[float, float, int]:
        """Train one episode following the specified algorithm"""
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]
            
        episode_reward = 0
        episode_data = []
        t = 0
        
        winning_trades = 0
        losing_trades = 0
        
        while True:
            # Step 6-7: Process state with LSTM
            features = self.process_state_with_lstm(state)
            
            # Step 8: Compute critic's value estimate
            with torch.no_grad():
                value = self.agent.network.critic(features).item()
            
            # Step 9: Sample action from policy
            with torch.no_grad():
                action_logits = self.agent.network.actor(features)
                dist = Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action).item()
            
            # Step 10: Execute action
            next_state, reward, done, truncated, info = self.env.step(action.item())
            episode_reward += reward
            
            # Track wins/losses
            if 'trade_closed' in info and info['trade_closed']:
                if info['pnl'] > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1
            
            # Get next value for advantage calculation
            if not done:
                next_features = self.process_state_with_lstm(next_state)
                with torch.no_grad():
                    next_value = self.agent.network.critic(next_features).item()
            else:
                next_value = 0
            
            # Step 11: Compute advantage
            advantage = self.compute_advantage(reward, value, next_value)
            
            # Step 12: Add to replay buffer
            transition = {
                'features': features.squeeze(),
                'action': action.item(),
                'advantage': advantage,
                'log_prob': log_prob,
                'return': reward + self.gamma * next_value,
                'reward': reward
            }
            self.replay_buffer.append(transition)
            episode_data.append(transition)
            
            # Step 13-16: Update networks at intervals
            if t % self.update_interval == 0 and t > 0:
                self.update_networks()
            
            state = next_state
            t += 1
            
            if done or truncated:
                break
        
        # Calculate win rate
        total_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Add to winning buffer if criteria met
        self.winning_buffer.add_episode(episode_data, episode_reward, win_rate)
        
        return episode_reward, win_rate, total_trades
    
    def train_from_winners_only(self, num_epochs: int = 10):
        """Train exclusively from winning episodes"""
        if len(self.winning_buffer) < self.update_interval:
            logger.warning(f"Not enough winning episodes ({len(self.winning_buffer)}), need {self.update_interval}")
            return
        
        logger.info(f"Training from {len(self.winning_buffer)} winning experiences")
        
        for epoch in range(num_epochs):
            # Sample from winning episodes only
            batch = self.winning_buffer.sample(min(self.update_interval, len(self.winning_buffer)))
            
            # Add to replay buffer for training
            self.replay_buffer = batch.copy()
            
            # Update networks
            self.update_networks()
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Trained on winning episodes")
    
    def save_winners_model(self, path: str):
        """Save the model trained on winners"""
        checkpoint = {
            'model_state_dict': self.agent.network.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'best_win_rate': self.best_win_rate,
            'winning_episodes_count': len(self.winning_buffer.winning_episodes),
            'total_winning_experiences': len(self.winning_buffer)
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved winners model to {path}")


def main():
    """Main training loop focusing on winning episodes"""
    # Create environment
    data_loader = HistoricalOptionsDataLoader()
    env = HistoricalOptionsEnvironment(
        data_loader=data_loader,
        initial_capital=100000,
        max_positions=5,
        commission=0.65
    )
    
    # Create trainer
    trainer = PPOTrainerWithWinners(
        env=env,
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.99,
        epsilon=0.2,
        update_interval=2048
    )
    
    # Training loop
    num_episodes = 10000
    winners_training_interval = 100  # Train from winners every N episodes
    
    for episode in range(num_episodes):
        # Regular episode training
        episode_reward, win_rate, trades = trainer.train_episode()
        
        logger.info(f"Episode {episode+1}: Reward={episode_reward:.2f}, "
                   f"Win Rate={win_rate:.2%}, Trades={trades}")
        
        # Periodically train from winners only
        if episode % winners_training_interval == 0 and episode > 0:
            trainer.train_from_winners_only(num_epochs=5)
            
        # Save best model
        if win_rate > trainer.best_win_rate and trades > 5:
            trainer.best_win_rate = win_rate
            trainer.save_winners_model(f'checkpoints/winners_model_wr{int(win_rate*100)}.pt')
            logger.info(f"New best win rate: {win_rate:.2%}")
    
    # Final training from all winners
    logger.info("Final training from all winning episodes...")
    trainer.train_from_winners_only(num_epochs=20)
    trainer.save_winners_model('checkpoints/final_winners_model.pt')


if __name__ == "__main__":
    main()