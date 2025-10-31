#!/usr/bin/env python3
"""
Enhanced Algorithm 2: PPO with LSTM using Historical Options Data
This implementation loads real historical data before training
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
import asyncio
from datetime import datetime, timedelta
import json

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.options_clstm_ppo import OptionsCLSTMPPONetwork
from src.historical_options_data import HistoricalOptionsDataLoader, HistoricalOptionsEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PPOLSTMTrainerWithHistory:
    """Enhanced PPO-LSTM trainer that loads historical data before training"""
    
    def __init__(
        self,
        env,
        learning_rate_actor: float = 3e-4,
        learning_rate_critic: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        update_interval: int = 128,
        device: str = 'cuda'
    ):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_interval = update_interval
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize network
        self.network = OptionsCLSTMPPONetwork(
            observation_space=env.observation_space,
            action_dim=11
        ).to(self.device)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(
            self.network.actor.parameters(), 
            lr=learning_rate_actor
        )
        self.critic_optimizer = optim.Adam(
            self.network.critic.parameters(), 
            lr=learning_rate_critic
        )
        
        # Replay buffer
        self.replay_buffer = {
            'features': [],
            'actions': [],
            'advantages': [],
            'returns': [],
            'old_log_probs': []
        }
        
        self.t = 0
        
        # Track training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_rates = []
        
    def process_state_with_lstm(self, state: Dict[str, np.ndarray]) -> torch.Tensor:
        """Process state with LSTM to obtain feature vector ft"""
        if state is None:
            # Return zero tensor if state is None
            return torch.zeros(1, self.network.clstm_encoder.hidden_dim).to(self.device)
            
        # Convert state components to tensors
        features = []
        for key in ['price_history', 'technical_indicators', 'options_chain', 
                   'portfolio_state', 'greeks_summary']:
            if key in state:
                tensor = torch.tensor(state[key], dtype=torch.float32).to(self.device)
                features.append(tensor.flatten())
        
        if not features:
            # Return zero tensor if no features
            return torch.zeros(1, self.network.clstm_encoder.hidden_dim).to(self.device)
            
        # Concatenate all features
        combined = torch.cat(features, dim=0).unsqueeze(0).unsqueeze(0)
        
        # Process through LSTM encoder
        with torch.no_grad():
            ft = self.network.clstm_encoder(combined)
            
        return ft.squeeze(0)
    
    def train_episode(self):
        """Train one episode following Algorithm 2"""
        # Step 4: Initialize environment with initial state s0
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]
            
        episode_reward = 0
        episode_steps = 0
        
        # Track episode metrics
        winning_trades = 0
        losing_trades = 0
        total_trades = 0
        
        # Step 5: For each step t in the episode
        while True:
            # Handle None state (no data available)
            if state is None:
                logger.warning("Received None state, ending episode")
                break
                
            # Step 7: Process st with LSTM to obtain feature vector ft
            ft = self.process_state_with_lstm(state)
            
            # Step 8: Compute critic's value estimate v̂t = Vϕ(ft)
            with torch.no_grad():
                vt = self.network.critic(ft.unsqueeze(0)).item()
            
            # Step 9: Sample action at from policy πθ(at|ft)
            with torch.no_grad():
                action_logits = self.network.actor(ft.unsqueeze(0))
                dist = Categorical(logits=action_logits)
                at = dist.sample()
                log_prob = dist.log_prob(at).item()
            
            # Step 10: Execute at in environment to receive rt and st+1
            step_result = self.env.step(at.item())
            if len(step_result) == 4:
                next_state, rt, done, info = step_result
                truncated = False
            else:
                next_state, rt, done, truncated, info = step_result
            episode_reward += rt
            
            # Track trading metrics
            if 'trade_closed' in info and info['trade_closed']:
                total_trades += 1
                if 'pnl' in info and info['pnl'] > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1
            
            # Get next value for advantage calculation
            if not done and not truncated and next_state is not None:
                # Process next state
                ft_next = self.process_state_with_lstm(next_state)
                with torch.no_grad():
                    vt_next = self.network.critic(ft_next.unsqueeze(0)).item()
            else:
                vt_next = 0
            
            # Step 11: Compute advantage estimate At = rt + γv̂t+1 - v̂t
            At = rt + self.gamma * vt_next - vt
            
            # Step 12: Add transition (ft, at, At) to replay buffer D
            self.replay_buffer['features'].append(ft)
            self.replay_buffer['actions'].append(at.item())
            self.replay_buffer['advantages'].append(At)
            self.replay_buffer['returns'].append(rt + self.gamma * vt_next)
            self.replay_buffer['old_log_probs'].append(log_prob)
            
            # Step 13: if t mod T = 0
            self.t += 1
            if self.t % self.update_interval == 0:
                self.update_networks()
                # Step 16: Clear replay buffer D
                self.clear_buffer()
            
            # Move to next state
            state = next_state
            episode_steps += 1
            
            if done or truncated:
                break
                
        # Calculate win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return episode_reward, episode_steps, win_rate
    
    def update_networks(self):
        """Steps 14-15: Update critic and actor networks"""
        if len(self.replay_buffer['features']) == 0:
            return
            
        # Convert buffer to tensors
        features = torch.stack(self.replay_buffer['features'])
        actions = torch.tensor(self.replay_buffer['actions'], device=self.device)
        advantages = torch.tensor(self.replay_buffer['advantages'], device=self.device)
        returns = torch.tensor(self.replay_buffer['returns'], device=self.device)
        old_log_probs = torch.tensor(self.replay_buffer['old_log_probs'], device=self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Step 14: Update critic by minimizing MSE
        values = self.network.critic(features).squeeze()
        critic_loss = F.mse_loss(values, returns)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Step 15: Update actor using PPO objective
        action_logits = self.network.actor(features)
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        
        # PPO objective: L^PPO(θ)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        logger.info(f"Updated networks - Critic Loss: {critic_loss:.4f}, Actor Loss: {actor_loss:.4f}")
    
    def clear_buffer(self):
        """Step 16: Clear the replay buffer D"""
        for key in self.replay_buffer:
            self.replay_buffer[key].clear()
    
    def train(self, num_episodes: int = 1000):
        """Main training loop"""
        logger.info(f"Starting PPO-LSTM training for {num_episodes} episodes")
        
        # Step 3: For each episode
        for episode in range(num_episodes):
            episode_reward, steps, win_rate = self.train_episode()
            
            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(steps)
            self.win_rates.append(win_rate)
            
            # Calculate rolling averages
            recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
            avg_reward = np.mean(recent_rewards)
            avg_win_rate = np.mean(self.win_rates[-100:]) if len(self.win_rates) >= 100 else np.mean(self.win_rates)
            
            logger.info(f"Episode {episode+1}/{num_episodes} - "
                       f"Reward: {episode_reward:.2f}, Steps: {steps}, "
                       f"Win Rate: {win_rate:.2%}, "
                       f"Avg Reward: {avg_reward:.2f}, "
                       f"Avg Win Rate: {avg_win_rate:.2%}")
            
            # Save checkpoint periodically
            if (episode + 1) % 100 == 0:
                self.save_checkpoint(f"checkpoints/ppo_lstm_historical_ep{episode+1}.pt")
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            't': self.t,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'win_rates': self.win_rates
        }, path)
        logger.info(f"Saved checkpoint to {path}")


async def load_historical_data(data_loader, symbols, start_date, end_date):
    """Load historical options data asynchronously"""
    logger.info(f"Loading historical data for {symbols} from {start_date} to {end_date}")
    
    historical_data = await data_loader.load_historical_options_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        use_cache=True
    )
    
    # Process the data
    processed_data = {}
    for symbol, data in historical_data.items():
        if data:
            logger.info(f"Loaded {len(data)} data points for {symbol}")
            processed_data[symbol] = data_loader.get_training_data(symbol)
        else:
            logger.warning(f"No data loaded for {symbol}")
            
    return processed_data


def main():
    """Main function to run Algorithm 2 with historical data"""
    # Get API keys from environment
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not api_secret:
        logger.error("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env file")
        sys.exit(1)
    
    # Load configuration
    config_path = 'symbols_config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        symbols = config.get('symbols', ['SPY', 'QQQ', 'IWM'])
        start_date = datetime.strptime(config.get('start_date', '2024-01-01'), '%Y-%m-%d')
        end_date = datetime.strptime(config.get('end_date', '2024-12-31'), '%Y-%m-%d')
    else:
        # Default configuration
        symbols = ['SPY', 'QQQ', 'IWM']
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now() - timedelta(days=1)
    
    # Create data loader
    data_loader = HistoricalOptionsDataLoader(
        api_key=api_key,
        api_secret=api_secret
    )
    
    # Load historical data
    logger.info("Loading historical options data...")
    historical_data = asyncio.run(load_historical_data(
        data_loader, symbols, start_date, end_date
    ))
    
    if not historical_data:
        logger.error("No historical data loaded. Cannot proceed with training.")
        sys.exit(1)
    
    # Create environment with historical data
    env = HistoricalOptionsEnvironment(
        historical_data=historical_data,
        data_loader=data_loader,
        symbols=list(historical_data.keys()),
        initial_capital=100000,
        max_positions=5,
        commission=0.65
    )
    
    # Create trainer
    trainer = PPOLSTMTrainerWithHistory(
        env=env,
        learning_rate_actor=3e-4,
        learning_rate_critic=1e-3,
        gamma=0.99,
        epsilon=0.2,
        update_interval=128
    )
    
    # Train
    trainer.train(num_episodes=1000)
    
    # Save final model
    trainer.save_checkpoint("checkpoints/ppo_lstm_historical_final.pt")
    logger.info("Training completed!")


if __name__ == "__main__":
    main()