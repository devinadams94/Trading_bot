#!/usr/bin/env python3
"""
LSTM-PPO Training Script for Options Trading
Following the architecture from training.png with technical indicators
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributions import Categorical
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from datetime import datetime, timedelta
import argparse
from collections import deque
import json
import time
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.options_clstm_ppo import OptionsCLSTMPPOAgent
from src.historical_options_data import HistoricalOptionsDataLoader, HistoricalOptionsEnvironment
from config.config import TradingConfig
from config.symbols_loader import SymbolsConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators for state representation"""
    
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        if len(prices) < slow:
            return 0, 0, 0
            
        prices_series = pd.Series(prices)
        exp1 = prices_series.ewm(span=fast, adjust=False).mean()
        exp2 = prices_series.ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50  # Neutral RSI
            
        prices_series = pd.Series(prices)
        delta = prices_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50
    
    @staticmethod
    def calculate_cci(high, low, close, period=20):
        """Calculate CCI indicator"""
        if len(high) < period:
            return 0
            
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        close_series = pd.Series(close)
        
        typical_price = (high_series + low_series + close_series) / 3
        moving_avg = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        
        cci = (typical_price - moving_avg) / (0.015 * mean_deviation)
        
        return cci.iloc[-1] if not np.isnan(cci.iloc[-1]) else 0
    
    @staticmethod
    def calculate_adx(high, low, close, period=14):
        """Calculate ADX indicator"""
        if len(high) < period + 1:
            return 0
            
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        close_series = pd.Series(close)
        
        # Calculate True Range
        tr1 = high_series - low_series
        tr2 = abs(high_series - close_series.shift(1))
        tr3 = abs(low_series - close_series.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate directional movements
        up_move = high_series - high_series.shift(1)
        down_move = low_series.shift(1) - low_series
        
        pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        # Calculate smoothed averages
        atr = tr.rolling(window=period).mean()
        pos_di = 100 * (pos_dm.rolling(window=period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(window=period).mean() / atr)
        
        # Calculate ADX
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(window=period).mean()
        
        return adx.iloc[-1] if not np.isnan(adx.iloc[-1]) else 0


class OptionsLSTMEnvironment(HistoricalOptionsEnvironment):
    """Enhanced environment with technical indicators and LSTM-compatible state"""
    
    def __init__(self, *args, window_length=50, **kwargs):
        # Initialize attributes before calling parent
        self.window_length = window_length
        self.price_history = deque(maxlen=window_length)
        self.high_history = deque(maxlen=window_length)
        self.low_history = deque(maxlen=window_length)
        self.volume_history = deque(maxlen=window_length)
        
        # Define action space for options trading
        self.actions = [
            'hold',
            'buy_call_atm',
            'buy_call_otm',
            'buy_put_atm', 
            'buy_put_otm',
            'sell_call_atm',
            'sell_call_otm',
            'sell_put_atm',
            'sell_put_otm',
            'close_all'
        ]
        
        # Call parent constructor
        super().__init__(*args, **kwargs)
        
        # Override action space
        self.action_space.n = len(self.actions)
        
        # Track positions and P&L
        self.positions = []
        self.cash_balance = self.initial_capital
        self.total_value = self.initial_capital
        
    def reset(self):
        """Reset environment and initialize price history"""
        # Call parent reset but don't use its observation
        super().reset()
        
        # Initialize price history with current data
        self.price_history.clear()
        self.high_history.clear()
        self.low_history.clear()
        self.volume_history.clear()
        
        # Fill history with initial data
        if hasattr(self, 'training_data') and not self.training_data.empty:
            start_idx = max(0, self.current_step - self.window_length)
            for i in range(start_idx, self.current_step):
                if i < len(self.training_data):
                    row = self.training_data.iloc[i]
                    self.price_history.append(row.get('underlying_price', 100))
                    self.high_history.append(row.get('underlying_price', 100) * 1.01)
                    self.low_history.append(row.get('underlying_price', 100) * 0.99)
                    self.volume_history.append(row.get('volume', 1000000))
        
        # Reset trading state
        self.positions = []
        self.cash_balance = self.initial_capital
        self.total_value = self.initial_capital
        
        return self._get_lstm_state()
    
    def _get_observation(self):
        """Override parent's observation method to return LSTM state"""
        return self._get_lstm_state()
    
    def _get_lstm_state(self):
        """Get state representation suitable for LSTM processing"""
        if len(self.price_history) < 2:
            # Return zero state if not enough history
            return np.zeros((self.window_length, 8), dtype=np.float32)
        
        # Calculate technical indicators
        prices = list(self.price_history)
        highs = list(self.high_history)
        lows = list(self.low_history)
        volumes = list(self.volume_history)
        
        # Calculate indicators
        macd, signal, histogram = TechnicalIndicators.calculate_macd(prices)
        rsi = TechnicalIndicators.calculate_rsi(prices)
        cci = TechnicalIndicators.calculate_cci(highs, lows, prices)
        adx = TechnicalIndicators.calculate_adx(highs, lows, prices)
        
        # Normalize values
        current_price = prices[-1] if prices else 100
        price_normalized = current_price / 1000  # Normalize to ~0-1 range
        balance_normalized = self.cash_balance / self.initial_capital
        positions_normalized = len(self.positions) / 10  # Assume max 10 positions
        
        # Create time series features
        time_series_features = []
        
        for i in range(len(prices)):
            if i == 0:
                price_change = 0
                volume_change = 0
            else:
                price_change = (prices[i] - prices[i-1]) / prices[i-1]
                volume_change = (volumes[i] - volumes[i-1]) / volumes[i-1] if volumes[i-1] > 0 else 0
            
            # Feature vector for each time step
            features = [
                prices[i] / 1000,  # Normalized price
                price_change,      # Price change
                volumes[i] / 1e6,  # Normalized volume (in millions)
                volume_change,     # Volume change
                macd / 100 if i == len(prices)-1 else 0,  # MACD (only latest)
                rsi / 100 if i == len(prices)-1 else 0,   # RSI (normalized to 0-1)
                cci / 200 if i == len(prices)-1 else 0,   # CCI (normalized)
                adx / 100 if i == len(prices)-1 else 0    # ADX (normalized to 0-1)
            ]
            
            time_series_features.append(features)
        
        # Pad if necessary
        while len(time_series_features) < self.window_length:
            time_series_features.insert(0, [0] * 8)
        
        # Keep only window_length most recent
        if len(time_series_features) > self.window_length:
            time_series_features = time_series_features[-self.window_length:]
        
        return np.array(time_series_features, dtype=np.float32)
    
    def step(self, action):
        """Execute trading action and return LSTM-compatible state"""
        if self.done:
            return self._get_lstm_state(), 0, True, {}
        
        # Get current market data
        current_data = self.training_data.iloc[self.current_step]
        current_price = current_data.get('underlying_price', 100)
        
        # Update price history
        self.price_history.append(current_price)
        self.high_history.append(current_price * 1.01)  # Simplified
        self.low_history.append(current_price * 0.99)   # Simplified
        self.volume_history.append(current_data.get('volume', 1000000))
        
        # Execute action
        action_name = self.actions[action]
        reward = 0
        
        if action_name == 'hold':
            pass
        elif action_name == 'close_all':
            reward = self._close_all_positions(current_data)
        elif 'buy' in action_name or 'sell' in action_name:
            reward = self._execute_option_trade(action_name, current_data)
        
        # Update positions
        self._update_positions(current_data)
        
        # Calculate portfolio value
        self.total_value = self._calculate_portfolio_value(current_data)
        
        # Calculate reward based on portfolio change
        value_change = (self.total_value - self.initial_capital) / self.initial_capital
        # Clip reward to reasonable bounds
        reward += np.clip(value_change * 10, -10, 10)  # Scale and clip reward
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        if self.current_step >= len(self.training_data) - 1:
            self.done = True
        elif self.total_value < self.initial_capital * 0.5:  # 50% loss limit
            self.done = True
            reward -= 10  # Penalty for large loss
        
        # Get next state
        next_state = self._get_lstm_state()
        
        info = {
            'portfolio_value': self.total_value,
            'cash_balance': self.cash_balance,
            'positions': len(self.positions),
            'action': action_name
        }
        
        return next_state, reward, self.done, info
    
    def _execute_option_trade(self, action_name, current_data):
        """Execute option buy/sell trade"""
        # Parse action
        parts = action_name.split('_')
        trade_type = parts[0]  # buy or sell
        option_type = parts[1]  # call or put
        moneyness = parts[2]    # atm or otm
        
        # Find suitable option
        current_price = current_data.get('underlying_price', 100)
        
        # Define strike based on moneyness
        if moneyness == 'atm':
            target_strike = current_price
        else:  # otm
            if option_type == 'call':
                target_strike = current_price * 1.05  # 5% OTM
            else:  # put
                target_strike = current_price * 0.95  # 5% OTM
        
        # Find closest option
        options = self.training_data[
            (self.training_data['timestamp'] == current_data['timestamp']) &
            (self.training_data['option_type'] == option_type)
        ].copy()  # Make a copy to avoid SettingWithCopyWarning
        
        if options.empty:
            return 0
        
        # Find closest strike
        options['strike_diff'] = abs(options['strike'] - target_strike)
        best_option = options.nsmallest(1, 'strike_diff').iloc[0]
        
        # Calculate trade size
        option_price = best_option['ask'] if trade_type == 'buy' else best_option['bid']
        if option_price <= 0:
            return 0
        
        # Position sizing: risk 2% of capital per trade
        risk_amount = self.cash_balance * 0.02
        contracts = max(1, int(risk_amount / (option_price * 100)))
        total_cost = contracts * option_price * 100 + self.commission
        
        if trade_type == 'buy' and total_cost <= self.cash_balance:
            # Open long position
            self.positions.append({
                'type': 'long',
                'option_type': option_type,
                'strike': best_option['strike'],
                'entry_price': option_price,
                'contracts': contracts,
                'entry_time': self.current_step
            })
            self.cash_balance -= total_cost
            return -0.1  # Small penalty for transaction cost
            
        elif trade_type == 'sell' and self.cash_balance > total_cost * 2:  # Margin requirement
            # Open short position
            self.positions.append({
                'type': 'short',
                'option_type': option_type,
                'strike': best_option['strike'],
                'entry_price': option_price,
                'contracts': contracts,
                'entry_time': self.current_step
            })
            self.cash_balance += total_cost - 2 * self.commission
            return -0.1
        
        return 0
    
    def _update_positions(self, current_data):
        """Update existing positions"""
        positions_to_close = []
        
        for i, pos in enumerate(self.positions):
            # Find current option price
            options = self.training_data[
                (self.training_data['timestamp'] == current_data['timestamp']) &
                (self.training_data['option_type'] == pos['option_type']) &
                (self.training_data['strike'] == pos['strike'])
            ]
            
            if options.empty:
                continue
            
            current_option = options.iloc[0]
            current_price = (current_option['bid'] + current_option['ask']) / 2
            
            # Calculate P&L
            if pos['type'] == 'long':
                pnl = (current_price - pos['entry_price']) * pos['contracts'] * 100
            else:  # short
                pnl = (pos['entry_price'] - current_price) * pos['contracts'] * 100
            
            pnl_pct = pnl / (pos['entry_price'] * pos['contracts'] * 100)
            
            # Exit conditions
            position_age = self.current_step - pos['entry_time']
            
            if pnl_pct >= 0.2:  # 20% profit
                positions_to_close.append(i)
                self.cash_balance += (pos['entry_price'] * pos['contracts'] * 100) + pnl - self.commission
            elif pnl_pct <= -0.1:  # 10% loss
                positions_to_close.append(i)
                self.cash_balance += (pos['entry_price'] * pos['contracts'] * 100) + pnl - self.commission
            elif position_age > 20:  # Time exit
                positions_to_close.append(i)
                self.cash_balance += (pos['entry_price'] * pos['contracts'] * 100) + pnl - self.commission
        
        # Close positions
        for i in reversed(positions_to_close):
            self.positions.pop(i)
    
    def _close_all_positions(self, current_data):
        """Close all open positions"""
        total_pnl = 0
        
        for pos in self.positions:
            # Find current option price
            options = self.training_data[
                (self.training_data['timestamp'] == current_data['timestamp']) &
                (self.training_data['option_type'] == pos['option_type']) &
                (self.training_data['strike'] == pos['strike'])
            ]
            
            if not options.empty:
                current_option = options.iloc[0]
                current_price = (current_option['bid'] + current_option['ask']) / 2
                
                if pos['type'] == 'long':
                    pnl = (current_price - pos['entry_price']) * pos['contracts'] * 100
                else:  # short
                    pnl = (pos['entry_price'] - current_price) * pos['contracts'] * 100
                
                total_pnl += pnl
                self.cash_balance += (pos['entry_price'] * pos['contracts'] * 100) + pnl - self.commission
        
        self.positions.clear()
        return total_pnl / 1000  # Normalized reward
    
    def _calculate_portfolio_value(self, current_data):
        """Calculate total portfolio value"""
        positions_value = 0
        
        for pos in self.positions:
            options = self.training_data[
                (self.training_data['timestamp'] == current_data['timestamp']) &
                (self.training_data['option_type'] == pos['option_type']) &
                (self.training_data['strike'] == pos['strike'])
            ]
            
            if not options.empty:
                current_option = options.iloc[0]
                bid = current_option.get('bid', 0)
                ask = current_option.get('ask', 0)
                
                # Handle invalid prices
                if bid <= 0 or ask <= 0:
                    current_price = pos['entry_price']  # Use entry price as fallback
                else:
                    current_price = (bid + ask) / 2
                
                if pos['type'] == 'long':
                    positions_value += current_price * pos['contracts'] * 100
                else:  # short
                    # For shorts, we need to account for the liability
                    positions_value -= (current_price - pos['entry_price']) * pos['contracts'] * 100
        
        return max(0, self.cash_balance + positions_value)  # Ensure non-negative


def create_lstm_ppo_model(state_dim, action_dim, hidden_dim=256, lstm_layers=2):
    """Create LSTM-PPO model for options trading"""
    
    class LSTMPPONetwork(nn.Module):
        def __init__(self):
            super().__init__()
            
            # LSTM for temporal feature extraction
            self.lstm = nn.LSTM(
                input_size=state_dim,
                hidden_size=hidden_dim,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=0.2 if lstm_layers > 1 else 0
            )
            
            # Shared feature extractor
            self.feature_extractor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            
            # Actor head (policy)
            self.actor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_dim),
                nn.Softmax(dim=-1)
            )
            
            # Critic head (value function)
            self.critic = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            
        def forward(self, x, hidden=None):
            # x shape: (batch, sequence_length, features)
            lstm_out, hidden = self.lstm(x, hidden)
            
            # Take the last timestep output
            last_output = lstm_out[:, -1, :]
            
            # Extract features
            features = self.feature_extractor(last_output)
            
            # Get policy and value
            policy = self.actor(features)
            value = self.critic(features)
            
            return policy, value.squeeze(-1), hidden
    
    return LSTMPPONetwork()


class PPOTrainer:
    """PPO trainer for LSTM-based options trading"""
    
    def __init__(
        self,
        env,
        model,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device='cuda'
    ):
        self.env = env
        self.model = model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Buffers for PPO
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def collect_rollout(self, num_steps=2048):
        """Collect experience for training"""
        state = self.env.reset()
        hidden = None
        
        for _ in range(num_steps):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action from policy
            with torch.no_grad():
                policy, value, hidden = self.model(state_tensor, hidden)
                dist = Categorical(policy)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            # Take action in environment
            next_state, reward, done, info = self.env.step(action.item())
            
            # Store experience
            self.states.append(state)
            self.actions.append(action.item())
            self.rewards.append(reward)
            self.values.append(value.item())
            self.log_probs.append(log_prob.item())
            self.dones.append(done)
            
            state = next_state
            
            if done:
                state = self.env.reset()
                hidden = None
    
    def compute_advantages(self):
        """Compute GAE advantages"""
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        # Bootstrap value for last state
        last_value = 0
        if not dones[-1]:
            state_tensor = torch.FloatTensor(self.states[-1]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, last_value, _ = self.model(state_tensor)
                last_value = last_value.item()
        
        # Compute advantages using GAE
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        # Compute returns
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update_policy(self, batch_size=64, epochs=10):
        """Update policy using PPO"""
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages()
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Training loop
        for _ in range(epochs):
            # Shuffle data
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                # Get batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                policy, values, _ = self.model(batch_states)
                dist = Categorical(policy)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # PPO loss
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = 0.5 * (values - batch_returns).pow(2).mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        # Clear buffers
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()


def train_lstm_ppo(
    symbols=['AAPL', 'GOOGL', 'MSFT'],
    num_episodes=1000,
    save_interval=100,
    checkpoint_dir='checkpoints/lstm_ppo'
):
    """Main training function"""
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load configuration
    config = TradingConfig()
    
    # Initialize data loader
    data_loader = HistoricalOptionsDataLoader(
        api_key=os.getenv('ALPACA_API_KEY'),
        api_secret=os.getenv('ALPACA_SECRET_KEY')
    )
    
    # Load historical data
    logger.info("Loading historical options data...")
    historical_data = asyncio.run(
        data_loader.load_historical_options_data(
            symbols=symbols,
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now() - timedelta(days=1)
        )
    )
    
    # Create environment
    env = OptionsLSTMEnvironment(
        historical_data=historical_data,
        initial_capital=100000,
        commission=0.65,
        max_positions=10,
        window_length=50
    )
    
    # Create model
    model = create_lstm_ppo_model(
        state_dim=8,  # 8 features per timestep
        action_dim=env.action_space.n,
        hidden_dim=256,
        lstm_layers=2
    )
    
    # Create trainer
    trainer = PPOTrainer(
        env=env,
        model=model,
        lr=3e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    win_rates = []
    
    # Training loop
    logger.info("Starting LSTM-PPO training...")
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        # Collect rollout
        trainer.collect_rollout(num_steps=2048)
        
        # Update policy
        trainer.update_policy(batch_size=64, epochs=10)
        
        # Evaluate performance
        if episode % 10 == 0:
            # Run evaluation episode
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            wins = 0
            total_trades = 0
            hidden = None
            
            while not env.done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(trainer.device)
                
                with torch.no_grad():
                    policy, _, hidden = model(state_tensor, hidden)
                    action = torch.argmax(policy).item()
                
                state, reward, done, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                # Track trades
                if 'trade_result' in info:
                    total_trades += 1
                    if info['trade_result'] > 0:
                        wins += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            win_rate = wins / total_trades if total_trades > 0 else 0
            win_rates.append(win_rate)
            
            logger.info(f"Episode {episode}: Reward={episode_reward:.2f}, "
                       f"Length={episode_length}, Win Rate={win_rate:.2%}")
        
        # Save checkpoint
        if episode % save_interval == 0 and episode > 0:
            checkpoint = {
                'episode': episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'episode_rewards': episode_rewards,
                'win_rates': win_rates
            }
            
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f'checkpoint_episode_{episode}.pt'
            )
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Save training metrics
            metrics = {
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'win_rates': win_rates
            }
            
            with open(os.path.join(checkpoint_dir, 'training_metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
    
    logger.info("Training complete!")
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_path)
    logger.info(f"Saved final model to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'GOOGL', 'MSFT'])
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--save-interval', type=int, default=100)
    parser.add_argument('--checkpoint-dir', default='checkpoints/lstm_ppo')
    
    args = parser.parse_args()
    
    train_lstm_ppo(
        symbols=args.symbols,
        num_episodes=args.episodes,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir
    )