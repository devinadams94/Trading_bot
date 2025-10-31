#!/usr/bin/env python3
"""Optimized training script with significant performance improvements"""

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributions import Categorical
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from datetime import datetime, timedelta
import argparse
import traceback
import asyncio
from dotenv import load_dotenv
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import deque


class TechnicalIndicators:
    """Calculate technical indicators for trading decisions"""
    
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """
        Calculate MACD (Moving Average Convergence Divergence)
        Returns: (macd_line, signal_line, histogram)
        """
        if len(prices) < slow:
            return 0.0, 0.0, 0.0
        
        prices_array = np.array(prices)
        
        # Calculate exponential moving averages
        ema_fast = TechnicalIndicators._calculate_ema(prices_array, fast)
        ema_slow = TechnicalIndicators._calculate_ema(prices_array, slow)
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line (EMA of MACD)
        signal_line = TechnicalIndicators._calculate_ema(macd_line, signal)
        
        # MACD histogram
        histogram = macd_line - signal_line
        
        return float(macd_line[-1]), float(signal_line[-1]), float(histogram[-1])
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        """
        Calculate RSI (Relative Strength Index)
        Returns: RSI value (0-100)
        """
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        prices_array = np.array(prices)
        deltas = np.diff(prices_array)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0  # Maximum RSI
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    @staticmethod
    def calculate_cci(prices, highs=None, lows=None, period=20):
        """
        Calculate CCI (Commodity Channel Index)
        Returns: CCI value (typically -100 to +100)
        """
        if len(prices) < period:
            return 0.0
        
        # Use prices for high/low if not provided
        if highs is None:
            highs = prices
        if lows is None:
            lows = prices
        
        # Calculate typical price
        typical_price = (np.array(highs) + np.array(lows) + np.array(prices)) / 3
        
        # Moving average of typical price
        sma = np.mean(typical_price[-period:])
        
        # Mean deviation
        mean_dev = np.mean(np.abs(typical_price[-period:] - sma))
        
        if mean_dev == 0:
            return 0.0
        
        # CCI calculation
        cci = (typical_price[-1] - sma) / (0.015 * mean_dev)
        
        return float(cci)
    
    @staticmethod
    def calculate_adx(prices, highs=None, lows=None, period=14):
        """
        Calculate ADX (Average Directional Index)
        Returns: ADX value (0-100, trend strength)
        """
        if len(prices) < period * 2:
            return 0.0
        
        # Use prices for high/low if not provided
        if highs is None:
            highs = [p * 1.001 for p in prices]  # Approximate
        if lows is None:
            lows = [p * 0.999 for p in prices]  # Approximate
        
        highs = np.array(highs)
        lows = np.array(lows)
        closes = np.array(prices)
        
        # Calculate True Range
        high_low = highs - lows
        high_close = np.abs(highs[1:] - closes[:-1])
        low_close = np.abs(lows[1:] - closes[:-1])
        
        # Pad to maintain array size
        high_close = np.concatenate([[0], high_close])
        low_close = np.concatenate([[0], low_close])
        
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        
        # Calculate directional movements
        up_move = highs[1:] - highs[:-1]
        down_move = lows[:-1] - lows[1:]
        
        # Pad arrays
        up_move = np.concatenate([[0], up_move])
        down_move = np.concatenate([[0], down_move])
        
        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Calculate smoothed averages
        atr = TechnicalIndicators._calculate_ema(tr, period)
        pos_di = 100 * TechnicalIndicators._calculate_ema(pos_dm, period) / atr
        neg_di = 100 * TechnicalIndicators._calculate_ema(neg_dm, period) / atr
        
        # Calculate DX
        di_diff = np.abs(pos_di - neg_di)
        di_sum = pos_di + neg_di
        dx = np.where(di_sum > 0, 100 * di_diff / di_sum, 0)
        
        # Calculate ADX
        adx = TechnicalIndicators._calculate_ema(dx, period)
        
        return float(adx[-1])
    
    @staticmethod
    def _calculate_ema(data, period):
        """Helper function to calculate Exponential Moving Average"""
        if len(data) < period:
            return np.array([np.mean(data[:i+1]) for i in range(len(data))])
        
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    @staticmethod
    def calculate_all_indicators(price_history, high_history=None, low_history=None):
        """Calculate all technical indicators at once"""
        if len(price_history) < 26:  # Minimum for MACD
            return {
                'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0,
                'rsi': 50.0, 'cci': 0.0, 'adx': 0.0
            }
        
        prices = list(price_history)
        
        # Calculate indicators
        macd, signal, histogram = TechnicalIndicators.calculate_macd(prices)
        rsi = TechnicalIndicators.calculate_rsi(prices)
        cci = TechnicalIndicators.calculate_cci(prices, high_history, low_history)
        adx = TechnicalIndicators.calculate_adx(prices, high_history, low_history)
        
        return {
            'macd': macd,
            'macd_signal': signal,
            'macd_histogram': histogram,
            'rsi': rsi,
            'cci': cci,
            'adx': adx
        }


import signal
import threading

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Don't set CUDA_VISIBLE_DEVICES for distributed training

from src.options_clstm_ppo import OptionsCLSTMPPOAgent
from src.options_trading_env import OptionsTradingEnvironment
from src.historical_options_data import HistoricalOptionsDataLoader, HistoricalOptionsEnvironment
from config.config import TradingConfig
from config.symbols_loader import SymbolsConfig

import multiprocessing as cpu_mp
from typing import List, Tuple, Any

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = threading.Event()


class UltraFastEnvironment(HistoricalOptionsEnvironment):
    """Ultra-fast environment with minimal overhead"""
    
    def __init__(self, *args, **kwargs):
        # Initialize attributes BEFORE calling super().__init__
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0
        self.consecutive_losses = 0
        self.price_history = []
        self._cache = {}  # Initialize cache BEFORE super().__init__
        self._last_action = 'hold'  # Initialize last action to prevent NameError
        
        # Call parent
        super().__init__(*args, **kwargs)
        
        # Ensure these attributes persist after parent init
        if not hasattr(self, 'winning_trades'):
            self.winning_trades = 0
        if not hasattr(self, 'losing_trades'):
            self.losing_trades = 0
        
    def reset(self):
        """Fast reset without precomputation"""
        obs = super().reset()
        
        # Reset tracking
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0
        self.consecutive_losses = 0
        self.price_history = []
        self._cache.clear()
        self._last_action = 'hold'  # Reset last action
        self.last_trade_reward = 0  # Reset trade reward
        
        # Convert training data to numpy once for fast access
        if hasattr(self, 'training_data') and self.training_data is not None:
            df = self.training_data
            self._prices = df['underlying_price'].values
            self._timestamps = df['timestamp'].values
            
            # Pre-filter options data
            self._option_data = {
                'strikes': df['strike'].values,
                'types': df['option_type'].values,
                'bids': df['bid'].values,
                'asks': df['ask'].values,
                'volumes': df['volume'].fillna(0).values,
                'moneyness': df['moneyness'].fillna(1.0).values,
                'timestamps': df['timestamp'].values
            }
            self._data_length = len(df)
        
        return obs
    
    def step(self, action: int):
        """Ultra-fast step with minimal overhead"""
        if self.done or self.current_step >= self._data_length:
            self.done = True
            return self._get_observation(), 0, True, {}
        
        # Fast action mapping
        actions = ['hold', 'buy_call', 'buy_put', 'sell_call', 'sell_put', 
                  'bull_call_spread', 'bear_put_spread', 'iron_condor', 
                  'straddle', 'strangle', 'close_all_positions']
        action_name = actions[action] if action < len(actions) else 'hold'
        
        # Get current price
        current_price = self._prices[self.current_step]
        self.price_history.append(current_price)
        if len(self.price_history) > 20:
            self.price_history.pop(0)
        
        # Simple portfolio value
        portfolio_value = self.capital + len(self.positions) * 1000  # Simplified
        
        # Execute action
        reward = 0
        
        if action_name == 'hold':
            pass
        elif action_name == 'close_all_positions' and self.positions:
            # Fast close all
            for pos in self.positions:
                pnl = np.random.uniform(-1000, 2000)  # Simplified P&L
                self.capital += pos['quantity'] * 100 * pos['entry_price'] + pnl
                if pnl > 0:
                    self.winning_trades += 1
                    # Add exponential reward for closing profitable position
                    self.last_trade_reward = self._calculate_trade_reward(pnl, pnl_pct, base_reward=5.0)
                else:
                    self.losing_trades += 1
                    # Penalty for losing trade
                    self.last_trade_reward = self._calculate_trade_reward(pnl, pnl_pct, base_reward=2.0)
            self.positions = []
            reward = 10 if self.winning_trades > self.losing_trades else -5
        elif action_name in ['buy_call', 'buy_put'] and len(self.positions) < self.max_positions:
            # Fast option selection - just pick first valid one
            current_time = self._timestamps[self.current_step]
            
            # Find options at current timestamp (cached)
            cache_key = (self.current_step, action_name)
            if cache_key not in self._cache:
                # Find matching timestamp indices
                time_mask = self._option_data['timestamps'] == current_time
                type_mask = self._option_data['types'] == ('call' if 'call' in action_name else 'put')
                money_mask = (self._option_data['moneyness'] >= 0.9) & (self._option_data['moneyness'] <= 1.1)
                bid_mask = self._option_data['bids'] > 0
                
                valid_mask = time_mask & type_mask & money_mask & bid_mask
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_indices) > 0:
                    # Pick the one with highest volume
                    volumes = self._option_data['volumes'][valid_indices]
                    best_idx = valid_indices[np.argmax(volumes)]
                    
                    option = {
                        'strike': self._option_data['strikes'][best_idx],
                        'bid': self._option_data['bids'][best_idx],
                        'ask': self._option_data['asks'][best_idx],
                        'volume': self._option_data['volumes'][best_idx]
                    }
                    self._cache[cache_key] = option
                else:
                    self._cache[cache_key] = None
            
            option = self._cache[cache_key]
            
            if option is not None:
                # Fast position entry
                mid_price = (option['bid'] + option['ask']) / 2
                quantity = min(5, int(self.capital * 0.1 / (mid_price * 100)))
                
                if quantity > 0:
                    cost = quantity * mid_price * 100 + 0.65
                    if cost <= self.capital * 0.3:
                        self.positions.append({
                            'entry_price': mid_price,
                            'quantity': quantity,
                            'entry_step': self.current_step,
                            'option_type': 'call' if 'call' in action_name else 'put',
                            'strike': option['strike']
                        })
                        self.capital -= cost
                        reward = 0.1
        
        # Fast position updates
        positions_to_close = []
        for i, pos in enumerate(self.positions):
            age = self.current_step - pos['entry_step']
            # Simple exit rules
            if age > 10 or np.random.random() < 0.1:  # 10% chance to exit
                pnl = np.random.uniform(-500, 1000) * pos['quantity']
                self.capital += pos['quantity'] * pos['entry_price'] * 100 + pnl
                
                if pnl > 0:
                    self.winning_trades += 1
                    reward += 5
                else:
                    self.losing_trades += 1
                    self.consecutive_losses += 1
                    reward -= 2
                
                positions_to_close.append(i)
        
        # Remove closed positions
        for i in reversed(positions_to_close):
            self.positions.pop(i)
        
        # Add any trade closing rewards
        if hasattr(self, 'last_trade_reward'):
            reward += self.last_trade_reward
            self.last_trade_reward = 0
        
        # Update step
        self.current_step += 1
        
        # Done conditions
        if self.current_step >= self._data_length - 1:
            self.done = True
        elif self.capital < 20000:  # 80% loss
            self.done = True
            reward -= 50
        
        # Final reward based on portfolio change
        new_portfolio = self.capital + len(self.positions) * 1000
        reward += (new_portfolio - portfolio_value) / 1000
        
        return self._get_observation(), reward, self.done, {
            'portfolio_value': new_portfolio,
            'positions': len(self.positions),
            'symbol': self.current_symbol
        }


class BalancedEnvironment(HistoricalOptionsEnvironment):
    """Balanced environment with realistic features and good performance"""
    
    def __init__(self, *args, **kwargs):
        # Initialize attributes BEFORE calling super().__init__
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0
        self.consecutive_losses = 0
        self.price_history = []
        self._cache = {}
        self._position_cache = {}
        
        # Realistic trading parameters with enhanced market awareness
        self.max_loss_per_trade = 0.015  # 1.5% max loss - tighter for better risk management
        self.max_profit_per_trade = 0.04  # 4% take profit - more realistic
        self.volatility_window = 20
        self.use_dynamic_exits = True  # Enable dynamic exit strategies
        self._last_action = 'hold'  # Initialize last action to prevent NameError
        
        # Market regime tracking
        self._market_regime_history = []
        self._regime_confidence = 0.0
        
        # Call parent
        
        # Price history for technical indicators
        self.price_history_window = 50  # Keep last 50 prices
        self.underlying_price_history = deque(maxlen=self.price_history_window)
        self.high_price_history = deque(maxlen=self.price_history_window)
        self.low_price_history = deque(maxlen=self.price_history_window)
        
        super().__init__(*args, **kwargs)
        
        # Ensure these attributes exist after parent init
        if not hasattr(self, 'winning_trades'):
            self.winning_trades = 0
        if not hasattr(self, 'losing_trades'):
            self.losing_trades = 0
        
    def reset(self):
        """Fast reset with minimal precomputation"""
        obs = super().reset()
        
        # Reset tracking
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0
        self.consecutive_losses = 0
        self.price_history = []
        self._cache.clear()
        self._position_cache.clear()
        self._last_action = 'hold'  # Reset last action
        self.last_trade_reward = 0  # Reset trade reward
        
        # Convert to numpy for fast access
        if hasattr(self, 'training_data') and self.training_data is not None:
            df = self.training_data
            self._data_length = len(df)
            
            # Essential data as numpy arrays
            self._prices = df['underlying_price'].values
            self._timestamps = df['timestamp'].values
            
            # Group by timestamp for fast option lookup
            self._timestamp_groups = df.groupby('timestamp').groups
            
            # Pre-compute option data arrays
            self._option_data = {
                'strikes': df['strike'].values,
                'types': df['option_type'].values,
                'bids': df['bid'].values,
                'asks': df['ask'].values,
                'volumes': df['volume'].fillna(0).values,
                'moneyness': df['moneyness'].fillna(1.0).values,
                'timestamps': df['timestamp'].values,
                'indices': np.arange(len(df))
            }
        
        return obs
    
    def step(self, action: int):
        """Balanced step with realistic features"""
        if self.done or self.current_step >= self._data_length:
            self.done = True
            return self._get_observation(), 0, True, {}
        
        # Action mapping
        actions = ['hold', 'buy_call', 'buy_put', 'sell_call', 'sell_put', 
                  'bull_call_spread', 'bear_put_spread', 'iron_condor', 
                  'straddle', 'strangle', 'close_all_positions']
        action_name = actions[action] if action < len(actions) else 'hold'
        
        # Track current price
        current_price = self._prices[self.current_step]
        self.price_history.append(current_price)
        if len(self.price_history) > self.volatility_window:
            self.price_history.pop(0)
        
        # Calculate realistic portfolio value
        portfolio_value_before = self._calculate_portfolio_value_fast()
        
        # Execute action
        reward = 0
        
        if action_name == 'hold':
            pass
        elif action_name == 'close_all_positions' and self.positions:
            reward = self._close_all_positions_realistic()
        elif action_name in ['buy_call', 'buy_put'] and len(self.positions) < self.max_positions:
            reward = self._execute_trade(action_name, current_price)
        
        # Update positions with realistic P&L
        self._update_positions_realistic()
        
        # Calculate reward based on portfolio change
        portfolio_value_after = self._calculate_portfolio_value_fast()
        step_pnl = portfolio_value_after - portfolio_value_before
        
        # Advanced reward shaping for better learning
        reward += step_pnl / 1000  # Normalized P&L
        
        # Win rate bonus (progressive) - encourage high win rates
        if self.winning_trades + self.losing_trades >= 5:  # Only after enough trades
            win_rate = self.winning_trades / max(1, self.winning_trades + self.losing_trades)
            if win_rate > 0.7:
                reward += 15  # Increased bonus for excellent win rate
            elif win_rate > 0.6:
                reward += 8
            elif win_rate > 0.5:
                reward += 4
            elif win_rate < 0.3:
                reward -= 5  # Penalty for poor win rate
        
        # Profit factor bonus - reward good risk/reward ratio
        if self.winning_trades > 0 and self.losing_trades > 0:
            # Calculate actual profit factor
            total_wins = 0
            total_losses = 0
            # This is simplified - in real implementation would track actual P&L per trade
            if self.total_pnl > 0:
                avg_win_size = self.total_pnl / self.winning_trades
                avg_loss_size = abs(self.total_pnl) / max(1, self.losing_trades)
                profit_factor = avg_win_size / max(0.01, avg_loss_size)
                
                if profit_factor > 2.0:
                    reward += 10  # Excellent profit factor
                elif profit_factor > 1.5:
                    reward += 5
                elif profit_factor < 0.8:
                    reward -= 3  # Poor profit factor
        
        # Risk management - heavily penalize consecutive losses
        if self.consecutive_losses > 5:
            reward -= self.consecutive_losses * 3  # Increased penalty
        elif self.consecutive_losses > 3:
            reward -= self.consecutive_losses * 2
        elif self.consecutive_losses == 0 and self.winning_trades > 5:
            reward += 5  # Increased bonus for consistent winning
        
        # Position management bonus with market consideration
        market_regime = self._detect_market_regime()
        if len(self.positions) > 0:
            if market_regime == 'trending':
                # Encourage holding positions in trends
                if len(self.positions) < self.max_positions - 1:
                    reward += 1.0
            elif market_regime == 'volatile':
                # Encourage smaller position sizes in volatile markets
                if len(self.positions) <= 2:
                    reward += 0.5
                else:
                    reward -= 0.5  # Penalty for too many positions in volatile market
        
        # Sharpe ratio component - reward risk-adjusted returns
        if step_pnl != 0:
            current_vol = self._calculate_volatility()
            if current_vol > 0:
                sharpe_component = step_pnl / (current_vol * 1000)
                reward += sharpe_component * 2  # Increased weight
        
        # Market timing bonus - reward good entries
        # Check if we just executed a trade action this step
        if hasattr(self, '_last_action') and self._last_action in ['buy_call', 'buy_put'] and step_pnl > 0:
            # Good entry that resulted in immediate profit
            reward += 2
            
        # Store the action for next step reference
        self._last_action = action_name
        
        # Add any trade closing rewards
        if hasattr(self, 'last_trade_reward'):
            reward += self.last_trade_reward
            self.last_trade_reward = 0
        
        # Update step
        self.current_step += 1
        
        # Done conditions
        if self.current_step >= self._data_length - 1:
            self.done = True
        elif portfolio_value_after < self.initial_capital * 0.5:  # 50% loss
            self.done = True
            reward -= 100
        
        return self._get_observation(), reward, self.done, {
            'portfolio_value': portfolio_value_after,
            'positions': len(self.positions),
            'symbol': self.current_symbol,
            'win_rate': self.winning_trades / max(1, self.winning_trades + self.losing_trades)
        }
    
    def _execute_trade(self, action_name, current_price):
        """Execute trade with realistic option selection and better timing"""
        current_time = self._timestamps[self.current_step]
        
        # Calculate market conditions
        momentum = self._calculate_momentum()
        rsi = self._calculate_rsi()
        volatility = self._calculate_volatility()
        market_regime = self._detect_market_regime()
        
        # Enhanced entry conditions with market regime consideration
        if action_name == 'buy_call':
            # Market regime-specific penalties/rewards
            if market_regime == 'volatile':
                return -0.3  # Avoid entering in volatile markets
            elif market_regime == 'ranging':
                return -0.2  # Calls less effective in ranging markets
            elif market_regime == 'trending' and momentum < 0:
                return -0.5  # Don't buy calls in downtrend
                
            # Stricter entry conditions for calls
            if momentum < -0.01:  # Negative momentum
                return -0.5  # Larger penalty for trading against trend
            elif rsi > 70:  # Overbought
                return -0.3  # Penalty for buying at extreme levels
            elif volatility > 0.04:  # Very high volatility
                return -0.2  # Penalty for entering in uncertain conditions
            elif momentum < 0.005 and rsi > 60:  # Weak momentum and high RSI
                return -0.1  # Small penalty for suboptimal timing
            
            # Positive conditions for calls
            if market_regime == 'trending' and momentum > 0.01 and 40 <= rsi <= 60:
                return 0.2  # Bonus for ideal conditions
                
        elif action_name == 'buy_put':
            # Market regime-specific penalties/rewards
            if market_regime == 'volatile':
                return -0.3  # Avoid entering in volatile markets
            elif market_regime == 'ranging':
                return -0.2  # Puts less effective in ranging markets
            elif market_regime == 'trending' and momentum > 0:
                return -0.5  # Don't buy puts in uptrend
                
            # Stricter entry conditions for puts
            if momentum > 0.01:  # Positive momentum
                return -0.5  # Larger penalty for trading against trend
            elif rsi < 30:  # Oversold
                return -0.3  # Penalty for buying at extreme levels
            elif volatility > 0.04:  # Very high volatility
                return -0.2  # Penalty for entering in uncertain conditions
            elif momentum > -0.005 and rsi < 40:  # Weak momentum and low RSI
                return -0.1  # Small penalty for suboptimal timing
            
            # Positive conditions for puts
            if market_regime == 'trending' and momentum < -0.01 and 40 <= rsi <= 60:
                return 0.2  # Bonus for ideal conditions
        
        # Check cache first
        cache_key = (self.current_step, action_name)
        if cache_key in self._cache:
            suitable_options = self._cache[cache_key]
        else:
            # Find options at current timestamp
            if current_time in self._timestamp_groups:
                indices = self._timestamp_groups[current_time]
                
                # Filter options efficiently
                option_type = 'call' if 'call' in action_name else 'put'
                
                # Use boolean indexing on pre-computed arrays
                mask = np.zeros(len(indices), dtype=bool)
                for i, idx in enumerate(indices):
                    if (self._option_data['types'][idx] == option_type and
                        0.9 <= self._option_data['moneyness'][idx] <= 1.1 and
                        self._option_data['bids'][idx] > 0 and
                        self._option_data['asks'][idx] > 0):
                        mask[i] = True
                
                valid_indices = np.array(list(indices))[mask]
                
                if len(valid_indices) > 0:
                    # Score by volume and spread
                    volumes = self._option_data['volumes'][valid_indices]
                    spreads = (self._option_data['asks'][valid_indices] - 
                              self._option_data['bids'][valid_indices])
                    
                    # Debug log occasionally
                    if self.current_step % 100 == 0:
                        logger.debug(f"Found {len(valid_indices)} {option_type} options at step {self.current_step}")
                    spread_pcts = spreads / self._option_data['asks'][valid_indices]
                    
                    # Simple scoring: high volume, low spread
                    scores = volumes / (1 + spread_pcts * 10)
                    best_idx = valid_indices[np.argmax(scores)]
                    
                    suitable_options = [{
                        'idx': best_idx,
                        'strike': self._option_data['strikes'][best_idx],
                        'bid': self._option_data['bids'][best_idx],
                        'ask': self._option_data['asks'][best_idx],
                        'volume': self._option_data['volumes'][best_idx]
                    }]
                else:
                    suitable_options = []
            else:
                suitable_options = []
            
            self._cache[cache_key] = suitable_options
        
        if suitable_options:
            option = suitable_options[0]
            
            # Realistic position sizing
            mid_price = (option['bid'] + option['ask']) / 2
            spread_cost = (option['ask'] - option['bid']) / 2
            
            # Advanced position sizing based on market conditions
            momentum_abs = abs(momentum)
            
            # Base risk: 5-15% of capital depending on conditions
            base_risk_pct = 0.1
            
            # Adjust based on volatility (lower size in high volatility)
            if volatility > 0.03:  # High volatility
                base_risk_pct *= 0.6
            elif volatility < 0.015:  # Low volatility  
                base_risk_pct *= 1.3
            
            # Adjust based on momentum strength
            if momentum_abs > 0.02:  # Strong trend
                base_risk_pct *= 1.2
            elif momentum_abs < 0.005:  # No clear trend
                base_risk_pct *= 0.8
            
            # Adjust based on RSI
            if 45 <= rsi <= 55:  # Neutral RSI, good for entry
                base_risk_pct *= 1.1
            elif rsi > 70 or rsi < 30:  # Extreme RSI
                base_risk_pct *= 0.7
            
            max_risk = self.capital * min(0.15, max(0.05, base_risk_pct))
            
            cost_per_contract = (mid_price + spread_cost) * 100 + self.commission
            ideal_contracts = int(max_risk / cost_per_contract)
            contracts = max(1, min(ideal_contracts, 10))
            
            total_cost = contracts * cost_per_contract
            
            if total_cost <= self.capital * 0.3:
                # Open position at ask price (realistic entry)
                self.positions.append({
                    'option_idx': option['idx'],
                    'entry_price': option['ask'],
                    'quantity': contracts,
                    'entry_step': self.current_step,
                    'option_type': 'call' if 'call' in action_name else 'put',
                    'strike': option['strike'],
                    'entry_underlying': current_price,
                    'peak_value': 0
                })
                self.capital -= total_cost
                
                # Debug log occasionally
                if self.current_step % 50 == 0:
                    logger.debug(f"Opened {action_name} position: {contracts} contracts at ${option['ask']:.2f}")
                
                # Small reward for entering position
                return 0.5
        
        return 0
    
    def _update_positions_realistic(self):
        """Update positions with realistic P&L tracking"""
        if self.current_step >= self._data_length:
            return
        
        current_time = self._timestamps[self.current_step]
        current_price = self._prices[self.current_step]
        positions_to_close = []
        
        for i, pos in enumerate(self.positions):
            # Simplified but realistic P&L based on underlying price movement
            position_age = self.current_step - pos['entry_step']
            entry_underlying = pos['entry_underlying']
            
            # Calculate price movement
            price_change = (current_price - entry_underlying) / entry_underlying
            
            # Option P&L estimation based on delta
            # Calls profit from price increase, puts from price decrease
            if pos['option_type'] == 'call':
                # Approximate delta with gamma effect (higher for profitable moves)
                if price_change > 0:
                    option_price_change = price_change * 1.5  # Leverage on profitable side
                else:
                    option_price_change = price_change * 0.8  # Less loss on downside
            else:  # put
                if price_change < 0:
                    option_price_change = -price_change * 1.5  # Leverage on profitable side
                else:
                    option_price_change = -price_change * 0.8  # Less loss on upside
            
            # Calculate P&L
            entry_cost = pos['entry_price'] * pos['quantity'] * 100
            current_value = pos['entry_price'] * (1 + option_price_change) * pos['quantity'] * 100
            
            # Ensure option value doesn't go negative
            current_value = max(0, current_value)
            
            pnl = current_value - entry_cost - self.commission
            pnl_pct = pnl / entry_cost
            
            # Track peak for trailing stop
            pos['peak_value'] = max(pos.get('peak_value', 0), current_value)
            
            # Exit conditions with dynamic adjustments
            should_exit = False
            
            # Calculate current market conditions
            current_momentum = self._calculate_momentum()
            current_rsi = self._calculate_rsi()
            
            # Enhanced dynamic stop loss based on multiple factors
            volatility = self._calculate_volatility()
            market_regime = self._detect_market_regime()
            
            # Base stop loss adjusted for volatility and market regime
            if volatility > 0.03:  # High volatility
                vol_multiplier = 0.6  # Much tighter stops in high volatility
            elif volatility < 0.015:  # Low volatility
                vol_multiplier = 1.3  # Wider stops in stable markets
            else:
                vol_multiplier = 1.0
            
            # Market regime adjustment for stop loss
            if market_regime == 'volatile':
                regime_multiplier = 0.7  # Tighter stops in volatile regime
            elif market_regime == 'trending':
                # Check if we're trading with or against the trend
                if (pos['option_type'] == 'call' and current_momentum > 0) or \
                   (pos['option_type'] == 'put' and current_momentum < 0):
                    regime_multiplier = 1.2  # Wider stops when with trend
                else:
                    regime_multiplier = 0.8  # Tighter stops when against trend
            else:
                regime_multiplier = 1.0
            
            # Adjust stop loss based on position age and performance
            age_multiplier = 1.0
            if position_age > 10 and pnl_pct > 0.05:  # Profitable position held for a while
                age_multiplier = 1.3  # Give it more room
            elif position_age < 3 and pnl_pct < -0.05:  # Quick loss
                age_multiplier = 0.8  # Tighter stop to prevent larger losses
            
            dynamic_stop_loss = -self.max_loss_per_trade * vol_multiplier * age_multiplier * regime_multiplier
            
            # Enhanced dynamic take profit based on market conditions
            # Consider momentum strength, RSI, and volatility
            base_take_profit = self.max_profit_per_trade
            
            # Momentum-based adjustment
            momentum_abs = abs(current_momentum)
            if momentum_abs > 0.02:  # Strong trend
                momentum_multiplier = 1.8
            elif momentum_abs > 0.01:  # Moderate trend
                momentum_multiplier = 1.4
            else:  # Weak trend
                momentum_multiplier = 1.0
            
            # RSI-based adjustment (take profits earlier at extremes)
            if current_rsi > 70 or current_rsi < 30:
                rsi_multiplier = 0.8  # Take profits quicker at extremes
            elif 45 <= current_rsi <= 55:
                rsi_multiplier = 1.2  # Let profits run in neutral conditions
            else:
                rsi_multiplier = 1.0
            
            # Direction-specific adjustments
            if (pos['option_type'] == 'call' and current_momentum > 0) or \
               (pos['option_type'] == 'put' and current_momentum < 0):
                # Trading with the trend
                dynamic_take_profit = base_take_profit * momentum_multiplier * rsi_multiplier
            else:
                # Trading against the trend - be more conservative
                dynamic_take_profit = base_take_profit * 0.8 * rsi_multiplier
            
            # Stop loss
            if pnl_pct <= dynamic_stop_loss:
                should_exit = True
                self.losing_trades += 1
                self.consecutive_losses += 1
            
            # Take profit
            elif pnl_pct >= dynamic_take_profit:
                should_exit = True
                self.winning_trades += 1
                self.consecutive_losses = 0
            
            # Momentum-based exit (exit if momentum reverses)
            elif position_age > 5:
                if pos['option_type'] == 'call' and current_momentum < -0.01:
                    should_exit = True  # Exit calls on negative momentum
                elif pos['option_type'] == 'put' and current_momentum > 0.01:
                    should_exit = True  # Exit puts on positive momentum
                
                if should_exit:
                    if pnl > 0:
                        self.winning_trades += 1
                        self.consecutive_losses = 0
                    else:
                        self.losing_trades += 1
                        self.consecutive_losses += 1
            
            # Trailing stop (dynamic based on volatility)
            elif pos['peak_value'] > 0:
                trailing_stop_pct = 0.8 if volatility < 0.02 else 0.85  # Tighter stop in low volatility
                if current_value < pos['peak_value'] * trailing_stop_pct:
                    should_exit = True
                    if pnl > 0:
                        self.winning_trades += 1
                        self.consecutive_losses = 0
                    else:
                        self.losing_trades += 1
                        self.consecutive_losses += 1
            
            # Time exit with RSI consideration
            elif position_age > 15:
                # Exit earlier if RSI is extreme
                if (pos['option_type'] == 'call' and current_rsi > 70) or \
                   (pos['option_type'] == 'put' and current_rsi < 30):
                    should_exit = True
                elif position_age > 20:
                    should_exit = True
                
                if should_exit:
                    if pnl > 0:
                        self.winning_trades += 1
                        self.consecutive_losses = 0
                    else:
                        self.losing_trades += 1
                        self.consecutive_losses += 1
            
            if should_exit:
                positions_to_close.append(i)
                self.capital += current_value - self.commission
                self.total_pnl += pnl
        
        # Remove closed positions
        for i in reversed(positions_to_close):
            self.positions.pop(i)
    
    def _close_all_positions_realistic(self):
        """Close all positions with realistic pricing"""
        total_reward = 0
        current_price = self._prices[self.current_step]
        
        for pos in self.positions:
            # Use same simplified P&L calculation as update method
            entry_underlying = pos['entry_underlying']
            price_change = (current_price - entry_underlying) / entry_underlying
            
            if pos['option_type'] == 'call':
                # Approximate delta with gamma effect (higher for profitable moves)
                if price_change > 0:
                    option_price_change = price_change * 1.5  # Leverage on profitable side
                else:
                    option_price_change = price_change * 0.8  # Less loss on downside
            else:  # put
                if price_change < 0:
                    option_price_change = -price_change * 1.5  # Leverage on profitable side
                else:
                    option_price_change = -price_change * 0.8  # Less loss on upside
            
            # Calculate final value
            entry_cost = pos['entry_price'] * pos['quantity'] * 100
            current_value = pos['entry_price'] * (1 + option_price_change) * pos['quantity'] * 100
            current_value = max(0, current_value)
            
            pnl = current_value - entry_cost - self.commission
            
            self.capital += current_value - self.commission
            self.total_pnl += pnl
            
            if pnl > 0:
                self.winning_trades += 1
                total_reward += 10
            else:
                self.losing_trades += 1
                self.consecutive_losses += 1
                total_reward -= 5
        
        self.positions = []
        return total_reward
    
    def _calculate_portfolio_value_fast(self):
        """Fast portfolio value calculation"""
        position_value = 0
        
        if self.positions and self.current_step < self._data_length:
            current_time = self._timestamps[self.current_step]
            
            for pos in self.positions:
                option_idx = pos['option_idx']
                
                if (option_idx < len(self._option_data['timestamps']) and 
                    self._option_data['timestamps'][option_idx] == current_time):
                    
                    current_bid = self._option_data['bids'][option_idx]
                    if current_bid > 0:
                        position_value += current_bid * pos['quantity'] * 100
        
        return self.capital + position_value
    
    def _calculate_volatility(self):
        """Calculate simple volatility from price history"""
        if len(self.price_history) < 2:
            return 0.02
        
        returns = np.diff(self.price_history) / self.price_history[:-1]
        return np.std(returns) if len(returns) > 0 else 0.02
    
    def _calculate_momentum(self):
        """Calculate price momentum for better entry timing"""
        if len(self.price_history) < 5:
            return 0
        
        # Short-term momentum (5 periods)
        recent_return = (self.price_history[-1] - self.price_history[-5]) / self.price_history[-5]
        return recent_return
    
    def _calculate_rsi(self, period=14):
        """Calculate RSI for overbought/oversold conditions"""
        if len(self.price_history) < period + 1:
            return 50  # Neutral RSI
        
        prices = np.array(self.price_history[-period-1:])
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _detect_market_regime(self):
        """Enhanced market regime detection with confidence scoring"""
        if len(self.price_history) < 20:
            return 'unknown'
        
        prices = np.array(self.price_history[-20:])
        returns = np.diff(prices) / prices[:-1]
        
        # Calculate comprehensive metrics
        volatility = np.std(returns)
        momentum = (prices[-1] - prices[0]) / prices[0]
        
        # Moving averages
        sma_5 = np.mean(prices[-5:])
        sma_10 = np.mean(prices[-10:])
        sma_20 = np.mean(prices)
        
        # Advanced metrics
        direction_changes = np.sum(np.diff(np.sign(returns)) != 0)
        
        # Calculate Average True Range (ATR) proxy
        high_low_ranges = []
        for i in range(1, len(prices)):
            high_low_ranges.append(abs(prices[i] - prices[i-1]))
        atr = np.mean(high_low_ranges) if high_low_ranges else 0
        
        # Trend strength indicator
        consecutive_ups = 0
        consecutive_downs = 0
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                consecutive_ups = consecutive_ups + 1 if consecutive_ups >= 0 else 1
                consecutive_downs = 0
            else:
                consecutive_downs = consecutive_downs + 1 if consecutive_downs >= 0 else 1
                consecutive_ups = 0
        
        max_consecutive = max(abs(consecutive_ups), abs(consecutive_downs))
        
        # Regime scoring
        regime_scores = {'trending': 0, 'ranging': 0, 'volatile': 0}
        
        # Volatile regime indicators
        if volatility > 0.025:
            regime_scores['volatile'] += 3
        if atr > np.mean(prices) * 0.02:
            regime_scores['volatile'] += 2
        if direction_changes > 12:
            regime_scores['volatile'] += 1
        
        # Trending regime indicators
        if abs(momentum) > 0.015 and direction_changes < 10:
            regime_scores['trending'] += 3
        if (momentum > 0 and sma_5 > sma_10 > sma_20) or (momentum < 0 and sma_5 < sma_10 < sma_20):
            regime_scores['trending'] += 2
        if max_consecutive >= 7:
            regime_scores['trending'] += 2
        
        # Ranging regime indicators
        if volatility < 0.01 and abs(momentum) < 0.005:
            regime_scores['ranging'] += 3
        if abs(sma_5 - sma_20) / sma_20 < 0.003:
            regime_scores['ranging'] += 2
        if direction_changes >= 8 and direction_changes <= 12:
            regime_scores['ranging'] += 1
        
        # Determine regime with confidence
        max_score = max(regime_scores.values())
        if max_score < 3:
            regime = 'mixed'
            self._regime_confidence = 0.3
        else:
            regime = max(regime_scores, key=regime_scores.get)
            self._regime_confidence = min(1.0, max_score / 6.0)
        
        # Store regime history for trend analysis
        self._market_regime_history.append(regime)
        if len(self._market_regime_history) > 50:
            self._market_regime_history.pop(0)
        
        return regime


    def _calculate_trade_reward(self, pnl, pnl_pct, base_reward=1.0):
        """
        Simplified reward structure focused on win rate and profitability
        """
        if pnl > 0:
            # Reward wins proportionally
            reward = 10 * pnl_pct  # 5% win = 0.5 reward
        else:
            # Smaller penalty for losses to encourage learning
            reward = 5 * pnl_pct   # 5% loss = -0.25 reward
        
        return reward * base_reward
    
    def _calculate_win_rate_bonus(self, episode_num):
        """Calculate bonus based on current win rate"""
        if self.winning_trades + self.losing_trades < 5:
            return 0  # Not enough trades
            
        win_rate = self.winning_trades / (self.winning_trades + self.losing_trades)
        
        # Progressive win rate bonuses
        if episode_num % 100 == 0:
            if win_rate > 0.6:
                return 50
            elif win_rate > 0.5:
                return 20
        
        # Regular bonuses
        if win_rate > 0.7:
            return 15
        elif win_rate > 0.6:
            return 8
        elif win_rate > 0.5:
            return 4
        elif win_rate < 0.3:
            return -5  # Penalty for poor win rate
        
        return 0


class FastProfitableEnvironment(HistoricalOptionsEnvironment):
    """Ultra-fast environment with aggressive optimizations"""
    
    def __init__(self, *args, **kwargs):
        # Initialize attributes BEFORE calling super().__init__
        # Pre-compute ALL data for maximum speed
        self._precompute_all_data = True
        self._step_data_cache = {}
        self._option_lookup_cache = {}
        
        # Initialize tracking variables
        self.consecutive_losses = 0
        self.force_close_losses = True
        self.max_loss_per_trade = 0.02  # 2% max loss
        self.max_profit_per_trade = 0.07  # 7% take profit
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0
        self.peak_capital = 100000  # Default, will be updated
        
        # Market analysis
        self.price_history = []
        self.volatility_window = 20
        self.historical_volatility = None
        self._last_action = 'hold'  # Initialize last action to prevent NameError
        
        # Now call super().__init__
        super().__init__(*args, **kwargs)
        self.peak_capital = self.initial_capital
        
    def _precompute_entire_episode(self):
        """Pre-compute all data for the entire episode"""
        if not hasattr(self, 'training_data') or self.training_data is None:
            return
            
        # Convert entire DataFrame to numpy arrays once
        df = self.training_data
        self._episode_length = len(df)
        
        # Pre-compute all data as numpy arrays
        self._all_underlying_prices = df['underlying_price'].values
        self._all_timestamps = df['timestamp'].values
        
        # Pre-compute option lookups for each step
        # Group by timestamp for faster processing
        grouped = df.groupby('timestamp')
        
        for step in range(self._episode_length):
            timestamp = self._all_timestamps[step]
            
            # Get all options for this timestamp
            try:
                step_options = grouped.get_group(timestamp)
            except KeyError:
                self._option_lookup_cache[step] = {'call': [], 'put': []}
                continue
            
            # Pre-filter and sort options
            call_options = step_options[
                (step_options['option_type'] == 'call') &
                (step_options['moneyness'] >= 0.9) &
                (step_options['moneyness'] <= 1.1) &
                (step_options['bid'] > 0) &
                (step_options['ask'] > 0)
            ].copy()
            
            put_options = step_options[
                (step_options['option_type'] == 'put') &
                (step_options['moneyness'] >= 0.9) &
                (step_options['moneyness'] <= 1.1) &
                (step_options['bid'] > 0) &
                (step_options['ask'] > 0)
            ].copy()
            
            # Pre-compute scores
            for opt_df in [call_options, put_options]:
                if len(opt_df) > 0:
                    opt_df['mid_price'] = (opt_df['bid'] + opt_df['ask']) / 2
                    opt_df['spread'] = opt_df['ask'] - opt_df['bid']
                    opt_df['spread_pct'] = opt_df['spread'] / opt_df['mid_price'].clip(lower=0.01)
                    
                    # Score calculation
                    opt_df['score'] = (
                        (1.0 / (1.0 + opt_df['spread_pct'] * 10)) * 0.6 +
                        np.minimum(opt_df['volume'] / 1000, 1.0) * 0.2 +
                        np.where(
                            (opt_df['mid_price'] >= 1.0) & (opt_df['mid_price'] <= 10.0),
                            1.0,
                            np.where(
                                opt_df['mid_price'] < 1.0,
                                opt_df['mid_price'],
                                10.0 / opt_df['mid_price']
                            )
                        ) * 0.2
                    )
            
            # Cache the top options
            self._option_lookup_cache[step] = {
                'call': call_options.nlargest(3, 'score').to_dict('records') if len(call_options) > 0 else [],
                'put': put_options.nlargest(3, 'score').to_dict('records') if len(put_options) > 0 else []
            }
    
    def reset(self):
        """Fast reset with pre-computation"""
        obs = super().reset()
        
        # Reset tracking variables
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0
        self.consecutive_losses = 0
        self.price_history = []
        self._last_action = 'hold'  # Reset last action
        self.last_trade_reward = 0  # Reset trade reward
        
        # Pre-compute entire episode data
        if self._precompute_all_data:
            self._precompute_entire_episode()
        
        return obs
    
    def step(self, action: int):
        """Ultra-fast step function"""
        if self.done:
            return None, 0, True, {}
            
        # Fast action mapping
        action_name = ['hold', 'buy_call', 'buy_put', 'sell_call', 'sell_put', 
                      'bull_call_spread', 'bear_put_spread', 'iron_condor', 
                      'straddle', 'strangle', 'close_all_positions'][action]
        
        # Fast bounds check
        if self.current_step >= self._episode_length:
            self.done = True
            return self._get_observation(), 0, True, {}
        
        # Get pre-computed data
        current_price = self._all_underlying_prices[self.current_step]
        
        # Update price history
        self.price_history.append(current_price)
        if len(self.price_history) > self.volatility_window:
            self.price_history.pop(0)
        
        # Fast portfolio value calculation
        portfolio_value_before = self.capital + sum(
            pos.get('current_value', pos['entry_price'] * pos['quantity'] * 100)
            for pos in self.positions
        )
        
        # Execute action
        reward = 0
        
        if action_name == 'hold':
            pass
        elif action_name == 'close_all_positions' and self.positions:
            reward = self._fast_close_all_positions()
        elif action_name in ['buy_call', 'buy_put'] and len(self.positions) < self.max_positions:
            # Use pre-computed options
            option_type = 'call' if 'call' in action_name else 'put'
            suitable_options = self._option_lookup_cache.get(self.current_step, {}).get(option_type, [])
            
            if suitable_options:
                option = suitable_options[0]
                
                # Fast position sizing
                cost_per_contract = option['mid_price'] * 100 + self.commission
                contracts_to_buy = min(5, max(1, int(self.capital * 0.25 / cost_per_contract)))
                total_cost = contracts_to_buy * cost_per_contract
                
                if total_cost <= self.capital * 0.30:
                    # Add position
                    self.positions.append({
                        'option_data': option,
                        'entry_price': option['mid_price'],
                        'quantity': contracts_to_buy,
                        'entry_step': self.current_step,
                        'option_type': option_type,
                        'strike': option['strike'],
                        'score': option.get('score', 0.5)
                    })
                    self.capital -= total_cost
                    reward = 0.1
        
        # Fast position updates
        self._fast_update_positions()
        
        # Calculate reward
        portfolio_value_after = self.capital + sum(
            pos.get('current_value', pos['entry_price'] * pos['quantity'] * 100)
            for pos in self.positions
        )
        
        step_pnl = portfolio_value_after - portfolio_value_before
        
        # Use exponential reward scaling for portfolio changes
        if step_pnl != 0:
            pnl_pct = step_pnl / portfolio_value_before if portfolio_value_before > 0 else 0
            reward += self._calculate_trade_reward(step_pnl, pnl_pct, base_reward=1.0)
        else:
            # Small penalty for holding without positions
            if len(self.positions) == 0 and action_name == 'hold':
                reward -= 0.01
        
        # Add any trade closing rewards
        if hasattr(self, 'last_trade_reward'):
            reward += self.last_trade_reward
            self.last_trade_reward = 0
        
        # Update step
        self.current_step += 1
        
        # Check done
        if self.current_step >= self._episode_length - 1:
            self.done = True
        elif portfolio_value_after < self.initial_capital * 0.2:
            self.done = True
        
        return self._get_observation(), reward, self.done, {
            'portfolio_value': portfolio_value_after,
            'positions': len(self.positions),
            'symbol': self.current_symbol
        }
    
    def _fast_update_positions(self):
        """Fast position update without DataFrame operations"""
        if self.current_step >= self._episode_length:
            return
            
        positions_to_close = []
        
        for i, pos in enumerate(self.positions):
            # Find current option value from cache
            option_type = pos['option_type']
            current_options = self._option_lookup_cache.get(self.current_step, {}).get(option_type, [])
            
            # Find matching option by strike
            current_price = None
            for opt in current_options:
                if opt['strike'] == pos['strike']:
                    current_price = opt['mid_price']
                    break
            
            if current_price is None:
                continue
            
            # Calculate P&L
            entry_cost = pos['entry_price'] * pos['quantity'] * 100
            current_value = current_price * pos['quantity'] * 100
            pos['current_value'] = current_value
            pnl = current_value - entry_cost
            pnl_pct = pnl / entry_cost
            
            # Simple exit rules
            position_age = self.current_step - pos['entry_step']
            
            if pnl_pct <= -0.5 or pnl_pct >= 0.5 or position_age > 20:
                positions_to_close.append(i)
                self.capital += current_value - self.commission
                
                if pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
        
        # Remove closed positions
        for i in reversed(positions_to_close):
            self.positions.pop(i)
    
    def _fast_close_all_positions(self):
        """Fast close all positions"""
        total_reward = 0
        
        for pos in self.positions:
            current_value = pos.get('current_value', pos['entry_price'] * pos['quantity'] * 100)
            self.capital += current_value - self.commission
            
            entry_cost = pos['entry_price'] * pos['quantity'] * 100
            pnl = current_value - entry_cost
            
            if pnl > 0:
                self.winning_trades += 1
                # Exponential reward for profitable trade
                pnl_pct = pnl / entry_cost
                total_reward += self._calculate_trade_reward(pnl, pnl_pct, base_reward=10.0)
            else:
                self.losing_trades += 1
                # Smaller penalty for losses to encourage risk-taking
                pnl_pct = pnl / entry_cost
                total_reward += self._calculate_trade_reward(pnl, pnl_pct, base_reward=3.0)
        
        self.positions = []
        return total_reward


class OptimizedProfitableEnvironment(HistoricalOptionsEnvironment):
    """Optimized environment with vectorized operations and caching"""
    
    def __init__(self, *args, **kwargs):
        # Initialize caches before calling super().__init__
        # Performance optimization: pre-compute and cache
        self._option_cache = {}
        self._timestamp_index = {}
        self._numpy_cache = {}
        
        super().__init__(*args, **kwargs)


    def _calculate_trade_reward(self, pnl, pnl_pct, base_reward=1.0):
        """
        Simplified reward structure focused on win rate and profitability
        """
        if pnl > 0:
            # Reward wins proportionally
            reward = 10 * pnl_pct  # 5% win = 0.5 reward
        else:
            # Smaller penalty for losses to encourage learning
            reward = 5 * pnl_pct   # 5% loss = -0.25 reward
        
        return reward * base_reward

        
        self.consecutive_losses = 0
        self.force_close_losses = True
        self.max_loss_per_trade = 0.02  # 2% max loss
        self.max_profit_per_trade = 0.07  # 7% take profit
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0
        self.peak_capital = self.initial_capital
        
        # Market analysis
        self.price_history = []
        self.volatility_window = 20
        self.historical_volatility = None
        
        # Pre-allocate arrays for faster operations (after parent init)
        # Note: observation_space is a Dict in parent class, so we don't pre-allocate
        self._obs_array = None
        self._precomputed_episodes = {}
        
    def reset(self):
        """Override reset with optimizations"""
        # Use pre-computed episode if available
        if hasattr(self, '_use_precomputed') and self._use_precomputed and self._precomputed_episodes:
            # Cycle through pre-computed episodes
            if not hasattr(self, '_episode_counter'):
                self._episode_counter = 0
            
            episode_idx = self._episode_counter % len(self._precomputed_episodes)
            self._episode_counter += 1
            
            # Load pre-computed data
            episode_data = self._precomputed_episodes[episode_idx]
            self.training_data = episode_data['data']
            self.current_symbol = episode_data['symbol']
            self._numpy_cache = episode_data['numpy_arrays']
            self._timestamp_index = episode_data['timestamp_index']
            
            # Reset state variables
            self.current_step = 0
            self.done = False
            self.capital = self.initial_capital
            self.positions = []
            self.winning_trades = 0
            self.losing_trades = 0
            self.total_pnl = 0
            self.consecutive_losses = 0
            self.price_history = []
            self._option_cache.clear()
            
            return self._get_observation()
        
        # Fall back to original reset
        obs = super().reset()
        
        # Reset tracking
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0
        self.consecutive_losses = 0
        self.price_history = []
        
        # Clear caches only if not using precomputed episodes
        if not (hasattr(self, '_use_precomputed') and self._use_precomputed):
            self._option_cache.clear()
            self._timestamp_index.clear()
            self._numpy_cache.clear()
        
        # Pre-compute data for this episode if available
        if hasattr(self, 'training_data') and self.training_data is not None:
            self._precompute_episode_data()
            
            # Log only occasionally
            if not hasattr(self, '_episode_count'):
                self._episode_count = 0
            self._episode_count += 1
            
            if self._episode_count % 500 == 0:
                logger.info(f"Episode {self._episode_count}: {len(self.training_data)} rows, symbol: {self.current_symbol}")
        
        return obs
    
    def _precompute_episode_data(self):
        """Pre-compute frequently used values for performance"""
        df = self.training_data
        
        # Pre-compute mid prices and spreads
        df['mid_price'] = (df['bid'] + df['ask']) / 2
        df['spread'] = df['ask'] - df['bid']
        df['spread_pct'] = df['spread'] / df['mid_price'].clip(lower=0.01)
        
        # Create timestamp index for fast lookups with relative indices
        unique_timestamps = df['timestamp'].unique()
        for ts in unique_timestamps:
            mask = df['timestamp'] == ts
            # Convert to relative indices (0-based) instead of absolute df indices
            self._timestamp_index[ts] = np.where(mask.values)[0].tolist()
        
        # Convert frequently accessed columns to numpy for speed
        self._numpy_cache['timestamps'] = df['timestamp'].values
        self._numpy_cache['strikes'] = df['strike'].values
        self._numpy_cache['option_types'] = df['option_type'].values
        self._numpy_cache['bids'] = df['bid'].values
        self._numpy_cache['asks'] = df['ask'].values
        self._numpy_cache['mid_prices'] = df['mid_price'].values
        self._numpy_cache['volumes'] = df['volume'].fillna(0).values
        self._numpy_cache['moneyness'] = df['moneyness'].fillna(1.0).values
        self._numpy_cache['spread_pcts'] = df['spread_pct'].values
        
    def step(self, action: int):
        """Optimized step function"""
        if self.done:
            return None, 0, True, {}
            
        # Map action
        action_mapping = {
            0: 'hold', 1: 'buy_call', 2: 'buy_put', 3: 'sell_call',
            4: 'sell_put', 5: 'bull_call_spread', 6: 'bear_put_spread',
            7: 'iron_condor', 8: 'straddle', 9: 'strangle', 10: 'close_all_positions'
        }
        action_name = action_mapping.get(action, 'hold')
        
        # Get current data efficiently
        if self.current_step >= len(self.training_data):
            self.done = True
            return self._get_observation(), 0, True, {}
            
        current_data = self.training_data.iloc[self.current_step]
        
        # Track price history for technical indicators
        current_price = current_data.get('underlying_price', 600)
        self.underlying_price_history.append(current_price)
        self.high_price_history.append(current_price * 1.01)  # Approximation
        self.low_price_history.append(current_price * 0.99)   # Approximation
        current_price = current_data.get('underlying_price', 600)
        
        # Update price history
        self.price_history.append(current_price)
        if len(self.price_history) > self.volatility_window:
            self.price_history.pop(0)
        
        self.historical_volatility = self._calculate_historical_volatility()
        
        # Portfolio value before
        portfolio_value_before = self._calculate_portfolio_value()
        
        # Risk management
        if portfolio_value_before < self.initial_capital * 0.95 and len(self.positions) > 0:
            if action_name == 'hold' and np.random.random() < 0.7:
                action_name = 'close_all_positions'
        
        if portfolio_value_before < self.initial_capital * 0.8:
            if action_name in ['buy_call', 'buy_put', 'sell_call', 'sell_put']:
                action_name = 'hold'
        
        # Execute action
        reward = 0
        wins_before = self.winning_trades
        
        if action_name == 'hold':
            pass
        elif action_name == 'close_all_positions':
            reward = self._close_all_positions()
            wins_added = self.winning_trades - wins_before
            if wins_added > 0:
                reward += wins_added * 20.0
        elif action_name in ['buy_call', 'buy_put'] and len(self.positions) < self.max_positions:
            if self._should_enter_trade_with_indicators(action_name):
                suitable_options = self._find_suitable_options_vectorized(current_data, action_name)
                
                if suitable_options:
                    option = suitable_options[0]
                    
                    # Position sizing
                    confidence = option.get('score', 0.5)
                    max_risk = self.capital * (0.10 + 0.15 * confidence)
                    cost_per_contract = option['mid_price'] * 100 + self.commission
                    
                    ideal_contracts = int(max_risk / cost_per_contract)
                    contracts_to_buy = max(1, min(ideal_contracts, 5))
                    
                    total_cost = contracts_to_buy * cost_per_contract
                    
                    if total_cost <= self.capital * 0.30:
                        # Open position with mid price
                        self.positions.append({
                            'option_data': option,
                            'entry_price': option['mid_price'],
                            'quantity': contracts_to_buy,
                            'entry_step': self.current_step,
                            'option_type': 'call' if 'call' in action_name else 'put',
                            'strike': option['strike'],
                            'score': option.get('score', 0.5),
                            'entry_reason': self._get_entry_reason(action_name),
                            'entry_spread': option['spread']
                        })
                        self.capital -= total_cost
                        # Debug log position open
                        if True:  # Always log for debugging
                            logger.info(f"[DEBUG-OPEN] Step {self.current_step}: Opened {action_name} - "
                                      f"strike=${option['strike']:.2f}, contracts={contracts_to_buy}, "
                                      f"cost=${total_cost:.2f}, capital=${self.capital:.2f}")
                        reward = 0.1 * confidence
        
        # Update positions
        self._update_positions_optimized()
        
        # Calculate reward
        portfolio_value_after = self._calculate_portfolio_value()
        step_pnl = portfolio_value_after - portfolio_value_before
        
        # Reward for winning trades
        wins_added = self.winning_trades - wins_before
        if wins_added > 0:
            reward += wins_added * 30.0
            
            if step_pnl > 0:
                profit_pct = step_pnl / portfolio_value_before
                if profit_pct >= 0.05:
                    reward += 30.0
                elif profit_pct >= 0.03:
                    reward += 20.0
                elif profit_pct >= 0.01:
                    reward += 10.0
        
        # Standard reward shaping
        if step_pnl > 0:
            reward += step_pnl / 1000 * (1 + 10 * (step_pnl / self.initial_capital))
        else:
            reward += step_pnl / 500

        # Reward for good risk-adjusted returns
        if portfolio_value_after > portfolio_value_before:
            risk_adjusted_reward = (step_pnl / portfolio_value_before) / max(0.01, abs(step_pnl) / portfolio_value_before)
            reward += risk_adjusted_reward * 5.0

        
        # Risk management rewards
        if self.consecutive_losses > 0:
            consecutive_loss_penalty = -3.0 * (self.consecutive_losses ** 1.2)
            reward += consecutive_loss_penalty
        
        current_return = (portfolio_value_after - self.initial_capital) / self.initial_capital
        if current_return > 0:
            reward += 2.0 * current_return
        
        if len(self.positions) > 0:
            position_value = sum(pos['entry_price'] * pos['quantity'] * 100 for pos in self.positions)
            leverage_ratio = position_value / portfolio_value_after
            if 0.2 <= leverage_ratio <= 0.5:
                reward += 1.0
            elif leverage_ratio > 0.7:
                reward -= 2.0
        
        # Next step
        self.current_step += 1
        
        # Check done
        if self.current_step >= len(self.training_data) - 1:
            self.done = True
            # Debug log episode end
            logger.info(f"[DEBUG-EPISODE-END] Episode complete: wins={self.winning_trades}, "
                      f"losses={self.losing_trades}, total_pnl=${self.total_pnl:.2f}, "
                      f"final_capital=${self.capital:.2f}")
        elif portfolio_value_after < self.initial_capital * 0.2:
            logger.info("Episode ended: Capital below 20%")
            self.done = True
            
        obs = self._get_observation()
        info = {
            'portfolio_value': portfolio_value_after,
            'positions': len(self.positions),
            'symbol': self.current_symbol,
            'total_pnl': self.total_pnl,
            'action': action_name
        }
        
        return obs, reward, self.done, info
    
    def _find_suitable_options_vectorized(self, current_data, action_name):
        """Vectorized option finding for massive speedup"""
        current_time = current_data['timestamp']
        
        # Use pre-computed index for fast lookup
        if current_time in self._timestamp_index:
            indices = self._timestamp_index[current_time]
        else:
            # Fallback to date-based search
            return self._find_suitable_options_fallback(current_data, action_name)
        
        if not indices:
            return []
        
        # Vectorized filtering using numpy arrays
        option_types = self._numpy_cache['option_types'][indices]
        moneyness = self._numpy_cache['moneyness'][indices]
        bids = self._numpy_cache['bids'][indices]
        asks = self._numpy_cache['asks'][indices]
        mid_prices = self._numpy_cache['mid_prices'][indices]
        volumes = self._numpy_cache['volumes'][indices]
        spread_pcts = self._numpy_cache['spread_pcts'][indices]
        strikes = self._numpy_cache['strikes'][indices]
        
        # Filter by option type
        if 'call' in action_name:
            type_mask = option_types == 'call'
        else:
            type_mask = option_types == 'put'
        
        # Filter by moneyness (0.9 to 1.1)
        money_mask = (moneyness >= 0.9) & (moneyness <= 1.1)
        
        # Filter by valid bid/ask
        valid_mask = (bids > 0) & (asks > 0)
        
        # Filter by spread (<10%)
        spread_mask = spread_pcts <= 0.10
        
        # Combine all filters
        final_mask = type_mask & money_mask & valid_mask & spread_mask
        
        if not final_mask.any():
            return []
        
        # Calculate scores vectorized
        filtered_indices = np.where(final_mask)[0]
        
        # Score components
        spread_scores = 1.0 / (1.0 + spread_pcts[filtered_indices] * 10)
        liquidity_scores = np.minimum(volumes[filtered_indices] / 1000, 1.0)
        
        # Price scores
        filtered_mid_prices = mid_prices[filtered_indices]
        price_scores = np.where(
            (filtered_mid_prices >= 1.0) & (filtered_mid_prices <= 10.0),
            1.0,
            np.where(
                filtered_mid_prices < 1.0,
                filtered_mid_prices,
                10.0 / filtered_mid_prices
            )
        )
        
        # Combined scores
        scores = spread_scores * 0.6 + liquidity_scores * 0.2 + price_scores * 0.2
        
        # Get top 3
        top_indices = np.argsort(scores)[-3:][::-1]
        
        # Build result dicts
        results = []
        for idx in top_indices:
            real_idx = indices[filtered_indices[idx]]
            results.append({
                'strike': strikes[filtered_indices[idx]],
                'option_type': 'call' if 'call' in action_name else 'put',
                'bid': bids[filtered_indices[idx]],
                'ask': asks[filtered_indices[idx]],
                'mid_price': mid_prices[filtered_indices[idx]],
                'spread': asks[filtered_indices[idx]] - bids[filtered_indices[idx]],
                'volume': volumes[filtered_indices[idx]],
                'score': scores[idx],
                'moneyness': moneyness[filtered_indices[idx]]
            })
        
        return results
    
    def _find_suitable_options_fallback(self, current_data, action_name):
        """Fallback method when exact timestamp not found"""
        # Similar to original but with some optimizations
        current_time = current_data['timestamp']
        
        # Try date-based search
        if hasattr(current_time, 'date'):
            date_mask = self.training_data['timestamp'].dt.date == current_time.date()
            current_options = self.training_data[date_mask]
        else:
            return []
        
        if len(current_options) == 0:
            return []
        
        # Filter and score
        option_type = 'call' if 'call' in action_name else 'put'
        mask = (
            (current_options['option_type'] == option_type) &
            (current_options['moneyness'] >= 0.9) &
            (current_options['moneyness'] <= 1.1) &
            (current_options['bid'] > 0) &
            (current_options['ask'] > 0) &
            (current_options['spread_pct'] <= 0.10)
        )
        
        filtered = current_options[mask]
        if len(filtered) == 0:
            return []
        
        # Use pre-computed values
        filtered = filtered.nlargest(3, 'volume')
        
        return filtered[['strike', 'option_type', 'bid', 'ask', 'mid_price', 
                        'spread', 'volume', 'moneyness']].to_dict('records')
    
    def _update_positions_optimized(self):
        """Optimized position update using vectorized operations where possible"""
        if self.current_step >= len(self.training_data):
            return
            
        current_data = self.training_data.iloc[self.current_step]
        current_time = current_data['timestamp']
        
        positions_to_close = []
        
        for i, pos in enumerate(self.positions):
            # Find current price using cached index
            if current_time in self._timestamp_index:
                indices = self._timestamp_index[current_time]
                
                # Fast lookup using numpy
                strike_matches = self._numpy_cache['strikes'][indices] == pos['strike']
                type_matches = self._numpy_cache['option_types'][indices] == pos['option_type']
                matches = strike_matches & type_matches
                
                if matches.any():
                    match_idx = indices[np.where(matches)[0][0]]
                    current_bid = self._numpy_cache['bids'][match_idx]
                    current_ask = self._numpy_cache['asks'][match_idx]
                    current_price = (current_bid + current_ask) / 2
                else:
                    continue
            else:
                # Fallback to DataFrame query
                current_options = self.training_data[
                    (self.training_data['timestamp'] == current_time) &
                    (self.training_data['strike'] == pos['strike']) &
                    (self.training_data['option_type'] == pos['option_type'])
                ]
                
                if current_options.empty:
                    continue
                    
                opt = current_options.iloc[0]
                current_price = (opt['bid'] + opt['ask']) / 2
            
            # Calculate P&L
            entry_cost = pos['entry_price'] * pos['quantity'] * 100
            current_value = current_price * pos['quantity'] * 100
            pnl = current_value - entry_cost
            pnl_pct = pnl / entry_cost
            
            # Position management
            position_age = self.current_step - pos['entry_step']
            position_score = pos.get('score', 0.5)
            
            adjusted_stop_loss = -self.max_loss_per_trade
            adjusted_take_profit = self.max_profit_per_trade * (0.5 + 0.5 * position_score)
            
            if position_age > 5:
                time_factor = min(position_age / 20, 1.0)
                adjusted_stop_loss *= (1 - 0.3 * time_factor)
                adjusted_take_profit *= (1 - 0.4 * time_factor)
            
            # Track peak (simplified)
            pos['peak_pnl_pct'] = max(pos.get('peak_pnl_pct', pnl_pct), pnl_pct)
            
            # Exit conditions
            should_exit = False
            exit_reason = ""
            
            if pos['peak_pnl_pct'] >= 0.03 and pnl_pct <= pos['peak_pnl_pct'] * 0.5:
                should_exit = True
                exit_reason = "trailing_stop"
                self.winning_trades += 1
                self.consecutive_losses = 0
            elif pnl_pct <= adjusted_stop_loss:
                should_exit = True
                exit_reason = "stop_loss"
                self.losing_trades += 1
                self.consecutive_losses += 1
            elif pnl_pct >= adjusted_take_profit:
                should_exit = True
                exit_reason = "take_profit"
                self.winning_trades += 1
                self.consecutive_losses = 0
            elif position_age > 15:
                should_exit = True
                exit_reason = "time_exit"
                if pnl > 0:
                    self.winning_trades += 1
                    self.consecutive_losses = 0
                else:
                    self.losing_trades += 1
                    self.consecutive_losses += 1
            elif position_age > 1 and pnl_pct > 0.02:
                should_exit = True
                exit_reason = "quick_profit"
                self.winning_trades += 1
                self.consecutive_losses = 0
            
            if should_exit:
                # Debug log for position close
                if True:  # Always log for debugging
                    logger.info(f"[DEBUG-CLOSE] Step {self.current_step}: Closing position - "
                              f"reason={exit_reason}, pnl=${pnl:.2f} ({pnl_pct:.2%}), "
                              f"wins={self.winning_trades}, losses={self.losing_trades}")
                positions_to_close.append(i)
                self.total_pnl += pnl - self.commission
                self.capital += current_value - self.commission
        
        # Close positions
        for i in reversed(positions_to_close):
            self.positions.pop(i)
    
    def _close_all_positions(self):
        """Optimized close all positions"""
        total_reward = 0
        if self.current_step >= len(self.training_data) or not self.positions:
            return total_reward
            
        current_data = self.training_data.iloc[self.current_step]
        current_time = current_data['timestamp']
        
        # Get all position strikes and types
        position_strikes = [pos['strike'] for pos in self.positions]
        position_types = [pos['option_type'] for pos in self.positions]
        
        # Batch lookup if possible
        for pos in self.positions:
            # Use cached lookup
            if current_time in self._timestamp_index:
                indices = self._timestamp_index[current_time]
                
                strike_matches = self._numpy_cache['strikes'][indices] == pos['strike']
                type_matches = self._numpy_cache['option_types'][indices] == pos['option_type']
                matches = strike_matches & type_matches
                
                if matches.any():
                    match_idx = indices[np.where(matches)[0][0]]
                    current_bid = self._numpy_cache['bids'][match_idx]
                    current_ask = self._numpy_cache['asks'][match_idx]
                    current_price = (current_bid + current_ask) / 2
                    
                    exit_value = current_price * pos['quantity'] * 100 - self.commission
                    entry_cost = pos['entry_price'] * pos['quantity'] * 100
                    pnl = exit_value - entry_cost
                    
                    if pnl > 0:
                        self.winning_trades += 1
                        self.consecutive_losses = 0
                        total_reward += 10.0
                    else:
                        self.losing_trades += 1
                        self.consecutive_losses += 1
                        total_reward -= 2.0
                    
                    self.total_pnl += pnl
                    self.capital += exit_value
        
        self.positions = []
        return total_reward
    
    def _should_enter_trade(self, action_name):
        """Quick entry validation"""
        if self.consecutive_losses >= 10:
            return False
            
        if len(self.price_history) < 5:
            return True


    def _should_enter_trade_with_indicators(self, action_name):
        """Enhanced entry decision using technical indicators"""
        # First check basic conditions
        if not self._should_enter_trade(action_name):
            return False
            
        # Need enough data for indicators
        if len(self.underlying_price_history) < 26:
            return True  # Fall back to basic method
        
        # Calculate current indicators
        indicators = TechnicalIndicators.calculate_all_indicators(
            self.underlying_price_history,
            self.high_price_history if hasattr(self, 'high_price_history') else None,
            self.low_price_history if hasattr(self, 'low_price_history') else None
        )
        
        # Base confidence
        confidence = 0.0
        
        # MACD signal
        macd_bullish = indicators['macd'] > indicators['macd_signal'] and indicators['macd_histogram'] > 0
        macd_bearish = indicators['macd'] < indicators['macd_signal'] and indicators['macd_histogram'] < 0
        
        # RSI conditions
        rsi = indicators['rsi']
        rsi_oversold = rsi < 30
        rsi_overbought = rsi > 70
        rsi_neutral = 30 <= rsi <= 70
        
        # CCI conditions
        cci = indicators['cci']
        cci_oversold = cci < -100
        cci_overbought = cci > 100
        
        # ADX trend strength
        adx = indicators['adx']
        strong_trend = adx > 25
        weak_trend = adx < 20
        
        # Decision logic based on action and indicators
        if 'call' in action_name:
            # Bullish indicators for calls
            if macd_bullish and rsi_oversold and strong_trend:
                confidence += 0.4  # Strong buy signal
            elif macd_bullish and rsi_neutral:
                confidence += 0.2  # Moderate buy signal
            elif rsi_overbought or (macd_bearish and strong_trend):
                confidence -= 0.3  # Warning signal
                
            # CCI confirmation
            if cci_oversold:
                confidence += 0.15
            elif cci_overbought:
                confidence -= 0.15
                
        elif 'put' in action_name:
            # Bearish indicators for puts
            if macd_bearish and rsi_overbought and strong_trend:
                confidence += 0.4  # Strong sell signal
            elif macd_bearish and rsi_neutral:
                confidence += 0.2  # Moderate sell signal
            elif rsi_oversold or (macd_bullish and strong_trend):
                confidence -= 0.3  # Warning signal
                
            # CCI confirmation
            if cci_overbought:
                confidence += 0.15
            elif cci_oversold:
                confidence -= 0.15
        
        # Weak trend penalty
        if weak_trend:
            confidence -= 0.15
        
        # Volatility check
        if hasattr(self, 'historical_volatility') and self.historical_volatility is not None:
            # High volatility bonus for options
            if self.historical_volatility > 0.02:
                confidence += 0.1
            elif self.historical_volatility < 0.005:
                confidence -= 0.2  # Low volatility penalty
        
        return confidence > 0.1  # Threshold for entry

            
        recent_prices = self.price_history[-5:]
        price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        if 'call' in action_name and price_trend < -0.02:
            return False
        if 'put' in action_name and price_trend > 0.02:
            return False
            
        return True
    
    def _calculate_historical_volatility(self):
        """Calculate volatility from price history"""
        if len(self.price_history) < 2:
            return 0.02
        prices = np.array(self.price_history)
        returns = np.diff(prices) / prices[:-1]
        returns = returns[np.abs(returns) < 0.1]  # Remove outliers
        return np.std(returns) if len(returns) > 0 else 0.02
    
    def _get_entry_reason(self, action_name):
        """Get entry reason for logging"""
        if len(self.price_history) < 5:
            return "insufficient_data"
        
        recent_prices = self.price_history[-5:]
        price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        if 'call' in action_name:
            return "bullish_trend" if price_trend > 0 else "oversold_bounce"
        else:
            return "bearish_trend" if price_trend < 0 else "overbought_reversal"
    
    def precompute_episodes(self, num_episodes=100):
        """Pre-compute episode data for faster training"""
        if num_episodes > 0:
            logger.info(f"Pre-computing {num_episodes} episodes for faster training...")
        
        for i in range(num_episodes):
            # Select random symbol
            symbol = np.random.choice(self.symbols)
            symbol_data = self.historical_data[symbol]
            
            if len(symbol_data) < self.episode_length:
                continue
            
            # Random start point
            max_start = len(symbol_data) - self.episode_length
            start_idx = np.random.randint(0, max_start)
            
            # Slice and pre-process data
            episode_data = symbol_data.iloc[start_idx:start_idx + self.episode_length].copy()
            
            # Pre-compute all derived values
            episode_data['mid_price'] = (episode_data['bid'] + episode_data['ask']) / 2
            episode_data['spread'] = episode_data['ask'] - episode_data['bid']
            episode_data['spread_pct'] = episode_data['spread'] / episode_data['mid_price'].clip(lower=0.01)
            
            # Build timestamp index with relative indices
            timestamp_index = {}
            for ts in episode_data['timestamp'].unique():
                mask = episode_data['timestamp'] == ts
                # Store relative indices (0-based for this episode)
                timestamp_index[ts] = np.where(mask)[0].tolist()
            
            # Convert to numpy arrays
            numpy_arrays = {
                'timestamps': episode_data['timestamp'].values,
                'strikes': episode_data['strike'].values,
                'option_types': episode_data['option_type'].values,
                'bids': episode_data['bid'].values,
                'asks': episode_data['ask'].values,
                'mid_prices': episode_data['mid_price'].values,
                'volumes': episode_data['volume'].fillna(0).values,
                'moneyness': episode_data['moneyness'].fillna(1.0).values,
                'spread_pcts': episode_data['spread_pct'].values,
                'underlying_prices': episode_data['underlying_price'].values
            }
            
            # Store pre-computed episode
            self._precomputed_episodes[i] = {
                'data': episode_data,
                'symbol': symbol,
                'numpy_arrays': numpy_arrays,
                'timestamp_index': timestamp_index
            }
        
        logger.info(f"Pre-computed {len(self._precomputed_episodes)} episodes")
        self._use_precomputed = True
        self._episode_counter = 0


class VectorizedEnvironment:
    """Vectorized environment wrapper for true parallel execution"""
    
    def __init__(self, env_fns: List[callable], start_method='spawn'):
        """
        Initialize vectorized environments
        
        Args:
            env_fns: List of functions that create environments
            start_method: Multiprocessing start method ('spawn' or 'fork')
        """
        self.num_envs = len(env_fns)
        self.envs = [fn() for fn in env_fns]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
        # Pre-allocate arrays for efficiency
        self._obs_dict_keys = list(self.envs[0].reset().keys())
        
        # Create thread pool for parallel execution
        # Use threads for I/O bound operations and processes for CPU bound
        self.thread_executor = ThreadPoolExecutor(max_workers=min(self.num_envs, 32))
        
        # For CPU-intensive operations, we could use ProcessPoolExecutor
        # but for now threads are sufficient since most ops are numpy/pandas
        
    def reset(self) -> List[dict]:
        """Reset all environments in parallel"""
        # Execute resets in parallel
        futures = [self.thread_executor.submit(env.reset) for env in self.envs]
        observations = [future.result() for future in futures]
        return observations
    
    def step(self, actions: List[int]) -> Tuple[List[dict], np.ndarray, np.ndarray, List[dict]]:
        """
        Step all environments in parallel
        
        Returns:
            observations: List of observation dicts
            rewards: Array of rewards
            dones: Array of done flags
            infos: List of info dicts
        """
        # Define step function for single environment
        def step_env(env_action_pair):
            env, action = env_action_pair
            obs, reward, done, info = env.step(action)
            
            # Auto-reset if done
            if done:
                obs = env.reset()
            
            return obs, reward, done, info
        
        # Execute all steps in parallel
        futures = [
            self.thread_executor.submit(step_env, (env, action)) 
            for env, action in zip(self.envs, actions)
        ]
        
        # Collect results
        observations = []
        rewards = np.zeros(self.num_envs)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = []
        
        for i, future in enumerate(futures):
            obs, reward, done, info = future.result()
            observations.append(obs)
            rewards[i] = reward
            dones[i] = done
            infos.append(info)
        
        return observations, rewards, dones, infos
    
    def get_attr(self, attr_name: str, indices=None):
        """Get attribute from environments in parallel"""
        if indices is None:
            indices = range(self.num_envs)
        
        # Execute getattr in parallel
        futures = [
            self.thread_executor.submit(getattr, self.envs[i], attr_name) 
            for i in indices
        ]
        return [future.result() for future in futures]
    
    def set_attr(self, attr_name: str, value, indices=None):
        """Set attribute in environments"""
        if indices is None:
            indices = range(self.num_envs)
        for i in indices:
            setattr(self.envs[i], attr_name, value)
    
    def close(self):
        """Close all environments and thread pool"""
        # Shutdown thread pool
        self.thread_executor.shutdown(wait=True)
        
        # Close all environments
        for env in self.envs:
            if hasattr(env, 'close'):
                env.close()


async def load_historical_data():
    """Load historical options data"""
    try:
        # Get API credentials from environment
        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_SECRET_KEY')
        
        if not api_key or not api_secret:
            logger.warning("Alpaca API credentials not found in environment, using dummy values")
        
        data_loader = HistoricalOptionsDataLoader(
            api_key=api_key or "dummy",
            api_secret=api_secret or "dummy",
            base_url='https://paper-api.alpaca.markets'
        )
        
        # Load symbols from configuration
        symbols_config = SymbolsConfig()
        
        # Get training recommendations with high liquidity
        symbols = symbols_config.get_training_recommendations(
            include_indices=True,
            include_memes=True,  # Include some high volatility stocks
            min_liquidity=7  # Only high liquidity symbols
        )
        
        # Limit to reasonable number for training
        max_symbols = 20  # Adjust based on your needs
        if len(symbols) > max_symbols:
            symbols = symbols[:max_symbols]
            
        logger.info(f"Loading historical data for {len(symbols)} symbols: {symbols}")
        
        # Set date range
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=60)  # 60 days of data
        
        # Load all symbols at once
        historical_data = await data_loader.load_historical_options_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )
        
        # Check what we loaded
        loaded_symbols = []
        for symbol in symbols:
            if symbol in historical_data and not historical_data[symbol].empty:
                logger.info(f"Loaded {len(historical_data[symbol])} option records for {symbol}")
                loaded_symbols.append(symbol)
        
        if loaded_symbols:
            logger.info(f"Successfully loaded data for {len(loaded_symbols)} symbols: {loaded_symbols}")
            return historical_data, data_loader
        else:
            logger.error("No historical data loaded")
            return None, None
            
    except Exception as e:
        logger.error(f"Error loading historical data: {e}")
        traceback.print_exc()
        return None, None


def find_free_port():
    """Find a free port to use for distributed training"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def setup_distributed(rank, world_size):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    # Port should already be set by the main process
    if 'MASTER_PORT' not in os.environ:
        # Fallback to finding a free port
        os.environ['MASTER_PORT'] = str(find_free_port())
    
    # Set device for this process
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # Initialize the process group with device_id
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=device
    )
    
    return device


def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()


def signal_handler(signum, frame):
    """Handle Ctrl+C for graceful shutdown"""
    logger.info("\n🛑 Shutdown requested (Ctrl+C detected). Saving checkpoint...")
    shutdown_requested.set()


def train_distributed(rank, world_size, num_episodes=10000, save_interval=100, use_real_data=True, resume=False, checkpoint_path=None):
    """Distributed training function for multi-GPU support"""
    
    # Configure NCCL timeout to prevent hanging (set to 30 minutes)
    if world_size > 1:
        os.environ['NCCL_TIMEOUT'] = '1800'  # 30 minutes
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'  # Enable async error handling
        os.environ['NCCL_DEBUG'] = 'WARN'  # Set to INFO for more debugging if needed
    
    # Register signal handler for graceful shutdown (only on rank 0)
    if rank == 0:
        signal.signal(signal.SIGINT, signal_handler)
    
    # Setup distributed training only if using multiple GPUs
    if world_size > 1:
        device = setup_distributed(rank, world_size)
    else:
        # Single GPU mode
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Setup logger for this process
    logger = logging.getLogger(f"rank_{rank}")
    
    # Only show logs from rank 0
    if rank == 0:
        if world_size > 1:
            logger.info(f"🚀 Distributed training on {world_size} GPUs")
        else:
            logger.info(f"🚀 Single GPU training")
        logger.info(f"Process {rank} using GPU: {torch.cuda.get_device_name(rank) if torch.cuda.is_available() else 'CPU'}")
        logger.info(f"Checkpoints will be saved every {save_interval} episodes")
        logger.info("Press Ctrl+C to gracefully stop and save checkpoint")
    
    # GPU optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
    torch.set_float32_matmul_precision('medium')
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.98)  # Use more GPU memory for larger batches
    
    # Enable mixed precision training
    use_mixed_precision = True
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
    if rank == 0 and use_mixed_precision:
        logger.info("🚀 Mixed precision training enabled with native PyTorch AMP")
    
    # Ensure checkpoint directory exists
    if rank == 0:
        os.makedirs('checkpoints/profitable_optimized', exist_ok=True)
    
    # Resume setup
    start_episode = 0
    loaded_performance_history = None
    loaded_all_returns = []
    loaded_all_win_rates = []
    loaded_best_avg_return = -float('inf')
    loaded_best_win_rate = 0.0
    loaded_best_avg_win_rate = 0.0  # Best average win rate over checkpoint interval
    loaded_exploration_rate = 0.5
    
    if resume and checkpoint_path:
        resume_checkpoint = checkpoint_path
        if os.path.exists(resume_checkpoint):
            try:
                logger.info(f"Loading checkpoint from {resume_checkpoint}")
                checkpoint_data = torch.load(resume_checkpoint, map_location='cpu', weights_only=False)
                
                if 'episode' in checkpoint_data:
                    start_episode = checkpoint_data['episode'] + 1
                    loaded_performance_history = checkpoint_data.get('performance_history', {})
                    loaded_best_avg_return = checkpoint_data.get('best_avg_return', -float('inf'))
                    loaded_best_win_rate = checkpoint_data.get('best_win_rate', 0.0)
                    loaded_best_avg_win_rate = checkpoint_data.get('best_avg_win_rate', 0.0)
                    loaded_exploration_rate = checkpoint_data.get('exploration_rate', 0.5)
                    
                    if loaded_performance_history and 'win_rate' in loaded_performance_history:
                        loaded_all_win_rates = loaded_performance_history['win_rate']
                        loaded_all_returns = loaded_performance_history.get('avg_return', [])
                    
                    logger.info(f"Will resume from episode {start_episode}")
                    logger.info(f"Previous best win rate: {loaded_best_win_rate:.2%}")
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
                resume = False
    
    # Load data (only on rank 0 for efficiency)
    if rank == 0:
        logger.info("Loading historical options data...")
        historical_data, data_loader = asyncio.run(load_historical_data())
        
        if not historical_data:
            logger.error("Failed to load data, exiting")
            cleanup_distributed()
            return
    else:
        # Other ranks wait for data loading
        historical_data = None
        data_loader = None
    
    # Synchronize all processes (only if distributed)
    if world_size > 1:
        try:
            dist.barrier()
        except Exception as e:
            logger.error(f"Rank {rank}: Initial barrier timeout: {e}")
            # Continue anyway
    
    # Broadcast data loaded status
    if rank != 0:
        # For simplicity, we'll have all ranks load data
        # In a production system, you'd serialize and broadcast the data
        historical_data, data_loader = asyncio.run(load_historical_data())
    
    # Number of parallel environments
    # In distributed mode, split environments across GPUs
    # With multi-GPU, we want each GPU to have many environments for better utilization
    total_envs = 96 if world_size > 1 else 48  # More envs when using multiple GPUs
    n_envs = total_envs // world_size if world_size > 1 else total_envs
    if rank == 0:
        logger.info(f"🚀 Creating {n_envs} parallel environments per GPU (total: {n_envs * world_size})")
    
    # Create environment functions
    def make_env(env_id):
        def _init():
            # Use BalancedEnvironment for realistic training
            # Change to UltraFastEnvironment for maximum speed testing
            env = BalancedEnvironment(
                historical_data=historical_data,
                data_loader=data_loader,
                symbols=list(historical_data.keys()),
                initial_capital=100000,
                max_positions=5,
                commission=0.65,
                episode_length=50  # Further reduced for even faster episodes
            )
            return env
        return _init
    
    # Create vectorized environment
    env_fns = [make_env(i) for i in range(n_envs)]
    env = VectorizedEnvironment(env_fns)
    
    if rank == 0:
        logger.info(f"✅ Vectorized environment created with {n_envs} parallel environments")
    
    # Log environment stats
    if rank == 0:
        first_env = env.envs[0]
        logger.info(f"Each environment initialized with: {len(first_env.symbols)} symbols, "
                   f"${first_env.initial_capital} capital, max {first_env.max_positions} positions")
    
    # Create agent with GPU support and optimized hyperparameters
    # Reduced learning rates to prevent overfitting
    base_lr_actor_critic = 3e-5  # Reduced from 1e-5
    base_lr_clstm = 1e-4  # Reduced from 5e-5
    
    # Adaptive learning rate based on performance (will be adjusted during training)
    current_lr_actor_critic = base_lr_actor_critic
    current_lr_clstm = base_lr_clstm
    
    agent = OptionsCLSTMPPOAgent(
        observation_space=env.observation_space,
        action_space=11,
        learning_rate_actor_critic=current_lr_actor_critic,
        learning_rate_clstm=current_lr_clstm,
        gamma=0.99,
        clip_epsilon=0.1,
        entropy_coef=0.02,  # Increased from 0.001 to encourage exploration
        batch_size=2048,  # Full batch size - DDP will handle gradient averaging
        n_epochs=2,  # Fewer epochs for faster training
        device=device  # Pass specific device
    )
    
    # Move agent to specific GPU
    agent.network = agent.network.to(device)
    agent.base_network = agent.base_network.to(device)

    # Create learning rate schedulers for adaptive training
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
    
    # Scheduler for PPO optimizer - reduce on plateau
    ppo_scheduler = ReduceLROnPlateau(
        agent.ppo_optimizer, 
        mode='max',  # Maximize win rate
        factor=0.5,  # Reduce LR by half
        patience=100,  # Wait 100 episodes
        min_lr=1e-7
    )
    
    # Scheduler for CLSTM - cosine annealing with warm restarts
    clstm_scheduler = CosineAnnealingWarmRestarts(
        agent.clstm_optimizer,
        T_0=500,  # Initial period
        T_mult=2,  # Double period after each restart
        eta_min=1e-6
    )

    
    # Compile the model for faster inference (PyTorch 2.0+)
    # NOTE: Disabled due to incompatibility with retain_graph=True in PPO training
    compile_model = False
    try:
        if compile_model and hasattr(torch, 'compile'):
            logger.info("🚀 Compiling model with torch.compile for faster inference...")
            agent.network = torch.compile(agent.network, mode="reduce-overhead", backend="inductor")
            # Update base_network reference after compilation
            agent.base_network = agent.network
            logger.info("✅ Model compiled successfully!")
        else:
            if rank == 0:
                logger.info("ℹ️  Model compilation disabled (incompatible with retain_graph=True)")
    except Exception as e:
        logger.warning(f"⚠️  Failed to compile model: {e}. Continuing without compilation.")
    
    # Wrap model with DDP only if using multiple GPUs
    if world_size > 1:
        agent.network = DDP(agent.network, device_ids=[rank], output_device=rank)
        # Update base_network reference for DDP
        agent.base_network = agent.network.module
        if rank == 0:
            logger.info(f"✅ Model distributed across {world_size} GPUs with DDP")
    else:
        # Single GPU mode - no DDP wrapper
        if rank == 0:
            logger.info("✅ Model on single GPU")
    
    # Load checkpoint if resuming
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            if 'network_state_dict' in checkpoint_data:
                state_dict = checkpoint_data['network_state_dict']
                
                # Handle loading from both DDP and non-DDP checkpoints
                # Check if the state dict has 'module.' prefix (from DDP)
                if any(key.startswith('module.') for key in state_dict.keys()):
                    # Loading from DDP checkpoint
                    if world_size > 1:
                        # DDP to DDP - load directly
                        agent.network.load_state_dict(state_dict)
                    else:
                        # DDP to non-DDP - remove 'module.' prefix
                        new_state_dict = {}
                        for key, value in state_dict.items():
                            new_key = key.replace('module.', '') if key.startswith('module.') else key
                            new_state_dict[new_key] = value
                        agent.network.load_state_dict(new_state_dict)
                else:
                    # Loading from non-DDP checkpoint
                    if world_size > 1:
                        # Non-DDP to DDP - add 'module.' prefix
                        new_state_dict = {}
                        for key, value in state_dict.items():
                            new_key = f'module.{key}'
                            new_state_dict[new_key] = value
                        agent.network.load_state_dict(new_state_dict)
                    else:
                        # Non-DDP to non-DDP - load directly
                        agent.network.load_state_dict(state_dict)
            
            if 'ppo_optimizer_state_dict' in checkpoint_data:
                agent.ppo_optimizer.load_state_dict(checkpoint_data['ppo_optimizer_state_dict'])
            if 'clstm_optimizer_state_dict' in checkpoint_data:
                agent.clstm_optimizer.load_state_dict(checkpoint_data['clstm_optimizer_state_dict'])
            
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            traceback.print_exc()
    
    # Pre-train CLSTM if we have historical data (disabled for speed)
    # Only do this if not resuming and on rank 0
    pretrain_clstm = False  # Disabled for faster startup
    if False and rank == 0 and not resume and pretrain_clstm:
        logger.info("Pre-training CLSTM encoder with market data patterns...")
        pretrain_samples = []
        
        # Generate pre-training samples from historical data
        with torch.no_grad():  # Disable gradients for speed
            for symbol in list(historical_data.keys())[:3]:  # Reduced from 5 to 3 symbols
                symbol_data = historical_data[symbol]
                if len(symbol_data) < 100:
                    continue
                    
                for i in range(0, min(50, len(symbol_data) - 50), 20):  # Reduced samples
                    # Create a mock observation from historical data
                    mock_obs = env._get_observation()
                    obs_tensor = agent._observation_to_tensor(mock_obs)
                    
                    # Combine features
                    combined_features = []
                    for key in ['price_history', 'technical_indicators', 'options_chain', 'portfolio_state', 'greeks_summary']:
                        if key in obs_tensor:
                            combined_features.append(obs_tensor[key].flatten())
                    
                    if combined_features:
                        features_tensor = torch.cat(combined_features, dim=-1).squeeze(0)
                        
                        # Simple targets based on historical data
                        price_target = symbol_data.iloc[i]['underlying_price']
                        volatility_target = 0.3  # Default volatility
                        volume_target = symbol_data.iloc[i].get('volume', 1000000)
                        
                        pretrain_samples.append({
                            'features': features_tensor,
                            'price_target': price_target,
                            'volatility_target': volatility_target,
                            'volume_target': volume_target
                        })
        
        if pretrain_samples:
            logger.info(f"Pre-training CLSTM with {len(pretrain_samples)} samples...")
            pretrain_metrics = agent.pretrain_clstm(pretrain_samples, epochs=5, batch_size=64)  # Reduced epochs, larger batch
            logger.info(f"CLSTM pre-training complete. Final loss: {pretrain_metrics['final_loss']:.4f}")
    
    # Training metrics
    all_returns = loaded_all_returns if loaded_all_returns else []
    all_win_rates = loaded_all_win_rates if loaded_all_win_rates else []
    best_avg_return = loaded_best_avg_return
    best_win_rate = loaded_best_win_rate
    best_avg_win_rate = loaded_best_avg_win_rate
    
    # Log current best performance (only on rank 0)
    if rank == 0 and resume:  # Only log if resuming, not for fresh starts
        if best_win_rate > 0:
            logger.info(f"🎯 Current best single episode win rate: {best_win_rate:.2%}")
        if best_avg_win_rate > 0:
            logger.info(f"📊 Current best average win rate (over {save_interval} episodes): {best_avg_win_rate:.2%}")
        if best_win_rate == 0 and best_avg_win_rate == 0:
            logger.info("🎯 No previous best win rates recorded - starting fresh!")
    
    # Comprehensive performance tracking
    if loaded_performance_history:
        performance_history = loaded_performance_history
        # Ensure all required keys exist (for compatibility with older checkpoints)
        required_keys = {
            'episode': [], 'win_rate': [], 'avg_return': [], 'total_trades': [],
            'avg_trade_size': [], 'max_drawdown': [], 'sharpe_ratio': [],
            'win_rate_ma_50': [], 'win_rate_ma_200': [], 'return_ma_50': [],
            'return_ma_200': [], 'improvement_rate': [], 'consistency_score': [],
            'profit_factor': [], 'avg_win': [], 'avg_loss': [],
            'win_loss_ratio': [], 'consecutive_wins_max': [], 'consecutive_losses_max': [],
            'exploration_rate': [], 'learning_efficiency': [], 'action_diversity': [],
            'position_hold_time': [], 'risk_adjusted_return': []
        }
        
        # Add any missing keys with empty lists
        for key, default_value in required_keys.items():
            if key not in performance_history:
                performance_history[key] = default_value
                if rank == 0:
                    logger.info(f"Added missing performance history key: {key}")
        
        # Ensure lists are the same length by padding with zeros/defaults
        if performance_history['episode']:
            max_len = len(performance_history['episode'])
            for key in required_keys:
                if key in performance_history and len(performance_history[key]) < max_len:
                    # Pad with appropriate default values
                    default_val = 0 if key != 'win_loss_ratio' and key != 'profit_factor' else 1
                    performance_history[key].extend([default_val] * (max_len - len(performance_history[key])))
    else:
        performance_history = {
            'episode': [], 'win_rate': [], 'avg_return': [], 'total_trades': [],
            'avg_trade_size': [], 'max_drawdown': [], 'sharpe_ratio': [],
            'win_rate_ma_50': [], 'win_rate_ma_200': [], 'return_ma_50': [],
            'return_ma_200': [], 'improvement_rate': [], 'consistency_score': [],
            'profit_factor': [], 'avg_win': [], 'avg_loss': [],
            'win_loss_ratio': [], 'consecutive_wins_max': [], 'consecutive_losses_max': [],
            'exploration_rate': [], 'learning_efficiency': [], 'action_diversity': [],
            'position_hold_time': [], 'risk_adjusted_return': [], 'success_rate_by_action': {}
        }
    
    # Initialize success rate tracking for each action
    if 'success_rate_by_action' not in performance_history or not performance_history['success_rate_by_action']:
        performance_history['success_rate_by_action'] = {
            'buy_call': {'wins': 0, 'total': 0},
            'buy_put': {'wins': 0, 'total': 0},
            'close_all': {'wins': 0, 'total': 0}
        }
    
    # Exploration - increased to find new strategies and prevent overfitting
    exploration_rate = loaded_exploration_rate if resume else 0.5  # Higher initial exploration
    base_exploration_decay = 0.9999  # Slower decay to maintain exploration longer
    min_exploration = 0.15  # Higher minimum to continue exploring
    max_exploration = 0.6  # Higher maximum for more exploration
    
    # Adaptive exploration based on performance
    exploration_boost_on_stagnation = True
    stagnation_threshold = 50  # Episodes without improvement
    episodes_without_improvement = 0

    # Early stopping configuration
    early_stopping_patience = 200  # Episodes without improvement
    early_stopping_counter = 0
    best_model_state = None
    best_model_episode = 0
    
    # Performance tracking for early stopping
    recent_win_rates = deque(maxlen=50)
    performance_trend = deque(maxlen=100)

    
    # Helper function for batch action selection
    
    def detect_market_regime(price_history, window=20):
        """Detect market regime based on volatility"""
        if len(price_history) < window:
            return "normal"
        
        recent_prices = list(price_history)[-window:]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        volatility = np.std(returns)
        
        if volatility < 0.01:
            return "low_volatility"
        elif volatility > 0.03:
            return "high_volatility"
        else:
            return "normal"


    def get_batch_actions(agent, observations_list, exploration_rate, random_actions_batch):
        """Get actions for all environments in parallel - truly parallel version"""
        n_envs = len(observations_list)
        
        # Determine which environments use random actions
        use_random = np.random.random(n_envs) < exploration_rate
        
        with torch.no_grad():
            # Stack all observations into a single batch tensor
            batch_obs = {}
            for key in observations_list[0].keys():
                batch_obs[key] = torch.stack([
                    torch.tensor(obs[key], dtype=torch.float32) 
                    for obs in observations_list
                ]).to(agent.device)
            
            # Get all actions and values in a single forward pass
            action_logits, values = agent.base_network.forward(batch_obs)

            # Apply temperature scaling for exploration
            temperature = 1.0 + exploration_rate * 2.0  # Higher temperature = more exploration
            action_logits = action_logits / temperature

            
            # Sample actions
            dist = Categorical(logits=action_logits)
            sampled_actions = dist.sample()
            log_probs = dist.log_prob(sampled_actions)
            
            # Convert to numpy
            sampled_actions = sampled_actions.cpu().numpy()
            log_probs = log_probs.cpu().numpy()
            values = values.squeeze(-1).cpu().numpy()
            
            # Apply exploration mask
            actions = np.where(use_random, random_actions_batch, sampled_actions)
            
            # Create action infos
            action_infos = [
                {'log_prob': log_probs[i], 'value': values[i]}
                for i in range(n_envs)
            ]
        
        return actions.tolist(), action_infos
    
    # Progress tracking (only on rank 0)
    total_episodes = start_episode + num_episodes
    if rank == 0:
        pbar = tqdm(range(start_episode, total_episodes), desc="Training", initial=start_episode, total=total_episodes)
    else:
        pbar = range(start_episode, total_episodes)
    
    # Performance monitoring
    iteration_times = []
    episode_times = []
    
    # Pre-allocate batch arrays for better memory efficiency
    batch_observations = []
    batch_actions = []
    batch_rewards = []
    batch_values = []
    batch_log_probs = []
    batch_dones = []
    
    # Track episodes processed by this rank
    episodes_processed = 0
    
    # Function to save emergency checkpoint
    def save_emergency_checkpoint(episode_num):
        if rank == 0:
            logger.info(f"💾 Saving emergency checkpoint at episode {episode_num}...")
            checkpoint = {
                'episode': episode_num,
                'network_state_dict': agent.network.module.state_dict() if world_size > 1 else agent.network.state_dict(),
                'ppo_optimizer_state_dict': agent.ppo_optimizer.state_dict(),
                'clstm_optimizer_state_dict': agent.clstm_optimizer.state_dict(),
                'performance_history': performance_history,
                'best_avg_return': best_avg_return,
                'best_win_rate': best_win_rate,
                'best_avg_win_rate': best_avg_win_rate,
                'exploration_rate': exploration_rate,
                'all_returns': all_returns,
                'all_win_rates': all_win_rates
            }
            
            emergency_file = f'checkpoints/profitable_optimized/emergency_checkpoint_episode_{episode_num}.pt'
            os.makedirs(os.path.dirname(emergency_file), exist_ok=True)
            torch.save(checkpoint, emergency_file)
            logger.info(f"✅ Emergency checkpoint saved to {emergency_file}")
            logger.info(f"To resume training, use: --resume --checkpoint {emergency_file}")
            
            # Export performance visualization data for emergency checkpoint
            export_performance_visualization(performance_history, 
                                           f'checkpoints/profitable_optimized/emergency_performance_ep{episode_num}.json')
    
    for episode in pbar:
        # Debug logging for hang investigation
        if rank == 0 and (episode % 10 == 0 or episode == start_episode):
            logger.info(f"Beginning episode {episode} (target: {total_episodes})")
            
        # Check for shutdown request
        if shutdown_requested.is_set():
            if rank == 0:
                logger.info("Shutdown requested, saving checkpoint and exiting...")
                save_emergency_checkpoint(episode)
            break
            
        episode_start = time.time()
        
        # In distributed mode, each GPU processes all episodes with its own environments
        # No need to skip episodes - each GPU trains independently
        if False:  # Disabled episode skipping for better parallelism
            # Still check if we need to save checkpoint on rank 0
            if rank == 0 and (episode + 1) % save_interval == 0:
                # We need to save even if we're not training this episode
                logger.info(f"📌 Episode {episode + 1} reached - saving checkpoint...")
                checkpoint = {
                    'episode': episode,
                    'network_state_dict': agent.network.module.state_dict() if world_size > 1 else agent.network.state_dict(),
                    'ppo_optimizer_state_dict': agent.ppo_optimizer.state_dict(),
                    'clstm_optimizer_state_dict': agent.clstm_optimizer.state_dict(),
                    'performance_history': performance_history,
                    'best_avg_return': best_avg_return,
                    'best_win_rate': best_win_rate,
                    'exploration_rate': exploration_rate,
                    'all_returns': all_returns[-100:],
                    'all_win_rates': all_win_rates[-100:]
                }
                
                checkpoint_file = f'checkpoints/profitable_optimized/checkpoint_episode_{episode + 1}.pt'
                try:
                    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
                    torch.save(checkpoint, checkpoint_file)
                    logger.info(f"💾 Checkpoint saved at episode {episode + 1} to {checkpoint_file}")
                    
                    # Verify the file was saved
                    if os.path.exists(checkpoint_file):
                        file_size = os.path.getsize(checkpoint_file) / (1024 * 1024)  # Size in MB
                        logger.info(f"✅ Checkpoint file verified: {file_size:.2f} MB")
                    else:
                        logger.error(f"❌ Checkpoint file not found after saving: {checkpoint_file}")
                except Exception as e:
                    logger.error(f"❌ Error saving checkpoint: {e}")
                    traceback.print_exc()
            continue
            
        # Reset all environments
        obs_list = env.reset()
        episode_rewards = [[] for _ in range(n_envs)]
        episode_dones = [False] * n_envs
        env_steps = [0] * n_envs
        
        # Collect supervised learning data for CLSTM training (disabled for speed)
        # This helps the CLSTM learn market patterns
        if False and episode % 20 == 0:
            # Create supervised samples from recent market data
            for i in range(min(2, len(env.price_history) - 5)):  # Reduced from 5 to 2 samples
                # Get historical context
                hist_obs = env._get_observation()  # Current observation
                
                # Future targets (for supervised learning)
                future_idx = min(i + 5, len(env.price_history) - 1)
                price_target = env.price_history[future_idx]
                
                # Calculate volatility from price movements
                if len(env.price_history) > i + 1:
                    price_changes = np.diff(env.price_history[i:future_idx+1])
                    volatility_target = np.std(price_changes) if len(price_changes) > 0 else 0.02
                else:
                    volatility_target = 0.02
                
                # Volume target (simplified - just use current volume as proxy)
                volume_target = 1000000  # Default volume
                
                # Convert observation to tensor and add to supervised buffer
                with torch.no_grad():  # Disable gradient computation for speed
                    obs_tensor = agent._observation_to_tensor(hist_obs)
                    combined_features = []
                    for key in ['price_history', 'technical_indicators', 'options_chain', 'portfolio_state', 'greeks_summary']:
                        if key in obs_tensor:
                            combined_features.append(obs_tensor[key].flatten())
                    
                    if combined_features:
                        features_tensor = torch.cat(combined_features, dim=-1)
                        agent.add_supervised_sample(
                            features=features_tensor.squeeze(0),
                            price_target=price_target,
                            volatility_target=volatility_target,
                            volume_target=volume_target
                        )
        
        # Adaptive exploration
        exploration_rate = max(min_exploration, exploration_rate * base_exploration_decay)
        
        # Clear batch arrays
        batch_observations.clear()
        batch_actions.clear()
        batch_rewards.clear()
        batch_values.clear()
        batch_log_probs.clear()
        batch_dones.clear()
        
        # Pre-generate random numbers for exploration (batch for efficiency)
        max_steps = 50  # Match reduced episode length
        # Bias random actions towards trading actions (1-4, 10) instead of just hold (0)
        action_weights = np.array([0.1, 0.25, 0.25, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.15])  # More buy/sell
        action_weights = action_weights / action_weights.sum()  # Pre-normalize
        random_actions_all = np.random.choice(11, size=(max_steps, n_envs), p=action_weights)
        
        step = 0
        while step < max_steps and not all(episode_dones):
            # Check for shutdown during episode
            if shutdown_requested.is_set():
                break
                
            # Get actions for all environments
            actions, action_infos = get_batch_actions(
                agent, obs_list, exploration_rate, random_actions_all[step]
            )
            
            # Step all environments
            next_obs_list, rewards, dones, infos = env.step(actions)
            
            # Debug: Log actions and rewards on first few steps of every 100th episode
            if rank == 0 and episode % 100 == 0 and step < 3:
                non_hold_actions = [a for a in actions if a != 0]
                non_zero_rewards = [r for r in rewards if r != 0]
                if non_hold_actions or non_zero_rewards:
                    logger.info(f"Episode {episode}, Step {step}: Actions: {actions[:5]}..., Rewards: {rewards[:5]}...")
            
            # Process results for each environment (vectorized)
            active_envs = np.where(np.logical_not(episode_dones))[0]
            
            for env_idx in active_envs:
                episode_rewards[env_idx].append(rewards[env_idx])
                env_steps[env_idx] += 1
                
                # Add to buffer
                batch_observations.append(obs_list[env_idx])
                batch_actions.append(actions[env_idx])
                batch_rewards.append(rewards[env_idx])
                batch_values.append(action_infos[env_idx]['value'])
                batch_log_probs.append(action_infos[env_idx]['log_prob'])
                batch_dones.append(dones[env_idx])
                
                if dones[env_idx]:
                    episode_dones[env_idx] = True
            
            # Update observations
            obs_list = next_obs_list
            step += 1
        
        # Add all experiences to buffer at once
        for i in range(len(batch_observations)):
            agent.buffer.add(
                observation=batch_observations[i],
                action=batch_actions[i],
                reward=batch_rewards[i],
                value=batch_values[i],
                log_prob=batch_log_probs[i],
                done=batch_dones[i]
            )
        
        # Episode metrics - aggregate across all environments
        total_episode_rewards = []
        total_winning_trades = 0
        total_losing_trades = 0
        
        for env_idx in range(n_envs):
            if len(episode_rewards[env_idx]) > 0:
                env_return = sum(episode_rewards[env_idx])
                total_episode_rewards.append(env_return)
                
                # Get trade stats from each environment - use direct access for now
                try:
                    winning_trades = env.envs[env_idx].winning_trades
                    losing_trades = env.envs[env_idx].losing_trades
                    total_winning_trades += winning_trades
                    total_losing_trades += losing_trades
                except Exception as e:
                    logger.warning(f"Error getting trade stats from env {env_idx}: {e}")
        
        # Average return across environments
        episode_return = np.mean(total_episode_rewards) if total_episode_rewards else 0
        all_returns.append(episode_return)
        
        # Calculate aggregate win rate
        total_trades = total_winning_trades + total_losing_trades
        win_rate = total_winning_trades / max(1, total_trades)
        all_win_rates.append(win_rate)

        # Track best model based on 50-episode rolling average
        if len(all_win_rates) >= 50:
            current_avg_wr = np.mean(all_win_rates[-50:])
            if current_avg_wr > best_avg_win_rate * 1.02:  # 2% improvement threshold
                best_avg_win_rate = current_avg_wr
                # Also update for decline detection
                if best_avg_win_rate == 0.0:
                    best_avg_win_rate = current_avg_wr
                best_model_state = agent.network.module.state_dict() if world_size > 1 else agent.network.state_dict()
                best_model_episode = episode
                if rank == 0:
                    logger.info(f"🏆 New best model! 50-MA Win Rate: {best_avg_win_rate:.2%} at episode {episode}")
                    
                    # Save best model immediately
                    best_model_path = 'checkpoints/profitable_optimized/best_rolling_avg_model.pt'
                    torch.save({
                        'model_state_dict': best_model_state,
                        'episode': episode,
                        'win_rate': best_avg_win_rate,
                        'timestamp': datetime.now().isoformat()
                    }, best_model_path)

        
        # Comprehensive learning analysis and debugging
        if rank == 0 and (episode % 10 == 0 or (total_trades > 0 and total_winning_trades == 0)):
            # Direct access to check if parallel get_attr is working
            direct_winning = sum([env.envs[i].winning_trades for i in range(n_envs)])
            direct_losing = sum([env.envs[i].losing_trades for i in range(n_envs)])
            
            avg_capital = np.mean([env.envs[i].capital for i in range(n_envs)])
            total_positions = sum([len(env.envs[i].positions) for i in range(n_envs)])
            
            # Count actions taken in this episode
            action_counts = np.bincount(batch_actions if batch_actions else [0], minlength=11)
            action_names = ['hold', 'buy_call', 'buy_put', 'sell_call', 'sell_put', 
                          'bull_call', 'bear_put', 'iron_condor', 'straddle', 'strangle', 'close_all']
            
            # Calculate performance metrics for analysis
            recent_episodes = 50
            if len(performance_history['win_rate']) >= recent_episodes:
                recent_win_rates = performance_history['win_rate'][-recent_episodes:]
                recent_returns = performance_history['avg_return'][-recent_episodes:]
                win_rate_trend = np.polyfit(range(recent_episodes), recent_win_rates, 1)[0]
                return_trend = np.polyfit(range(recent_episodes), recent_returns, 1)[0]
                win_rate_std = np.std(recent_win_rates)
                avg_recent_wr = np.mean(recent_win_rates)
                avg_recent_return = np.mean(recent_returns)
            else:
                win_rate_trend = 0
                return_trend = 0
                win_rate_std = 0
                avg_recent_wr = win_rate
                avg_recent_return = episode_return
            
            # Print performance summary
            logger.info(f"\n{'='*80}")
            logger.info(f"📊 TRAINING PROGRESS REPORT - Episode {episode}")
            logger.info(f"{'='*80}")
            
            # Current episode metrics
            logger.info(f"🎯 Current Episode:")
            logger.info(f"   • Win Rate: {win_rate:.1%} ({total_winning_trades}W/{total_losing_trades}L)")
            logger.info(f"   • Return: ${episode_return:.2f}")
            logger.info(f"   • Capital: ${avg_capital:.0f} (started with $100,000)")
            logger.info(f"   • Active Positions: {total_positions}")
            
            # Recent performance (last 50 episodes)
            if len(performance_history['win_rate']) >= 10:
                logger.info(f"\n📈 Recent Performance (last {min(recent_episodes, len(performance_history['win_rate']))} episodes):")
                logger.info(f"   • Average Win Rate: {avg_recent_wr:.1%}")
                logger.info(f"   • Average Return: ${avg_recent_return:.2f}")
                logger.info(f"   • Win Rate Trend: {'+' if win_rate_trend > 0 else ''}{win_rate_trend*100:.3f}% per episode")
                logger.info(f"   • Win Rate Volatility: {win_rate_std:.1%}")
            
            # Best performance tracking
            if best_win_rate > 0 or best_avg_win_rate > 0:
                logger.info(f"\n🏆 Best Performance:")
                if best_win_rate > 0:
                    logger.info(f"   • Best Single Episode: {best_win_rate:.1%}")
                if best_avg_win_rate > 0:
                    logger.info(f"   • Best Average (100 eps): {best_avg_win_rate:.1%}")
            
            # Action distribution
            logger.info(f"\n🎮 Action Distribution:")
            action_percentages = [(name, count/sum(action_counts)*100) for name, count in zip(action_names, action_counts) if count > 0]
            action_percentages.sort(key=lambda x: x[1], reverse=True)
            for name, pct in action_percentages[:5]:  # Top 5 actions
                logger.info(f"   • {name}: {pct:.1f}%")
            
            # Advanced metrics if available
            if len(performance_history['profit_factor']) > 0:
                recent_pf = performance_history['profit_factor'][-1]
                recent_sharpe = performance_history['risk_adjusted_return'][-1]
                recent_diversity = performance_history['action_diversity'][-1]
                logger.info(f"\n📊 Advanced Metrics:")
                logger.info(f"   • Profit Factor: {recent_pf:.2f}")
                logger.info(f"   • Sharpe Ratio: {recent_sharpe:.2f}")
                logger.info(f"   • Action Diversity: {recent_diversity:.2f}")
                logger.info(f"   • Exploration Rate: {exploration_rate:.1%}")
            
            # Performance Analysis and Recommendations
            problems = []
            suggestions = []
            
            # Check for issues
            if len(performance_history['win_rate']) >= 50:
                # 1. Stagnant learning
                if abs(win_rate_trend) < 0.00001 and avg_recent_wr < 0.4:
                    problems.append("🔴 Stagnant Learning: Win rate not improving")
                    suggestions.append("• Increase exploration rate (currently {:.1%})".format(exploration_rate))
                    suggestions.append("• Check if reward shaping is appropriate")
                    suggestions.append("• Consider adjusting learning rate")
                
                # 2. High variance
                if win_rate_std > 0.25:
                    problems.append("🟡 High Variance: Unstable performance (std={:.1%})".format(win_rate_std))
                    suggestions.append("• Reduce learning rate for stability")
                    suggestions.append("• Increase batch size (currently {})".format(agent.batch_size))
                    suggestions.append("• Check for environment randomness")
                
                # 3. Declining performance
                if win_rate_trend < -0.0005:
                    problems.append("🔴 Declining Performance: Win rate decreasing")
                    suggestions.append("• Possible overfitting - reduce learning rate")
                    suggestions.append("• Increase exploration to find new strategies")
                    suggestions.append("• Check if best model checkpoint should be loaded")
                
                # 4. Low action diversity
                if 'action_diversity' in locals() and recent_diversity < 0.5:
                    problems.append("🟡 Low Action Diversity: Model stuck in limited strategy")
                    suggestions.append("• Increase exploration rate")
                    suggestions.append("• Add entropy bonus to encourage exploration")
                    suggestions.append("• Check if certain actions are being rewarded too heavily")
                
                # 5. Poor risk-adjusted returns
                if 'recent_sharpe' in locals() and recent_sharpe < -0.5:
                    problems.append("🟡 Poor Risk-Adjusted Returns: High volatility relative to returns")
                    suggestions.append("• Focus on consistent strategies")
                    suggestions.append("• Adjust position sizing logic")
                    suggestions.append("• Consider adding risk penalties to reward function")
                
                # 6. Low win rate
                if avg_recent_wr < 0.3:
                    problems.append("🔴 Low Win Rate: Below 30%")
                    suggestions.append("• Review entry/exit conditions")
                    suggestions.append("• Check if stop-loss/take-profit levels are appropriate")
                    suggestions.append("• Consider market conditions in decision making")
            
            # Print analysis
            if problems:
                logger.info(f"\n⚠️  ISSUES DETECTED:")
                for problem in problems:
                    logger.info(f"   {problem}")
                
                logger.info(f"\n💡 RECOMMENDATIONS:")
                for suggestion in suggestions:
                    logger.info(f"   {suggestion}")
            else:
                logger.info(f"\n✅ Performance Status: HEALTHY")
                if avg_recent_wr > 0.5:
                    logger.info(f"   • Excellent win rate above 50%")
                if win_rate_trend > 0.0001:
                    logger.info(f"   • Positive learning trend")
                if win_rate_std < 0.1:
                    logger.info(f"   • Stable performance")
            
            # Special warnings
            if total_trades > 0 and total_winning_trades == 0:
                logger.warning(f"\n🚨 CRITICAL: ALL TRADES ARE LOSSES!")
                sample_env = env.envs[0]
                logger.warning(f"   • Max loss: {sample_env.max_loss_per_trade:.1%}")
                logger.warning(f"   • Max profit: {sample_env.max_profit_per_trade:.1%}")
                logger.warning(f"   • Consider reviewing trading logic immediately")
            
            logger.info(f"{'='*80}\n")
        
        # Initialize is_new_best for this episode
        is_new_best = False
        
        # Adaptive learning rate and model management
        if rank == 0 and episode % 10 == 0:
            # Track improvement (will be updated later if new best is found)
            # For now, assume no improvement
            episodes_without_improvement += 10
            
            # Check if performance is declining or stagnant
            if len(performance_history['win_rate']) >= 50:
                recent_win_rates = performance_history['win_rate'][-50:]
                recent_avg = np.mean(recent_win_rates)
                
                # Declining performance - adjust learning but maintain model continuity
                if episodes_without_improvement >= stagnation_threshold and best_avg_win_rate > 0:
                    # Calculate actual performance drop percentage
                    performance_drop = (best_avg_win_rate - recent_avg) / best_avg_win_rate
                    
                    if performance_drop >= 0.20:  # 20% or more drop from best average
                        logger.info(f"\n⚠️  PERFORMANCE DECLINE DETECTED!")

                        logger.info(f"   Recent 50-ep avg: {recent_avg:.2%}")
                        logger.info(f"   Best 50-ep avg: {best_avg_win_rate:.2%}")
                        logger.info(f"   Performance drop: {performance_drop:.1%}")
                        logger.info(f"   Episodes without improvement: {episodes_without_improvement}")
                        
                        # Reload model only on severe decline (30% or more drop)
                        reload_best_model = performance_drop >= 0.30
                        
                        if reload_best_model:
                            best_model_path = 'checkpoints/profitable_optimized/best_model.pt'
                            if os.path.exists(best_model_path):
                                logger.info(f"🔄 CRITICAL: Reloading best model due to severe decline...")
                                try:
                                    best_checkpoint = torch.load(best_model_path, map_location=device)
                                    if world_size > 1:
                                        agent.network.module.load_state_dict(best_checkpoint['network_state_dict'])
                                    else:
                                        agent.network.load_state_dict(best_checkpoint['network_state_dict'])
                                    
                                    # Don't reload optimizer states - start fresh with lower LR
                                    logger.info(f"✅ Best model reloaded!")
                                    episodes_without_improvement = 0
                                    
                                    # Significant exploration boost for recovery
                                    exploration_rate = min(max_exploration, exploration_rate + 0.15)
                                    logger.info(f"📈 Exploration rate boosted significantly to {exploration_rate:.1%}")
                                except Exception as e:
                                    logger.error(f"Failed to reload best model: {e}")
                        else:
                            # Just reduce learning rates without reloading
                            logger.info(f"📉 Moderate decline ({performance_drop:.1%}) - reducing learning rates without reload...")
                        
                        # Always reduce learning rates on decline
                        for param_group in agent.ppo_optimizer.param_groups:
                            param_group['lr'] = param_group['lr'] * 0.7  # Less aggressive reduction
                            current_lr_actor_critic = param_group['lr']
                        for param_group in agent.clstm_optimizer.param_groups:
                            param_group['lr'] = param_group['lr'] * 0.7
                            current_lr_clstm = param_group['lr']
                        logger.info(f"📉 Learning rates reduced by 30%")
                        logger.info(f"   Actor-Critic LR: {current_lr_actor_critic:.2e}")
                        logger.info(f"   CLSTM LR: {current_lr_clstm:.2e}")
                        
                        # Moderate exploration boost even without reload
                        if not reload_best_model:
                            exploration_rate = min(max_exploration, exploration_rate + 0.05)
                            logger.info(f"📈 Exploration rate boosted to {exploration_rate:.1%}")
                
                # Stagnant learning - boost exploration
                elif abs(np.polyfit(range(len(recent_win_rates)), recent_win_rates, 1)[0]) < 0.00001:
                    if exploration_boost_on_stagnation:
                        exploration_rate = min(max_exploration, exploration_rate + 0.05)
                        logger.info(f"📈 Stagnant learning detected - exploration boosted to {exploration_rate:.1%}")
        
        # Check if we're approaching the best win rate
        if rank == 0 and episode % 10 == 0 and best_win_rate > 0 and abs(win_rate - best_win_rate) < 0.05:
            if win_rate < best_win_rate:
                logger.info(f"📊 Approaching best win rate! Current: {win_rate:.2%}, Best: {best_win_rate:.2%} (gap: {best_win_rate - win_rate:.2%})")
            elif win_rate == best_win_rate:
                logger.info(f"📊 Matched best win rate: {win_rate:.2%}!")
        
        # Check if this is a new best model (check after EVERY episode, not just at checkpoints)
        # First, determine if we have a new best model (rank 0 decides)
        is_new_best = False
        if rank == 0:
            is_new_best = win_rate > best_win_rate and total_trades >= 1
            
        # Broadcast the decision to all ranks with timeout handling
        if world_size > 1 and dist.is_initialized():
            try:
                is_new_best_tensor = torch.tensor([1.0 if is_new_best else 0.0], device=device)
                # Use a timeout to prevent hanging
                dist.broadcast(is_new_best_tensor, src=0)
                is_new_best = bool(is_new_best_tensor.item())
            except Exception as e:
                logger.error(f"Rank {rank}: Error broadcasting best model decision: {e}")
                # Fallback to local decision
                is_new_best = False
        
        if is_new_best:
            if rank == 0:
                if win_rate <= best_win_rate or total_trades < 1:
                    # This shouldn't happen but just in case
                    logger.info(f"📈 Higher win rate {win_rate:.2%} (vs best {best_win_rate:.2%}) but only {total_trades} trades - need at least 1")
                else:  # total_trades >= 1
                    previous_best = best_win_rate
                    best_win_rate = win_rate
                    improvement = best_win_rate - previous_best
                    episodes_without_improvement = 0  # Reset counter on new best
                    
                    logger.info(f"\n{'='*60}")
                    logger.info(f"🏆 NEW BEST MODEL! Win rate: {best_win_rate:.2%} (was {previous_best:.2%}, +{improvement:.2%}) at episode {episode + 1}")
                    logger.info(f"   Total trades: {total_trades}, Return: {episode_return:.1f}")
                    
                    # Save best model immediately
                    best_checkpoint = {
                        'episode': episode,
                        'network_state_dict': agent.network.module.state_dict() if world_size > 1 else agent.network.state_dict(),
                        'ppo_optimizer_state_dict': agent.ppo_optimizer.state_dict(),
                        'clstm_optimizer_state_dict': agent.clstm_optimizer.state_dict(),
                        'performance_history': performance_history,
                        'best_avg_return': best_avg_return,
                        'best_win_rate': best_win_rate,
                        'exploration_rate': exploration_rate,
                        'all_returns': all_returns[-100:] if all_returns else [],
                        'all_win_rates': all_win_rates[-100:] if all_win_rates else [],
                        'episode_metrics': {
                            'total_trades': total_trades,
                            'winning_trades': total_winning_trades,
                            'losing_trades': total_losing_trades,
                            'episode_return': episode_return,
                            'final_capital': avg_capital if 'avg_capital' in locals() else 100000
                        }
                    }
                    
                    try:
                        torch.save(best_checkpoint, 'checkpoints/profitable_optimized/best_model.pt')
                        logger.info(f"💾 Best model saved immediately!")
                        
                        # Also save a timestamped backup
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        backup_path = f'checkpoints/profitable_optimized/best_model_wr{int(best_win_rate*100)}_ep{episode+1}_{timestamp}.pt'
                        torch.save(best_checkpoint, backup_path)
                        logger.info(f"📁 Backup saved to: {backup_path}")
                        
                        # Export performance visualization data for best model
                        export_performance_visualization(performance_history, 
                                                       f'checkpoints/profitable_optimized/best_model_performance_ep{episode + 1}.json')
                        
                        # NOTE: We DON'T reload the model here to maintain training continuity
                        # The saved checkpoint is for future use (resume) but we continue with current model
                        logger.info(f"ℹ️  Continuing training with current model (best model saved for future use)")
                        logger.info(f"{'='*60}\n")
                    except Exception as e:
                        logger.error(f"❌ Error saving/reloading best model: {e}")
                        traceback.print_exc()
                        logger.info("="*60 + "\n")
            
            # Synchronize across all processes if using distributed training
            if world_size > 1 and dist.is_initialized():
                try:
                    # All ranks must participate in the barrier and broadcast
                    # Set a timeout to prevent hanging
                    dist.barrier()
                    
                    if rank == 0:
                        logger.info(f"🔄 Broadcasting best model to all GPUs...")
                    
                    # Broadcast the best model state from rank 0 to all other ranks
                    # Do this in smaller chunks to avoid timeout
                    param_count = 0
                    for param in agent.network.parameters():
                        try:
                            dist.broadcast(param.data, src=0)
                            param_count += 1
                        except Exception as e:
                            logger.error(f"Rank {rank}: Error broadcasting parameter {param_count}: {e}")
                            break
                    
                    # Final synchronization barrier
                    dist.barrier()
                    
                    if rank == 0:
                        logger.info(f"✅ All GPUs synchronized with best model!")
                except Exception as e:
                    logger.error(f"Rank {rank}: Error during model synchronization: {e}")
                    logger.error(f"Continuing without full synchronization")
        
        # Increment episodes processed counter
        episodes_processed += 1
        
        # Update agent - train with larger buffer for efficiency
        # With 48-96 parallel envs, we collect data much faster
        if len(agent.buffer) >= agent.batch_size:  # Train when we have enough for one batch
            if rank == 0 and episode % 50 == 0:
                logger.info(f"Buffer size: {len(agent.buffer)}, Batch size: {agent.batch_size} - Training triggered")
            if world_size > 1 and dist.is_initialized():
                # Distributed training: All GPUs train together with gradient averaging
                try:
                    # Log buffer sizes across GPUs
                    local_buffer_size = len(agent.buffer)
                    buffer_sizes = [0] * world_size
                    dist.all_gather_object(buffer_sizes, local_buffer_size)
                    
                    if rank == 0:
                        total_experiences = sum(buffer_sizes)
                        logger.info(f"Training with {total_experiences} total experiences across {world_size} GPUs: {buffer_sizes}")
                    
                    # All GPUs train on their local data with DDP gradient synchronization
                    train_metrics = agent.train()
                    
                    # Ensure all GPUs finish training before continuing
                    # Use try-except to handle potential timeout
                    try:
                        dist.barrier()
                    except Exception as e:
                        logger.error(f"Rank {rank}: Training barrier timeout: {e}")
                        # Continue anyway to prevent deadlock
                    
                except Exception as e:
                    logger.error(f"Rank {rank}: Distributed training error: {e}")
                    traceback.print_exc()
                    # Fallback to local training
                    train_metrics = agent.train()
            else:
                # Single GPU training
                try:
                    train_metrics = agent.train()
                except Exception as e:
                    logger.error(f"Training error: {e}")
                    train_metrics = {}
            
            # Clear buffer after training
            if rank == 0 and episode % 50 == 0:
                logger.info(f"Clearing buffer after training")
            agent.buffer.clear()
            
            # Clear GPU cache periodically to prevent memory fragmentation
            if episodes_processed % 100 == 0:
                torch.cuda.empty_cache()
            
            # Log training metrics periodically (only on rank 0)
            if rank == 0 and episodes_processed % 200 == 0 and train_metrics:
                logger.info(f"Episode {episode}: PPO Loss: {train_metrics.get('policy_loss', 0):.4f}, "
                          f"Value Loss: {train_metrics.get('value_loss', 0):.4f}, "
                          f"CLSTM Loss: {train_metrics.get('clstm_loss', 'N/A')}")
        
        # Track episode time
        episode_time = time.time() - episode_start
        episode_times.append(episode_time)
        
        # Collect comprehensive metrics for tracking
        if rank == 0:  # Only on main process
            # Calculate advanced metrics
            winning_pnls = []
            losing_pnls = []
            position_durations = []
            action_counts_episode = {}
            
            # Collect detailed trade statistics from all environments
            for env_idx in range(n_envs):
                env_obj = env.envs[env_idx]
                if hasattr(env_obj, '_position_cache'):
                    for pos_data in env_obj._position_cache.values():
                        if 'pnl' in pos_data:
                            if pos_data['pnl'] > 0:
                                winning_pnls.append(pos_data['pnl'])
                            else:
                                losing_pnls.append(pos_data['pnl'])
                            if 'exit_step' in pos_data and 'entry_step' in pos_data:
                                position_durations.append(pos_data['exit_step'] - pos_data['entry_step'])
            
            # Calculate metrics
            profit_factor = sum(winning_pnls) / abs(sum(losing_pnls)) if losing_pnls else float('inf')
            avg_win = np.mean(winning_pnls) if winning_pnls else 0
            avg_loss = np.mean(losing_pnls) if losing_pnls else 0
            win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            # Track consecutive wins/losses
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            current_streak = 0
            for env_idx in range(n_envs):
                env_obj = env.envs[env_idx]
                if hasattr(env_obj, 'consecutive_losses'):
                    max_consecutive_losses = max(max_consecutive_losses, env_obj.consecutive_losses)
            
            # Action diversity (entropy of action distribution)
            if batch_actions:
                action_probs = np.bincount(batch_actions, minlength=11) / len(batch_actions)
                action_diversity = -np.sum(action_probs * np.log(action_probs + 1e-10))
            else:
                action_diversity = 0
            
            # Risk-adjusted returns (Sharpe-like metric)
            if len(all_returns) > 20:
                recent_returns = all_returns[-20:]
                risk_adjusted_return = np.mean(recent_returns) / (np.std(recent_returns) + 1e-6)
            else:
                risk_adjusted_return = 0
            
            # Update performance history
            # Ensure lists are initialized
            if 'episode' not in performance_history:
                performance_history['episode'] = []
            if 'win_rate' not in performance_history:
                performance_history['win_rate'] = []
            if 'avg_return' not in performance_history:
                performance_history['avg_return'] = []
            
            performance_history['episode'].append(episode)
            performance_history['win_rate'].append(win_rate)
            performance_history['avg_return'].append(episode_return)
            performance_history['total_trades'].append(total_trades)
            performance_history['profit_factor'].append(min(10, profit_factor))  # Cap at 10
            performance_history['avg_win'].append(avg_win)
            performance_history['avg_loss'].append(avg_loss)
            performance_history['win_loss_ratio'].append(min(10, win_loss_ratio))  # Cap at 10
            performance_history['consecutive_wins_max'].append(max_consecutive_wins)  # Added this
            performance_history['consecutive_losses_max'].append(max_consecutive_losses)
            performance_history['exploration_rate'].append(exploration_rate)
            performance_history['action_diversity'].append(action_diversity)
            performance_history['risk_adjusted_return'].append(risk_adjusted_return)
            
            # Add missing metrics with default values
            avg_position_duration = np.mean(position_durations) if position_durations else 0
            performance_history['position_hold_time'].append(avg_position_duration)
            
            # Add placeholder values for metrics we're not tracking yet
            if 'avg_trade_size' in performance_history:
                performance_history['avg_trade_size'].append(0)
            if 'max_drawdown' in performance_history:
                performance_history['max_drawdown'].append(0)
            if 'sharpe_ratio' in performance_history:
                performance_history['sharpe_ratio'].append(risk_adjusted_return)  # Use risk-adjusted return as proxy
            if 'return_ma_50' in performance_history:
                if len(all_returns) >= 50:
                    performance_history['return_ma_50'].append(np.mean(all_returns[-50:]))
                else:
                    performance_history['return_ma_50'].append(np.mean(all_returns) if all_returns else 0)
            if 'return_ma_200' in performance_history:
                if len(all_returns) >= 200:
                    performance_history['return_ma_200'].append(np.mean(all_returns[-200:]))
                else:
                    performance_history['return_ma_200'].append(np.mean(all_returns) if all_returns else 0)
            if 'improvement_rate' in performance_history:
                performance_history['improvement_rate'].append(0)
            if 'consistency_score' in performance_history:
                performance_history['consistency_score'].append(0)
            if 'learning_efficiency' in performance_history:
                performance_history['learning_efficiency'].append(0)
            
            # Calculate rolling averages
            if len(performance_history['win_rate']) >= 50:
                performance_history['win_rate_ma_50'].append(np.mean(performance_history['win_rate'][-50:]))
            else:
                performance_history['win_rate_ma_50'].append(np.mean(performance_history['win_rate']))
            
            if len(performance_history['win_rate']) >= 200:
                performance_history['win_rate_ma_200'].append(np.mean(performance_history['win_rate'][-200:]))
            else:
                performance_history['win_rate_ma_200'].append(np.mean(performance_history['win_rate']))
            
            # Analyze learning quality periodically with detailed report
            if episode % 50 == 0 and len(performance_history['win_rate']) >= 50:
                # Print a more detailed analysis every 50 episodes
                logger.info(f"\n{'='*80}")
                logger.info(f"📊 DETAILED PERFORMANCE ANALYSIS - Episode {episode}")
                logger.info(f"{'='*80}")
                
                # Performance over different time windows
                windows = [10, 50, 100, 200]
                logger.info(f"\n📈 Performance Over Time:")
                for window in windows:
                    if len(performance_history['win_rate']) >= window:
                        window_wr = np.mean(performance_history['win_rate'][-window:])
                        window_ret = np.mean(performance_history['avg_return'][-window:])
                        logger.info(f"   • Last {window} episodes: WR={window_wr:.1%}, Avg Return=${window_ret:.2f}")
                
                # Trading statistics
                if len(performance_history['total_trades']) > 0:
                    recent_trades = performance_history['total_trades'][-50:]
                    avg_trades_per_ep = np.mean(recent_trades)
                    logger.info(f"\n📊 Trading Statistics:")
                    logger.info(f"   • Average trades per episode: {avg_trades_per_ep:.1f}")
                    if 'profit_factor' in performance_history and len(performance_history['profit_factor']) > 0:
                        avg_pf = np.mean([pf for pf in performance_history['profit_factor'][-50:] if pf < 10])
                        logger.info(f"   • Average profit factor: {avg_pf:.2f}")
                    if 'avg_win' in performance_history and 'avg_loss' in performance_history:
                        recent_avg_wins = [w for w in performance_history['avg_win'][-50:] if w > 0]
                        recent_avg_losses = [l for l in performance_history['avg_loss'][-50:] if l < 0]
                        if recent_avg_wins and recent_avg_losses:
                            logger.info(f"   • Average win size: ${np.mean(recent_avg_wins):.2f}")
                            logger.info(f"   • Average loss size: ${abs(np.mean(recent_avg_losses)):.2f}")
                
                # Learning progress
                if len(performance_history['win_rate']) >= 200:
                    early_wr = np.mean(performance_history['win_rate'][:50])
                    recent_wr = np.mean(performance_history['win_rate'][-50:])
                    improvement = recent_wr - early_wr
                    logger.info(f"\n📈 Learning Progress:")
                    logger.info(f"   • Starting win rate (first 50 eps): {early_wr:.1%}")
                    logger.info(f"   • Current win rate (last 50 eps): {recent_wr:.1%}")
                    logger.info(f"   • Total improvement: {'+' if improvement > 0 else ''}{improvement:.1%}")
                    # Calculate episodes since best safely
                    if performance_history['win_rate']:
                        max_wr = max(performance_history['win_rate'])
                        max_wr_idx = performance_history['win_rate'].index(max_wr)
                        if max_wr_idx < len(performance_history['episode']):
                            best_episode = performance_history['episode'][max_wr_idx]
                            episodes_since_best = episode - best_episode
                        else:
                            episodes_since_best = "N/A"
                    else:
                        episodes_since_best = "N/A"
                    logger.info(f"   • Episodes since best: {episodes_since_best}")
                
                # Call the detailed analysis function
                analyze_learning_quality(performance_history, episode, logger)
                logger.info(f"Detailed analysis complete for episode {episode}")
        
        # Update progress bar (only on rank 0)
        if rank == 0:
            avg_episode_time = np.mean(episode_times[-100:]) if episode_times else episode_time
            steps_per_second = len(batch_observations) / episode_time if episode_time > 0 else 0
            
            # Calculate true episodes per second (all environments combined across all GPUs)
            # Each GPU processes n_envs environments, so multiply by world_size
            true_eps_per_second = (n_envs * world_size) / avg_episode_time if avg_episode_time > 0 else 0
            total_steps_per_second = steps_per_second * n_envs * world_size
            
            # Enhanced progress bar with learning indicators
            postfix_dict = {
                'WR': f"{win_rate:.1%}",
                'Ret': f"{episode_return:.1f}",
                'ep/s': f"{true_eps_per_second:.1f}",
                'envs': f"{n_envs}x{world_size}"
            }
            
            # Add trend indicators
            if len(performance_history['win_rate']) > 50:
                recent_trend = performance_history['win_rate'][-1] - performance_history['win_rate'][-50]
                postfix_dict['Trend'] = f"{'↑' if recent_trend > 0 else '↓'}{abs(recent_trend)*100:.1f}%"
                
                # Add status indicator based on recent performance
                recent_wr = np.mean(performance_history['win_rate'][-50:])
                if recent_wr > 0.5:
                    status = "🟢"  # Green - Excellent
                elif recent_wr > 0.4:
                    status = "🔵"  # Blue - Good
                elif recent_wr > 0.3:
                    status = "🟡"  # Yellow - Needs improvement
                else:
                    status = "🔴"  # Red - Poor
                postfix_dict['Status'] = status
            
            # Add best performance tracking
            if best_win_rate > 0:
                postfix_dict['Best'] = f"{best_win_rate:.1%}"
            
            pbar.set_postfix(postfix_dict)
            pbar.update(0)  # Force refresh of progress bar
        
        # Save checkpoint logic (only on rank 0, after processing episodes)
        # This should happen whether or not this rank processed the episode
        # IMPORTANT: All ranks must participate in checkpoint synchronization
        if (episode + 1) % save_interval == 0:
            if rank == 0:
                logger.info(f"📌 Episode {episode + 1} reached - saving checkpoint...")
            
                # Calculate average win rate over the checkpoint interval
                if len(all_win_rates) >= save_interval:
                    current_avg_win_rate = np.mean(all_win_rates[-save_interval:])
                else:
                    current_avg_win_rate = np.mean(all_win_rates) if all_win_rates else 0.0
                
                logger.info(f"📊 Average win rate over last {min(len(all_win_rates), save_interval)} episodes: {current_avg_win_rate:.2%}")
                
                checkpoint = {
                    'episode': episode,
                    'network_state_dict': agent.network.module.state_dict() if world_size > 1 else agent.network.state_dict(),
                    'ppo_optimizer_state_dict': agent.ppo_optimizer.state_dict(),
                    'clstm_optimizer_state_dict': agent.clstm_optimizer.state_dict(),
                    'performance_history': performance_history,
                    'best_avg_return': best_avg_return,
                    'best_win_rate': best_win_rate,
                    'best_avg_win_rate': best_avg_win_rate,
                    'exploration_rate': exploration_rate,
                    'all_returns': all_returns[-100:] if all_returns else [],
                    'all_win_rates': all_win_rates[-100:] if all_win_rates else []
                }
                
                checkpoint_file = f'checkpoints/profitable_optimized/checkpoint_episode_{episode + 1}.pt'
                try:
                    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
                    torch.save(checkpoint, checkpoint_file)
                    logger.info(f"💾 Checkpoint saved at episode {episode + 1} to {checkpoint_file}")
                    
                    # Verify the file was saved
                    if os.path.exists(checkpoint_file):
                        file_size = os.path.getsize(checkpoint_file) / (1024 * 1024)  # Size in MB
                        logger.info(f"✅ Checkpoint file verified: {file_size:.2f} MB")
                        
                        # Export performance visualization data
                        export_performance_visualization(performance_history, 
                                                       f'checkpoints/profitable_optimized/performance_ep{episode + 1}.json')
                    else:
                        logger.error(f"❌ Checkpoint file not found after saving: {checkpoint_file}")
                except Exception as e:
                    logger.error(f"❌ Error saving checkpoint: {e}")
                    traceback.print_exc()
            else:
                # Non-rank 0 processes must wait for rank 0 to complete checkpoint saving
                current_avg_win_rate = 0.0  # Placeholder for non-rank 0
            
            # Check if this is a new best average win rate
            # First, determine if we have a new best average (rank 0 decides)
            is_new_best_avg = False
            if rank == 0:
                # Check if this is a new best average (and initialize if needed)
                if best_avg_win_rate == 0.0 and len(all_win_rates) >= save_interval:
                    best_avg_win_rate = current_avg_win_rate
                is_new_best_avg = current_avg_win_rate > best_avg_win_rate and len(all_win_rates) >= save_interval
                
            # Ensure all ranks synchronize before broadcasting
            if world_size > 1 and dist.is_initialized():
                try:
                    # First barrier: ensure all ranks reach this point
                    dist.barrier()
                    
                    # Broadcast the decision to all ranks
                    is_new_best_avg_tensor = torch.tensor([1.0 if is_new_best_avg else 0.0], device=device)
                    dist.broadcast(is_new_best_avg_tensor, src=0)
                    is_new_best_avg = bool(is_new_best_avg_tensor.item())
                except Exception as e:
                    logger.error(f"Rank {rank}: Error in checkpoint synchronization: {e}")
                    # Fallback to local decision
                    is_new_best_avg = False
            
            if is_new_best_avg:
                if rank == 0:
                    previous_best_avg = best_avg_win_rate
                    best_avg_win_rate = current_avg_win_rate
                    improvement = best_avg_win_rate - previous_best_avg
                    
                    logger.info(f"\n{'='*60}")
                    logger.info(f"🏆 NEW BEST AVERAGE WIN RATE! Average: {best_avg_win_rate:.2%} (was {previous_best_avg:.2%}, +{improvement:.2%})")
                    logger.info(f"   Calculated over last {save_interval} episodes")
                    
                    # Save best average model
                    best_avg_checkpoint = checkpoint.copy()
                    best_avg_checkpoint['best_avg_win_rate'] = best_avg_win_rate
                    
                    try:
                        torch.save(best_avg_checkpoint, 'checkpoints/profitable_optimized/best_avg_model.pt')
                        logger.info(f"💾 Best average model saved!")
                        
                        # Also save a timestamped backup
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        backup_path = f'checkpoints/profitable_optimized/best_avg_model_wr{int(best_avg_win_rate*100)}_ep{episode+1}_{timestamp}.pt'
                        torch.save(best_avg_checkpoint, backup_path)
                        logger.info(f"📁 Backup saved to: {backup_path}")
                        
                        # Export performance visualization data for best average model
                        export_performance_visualization(performance_history, 
                                                       f'checkpoints/profitable_optimized/best_avg_performance_ep{episode + 1}.json')
                        
                        # NOTE: We DON'T reload the model here to maintain training continuity
                        # The saved checkpoint is for future use (resume) but we continue with current model
                        logger.info(f"ℹ️  Continuing training with current model (best avg saved for future use)")
                        logger.info(f"{'='*60}\n")
                    except Exception as e:
                        logger.error(f"❌ Error saving/reloading best average model: {e}")
                        traceback.print_exc()
                        logger.info("="*60 + "\n")
                
                # Synchronize across all processes if using distributed training
                if world_size > 1 and dist.is_initialized():
                    try:
                        # All ranks must participate in the barrier and broadcast
                        dist.barrier()
                        
                        if rank == 0:
                            logger.info(f"🔄 Broadcasting best average model to all GPUs...")
                        
                        # Broadcast the best model state from rank 0 to all other ranks
                        # Do this in smaller chunks to avoid timeout
                        param_count = 0
                        for param in agent.network.parameters():
                            try:
                                dist.broadcast(param.data, src=0)
                                param_count += 1
                            except Exception as e:
                                logger.error(f"Rank {rank}: Error broadcasting parameter {param_count}: {e}")
                                break
                        
                        # Ensure all ranks are synced
                        dist.barrier()
                        
                        if rank == 0:
                            logger.info(f"✅ All GPUs synchronized with best average model!")
                    except Exception as e:
                        logger.error(f"Rank {rank}: Error during average model synchronization: {e}")
                        logger.error(f"Continuing without full synchronization")
            
            # Note: Best single episode model saving is done immediately after each episode
            # This ensures we never miss a high-performing episode
            
            # Final synchronization barrier for checkpoint saving
            if world_size > 1 and dist.is_initialized():
                try:
                    dist.barrier()
                    if rank == 0:
                        logger.info("✅ All ranks synchronized after checkpoint save")
                except Exception as e:
                    logger.error(f"Rank {rank}: Error in final checkpoint barrier: {e}")
        
        # Debug: Log end of episode
        if rank == 0 and (episode % 10 == 0 or episode == start_episode + num_episodes - 1):
            logger.info(f"Completed episode {episode}, moving to next...")
    
    if rank == 0:
        if shutdown_requested.is_set():
            logger.info("🛑 Training stopped by user")
            logger.info(f"Completed {len(all_returns)} episodes before shutdown")
        else:
            logger.info("✅ Training complete!")
        
        if all_win_rates:
            logger.info(f"Final win rate: {all_win_rates[-1]:.2%}")
            logger.info(f"Best single episode win rate achieved: {best_win_rate:.2%}")
            logger.info(f"Best average win rate achieved (over {save_interval} episodes): {best_avg_win_rate:.2%}")
            
            # Summary of performance
            if len(all_win_rates) > 100:
                recent_avg_win_rate = np.mean(all_win_rates[-100:])
                logger.info(f"Average win rate (last 100 episodes): {recent_avg_win_rate:.2%}")
                
                # Check if model is consistently performing well
                if recent_avg_win_rate >= best_avg_win_rate * 0.95:
                    logger.info("✅ Model is performing consistently near its best average!")
                else:
                    logger.info(f"ℹ️  Recent performance is {((best_avg_win_rate - recent_avg_win_rate) / best_avg_win_rate * 100):.1f}% below best average")
        if episode_times:
            logger.info(f"Average episode time: {np.mean(episode_times):.3f}s ({1/np.mean(episode_times):.2f} ep/s)")
            logger.info(f"Average steps per second: ~{200/np.mean(episode_times):.1f} step/s")
    
    # Clean up distributed training (only if initialized)
    if world_size > 1 and dist.is_initialized():
        cleanup_distributed()
    
    # Close vectorized environments
    if 'env' in locals():
        env.close()


def analyze_learning_quality(performance_history, episode, logger):
    """Analyze if the model is learning effectively"""
    if len(performance_history['win_rate']) < 100:
        return  # Need more data
    
    # Get recent performance
    recent_win_rates = performance_history['win_rate'][-100:]
    recent_returns = performance_history['avg_return'][-100:]
    recent_exploration = performance_history['exploration_rate'][-100:]
    
    # Calculate trends
    win_rate_trend = np.polyfit(range(100), recent_win_rates, 1)[0]
    return_trend = np.polyfit(range(100), recent_returns, 1)[0]
    
    # Learning indicators
    win_rate_improving = win_rate_trend > 0.0001
    returns_improving = return_trend > 0.01
    win_rate_stable = np.std(recent_win_rates[-20:]) < 0.15
    
    # Check for problems
    problems = []
    
    # 1. Stagnant learning
    if abs(win_rate_trend) < 0.00001 and np.mean(recent_win_rates) < 0.4:
        problems.append("⚠️ Learning appears stagnant - win rate not improving")
    
    # 2. High variance
    if np.std(recent_win_rates) > 0.3:
        problems.append("⚠️ High variance in performance - unstable learning")
    
    # 3. Declining performance
    if win_rate_trend < -0.0005:
        problems.append("🚨 Performance declining - possible overfitting or exploration issues")
    
    # 4. Low diversity
    if 'action_diversity' in performance_history and len(performance_history['action_diversity']) > 50:
        recent_diversity = np.mean(performance_history['action_diversity'][-50:])
        if recent_diversity < 0.5:
            problems.append("⚠️ Low action diversity - model may be stuck in suboptimal strategy")
    
    # 5. Poor risk-adjusted returns
    if 'risk_adjusted_return' in performance_history and len(performance_history['risk_adjusted_return']) > 20:
        recent_sharpe = np.mean(performance_history['risk_adjusted_return'][-20:])
        if recent_sharpe < -0.5:
            problems.append("⚠️ Poor risk-adjusted returns - high volatility relative to returns")
    
    # Report findings
    if episode % 100 == 0:  # Detailed report every 100 episodes
        logger.info("\n" + "="*60)
        logger.info("📊 LEARNING QUALITY ANALYSIS")
        logger.info("="*60)
        
        # Current performance
        current_win_rate = np.mean(recent_win_rates[-10:])
        current_return = np.mean(recent_returns[-10:])
        logger.info(f"Current Performance (last 10 eps): WR={current_win_rate:.1%}, Return={current_return:.1f}")
        
        # Trends
        logger.info(f"Win Rate Trend: {'+' if win_rate_improving else '-'}{abs(win_rate_trend)*100:.3f}% per episode")
        logger.info(f"Return Trend: {'+' if returns_improving else '-'}{abs(return_trend):.2f} per episode")
        
        # Best performance
        if len(performance_history['win_rate_ma_50']) > 0:
            best_ma50_wr = max(performance_history['win_rate_ma_50'])
            current_ma50_wr = performance_history['win_rate_ma_50'][-1]
            logger.info(f"Best 50-MA Win Rate: {best_ma50_wr:.1%} (current: {current_ma50_wr:.1%})")
        
        # Learning efficiency
        total_episodes = len(performance_history['episode'])
        episodes_above_50_wr = sum(1 for wr in performance_history['win_rate'] if wr > 0.5)
        learning_efficiency = episodes_above_50_wr / total_episodes if total_episodes > 0 else 0
        logger.info(f"Learning Efficiency: {learning_efficiency:.1%} episodes with >50% win rate")
        
        # Report problems
        if problems:
            logger.warning("\n⚠️ POTENTIAL ISSUES DETECTED:")
            for problem in problems:
                logger.warning(problem)
            
            # Suggestions
            logger.info("\n💡 SUGGESTIONS:")
            if "stagnant" in str(problems):
                logger.info("- Consider increasing exploration rate temporarily")
                logger.info("- Check if reward shaping is appropriate")
            if "High variance" in str(problems):
                logger.info("- Reduce learning rate")
                logger.info("- Increase batch size")
            if "declining" in str(problems):
                logger.info("- Reduce learning rate")
                logger.info("- Check for overfitting")
            if "Low action diversity" in str(problems):
                logger.info("- Increase exploration rate")
                logger.info("- Add entropy bonus to encourage exploration")
        else:
            logger.info("✅ Learning appears healthy!")
        
        logger.info("="*60 + "\n")


def export_performance_visualization(performance_history, output_path='training_performance.json'):
    """Export performance data for visualization"""
    try:
        import json
        
        # Prepare data for export with safe gets
        export_data = {
            'episodes': performance_history.get('episode', []),
            'win_rates': performance_history.get('win_rate', []),
            'returns': performance_history.get('avg_return', []),
            'win_rate_ma50': performance_history.get('win_rate_ma_50', []),
            'win_rate_ma200': performance_history.get('win_rate_ma_200', []),
            'profit_factors': performance_history.get('profit_factor', []),
            'risk_adjusted_returns': performance_history.get('risk_adjusted_return', []),
            'action_diversity': performance_history.get('action_diversity', []),
            'exploration_rates': performance_history.get('exploration_rate', []),
            'avg_wins': performance_history.get('avg_win', []),
            'avg_losses': performance_history.get('avg_loss', []),
            'win_loss_ratios': performance_history.get('win_loss_ratio', []),
            'metadata': {
                'best_win_rate': max(performance_history.get('win_rate', [0])) if performance_history.get('win_rate') else 0,
                'best_return': max(performance_history.get('avg_return', [0])) if performance_history.get('avg_return') else 0,
                'total_episodes': len(performance_history.get('episode', [])),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"📊 Performance data exported to {output_path}")
        
        # Also create a simple CSV for Excel
        csv_path = output_path.replace('.json', '.csv')
        episodes = performance_history.get('episode', [])
        if episodes:
            with open(csv_path, 'w') as f:
                f.write("Episode,Win Rate,Return,MA50,MA200,Profit Factor,Sharpe,Action Diversity\n")
                for i in range(len(episodes)):
                    f.write(f"{episodes[i]},")
                    f.write(f"{performance_history.get('win_rate', [0]*len(episodes))[i]:.4f},")
                    f.write(f"{performance_history.get('avg_return', [0]*len(episodes))[i]:.2f},")
                    f.write(f"{performance_history.get('win_rate_ma_50', [0]*len(episodes))[i]:.4f},")
                    f.write(f"{performance_history.get('win_rate_ma_200', [0]*len(episodes))[i]:.4f},")
                    f.write(f"{performance_history.get('profit_factor', [0]*len(episodes))[i]:.2f},")
                    f.write(f"{performance_history.get('risk_adjusted_return', [0]*len(episodes))[i]:.2f},")
                    f.write(f"{performance_history.get('action_diversity', [0]*len(episodes))[i]:.2f}\n")
        
        logger.info(f"📊 CSV data exported to {csv_path}")
        
    except Exception as e:
        logger.error(f"Failed to export performance data: {e}")


def find_latest_checkpoint():
    """Find the latest checkpoint in the checkpoint directory"""
    checkpoint_dir = 'checkpoints/profitable_optimized'
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Look for checkpoint files
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith('checkpoint_episode_') and file.endswith('.pt'):
            try:
                # Extract episode number from filename
                episode_num = int(file.replace('checkpoint_episode_', '').replace('.pt', ''))
                checkpoint_files.append((episode_num, os.path.join(checkpoint_dir, file)))
            except ValueError:
                continue
    
    # Also check for emergency checkpoints
    for file in os.listdir(checkpoint_dir):
        if file.startswith('emergency_checkpoint_episode_') and file.endswith('.pt'):
            try:
                # Extract episode number from filename
                episode_num = int(file.replace('emergency_checkpoint_episode_', '').replace('.pt', ''))
                checkpoint_files.append((episode_num, os.path.join(checkpoint_dir, file)))
            except ValueError:
                continue
    
    if not checkpoint_files:
        return None
    
    # Sort by episode number and return the latest
    checkpoint_files.sort(key=lambda x: x[0], reverse=True)
    latest_episode, latest_path = checkpoint_files[0]
    
    logger.info(f"📂 Found latest checkpoint: Episode {latest_episode} at {latest_path}")
    return latest_path


def train(num_episodes=10000, save_interval=100, use_real_data=True, resume=False, checkpoint_path=None, no_distributed=False):
    """Main training function that spawns distributed processes"""
    
    # Check if we have multiple GPUs
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        
        if world_size > 1 and not no_distributed and not args.force_single_gpu:
            logger.info(f"Starting distributed training on {world_size} GPUs")
            
            # Find a free port to avoid conflicts
            free_port = find_free_port()
            os.environ['MASTER_PORT'] = str(free_port)
            logger.info(f"Using port {free_port} for distributed training")
            
            # Spawn processes for distributed training
            mp.spawn(
                train_distributed,
                args=(world_size, num_episodes, save_interval, use_real_data, resume, checkpoint_path),
                nprocs=world_size,
                join=True
            )
        else:
            if no_distributed or args.force_single_gpu:
                logger.info("Single GPU mode (distributed training disabled)")
            else:
                logger.info("Single GPU detected, running standard training")
            # Run single GPU training
            train_distributed(0, 1, num_episodes, save_interval, use_real_data, resume, checkpoint_path)
    else:
        logger.error("No GPU available! This training script requires GPU.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes')
    parser.add_argument('--save-interval', type=int, default=100, help='Save interval')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path')
    parser.add_argument('--no-distributed', action='store_true', help='Disable distributed training')
    parser.add_argument('--no-auto-resume', action='store_true', help='Disable automatic resume from latest checkpoint')
    parser.add_argument('--use-latest', action='store_true', help='Use latest checkpoint instead of best model when resuming')
    parser.add_argument('--force-single-gpu', action='store_true', help='Force single GPU mode even with multiple GPUs')
    args = parser.parse_args()
    
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Auto-resume logic: if no explicit resume/checkpoint args, try to find latest checkpoint
    resume = args.resume
    checkpoint_path = args.checkpoint
    
    if not resume and not checkpoint_path and not args.no_auto_resume:
        # Priority order: best_avg_model > best_model > latest checkpoint
        best_avg_model_path = 'checkpoints/profitable_optimized/best_avg_model.pt'
        best_model_path = 'checkpoints/profitable_optimized/best_model.pt'
        
        if not args.use_latest and os.path.exists(best_avg_model_path):
            logger.info(f"🏆 Found best average model at {best_avg_model_path}")
            logger.info("   Using best average model for resume (highest priority)")
            logger.info("   (Use --use-latest to resume from latest checkpoint instead)")
            resume = True
            checkpoint_path = best_avg_model_path
        elif not args.use_latest and os.path.exists(best_model_path):
            logger.info(f"🎯 Found best single episode model at {best_model_path}")
            logger.info("   Using best single episode model for resume")
            logger.info("   (Use --use-latest to resume from latest checkpoint instead)")
            resume = True
            checkpoint_path = best_model_path
        else:
            # Try to find the latest checkpoint automatically
            latest_checkpoint = find_latest_checkpoint()
            if latest_checkpoint:
                logger.info(f"🔄 Auto-resuming from latest checkpoint: {latest_checkpoint}")
                logger.info("   (Use --no-auto-resume to start fresh training)")
                resume = True
                checkpoint_path = latest_checkpoint
            else:
                logger.info("🆕 No existing checkpoints found, starting fresh training")
    
    train(
        num_episodes=args.episodes,
        save_interval=args.save_interval,
        resume=resume,
        checkpoint_path=checkpoint_path,
        no_distributed=args.no_distributed
    )