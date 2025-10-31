#!/usr/bin/env python3
"""Optimized training script with significant performance improvements"""

import os
import sys
import torch
import torch.nn as nn
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

# Live trading imports
import alpaca_trade_api as tradeapi
from src.options_data_collector import AlpacaOptionsDataCollector
from typing import Union, Dict, List, Optional, Tuple, Any
import traceback
import asyncio
from dotenv import load_dotenv
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import deque


def safe_save_checkpoint(checkpoint_data, filepath, logger=None):
    """
    Safely save a checkpoint using torch.save with proper error handling.
    This function helps avoid the 'weights_only load failed' error by:
    1. Using map_location when loading
    2. Saving with proper metadata
    3. Including version information for compatibility
    """
    try:
        # Add metadata for safe loading
        checkpoint_data['_save_metadata'] = {
            'pytorch_version': torch.__version__,
            'save_timestamp': datetime.now().isoformat(),
            'safe_format': True
        }
        
        # Save the checkpoint
        torch.save(checkpoint_data, filepath)
        
        # Verify the checkpoint can be loaded (optional)
        try:
            test_load = torch.load(filepath, map_location='cpu', weights_only=False)
            del test_load  # Free memory
            if logger:
                logger.debug(f"Checkpoint saved and verified: {filepath}")
        except Exception as verify_error:
            if logger:
                logger.warning(f"Checkpoint saved but verification failed: {verify_error}")
        
        return True
    except Exception as e:
        if logger:
            logger.error(f"Failed to save checkpoint to {filepath}: {e}")
        return False


def safe_load_checkpoint(filepath, device='cpu', logger=None):
    """
    Safely load a checkpoint with proper error handling.
    Handles the 'weights_only load failed' error gracefully.
    """
    try:
        # First try with weights_only=True (most secure)
        try:
            checkpoint = torch.load(filepath, map_location=device, weights_only=True)
            if logger:
                logger.info(f"Checkpoint loaded securely (weights_only=True): {filepath}")
            return checkpoint
        except Exception as weights_only_error:
            # If weights_only fails, try with weights_only=False
            if logger:
                logger.warning(f"Weights-only load failed, attempting full load: {weights_only_error}")
            
            checkpoint = torch.load(filepath, map_location=device, weights_only=False)
            if logger:
                logger.info(f"Checkpoint loaded with full compatibility mode: {filepath}")
            return checkpoint
            
    except Exception as e:
        if logger:
            logger.error(f"Failed to load checkpoint from {filepath}: {e}")
        return None


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


# Add symbol encoding mixin
class SymbolEncodingMixin:
    """Mixin to add symbol encoding functionality to environments"""
    
    def _get_symbol_encoding(self, symbol):
        """Create one-hot encoding for symbol to enable symbol-specific strategies"""
        # Create a mapping of symbols to indices if not exists
        if not hasattr(self, 'symbol_to_idx'):
            self.symbol_to_idx = {s: i for i, s in enumerate(self.symbols)}
        
        # Create one-hot encoding
        encoding = np.zeros(len(self.symbols), dtype=np.float32)
        if symbol in self.symbol_to_idx:
            encoding[self.symbol_to_idx[symbol]] = 1.0
        
        return encoding

import multiprocessing as cpu_mp
from typing import List, Tuple, Any

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = threading.Event()


class UltraFastEnvironment(HistoricalOptionsEnvironment, SymbolEncodingMixin):
    """Ultra-fast environment with minimal overhead"""
    
    def __init__(self, *args, **kwargs):
        # Initialize attributes BEFORE calling super().__init__
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_trades = 0
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
            """Simplified step focusing on trade returns"""
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
            
            # Simple reward based on trade returns
            reward = 0
            
            # Execute action
            if action_name == 'hold':
                # Small penalty for holding when no positions (encourage trading)
                if len(self.positions) == 0:
                    reward = -0.01
            elif action_name == 'close_all_positions' and self.positions:
                # Close all positions and get the total P&L as reward
                total_pnl = self._close_all_positions_realistic()
                # Scale P&L to reasonable reward range
                reward = total_pnl / 1000.0  # Divide by 1000 to normalize
            elif action_name in ['buy_call', 'buy_put'] and len(self.positions) < self.max_positions:
                # Execute trade (small penalty for transaction cost)
                trade_reward = self._execute_trade(action_name, current_price)
                reward = trade_reward if trade_reward < 0 else -0.01  # Transaction cost
            
            # Update positions and check for exits
            closed_pnl = self._update_positions_realistic()
            
            # Add P&L from any positions that were closed this step
            if closed_pnl != 0:
                reward += closed_pnl / 1000.0  # Normalize by dividing by 1000
            
            # Update step
            self.current_step += 1
            
            # Done conditions
            portfolio_value = self._calculate_portfolio_value_fast()
            if self.current_step >= self._data_length - 1:
                self.done = True
            elif portfolio_value < self.initial_capital * 0.2:  # 80% loss
                self.done = True
                reward -= 1.0  # Additional penalty for blowing up the account
            
            return self._get_observation(), reward, self.done, {
                'portfolio_value': portfolio_value,
                'positions': len(self.positions),
                'symbol': self.current_symbol,
                'win_rate': self.winning_trades / max(1, self.winning_trades + self.losing_trades) if hasattr(self, 'winning_trades') else 0
            }

class BalancedEnvironment(HistoricalOptionsEnvironment, SymbolEncodingMixin):
    """Balanced environment with realistic features and good performance"""
    
    def __init__(self, *args, **kwargs):
        # Initialize attributes BEFORE calling super().__init__
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_trades = 0
        self.total_pnl = 0
        self.consecutive_losses = 0
        self.price_history = []
        self._cache = {}
        self._position_cache = {}
        
        # Realistic trading parameters with enhanced market awareness
        self.max_loss_per_trade = 0.03  # 3% max loss - more room for options volatility
        self.max_profit_per_trade = 0.02  # 2% take profit - more achievable target
        self.volatility_window = 20
        self.use_dynamic_exits = True  # Enable dynamic exit strategies
        self._last_action = 'hold'  # Initialize last action to prevent NameError
        
        # Market regime tracking
        self._market_regime_history = []
        self._regime_confidence = 0.0
        
        # Risk tracking
        self._position_risk_history = []
        self._max_drawdown = 0.0
        self._peak_value = 100000  # Will be updated after parent init
        self._recent_returns = deque(maxlen=50)
        
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
        
        # Update peak value with actual initial capital
        self._peak_value = self.initial_capital
        
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
        
        # Risk-based action filtering during drawdowns
        current_drawdown = self._calculate_max_drawdown()
        position_risk = self._calculate_position_risk()
        
        # Initialize reward
        reward = 0
        
        # Prevent risky actions during significant drawdown
        if current_drawdown > 0.15 and position_risk > 0.6:
            # Force conservative actions during drawdown with high risk
            if action_name in ['buy_call', 'buy_put'] and len(self.positions) >= 2:
                action_name = 'hold'  # Override to hold
                reward = -2  # Small penalty for attempted risky action during drawdown
        
        # Execute action
        if action_name == 'hold':
            pass
        elif action_name == 'close_all_positions' and self.positions:
            reward = self._close_all_positions_realistic()
            # Bonus for risk reduction during drawdown
            if current_drawdown > 0.1:
                reward += 3  # Encourage de-risking during drawdowns
        elif action_name in ['buy_call', 'buy_put'] and len(self.positions) < self.max_positions:
            # Check if we should allow new positions based on risk
            if current_drawdown > 0.2 and len(self.positions) > 0:
                # During severe drawdown, limit new positions
                reward = -1  # Penalty for trying to add risk during drawdown
            else:
                reward = self._execute_trade(action_name, current_price)
        
        # Update positions with realistic P&L
        self._update_positions_realistic()
        
        # Calculate reward based on portfolio change
        portfolio_value_after = self._calculate_portfolio_value_fast()
        step_pnl = portfolio_value_after - portfolio_value_before
        
        # Advanced reward shaping with risk-based adjustments
        base_reward = step_pnl / 1000  # Normalized P&L
        
        # Calculate risk metrics for this step
        position_risk = self._calculate_position_risk()
        volatility = self._calculate_volatility()
        leverage = self._calculate_leverage()
        
        # Risk-adjusted reward calculation
        if step_pnl > 0:  # Profitable trade
            # Bonus for profitable risky trades
            if position_risk > 0.7:  # High risk position
                risk_bonus = base_reward * 2.0  # Double reward for successful high-risk trades
                reward += base_reward + risk_bonus
            elif position_risk > 0.5:  # Medium risk
                risk_bonus = base_reward * 1.5
                reward += base_reward + risk_bonus
            else:  # Low risk
                reward += base_reward
        else:  # Loss
            # Extra penalty for failed risky trades
            if position_risk > 0.7:  # High risk position failed
                risk_penalty = abs(base_reward) * 2.5  # 2.5x penalty for failed high-risk trades
                reward += base_reward - risk_penalty
            elif position_risk > 0.5:  # Medium risk failed
                risk_penalty = abs(base_reward) * 1.5
                reward += base_reward - risk_penalty
            else:  # Low risk loss
                reward += base_reward
        
        # Sharpe ratio reward component (risk-adjusted returns)
        if hasattr(self, '_recent_returns'):
            self._recent_returns.append(step_pnl / portfolio_value_before if portfolio_value_before > 0 else 0)
            if len(self._recent_returns) > 20:
                returns_array = np.array(list(self._recent_returns))
                if np.std(returns_array) > 0:
                    sharpe = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)  # Annualized
                    reward += sharpe * 0.5  # Reward good risk-adjusted returns
        else:
            self._recent_returns = deque(maxlen=50)
        
        # Win rate bonus with risk consideration
        if self.winning_trades + self.losing_trades >= 5:
            win_rate = self.winning_trades / max(1, self.winning_trades + self.losing_trades)
            avg_position_risk = np.mean(self._position_risk_history) if hasattr(self, '_position_risk_history') and self._position_risk_history else 0.5
            
            if win_rate > 0.7:
                if avg_position_risk > 0.6:  # High win rate with high risk
                    reward += 25  # Exceptional - managing risk well
                else:
                    reward += 15  # Good win rate with normal risk
            elif win_rate > 0.6:
                if avg_position_risk > 0.6:
                    reward += 15  # Good performance with high risk
                else:
                    reward += 8
            elif win_rate > 0.5:
                reward += 4
            elif win_rate < 0.3:
                if avg_position_risk > 0.6:
                    reward -= 15  # Terrible - high risk and low win rate
                else:
                    reward -= 5
        
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
        
        # Risk management - penalize consecutive losses with risk consideration
        if self.consecutive_losses > 5:
            # Extra penalty if taking high risks during losing streak
            if position_risk > 0.6:
                reward -= self.consecutive_losses * 5  # Severe penalty for high risk during drawdown
            else:
                reward -= self.consecutive_losses * 3
        elif self.consecutive_losses > 3:
            if position_risk > 0.6:
                reward -= self.consecutive_losses * 3
            else:
                reward -= self.consecutive_losses * 2
        elif self.consecutive_losses == 0 and self.winning_trades > 5:
            # Bonus for consistent winning, extra if managing risk well
            if position_risk < 0.4:
                reward += 8  # Conservative risk with good results
            else:
                reward += 5
        
        # Drawdown penalty
        current_drawdown = self._calculate_max_drawdown()
        if current_drawdown > 0.2:  # 20% drawdown
            reward -= 10 * current_drawdown  # Progressive penalty
        elif current_drawdown > 0.1:  # 10% drawdown
            reward -= 5 * current_drawdown
        
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
        
        # Market timing and risk management bonus
        if hasattr(self, '_last_action') and self._last_action in ['buy_call', 'buy_put']:
            # Calculate entry quality
            momentum = self._calculate_momentum()
            rsi = self._calculate_rsi()
            
            # Good entry conditions
            if action_name == 'buy_call' and momentum > 0.01 and 40 <= rsi <= 65:
                entry_quality = 1.0
            elif action_name == 'buy_put' and momentum < -0.01 and 35 <= rsi <= 60:
                entry_quality = 1.0
            else:
                entry_quality = 0.5
            
            # Risk-adjusted entry bonus
            if step_pnl > 0:  # Immediate profit
                if position_risk > 0.6:  # High risk entry that worked
                    reward += 5 * entry_quality
                else:
                    reward += 2 * entry_quality
            elif volatility > 0.025:  # Entered during high volatility
                if step_pnl < 0:
                    reward -= 2  # Penalty for bad timing in volatile market
                else:
                    reward += 1  # Small bonus for navigating volatility
        
        # Leverage penalty - discourage excessive leverage
        if leverage > 5:
            reward -= (leverage - 5) * 2  # Progressive penalty above 5x leverage
        elif leverage > 3:
            reward -= (leverage - 3) * 0.5
            
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
            # Print trade signal summary if in live mode
            if hasattr(self, 'trade_signals') and self.trade_signals:
                logger.info(f"\n📊 Episode Trade Signals Summary:")
                logger.info(f"   Total signals: {len(self.trade_signals)}")
                buy_calls = sum(1 for s in self.trade_signals if s['action'] == 'buy_call')
                buy_puts = sum(1 for s in self.trade_signals if s['action'] == 'buy_put')
                logger.info(f"   Buy calls: {buy_calls}, Buy puts: {buy_puts}")
                if self.trade_signals:
                    avg_momentum = np.mean([s['momentum'] for s in self.trade_signals])
                    logger.info(f"   Average momentum at signal: {avg_momentum:.3f}")
        elif portfolio_value_after < self.initial_capital * 0.5:  # 50% loss
            self.done = True
            reward -= 100
        
        return self._get_observation(), reward, self.done, {
            'portfolio_value': portfolio_value_after,
            'positions': len(self.positions),
            'symbol': self.current_symbol,
            'win_rate': self.winning_trades / max(1, self.winning_trades + self.losing_trades)
        }
    
    def _validate_trade_entry(self, action_name, current_price, option_data=None):
        """Comprehensive trade validation before entry"""
        validation_score = 0.0
        reasons = []
        
        # Get market conditions
        momentum = self._calculate_momentum()
        rsi = self._calculate_rsi()
        volatility = self._calculate_volatility()
        market_regime = self._detect_market_regime()
        
        # 1. Market Regime Validation
        if market_regime == 'volatile' and volatility > 0.03:
            validation_score -= 0.3
            reasons.append("High volatility environment")
        elif market_regime == 'trending':
            if (action_name == 'buy_call' and momentum > 0) or (action_name == 'buy_put' and momentum < 0):
                validation_score += 0.2
                reasons.append("Aligned with trend")
            else:
                validation_score -= 0.4
                reasons.append("Against trend")
        
        # 2. Technical Indicator Validation
        if action_name == 'buy_call':
            if momentum < 0:
                validation_score -= 0.3
                reasons.append("Negative momentum for call")
            if rsi > 70:
                validation_score -= 0.2
                reasons.append("Overbought RSI")
            if 40 <= rsi <= 60 and momentum > 0.01:
                validation_score += 0.3
                reasons.append("Good technical setup")
        else:  # buy_put
            if momentum > 0:
                validation_score -= 0.3
                reasons.append("Positive momentum for put")
            if rsi < 30:
                validation_score -= 0.2
                reasons.append("Oversold RSI")
            if 40 <= rsi <= 60 and momentum < -0.01:
                validation_score += 0.3
                reasons.append("Good technical setup")
        
        # 3. Option Pricing Validation (if option data provided)
        if option_data:
            bid = option_data.get('bid', 0)
            ask = option_data.get('ask', 0)
            volume = option_data.get('volume', 0)
            
            # Spread validation
            if ask > 0 and bid > 0:
                spread_pct = (ask - bid) / ask
                if spread_pct > 0.1:  # More than 10% spread
                    validation_score -= 0.2
                    reasons.append("Wide bid-ask spread")
                elif spread_pct < 0.05:  # Less than 5% spread
                    validation_score += 0.1
                    reasons.append("Tight spread")
            
            # Volume validation
            if volume < 100:
                validation_score -= 0.2
                reasons.append("Low volume")
            elif volume > 1000:
                validation_score += 0.1
                reasons.append("Good liquidity")
            
            # Price sanity check
            mid_price = (bid + ask) / 2
            if mid_price < 0.10:  # Too cheap, likely to expire worthless
                validation_score -= 0.3
                reasons.append("Option too cheap")
            elif mid_price > current_price * 0.1:  # Too expensive
                validation_score -= 0.2
                reasons.append("Option too expensive")
        
        # 4. Risk/Reward Validation
        if hasattr(self, 'winning_trades') and hasattr(self, 'losing_trades'):
            total_trades = self.winning_trades + self.losing_trades
            if total_trades > 10:
                win_rate = self.winning_trades / total_trades
                if win_rate < 0.3:  # Poor historical performance
                    validation_score -= 0.2
                    reasons.append("Low historical win rate")
        
        # 5. Portfolio Risk Check
        if len(self.positions) >= self.max_positions - 1:
            validation_score -= 0.3
            reasons.append("Near position limit")
        
        current_exposure = sum([pos.get('cost', 0) for pos in self.positions])
        if current_exposure > self.capital * 0.5:
            validation_score -= 0.3
            reasons.append("High portfolio exposure")
        
        # 6. Momentum Confirmation Check
        if len(self.price_history) >= 10:
            # Check multiple timeframes for confirmation
            momentum_3 = self._calculate_momentum_window(3)
            momentum_5 = self._calculate_momentum_window(5)
            momentum_10 = momentum  # Already calculated
            
            if action_name == 'buy_call':
                confirmations = sum([1 for m in [momentum_3, momentum_5, momentum_10] if m > 0])
                if confirmations == 3:
                    validation_score += 0.2
                    reasons.append("Strong bullish momentum confirmation")
                elif confirmations == 0:
                    validation_score -= 0.3
                    reasons.append("No bullish momentum confirmation")
                    
                # Check for momentum divergence
                if momentum_3 < 0 and momentum_10 > 0:
                    validation_score -= 0.2
                    reasons.append("Bearish momentum divergence")
                    
            elif action_name == 'buy_put':
                confirmations = sum([1 for m in [momentum_3, momentum_5, momentum_10] if m < 0])
                if confirmations == 3:
                    validation_score += 0.2
                    reasons.append("Strong bearish momentum confirmation")
                elif confirmations == 0:
                    validation_score -= 0.3
                    reasons.append("No bearish momentum confirmation")
                    
                # Check for momentum divergence
                if momentum_3 > 0 and momentum_10 < 0:
                    validation_score -= 0.2
                    reasons.append("Bullish momentum divergence")
        
        return validation_score, reasons
    
    def _execute_live_trade(self, action_name, current_price):
        """Execute a real trade through Alpaca API"""
        try:
            # Check if we've already determined API is invalid
            if hasattr(self, '_api_invalid') and self._api_invalid:
                return
            
            config = __builtins__.live_trading_config
            
            # Create API connection if not already created
            if not hasattr(self, '_live_api'):
                try:
                    self._live_api = tradeapi.REST(
                        config['api_key'],
                        config['api_secret'],
                        config['base_url']
                    )
                    # Test the connection
                    account = self._live_api.get_account()
                    self._live_buying_power = float(account.buying_power)
                    logger.info(f"✅ Live API connected - Buying power: ${self._live_buying_power:,.2f}")
                except Exception as e:
                    if "not authorized" in str(e):
                        logger.warning("❌ Alpaca API not authorized - switching to simulation mode")
                        logger.info("💡 To enable live trading:")
                        logger.info("   1. Get API keys from https://alpaca.markets/")
                        logger.info("   2. Add to .env file: ALPACA_API_KEY and ALPACA_SECRET_KEY")
                    else:
                        logger.error(f"Failed to connect to Alpaca API: {e}")
                    self._api_invalid = True
                    return
            
            # Get current symbol
            symbol = self.current_symbol
            
            # Calculate position size
            position_size_dollars = self._live_buying_power * config['position_size_pct']
            
            # Log the trade signal (simulation mode)
            logger.info(f"📈 TRADE SIGNAL: {action_name} for {symbol}")
            logger.info(f"   Current price: ${current_price:.2f}")
            logger.info(f"   Position size: ${position_size_dollars:.2f}")
            logger.info(f"   Momentum: {self._calculate_momentum():.3f}")
            logger.info(f"   RSI: {self._calculate_rsi():.1f}")
            logger.info(f"   Market regime: {self._detect_market_regime()}")
            
            # In a real implementation with valid API:
            # 1. Find the appropriate option contract using AlpacaOptionsDataCollector
            # 2. Place the order through Alpaca's options API
            # 3. Track the position in real_positions list
            
            # Track trade signals
            if not hasattr(self, 'trade_signals'):
                self.trade_signals = []
            
            self.trade_signals.append({
                'timestamp': datetime.now(),
                'action': action_name,
                'symbol': symbol,
                'price': current_price,
                'momentum': self._calculate_momentum(),
                'rsi': self._calculate_rsi(),
                'size': position_size_dollars
            })
            
            # Track that we attempted a live trade
            if not hasattr(self, 'live_trade_attempts'):
                self.live_trade_attempts = 0
            self.live_trade_attempts += 1
            
        except Exception as e:
            logger.error(f"Unexpected error in live trade execution: {e}")
            import traceback
            traceback.print_exc()
    
    def _execute_trade(self, action_name, current_price):
            """Execute trade with minimal validation"""
            current_time = self._timestamps[self.current_step]
            
            # Check if we should execute a live trade
            if hasattr(__builtins__, 'live_trading_config') and __builtins__.live_trading_config.get('enabled'):
                # Randomly decide whether to execute this trade live based on execution probability
                if np.random.random() < __builtins__.live_trading_config.get('execution_probability', 0.1):
                    self._execute_live_trade(action_name, current_price)
            
            # Simple momentum check for basic filtering
            if len(self.price_history) >= 5:
                momentum = (self.price_history[-1] - self.price_history[-5]) / self.price_history[-5]
                
                # Basic directional filter
                if action_name == 'buy_call' and momentum < -0.02:
                    return -0.01  # Small penalty for buying calls in downtrend
                elif action_name == 'buy_put' and momentum > 0.02:
                    return -0.01  # Small penalty for buying puts in uptrend
            
            # Find suitable option
            cache_key = (self.current_step, action_name)
            if cache_key in self._cache:
                option = self._cache[cache_key]
            else:
                # Find options at current timestamp
                time_mask = self._option_data['timestamps'] == current_time
                type_mask = self._option_data['types'] == ('call' if 'call' in action_name else 'put')
                money_mask = (self._option_data['moneyness'] >= 0.95) & (self._option_data['moneyness'] <= 1.05)
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
                    return 0
            
            option = self._cache[cache_key]
            
            if option is not None:
                # Calculate position size (use larger percentage for better returns)
                mid_price = (option['bid'] + option['ask']) / 2
                # Dynamic position sizing based on performance
                total_trades = getattr(self, 'total_trades', self.winning_trades + self.losing_trades)
                if total_trades < 50:
                    size_pct = 0.30  # 30% during early exploration
                elif self.winning_trades / max(1, total_trades) < 0.3:
                    size_pct = 0.25  # 25% when struggling
                else:
                    size_pct = 0.20  # 20% normally
                position_size = min(20, max(2, int(self.capital * size_pct / (mid_price * 100))))
                
                if position_size > 0:
                    cost = position_size * option['ask'] * 100 + self.commission
                    if cost <= self.capital * 0.5:  # Don't use more than 50% of capital
                        self.positions.append({
                            'entry_price': option['ask'],
                            'quantity': position_size,
                            'entry_step': self.current_step,
                            'entry_underlying': current_price,
                            'option_type': 'call' if 'call' in action_name else 'put',
                            'strike': option['strike']
                        })
                        self.capital -= cost
                        return 0  # No immediate reward for opening position
            
            return 0
    def _update_positions_realistic(self):
            """Update positions and return total P&L from closed positions"""
            if self.current_step >= self._data_length:
                return 0
            
            current_time = self._timestamps[self.current_step]
            current_price = self._prices[self.current_step]
            positions_to_close = []
            total_closed_pnl = 0
            
            for i, pos in enumerate(self.positions):
                # Calculate position age
                position_age = self.current_step - pos['entry_step']
                entry_underlying = pos['entry_underlying']
                
                # Calculate price movement
                price_change = (current_price - entry_underlying) / entry_underlying
                
                # Simple option P&L calculation
                if pos['option_type'] == 'call':
                    option_price_change = price_change * 1.5 if price_change > 0 else price_change * 0.8
                else:  # put
                    option_price_change = -price_change * 1.5 if price_change < 0 else -price_change * 0.8
                
                # Calculate P&L
                entry_cost = pos['entry_price'] * pos['quantity'] * 100
                current_value = pos['entry_price'] * (1 + option_price_change) * pos['quantity'] * 100
                current_value = max(0, current_value)
                
                pnl = current_value - entry_cost - self.commission
                pnl_pct = pnl / entry_cost
                
                # Calculate position-specific risk
                pos_size_risk = min(1.0, entry_cost / (self.capital * 0.2))
                pos_time_risk = min(1.0, position_age / 30)
                position_risk_score = (pos_size_risk + pos_time_risk) / 2
                
                # Dynamic exit rules based on risk
                should_exit = False
                exit_reason = None
                
                # Risk-adjusted stop loss
                if position_risk_score > 0.7:  # High risk position
                    stop_loss = -0.08  # 8% stop for risky positions (was 15%)
                else:
                    stop_loss = -0.10  # 10% normal stop loss (was 20%)
                
                # Risk-adjusted take profit
                if position_risk_score > 0.7:  # High risk position
                    take_profit = 0.15  # 15% take profit for risky positions (was 25%)
                else:
                    take_profit = 0.20  # 20% normal take profit (was 30%)
                
                # Exit decisions
                if pnl_pct <= stop_loss:
                    should_exit = True
                    exit_reason = 'stop_loss'
                    if hasattr(self, 'losing_trades'):
                        self.losing_trades += 1
                    if hasattr(self, 'total_trades'):
                        self.total_trades += 1
                    if hasattr(self, 'consecutive_losses'):
                        self.consecutive_losses += 1
                
                elif pnl_pct >= take_profit:
                    should_exit = True
                    exit_reason = 'take_profit'
                    if hasattr(self, 'winning_trades'):
                        self.winning_trades += 1
                    if hasattr(self, 'total_trades'):
                        self.total_trades += 1
                    if hasattr(self, 'consecutive_losses'):
                        self.consecutive_losses = 0
                
                # Time-based exit with risk consideration
                elif position_age > (15 if position_risk_score > 0.7 else 20):
                    should_exit = True
                    exit_reason = 'time_exit'
                    if pnl > 0:
                        if hasattr(self, 'winning_trades'):
                            self.winning_trades += 1
                        if hasattr(self, 'consecutive_losses'):
                            self.consecutive_losses = 0
                    else:
                        if hasattr(self, 'losing_trades'):
                            self.losing_trades += 1
                        if hasattr(self, 'consecutive_losses'):
                            self.consecutive_losses += 1
                
                if should_exit:
                    positions_to_close.append(i)
                    self.capital += current_value - self.commission
                    total_closed_pnl += pnl
                    if hasattr(self, 'total_pnl'):
                        self.total_pnl += pnl
                    
                    # Track recent PnLs for consistency analysis
                    if hasattr(self, 'recent_pnls'):
                        self.recent_pnls.append(pnl_pct)
                    
                    # Calculate risk-adjusted trade reward
                    if hasattr(self, 'last_trade_reward'):
                        if pnl > 0:
                            # Bonus for profitable risky trades
                            if position_risk_score > 0.7:
                                self.last_trade_reward = pnl / 500  # Double reward for risky wins
                            else:
                                self.last_trade_reward = pnl / 1000
                        else:
                            # Extra penalty for failed risky trades
                            if position_risk_score > 0.7:
                                self.last_trade_reward = pnl / 400  # Harsher penalty for risky losses
                            else:
                                self.last_trade_reward = pnl / 1000
            
            # Remove closed positions
            for i in reversed(positions_to_close):
                self.positions.pop(i)
            
            return total_closed_pnl
    
    def _close_all_positions_realistic(self):
            """Close all positions and return total P&L"""
            total_pnl = 0
            current_price = self._prices[self.current_step]
            
            for pos in self.positions:
                # Calculate P&L
                entry_underlying = pos['entry_underlying']
                price_change = (current_price - entry_underlying) / entry_underlying
                
                if pos['option_type'] == 'call':
                    option_price_change = price_change * 1.5 if price_change > 0 else price_change * 0.8
                else:  # put
                    option_price_change = -price_change * 1.5 if price_change < 0 else -price_change * 0.8
                
                # Calculate final value
                entry_cost = pos['entry_price'] * pos['quantity'] * 100
                current_value = pos['entry_price'] * (1 + option_price_change) * pos['quantity'] * 100
                current_value = max(0, current_value)
                
                pnl = current_value - entry_cost - self.commission
                
                self.capital += current_value - self.commission
                total_pnl += pnl
                
                if hasattr(self, 'total_pnl'):
                    self.total_pnl += pnl
                
                if pnl > 0 and hasattr(self, 'winning_trades'):
                    self.winning_trades += 1
                elif hasattr(self, 'losing_trades'):
                    self.losing_trades += 1
            
            self.positions = []
            return total_pnl
    def _calculate_portfolio_value_fast(self):
        """Fast portfolio value calculation"""
        position_value = 0
        
        if self.positions and self.current_step < self._data_length:
            current_price = self._prices[self.current_step]
            
            for pos in self.positions:
                # Calculate option value based on price movement
                entry_underlying = pos['entry_underlying']
                price_change = (current_price - entry_underlying) / entry_underlying
                
                # Simple option pricing based on underlying movement
                if pos['option_type'] == 'call':
                    option_price_change = price_change * 1.5 if price_change > 0 else price_change * 0.8
                else:  # put
                    option_price_change = -price_change * 1.5 if price_change < 0 else -price_change * 0.8
                
                # Calculate current value
                current_value = pos['entry_price'] * (1 + option_price_change) * pos['quantity'] * 100
                current_value = max(0, current_value)  # Options can't be negative
                
                position_value += current_value
        
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
    
    def _calculate_position_risk(self):
        """Calculate risk score for current positions (0-1, higher = riskier)"""
        if not self.positions:
            return 0.0
        
        risk_factors = []
        current_price = self._prices[self.current_step] if self.current_step < self._data_length else 100
        
        for pos in self.positions:
            # 1. Position size risk (larger positions = more risk)
            position_value = pos['entry_price'] * pos['quantity'] * 100
            size_risk = min(1.0, position_value / (self.capital * 0.2))  # 20% of capital is max normal size
            
            # 2. Time decay risk (options lose value over time)
            holding_time = self.current_step - pos['entry_step']
            time_risk = min(1.0, holding_time / 30)  # 30 steps is considered long hold
            
            # 3. Moneyness risk (OTM options are riskier)
            if pos['option_type'] == 'call':
                moneyness = current_price / pos.get('strike', current_price)
                if moneyness < 0.95:  # OTM call
                    moneyness_risk = 0.8
                elif moneyness > 1.05:  # ITM call
                    moneyness_risk = 0.3
                else:  # ATM
                    moneyness_risk = 0.5
            else:  # put
                moneyness = pos.get('strike', current_price) / current_price
                if moneyness < 0.95:  # OTM put
                    moneyness_risk = 0.8
                elif moneyness > 1.05:  # ITM put
                    moneyness_risk = 0.3
                else:  # ATM
                    moneyness_risk = 0.5
            
            # 4. P&L risk (losing positions are riskier to hold)
            entry_underlying = pos['entry_underlying']
            price_change = (current_price - entry_underlying) / entry_underlying
            if pos['option_type'] == 'call':
                pnl_direction = price_change
            else:
                pnl_direction = -price_change
            
            if pnl_direction < -0.02:  # Losing more than 2%
                pnl_risk = 0.8
            elif pnl_direction > 0.05:  # Winning more than 5%
                pnl_risk = 0.4  # Still some risk of giving back profits
            else:
                pnl_risk = 0.6
            
            # Combine risk factors
            position_risk = (size_risk * 0.3 + time_risk * 0.2 + moneyness_risk * 0.3 + pnl_risk * 0.2)
            risk_factors.append(position_risk)
        
        # Overall portfolio risk
        avg_position_risk = np.mean(risk_factors)
        concentration_risk = len(self.positions) / self.max_positions  # More positions = more risk
        volatility_risk = min(1.0, self._calculate_volatility() / 0.03)  # 3% vol is high
        
        total_risk = (avg_position_risk * 0.5 + concentration_risk * 0.3 + volatility_risk * 0.2)
        
        # Track risk history
        if not hasattr(self, '_position_risk_history'):
            self._position_risk_history = []
        self._position_risk_history.append(total_risk)
        if len(self._position_risk_history) > 100:
            self._position_risk_history.pop(0)
        
        return total_risk
    
    def _calculate_leverage(self):
        """Calculate effective leverage of current positions"""
        if not self.positions:
            return 0.0
        
        total_notional = 0
        for pos in self.positions:
            # Options provide leverage through control of 100 shares
            notional_value = pos.get('strike', 100) * pos['quantity'] * 100
            total_notional += notional_value
        
        # Leverage = notional exposure / capital
        leverage = total_notional / max(1, self.capital)
        return min(10.0, leverage)  # Cap at 10x for sanity
    
    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown from peak"""
        current_value = self._calculate_portfolio_value_fast()
        
        # Update peak
        if current_value > self._peak_value:
            self._peak_value = current_value
        
        # Calculate drawdown
        if self._peak_value > 0:
            drawdown = (self._peak_value - current_value) / self._peak_value
            self._max_drawdown = max(self._max_drawdown, drawdown)
        
        return self._max_drawdown
    
    def _calculate_momentum_window(self, window):
        """Calculate momentum over specified window"""
        if len(self.price_history) < window:
            return 0
        
        old_price = self.price_history[-window]
        current_price = self.price_history[-1]
        if old_price == 0:
            return 0
        return (current_price - old_price) / old_price
    
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


    def _get_enhanced_observation(self, base_obs):
        """Enhanced observation with properly integrated technical indicators"""
        if base_obs is None:
            return None
        
        # Calculate technical indicators using price history
        if len(self.price_history) >= 26:  # Minimum for MACD
            prices = self.price_history
            
            # Calculate all indicators
            macd_line, signal_line, macd_hist = TechnicalIndicators.calculate_macd(prices)
            rsi = TechnicalIndicators.calculate_rsi(prices)
            cci = TechnicalIndicators.calculate_cci(prices)
            adx = TechnicalIndicators.calculate_adx(prices)
            
            # Additional calculations
            volatility = self._calculate_volatility()
            momentum = self._calculate_momentum()
            
            # Market regime
            regime = self._detect_market_regime()
            regime_encoding = {"volatile": 0, "trending": 1, "ranging": 2, "mixed": 3}
            regime_value = regime_encoding.get(regime, 3)
            
            # Win rate
            win_rate = self.winning_trades / max(1, self.winning_trades + self.losing_trades)
            
            # Fill technical indicators array
            technical_features = np.array([
                macd_hist / 10.0,  # Normalized MACD histogram
                (rsi - 50) / 50.0,  # Normalized RSI (-1 to 1)
                cci / 200.0,  # Normalized CCI
                adx / 50.0,  # Normalized ADX
                volatility * 100,  # Volatility percentage
                momentum * 100,  # Momentum percentage
                regime_value / 3.0,  # Normalized regime
                win_rate,  # Current win rate
                len(self.positions) / float(self.max_positions),  # Position utilization
                self.consecutive_losses / 10.0,  # Normalized consecutive losses
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Padding to 20
            ], dtype=np.float32)
            
            base_obs['technical_indicators'] = technical_features[:20]
        
        # Add symbol encoding for symbol-specific strategies
        if hasattr(self, 'current_symbol'):
            base_obs['symbol_encoding'] = self._get_symbol_encoding(self.current_symbol)
        
        return base_obs

    def _calculate_trade_reward(self, pnl, pnl_pct, base_reward=1.0):
        """
        Improved reward structure with risk penalties and variance reduction
        """
        if pnl > 0:
            # Reward for wins with diminishing returns
            reward = 10 * np.tanh(pnl_pct * 20)  # Smooth scaling with tanh
            # Small bonus for larger wins
            if pnl_pct > 0.03:  # 3%+ wins
                reward += 1.0
            # Risk-adjusted reward: penalize excessive risk even on wins
            if pnl_pct > 0.10:  # 10%+ win suggests excessive risk
                reward *= 0.8  # Reduce reward for risky wins
        else:
            # Penalty for losses with diminishing impact
            reward = 5 * np.tanh(pnl_pct * 20)  # Smooth scaling with tanh
            # Extra penalty for large losses
            if pnl_pct < -0.03:  # 3%+ losses
                reward -= 1.0
            # Severe penalty for catastrophic losses
            if pnl_pct < -0.05:  # 5%+ losses
                reward -= 2.0  # Additional penalty for poor risk management
        
        # Risk penalties based on portfolio exposure
        if hasattr(self, 'capital') and hasattr(self, 'positions'):
            total_exposure = sum(pos.get('entry_cost', 0) for pos in self.positions)
            exposure_ratio = total_exposure / self.capital if self.capital > 0 else 0
            
            # Penalty for over-exposure
            if exposure_ratio > 0.7:  # Using more than 70% of capital
                reward -= 0.5 * (exposure_ratio - 0.7) * 10  # Progressive penalty
        
        # Consistency bonus: reward consistent small wins over volatile results
        if hasattr(self, 'recent_pnls'):
            if len(self.recent_pnls) > 5:
                pnl_std = np.std(self.recent_pnls[-10:])
                if pnl_std < 0.02 and pnl > 0:  # Low volatility with profit
                    reward += 0.5
        
        # Clip final reward to reasonable range
        reward = np.clip(reward * base_reward, -15.0, 10.0)
        
        return reward

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

    def _calculate_dynamic_position_size(self, confidence, volatility, win_rate):
        """Implement dynamic position sizing based on multiple factors"""
        base_size = 0.15  # 15% base position
        
        # Adjust for confidence (0.5x to 1.5x)
        size = base_size * (0.5 + confidence)
        
        # Adjust for volatility (inverse relationship)
        volatility_factor = 1.0 - min(0.5, volatility * 10)
        size *= volatility_factor
        
        # Bonus for high win rate
        if win_rate > 0.6:
            size *= 1.2
        elif win_rate < 0.4 and self.winning_trades + self.losing_trades > 10:
            size *= 0.8
        
        return min(0.3, max(0.05, size))  # 5-30% of capital


    def _should_exit_position(self, position, current_price, market_regime):
        """Smart exit logic based on market regime and position metrics"""
        pnl = self._calculate_position_pnl(position, current_price)
        pnl_pct = pnl / (position['entry_price'] * position['quantity'] * 100)
        holding_time = self.current_step - position['entry_step']
        
        # Dynamic exit thresholds based on market regime
        if market_regime == "volatile":
            take_profit = 0.025  # 2.5% in volatile markets (reduced from 3%)
            stop_loss = -0.015   # 1.5% stop loss (tighter from 2%)
            max_holding = 20    # Shorter holding period
        elif market_regime == "trending":
            momentum = self._calculate_momentum()
            if (position['option_type'] == 'call' and momentum > 0) or \
               (position['option_type'] == 'put' and momentum < 0):
                take_profit = 0.05  # 5% for trend-following positions (was 8%)
                stop_loss = -0.025   # 2.5% stop loss (was 3%)
                max_holding = 40    # Longer holding period
            else:
                take_profit = 0.02  # 2% quick exit for counter-trend
                stop_loss = -0.012  # 1.2% tight stop (was 1.5%)
                max_holding = 10
        else:  # ranging
            take_profit = 0.03  # 3% in ranging markets (was 4%)
            stop_loss = -0.018   # 1.8% stop loss (was 2%)
            max_holding = 25
        
        # Exit conditions
        if pnl_pct >= take_profit:
            return True, "take_profit"
        elif pnl_pct <= stop_loss:
            return True, "stop_loss"
        elif holding_time > max_holding:
            return True, "time_exit"
        
        return False, None



class LiveTradingEnvironment(HistoricalOptionsEnvironment):
    """Environment for live trading with real-time data from Alpaca"""
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = 'https://paper-api.alpaca.markets',
        symbols: List[str] = None,
        initial_capital: float = 100000,
        max_positions: int = 5,
        commission: float = 0.65,
        episode_length: int = 390,  # Trading minutes in a day
        position_size_pct: float = 0.05,  # 5% of capital per trade
        stop_loss_pct: float = 0.10,  # 10% stop loss
        take_profit_pct: float = 0.20,  # 20% take profit
        max_daily_loss_pct: float = 0.02,  # 2% daily loss limit
        live_mode: bool = True
    ):
        # Set live_mode before calling super().__init__() as it's needed in reset()
        self.live_mode = live_mode
        
        # Set symbols first
        self.symbols = symbols or ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA']
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        
        # For live trading, we don't use historical data
        # Initialize parent with minimal setup
        self.historical_data = {}
        self.data_loader = None
        self.initial_capital = initial_capital
        self.commission = commission
        self.episode_length = episode_length
        
        # Set up observation and action spaces manually
        import gymnasium as gym
        from gymnasium import spaces
        
        lookback_window = 20
        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Dict({
            'price_history': spaces.Box(low=0, high=np.inf, shape=(lookback_window, 5), dtype=np.float32),
            'technical_indicators': spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32),
            'options_chain': spaces.Box(low=0, high=np.inf, shape=(20, 15), dtype=np.float32),
            'portfolio_state': spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
            'greeks_summary': spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
            'symbol_encoding': spaces.Box(low=0, high=1, shape=(len(self.symbols),), dtype=np.float32)
        })
        
        # Initialize base attributes needed by parent methods
        self.lookback_window = lookback_window
        self.training_data = pd.DataFrame()  # Empty dataframe
        self.current_step = 0
        self.done = False
        
        # Initialize trading statistics
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_trades = 0
        self.consecutive_losses = 0
        self.total_pnl = 0
        
        # Initialize trading state
        self.capital = initial_capital
        self.positions = []
        self.closed_positions = []
        self.current_symbol = self.symbols[0] if self.symbols else 'SPY'
        
        # Live trading specific attributes
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.daily_starting_capital = initial_capital
        
        
        # Initialize Alpaca API
        try:
            self.api = tradeapi.REST(api_key, api_secret, base_url)
            # Test the connection
            account = self.api.get_account()
            logger.info(f"✅ Connected to Alpaca - Account status: {account.status}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to connect to Alpaca: {e}")
            logger.warning("Running in simulation mode - no real trades will be executed")
            self.live_mode = False  # Fallback to simulation
            
        self.data_collector = AlpacaOptionsDataCollector(
            api_key=api_key,
            api_secret=api_secret,
            base_url=base_url
        )
        
        # Track real positions
        self.real_positions = []
        self.daily_pnl = 0
        self.trade_log = []
        
        # Real-time data cache
        self.realtime_cache = {}
        self.last_update_time = {}
        
        logger.info(f"🔴 LIVE TRADING MODE {'ACTIVE' if live_mode else 'SIMULATED'}")
        logger.info(f"   Capital: ${initial_capital:,.2f}")
        logger.info(f"   Max positions: {max_positions}")
        logger.info(f"   Position size: {position_size_pct*100:.1f}% of capital")
        logger.info(f"   Stop loss: {stop_loss_pct*100:.1f}%")
        logger.info(f"   Daily loss limit: {max_daily_loss_pct*100:.1f}%")
    
    def reset(self):
        """Reset for a new trading day"""
        # For live trading, we need to create a proper observation
        if self.live_mode:
            # Initialize basic state
            self.current_step = 0
            self.capital = self.initial_capital
            self.positions = []
            self.closed_positions = []
            self.winning_trades = 0
            self.losing_trades = 0
            self.total_trades = 0
            self.consecutive_losses = 0
            self.total_pnl = 0
            self.current_symbol = self.symbols[0] if self.symbols else 'SPY'
            
            # Update account info
            if hasattr(self, 'api') and self.live_mode:
                try:
                    account = self.api.get_account()
                    self.capital = float(account.buying_power)
                    self.daily_starting_capital = self.capital
                    self.daily_pnl = 0
                    logger.info(f"Account updated - Buying power: ${self.capital:,.2f}")
                except Exception as e:
                    logger.warning(f"Failed to get account info: {e}")
                    logger.info("Using default capital for simulation")
            
            # Create a valid observation for live trading
            obs = self._create_live_observation()
            return obs
        else:
            # Use parent's reset for historical data
            return super().reset()
    
    def _create_live_observation(self):
        """Create a valid observation for live trading"""
        # Initialize with default values - only numeric data for tensor conversion
        observation = {
            'price_history': np.zeros((20, 5), dtype=np.float32),  # Match parent class shape
            'technical_indicators': np.zeros(20, dtype=np.float32),  # Match parent class shape
            'options_chain': np.zeros((20, 15), dtype=np.float32),  # Match parent class shape
            'portfolio_state': np.array([
                self.capital / self.initial_capital,  # Capital ratio
                len(self.positions) / self.max_positions,  # Position utilization
                0.0,  # Total P&L
                0.0,  # Win rate
                0.0,  # Current exposure
            ], dtype=np.float32),
            'greeks_summary': np.zeros(5, dtype=np.float32),  # Delta, gamma, theta, vega, rho
            'symbol_encoding': self._get_symbol_encoding(self.current_symbol),  # One-hot encoding for symbol-specific learning
        }
        
        # Try to get live data for the first symbol
        if self.symbols:
            live_data = self._get_live_market_data(self.symbols[0])
            if live_data:
                # Update price history with current price (OHLCV format)
                observation['price_history'][-1] = [
                    live_data['price'],  # Open
                    live_data['price'] * 1.001,  # High (simulated)
                    live_data['price'] * 0.999,  # Low (simulated)
                    live_data['price'],  # Close
                    live_data.get('volume', 1000000)  # Volume
                ]
                
                # Update options chain if available
                if live_data['options'] and len(live_data['options']) > 0:
                    for i, opt in enumerate(live_data['options'][:20]):
                        # Fill all 15 features to match parent class shape
                        observation['options_chain'][i] = [
                            opt.get('strike', 0),
                            opt.get('bid', 0),
                            opt.get('ask', 0),
                            opt.get('volume', 0),
                            opt.get('open_interest', 0),
                            opt.get('implied_volatility', 0.3),
                            1 if opt.get('type') == 'call' else 0,
                            opt.get('days_to_expiry', 30),
                            opt.get('delta', 0.5),
                            opt.get('gamma', 0.01),
                            opt.get('theta', -0.05),
                            opt.get('vega', 0.1),
                            opt.get('rho', 0.01),
                            0,  # Additional padding
                            0   # Additional padding
                        ]
        
        return observation
    
    def _get_live_market_data(self, symbol: str):
        """Fetch real-time market data"""
        try:
            # Initialize with simulated values
            current_price = 100.0 + np.random.randn() * 2  # Random price around 100
            volume = 1000000
            
            # Try to get real data if available
            if hasattr(self, 'api') and self.live_mode:
                try:
                    bars = self.api.get_latest_bar(symbol)
                    current_price = bars.c  # Close price
                    volume = bars.v
                except Exception as api_error:
                    logger.debug(f"Using simulated data for {symbol}: {api_error}")
            
            # For now, simulate options data to avoid API issues
            # TODO: Fix the async options chain fetching
            simulated_options = []
            
            # Generate some simulated options around the current price
            for strike_offset in [-5, -2.5, 0, 2.5, 5]:
                strike = current_price + strike_offset
                for option_type in ['call', 'put']:
                    # Calculate option premium (ensure non-zero for ATM options)
                    intrinsic_value = max(0, current_price - strike) if option_type == 'call' else max(0, strike - current_price)
                    time_value = 2.0  # Base time value
                    premium = intrinsic_value + time_value + abs(strike_offset) * 0.1
                    
                    simulated_options.append({
                        'symbol': f"{symbol}_SIMULATED",
                        'strike': strike,
                        'type': option_type,
                        'bid': premium * 0.95,  # Bid is slightly below mid
                        'ask': premium * 1.05,  # Ask is slightly above mid
                        'volume': 1000,
                        'open_interest': 5000,
                        'implied_volatility': 0.3,
                        'underlying_price': current_price,
                        'days_to_expiry': 30
                    })
            
            return {
                'price': current_price,
                'volume': volume,
                'options': simulated_options,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error fetching live data for {symbol}: {e}")
            # Return simulated data as fallback
            simulated_price = 100.0  # Default price
            return {
                'price': simulated_price,
                'volume': 1000000,
                'options': [],
                'timestamp': datetime.now()
            }
    
    def _execute_real_trade(self, action_name: str, symbol: str, quantity: int, contract_symbol: str):
        """Execute a real trade through Alpaca"""
        if not self.live_mode:
            logger.info(f"SIMULATED: Would execute {action_name} for {quantity} {contract_symbol}")
            return True
        
        try:
            # Check daily loss limit
            if self.daily_pnl < -self.daily_starting_capital * self.max_daily_loss_pct:
                logger.warning("Daily loss limit reached - no new trades allowed")
                return False
            
            # Submit order
            if action_name in ['buy_call', 'buy_put']:
                order = self.api.submit_order(
                    symbol=contract_symbol,
                    qty=quantity,
                    side='buy',
                    type='limit',
                    time_in_force='day',
                    limit_price=None  # Use market order for now
                )
                
                logger.info(f"✅ Order submitted: {action_name} {quantity} {contract_symbol}")
                logger.info(f"   Order ID: {order.id}")
                
                # Track position
                self.real_positions.append({
                    'order_id': order.id,
                    'symbol': symbol,
                    'contract_symbol': contract_symbol,
                    'quantity': quantity,
                    'entry_price': float(order.limit_price or order.filled_avg_price or 0),
                    'entry_time': datetime.now(),
                    'action': action_name
                })
                
                return True
            
            elif action_name == 'close_position':
                # Find matching position
                position = next((p for p in self.real_positions if p['contract_symbol'] == contract_symbol), None)
                if position:
                    order = self.api.submit_order(
                        symbol=contract_symbol,
                        qty=position['quantity'],
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
                    
                    logger.info(f"✅ Close order submitted for {contract_symbol}")
                    self.real_positions.remove(position)
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to execute trade: {e}")
            return False
        
        return False
    
    def step(self, action: int):
        """Execute action with live trading"""
        # For live mode, we handle the step ourselves
        if self.live_mode:
            self.current_step += 1
            
            # Initialize return values
            reward = 0.0
            done = self.current_step >= self.episode_length
            info = {}
            
            # Create observation before processing action
            obs = self._create_live_observation()
        else:
            # Get base observation and reward from parent
            obs, reward, done, info = super().step(action)
        
        # If we're in live mode and action involves trading
        actions = ['hold', 'buy_call', 'buy_put', 'sell_call', 'sell_put', 
                  'bull_call_spread', 'bear_put_spread', 'iron_condor', 
                  'straddle', 'strangle', 'close_all_positions']
        action_name = actions[action] if action < len(actions) else 'hold'
        
        # Handle live trades
        if self.live_mode and action_name in ['buy_call', 'buy_put', 'close_all_positions']:
            symbol = self.current_symbol
            
            # Get live data
            live_data = self._get_live_market_data(symbol)
            if live_data:
                if action_name == 'close_all_positions':
                    # Close all real positions
                    for position in self.real_positions[:]:
                        self._execute_real_trade('close_position', position['symbol'], 
                                               position['quantity'], position['contract_symbol'])
                else:
                    # Calculate position size
                    position_value = self.capital * self.position_size_pct
                    # Find suitable option contract
                    if live_data['options']:
                        # Pick ATM option
                        option = self._select_best_option(live_data['options'], action_name)
                        if option and option.get('ask', 0) > 0:
                            quantity = int(position_value / (option['ask'] * 100))
                            if quantity > 0:
                                self._execute_real_trade(action_name, symbol, quantity, option['symbol'])
        
        # Update P&L tracking
        if self.live_mode:
            self._update_real_pnl()
        
        return obs, reward, done, info
    
    def _select_best_option(self, options: List[Dict], action_name: str):
        """Select the best option contract based on criteria"""
        option_type = 'call' if 'call' in action_name else 'put'
        
        # Filter by type and liquidity
        filtered = [
            opt for opt in options 
            if opt['type'] == option_type and opt['volume'] > 100
        ]
        
        if not filtered:
            return None
        
        # Sort by distance to ATM
        filtered.sort(key=lambda x: abs(x['strike'] - x['underlying_price']))
        
        # Return the most liquid near-ATM option
        return filtered[0]
    
    def _update_real_pnl(self):
        """Update real P&L from actual positions"""
        if not self.real_positions:
            return
        
        try:
            # Get current positions from Alpaca
            positions = self.api.list_positions()
            
            total_pnl = 0
            for pos in positions:
                # Match with our tracked positions
                tracked = next((p for p in self.real_positions 
                              if p['contract_symbol'] == pos.symbol), None)
                if tracked:
                    current_value = float(pos.market_value)
                    entry_value = tracked['entry_price'] * tracked['quantity'] * 100
                    pnl = current_value - entry_value
                    total_pnl += pnl
                    
                    # Check stop loss / take profit
                    pnl_pct = pnl / entry_value
                    if pnl_pct <= -self.stop_loss_pct or pnl_pct >= self.take_profit_pct:
                        self._execute_real_trade('close_position', tracked['symbol'],
                                               tracked['quantity'], tracked['contract_symbol'])
            
            self.daily_pnl = total_pnl
            
        except Exception as e:
            logger.error(f"Error updating P&L: {e}")
    
    def _calculate_position_pnl(self, position, current_price):
        """Calculate P&L for a position"""
        # Simple P&L calculation for options
        entry_value = position['entry_price'] * position['quantity'] * 100
        current_value = current_price * position['quantity'] * 100
        return current_value - entry_value
    
    def _get_observation(self):
        """Get current observation for live trading"""
        return self._create_live_observation()
    
    def _get_symbol_encoding(self, symbol):
        """Create one-hot encoding for symbol to enable symbol-specific strategies"""
        # Create a mapping of symbols to indices if not exists
        if not hasattr(self, 'symbol_to_idx'):
            self.symbol_to_idx = {s: i for i, s in enumerate(self.symbols)}
        
        # Create one-hot encoding
        encoding = np.zeros(len(self.symbols), dtype=np.float32)
        if symbol in self.symbol_to_idx:
            encoding[self.symbol_to_idx[symbol]] = 1.0
        
        return encoding


class FastProfitableEnvironment(HistoricalOptionsEnvironment, SymbolEncodingMixin):
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
        self.max_loss_per_trade = 0.04  # 4% max loss - more room for options volatility
        self.max_profit_per_trade = 0.015  # 1.5% take profit - quick wins
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_trades = 0
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

        # Trade quality metrics
        self.trade_metrics = {
            'avg_win_size': deque(maxlen=100),
            'avg_loss_size': deque(maxlen=100),
            'profit_factors': deque(maxlen=100),
            'max_drawdowns': deque(maxlen=100),
            'sharpe_ratios': deque(maxlen=100),
            'win_rates': deque(maxlen=100)
        }
        
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
                # Larger position sizes for better profit potential
                total_trades = getattr(self, 'total_trades', self.winning_trades + self.losing_trades)
                if total_trades < 50:
                    size_pct = 0.40  # 40% during early exploration
                elif self.winning_trades / max(1, total_trades) < 0.3:
                    size_pct = 0.35  # 35% when struggling  
                else:
                    size_pct = 0.30  # 30% normally
                contracts_to_buy = min(10, max(2, int(self.capital * size_pct / cost_per_contract)))
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
        
        # Action diversity tracking
        self.recent_actions = deque(maxlen=20)
        self.action_counts = {i: 0 for i in range(11)}
        
        # Risk tracking
        self.recent_pnls = deque(maxlen=20)
        
        super().__init__(*args, **kwargs)


    def _calculate_trade_reward(self, pnl, pnl_pct, base_reward=1.0):
        """
        Improved reward structure with risk penalties and variance reduction
        """
        if pnl > 0:
            # Reward for wins with diminishing returns
            reward = 10 * np.tanh(pnl_pct * 20)  # Smooth scaling with tanh
            # Small bonus for larger wins
            if pnl_pct > 0.03:  # 3%+ wins
                reward += 1.0
            # Risk-adjusted reward: penalize excessive risk even on wins
            if pnl_pct > 0.10:  # 10%+ win suggests excessive risk
                reward *= 0.8  # Reduce reward for risky wins
        else:
            # Penalty for losses with diminishing impact
            reward = 5 * np.tanh(pnl_pct * 20)  # Smooth scaling with tanh
            # Extra penalty for large losses
            if pnl_pct < -0.03:  # 3%+ losses
                reward -= 1.0
            # Severe penalty for catastrophic losses
            if pnl_pct < -0.05:  # 5%+ losses
                reward -= 2.0  # Additional penalty for poor risk management
        
        # Risk penalties based on portfolio exposure
        if hasattr(self, 'capital') and hasattr(self, 'positions'):
            total_exposure = sum(pos.get('entry_cost', 0) for pos in self.positions)
            exposure_ratio = total_exposure / self.capital if self.capital > 0 else 0
            
            # Penalty for over-exposure
            if exposure_ratio > 0.7:  # Using more than 70% of capital
                reward -= 0.5 * (exposure_ratio - 0.7) * 10  # Progressive penalty
        
        # Consistency bonus: reward consistent small wins over volatile results
        if hasattr(self, 'recent_pnls'):
            if len(self.recent_pnls) > 5:
                pnl_std = np.std(self.recent_pnls[-10:])
                if pnl_std < 0.02 and pnl > 0:  # Low volatility with profit
                    reward += 0.5
        
        # Clip final reward to reasonable range
        reward = np.clip(reward * base_reward, -15.0, 10.0)
        
        return reward

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

        
        self.consecutive_losses = 0
        self.force_close_losses = True
        self.max_loss_per_trade = 0.04  # 4% max loss - more room for options volatility
        self.max_profit_per_trade = 0.015  # 1.5% take profit - quick wins
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
            
        # Track action for diversity
        self.recent_actions.append(action)
        self.action_counts[action] += 1
            
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
                    
                    # Position sizing - SMALLER POSITIONS FOR MORE OPPORTUNITIES
                    confidence = option.get('score', 0.5)
                    max_risk = self.capital * (0.05 + 0.05 * confidence)  # 5-10% per trade
                    cost_per_contract = option['mid_price'] * 100 + self.commission
                    
                    ideal_contracts = int(max_risk / cost_per_contract)
                    contracts_to_buy = max(1, min(ideal_contracts, 3))  # Max 3 contracts
                    
                    total_cost = contracts_to_buy * cost_per_contract
                    
                    if total_cost <= self.capital * 0.15:  # Max 15% per position
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
        
        # Reward for winning trades - ENHANCED REWARDS FOR ANY WIN
        wins_added = self.winning_trades - wins_before
        if wins_added > 0:
            # Big reward for ANY winning trade to encourage wins
            reward += wins_added * 50.0
            
            if step_pnl > 0:
                profit_pct = step_pnl / portfolio_value_before
                # Reward any profitable outcome
                if profit_pct >= 0.02:
                    reward += 40.0
                elif profit_pct >= 0.01:
                    reward += 30.0
                elif profit_pct >= 0.005:
                    reward += 20.0
                elif profit_pct > 0:
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
        
        # Win rate bonus rewards
        if self.winning_trades + self.losing_trades >= 5:
            current_win_rate = self.winning_trades / (self.winning_trades + self.losing_trades)
            if current_win_rate >= 0.6:
                reward += 20.0  # Bonus for 60%+ win rate
        
        # Strategy consistency rewards
        if hasattr(self, 'recent_actions') and len(self.recent_actions) >= 10:
            # Check for consistent strategy patterns
            recent_10 = list(self.recent_actions)[-10:]
            
            # Reward for consistent buy/sell patterns (not just holding)
            non_hold_actions = [a for a in recent_10 if a != 0]
            if len(non_hold_actions) >= 5:  # At least 50% active trading
                # Check for pattern consistency
                if len(set(non_hold_actions)) <= 3:  # Using 3 or fewer different strategies
                    reward += 2.0  # Consistency bonus
                
                # Check for market-appropriate actions
                if hasattr(self, '_market_regime_history'):
                    last_regime = self._market_regime_history[-1] if self._market_regime_history else 'mixed'
                    # Reward trend-following in trending markets
                    if last_regime == 'trending' and (1 in recent_10 or 2 in recent_10):  # Buy calls/puts
                        reward += 1.5
                    # Reward spreads in volatile markets
                    elif last_regime == 'volatile' and any(a in [5, 6, 7] for a in recent_10):
                        reward += 1.5
                    # Reward neutral strategies in ranging markets
                    elif last_regime == 'ranging' and any(a in [7, 8, 9] for a in recent_10):
                        reward += 1.5
            elif current_win_rate >= 0.5:
                reward += 10.0  # Bonus for 50%+ win rate
            elif current_win_rate >= 0.4:
                reward += 5.0   # Bonus for 40%+ win rate
            elif current_win_rate < 0.2:
                reward -= 10.0  # Penalty for very low win rate
        
        # Next step
        self.current_step += 1
        
        # Check done
        if self.current_step >= len(self.training_data) - 1:
            self.done = True
            # Episode completion bonus based on win rate
            if self.winning_trades + self.losing_trades > 0:
                final_win_rate = self.winning_trades / (self.winning_trades + self.losing_trades)
                if final_win_rate >= 0.5:
                    reward += 100.0 * final_win_rate  # Big bonus for high win rate episodes
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
        
        # Score components - PRIORITIZE TIGHT SPREADS FOR BETTER ENTRY/EXIT
        spread_scores = 1.0 / (1.0 + spread_pcts[filtered_indices] * 20)  # More penalty for wide spreads
        liquidity_scores = np.minimum(volumes[filtered_indices] / 1000, 1.0)
        
        # Price scores - favor moderately priced options
        filtered_mid_prices = mid_prices[filtered_indices]
        price_scores = np.where(
            (filtered_mid_prices >= 2.0) & (filtered_mid_prices <= 8.0),
            1.0,
            np.where(
                filtered_mid_prices < 2.0,
                filtered_mid_prices / 2.0,
                8.0 / filtered_mid_prices
            )
        )
        
        # Moneyness score - prefer slightly OTM for better risk/reward
        filtered_moneyness = moneyness[filtered_indices]
        moneyness_scores = np.where(
            (filtered_moneyness >= 0.95) & (filtered_moneyness <= 1.05),
            1.0,
            1.0 - np.abs(filtered_moneyness - 1.0) * 2
        )
        
        # Combined scores - heavy weight on spread for profitability
        scores = spread_scores * 0.7 + liquidity_scores * 0.15 + price_scores * 0.1 + moneyness_scores * 0.05
        
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
            
            # Calculate P&L (excluding commission from PnL calculation for win/loss determination)
            entry_cost = pos['entry_price'] * pos['quantity'] * 100
            current_value = current_price * pos['quantity'] * 100
            # Raw PnL without commission for win/loss determination
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
            
            # Exit conditions - OPTIMIZED FOR HIGHER WIN RATE
            should_exit = False
            exit_reason = ""
            
            # Quick win exit - take profits fast before theta decay
            if pnl > 0 and position_age >= 1:
                # Any profitable position after 1 step is a win
                should_exit = True
                exit_reason = "quick_win"
                self.winning_trades += 1
                self.consecutive_losses = 0
            # Trailing stop for larger profits
            elif pos['peak_pnl_pct'] >= 0.01 and pnl_pct <= pos['peak_pnl_pct'] * 0.7:
                should_exit = True
                exit_reason = "trailing_stop"
                if pnl > 0:
                    self.winning_trades += 1
                    self.consecutive_losses = 0
                else:
                    self.losing_trades += 1
                    self.consecutive_losses += 1
            # Take profit - lowered threshold
            elif pnl_pct >= adjusted_take_profit:
                should_exit = True
                exit_reason = "take_profit"
                self.winning_trades += 1
                self.consecutive_losses = 0
            # Stop loss - only if significant loss
            elif pnl_pct <= adjusted_stop_loss:
                should_exit = True
                exit_reason = "stop_loss"
                self.losing_trades += 1
                self.consecutive_losses += 1
            # Time exit - shorter holding period to avoid theta decay
            elif position_age > 10:
                should_exit = True
                exit_reason = "time_exit"
                if pnl > 0:
                    self.winning_trades += 1
                    self.consecutive_losses = 0
                else:
                    self.losing_trades += 1
                    self.consecutive_losses += 1
            
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


def train_distributed(rank, world_size, num_episodes=10000, save_interval=100, use_real_data=True, resume=False, checkpoint_path=None, live_mode=False, live_config=None, winners_only=False, winners_interval=50):
    """Distributed training function for multi-GPU support"""
    
    # Configure NCCL timeout to prevent hanging (set to 30 minutes)
    if world_size > 1:
        os.environ['NCCL_TIMEOUT'] = '3600'  # 60 minutes - increase for large models
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'  # Enable async error handling
        os.environ['NCCL_DEBUG'] = 'WARN'  # Set to INFO for more debugging if needed
        os.environ['NCCL_P2P_DISABLE'] = '1'  # Disable P2P to avoid some GPU communication issues
        os.environ['NCCL_IB_DISABLE'] = '1'  # Disable InfiniBand if not using it
    
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
    
    
    # Create environment based on mode
    logger.info(f"Environment mode - Live: {live_mode}, Config: {live_config}")
    
    # For live mode, we'll create a regular training environment but add live execution
    if live_mode:
        logger.info("🔴 LIVE TRADING MODE ENABLED")
        logger.info("📚 Using historical data for training with live execution capability")
        
        # Get API credentials
        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_SECRET_KEY')
        
        # Store live trading config globally for environments to use
        import builtins
        builtins.live_trading_config = {
            'enabled': True,
            'api_key': api_key,
            'api_secret': api_secret,
            'base_url': 'https://paper-api.alpaca.markets' if live_config.get('paper_trading', True) else 'https://api.alpaca.markets',
            'position_size_pct': live_config.get('position_size', 0.05),
            'max_daily_loss_pct': live_config.get('daily_loss_limit', 0.02),
            'execution_probability': 0.1  # Only execute 10% of trades live
        }
        
        if api_key and api_secret:
            logger.info("✅ Live trading credentials found")
            try:
                # Test the connection
                test_api = tradeapi.REST(api_key, api_secret, builtins.live_trading_config['base_url'])
                account = test_api.get_account()
                logger.info(f"✅ Connected to Alpaca - Account status: {account.status}")
            except Exception as e:
                logger.warning(f"⚠️ Could not connect to Alpaca: {e}")
                logger.info("Will continue with simulated trading only")
        else:
            logger.warning("⚠️ No API credentials - will train with simulated trades only")
        
        # Use regular training environment with historical data
        # The environment will check for live_trading_config and execute trades if configured
        n_envs = 1  # Single environment for live mode
        
        # Create a BalancedEnvironment with historical data for training
        if historical_data and len(historical_data) > 0:
            # Use intersection of requested symbols and available symbols
            requested_symbols = live_config.get('symbols', [])
            available_symbols = list(historical_data.keys())
            
            if requested_symbols:
                # Filter to only symbols that we have data for
                symbols_to_use = [s for s in requested_symbols if s in available_symbols]
                if not symbols_to_use:
                    logger.warning(f"None of the requested symbols {requested_symbols} have historical data")
                    logger.info(f"Using all available symbols instead: {available_symbols}")
                    symbols_to_use = available_symbols
                else:
                    logger.info(f"Using symbols with historical data: {symbols_to_use}")
            else:
                symbols_to_use = available_symbols
            
            logger.info(f"Creating training environment with historical data for {len(symbols_to_use)} symbols")
            base_env = BalancedEnvironment(
                historical_data=historical_data,
                data_loader=data_loader,
                symbols=symbols_to_use,
                initial_capital=live_config.get('capital', 100000),
                max_positions=live_config.get('max_positions', 5),
                commission=0.65,
                episode_length=100  # Reasonable episode length for live mode
            )
        else:
            logger.error("No historical data available for training environment")
            return
        
        # Wrap in a simple vectorized wrapper for compatibility
        class SingleEnvWrapper:
            def __init__(self, env):
                self.envs = [env]
                self.num_envs = 1
                self.observation_space = env.observation_space
                self.action_space = env.action_space
            
            def reset(self):
                return [self.envs[0].reset()]
            
            def step(self, actions):
                obs, reward, done, info = self.envs[0].step(actions[0])
                return [obs], [reward], [done], [info]
            
            def close(self):
                pass
        
        env = SingleEnvWrapper(base_env)
        n_envs = 1
        
        logger.info("✅ Live trading environment created")
        logger.info(f"   Symbols: {live_config.get('symbols')}")
        logger.info(f"   Capital: ${live_config.get('capital'):,.2f}")
        logger.info(f"   {'PAPER' if live_config.get('paper_trading') else 'REAL'} trading account")
    else:
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
    # Increased learning rates for better learning with normalized rewards
    base_lr_actor_critic = 1e-3  # Higher learning rate for breaking through plateau
    base_lr_clstm = 3e-3  # Higher learning rate for CLSTM features
    
    # Start with lower learning rates for warmup
    warmup_episodes = 100
    if not resume or start_episode < warmup_episodes:
        current_lr_actor_critic = base_lr_actor_critic * 0.1  # Start at 10% of base
        current_lr_clstm = base_lr_clstm * 0.1
        if rank == 0:
            logger.info(f"Using warmup learning rates: AC={current_lr_actor_critic}, CLSTM={current_lr_clstm}")
    else:
        current_lr_actor_critic = base_lr_actor_critic
        current_lr_clstm = base_lr_clstm
    
    agent = OptionsCLSTMPPOAgent(
        observation_space=env.observation_space,
        action_space=11,
        learning_rate_actor_critic=current_lr_actor_critic,
        learning_rate_clstm=current_lr_clstm,
        gamma=0.99,  # Higher discount factor for long-term rewards
        gae_lambda=0.95,  # GAE lambda for advantage estimation
        clip_epsilon=0.2,  # Standard PPO clip range
        value_coef=0.5,  # Value loss coefficient
        entropy_coef=0.02,  # Higher entropy for more exploration
        max_grad_norm=0.5,  # Gradient clipping
        batch_size=4096,  # Larger batch size for stable updates
        n_epochs=10,  # More epochs for better convergence
        device=device  # Pass specific device
    )
    
    # Move agent to specific GPU
    agent.network = agent.network.to(device)
    agent.base_network = agent.base_network.to(device)
    
    # Initialize network weights properly if starting fresh
    if not resume:
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)  # Small initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        agent.network.apply(init_weights)
        if rank == 0:
            logger.info("Network weights initialized with Xavier uniform (small gain)")
    

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
            pretrain_metrics = agent.pretrain_clstm(pretrain_samples, epochs=5, batch_size = 4096)  # Reduced epochs, larger batch
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
    base_exploration_decay = 0.9995  # Slower decay to maintain exploration
    min_exploration = 0.15  # Higher minimum for continuous exploration
    max_exploration = 0.95  # Much higher maximum for aggressive recovery
    
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
    
    # Reward normalization tracking
    reward_stats = {
        'count': 0,
        'mean': 0.0,
        'M2': 0.0,  # For running variance calculation
        'std': 1.0
    }

    
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
            safe_save_checkpoint(checkpoint, emergency_file, logger)
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
                    safe_save_checkpoint(checkpoint, checkpoint_file, logger)
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
        
        # Boost exploration if performance is stagnating
        if exploration_boost_on_stagnation and episodes_without_improvement > stagnation_threshold:
            # More aggressive boost based on stagnation duration
            if episodes_without_improvement > stagnation_threshold * 3:  # Very long stagnation
                boost_factor = 3.0
            elif episodes_without_improvement > stagnation_threshold * 2:  # Long stagnation
                boost_factor = 2.5
            else:
                boost_factor = 2.0
            
            boosted_rate = min(max_exploration, exploration_rate * boost_factor)
            # Also ensure minimum boost of +0.2 to make a real difference
            boosted_rate = max(boosted_rate, min(max_exploration, exploration_rate + 0.2))
            
            if rank == 0 and boosted_rate != exploration_rate:
                logger.info(f"🔄 Boosting exploration from {exploration_rate:.3f} to {boosted_rate:.3f} due to {episodes_without_improvement} episodes without improvement")
            exploration_rate = boosted_rate
        
        # Learning rate warmup
        if episode < warmup_episodes and not resume:
            warmup_factor = (episode + 1) / warmup_episodes
            new_lr_ac = current_lr_actor_critic * warmup_factor
            new_lr_clstm = current_lr_clstm * warmup_factor
            
            for param_group in agent.ppo_optimizer.param_groups:
                param_group['lr'] = new_lr_ac
            for param_group in agent.clstm_optimizer.param_groups:
                param_group['lr'] = new_lr_clstm
                
            if rank == 0 and episode % 10 == 0:
                logger.info(f"Warmup LR - AC: {new_lr_ac:.6f}, CLSTM: {new_lr_clstm:.6f}")
        
        # Clear batch arrays
        batch_observations.clear()
        batch_actions.clear()
        batch_rewards.clear()
        batch_values.clear()
        batch_log_probs.clear()
        batch_dones.clear()
        
        # Pre-generate random numbers for exploration (batch for efficiency)
        max_steps = 50  # Match reduced episode length
        
        # Dynamic action weights based on recent performance
        # Calculate average recent return for decision making
        recent_avg_return = np.mean(all_returns[-10:]) if len(all_returns) > 0 else -300
        
        if episode < 100 or (episode % 50 == 0 and np.random.random() < 0.2):
            # Full random episodes periodically for exploration
            action_weights = np.ones(11) / 11  # Equal probability for all actions
            if rank == 0 and episode % 50 == 0:
                logger.info(f"🎲 Episode {episode}: Full random exploration episode")
        elif recent_avg_return < -200:  # If losing money, encourage more diverse trading
            # Strong bias towards trading actions when losing
            action_weights = np.array([0.05, 0.20, 0.20, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10])
            if rank == 0 and episode % 50 == 0:
                logger.info(f"📉 Recent avg return: ${recent_avg_return:.2f} - Using aggressive exploration")
        else:
            # Standard exploration with moderate trading bias
            action_weights = np.array([0.1, 0.15, 0.15, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.04])
        
        action_weights = action_weights / action_weights.sum()  # Pre-normalize
        random_actions_all = np.random.choice(11, size=(max_steps, n_envs), p=action_weights)
        
        step = 0
        while step < max_steps and not all(episode_dones):
            # Check for shutdown during episode
            if shutdown_requested.is_set():
                break
                
            # Algorithm 2 Implementation:
            # Step 6: Receive state st from environment (obs_list contains states)
            
            # Step 7: Process st with LSTM to obtain feature vector ft
            # This is handled inside get_batch_actions through agent.network
            
            # Steps 8-9: Compute critic value and sample action
            # get_batch_actions performs these steps internally
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
                # Enhanced exploration bonus system
                exploration_bonus = 0.0
                
                # 1. Action diversity bonus
                if actions[env_idx] != 0:  # Not hold
                    exploration_bonus = 0.05 * exploration_rate
                    
                    # Extra bonus for less common actions
                    if actions[env_idx] in [3, 4, 5, 6, 7, 8, 9]:  # Sell options, partial closes
                        exploration_bonus += 0.1 * exploration_rate
                
                # 2. Curiosity bonus for trying different strategies
                try:
                    # Access the underlying environment through the VectorizedEnvironment wrapper
                    if hasattr(env.envs[env_idx], 'recent_actions'):
                        recent_actions = env.envs[env_idx].recent_actions
                        if len(recent_actions) > 5 and actions[env_idx] not in recent_actions[-5:]:
                            exploration_bonus += 0.2  # Bonus for trying something new
                except (AttributeError, IndexError):
                    # Skip if environment doesn't support this feature
                    pass
                
                # 3. Anti-stagnation bonus
                if recent_avg_return < -200 and actions[env_idx] in [1, 2]:  # Buy actions when losing
                    exploration_bonus += 0.3
                
                adjusted_reward = rewards[env_idx] + exploration_bonus
                episode_rewards[env_idx].append(adjusted_reward)
                env_steps[env_idx] += 1
                
                # Add to buffer
                batch_observations.append(obs_list[env_idx])
                batch_actions.append(actions[env_idx])
                batch_rewards.append(adjusted_reward)
                batch_values.append(action_infos[env_idx]['value'])
                batch_log_probs.append(action_infos[env_idx]['log_prob'])
                batch_dones.append(dones[env_idx])
                
                if dones[env_idx]:
                    episode_dones[env_idx] = True
            
            # Update observations
            obs_list = next_obs_list
            step += 1
        
        # Normalize rewards before adding to buffer using running statistics
        # This is critical for stable learning
        if len(batch_rewards) > 0:
            rewards_array = np.array(batch_rewards)
            
            # Update running statistics (Welford's online algorithm)
            for reward in rewards_array:
                reward_stats['count'] += 1
                delta = reward - reward_stats['mean']
                reward_stats['mean'] += delta / reward_stats['count']
                delta2 = reward - reward_stats['mean']
                reward_stats['M2'] += delta * delta2
                
            if reward_stats['count'] > 1:
                reward_stats['std'] = np.sqrt(reward_stats['M2'] / (reward_stats['count'] - 1))
            
            # Clip extreme rewards to prevent instability
            rewards_array = np.clip(rewards_array, -10, 10)
            
            # Normalize using running statistics
            if reward_stats['count'] > 100:  # Wait for stable statistics
                normalized_rewards = (rewards_array - reward_stats['mean']) / (reward_stats['std'] + 1e-8)
                # Additional clipping after normalization
                normalized_rewards = np.clip(normalized_rewards, -5, 5)
            else:
                # Simple normalization for early episodes
                normalized_rewards = rewards_array / 10.0
        else:
            normalized_rewards = batch_rewards
        
        # Add all experiences to buffer at once
        for i in range(len(batch_observations)):
            agent.buffer.add(
                observation=batch_observations[i],
                action=batch_actions[i],
                reward=normalized_rewards[i] if len(batch_rewards) > 0 else batch_rewards[i],
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
                
                # Add bonus for breaking out of negative return pattern
                if env_return > -100:  # Better than -$300 pattern
                    bonus = 5.0
                    if env_return > 0:  # Profitable!
                        bonus = 20.0
                    elif env_return > 100:  # Very profitable!
                        bonus = 50.0
                    
                    # Add bonus to last step's reward
                    if len(batch_rewards) > 0 and env_steps[env_idx] > 0:
                        last_idx = sum(env_steps[:env_idx]) + env_steps[env_idx] - 1
                        if last_idx < len(batch_rewards):
                            batch_rewards[last_idx] += bonus
                            if rank == 0 and episode % 100 == 0:
                                logger.info(f"💰 Episode {episode} Env {env_idx}: Return ${env_return:.2f}, Bonus +{bonus}")
                
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
                    safe_save_checkpoint({
                        'model_state_dict': best_model_state,
                        'episode': episode,
                        'win_rate': best_avg_win_rate,
                        'timestamp': datetime.now().isoformat()
                    }, best_model_path, logger)

        
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
                        
                        # Model reloading disabled due to checkpoint compatibility issues
                        # PyTorch's weights_only loading can fail with complex checkpoints
                        # Instead, we use adaptive hyperparameter adjustments for recovery:
                        # 1. Increase exploration to find new strategies
                        # 2. Reduce learning rates to stabilize training
                        # 3. Trust the model to recover through continued training
                        reload_best_model = False  # Always false - reloading disabled
                        
                        if performance_drop >= 0.30:
                            logger.info(f"🔄 SEVERE DECLINE DETECTED ({performance_drop:.1%})")
                            logger.info(f"   Model reloading is disabled - using recovery strategies:")
                            
                            # Recovery Strategy 1: Reset episodes without improvement counter
                            episodes_without_improvement = 0
                            logger.info(f"   ✓ Reset improvement tracking")
                            
                            # Recovery Strategy 2: Significant exploration boost
                            old_exploration = exploration_rate
                            exploration_rate = min(max_exploration, exploration_rate + 0.15)
                            logger.info(f"   ✓ Exploration: {old_exploration:.1%} → {exploration_rate:.1%}")
                            
                            # Recovery Strategy 3: Moderate learning rate reduction (not too aggressive)
                            lr_reduction_factor = 0.5  # 50% reduction
                            
                            # Recovery Strategy 4: Increase entropy coefficient temporarily
                            if hasattr(agent, 'entropy_coef'):
                                old_entropy = agent.entropy_coef
                                agent.entropy_coef = min(0.05, agent.entropy_coef * 2)
                                logger.info(f"   ✓ Entropy coefficient: {old_entropy:.3f} → {agent.entropy_coef:.3f}")
                            
                            # Recovery Strategy 5: Adjust PPO clip range for more exploration
                            if hasattr(agent, 'clip_epsilon'):
                                old_clip = agent.clip_epsilon
                                agent.clip_epsilon = min(0.3, agent.clip_epsilon * 1.5)
                                logger.info(f"   ✓ PPO clip range: {old_clip:.2f} → {agent.clip_epsilon:.2f}")
                            
                        else:
                            # Moderate decline
                            logger.info(f"📉 Moderate decline ({performance_drop:.1%}) - mild adjustments...")
                            lr_reduction_factor = 0.7  # 30% reduction for moderate decline
                            
                            # Mild entropy boost
                            if hasattr(agent, 'entropy_coef'):
                                agent.entropy_coef = min(0.03, agent.entropy_coef * 1.2)
                        
                        # Always reduce learning rates on decline
                        for param_group in agent.ppo_optimizer.param_groups:
                            param_group['lr'] = param_group['lr'] * lr_reduction_factor
                            current_lr_actor_critic = param_group['lr']
                        for param_group in agent.clstm_optimizer.param_groups:
                            param_group['lr'] = param_group['lr'] * lr_reduction_factor
                            current_lr_clstm = param_group['lr']
                        
                        reduction_pct = (1 - lr_reduction_factor) * 100
                        logger.info(f"📉 Learning rates reduced by {reduction_pct:.0f}%")
                        logger.info(f"   Actor-Critic LR: {current_lr_actor_critic:.2e}")
                        logger.info(f"   CLSTM LR: {current_lr_clstm:.2e}")
                        
                        # Moderate exploration boost for moderate decline (already handled above for severe)
                        if performance_drop < 0.30:
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
                        safe_save_checkpoint(best_checkpoint, 'checkpoints/profitable_optimized/best_model.pt', logger)
                        logger.info(f"💾 Best model saved immediately!")
                        
                        # Also save a timestamped backup
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        backup_path = f'checkpoints/profitable_optimized/best_model_wr{int(best_win_rate*100)}_ep{episode+1}_{timestamp}.pt'
                        safe_save_checkpoint(best_checkpoint, backup_path, logger)
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
        
        # Algorithm 2 Step 13: if t mod T = 0 then update networks
        # Update agent when buffer reaches update interval
        if len(agent.buffer) >= agent.batch_size:
            if rank == 0 and episode % 50 == 0:
                logger.info(f"Buffer size: {len(agent.buffer)}, Batch size: {agent.batch_size} - Training triggered")
            if world_size > 1 and dist.is_initialized():
                # Distributed training: All GPUs train together with gradient averaging
                try:
                    # Log buffer sizes across GPUs - with timeout protection
                    if rank == 0 and episode % 50 == 0:
                        try:
                            local_buffer_size = len(agent.buffer)
                            buffer_sizes = [0] * world_size
                            # Add timeout by using a future
                            import concurrent.futures
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(dist.all_gather_object, buffer_sizes, local_buffer_size)
                                try:
                                    future.result(timeout=5.0)  # 5 second timeout
                                    total_experiences = sum(buffer_sizes)
                                    logger.info(f"Training with {total_experiences} total experiences across {world_size} GPUs: {buffer_sizes}")
                                except concurrent.futures.TimeoutError:
                                    logger.warning("all_gather_object timed out - skipping buffer size logging")
                        except Exception as e:
                            logger.warning(f"Error gathering buffer sizes: {e}")
                    
                    # All GPUs train on their local data with DDP gradient synchronization
                    # Train on winning episodes periodically for better performance
                    use_winners = (winners_only or (episode % winners_interval == 0 and episode > 100))
                    if use_winners and rank == 0:
                        logger.info("Training on winning episodes only")
                    train_metrics = agent.train(winners_only=use_winners)
                    
                    # Skip barrier synchronization to avoid hangs
                    # Each GPU can proceed independently after training
                    if False:  # Disabled to prevent NCCL timeout
                        try:
                            dist.barrier()
                        except Exception as e:
                            logger.error(f"Rank {rank}: Training barrier timeout: {e}")
                        # Continue anyway to prevent deadlock
                    
                except Exception as e:
                    logger.error(f"Rank {rank}: Distributed training error: {e}")
                    traceback.print_exc()
                    # Fallback to local training
                    use_winners = (winners_only or (episode % winners_interval == 0 and episode > 100))
                    train_metrics = agent.train(winners_only=use_winners)
            else:
                # Single GPU training
                try:
                    use_winners = (winners_only or (episode % winners_interval == 0 and episode > 100))
                    if use_winners:
                        logger.info("Training on winning episodes only")
                    train_metrics = agent.train(winners_only=use_winners)
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
            if rank == 0 and train_metrics:
                # Log every training update for better debugging
                logger.info(f"Episode {episode} Training Update:")
                logger.info(f"  PPO Loss: {train_metrics.get('policy_loss', 0):.4f}")
                logger.info(f"  Value Loss: {train_metrics.get('value_loss', 0):.4f}")
                logger.info(f"  Entropy: {train_metrics.get('entropy', 0):.4f}")
                logger.info(f"  CLSTM Loss: {train_metrics.get('clstm_loss', 'N/A')}")
                logger.info(f"  Reward Stats - Mean: {reward_stats['mean']:.4f}, Std: {reward_stats['std']:.4f}")
                
                # Check if losses are NaN or too high
                if train_metrics.get('policy_loss', 0) > 100 or np.isnan(train_metrics.get('policy_loss', 0)):
                    logger.warning("⚠️ Very high or NaN policy loss detected! Reducing learning rate.")
                    for param_group in agent.ppo_optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    for param_group in agent.clstm_optimizer.param_groups:
                        param_group['lr'] *= 0.5
        
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
                    safe_save_checkpoint(checkpoint, checkpoint_file, logger)
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
                    episodes_without_improvement = 0  # Reset counter on improvement
                    improvement = best_avg_win_rate - previous_best_avg
                    
                    logger.info(f"\n{'='*60}")
                    logger.info(f"🏆 NEW BEST AVERAGE WIN RATE! Average: {best_avg_win_rate:.2%} (was {previous_best_avg:.2%}, +{improvement:.2%})")
                    logger.info(f"   Calculated over last {save_interval} episodes")
                    
                    # Save best average model
                    best_avg_checkpoint = checkpoint.copy()
                    best_avg_checkpoint['best_avg_win_rate'] = best_avg_win_rate
                    
                    try:
                        safe_save_checkpoint(best_avg_checkpoint, 'checkpoints/profitable_optimized/best_avg_model.pt', logger)
                        logger.info(f"💾 Best average model saved!")
                        
                        # Also save a timestamped backup
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        backup_path = f'checkpoints/profitable_optimized/best_avg_model_wr{int(best_avg_win_rate*100)}_ep{episode+1}_{timestamp}.pt'
                        safe_save_checkpoint(best_avg_checkpoint, backup_path, logger)
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


def train(num_episodes=10000, save_interval=100, use_real_data=True, resume=False, checkpoint_path=None, no_distributed=False, live_mode=False, live_config=None):
    """Main training function that spawns distributed processes"""
    
    # Check if we have multiple GPUs
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        
        # Live mode only supports single GPU
        if live_mode and world_size > 1:
            logger.warning("Live trading mode only supports single GPU - forcing single GPU mode")
            world_size = 1
            no_distributed = True
        
        if world_size > 1 and not no_distributed and not args.force_single_gpu:
            logger.info(f"Starting distributed training on {world_size} GPUs")
            
            # Find a free port to avoid conflicts
            free_port = find_free_port()
            os.environ['MASTER_PORT'] = str(free_port)
            logger.info(f"Using port {free_port} for distributed training")
            
            # Spawn processes for distributed training
            mp.spawn(
                train_distributed,
                args=(world_size, num_episodes, save_interval, use_real_data, resume, checkpoint_path, live_mode, live_config, args.winners_only, args.winners_interval),
                nprocs=world_size,
                join=True
            )
        else:
            if no_distributed or args.force_single_gpu:
                logger.info("Single GPU mode (distributed training disabled)")
            else:
                logger.info("Single GPU detected, running standard training")
            # Run single GPU training
            train_distributed(0, 1, num_episodes, save_interval, use_real_data, resume, checkpoint_path, live_mode, live_config, args.winners_only, args.winners_interval)
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
    parser.add_argument('--live-mode', action='store_true', help='Enable live trading through Alpaca')
    parser.add_argument('--paper-trading', action='store_true', help='Use paper trading account (default)')
    parser.add_argument('--live-capital', type=float, default=10000, help='Capital for live trading')
    parser.add_argument('--live-symbols', nargs='+', default=['SPY', 'QQQ'], help='Symbols for live trading')
    parser.add_argument('--position-size', type=float, default=0.05, help='Position size as fraction of capital')
    parser.add_argument('--daily-loss-limit', type=float, default=0.02, help='Daily loss limit as fraction of capital')
    parser.add_argument('--winners-only', action='store_true', help='Train only on winning episodes')
    parser.add_argument('--winners-interval', type=int, default=50, help='Train on winners every N episodes')
    parser.add_argument('--use-algorithm2', action='store_true', help='Use exact Algorithm 2 PPO-LSTM implementation')

    args = parser.parse_args()
    
    # If using Algorithm 2, run the dedicated implementation
    if args.use_algorithm2:
        logger.info("Using exact Algorithm 2 PPO-LSTM implementation")
        from train_ppo_lstm import main as run_algorithm2
        run_algorithm2()
        return
    
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
        no_distributed=args.no_distributed,
        live_mode=args.live_mode,
        live_config={
            'paper_trading': args.paper_trading,
            'capital': args.live_capital,
            'symbols': args.live_symbols,
            'position_size': args.position_size,
            'daily_loss_limit': args.daily_loss_limit
        } if args.live_mode else None
    )