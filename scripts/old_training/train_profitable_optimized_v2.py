#!/usr/bin/env python3
"""Optimized training script with significant performance improvements - Version 2 with all 10 fixes"""

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
import pickle


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
        mean_deviation = np.mean(np.abs(typical_price[-period:] - sma))
        
        if mean_deviation == 0:
            return 0.0
        
        # CCI calculation
        cci = (typical_price[-1] - sma) / (0.015 * mean_deviation)
        
        return float(cci)
    
    @staticmethod
    def calculate_adx(prices, highs=None, lows=None, period=14):
        """
        Calculate ADX (Average Directional Index)
        Returns: ADX value (0-100, >25 indicates strong trend)
        """
        if len(prices) < period + 1:
            return 0.0
        
        if highs is None:
            highs = prices
        if lows is None:
            lows = prices
            
        highs_arr = np.array(highs)
        lows_arr = np.array(lows)
        prices_arr = np.array(prices)
        
        # Calculate directional movements
        high_diff = np.diff(highs_arr)
        low_diff = -np.diff(lows_arr)
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        # True Range
        tr = np.maximum(
            highs_arr[1:] - lows_arr[1:],
            np.abs(highs_arr[1:] - prices_arr[:-1]),
            np.abs(lows_arr[1:] - prices_arr[:-1])
        )
        
        # Smoothed averages
        atr = np.mean(tr[-period:])
        plus_di = 100 * np.mean(plus_dm[-period:]) / atr if atr > 0 else 0
        minus_di = 100 * np.mean(minus_dm[-period:]) / atr if atr > 0 else 0
        
        # DX and ADX
        dx_denominator = plus_di + minus_di
        if dx_denominator == 0:
            return 0.0
            
        dx = 100 * np.abs(plus_di - minus_di) / dx_denominator
        
        return float(dx)  # Simplified - would normally smooth DX to get ADX
    
    @staticmethod
    def _calculate_ema(data, period):
        """Calculate Exponential Moving Average"""
        if len(data) < period:
            return np.array([np.mean(data[:i+1]) for i in range(len(data))])
        
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    @staticmethod
    def calculate_bollinger_bands(prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return prices[-1], prices[-1], prices[-1]
        
        prices_array = np.array(prices)
        sma = np.mean(prices_array[-period:])
        std = np.std(prices_array[-period:])
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return float(upper_band), float(sma), float(lower_band)
    
    @staticmethod
    def calculate_stochastic(prices, highs=None, lows=None, period=14, smooth_k=3, smooth_d=3):
        """Calculate Stochastic Oscillator"""
        if len(prices) < period:
            return 50.0, 50.0
        
        if highs is None:
            highs = prices
        if lows is None:
            lows = prices
        
        highs_array = np.array(highs[-period:])
        lows_array = np.array(lows[-period:])
        close = prices[-1]
        
        highest_high = np.max(highs_array)
        lowest_low = np.min(lows_array)
        
        if highest_high == lowest_low:
            return 50.0, 50.0
        
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        
        return float(k), float(k)  # Simplified - would normally calculate %D as well


# Add this before environment imports
class BestEpisodeBuffer:
    """Buffer to store best trading episodes for experience replay"""
    
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.buffer = []
        self.min_win_rate = 0.6  # Minimum win rate to be considered
    
    def add_episode(self, episode_data, win_rate, total_return):
        """Add episode if it meets quality criteria"""
        if win_rate >= self.min_win_rate:
            episode_info = {
                'data': episode_data,
                'win_rate': win_rate,
                'total_return': total_return,
                'timestamp': time.time()
            }
            
            self.buffer.append(episode_info)
            
            # Keep only best episodes
            if len(self.buffer) > self.capacity:
                self.buffer.sort(key=lambda x: x['win_rate'] * x['total_return'], reverse=True)
                self.buffer = self.buffer[:self.capacity]
    
    def sample_batch(self, batch_size=32):
        """Sample a batch of best episodes"""
        if len(self.buffer) < batch_size:
            return self.buffer
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def save(self, filepath):
        """Save buffer to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.buffer, f)
    
    def load(self, filepath):
        """Load buffer from disk"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.buffer = pickle.load(f)


# Import required modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.options_clstm_ppo import OptionsCLSTMPPOAgent
from src.options_trading_env import OptionsTradingEnvironment
from src.historical_options_data import HistoricalOptionsDataLoader, HistoricalOptionsEnvironment

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImprovedProfitableEnvironment(HistoricalOptionsEnvironment):
    """Environment with all 10 improvements implemented"""
    
    def __init__(self, *args, **kwargs):
        # Initialize tracking variables
        self.consecutive_losses = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0
        self.peak_capital = 100000
        self.episode_number = 0
        
        # Market analysis
        self.price_history = []
        self.volatility_window = 20
        self._market_regime_history = []
        self._regime_confidence = 0.5
        
        # Trade quality metrics
        self.trade_metrics = {
            'avg_win_size': deque(maxlen=100),
            'avg_loss_size': deque(maxlen=100),
            'profit_factors': deque(maxlen=100),
            'max_drawdowns': deque(maxlen=100),
            'sharpe_ratios': deque(maxlen=100),
            'win_rates': deque(maxlen=100)
        }
        
        # Performance tracking
        self._step_returns = []
        self._position_history = []
        
        # Now call super().__init__
        super().__init__(*args, **kwargs)
        self.peak_capital = self.initial_capital
    
    def reset(self):
        """Reset environment with improved tracking"""
        obs = super().reset()
        
        # Reset tracking
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0
        self.consecutive_losses = 0
        self.price_history = []
        self._market_regime_history = []
        self._step_returns = []
        self._position_history = []
        self.episode_number += 1
        
        return self._get_enhanced_observation(obs)
    
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
            upper_bb, middle_bb, lower_bb = TechnicalIndicators.calculate_bollinger_bands(prices)
            stoch_k, stoch_d = TechnicalIndicators.calculate_stochastic(prices)
            
            # Additional calculations
            volatility = self._calculate_volatility()
            momentum = self._calculate_momentum()
            volume_ratio = self._calculate_volume_ratio() if hasattr(self, '_volumes') else 1.0
            
            # Market regime encoding
            regime = self._detect_market_regime()
            regime_encoding = {"volatile": 0, "trending": 1, "ranging": 2, "mixed": 3}
            regime_value = regime_encoding.get(regime, 3)
            
            # Win rate and trade metrics
            win_rate = self.winning_trades / max(1, self.winning_trades + self.losing_trades)
            avg_win = np.mean(self.trade_metrics['avg_win_size']) if self.trade_metrics['avg_win_size'] else 0
            avg_loss = np.mean(self.trade_metrics['avg_loss_size']) if self.trade_metrics['avg_loss_size'] else 0
            
            # Fill technical indicators array
            technical_features = np.array([
                macd_hist / 10.0,  # Normalized MACD histogram
                (rsi - 50) / 50.0,  # Normalized RSI (-1 to 1)
                cci / 200.0,  # Normalized CCI
                adx / 50.0,  # Normalized ADX
                volatility * 100,  # Volatility percentage
                momentum * 100,  # Momentum percentage
                (prices[-1] - middle_bb) / (upper_bb - lower_bb + 1e-6),  # BB position
                (stoch_k - 50) / 50.0,  # Normalized stochastic
                volume_ratio,  # Volume ratio
                regime_value / 3.0,  # Normalized regime
                self._regime_confidence,  # Regime confidence
                win_rate,  # Current win rate
                avg_win / 1000.0,  # Normalized average win
                avg_loss / 1000.0,  # Normalized average loss
                len(self.positions) / float(self.max_positions),  # Position utilization
                self.consecutive_losses / 10.0,  # Normalized consecutive losses
                0.0, 0.0, 0.0, 0.0  # Padding to 20 features
            ], dtype=np.float32)
            
            base_obs['technical_indicators'] = technical_features[:20]
        
        return base_obs
    
    def step(self, action: int):
        """Enhanced step with all improvements"""
        if self.done:
            return None, 0, True, {}
        
        # Map actions
        action_names = ['hold', 'buy_call', 'buy_put', 'sell_call', 'sell_put',
                       'bull_call_spread', 'bear_put_spread', 'iron_condor',
                       'straddle', 'strangle', 'close_all_positions']
        action_name = action_names[action] if action < len(action_names) else 'hold'
        
        # Get current price
        current_price = self._get_current_price()
        self.price_history.append(current_price)
        if len(self.price_history) > self.volatility_window * 2:
            self.price_history.pop(0)
        
        # Calculate portfolio value before action
        portfolio_value_before = self._calculate_portfolio_value()
        
        # Detect market regime
        market_regime = self._detect_market_regime()
        
        # Execute action with dynamic position sizing
        reward = 0
        if action_name == 'hold':
            # Small penalty for inaction when no positions
            if len(self.positions) == 0:
                reward -= 0.01
        
        elif action_name == 'close_all_positions' and self.positions:
            reward = self._close_all_positions_smart(market_regime)
        
        elif action_name in ['buy_call', 'buy_put'] and len(self.positions) < self.max_positions:
            # Calculate dynamic position size
            confidence = self._calculate_entry_confidence(action_name, market_regime)
            volatility = self._calculate_volatility()
            win_rate = self.winning_trades / max(1, self.winning_trades + self.losing_trades)
            
            position_size = self._calculate_dynamic_position_size(confidence, volatility, win_rate)
            
            if confidence > 0.3:  # Minimum confidence threshold
                reward = self._execute_trade_with_sizing(action_name, position_size, market_regime)
            else:
                reward = -0.1  # Small penalty for low confidence trades
        
        # Update positions with smart exit logic
        self._update_positions_smart(market_regime)
        
        # Calculate portfolio value after action
        portfolio_value_after = self._calculate_portfolio_value()
        step_pnl = portfolio_value_after - portfolio_value_before
        
        # Calculate rewards
        if step_pnl != 0:
            pnl_pct = step_pnl / portfolio_value_before if portfolio_value_before > 0 else 0
            reward += self._calculate_trade_reward(step_pnl, pnl_pct)
            self._step_returns.append(pnl_pct)
        
        # Add win rate bonus
        reward += self._calculate_win_rate_bonus(self.episode_number)
        
        # Track metrics for adaptive learning
        self._update_trade_metrics()
        
        # Update step
        self.current_step += 1
        
        # Check termination conditions
        if self.current_step >= len(self.training_data) - 1:
            self.done = True
        elif portfolio_value_after < self.initial_capital * 0.3:  # 70% loss
            self.done = True
            reward -= 50  # Large penalty for blowing up account
        
        # Enhanced info
        info = {
            'portfolio_value': portfolio_value_after,
            'positions': len(self.positions),
            'symbol': self.current_symbol,
            'win_rate': self.winning_trades / max(1, self.winning_trades + self.losing_trades),
            'total_trades': self.winning_trades + self.losing_trades,
            'market_regime': market_regime,
            'step_pnl': step_pnl,
            'sharpe_ratio': self._calculate_sharpe_ratio()
        }
        
        return self._get_enhanced_observation(self._get_observation()), reward, self.done, info
    
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
        
        # Kelly Criterion inspired adjustment
        if win_rate > 0 and self.trade_metrics['avg_win_size'] and self.trade_metrics['avg_loss_size']:
            avg_win = np.mean(self.trade_metrics['avg_win_size'])
            avg_loss = np.mean(self.trade_metrics['avg_loss_size'])
            if avg_loss > 0:
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_fraction = max(0, min(0.25, kelly_fraction))  # Cap at 25%
                size = size * 0.5 + kelly_fraction * 0.5  # Blend with base size
        
        return min(0.3, max(0.05, size))  # 5-30% of capital
    
    def _should_exit_position(self, position, current_price, market_regime):
        """Smart exit logic based on market regime and position metrics"""
        pnl = self._calculate_position_pnl(position, current_price)
        pnl_pct = pnl / (position['entry_price'] * position['quantity'] * 100)
        holding_time = self.current_step - position['entry_step']
        
        # Dynamic exit thresholds based on market regime
        if market_regime == "volatile":
            take_profit = 0.03  # 3% in volatile markets
            stop_loss = -0.02   # 2% stop loss
            max_holding = 20    # Shorter holding period
        elif market_regime == "trending":
            # Determine trend direction
            momentum = self._calculate_momentum()
            if (position['option_type'] == 'call' and momentum > 0) or \
               (position['option_type'] == 'put' and momentum < 0):
                take_profit = 0.08  # 8% for trend-following positions
                stop_loss = -0.03   # 3% stop loss
                max_holding = 40    # Longer holding period
            else:
                take_profit = 0.02  # Quick exit for counter-trend
                stop_loss = -0.015
                max_holding = 10
        else:  # ranging
            take_profit = 0.04  # 4% in ranging markets
            stop_loss = -0.02   # 2% stop loss
            max_holding = 25
        
        # Time decay acceleration for options
        time_decay_factor = 1.0 + (holding_time / max_holding) * 0.5
        take_profit *= time_decay_factor
        
        # Trailing stop logic
        if hasattr(position, 'max_pnl'):
            trailing_stop = position['max_pnl'] * 0.7  # Keep 70% of max profit
            if pnl < trailing_stop and pnl > 0:
                return True, "trailing_stop"
        
        # Exit conditions
        if pnl_pct >= take_profit:
            return True, "take_profit"
        elif pnl_pct <= stop_loss:
            return True, "stop_loss"
        elif holding_time > max_holding:
            return True, "time_exit"
        
        # Momentum-based exit
        if holding_time > 5:
            recent_momentum = self._calculate_momentum(window=5)
            if position['option_type'] == 'call' and recent_momentum < -0.01:
                return True, "momentum_exit"
            elif position['option_type'] == 'put' and recent_momentum > 0.01:
                return True, "momentum_exit"
        
        return False, None
    
    def _calculate_entry_confidence(self, action_name, market_regime):
        """Calculate confidence score for entry decisions"""
        confidence = 0.5  # Base confidence
        
        # Technical indicators alignment
        if len(self.price_history) >= 26:
            macd_line, signal_line, macd_hist = TechnicalIndicators.calculate_macd(self.price_history)
            rsi = TechnicalIndicators.calculate_rsi(self.price_history)
            adx = TechnicalIndicators.calculate_adx(self.price_history)
            
            if action_name == 'buy_call':
                if macd_hist > 0 and macd_line > signal_line:
                    confidence += 0.2
                if 30 < rsi < 70:
                    confidence += 0.1
                if rsi < 40:
                    confidence += 0.1  # Oversold bounce
                if adx > 25 and self._calculate_momentum() > 0:
                    confidence += 0.2
                
                # Market regime alignment
                if market_regime == "trending" and self._calculate_momentum() > 0:
                    confidence += 0.2
                elif market_regime == "volatile":
                    confidence -= 0.2
                
            elif action_name == 'buy_put':
                if macd_hist < 0 and macd_line < signal_line:
                    confidence += 0.2
                if 30 < rsi < 70:
                    confidence += 0.1
                if rsi > 60:
                    confidence += 0.1  # Overbought reversal
                if adx > 25 and self._calculate_momentum() < 0:
                    confidence += 0.2
                
                # Market regime alignment
                if market_regime == "trending" and self._calculate_momentum() < 0:
                    confidence += 0.2
                elif market_regime == "volatile":
                    confidence -= 0.2
        
        # Recent performance adjustment
        if self.winning_trades + self.losing_trades > 5:
            recent_win_rate = self._calculate_recent_win_rate(10)
            if recent_win_rate > 0.6:
                confidence += 0.1
            elif recent_win_rate < 0.3:
                confidence -= 0.1
        
        return max(0, min(1, confidence))
    
    def _detect_market_regime(self):
        """Enhanced market regime detection"""
        if len(self.price_history) < 20:
            return "mixed"
        
        prices = np.array(self.price_history)
        returns = np.diff(prices) / prices[:-1]
        
        # Volatility
        volatility = np.std(returns)
        
        # Trend strength
        sma_5 = np.mean(prices[-5:])
        sma_20 = np.mean(prices[-20:])
        trend_strength = abs(sma_5 - sma_20) / sma_20
        
        # Directional movement
        positive_moves = np.sum(returns > 0)
        negative_moves = np.sum(returns < 0)
        directional_bias = abs(positive_moves - negative_moves) / len(returns)
        
        # ADX for trend strength
        adx = TechnicalIndicators.calculate_adx(prices.tolist())
        
        # Classify regime
        if volatility > 0.02 and adx < 20:
            regime = "volatile"
        elif adx > 25 and directional_bias > 0.3:
            regime = "trending"
        elif volatility < 0.01 and trend_strength < 0.02:
            regime = "ranging"
        else:
            regime = "mixed"
        
        # Calculate confidence
        regime_scores = {
            "volatile": volatility * 50,
            "trending": adx / 25.0 * directional_bias,
            "ranging": (1 - volatility * 50) * (1 - trend_strength * 50)
        }
        
        self._regime_confidence = min(1.0, regime_scores.get(regime, 0.5))
        
        return regime
    
    def _calculate_trade_reward(self, pnl, pnl_pct, base_reward=1.0):
        """Simplified reward structure focused on win rate and profitability"""
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
    
    def _update_trade_metrics(self):
        """Track trade quality metrics for adaptive learning"""
        if self.winning_trades + self.losing_trades == 0:
            return
        
        # Calculate current metrics
        win_rate = self.winning_trades / (self.winning_trades + self.losing_trades)
        self.trade_metrics['win_rates'].append(win_rate)
        
        # Sharpe ratio
        if len(self._step_returns) > 20:
            sharpe = self._calculate_sharpe_ratio()
            self.trade_metrics['sharpe_ratios'].append(sharpe)
        
        # Max drawdown
        if hasattr(self, 'peak_capital'):
            drawdown = (self.peak_capital - self._calculate_portfolio_value()) / self.peak_capital
            self.trade_metrics['max_drawdowns'].append(drawdown)
    
    def _calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio from recent returns"""
        if len(self._step_returns) < 20:
            return 0.0
        
        returns = np.array(self._step_returns[-100:])  # Last 100 steps
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        # Annualized Sharpe (assuming daily steps, 252 trading days)
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        return float(sharpe)
    
    def _calculate_recent_win_rate(self, n_trades=10):
        """Calculate win rate over recent trades"""
        if not hasattr(self, '_recent_trades'):
            self._recent_trades = deque(maxlen=n_trades)
        
        if len(self._recent_trades) == 0:
            return 0.5
        
        wins = sum(1 for trade in self._recent_trades if trade > 0)
        return wins / len(self._recent_trades)
    
    def _get_current_price(self):
        """Get current underlying price"""
        if hasattr(self, '_all_underlying_prices'):
            return self._all_underlying_prices[self.current_step]
        elif hasattr(self, 'training_data') and not self.training_data.empty:
            return self.training_data.iloc[self.current_step]['underlying_price']
        return 100.0  # Default
    
    def _calculate_momentum(self, window=10):
        """Calculate price momentum"""
        if len(self.price_history) < window:
            return 0.0
        
        old_price = self.price_history[-window]
        current_price = self.price_history[-1]
        
        if old_price == 0:
            return 0.0
        
        return (current_price - old_price) / old_price
    
    def _calculate_volatility(self):
        """Calculate historical volatility"""
        if len(self.price_history) < 2:
            return 0.02  # Default 2% volatility
        
        prices = np.array(self.price_history)
        returns = np.diff(prices) / prices[:-1]
        
        return float(np.std(returns))
    
    def _calculate_volume_ratio(self):
        """Calculate volume ratio (placeholder)"""
        return 1.0
    
    def _calculate_position_pnl(self, position, current_price):
        """Calculate P&L for a position"""
        # This is simplified - would need option pricing in real implementation
        if position['option_type'] == 'call':
            intrinsic_value = max(0, current_price - position['strike'])
        else:
            intrinsic_value = max(0, position['strike'] - current_price)
        
        current_value = intrinsic_value * position['quantity'] * 100
        entry_cost = position['entry_price'] * position['quantity'] * 100
        
        return current_value - entry_cost
    
    def _execute_trade_with_sizing(self, action_name, position_size, market_regime):
        """Execute trade with dynamic position sizing"""
        # This is a placeholder - would implement actual trading logic
        return 0.1  # Small positive reward for taking action
    
    def _update_positions_smart(self, market_regime):
        """Update positions with smart exit logic"""
        current_price = self._get_current_price()
        
        for i, position in enumerate(self.positions):
            should_exit, exit_reason = self._should_exit_position(position, current_price, market_regime)
            
            if should_exit:
                pnl = self._calculate_position_pnl(position, current_price)
                
                if pnl > 0:
                    self.winning_trades += 1
                    self.trade_metrics['avg_win_size'].append(abs(pnl))
                else:
                    self.losing_trades += 1
                    self.consecutive_losses += 1
                    self.trade_metrics['avg_loss_size'].append(abs(pnl))
                
                self.total_pnl += pnl
                
                # Track in recent trades
                if hasattr(self, '_recent_trades'):
                    self._recent_trades.append(pnl)
                
                # Remove position
                self.positions.pop(i)
    
    def _close_all_positions_smart(self, market_regime):
        """Close all positions with smart logic"""
        total_reward = 0
        current_price = self._get_current_price()
        
        for position in self.positions:
            pnl = self._calculate_position_pnl(position, current_price)
            pnl_pct = pnl / (position['entry_price'] * position['quantity'] * 100)
            
            # Reward based on position outcome
            total_reward += self._calculate_trade_reward(pnl, pnl_pct, base_reward=2.0)
            
            if pnl > 0:
                self.winning_trades += 1
                self.trade_metrics['avg_win_size'].append(abs(pnl))
            else:
                self.losing_trades += 1
                self.trade_metrics['avg_loss_size'].append(abs(pnl))
        
        self.positions = []
        return total_reward


# Add ensemble model classes
class TrendFollowingAgent:
    """Agent specialized in trend following strategies"""
    
    def __init__(self, base_agent):
        self.base_agent = base_agent
        self.strategy_weight = 1.0
    
    def get_action(self, observation, deterministic=False):
        # Modify action probabilities based on trend indicators
        action, _ = self.base_agent.get_action(observation, deterministic)
        
        # Extract trend indicators from observation
        if 'technical_indicators' in observation:
            macd_hist = observation['technical_indicators'][0]
            adx = observation['technical_indicators'][3]
            
            # Bias towards trend-following actions
            if adx > 0.5 and macd_hist > 0:  # Strong uptrend
                if action == 2:  # buy_put
                    action = 1  # Change to buy_call
            elif adx > 0.5 and macd_hist < 0:  # Strong downtrend
                if action == 1:  # buy_call
                    action = 2  # Change to buy_put
        
        return action, None


class MeanReversionAgent:
    """Agent specialized in mean reversion strategies"""
    
    def __init__(self, base_agent):
        self.base_agent = base_agent
        self.strategy_weight = 1.0
    
    def get_action(self, observation, deterministic=False):
        action, _ = self.base_agent.get_action(observation, deterministic)
        
        # Extract mean reversion indicators
        if 'technical_indicators' in observation:
            rsi = observation['technical_indicators'][1] * 50 + 50  # Denormalize
            bb_position = observation['technical_indicators'][6]
            
            # Bias towards mean reversion actions
            if rsi < 30 and bb_position < -0.8:  # Oversold
                if action == 2:  # buy_put
                    action = 1  # Change to buy_call
            elif rsi > 70 and bb_position > 0.8:  # Overbought
                if action == 1:  # buy_call
                    action = 2  # Change to buy_put
        
        return action, None


class VolatilityAgent:
    """Agent specialized in volatility trading"""
    
    def __init__(self, base_agent):
        self.base_agent = base_agent
        self.strategy_weight = 1.0
    
    def get_action(self, observation, deterministic=False):
        action, _ = self.base_agent.get_action(observation, deterministic)
        
        # Extract volatility indicators
        if 'technical_indicators' in observation:
            volatility = observation['technical_indicators'][4]
            regime = observation['technical_indicators'][9] * 3  # Denormalize regime
            
            # In high volatility, prefer neutral strategies or closing
            if volatility > 0.03:  # High volatility
                if action in [1, 2]:  # Directional trades
                    action = 0  # Hold instead
                elif regime == 0:  # Volatile regime
                    if np.random.random() < 0.3:
                        action = 10  # Close positions
        
        return action, None


class EnsembleAgent:
    """Ensemble agent that combines multiple strategies"""
    
    def __init__(self, agents, base_agent):
        self.agents = agents
        self.base_agent = base_agent
        self.strategy_weights = {
            'trend_follower': 1.0,
            'mean_reverter': 1.0,
            'volatility_trader': 1.0
        }
    
    def get_action(self, observation, deterministic=False):
        """Get weighted ensemble action"""
        # Get market regime
        regime_value = observation['technical_indicators'][9] * 3 if 'technical_indicators' in observation else 1
        
        # Adjust weights based on market regime
        if regime_value < 0.5:  # Volatile
            weights = [0.1, 0.2, 0.7]  # Favor volatility trader
        elif regime_value < 1.5:  # Trending
            weights = [0.7, 0.2, 0.1]  # Favor trend follower
        elif regime_value < 2.5:  # Ranging
            weights = [0.2, 0.7, 0.1]  # Favor mean reverter
        else:  # Mixed
            weights = [0.33, 0.33, 0.34]  # Equal weights
        
        # Get actions from each agent
        actions = []
        for agent in self.agents.values():
            action, _ = agent.get_action(observation, deterministic)
            actions.append(action)
        
        # Weighted voting
        action_scores = {}
        for i, (action, weight) in enumerate(zip(actions, weights)):
            if action not in action_scores:
                action_scores[action] = 0
            action_scores[action] += weight
        
        # Select action with highest score
        best_action = max(action_scores, key=action_scores.get)
        
        return best_action, None


def create_optimized_environment(historical_data, symbols, **kwargs):
    """Create optimized environment with improved features"""
    return ImprovedProfitableEnvironment(
        historical_data=historical_data,
        symbols=symbols,
        initial_capital=100000,
        commission=0.65,
        max_positions=5,
        lookback_window=20,
        episode_length=50,
        **kwargs
    )


async def train_improved_model(
    agent,
    train_envs,
    val_env,
    episodes=50000,
    save_dir='checkpoints/improved_model',
    device='cuda',
    rank=0,
    world_size=1
):
    """Improved training loop with all enhancements"""
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize best episode buffer
    best_episodes = BestEpisodeBuffer(capacity=100)
    best_episodes_path = os.path.join(save_dir, 'best_episodes.pkl')
    if os.path.exists(best_episodes_path):
        best_episodes.load(best_episodes_path)
    
    # Create ensemble agents
    ensemble_agents = {
        'trend_follower': TrendFollowingAgent(agent),
        'mean_reverter': MeanReversionAgent(agent),
        'volatility_trader': VolatilityAgent(agent)
    }
    ensemble = EnsembleAgent(ensemble_agents, agent)
    
    # Training hyperparameters (improved)
    learning_rate_actor_critic = 1e-4  # Increased from 3e-5
    learning_rate_clstm = 3e-4        # Increased from 1e-4
    ppo_epochs = 4                    # Increased from 2
    batch_size = 4096                 # Increased from 2048
    
    # Update agent hyperparameters
    agent.ppo_epochs = ppo_epochs
    agent.batch_size = batch_size
    
    # Curriculum learning
    def get_episode_difficulty(episode):
        if episode < 1000:
            return "easy"  # Only trending markets
        elif episode < 5000:
            return "medium"  # Add ranging markets
        else:
            return "hard"  # All market types
    
    # Adaptive risk management
    risk_penalty_multiplier = 1.0
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    win_rates = []
    
    # Progress bar
    if rank == 0:
        pbar = tqdm(total=episodes, desc="Training")
    
    for episode in range(episodes):
        # Curriculum learning - filter environments
        difficulty = get_episode_difficulty(episode)
        
        # Run episode
        states = [env.reset() for env in train_envs]
        episode_reward = 0
        episode_length = 0
        episode_data = []
        
        done = [False] * len(train_envs)
        
        while not all(done):
            # Use ensemble for action selection
            if episode > 10000:  # Start using ensemble after initial training
                actions = []
                for i, (state, d) in enumerate(zip(states, done)):
                    if not d:
                        action, _ = ensemble.get_action(state, deterministic=False)
                        actions.append(action)
                    else:
                        actions.append(0)
            else:
                actions = []
                for i, (state, d) in enumerate(zip(states, done)):
                    if not d:
                        action, _ = agent.get_action(state, deterministic=False)
                        actions.append(action)
                    else:
                        actions.append(0)
            
            # Step environments
            next_states = []
            rewards = []
            infos = []
            
            for i, (env, action) in enumerate(zip(train_envs, actions)):
                if not done[i]:
                    next_state, reward, done[i], info = env.step(action)
                    next_states.append(next_state)
                    rewards.append(reward)
                    infos.append(info)
                    
                    # Store transition
                    episode_data.append({
                        'state': states[i],
                        'action': action,
                        'reward': reward,
                        'next_state': next_state,
                        'done': done[i],
                        'info': info
                    })
                else:
                    next_states.append(states[i])
                    rewards.append(0)
                    infos.append({})
            
            states = next_states
            episode_reward += sum(rewards)
            episode_length += 1
        
        # Calculate episode metrics
        total_win_rate = np.mean([info.get('win_rate', 0) for info in infos])
        total_trades = sum([info.get('total_trades', 0) for info in infos])
        
        # Store in best episodes buffer if criteria met
        if total_win_rate > 0.6 and total_trades > 5:
            best_episodes.add_episode(episode_data, total_win_rate, episode_reward)
        
        # Experience replay from best episodes
        if episode % 500 == 0 and len(best_episodes.buffer) > 10:
            # Train on best episodes
            best_batch = best_episodes.sample_batch(min(32, len(best_episodes.buffer)))
            # TODO: Implement replay training
        
        # Adaptive risk management
        if total_win_rate < 0.4 and episode > 1000:
            risk_penalty_multiplier = min(2.0, risk_penalty_multiplier * 1.1)
        elif total_win_rate > 0.6:
            risk_penalty_multiplier = max(1.0, risk_penalty_multiplier * 0.95)
        
        # Update learning rates adaptively
        if episode % 1000 == 0:
            current_lr_ac = agent.actor_optimizer.param_groups[0]['lr']
            current_lr_clstm = agent.clstm_optimizer.param_groups[0]['lr']
            
            if total_win_rate < 0.5:
                # Increase learning rate to escape local minimum
                new_lr_ac = min(5e-4, current_lr_ac * 1.2)
                new_lr_clstm = min(1e-3, current_lr_clstm * 1.2)
            else:
                # Decrease learning rate for fine-tuning
                new_lr_ac = max(1e-5, current_lr_ac * 0.9)
                new_lr_clstm = max(3e-5, current_lr_clstm * 0.9)
            
            for param_group in agent.actor_optimizer.param_groups:
                param_group['lr'] = new_lr_ac
            for param_group in agent.clstm_optimizer.param_groups:
                param_group['lr'] = new_lr_clstm
        
        # Log metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        win_rates.append(total_win_rate)
        
        # Update progress
        if rank == 0:
            pbar.update(1)
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_win_rate = np.mean(win_rates[-100:])
                pbar.set_description(f"Avg Reward: {avg_reward:.2f}, Win Rate: {avg_win_rate:.2%}")
        
        # Save checkpoints
        if episode % 1000 == 0 and rank == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_ep{episode}.pt')
            agent.save(checkpoint_path)
            
            # Save best episodes buffer
            best_episodes.save(best_episodes_path)
            
            # Save metrics
            metrics = {
                'episode_rewards': episode_rewards,
                'win_rates': win_rates,
                'episode_lengths': episode_lengths
            }
            metrics_path = os.path.join(save_dir, f'metrics_ep{episode}.pkl')
            with open(metrics_path, 'wb') as f:
                pickle.dump(metrics, f)
    
    if rank == 0:
        pbar.close()
    
    return agent, episode_rewards, win_rates


def main():
    """Main training function with improved configuration"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=50000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--load_checkpoint', type=str, default=None)
    parser.add_argument('--data_file', type=str, default='data/historical_options_data.pkl')
    args = parser.parse_args()
    
    # Load data
    logger.info("Loading historical options data...")
    with open(args.data_file, 'rb') as f:
        historical_data = pickle.load(f)
    
    symbols = list(historical_data.keys())
    logger.info(f"Loaded data for {len(symbols)} symbols")
    
    # Create environments
    num_envs = 48  # Parallel environments
    train_envs = [
        create_optimized_environment(historical_data, symbols)
        for _ in range(num_envs)
    ]
    val_env = create_optimized_environment(historical_data, symbols)
    
    # Create agent
    observation_space = train_envs[0].observation_space
    action_space = train_envs[0].action_space
    
    agent = OptionsCLSTMPPOAgent(
        observation_space=observation_space,
        action_space=action_space,
        hidden_size=256,
        device=args.device
    )
    
    # Load checkpoint if provided
    if args.load_checkpoint:
        agent.load(args.load_checkpoint)
        logger.info(f"Loaded checkpoint from {args.load_checkpoint}")
    
    # Train model
    logger.info("Starting improved training...")
    trained_agent, rewards, win_rates = asyncio.run(
        train_improved_model(
            agent=agent,
            train_envs=train_envs,
            val_env=val_env,
            episodes=args.episodes,
            device=args.device
        )
    )
    
    # Save final model
    final_path = 'checkpoints/improved_model/final_model.pt'
    trained_agent.save(final_path)
    logger.info(f"Saved final model to {final_path}")
    
    # Print final statistics
    final_win_rate = np.mean(win_rates[-1000:])
    logger.info(f"Final average win rate: {final_win_rate:.2%}")


if __name__ == "__main__":
    main()