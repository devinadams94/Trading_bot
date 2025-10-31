#!/usr/bin/env python3
"""
Paper-based optimizations for CLSTM-PPO trading system
Based on "A Novel Deep Reinforcement Learning Based Automated Stock Trading System Using Cascaded LSTM Networks"
https://arxiv.org/abs/2212.02721
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TurbulenceCalculator:
    """
    Turbulence threshold calculation from the paper
    Measures extreme asset price movements to avoid market crashes
    """
    
    def __init__(self, percentile: float = 90):
        self.percentile = percentile
        self.historical_turbulence = []
        self.threshold = None
        
    def calculate_turbulence(self, returns: np.ndarray, 
                           historical_returns: np.ndarray) -> float:
        """
        Calculate turbulence index as defined in paper:
        turbulence_t = (y_t - μ) * Σ^(-1) * (y_t - μ)'
        
        Args:
            returns: Current period returns (n_assets,)
            historical_returns: Historical returns matrix (n_periods, n_assets)
        
        Returns:
            Turbulence index value
        """
        try:
            # Calculate mean and covariance of historical returns
            mu = np.mean(historical_returns, axis=0)
            sigma = np.cov(historical_returns.T)
            
            # Add small regularization to avoid singular matrix
            sigma += np.eye(sigma.shape[0]) * 1e-6
            
            # Calculate turbulence
            diff = returns - mu
            turbulence = diff.T @ np.linalg.inv(sigma) @ diff
            
            return float(turbulence)
            
        except Exception as e:
            logger.warning(f"Error calculating turbulence: {e}")
            return 0.0
    
    def update_threshold(self, turbulence_values: List[float]):
        """Update turbulence threshold to 90th percentile"""
        if len(turbulence_values) > 10:  # Need minimum data
            old_threshold = self.threshold
            self.threshold = np.percentile(turbulence_values, self.percentile)
            # Only log if threshold changed significantly (more than 1%) or first time
            if old_threshold is None or abs(self.threshold - old_threshold) / max(old_threshold, 1e-6) > 0.01:
                logger.info(f"Updated turbulence threshold to {self.threshold:.4f}")
    
    def should_stop_trading(self, current_turbulence: float) -> bool:
        """Check if trading should be stopped due to high turbulence"""
        if self.threshold is None:
            return False
        return current_turbulence > self.threshold


class EnhancedRewardFunction:
    """
    Enhanced reward function based on paper's portfolio return approach
    Includes transaction costs and risk management
    """
    
    def __init__(self, transaction_cost_rate: float = 0.001, 
                 reward_scaling: float = 1e-4):
        self.transaction_cost_rate = transaction_cost_rate
        self.reward_scaling = reward_scaling
        
    def calculate_portfolio_return_reward(self, 
                                        prev_portfolio_value: float,
                                        current_portfolio_value: float,
                                        transaction_costs: float) -> float:
        """
        Calculate reward as portfolio value change minus transaction costs
        As defined in paper equation (1)
        """
        portfolio_change = current_portfolio_value - prev_portfolio_value
        net_return = portfolio_change - transaction_costs
        
        # Apply reward scaling as in paper
        scaled_reward = net_return * self.reward_scaling
        
        return scaled_reward
    
    def calculate_transaction_costs(self, trades: Dict[str, float], 
                                  prices: Dict[str, float]) -> float:
        """
        Calculate transaction costs as 0.1% of trade value
        As defined in paper equation (2)
        """
        total_cost = 0.0
        for symbol, quantity in trades.items():
            if symbol in prices and quantity != 0:
                trade_value = abs(quantity * prices[symbol])
                cost = trade_value * self.transaction_cost_rate
                total_cost += cost
        
        return total_cost


class CascadedLSTMFeatureExtractor(nn.Module):
    """
    Cascaded LSTM feature extractor as described in the paper
    Extracts time-series features from market data before PPO training
    """
    
    def __init__(self, input_size: int = 181, hidden_size: int = 128, 
                 output_size: int = 128, time_window: int = 30):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.time_window = time_window
        
        # LSTM layer for feature extraction
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        
        # Linear layers as described in paper
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
        # Activation function
        self.tanh = nn.Tanh()
        
        logger.info(f"Initialized Cascaded LSTM Feature Extractor:")
        logger.info(f"  Input size: {input_size}")
        logger.info(f"  Hidden size: {hidden_size}")
        logger.info(f"  Output size: {output_size}")
        logger.info(f"  Time window: {time_window}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through cascaded LSTM
        
        Args:
            x: Input tensor of shape (batch_size, time_window, input_size)
            
        Returns:
            Feature tensor of shape (batch_size, output_size)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last time step output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Pass through linear layers with Tanh activation
        x = self.tanh(self.linear1(last_output))
        x = self.tanh(self.linear2(x))
        features = self.linear3(x)
        
        return features


class TechnicalIndicators:
    """
    Technical indicators used in the paper: MACD, RSI, CCI, ADX
    """
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> float:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            return float(macd_line.iloc[-1]) if len(macd_line) > 0 else 0.0
        except:
            return 0.0
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1]) if len(rsi) > 0 else 50.0
        except:
            return 50.0
    
    @staticmethod
    def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> float:
        """Calculate CCI (Commodity Channel Index)"""
        try:
            typical_price = (high + low + close) / 3
            sma = typical_price.rolling(window=period).mean()
            mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (typical_price - sma) / (0.015 * mad)
            return float(cci.iloc[-1]) if len(cci) > 0 else 0.0
        except:
            return 0.0
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Calculate ADX (Average Directional Index)"""
        try:
            # Simplified ADX calculation
            high_diff = high.diff()
            low_diff = low.diff()
            
            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            minus_dm = (-low_diff).where((low_diff > high_diff) & (low_diff < 0), 0)
            
            tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
            
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
            
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            
            return float(adx.iloc[-1]) if len(adx) > 0 else 25.0
        except:
            return 25.0


def create_paper_optimized_config() -> Dict:
    """Create configuration optimized based on paper findings"""
    return {
        # Paper's optimal hyperparameters (Table 1)
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'clip_epsilon': 0.2,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'batch_size': 128,
        'max_grad_norm': 0.5,
        
        # LSTM parameters (paper found TW=30 optimal)
        'lstm_time_window': 30,
        'lstm_hidden_size': 128,
        'lstm_features_out': 128,
        
        # Risk management
        'use_turbulence_threshold': True,
        'turbulence_percentile': 90,
        'transaction_cost_rate': 0.001,
        'reward_scaling': 1e-4,
        
        # Portfolio settings (paper used 30 stocks, $1M capital)
        'initial_capital': 1000000,
        'max_positions': 100,  # Paper allowed up to 100 shares per trade
        
        # Technical indicators from paper
        'technical_indicators': ['MACD', 'RSI', 'CCI', 'ADX']
    }
