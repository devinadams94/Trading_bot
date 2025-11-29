"""Technical indicators and turbulence calculation"""

import numpy as np
import pandas as pd
from typing import List
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
        
    def calculate_turbulence(self, returns: np.ndarray, historical_returns: np.ndarray) -> float:
        """
        Calculate turbulence index as defined in paper:
        turbulence_t = (y_t - μ) * Σ^(-1) * (y_t - μ)'
        """
        try:
            mu = np.mean(historical_returns, axis=0)
            sigma = np.cov(historical_returns.T)
            sigma += np.eye(sigma.shape[0]) * 1e-6
            diff = returns - mu
            turbulence = diff.T @ np.linalg.inv(sigma) @ diff
            return float(turbulence)
        except Exception as e:
            logger.warning(f"Error calculating turbulence: {e}")
            return 0.0
    
    def update_threshold(self, turbulence_values: List[float]):
        """Update turbulence threshold to 90th percentile"""
        if len(turbulence_values) > 10:
            old_threshold = self.threshold
            self.threshold = np.percentile(turbulence_values, self.percentile)
            if old_threshold is None or abs(self.threshold - old_threshold) / max(old_threshold, 1e-6) > 0.01:
                logger.info(f"Updated turbulence threshold to {self.threshold:.4f}")
    
    def should_stop_trading(self, current_turbulence: float) -> bool:
        """Check if trading should be stopped due to high turbulence"""
        if self.threshold is None:
            return False
        return current_turbulence > self.threshold


class TechnicalIndicators:
    """Technical indicators used in the paper: MACD, RSI, CCI, ADX"""
    
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

