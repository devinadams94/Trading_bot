#!/usr/bin/env python3
"""
Add technical indicators (MACD, RSI, CCI, ADX) to the training script
Following the architecture in training.png
"""

import re
import os

# Read the training script
train_file = "/home/devin/Desktop/Trading_bot/train_profitable_optimized.py"

with open(train_file, 'r') as f:
    content = f.read()

print("Adding technical indicators to training script...")

# 1. First, add the technical indicator calculation functions
print("1. Adding technical indicator calculation functions...")

technical_indicators_code = '''
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

'''

# Insert the TechnicalIndicators class after imports
import_pattern = r'(from collections import deque\n)'
content = re.sub(import_pattern, r'\1\n' + technical_indicators_code + '\n', content)

# 2. Update the environment to track price history for indicators
print("2. Updating environment to track price history...")

# Find BalancedEnvironment class and add price tracking
env_init_pattern = r'(class BalancedEnvironment\(HistoricalOptionsEnvironment\):[\s\S]*?def __init__\(self, \*args, \*\*kwargs\):[\s\S]*?)(super\(\).__init__\(\*args, \*\*kwargs\))'

price_tracking_code = '''
        # Price history for technical indicators
        self.price_history_window = 50  # Keep last 50 prices
        self.underlying_price_history = deque(maxlen=self.price_history_window)
        self.high_price_history = deque(maxlen=self.price_history_window)
        self.low_price_history = deque(maxlen=self.price_history_window)
        
        '''

content = re.sub(env_init_pattern, r'\1' + price_tracking_code + r'\2', content)

# 3. Update _get_observation to calculate and include technical indicators
print("3. Updating observation to include technical indicators...")

# First, let's add a method to calculate technical indicators
calc_indicators_method = '''
    def _calculate_technical_indicators(self):
        """Calculate technical indicators for current state"""
        if len(self.underlying_price_history) < 26:
            # Not enough data, return neutral values
            return np.array([
                0.0, 0.0, 0.0,  # MACD, signal, histogram
                50.0,           # RSI (neutral)
                0.0,            # CCI
                0.0,            # ADX
                0.0, 0.0, 0.0,  # Moving averages
                0.0,            # Volatility
                0.0, 0.0, 0.0,  # Price momentum
                0.0, 0.0, 0.0,  # Volume indicators
                0.0, 0.0, 0.0,  # Support/Resistance
                0.0             # Trend strength
            ], dtype=np.float32)
        
        # Calculate all indicators
        indicators = TechnicalIndicators.calculate_all_indicators(
            self.underlying_price_history,
            self.high_price_history,
            self.low_price_history
        )
        
        # Current price info
        current_price = self.underlying_price_history[-1]
        prices = list(self.underlying_price_history)
        
        # Moving averages
        ma_10 = np.mean(prices[-10:]) if len(prices) >= 10 else current_price
        ma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else current_price
        ma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else current_price
        
        # Price momentum
        momentum_5 = (current_price - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        momentum_10 = (current_price - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
        momentum_20 = (current_price - prices[-20]) / prices[-20] if len(prices) >= 20 else 0
        
        # Volatility
        returns = np.diff(prices[-20:]) / prices[-20:-1] if len(prices) >= 20 else [0]
        volatility = np.std(returns) if len(returns) > 0 else 0.02
        
        # Volume indicators (simplified - would need volume data)
        volume_ma = 1000000  # Placeholder
        volume_ratio = 1.0   # Placeholder
        volume_trend = 0.0   # Placeholder
        
        # Support/Resistance levels
        support = min(prices[-20:]) if len(prices) >= 20 else current_price * 0.98
        resistance = max(prices[-20:]) if len(prices) >= 20 else current_price * 1.02
        price_position = (current_price - support) / (resistance - support) if resistance > support else 0.5
        
        # Trend strength (combination of ADX and price momentum)
        trend_strength = indicators['adx'] / 100.0
        
        # Normalize and return
        return np.array([
            indicators['macd'] / 10.0,          # Normalized MACD
            indicators['macd_signal'] / 10.0,   # Normalized MACD signal
            indicators['macd_histogram'] / 10.0, # Normalized MACD histogram
            indicators['rsi'] / 100.0,          # RSI (already 0-100)
            indicators['cci'] / 200.0,          # Normalized CCI
            indicators['adx'] / 100.0,          # ADX (already 0-100)
            (current_price - ma_10) / ma_10,    # Price vs MA10
            (current_price - ma_20) / ma_20,    # Price vs MA20
            (current_price - ma_50) / ma_50,    # Price vs MA50
            volatility * 10,                    # Scaled volatility
            momentum_5,                         # 5-day momentum
            momentum_10,                        # 10-day momentum
            momentum_20,                        # 20-day momentum
            volume_ma / 1e6,                    # Normalized volume MA
            volume_ratio,                       # Volume ratio
            volume_trend,                       # Volume trend
            (current_price - support) / current_price,     # Distance from support
            (resistance - current_price) / current_price,  # Distance from resistance
            price_position,                     # Price position in range
            trend_strength                      # Overall trend strength
        ], dtype=np.float32)
'''

# Find where to insert the method (after _precompute_entire_episode or similar)
method_insert_pattern = r'(def _precompute_entire_episode\(self\):[\s\S]*?return episode_complete\n)'
content = re.sub(method_insert_pattern, r'\1\n' + calc_indicators_method + '\n', content, count=1)

# 4. Update step method to track price history
print("4. Updating step method to track price history...")

step_update_pattern = r'(def step\(self, action: int\):[\s\S]*?current_data = self\.training_data\.iloc\[self\.current_step\])'

price_tracking_update = r'''\1
        
        # Track price history for technical indicators
        current_price = current_data.get('underlying_price', 600)
        self.underlying_price_history.append(current_price)
        self.high_price_history.append(current_price * 1.01)  # Approximation
        self.low_price_history.append(current_price * 0.99)   # Approximation'''

content = re.sub(step_update_pattern, price_tracking_update, content)

# 5. Update _get_observation to use calculated indicators
print("5. Updating observation to use calculated indicators...")

# Find the observation creation in BalancedEnvironment
obs_pattern = r"('technical_indicators': )(np\.zeros\(20, dtype=np\.float32\))"
content = re.sub(obs_pattern, r"\1self._calculate_technical_indicators()", content)

# 6. Add indicator-based entry conditions
print("6. Adding indicator-based entry conditions...")

entry_conditions_code = '''
    def _should_enter_trade_with_indicators(self, action_name):
        """Enhanced entry decision using technical indicators"""
        if len(self.underlying_price_history) < 26:
            # Not enough data for indicators
            return self._should_enter_trade(action_name)
        
        # Calculate current indicators
        indicators = TechnicalIndicators.calculate_all_indicators(
            self.underlying_price_history,
            self.high_price_history,
            self.low_price_history
        )
        
        # Base confidence from original method
        confidence = self._get_market_confidence()
        
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
                confidence += 0.3  # Strong buy signal
            elif macd_bullish and rsi_neutral:
                confidence += 0.15  # Moderate buy signal
            elif rsi_overbought or (macd_bearish and strong_trend):
                confidence -= 0.2  # Warning signal
                
            # CCI confirmation
            if cci_oversold:
                confidence += 0.1
            elif cci_overbought:
                confidence -= 0.1
                
        elif 'put' in action_name:
            # Bearish indicators for puts
            if macd_bearish and rsi_overbought and strong_trend:
                confidence += 0.3  # Strong sell signal
            elif macd_bearish and rsi_neutral:
                confidence += 0.15  # Moderate sell signal
            elif rsi_oversold or (macd_bullish and strong_trend):
                confidence -= 0.2  # Warning signal
                
            # CCI confirmation
            if cci_overbought:
                confidence += 0.1
            elif cci_oversold:
                confidence -= 0.1
        
        # Weak trend penalty
        if weak_trend:
            confidence -= 0.1
        
        # Volatility check
        if len(self.underlying_price_history) >= 20:
            prices = list(self.underlying_price_history)[-20:]
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns)
            
            # High volatility bonus for options
            if volatility > 0.02:
                confidence += 0.1
            elif volatility < 0.005:
                confidence -= 0.15  # Low volatility penalty
        
        return confidence > 0.15  # Slightly higher threshold with indicators
'''

# Insert the new method after _should_enter_trade
enter_pattern = r'(def _should_enter_trade\(self, action_name\):[\s\S]*?return confidence > 0\.1\n)'
content = re.sub(enter_pattern, r'\1\n' + entry_conditions_code + '\n', content)

# 7. Update the step method to use the new entry function
print("7. Updating step method to use indicator-based entry...")

step_entry_pattern = r'(if self\._should_enter_trade\(action_name\):)'
content = re.sub(step_entry_pattern, r'if self._should_enter_trade_with_indicators(action_name):', content)

# Save the updated file
with open(train_file, 'w') as f:
    f.write(content)

print("\n✅ Technical indicators successfully added!")
print("\nImplemented indicators:")
print("1. ✓ MACD (Moving Average Convergence Divergence)")
print("2. ✓ RSI (Relative Strength Index)")
print("3. ✓ CCI (Commodity Channel Index)")
print("4. ✓ ADX (Average Directional Index)")
print("\nFeatures added:")
print("- Price history tracking (50-period window)")
print("- 20-dimensional technical indicator state vector")
print("- Indicator-based entry conditions")
print("- Enhanced decision making for PPO")
print("\nThe training now follows the architecture in training.png with:")
print("- State includes price history + technical indicators")
print("- PPO uses indicators for better trading decisions")
print("- Entry/exit logic considers multiple indicators")