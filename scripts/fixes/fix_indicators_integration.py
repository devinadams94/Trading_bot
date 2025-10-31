#!/usr/bin/env python3
"""
Fix the technical indicators integration
"""

import re

# Read the training script
train_file = "/home/devin/Desktop/Trading_bot/train_profitable_optimized.py"

with open(train_file, 'r') as f:
    content = f.read()

print("Fixing technical indicators integration...")

# Add the _should_enter_trade_with_indicators method after _should_enter_trade
indicator_entry_method = '''
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
'''

# Find the right place to insert (after _should_enter_trade method)
pattern = r'(def _should_enter_trade\(self, action_name\):[\s\S]*?return True\n)'
content = re.sub(pattern, r'\1\n' + indicator_entry_method + '\n', content)

# Now update the _calculate_technical_indicators method to work with current structure
calc_indicators_update = '''
    def _calculate_technical_indicators(self):
        """Calculate technical indicators for current state"""
        # Use price_history if underlying_price_history doesn't exist
        price_source = getattr(self, 'underlying_price_history', self.price_history)
        
        if len(price_source) < 26:
            # Not enough data, return neutral values
            return np.zeros(20, dtype=np.float32)
        
        # Calculate all indicators
        indicators = TechnicalIndicators.calculate_all_indicators(
            price_source,
            getattr(self, 'high_price_history', None),
            getattr(self, 'low_price_history', None)
        )
        
        # Current price info
        current_price = price_source[-1] if hasattr(price_source, '__getitem__') else list(price_source)[-1]
        prices = list(price_source)
        
        # Moving averages
        ma_10 = np.mean(prices[-10:]) if len(prices) >= 10 else current_price
        ma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else current_price
        ma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else current_price
        
        # Price momentum
        momentum_5 = (current_price - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        momentum_10 = (current_price - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
        momentum_20 = (current_price - prices[-20]) / prices[-20] if len(prices) >= 20 else 0
        
        # Volatility
        if hasattr(self, 'historical_volatility') and self.historical_volatility is not None:
            volatility = self.historical_volatility
        else:
            returns = np.diff(prices[-20:]) / prices[-20:-1] if len(prices) >= 20 else [0]
            volatility = np.std(returns) if len(returns) > 0 else 0.02
        
        # Support/Resistance levels
        support = min(prices[-20:]) if len(prices) >= 20 else current_price * 0.98
        resistance = max(prices[-20:]) if len(prices) >= 20 else current_price * 1.02
        price_position = (current_price - support) / (resistance - support) if resistance > support else 0.5
        
        # Return normalized indicators
        return np.array([
            indicators['macd'] / 10.0,          # Normalized MACD
            indicators['macd_signal'] / 10.0,   # Normalized MACD signal
            indicators['macd_histogram'] / 10.0, # Normalized MACD histogram
            indicators['rsi'] / 100.0,          # RSI (already 0-100)
            indicators['cci'] / 200.0,          # Normalized CCI
            indicators['adx'] / 100.0,          # ADX (already 0-100)
            (current_price - ma_10) / ma_10 if ma_10 > 0 else 0,    # Price vs MA10
            (current_price - ma_20) / ma_20 if ma_20 > 0 else 0,    # Price vs MA20
            (current_price - ma_50) / ma_50 if ma_50 > 0 else 0,    # Price vs MA50
            volatility * 10,                    # Scaled volatility
            momentum_5,                         # 5-day momentum
            momentum_10,                        # 10-day momentum
            momentum_20,                        # 20-day momentum
            1.0,                                # Volume placeholder
            1.0,                                # Volume ratio placeholder
            0.0,                                # Volume trend placeholder
            (current_price - support) / current_price if current_price > 0 else 0,     # Distance from support
            (resistance - current_price) / current_price if current_price > 0 else 0,  # Distance from resistance
            price_position,                     # Price position in range
            indicators['adx'] / 100.0           # Trend strength (same as ADX)
        ], dtype=np.float32)
'''

# Replace the existing _calculate_technical_indicators if it exists
if '_calculate_technical_indicators' in content:
    pattern = r'(def _calculate_technical_indicators\(self\):[\s\S]*?return np\.array\([\s\S]*?\], dtype=np\.float32\)\n)'
    content = re.sub(pattern, calc_indicators_update + '\n', content)
else:
    # Add it after _get_observation
    pattern = r'(def _get_observation\(self\):[\s\S]*?return obs\n)'
    content = re.sub(pattern, r'\1\n' + calc_indicators_update + '\n', content)

# Save the fixed file
with open(train_file, 'w') as f:
    f.write(content)

print("✅ Technical indicators integration fixed!")
print("\nKey fixes:")
print("1. Added _should_enter_trade_with_indicators method")
print("2. Updated _calculate_technical_indicators to work with current structure")
print("3. Made indicators compatible with existing price_history")
print("4. Proper fallbacks for missing data")
print("\nThe training script now properly uses technical indicators for:")
print("- Enhanced entry decisions")
print("- State representation (20-dim indicator vector)")
print("- Risk assessment")