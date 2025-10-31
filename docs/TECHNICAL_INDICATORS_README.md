# Technical Indicators Implementation

## Overview
The training script now uses real market indicators from Alpaca data to make informed trading decisions.

## Implemented Indicators

### 1. MACD (Moving Average Convergence Divergence)
- **Purpose**: Identifies trend changes and momentum
- **Components**:
  - MACD Line: 12-day EMA - 26-day EMA
  - Signal Line: 9-day EMA of MACD
  - Histogram: MACD - Signal Line
- **Trading Signals**:
  - Bullish: MACD crosses above signal line
  - Bearish: MACD crosses below signal line

### 2. RSI (Relative Strength Index)
- **Purpose**: Measures overbought/oversold conditions
- **Range**: 0-100
- **Key Levels**:
  - Oversold: < 30 (potential buy signal)
  - Overbought: > 70 (potential sell signal)
- **Calculation**: 14-period RSI using average gains vs losses

### 3. CCI (Commodity Channel Index)
- **Purpose**: Identifies cyclical trends
- **Typical Range**: -100 to +100
- **Calculation**: (Typical Price - 20-period SMA) / (0.015 * Mean Deviation)
- **Trading Signals**:
  - > +100: Overbought
  - < -100: Oversold

### 4. ADX (Average Directional Index)
- **Purpose**: Measures trend strength (not direction)
- **Range**: 0-100
- **Key Levels**:
  - < 25: Weak or no trend
  - 25-50: Moderate trend
  - > 50: Strong trend
- **Components**:
  - +DI: Positive directional indicator
  - -DI: Negative directional indicator
  - ADX: Smoothed average of DX

## Neural Network Input Normalization

All indicators are normalized for optimal neural network performance:

1. **RSI**: Divided by 100 (0-1 range)
2. **MACD Components**: Normalized using tanh relative to price
3. **CCI**: Normalized using tanh(CCI/100)
4. **ADX**: Divided by 100 (0-1 range)
5. **Directional Indicators**: Divided by 50 and clipped to 0-1
6. **Moving Average Differences**: Normalized relative to price
7. **Bollinger Band Position**: Normalized to 0-1 range
8. **Volume Ratio**: Clipped to 0-3 range

## Additional Features

- **Price Position Indicators**: Price relative to moving averages
- **Trend Signals**: Binary signals for RSI extremes, MACD crossovers
- **Volume Analysis**: Volume relative to 20-day average
- **Volatility Measures**: Bollinger Band width, High/Low ratio

## Data Source

All price and volume data is fetched from Alpaca Markets API:
- Historical data: 2 years of daily bars
- Real-time updates during training
- Indicators calculated on actual market data

## Usage in Trading Decisions

The CLSTM-PPO model receives these normalized indicators as part of its observation space, allowing it to:
- Identify trend strength and direction
- Detect overbought/oversold conditions
- Recognize momentum shifts
- Make informed options trading decisions

## Future Enhancements

- Real-time options chain data integration
- Intraday technical indicators
- Market microstructure features
- Cross-asset correlations