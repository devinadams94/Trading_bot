# Intraday Data Options for Training

## Available Alpaca Timeframes

### Current Implementation (Hourly)
- **Timeframe**: 1Hour
- **Data points per day**: ~6.5 (market hours)
- **Data points per year**: ~1,638
- **Total for 2 years**: ~3,276 data points

### Other Available Options

#### 1. **15-Minute Bars** (More granular)
```python
bars = self.data_collector.api.get_bars(
    symbol, 
    '15Min',  # 26 bars per trading day
    start=start_date.strftime('%Y-%m-%d'),
    end=end_date.strftime('%Y-%m-%d'),
    feed='iex'
).df
```
- **Data points per day**: 26
- **Data points per year**: ~6,552
- **Observation window**: 520 bars (~20 days)

#### 2. **5-Minute Bars** (Very granular)
```python
bars = self.data_collector.api.get_bars(
    symbol, 
    '5Min',  # 78 bars per trading day
    start=start_date.strftime('%Y-%m-%d'),
    end=end_date.strftime('%Y-%m-%d'),
    feed='iex'
).df
```
- **Data points per day**: 78
- **Data points per year**: ~19,656
- **Observation window**: 1,560 bars (~20 days)
- **Note**: May hit API rate limits for multiple symbols

#### 3. **30-Minute Bars** (Balanced)
```python
bars = self.data_collector.api.get_bars(
    symbol, 
    '30Min',  # 13 bars per trading day
    start=start_date.strftime('%Y-%m-%d'),
    end=end_date.strftime('%Y-%m-%d'),
    feed='iex'
).df
```
- **Data points per day**: 13
- **Data points per year**: ~3,276
- **Observation window**: 260 bars (~20 days)

## Additional Data Features

### 1. **Volume Profile**
- Add volume-weighted indicators
- Track volume at different price levels
- Identify high-volume trading periods

### 2. **Market Microstructure**
```python
# Add bid-ask spread data
trades = self.data_collector.api.get_trades(symbol, start, end)
quotes = self.data_collector.api.get_quotes(symbol, start, end)
```

### 3. **Multi-Timeframe Features**
Combine multiple timeframes for richer features:
```python
# Get both hourly and daily data
hourly_bars = api.get_bars(symbol, '1Hour', ...)
daily_bars = api.get_bars(symbol, '1Day', ...)
```

### 4. **Extended Hours Data**
```python
# Include pre-market and after-hours
bars = api.get_bars(
    symbol,
    '15Min',
    start=start,
    end=end,
    feed='iex',
    asof='raw'  # Includes extended hours
)
```

## Technical Indicators on Intraday Data

With more frequent data, you can calculate:
- **Intraday momentum**: Shorter RSI periods (e.g., RSI-9)
- **VWAP**: Volume-weighted average price
- **Intraday support/resistance**: Based on hourly highs/lows
- **Market profile**: Volume at price levels
- **Order flow**: Buy/sell pressure from tick data

## Memory Considerations

- **1Hour**: ~650KB per symbol per year
- **15Min**: ~2.6MB per symbol per year
- **5Min**: ~7.8MB per symbol per year

## Recommended Approach

For options trading, **15-minute bars** offer the best balance:
- Captures intraday movements
- Sufficient for options pricing changes
- Manageable data size
- Good for technical indicators

To implement 15-minute bars, change line 149 to:
```python
'15Min',  # 15-minute bars for intraday patterns
```

And update the observation window (line 114) to:
```python
shape=(len(self.symbols), 520, 5),  # 520 = 20 days * 26 bars/day
```