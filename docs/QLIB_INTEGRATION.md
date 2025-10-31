# Qlib Integration for Enhanced Trading

## Overview

Based on the research paper "A Novel Deep Reinforcement Learning Based Automated Stock Trading System Using Cascaded LSTM Networks", we've integrated Microsoft's Qlib for enhanced feature engineering and market analysis.

## What is Qlib?

Qlib is an AI-oriented quantitative investment platform that provides:
- Advanced technical indicators (158+ features)
- Market microstructure analysis
- Cross-sectional features
- High-quality financial data infrastructure

## Installation

### 1. Install Qlib Package
```bash
pip install pyqlib
```

### 2. Download Market Data
```bash
# Download US market data (recommended)
python download_qlib_data.py --region us

# Download Chinese market data
python download_qlib_data.py --region cn

# Specify custom directory
python download_qlib_data.py --target-dir /path/to/qlib/data --region us
```

### 3. Verify Installation
```bash
python download_qlib_data.py --test-only
```

## Using Qlib in Training

### Basic Usage
```bash
# Enable Qlib features
python train_ppo_lstm.py --use-qlib --episodes 1000

# With custom data path
python train_ppo_lstm.py --use-qlib --qlib-data-path /path/to/qlib/data

# Different market region
python train_ppo_lstm.py --use-qlib --qlib-region cn
```

### Multi-GPU with Qlib
```bash
python train_ppo_lstm.py --distributed --use-qlib --episodes 5000
```

## Features Added by Qlib

### 1. Enhanced Technical Indicators
Based on the research paper, we include:
- **MACD**: Moving Average Convergence Divergence
- **RSI**: Relative Strength Index (14-period)
- **CCI**: Commodity Channel Index
- **ADX**: Average Directional Index
- **Volatility**: 20-day rolling standard deviation
- **Volume patterns**: Volume-weighted indicators

### 2. Market Microstructure Features
- Price-volume correlation
- Bid-ask spread proxies
- Intraday patterns
- Cross-sectional rankings

### 3. Advanced Time-Series Features
- Kaufman's Adaptive Moving Average (KAMA)
- Residuals from trend regression
- Distance to recent highs/lows
- Multi-timeframe returns

### 4. Cross-Sectional Features
- Relative volume ranking
- Return rankings among peer stocks
- Market dispersion metrics

## Implementation Details

### Feature Enhancement Process

1. **Data Collection**: Qlib fetches and aligns market data
2. **Feature Engineering**: 30+ technical indicators calculated
3. **Normalization**: Features normalized for neural network input
4. **Integration**: Features added to PPO observation space

### Architecture Changes

```python
# Original observation space
obs = {
    'price_history': [...],
    'technical_indicators': [...],
    'options_chain': [...],
    'portfolio_state': [...],
    'greeks_summary': [...]
}

# Enhanced with Qlib
obs = {
    # ... original features ...
    'qlib_features': [600 values],  # 30 features × 20 days
    'market_features': [5 values]    # Market-wide statistics
}
```

## Performance Impact

Based on the research paper's findings:
- **Cumulative returns**: +5% to +52% improvement
- **Sharpe ratio**: +37.4% in emerging markets
- **Win rate**: Improved consistency
- **Training efficiency**: Better feature extraction

## Configuration Options

### Environment Variables
```bash
# Set Qlib data directory
export QLIB_DATA_PATH=~/.qlib/qlib_data/us_data

# Set market region
export QLIB_REGION=us
```

### Command Line Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--use-qlib` | Enable Qlib features | False |
| `--qlib-region` | Market region (us, cn) | us |
| `--qlib-data-path` | Path to Qlib data | Auto-detect |

## Troubleshooting

### Qlib Not Initialized
```
WARNING: Qlib initialization failed. Proceeding without Qlib features.
```
**Solution**: Download Qlib data using `download_qlib_data.py`

### Missing Data
```
ERROR: No data loaded for symbol XYZ
```
**Solution**: Ensure the symbol exists in your Qlib dataset and date range is valid

### Memory Issues
If using Qlib increases memory usage:
1. Reduce lookback window in features
2. Use fewer symbols
3. Enable data caching

## Advanced Usage

### Custom Feature Sets
Modify `src/qlib_features.py` to add custom indicators:
```python
fields = [
    # Add your custom Qlib expressions
    "TA('BBANDS', $close, timeperiod=20)",
    "TA('STOCH', $high, $low, $close)",
    # ... more indicators
]
```

### Hybrid Models
The research paper used Qlib models (MLP, LSTM, LightGBM) as baselines. You can create ensemble strategies by combining PPO with Qlib predictions.

## Best Practices

1. **Start Simple**: Begin with basic Qlib features before adding complex ones
2. **Monitor Performance**: Compare training with/without Qlib
3. **Data Quality**: Ensure Qlib data covers your training period
4. **Feature Selection**: Not all features improve performance - experiment
5. **Computational Cost**: Qlib adds overhead - balance with benefits

## Research Paper Implementation

Following the paper's approach:
1. ✅ Integrated technical indicators (MACD, RSI, CCI, ADX)
2. ✅ Added time-series feature extraction
3. ✅ Implemented cross-sectional features
4. ✅ Enhanced observation space for LSTM processing

## Future Enhancements

- [ ] Implement Qlib prediction models (MLP, LSTM) for signal generation
- [ ] Add portfolio optimization using Qlib's tools
- [ ] Integrate Qlib's backtesting framework
- [ ] Support more market regions (EU, JP)

## References

- Research Paper: "A Novel Deep Reinforcement Learning Based Automated Stock Trading System Using Cascaded LSTM Networks"
- Qlib Documentation: https://qlib.readthedocs.io/
- Microsoft Research: https://github.com/microsoft/qlib