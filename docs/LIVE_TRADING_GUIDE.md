# Live Trading Guide

## Overview

The trading bot now supports a hybrid mode that trains on historical data while occasionally executing real trades through Alpaca's API. This solves the problem of "no training data available in live mode" by using historical options data for learning while applying that knowledge to live market conditions.

## How It Works

1. **Historical Data Training**: The bot loads historical options data for your selected symbols
2. **Live Signal Generation**: As it trains, it generates trading signals based on learned patterns
3. **Selective Execution**: A configurable percentage of these signals are executed as real trades
4. **Continuous Learning**: The model continues to improve while trading

## Setup

### 1. Get Alpaca API Credentials

1. Sign up at [https://alpaca.markets/](https://alpaca.markets/)
2. Generate API keys (use paper trading keys for testing!)
3. Run the setup script:
   ```bash
   source venv/bin/activate
   python setup_alpaca_live_trading.py
   ```

### 2. Configure Environment

Create or update your `.env` file:
```bash
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
```

### 3. Test Connection

```bash
python check_alpaca_credentials.py
```

## Running Live Trading

### Basic Command

```bash
python train.py --live-mode
```

### With Options

```bash
python train.py --live-mode \
    --paper-trading \              # Use paper trading account
    --live-capital 10000 \         # Trading capital
    --live-symbols SPY QQQ AAPL \  # Symbols to trade
    --position-size 0.05 \         # 5% position size
    --daily-loss-limit 0.02        # 2% daily loss limit
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--live-mode` | False | Enable live trading mode |
| `--paper-trading` | False | Use paper trading account |
| `--live-capital` | 10000 | Trading capital in dollars |
| `--live-symbols` | SPY QQQ | Space-separated list of symbols |
| `--position-size` | 0.05 | Position size as fraction of capital (5%) |
| `--daily-loss-limit` | 0.02 | Daily loss limit as fraction (2%) |

## Trade Execution

### Execution Probability

By default, only 10% of trading signals are executed live. This is controlled by the `execution_probability` setting in the code:

```python
'execution_probability': 0.1  # Execute 10% of signals
```

### Trade Signals

When a trade signal is generated, you'll see:
```
📈 TRADE SIGNAL: buy_call for SPY
   Current price: $450.23
   Position size: $500.00
   Momentum: 0.015
   RSI: 55.2
   Market regime: trending
```

### Trade Summary

At the end of each episode:
```
📊 Episode Trade Signals Summary:
   Total signals: 5
   Buy calls: 3, Buy puts: 2
   Average momentum at signal: 0.012
```

## Safety Features

1. **API Validation**: Credentials are validated before trading
2. **Fallback Mode**: Automatically switches to simulation if API fails
3. **Position Limits**: Enforces maximum position sizes
4. **Loss Limits**: Stops trading if daily loss limit is hit
5. **Signal Filtering**: Only trades when technical conditions are favorable

## Monitoring

### Real-time Logs

The system provides detailed logs:
- Trade signals with technical indicators
- API connection status
- Training progress and performance
- Win rates and returns

### Performance Tracking

- Win rate tracking
- P&L monitoring
- Position management
- Risk metrics

## Troubleshooting

### "Request is not authorized"

1. Check your API credentials in `.env`
2. Ensure keys are for the correct environment (paper vs live)
3. Verify account is activated and funded
4. Run `python setup_alpaca_live_trading.py` to update credentials

### No Trade Signals

1. Increase `execution_probability` in the code
2. Check if symbols have available options data
3. Verify market hours (options trade 9:30 AM - 4:00 PM ET)
4. Review momentum and RSI thresholds

### Connection Issues

1. Check internet connection
2. Verify Alpaca API status
3. Try paper trading endpoint first
4. Check for rate limiting

## Best Practices

1. **Start with Paper Trading**: Always test with paper trading first
2. **Small Position Sizes**: Use 1-5% position sizes initially
3. **Monitor Actively**: Watch the logs during initial runs
4. **Gradual Scaling**: Increase position sizes gradually
5. **Risk Management**: Set appropriate daily loss limits

## Example Session

```bash
# Activate environment
source venv/bin/activate

# Setup credentials (one time)
python setup_alpaca_live_trading.py

# Run paper trading with conservative settings
python train.py --live-mode \
    --paper-trading \
    --live-capital 10000 \
    --live-symbols SPY QQQ \
    --position-size 0.02 \
    --daily-loss-limit 0.01 \
    --episodes 100
```

## Advanced Configuration

To modify the execution probability or other advanced settings, edit the live trading configuration in `train.py`:

```python
builtins.live_trading_config = {
    'enabled': True,
    'api_key': api_key,
    'api_secret': api_secret,
    'base_url': base_url,
    'position_size_pct': 0.05,
    'max_daily_loss_pct': 0.02,
    'execution_probability': 0.1  # Increase to execute more trades
}
```

## Integration with Training

The live mode seamlessly integrates with the training process:
- Loads historical data for your symbols
- Trains the CLSTM-PPO model on this data
- Generates trading signals based on learned patterns
- Executes a portion of these signals as real trades
- Continues learning from both historical and live data

This hybrid approach ensures the model has sufficient data to learn from while still being able to execute real trades based on current market conditions.