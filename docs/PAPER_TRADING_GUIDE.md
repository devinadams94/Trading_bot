# Paper Trading Bot Guide

## Overview
A simplified paper trading bot that uses your trained PPO-LSTM model to simulate options trading without requiring Alpaca API credentials.

## Quick Start

```bash
# Basic usage with default settings
python paper_trading_bot.py --model checkpoints/ppo_lstm/best_model.pt

# Trade specific symbols for 2 hours
python paper_trading_bot.py \
    --model checkpoints/ppo_lstm/best_model.pt \
    --symbols AAPL MSFT GOOGL \
    --duration 120

# Start with different capital
python paper_trading_bot.py \
    --model checkpoints/ppo_lstm/best_model.pt \
    --capital 50000 \
    --duration 30
```

## Features

1. **No API Required**: Uses simulated market data
2. **Realistic Options**: Generates realistic options chains
3. **Risk Management**: 
   - 20% profit taking
   - 10% stop loss
   - Position limits (5 max)
4. **Live Logging**: See trades in real-time
5. **Trade History**: Saves all trades to JSON

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | Required | Path to trained model |
| `--symbols` | SPY QQQ IWM | Symbols to trade |
| `--capital` | 100000 | Starting capital |
| `--duration` | 60 | Minutes to run |

## Example Output

```
2024-01-10 10:00:00 - Starting paper trading session for 60 minutes
2024-01-10 10:00:00 - Symbols: ['SPY', 'QQQ', 'IWM']

--- Iteration 1 ---
2024-01-10 10:00:10 - SPY: Bought 5 call contracts (confidence: 75.3%)
2024-01-10 10:00:10 - Portfolio Value: $99,850.25
2024-01-10 10:00:10 - Daily P&L: $0.00
2024-01-10 10:00:10 - Open Positions: 1
2024-01-10 10:00:10 - Win Rate: 0.0% (0/0)

--- Iteration 2 ---
2024-01-10 10:00:20 - Taking profit on SPY_call_0: 21.5%
2024-01-10 10:00:20 - Closed SPY_call_0 - P&L: $325.00 (21.5%)
2024-01-10 10:00:20 - QQQ: Bought 3 put contracts (confidence: 62.1%)
...
```

## Understanding the Output

### Per Iteration:
- **Action Taken**: What the model decided (buy/sell/hold)
- **Confidence**: Model's confidence in the action
- **Portfolio Value**: Current total value
- **Daily P&L**: Realized + unrealized profit/loss
- **Open Positions**: Number of active positions
- **Win Rate**: Percentage of profitable trades

### Position Management:
- **Taking profit**: Automatically closes at 20% gain
- **Stop loss**: Automatically closes at 10% loss
- **Expiring**: Closes positions near expiration

## Trade History

After the session, find your trades in:
```
paper_trades_YYYYMMDD_HHMMSS.json
```

Example entry:
```json
{
  "symbol": "SPY",
  "type": "call",
  "quantity": 5,
  "entry_price": 2.85,
  "exit_price": 3.46,
  "pnl": 305.0,
  "pnl_percent": 0.214,
  "holding_time": 0.5
}
```

## Tips for Better Results

1. **Run Longer Sessions**: 2-4 hours gives more realistic results
2. **Test Different Models**: Compare checkpoints
3. **Vary Market Conditions**: The simulator adds realistic volatility
4. **Monitor Win Rate**: Should improve with closing incentives

## Comparing Models

```bash
# Test your best model
python paper_trading_bot.py --model checkpoints/ppo_lstm/best_model.pt --duration 120

# Test a specific checkpoint
python paper_trading_bot.py --model checkpoints/ppo_lstm/ep1000.pt --duration 120

# Compare results in the JSON files
```

## Limitations

1. **Simulated Data**: Not real market prices
2. **No Slippage**: Perfect fills at bid/ask
3. **Limited Strategies**: Only simple options (no spreads yet)
4. **No Greeks Updates**: Greeks are static

## Next Steps

1. Run paper trading to validate your model
2. Compare different trained models
3. Adjust risk parameters if needed
4. Move to live paper trading with Alpaca when ready

## Troubleshooting

### "Model not found"
```bash
ls checkpoints/ppo_lstm/  # Check available models
```

### "Out of memory"
- Use CPU instead: Add `--device cpu` support
- Reduce batch operations

### "No trades executed"
- Model confidence might be too low
- Try different symbols
- Check if model was trained properly