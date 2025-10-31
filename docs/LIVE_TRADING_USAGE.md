# Live Trading with PPO-LSTM Model

## Overview
This script enables live/paper trading using your trained PPO-LSTM options trading model with Alpaca's API.

## Prerequisites

1. **Alpaca Account**: Set up an account at https://alpaca.markets
2. **API Keys**: Get your API keys and add to `.env`:
   ```
   ALPACA_API_KEY=your_api_key_here
   ALPACA_SECRET_KEY=your_secret_key_here
   ```
3. **Trained Model**: Have a trained model in `checkpoints/ppo_lstm/`

## Basic Usage

### Paper Trading (Recommended for Testing)
```bash
# Use the best model for paper trading
python live_trading_ppo_lstm.py --model checkpoints/ppo_lstm/best_model.pt --paper

# Use a specific checkpoint
python live_trading_ppo_lstm.py --model checkpoints/ppo_lstm/ep1000.pt --paper

# Trade specific symbols
python live_trading_ppo_lstm.py --model checkpoints/ppo_lstm/best_model.pt --symbols AAPL MSFT GOOGL --paper
```

### Live Trading (Use with Caution!)
```bash
# Remove --paper flag for live trading (BE VERY CAREFUL!)
python live_trading_ppo_lstm.py --model checkpoints/ppo_lstm/best_model.pt --max-capital-per-trade 1000
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | Required | Path to model checkpoint |
| `--symbols` | SPY QQQ IWM | Symbols to trade |
| `--max-positions` | 5 | Maximum simultaneous positions |
| `--max-capital-per-trade` | 10000 | Max $ per trade |
| `--stop-loss` | 0.10 | Stop loss (10%) |
| `--take-profit` | 0.20 | Take profit (20%) |
| `--paper` | True | Use paper trading |
| `--duration` | None | Run duration in minutes |

## Examples

### Conservative Paper Trading
```bash
python live_trading_ppo_lstm.py \
    --model checkpoints/ppo_lstm/best_model.pt \
    --symbols SPY \
    --max-positions 2 \
    --max-capital-per-trade 5000 \
    --stop-loss 0.05 \
    --paper
```

### Aggressive Paper Trading
```bash
python live_trading_ppo_lstm.py \
    --model checkpoints/ppo_lstm/best_model.pt \
    --symbols SPY QQQ IWM AAPL MSFT \
    --max-positions 10 \
    --max-capital-per-trade 20000 \
    --stop-loss 0.15 \
    --take-profit 0.30 \
    --paper
```

### Time-Limited Session
```bash
# Run for 2 hours (120 minutes)
python live_trading_ppo_lstm.py \
    --model checkpoints/ppo_lstm/best_model.pt \
    --duration 120 \
    --paper
```

## Risk Management Features

1. **Position Limits**: Maximum number of concurrent positions
2. **Capital Limits**: Maximum capital per trade
3. **Stop Loss**: Automatic position closing at loss threshold
4. **Take Profit**: Automatic profit taking
5. **Daily Loss Limit**: Stops trading if daily loss exceeds $5000
6. **Confidence Threshold**: Only trades when model confidence > 30%

## Monitoring

The script logs to both console and `live_trading.log`:
- Real-time position updates
- P&L tracking
- All trade executions
- Risk limit triggers

## Important Notes

1. **Start with Paper Trading**: Always test with paper trading first!
2. **Monitor Closely**: Even with automation, monitor the bot closely
3. **Market Hours**: Bot only trades during market hours (9 AM - 4 PM ET)
4. **Options Specifics**: Alpaca's options API may have limitations
5. **Slippage**: Real trading has slippage not present in training

## Troubleshooting

### "Model not found"
- Check the model path: `ls checkpoints/ppo_lstm/`
- Use absolute path if needed

### "API Key Error"
- Verify `.env` file has correct keys
- Check if using paper vs live keys correctly

### "No trades being placed"
- Check if model confidence is too low
- Verify market hours
- Check account balance

### "Position not closing"
- Verify stop loss/take profit settings
- Check if options data is updating

## Safety Checklist

Before going live:
- [ ] Tested extensively with paper trading
- [ ] Verified all risk limits are set appropriately  
- [ ] Confirmed model performance matches expectations
- [ ] Set up monitoring/alerts
- [ ] Started with small position sizes
- [ ] Have manual override plan ready

## Performance Metrics

Track these metrics:
- Win rate
- Average P&L per trade
- Maximum drawdown
- Sharpe ratio
- Number of trades per day

## Next Steps

1. Run paper trading for at least 1 week
2. Analyze performance vs backtest
3. Adjust parameters based on results
4. Consider implementing additional strategies
5. Gradually increase position sizes if profitable