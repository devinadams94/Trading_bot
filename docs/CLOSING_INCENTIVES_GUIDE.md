# Position Closing Incentives Guide

## Problem
Your model is achieving positive returns but most are **unrealized** because positions are held indefinitely. The model needs incentives to close profitable positions.

## Solution: Paper-Compliant Closing Incentives

We've implemented closing incentives that comply with the research paper (no holding time penalties):

### 1. Profit Realization Bonus
- **20% bonus** for closing profitable positions
- Encourages locking in gains
- Scaled by 1e-4 as per paper

### 2. Automatic Profit Taking
- Positions automatically close at **10% profit**
- Prevents gains from evaporating
- Based on market behavior, not penalties

### 3. Stop Loss Protection
- Positions automatically close at **5% loss**
- Limits downside risk
- Standard risk management practice

### 4. Trailing Stop
- **3% trailing stop** from peak value
- Protects profits while allowing upside
- Closes positions if they drop 3% from their highest point

### 5. Turbulence-Based Trading
- Implements paper's turbulence threshold
- Stops trading during extreme market conditions
- 90th percentile threshold as specified

## Usage

### Basic Training with Closing Incentives
```bash
python train_ppo_lstm.py \
    --closing-incentives \
    --fix-zero-trading \
    --episodes 1000
```

### Full Configuration
```bash
python train_ppo_lstm.py \
    --closing-incentives \
    --fix-zero-trading \
    --entropy-coef 0.02 \
    --episodes 1000 \
    --symbols SPY QQQ IWM \
    --checkpoint-interval 50
```

### With Historical Data
```bash
python train_ppo_lstm.py \
    --closing-incentives \
    --fix-zero-trading \
    --symbols AAPL MSFT GOOGL \
    --start-date 2024-01-01 \
    --episodes 1000
```

## What This Fixes

### Before:
- Episode 883: Return: $4,063 (**unrealized** from 9 open positions)
- Win Rate: 0% (0 closed trades)
- Model hoards positions indefinitely

### After (Expected):
- More **realized** profits
- Positive win rate (positions actually close)
- Better risk management
- Consistent profit taking

## Key Differences from Original

| Feature | Without Incentives | With Incentives |
|---------|-------------------|-----------------|
| Profit Taking | Manual/Never | Automatic at 10% |
| Stop Loss | None | Automatic at 5% |
| Trailing Stop | None | 3% from peak |
| Realization Bonus | None | 20% of profits |
| Win Rate | Often 0% | Should improve |

## Implementation Details

The closing incentives are implemented in `fix_position_closing_paper_compliant.py`:

1. **PaperCompliantClosingIncentives**: Core logic for profit/loss targets
2. **SmartPositionManagement**: Trailing stops and position tracking
3. **TurbulenceBasedTradingEnvironment**: Market condition monitoring

## Monitoring Progress

Watch for these improvements:
```
Episode X - Return: $Y (realized), Win Rate: Z% (N closed trades)
```

Key metrics to track:
- **Realized vs Unrealized**: Should see more realized
- **Win Rate**: Should be > 0%
- **Closed Trades**: Should increase
- **Average Holding Time**: Should decrease

## Advanced Options

### Adjust Thresholds
Modify in `fix_position_closing_paper_compliant.py`:
```python
self.profit_target = 0.10  # Change to 0.15 for 15%
self.stop_loss = 0.05     # Change to 0.03 for 3%
self.trailing_stop_pct = 0.03  # Change to 0.05 for 5%
```

### Combine with Other Features
```bash
python train_ppo_lstm.py \
    --closing-incentives \
    --fix-zero-trading \
    --use-qlib \
    --distributed \
    --episodes 5000
```

## Troubleshooting

### Still Not Closing Positions?
1. Increase profit realization bonus
2. Lower profit target threshold
3. Increase entropy coefficient

### Closing Too Early?
1. Increase profit target (e.g., 15%)
2. Reduce realization bonus
3. Widen trailing stop

### Performance Dropped?
- This is normal initially as the model learns new behavior
- Give it 100-200 episodes to adapt
- The long-term performance should improve

## Next Steps

1. Train with closing incentives for at least 1000 episodes
2. Compare realized vs unrealized profits
3. Adjust thresholds based on results
4. Consider market-specific parameters