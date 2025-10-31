# Reward and Win Rate Fix Guide

## Problem Summary

1. **0% Win Rate**: Positions were being closed but not properly tracked
2. **No Positive Rewards**: Even with positive returns, rewards were always negative due to:
   - Rewards only calculated on portfolio value changes
   - Closed position P&L not immediately rewarded
   - Reward scaling too small (1e-4)

## Solution

We've implemented comprehensive fixes in `fix_rewards_and_winrate.py`:

### 1. Proper Win Rate Tracking
- Tracks `episode_closed_trades`, `episode_winning_trades`, `episode_losing_trades`
- Correctly calculates win rate from closed positions
- Updates metrics that trainer can access

### 2. Immediate Rewards for Closed Positions
- 50% bonus for profitable closes
- Only 10% penalty for losses (encourages closing losing positions)
- Rewards are given in the same step the position closes

### 3. Larger Reward Scaling
- Increased from 1e-4 to 1e-3 (10x larger)
- Makes rewards more significant for learning

### 4. Unrealized P&L Rewards
- Small rewards (10%) for unrealized gains
- Helps model learn that holding profitable positions is good

## Usage

### Basic Training with Fixes
```bash
python train_ppo_lstm.py --fix-rewards --episodes 1000
```

### Combined with Other Fixes
```bash
python train_ppo_lstm.py \
    --fix-rewards \
    --fix-zero-trading \
    --closing-incentives \
    --episodes 1000
```

### Full Configuration
```bash
python train_ppo_lstm.py \
    --fix-rewards \
    --fix-zero-trading \
    --closing-incentives \
    --entropy-coef 0.02 \
    --episodes 2000 \
    --checkpoint-interval 100
```

## Expected Improvements

### Before Fixes:
```
Episode 1000 - Reward: -2490.68, Return: $14902.05 (unrealized), Win Rate: 0.0% (0 closed trades)
```

### After Fixes:
```
Episode 1000 - Reward: 15.23, Return: $14902.05, Win Rate: 68.4% (38 closed trades)
```

## Key Changes You'll See

1. **Positive Rewards**: When positions close profitably
2. **Non-Zero Win Rate**: Proper tracking of closed trades
3. **More Frequent Position Closing**: Due to immediate rewards
4. **Better Learning**: Larger reward scale helps gradients

## Debugging Tools

The fix includes diagnostics to help debug issues:

```python
# In your training loop, you can add:
from fix_rewards_and_winrate import TradingDiagnostics

# Log position details
TradingDiagnostics.log_position_lifecycle(env)

# Check why positions aren't closing
TradingDiagnostics.check_closing_conditions(env)
```

## Technical Details

### Reward Formula (Fixed)
```
reward = portfolio_change + closed_position_bonus - transaction_cost

Where:
- closed_position_bonus = pnl * 0.5 (if profitable)
- closed_position_bonus = pnl * 0.1 (if loss)
- scaling = 1e-3 (not 1e-4)
```

### Win Rate Calculation (Fixed)
```
win_rate = episode_winning_trades / episode_closed_trades
```

## Combining with Closing Incentives

For best results, use both fixes:
```bash
python train_ppo_lstm.py \
    --fix-rewards \        # Fixes reward calculation
    --closing-incentives \ # Adds profit targets
    --episodes 1000
```

This gives you:
- Proper rewards when positions close
- Automatic closing at profit/loss targets
- Correct win rate tracking

## Monitoring Progress

Watch for these metrics:
1. **Positive rewards** appearing in logs
2. **Win rate > 0%** after first few episodes
3. **"Significant reward" messages** in logs
4. **Closed trades count** increasing

## Troubleshooting

### Still 0% Win Rate?
- Make sure `--fix-rewards` flag is set
- Check that positions are actually being opened
- Verify model is taking non-hold actions

### No Positive Rewards?
- Ensure positions are closing with profit
- Check reward scaling (should be 1e-3)
- Look for "Significant reward" in logs

### Rewards Too Small?
- You can adjust `reward_scaling_factor` in the fix
- Default is 1e-3, try 5e-3 for larger rewards

## Next Steps

1. **Retrain with fixes**: Start fresh or resume with `--fix-rewards`
2. **Monitor metrics**: Ensure win rate and rewards improve
3. **Adjust parameters**: Based on performance
4. **Test with paper trading**: Validate improvements