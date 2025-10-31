# Best Model Selection Fix

## Problem
The best model selection was broken:
- A model with $1.63 average return was selected as "best"
- While episodes with $4,063+ returns were ignored
- Models with 0% win rate (never closing trades) were preferred

## Root Cause
1. The selection was based on cumulative step rewards, not actual portfolio returns
2. With paper's reward formula, opening positions gives negative rewards
3. Models that just accumulate positions without closing looked "better"

## Solution Implemented

### 1. New Scoring Formula
```python
if avg_win_rate == 0 and avg_closed_trades == 0:
    # Model is just accumulating positions, not trading
    combined_score = avg_return * 0.1  # Heavily discounted
else:
    # Normal scoring for models that actually trade
    combined_score = avg_return * (1 + avg_win_rate)  # Returns amplified by win rate
```

### 2. Key Changes
- **Tracks closed trades**: Added `episode_closed_trades` list
- **Penalizes position accumulation**: Models that never close get 90% score penalty
- **Rewards actual trading**: Score = returns × (1 + win_rate)
- **Uses portfolio returns**: Not cumulative rewards

### 3. Example Scoring

#### Before Fix:
- Model A: $1.63 avg return, 0% win rate → Score: 1.14
- Model B: $40.63 avg return (single episode), 0% win rate → Score: 0.41

#### After Fix:
- Model A: $1.63 avg return, 0% win rate, 0 trades → Score: 0.163 (penalized)
- Model B: $40.63 avg return, 20% win rate, 5 trades → Score: 48.76 (rewarded)

## Impact
- Best model now reflects actual trading performance
- Models are incentivized to close profitable positions
- No more rewarding position hoarders

## Testing
To verify the fix works:
```bash
# Check the logs for new best model selections
grep "🏆 New best model" your_training.log

# Should see:
# - Higher returns being selected
# - Models with positive win rates preferred
# - Score value included in the log
```