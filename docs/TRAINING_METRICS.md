# Training Metrics Guide

## Metrics Displayed During Training

The updated `train_ppo_lstm.py` script now displays comprehensive metrics for each episode:

### Per-Episode Metrics

```
Episode 1/1000 - Reward: -403.38, Return: $-94452.90, Win Rate: 0.0% (0 trades), Steps: 1
```

- **Reward**: The cumulative RL reward signal used for training
- **Return**: Actual dollar P&L (final portfolio value - initial capital)
- **Win Rate**: Percentage of trades that were profitable
- **Trades**: Number of trades executed during the episode
- **Steps**: Number of time steps in the episode

### Rolling Averages (Every 10 Episodes)

```
→ 100-Episode Averages - Return: $-2543.21, Win Rate: 45.3%
```

- Shows 100-episode moving averages for returns and win rates
- Helps identify long-term trends in performance

## Understanding the Metrics

### Reward vs Return
- **Reward** is the training signal (can be negative/positive based on actions)
- **Return** is the actual money made/lost in dollars

### Win Rate Components
- Calculated as: `winning_trades / total_trades`
- A trade is counted when a position is closed
- 0% win rate with 0 trades means no positions were closed

### Common Patterns

1. **Early Training**: Often shows 0% win rate with few/no trades as the model explores
2. **Mid Training**: Win rate may fluctuate as the model learns different strategies
3. **Late Training**: Should see stabilization of both win rate and returns

## Monitoring Training Progress

### Good Signs
- Increasing average returns over time
- Win rate stabilizing above 50%
- Consistent trade execution (not 0 trades)
- Decreasing variance in returns

### Warning Signs
- Consistently 0 trades (model not learning to trade)
- Win rate stuck at 0% or 100% (overfitting)
- Returns becoming more negative over time
- High variance that doesn't decrease

## Example Training Output

```
INFO:__main__:Episode 100/1000 - Reward: 125.43, Return: $1234.56, Win Rate: 62.5% (8 trades), Steps: 245
INFO:__main__:  → 100-Episode Averages - Return: $543.21, Win Rate: 58.3%
INFO:__main__:Updated networks - Critic Loss: 0.0234, Actor Loss: 0.0012
INFO:__main__:Saved checkpoint to checkpoints/ppo_lstm_ep100.pt
```

This shows:
- Positive returns ($1234.56 profit)
- Good win rate (62.5%)
- Active trading (8 trades)
- Network learning (loss updates)
- Checkpoint saved for later use