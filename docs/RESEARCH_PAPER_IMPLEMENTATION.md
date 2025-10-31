# Research Paper Implementation Guide

## Paper: "A Novel Deep Reinforcement Learning Based Automated Stock Trading System Using Cascaded LSTM Networks"

### Reward Structure Implementation

The paper specifies a precise reward formula in Section 3.1.3:

```
Return_t(s_t, a_t, s_{t+1}) = (b_{t+1} + p^T_{t+1} * h_{t+1}) - (b_t + p^T_t * h_t) - c_t
```

Where:
- `b_t`: Available balance at time t
- `p_t`: Price vector at time t  
- `h_t`: Holdings vector at time t
- `c_t`: Transaction cost (0.1% of trade value)

### Key Parameters from the Paper

From Section 3.1.5:
- **Initial capital**: $1,000,000
- **Max shares per trade**: 100 (h_max)
- **Transaction cost**: 0.1% per trade
- **Reward scaling factor**: 1e-4
- **Turbulence threshold**: 90th percentile of historical turbulence

### Current Implementation

The reward structure has been updated in `src/options_trading_env.py` to match the paper:

```python
# Calculate raw return (portfolio value change)
raw_return = portfolio_value - self.last_portfolio_value

# Apply 0.1% transaction cost
transaction_cost = 0.001 * abs(trade_value)

# Paper's reward formula
reward = raw_return - transaction_cost

# Apply scaling factor (1e-4)
scaled_reward = reward * 1e-4
```

### Running with Paper's Configuration

#### Basic Training
```bash
python train_ppo_lstm.py --episodes 1000
```

#### With Zero-Trading Fix + Paper Rewards
```bash
python train_ppo_lstm.py \
    --fix-zero-trading \
    --entropy-coef 0.01 \
    --episodes 1000
```

#### Full Paper Implementation (Future)
```bash
python train_ppo_lstm.py \
    --use-paper-reward \
    --initial-capital 1000000 \
    --max-shares 100 \
    --episodes 1000
```

### Differences from Original Implementation

| Feature | Original | Paper | Impact |
|---------|----------|-------|---------|
| Reward Scale | /1000 | *1e-4 | More stable gradients |
| Transaction Cost | 0.065% | 0.1% | Slightly higher costs |
| Initial Capital | $100,000 | $1,000,000 | Larger position sizes |
| Bonus/Penalties | Many | None | Simpler, cleaner signal |

### Additional Paper Suggestions (Section 5)

The paper suggests future improvements:

1. **Sharpe Ratio Reward**: Balance risk and return
   ```python
   sharpe = (mean_return - risk_free_rate) / std_return * sqrt(252)
   ```

2. **Reward Normalization**: Clip to [-1, 1] for stability
   ```python
   normalized_reward = tanh(reward / 2)
   ```

3. **More Training Data**: Expand historical data range

### State Space (Section 3.1.1)

The paper uses:
1. Balance `b_t`
2. Prices `p_t` (adjusted close)
3. Holdings `h_t`
4. Technical indicators:
   - MACD
   - RSI
   - CCI
   - ADX

### Action Space (Section 3.1.2)

- Discrete actions: {-k, ..., -1, 0, 1, ..., k}
- k = max shares to buy/sell
- Normalized to [-1, 1]
- High-dimensional: (2k+1)^30 for 30 stocks

### LSTM Feature Extraction

The paper's key innovation is using LSTM for feature extraction:
1. Takes past T states: F_t = [S_{t-T+1}, ..., S_t]
2. Extracts hidden patterns
3. Outputs encoded feature vector F'_t
4. Feeds to PPO agent

### Training Results from Paper

- **US Market (DJI)**: +10% cumulative return vs baseline
- **Chinese Market**: +84.4% cumulative return improvement
- **Sharpe Ratio**: +37.4% improvement
- **Win Rate**: Improved consistency

### Recommended Training Approach

1. **Start Simple**: Use current implementation with fixes
2. **Add Paper Elements Gradually**:
   - First: Transaction cost adjustment
   - Second: Reward scaling factor
   - Third: Full state/action space
3. **Monitor Carefully**: Paper's reward is very small (1e-4 scale)

### Troubleshooting

#### If Returns Stay at Zero
- The paper's 1e-4 scaling makes rewards tiny
- Consider starting with 1e-3 or 1e-2 
- Add small exploration bonus

#### If Training is Unstable
- Implement reward normalization
- Reduce learning rate
- Increase batch size

### Code to Match Paper Exactly

```python
# In options_trading_env.py
class PaperCompliantEnvironment(OptionsTradingEnvironment):
    def __init__(self):
        super().__init__(
            initial_capital=1000000,  # $1M as per paper
            commission=0.65,  # Assuming per-contract commission
            max_positions=10
        )
        self.reward_scaling_factor = 1e-4
        self.transaction_cost_rate = 0.001  # 0.1%
        self.max_shares_per_trade = 100
```

### Next Steps

1. Test current implementation with zero-trading fix
2. Gradually adjust parameters to match paper
3. Implement turbulence threshold for risk management
4. Add Sharpe ratio reward shaping if needed