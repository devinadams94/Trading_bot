# Risk-Based Reward System

## Overview

The trading bot now implements a sophisticated risk-based reward system that teaches the model to balance risk and reward effectively. The system provides larger rewards for successful risky trades while imposing harsher penalties for failed risky strategies.

## Key Components

### 1. Position Risk Calculation (0-1 scale)

The system calculates a comprehensive risk score for each position based on:

- **Size Risk (30%)**: Larger positions relative to capital are riskier
- **Time Risk (20%)**: Longer holding periods increase risk due to theta decay
- **Moneyness Risk (30%)**: OTM options are riskier than ITM options
- **P&L Risk (20%)**: Losing positions are riskier to hold

```python
position_risk = _calculate_position_risk()  # Returns 0.0 (low) to 1.0 (high)
```

### 2. Risk-Adjusted Rewards

#### Profitable Trades
- **High Risk (>0.7)**: 2x base reward for successful risky trades
- **Medium Risk (0.5-0.7)**: 1.5x base reward
- **Low Risk (<0.5)**: Normal base reward

#### Losing Trades
- **High Risk (>0.7)**: 2.5x penalty for failed risky trades
- **Medium Risk (0.5-0.7)**: 1.5x penalty
- **Low Risk (<0.5)**: Normal penalty

### 3. Sharpe Ratio Component

The system rewards risk-adjusted returns by calculating a rolling Sharpe ratio:
```python
sharpe = mean(returns) / std(returns) * sqrt(252)
reward += sharpe * 0.5
```

### 4. Win Rate with Risk Consideration

Win rate bonuses are adjusted based on average position risk:

| Win Rate | Low Risk | High Risk |
|----------|----------|-----------|
| >70%     | +15      | +25       |
| >60%     | +8       | +15       |
| >50%     | +4       | +4        |
| <30%     | -5       | -15       |

### 5. Drawdown Management

Progressive penalties based on maximum drawdown:
- **>20% drawdown**: -10 × drawdown percentage
- **>10% drawdown**: -5 × drawdown percentage
- **Action filtering**: Prevents new risky positions during severe drawdowns

### 6. Risk-Based Exit Rules

Dynamic stop-loss and take-profit levels:

| Risk Level | Stop Loss | Take Profit |
|------------|-----------|-------------|
| High (>0.7)| -15%      | +25%        |
| Normal     | -20%      | +30%        |

### 7. Market Regime Adaptation

Different risk tolerances for different market conditions:
- **Volatile markets**: Penalties for large position sizes
- **Trending markets**: Rewards for holding positions
- **Ranging markets**: Neutral risk assessment

### 8. Leverage Control

Progressive penalties for excessive leverage:
- **>5x leverage**: -(leverage - 5) × 2
- **>3x leverage**: -(leverage - 3) × 0.5

## Risk Metrics

### Position Risk Factors

1. **Size Risk**: `position_value / (capital × 0.2)`
2. **Time Risk**: `holding_time / 30`
3. **Moneyness Risk**: 
   - OTM: 0.8
   - ATM: 0.5
   - ITM: 0.3
4. **P&L Risk**:
   - Losing >2%: 0.8
   - Winning >5%: 0.4
   - Neutral: 0.6

### Portfolio Risk

```python
total_risk = (avg_position_risk × 0.5 + 
              concentration_risk × 0.3 + 
              volatility_risk × 0.2)
```

## Learning Objectives

The risk-based reward system teaches the model to:

1. **Take calculated risks**: Larger rewards for successful risky trades encourage controlled risk-taking
2. **Manage drawdowns**: Penalties and action filtering during drawdowns teach capital preservation
3. **Size positions appropriately**: Risk-based position sizing prevents oversized bets
4. **Exit dynamically**: Tighter stops for risky positions teach adaptive risk management
5. **Consider market conditions**: Different rewards for different market regimes
6. **Balance portfolio risk**: Penalties for concentration and excessive leverage

## Implementation Details

### Entry Quality Assessment

Good entry conditions are rewarded:
- **Calls**: Positive momentum + RSI 40-65
- **Puts**: Negative momentum + RSI 35-60

Risk-adjusted entry bonuses:
- **High risk + immediate profit**: +5 × entry_quality
- **Low risk + immediate profit**: +2 × entry_quality
- **High volatility + loss**: -2 penalty

### Consecutive Loss Management

Enhanced penalties during losing streaks:
- **>5 losses + high risk**: losses × 5 penalty
- **>5 losses + normal risk**: losses × 3 penalty
- **>3 losses + high risk**: losses × 3 penalty
- **>3 losses + normal risk**: losses × 2 penalty

### Risk Reduction Incentives

- **Closing positions during drawdown**: +3 bonus
- **Maintaining low risk with wins**: +8 bonus
- **Attempting new positions at 20%+ drawdown**: -1 penalty

## Configuration

Key parameters that can be adjusted:

```python
# Risk thresholds
HIGH_RISK_THRESHOLD = 0.7
MEDIUM_RISK_THRESHOLD = 0.5

# Reward multipliers
HIGH_RISK_WIN_MULTIPLIER = 2.0
HIGH_RISK_LOSS_MULTIPLIER = 2.5

# Drawdown limits
DRAWDOWN_ACTION_FILTER = 0.15
SEVERE_DRAWDOWN_LIMIT = 0.20

# Position limits
MAX_CAPITAL_PER_POSITION = 0.20
MAX_LEVERAGE = 10.0
```

## Monitoring Risk Metrics

The system tracks:
- Position risk scores over time
- Maximum drawdown
- Average position risk
- Sharpe ratio (rolling 50 periods)
- Win rate by risk level
- Risk-adjusted returns

## Expected Outcomes

With this risk-based reward system, the model should learn to:

1. Take larger positions when confidence is high
2. Reduce risk during drawdowns
3. Exit losing positions faster when risk is high
4. Hold winning positions longer when risk is low
5. Avoid new positions during severe drawdowns
6. Balance position sizes based on market volatility
7. Achieve better risk-adjusted returns (higher Sharpe ratio)

The ultimate goal is to create a model that not only maximizes returns but does so with appropriate risk management, leading to more consistent and sustainable trading performance.