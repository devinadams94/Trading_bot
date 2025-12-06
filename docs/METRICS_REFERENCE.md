# Metrics Reference Card

**Quick reference for interpreting training and evaluation metrics**

---

## Training Metrics (Logged Every Iteration)

### Reward Metrics
```
Reward: 0.234 ± 1.456
```
| Metric | Good Range | Warning Signs |
|--------|------------|---------------|
| **Mean Reward** | Increasing over time | Stuck at negative values |
| **Std Reward** | Decreasing over time | Very high (>5.0) |

**What it means**: Average reward across all parallel environments. Should increase as policy improves.

---

### Return Metrics
```
Return: 0.023% | Sharpe: 0.12
```
| Metric | Good | Acceptable | Warning |
|--------|------|------------|---------|
| **Return %** | > 0.5% | 0-0.5% | < 0% |
| **Sharpe Ratio** | > 2.0 | 1.0-2.0 | < 0.5 |

**What it means**: 
- **Return**: Average portfolio return per episode
- **Sharpe**: Risk-adjusted return (higher is better)

---

### PPO Loss Metrics
```
Policy Loss: 0.0234 | Value Loss: 0.1234 | Entropy: 2.45
```

| Metric | Good Range | Warning Signs |
|--------|------------|---------------|
| **Policy Loss** | Decreasing, 0.01-0.5 | Stuck at high value, or drops to 0 too fast |
| **Value Loss** | Decreasing, 0.01-1.0 | Increasing over time |
| **Entropy** | 2.7 → 0.5 (decaying) | Drops to 0 in first 100 iterations |

**What it means**:
- **Policy Loss**: How much the policy is changing (should decrease)
- **Value Loss**: How well critic predicts returns (should decrease)
- **Entropy**: Exploration measure (should decay smoothly)

---

### PPO Diagnostic Metrics
```
KL Div: 0.0012 | Clip Frac: 0.15 | Explained Var: 0.45
```

| Metric | Good Range | Warning Signs |
|--------|------------|---------------|
| **KL Divergence** | < 0.01 | > 0.05 (policy changing too fast) |
| **Clip Fraction** | 0.1-0.3 | 0 (updates too small) or >0.5 (too large) |
| **Explained Variance** | > 0.5 (increasing) | < 0 (critic worse than mean) |

**What it means**:
- **KL Div**: How much policy changed this update
- **Clip Frac**: % of updates that were clipped by PPO
- **Explained Var**: How well value function predicts returns

---

### Optimization Metrics
```
LR: 3.00e-4 | Grad Norm: 0.23
```

| Metric | Good Range | Warning Signs |
|--------|------------|---------------|
| **Learning Rate** | Decaying smoothly | Stuck at initial value |
| **Grad Norm** | 0.1-1.0 | > 5.0 (exploding) or < 0.01 (vanishing) |

**What it means**:
- **LR**: Current learning rate (decays with cosine schedule)
- **Grad Norm**: Magnitude of gradients (should be stable)

---

### Performance Metrics
```
SPS: 12,543
```

| Hardware | Expected SPS |
|----------|--------------|
| **CPU** | 1,000-5,000 |
| **GPU (RTX 3090)** | 10,000-20,000 |
| **GPU (H100/H200)** | 30,000-50,000 |

**What it means**: Steps per second (training throughput)

---

## Evaluation Metrics (After Training)

### Primary Metrics

| Metric | Formula | Good | Acceptable | Warning |
|--------|---------|------|------------|---------|
| **Sharpe Ratio** | sqrt(252) × mean(returns) / std(returns) | > 2.0 | 1.0-2.0 | < 0.5 |
| **Total Return %** | (final - initial) / initial × 100 | > 1.0% | 0-1.0% | < 0% |
| **Max Drawdown %** | max(peak - current) / peak × 100 | < 10% | 10-20% | > 20% |
| **Win Rate %** | profitable_episodes / total_episodes × 100 | > 55% | 50-55% | < 50% |
| **Turnover %** | avg_daily_portfolio_change × 100 | < 5% | 5-20% | > 50% |

---

### Sharpe Ratio Interpretation

| Sharpe | Interpretation | Example |
|--------|----------------|---------|
| **< 0** | Losing money | Worse than cash |
| **0-1** | Weak performance | Barely beating risk-free rate |
| **1-2** | Good performance | Solid risk-adjusted returns |
| **2-3** | Very good | Professional-grade performance |
| **> 3** | Excellent | Top-tier hedge fund level |

**Current System**: Achieves **2.87** on 2020-2024 test data (very good)

---

### Drawdown Interpretation

| Max Drawdown | Interpretation | Risk Level |
|--------------|----------------|------------|
| **< 5%** | Very low risk | Conservative |
| **5-10%** | Low risk | Moderate |
| **10-20%** | Moderate risk | Acceptable |
| **20-30%** | High risk | Aggressive |
| **> 30%** | Very high risk | Dangerous |

**Current System**: Achieves **-8.1%** on test data (low risk)

---

### Turnover Interpretation

| Turnover | Interpretation | Trading Style |
|----------|----------------|---------------|
| **< 1%** | Very low | Buy and hold |
| **1-5%** | Low | Infrequent rebalancing |
| **5-20%** | Moderate | Regular rebalancing |
| **20-50%** | High | Active trading |
| **> 50%** | Very high | Day trading / thrashing |

**Current System**: Achieves **0.3%** on test data (very low - learned to hold)

---

## Training Progress Patterns

### Early Training (Iterations 1-100)
```
✅ Reward: -0.5 → 0.0 (improving)
✅ Entropy: 2.7 → 2.5 (exploring)
✅ Policy Loss: 0.5 → 0.2 (learning)
✅ Value Loss: 1.0 → 0.5 (improving)
✅ Explained Var: 0.0 → 0.3 (critic learning)
```

### Mid Training (Iterations 100-1000)
```
✅ Reward: 0.0 → 0.5 (positive returns)
✅ Sharpe: 0.0 → 1.0 (risk-adjusted gains)
✅ Entropy: 2.5 → 1.5 (less exploration)
✅ Policy Loss: 0.2 → 0.1 (stabilizing)
✅ Explained Var: 0.3 → 0.7 (good predictions)
```

### Late Training (Iterations 1000+)
```
✅ Reward: 0.5 → 0.8 (converging)
✅ Sharpe: 1.0 → 2.0 (strong performance)
✅ Entropy: 1.5 → 0.5 (exploiting)
✅ Policy Loss: 0.1 → 0.05 (stable)
✅ Explained Var: 0.7 → 0.9 (excellent)
```

---

## Warning Signs

### 🚨 Policy Collapse
```
❌ Entropy drops to 0 in first 100 iterations
❌ All actions become the same (e.g., always HOLD)
❌ Clip fraction = 0
```
**Fix**: Increase `--entropy-coef`, reduce `--lr`

### 🚨 Unstable Training
```
❌ Reward oscillates wildly
❌ KL divergence > 0.1
❌ Grad norm > 5.0
```
**Fix**: Reduce `--lr`, increase `--max-grad-norm`

### 🚨 No Learning
```
❌ Losses don't decrease after 500 iterations
❌ Explained variance stays near 0
❌ Sharpe stays negative
```
**Fix**: Check data quality, increase `--lr`, check reward function

### 🚨 Overfitting
```
❌ Training reward increases but test reward decreases
❌ Entropy drops to 0 too early
❌ Policy becomes deterministic
```
**Fix**: Increase `--entropy-coef`, add regularization, use more data

---

## Success Criteria

### Minimum (Regression Checks)
```yaml
✅ RL Sharpe >= SPY Sharpe - 0.5
✅ RL Sharpe >= 0.0 (not losing money)
✅ Turnover <= 50% (not thrashing)
```

### Good Performance
```yaml
✅ RL Sharpe > 1.0
✅ RL beats SPY on Sharpe
✅ Max drawdown < 15%
✅ Win rate > 52%
✅ Turnover < 10%
```

### Excellent Performance
```yaml
✅ RL Sharpe > 2.0
✅ RL ranks in top 3 vs baselines
✅ Max drawdown < 10%
✅ Win rate > 55%
✅ Turnover < 5%
```

---

**For detailed explanations, see `TRAINING_GUIDE.md`**

