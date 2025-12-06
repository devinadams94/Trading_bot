# Architecture Analysis and Improvements

## Current Architecture: GRU-PPO (Not CLSTM-PPO)

### ⚠️ Architecture Clarification

The current implementation is **GRU-PPO**, not CLSTM-PPO:

| Component | Current (GRU-PPO) | True CLSTM-PPO |
|-----------|-------------------|----------------|
| Recurrent Unit | GRU (1 layer) | LSTM (2+ layers) |
| Hidden State | Single tensor `h` | Tuple `(h, c)` |
| Gating | 2 gates (reset, update) | 3 gates (input, forget, output) |
| Cell State | None | Explicit `c` state |
| Parameters | ~110K | ~3.8M (original v1) |

### Why GRU Instead of LSTM?

1. **Faster Training**: GRU has fewer parameters (2 gates vs 3)
2. **Simpler Hidden State**: Single tensor vs tuple
3. **Sufficient for Task**: 16 discrete actions don't need LSTM's complexity
4. **Stability**: Fewer parameters = easier optimization

---

## Potential Improvements

### 1. 🔄 Upgrade to True CLSTM (Convolutional LSTM)

**What is CLSTM?**
CLSTM (Convolutional LSTM) combines CNN and LSTM for spatiotemporal learning. For time series, it processes sequences with learned convolutional kernels.

**Implementation:**
```python
class CLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)
        self.hidden_dim = hidden_dim
    
    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new
```

**Expected Benefit**: Better pattern recognition across time, +10-20% Sharpe improvement

### 2. 📊 Dual-Trunk Architecture

Separate encoders for policy and value networks reduce interference:

```
Observation (64-dim)
        │
   ┌────┴────┐
   ▼         ▼
[Policy    [Value
 Encoder]   Encoder]
   │         │
  GRU       GRU
   │         │
[Policy    [Value
 Head]      Head]
   │         │
Actions    V(s)
```

**Expected Benefit**: More stable training, faster value function convergence

### 3. 🎯 Attention Mechanism

Add self-attention to capture long-range dependencies:

```python
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)
```

**Expected Benefit**: Better regime detection, +5-15% Sharpe improvement

### 4. 📈 Richer Observation Space

Current: 15 features + 49 zeros = 64 dim

**Add:**
- Rolling volatility (5, 10, 20, 60 day windows)
- Cross-asset correlations (SPY-QQQ, SPY-IWM, QQQ-IWM)
- VIX level and term structure
- Market regime indicators
- Technical indicators (RSI, MACD, Bollinger bands)

**New obs_dim**: 128-256

### 5. 🎲 Action Space Improvements

**Current**: 16 discrete portfolio regimes

**Options:**
1. **Continuous Actions**: Output weights directly [0,1]^4, normalized
2. **Hybrid**: Discrete regime + continuous adjustment
3. **Hierarchical**: Select regime → fine-tune weights

### 6. 🧠 Advanced RL Algorithms

| Algorithm | Pros | Cons |
|-----------|------|------|
| **SAC** | Better exploration, continuous | Complex for discrete |
| **TD3** | Stable, continuous | Not for discrete |
| **IMPALA** | Distributed, fast | Infrastructure heavy |
| **DARC** | Distributional, risk-aware | Complex |

### 7. 💹 Reward Shaping

**Current**: `reward = alpha - trading_cost`

**Improvements:**
```python
# Sharpe-aware reward
rolling_sharpe = returns.mean() / (returns.std() + 1e-8)
reward = alpha * (1 + rolling_sharpe) - trading_cost

# Regime-adjusted
regime_bonus = 0.1 if detected_regime == optimal_regime else 0
reward = alpha + regime_bonus - trading_cost

# Risk-adjusted
drawdown_penalty = 0.1 * max(0, drawdown - 0.05)
reward = alpha - trading_cost - drawdown_penalty
```

### 8. 🔧 Training Improvements

| Improvement | Current | Suggested |
|-------------|---------|-----------|
| Batch size | 8192 | 16384+ |
| Hidden dim | 256 | 512 |
| GRU layers | 1 | 2-3 |
| Learning rate | 3e-4 | Cyclical (1e-5 to 3e-4) |
| Epochs | 4 | 8-10 |
| Entropy coef | 0.03 | Adaptive (0.01-0.1) |

---

## Implementation Priority

1. **Quick Wins** (1-2 hours):
   - ✅ Increase batch size and n_envs
   - ✅ Optimize entropy coefficient
   - Add more technical indicators

2. **Medium Effort** (1 day):
   - Dual-trunk architecture
   - Richer observation space
   - Improved reward shaping

3. **Major Refactor** (1 week):
   - True CLSTM implementation
   - Attention mechanism
   - Continuous action space

---

## Recommended Next Steps

1. **Run extended training** with current optimizations
2. **Monitor metrics**: Look for Sharpe > 2.0, win rate > 55%
3. **Add observation features** (technical indicators)
4. **Implement attention** for regime detection
5. **Consider CLSTM** only if GRU hits performance ceiling

