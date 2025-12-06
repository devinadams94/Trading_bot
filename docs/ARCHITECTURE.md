# Trading Bot Architecture

**Visual guide to system architecture and data flow**

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRADING BOT SYSTEM                           │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
            ┌───────▼────────┐         ┌────────▼────────┐
            │  TRAINING      │         │  EVALUATION     │
            │  (train.py)    │         │  (eval_*.py)    │
            └───────┬────────┘         └────────┬────────┘
                    │                           │
            ┌───────▼────────────────────────────▼────────┐
            │     MULTI-ASSET ENVIRONMENT                 │
            │     (multi_asset_env.py)                    │
            │  - 4 assets: CASH, SPY, QQQ, IWM           │
            │  - 16 portfolio regimes                     │
            │  - Reward: alpha - trading_cost             │
            └───────┬─────────────────────────────────────┘
                    │
            ┌───────▼────────┐
            │  POLICY NET    │
            │  (GRU-based)   │
            │  ~110K params  │
            └────────────────┘
```

---

## Data Flow

### Training Loop

```
1. LOAD DATA
   ├─ GPU Cache: data/v2_train_2015_2019/gpu_cache_train.pt
   ├─ Contains: Daily OHLCV for SPY, QQQ, IWM (2015-2019)
   └─ Format: PyTorch tensors on GPU

2. INITIALIZE ENVIRONMENT
   ├─ Create 2048 parallel environments
   ├─ Each env: $100K initial capital, 256-step episodes
   └─ Reset to random starting dates

3. COLLECT ROLLOUTS (256 steps × 2048 envs)
   ├─ For each step:
   │   ├─ Get observation (64-dim vector)
   │   ├─ Policy network → action logits + value
   │   ├─ Sample action from policy
   │   ├─ Execute action in environment
   │   ├─ Receive reward and next observation
   │   └─ Store (obs, action, reward, value, log_prob)
   └─ Result: 524,288 transitions

4. COMPUTE ADVANTAGES
   ├─ Calculate returns using GAE (Generalized Advantage Estimation)
   ├─ Normalize advantages (mean=0, std=1)
   └─ Prepare mini-batches (size=2048)

5. PPO UPDATE (4 epochs)
   ├─ For each mini-batch:
   │   ├─ Compute policy loss (clipped surrogate)
   │   ├─ Compute value loss (MSE to returns)
   │   ├─ Compute entropy bonus
   │   ├─ Total loss = policy_loss + 0.5*value_loss - entropy_coef*entropy
   │   ├─ Backprop and clip gradients (max_norm=0.5)
   │   └─ Update weights
   └─ Track metrics: KL div, clip fraction, explained variance

6. LOG & CHECKPOINT
   ├─ Log metrics to console and TensorBoard
   ├─ Save checkpoint every 300 seconds
   └─ Update learning rate (cosine schedule)

7. REPEAT
   └─ Go to step 3 until total_timesteps reached
```

---

## Network Architecture

### SimplePolicyNetwork (GRU-based)

```
INPUT (64-dim observation)
  │
  ├─ [0:3]   price_normalized    (SPY, QQQ, IWM prices)
  ├─ [3:6]   returns_scaled       (daily returns × 10)
  ├─ [6:9]   volatility_scaled    (20-day vol × 10)
  ├─ [9:13]  weights              (current allocation)
  ├─ [13]    portfolio_return     (total return)
  ├─ [14]    drawdown             (current drawdown)
  └─ [15:64] padding              (zeros, reserved)
  │
  ▼
┌─────────────────────────────────────────────────┐
│ ENCODER                                         │
│  Linear(64 → 128) + ReLU + LayerNorm           │
│  Output: 128-dim encoded features              │
└─────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────┐
│ GRU (Recurrent Layer)                           │
│  Input: 128-dim                                 │
│  Hidden: 128-dim (1 layer)                      │
│  Output: 128-dim sequence features              │
│  Hidden state: Carries temporal context         │
└─────────────────────────────────────────────────┘
  │
  ├──────────────────────┬──────────────────────┐
  ▼                      ▼                      ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ POLICY HEAD  │  │  VALUE HEAD  │  │ HIDDEN STATE │
│ Linear(128→16)│  │ Linear(128→1)│  │ (for next    │
│              │  │              │  │  timestep)   │
│ Output:      │  │ Output:      │  └──────────────┘
│ - 16 logits  │  │ - 1 value    │
│   (1 per     │  │   (state     │
│    action)   │  │    value)    │
└──────────────┘  └──────────────┘
  │                      │
  ▼                      ▼
ACTION                VALUE
(portfolio regime)    (expected return)
```

**Parameter Count:**
```
Encoder:     64 × 128 + 128 = 8,320
GRU:         4 × (128 × 128 + 128 × 128) = 131,072
Policy Head: 128 × 16 + 16 = 2,064
Value Head:  128 × 1 + 1 = 129
LayerNorm:   128 × 2 = 256
─────────────────────────────────────────
TOTAL:       ~110,592 parameters
```

---

## Environment Dynamics

### State Transition

```
STATE (t)
  ├─ Portfolio: [CASH, SPY, QQQ, IWM] weights
  ├─ Prices: Current market prices
  ├─ History: 20-day rolling statistics
  └─ Value: Current portfolio value
  │
  ▼
ACTION (t)
  ├─ Select portfolio regime (0-15)
  │   Examples:
  │   - 0: HOLD (no change)
  │   - 2: ALL_SPY (100% S&P 500)
  │   - 10: GROWTH_TILT (60% QQQ, 30% SPY, 10% cash)
  └─ Rebalance portfolio to target weights
  │
  ▼
TRANSITION
  ├─ Calculate trading cost (0.1% of turnover)
  ├─ Execute rebalance (instant, no slippage)
  ├─ Advance time by 1 day
  └─ Update portfolio value with new prices
  │
  ▼
REWARD (t)
  ├─ alpha = portfolio_return - benchmark_return
  │   where benchmark = equal_weight(SPY, QQQ, IWM)
  ├─ cost = 0.001 × portfolio_value × turnover
  └─ reward = alpha - cost
  │
  ▼
STATE (t+1)
  └─ Updated portfolio, prices, statistics
```

### Episode Structure

```
EPISODE (256 steps)
  │
  ├─ Step 0: Reset to random date, $100K capital
  ├─ Step 1: Observe → Act → Reward
  ├─ Step 2: Observe → Act → Reward
  ├─ ...
  ├─ Step 255: Observe → Act → Reward
  └─ Step 256: Terminal state
  │
  ▼
EPISODE METRICS
  ├─ Total reward: Sum of all rewards
  ├─ Total return: (final_value - initial_value) / initial_value
  ├─ Sharpe ratio: sqrt(252) × mean(returns) / std(returns)
  ├─ Max drawdown: Max(peak - current) / peak
  └─ Turnover: Average daily portfolio change
```

---

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ 1. DATA PREPARATION                                         │
│    - Download historical data (Polygon.io API)              │
│    - Process to daily OHLCV                                 │
│    - Create GPU cache (PyTorch tensors)                     │
│    - Split: 2015-2019 (train), 2020-2024 (test)           │
└─────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. TRAINING (train.py)                                      │
│    - Load config (rl_v2_multi_asset.yaml)                  │
│    - Initialize environment (2048 parallel envs)            │
│    - Initialize policy network (SimplePolicyNetwork)        │
│    - Train with PPO (5M timesteps, ~10-20 hours)           │
│    - Save checkpoints every 300 seconds                     │
└─────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. EVALUATION (eval_with_baselines.py)                     │
│    - Load trained model                                     │
│    - Run on test data (2020-2024)                          │
│    - Compare to baselines (ALL_SPY, ALL_QQQ, etc.)         │
│    - Generate JSON report with metrics                      │
└─────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. ANALYSIS                                                 │
│    - Check Sharpe ratio (target: > 1.0)                    │
│    - Check max drawdown (target: < 15%)                    │
│    - Check turnover (target: < 10%)                        │
│    - Verify beats SPY on risk-adjusted basis                │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### Why GRU instead of LSTM?
- **Simpler**: Fewer parameters (110K vs 3.8M)
- **Faster**: Less computation per step
- **More stable**: Easier to train, less prone to vanishing gradients
- **Better generalization**: Less overfitting on small datasets

### Why 16 discrete actions instead of continuous?
- **Interpretable**: Each action has clear meaning (e.g., "ALL_SPY")
- **Stable**: Discrete actions avoid continuous action space exploration issues
- **Efficient**: Faster sampling and evaluation
- **Practical**: Easier to implement in live trading

### Why alpha-based reward instead of raw returns?
- **Benchmark-relative**: Encourages beating the market, not just making money
- **Risk-aware**: Alpha accounts for market conditions
- **Cost-aware**: Penalizes excessive trading
- **Clean signal**: No conflicting objectives (vol/DD tracked separately)

### Why no options yet?
- **Foundation first**: Master ETF allocation before adding options complexity
- **Incremental**: Add options as overlays on top of working system
- **Validation**: Ensure RL works on simpler problem first
- **Roadmap**: Options layer planned for Phase 4 (see PROJECT_STATUS.md)

---

## Performance Characteristics

### Training Speed
- **GPU (RTX 3090)**: ~15,000 steps/sec
- **GPU (H100)**: ~40,000 steps/sec
- **CPU**: ~1,000 steps/sec

### Memory Usage
- **GPU**: ~4-8 GB (2048 envs, batch_size=2048)
- **RAM**: ~8-16 GB
- **Disk**: ~500 MB (data cache)

### Training Time
- **5M timesteps**: 10-20 hours (GPU)
- **10M timesteps**: 20-40 hours (GPU)
- **100K timesteps**: ~5 minutes (quick test)

---

**For more details, see `TRAINING_GUIDE.md`**

