# RL Trader Status — v2 Multi-Asset Baseline

## Current State (as of 2024-12)

We have a **working multi-asset RL trader** that:

- Trades **portfolio regimes over [CASH, SPY, QQQ, IWM]** (16 discrete actions)
- Uses a **simple GRU policy network** (~110K params)
- Trains with **PPO** and produces **non-collapsed policies**
- Has a **clean reward**: `reward = alpha_vs_benchmark - trading_cost`
- Has been **validated out-of-sample** (train: 2015-2019, test: 2020-2024)

### OOS Results (2020-2024)

| Rank | Policy      | Sharpe | Return% | Turnover |
|------|-------------|--------|---------|----------|
| #1   | ALL_QQQ     | 5.84   | +2.56%  | 0.0%     |
| #2   | **RL_POLICY** | **2.87** | **+0.84%** | **0.3%** |
| #3   | ALL_SPY     | 0.96   | +0.32%  | 0.0%     |
| #4   | ALL_IWM     | -5.87  | -3.12%  | 0.0%     |

**Key finding**: RL policy beats ALL_SPY and learned cost-aware behavior (very low turnover).

---

## Quick Start

### Train a new model
```bash
# Using config file
python src/apps/train.py \
    --config configs/rl_v2_multi_asset.yaml \
    --timesteps 5000000

# Or with explicit args
python src/apps/train.py \
    --cache-path data/v2_train_2015_2019/gpu_cache_train.pt \
    --timesteps 5000000 \
    --n-envs 2048
```

### Evaluate with baselines
```bash
python src/apps/eval_with_baselines.py \
    --model checkpoints/clstm_full/model_final_XXXXXXXX.pt \
    --cache-path data/v2_test_2020_2024/gpu_cache_test.pt \
    --output reports/eval_$(date +%Y%m%d).json
```

---

## Key Files

| File | Purpose |
|------|---------|
| `src/apps/train.py` | Training script (SimplePolicyNetwork + PPO) |
| `src/apps/eval_with_baselines.py` | Evaluation with JSON report |
| `src/envs/multi_asset_env.py` | Multi-asset environment (16 regimes) |
| `configs/rl_v2_multi_asset.yaml` | Canonical config for v2 baseline |

---

## Regression Checks

Before merging any changes, verify the model still works:

### Quick Sanity Check
```bash
# 1. Train for a short run
python src/apps/train.py \
    --cache-path data/v2_train_2015_2019/gpu_cache_train.pt \
    --timesteps 1000000

# 2. Evaluate on test data
python src/apps/eval_with_baselines.py \
    --model checkpoints/clstm_full/model_final_*.pt \
    --cache-path data/v2_test_2020_2024/gpu_cache_test.pt
```

### Pass Criteria

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| RL Sharpe | ≥ 0 | Should not be losing money on average |
| RL Sharpe vs SPY | ≥ SPY - 0.5 | Should be competitive with buy-and-hold |
| Turnover | ≤ 50% | Should not be thrashing (cost-aware) |
| RL Rank | ≤ #4 | Should beat at least half the baselines |

**If any of these fail, something is broken in env/model/reward.**

---

## Architecture Summary

### Model: SimplePolicyNetwork
```
Input: obs_dim=64
  ↓
Encoder: Linear(64→128) + ReLU + LayerNorm
  ↓
GRU: 1 layer, hidden=128 (shared actor+critic)
  ↓
Policy Head: Linear(128→16) → action logits
Value Head:  Linear(128→1)  → state value
```
Total: ~110K parameters

### Environment: MultiAssetEnvironment
- **Assets**: CASH (idx=0), SPY (idx=1), QQQ (idx=2), IWM (idx=3)
- **Actions**: 16 portfolio regimes (HOLD, ALL_SPY, ALL_QQQ, etc.)
- **Observation**: 64-dim vector (prices, returns, vol, weights, drawdown)
- **Reward**: `alpha - trading_cost` (no vol/dd penalties in training)

### Data
```
data/
├── v2_train_2015_2019/gpu_cache_train.pt  # Training (1257 days)
└── v2_test_2020_2024/gpu_cache_test.pt    # Testing (1257 days)
```

---

## What's NOT Done Yet

1. **Options trading** — No explicit calls/puts, Greeks, term structure
2. **Richer features** — Obs is just prices/returns/vol, no macro/regime flags
3. **Risk constraints** — No hard caps on weights or exposure
4. **Walk-forward validation** — Only one train/test split tested

See `configs/rl_v2_multi_asset.yaml` for TODO notes on future enhancements.

---

## Troubleshooting

### Policy collapses to single action
- Check entropy coefficient (should be ~0.01, not 0)
- Check if reward scale is too large (should be ~0.1)
- Verify observation space has variation (not all zeros)

### RL policy ranks last
- Check trading_cost is reasonable (0.001 = 0.1%)
- Verify reward formula: should be `alpha - cost`, not `alpha - vol - dd - cost`
- Ensure benchmark is equal-weight equity, not including cash

### Training is slow
- Use `--n-envs 2048` for GPU parallelism
- Ensure cache is loaded (not creating synthetic data)
- Check GPU utilization with `nvidia-smi`

