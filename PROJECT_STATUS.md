# Trading Bot Project Status

**Last Updated**: 2024-12-04  
**Goal**: Build an RL-based options trading system for a quant desk

---

## 🎯 The Big Picture

We're building a **reinforcement learning agent** that trades options like a quant desk would. The end state is an agent that can:
1. Allocate capital across multiple assets
2. Use options for leverage, hedging, and income generation
3. Manage risk (Greeks, drawdowns, volatility)
4. Beat simple baselines on risk-adjusted returns

**Current phase**: We have a working **multi-asset ETF allocation agent** (no options yet). This is the foundation before adding options complexity.

---

## ✅ What We've Done

### Phase 0: Fixed Policy Collapse
- **Problem**: Original agent collapsed to single deterministic action
- **Root cause**: Single-asset environment (buy/sell SPY) was too constrained
- **Solution**: Multi-asset environment with 16 discrete portfolio regimes

### Phase 1: Simplified Architecture  
- Replaced complex CLSTM dual-trunk (3.8M params) with **SimplePolicyNetwork** (110K params)
- Single GRU, shared encoder, simple heads
- Training became stable, PPO diagnostics healthy

### Phase 2: Reward Calibration
- **Problem**: Vol/DD penalties made CASH win everything
- **Solution**: Simplified to `reward = alpha - trading_cost`
- Vol/DD tracked for evaluation only, not in training signal
- Baselines now rank correctly: QQQ > SPY > IWM > CASH

### Phase 3: Train/Test Split (OOS Validation)
- Created proper data splits:
  - Train: `data/v2_train_2015_2019/` (2015-2019)
  - Test: `data/v2_test_2020_2024/` (2020-2024)
- **Result**: Model trained on 2015-2019 generalizes to 2020-2024
- RL policy ranks #2 (after ALL_QQQ), beats ALL_SPY on Sharpe

### Phase 4: Engineering Cleanup (Just Completed)
- Added v2 baseline header to `train.py`
- Created `configs/rl_v2_multi_asset.yaml` (canonical config)
- Added `--config` flag support to training
- Created `src/apps/eval_with_baselines.py` (JSON report output)
- Documented observation space in `multi_asset_env.py`
- Created `docs/rl_trader_status.md` (regression checks)

---

## 📍 Where We Are Now

### What Works
```
✅ Multi-asset env with 16 portfolio regimes [CASH, SPY, QQQ, IWM]
✅ SimplePolicyNetwork (GRU, 110K params) - stable training
✅ Clean reward: alpha_vs_benchmark - trading_cost
✅ OOS validation passing (RL beats SPY on 2020-2024 data)
✅ Proper train/test split and evaluation infrastructure
✅ Config-driven training and evaluation
```

### Current Performance (OOS 2020-2024)
| Rank | Policy    | Sharpe | Return |
|------|-----------|--------|--------|
| #1   | ALL_QQQ   | 5.84   | +2.56% |
| #2   | RL_POLICY | 2.87   | +0.84% |
| #3   | ALL_SPY   | 0.96   | +0.32% |

### Key Files
| File | Purpose |
|------|---------|
| `src/apps/train.py` | Training (use `--config configs/rl_v2_multi_asset.yaml`) |
| `src/apps/eval_with_baselines.py` | Evaluation with JSON reports |
| `src/envs/multi_asset_env.py` | Multi-asset environment |
| `configs/rl_v2_multi_asset.yaml` | Canonical v2 config |
| `docs/rl_trader_status.md` | Detailed status + regression checks |

### Quick Commands
```bash
# Train
python src/apps/train.py --config configs/rl_v2_multi_asset.yaml --timesteps 5000000

# Evaluate
python src/apps/eval_with_baselines.py \
    --model checkpoints/clstm_full/model_final_*.pt \
    --cache-path data/v2_test_2020_2024/gpu_cache_test.pt
```

---

## 🚀 Where We're Going

### Next Steps (in order)

#### 1. Feature Enrichment (still ETFs, no options)
- Add rolling volatility features (5d, 20d, 60d)
- Add cross-asset spreads (QQQ/SPY, IWM/SPY)
- Add regime detection flags (high vol, trending, etc.)
- Goal: Help agent detect market conditions

#### 2. Risk Constraints
- Add hard caps: max 80% in one asset, min 5% cash
- Track risk metrics in `info` dict
- Goal: Prevent blow-ups before they happen

#### 3. Walk-Forward Validation
- Test on multiple train/test windows, not just one
- Goal: Confirm robustness across different market regimes

#### 4. Options Layer (THE BIG ONE)
- Extend action space with simple option overlays:
  - "SPY + protective put"
  - "Covered call on QQQ"
  - "Call spread overlay"
- Add Greeks-related observations (IV, term structure)
- Keep reward structure same: alpha - cost
- Goal: Options as overlays on the portfolio allocation

#### 5. Paper Trading Harness
- Replay daily data forward
- Log actions, PnL, risk metrics
- Goal: Bridge to live trading

---

## ⚠️ Known Limitations

1. **No options yet** — Just ETF allocation
2. **Limited features** — Only prices, returns, 20d vol
3. **Single train/test split** — Need walk-forward validation
4. **No execution model** — Instant rebalance, no slippage
5. **QQQ always wins** — In a bull market, growth wins; need regime-conditional logic

---

## 🧠 Key Learnings

1. **Simpler is better**: 110K param GRU beats 3.8M param CLSTM
2. **Clean reward signal**: Remove penalties that distort ordering
3. **OOS validation is essential**: In-sample performance means nothing
4. **Policy collapse = problem formulation**: More actions = more exploration
5. **Cost-aware agents hold more**: 0.3% turnover means model learned costs matter

---

## 📁 Data Layout

```
data/
├── v2_train_2015_2019/gpu_cache_train.pt  # Training data
├── v2_test_2020_2024/gpu_cache_test.pt    # Test data (OOS)
├── v1_train_2020_2024/                    # Legacy (don't use)
└── v1_eval_2015_2019/                     # Legacy (don't use)
```

---

## 🔧 To Resume Tomorrow

1. Read this file and `docs/rl_trader_status.md`
2. Run evaluation to verify nothing broke:
   ```bash
   python src/apps/eval_with_baselines.py \
       --model checkpoints/clstm_full/model_final_20251204_204650.pt \
       --cache-path data/v2_test_2020_2024/gpu_cache_test.pt
   ```
3. Pick next step from "Where We're Going" section
4. Ask user which direction they want to go

