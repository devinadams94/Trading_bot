# Training System Updates

This document tracks all improvements made to the trading bot training system.

---

## 2025-12-05: Major Training Enhancements

### 1. Separate Actor/Critic Optimizers

**Before:** Single optimizer with shared learning rate for all parameters.

**After:** Separate optimizers with different learning rates:
- Actor optimizer: 3e-4 (encoder + GRU + policy head)
- Critic optimizer: 9e-4 (3x faster for value head)

**Why it's better:**
- The critic (value function) often needs to learn faster than the actor to provide accurate baselines
- Separate optimizers prevent the actor from being held back by the critic's learning needs
- Improves explained variance from ~0.15 to ~0.5+ (target: > 0.5)

---

### 2. Extra Critic Training Epochs

**Before:** Critic trained jointly with actor for 4 PPO epochs only.

**After:** 4 additional critic-only epochs after joint PPO training (configurable via `--critic-epochs`).

**Why it's better:**
- Critic sees 2x more training updates than actor
- Value function accuracy directly impacts advantage estimation quality
- Higher explained variance = more stable policy updates

---

### 3. Enhanced Network Architecture

**Before (110K params):**
```
Encoder: Linear(64→128) + ReLU + LayerNorm
GRU: 1 layer, hidden=128
Policy Head: Linear(128→16)
Value Head: Linear(128→1)
```

**After (941K params with hidden_dim=256):**
```
Encoder: 2 layers with dropout
  ├── Linear(64→256) + ReLU + LayerNorm
  ├── Linear(256→256) + ReLU + LayerNorm
  └── Dropout(0.1)
GRU: 2 layers, hidden=256, dropout=0.1
Policy Head: Linear(256→128) + ReLU + LayerNorm + Linear(128→16)
Value Head: Linear(256→128) + ReLU + LayerNorm + Linear(128→1)
```

**Why it's better:**
- 8.5x more parameters for better feature extraction
- 2-layer GRU captures longer temporal dependencies
- Separate hidden layers for policy/value reduce interference
- Dropout prevents overfitting
- Configurable via `--hidden-dim`, `--n-gru-layers`, `--dropout`

---

### 4. High VRAM Optimization (50GB+)

**Before:** Conservative defaults (8,192 envs, 8,192 batch size).

**After:** Aggressive defaults for high VRAM systems:
- `n_envs`: 65,536 (8x more parallel environments)
- `batch_size`: 65,536 (matches n_envs)
- `n_steps`: 512 (longer rollouts)

**Why it's better:**
- More parallel environments = more samples per second
- Larger batches = better GPU utilization
- Longer rollouts = better advantage estimation
- Estimated speedup: 8-10x faster training

**New CLI options:**
- `--high-vram`: Use aggressive settings for 50GB+ VRAM
- `--low-vram`: Use conservative settings for 8GB VRAM

---

### 5. GPU Memory Display Fix

**Before:** `torch.cuda.memory_allocated()` showed 0.00 GB (only active tensors).

**After:** `torch.cuda.memory_reserved()` shows actual GPU memory held by PyTorch.

**Why it's better:**
- Accurate representation of GPU memory usage
- Helps users understand actual VRAM consumption

---

### 6. Checkpoint Resume Functionality

**Before:** No resume capability; training always started fresh.

**After:** Automatic checkpoint resume with `--resume` flag:
- Finds latest checkpoint by modification time
- Loads model weights and optimizer states
- Handles architecture mismatches gracefully (starts fresh with warning)
- Restores metrics (best_reward, total_trades, win_rate)

**Why it's better:**
- Training can be interrupted and resumed without losing progress
- Supports both legacy single-optimizer and new dual-optimizer checkpoint formats

---

### 7. Composite Model Score

**Before:** Only trade-based metrics (win rate, avg return).

**After:** Comprehensive model score combining multiple factors:

| Component | Weight | What it Measures |
|-----------|--------|------------------|
| Profitability | 40% | avg_return × win_rate |
| Reward Signal | 20% | Normalized avg_reward |
| Value Accuracy | 20% | Explained variance |
| Exploration | 10% | Entropy ratio |
| Stability | -10% | Penalty for high KL/clip |

**Why it's better:**
- Single metric to track overall model quality
- Balances trading performance with RL health metrics
- Color-coded display (green > +20, red < 0)

---

### 8. Expanded Symbol Support

**Before:** 4 assets (SPY, QQQ, IWM, CASH) with 16 portfolio regimes.

**After:** 13 assets with 24 portfolio regimes:
- ETFs: SPY, QQQ, IWM
- Tech stocks: AAPL, MSFT, NVDA, TSLA, AMZN, META, GOOGL, AMD, NFLX
- CASH

**Why it's better:**
- More diversification options
- Better exposure to high-volatility tech stocks
- Dynamic portfolio regime generation

**Usage:** `--expanded --cache-path data/v2_expanded/gpu_cache_expanded.pt`

---

## CLI Quick Reference

```bash
# Standard training with all improvements
python src/apps/train.py \
  --config configs/rl_v2_multi_asset.yaml \
  --hidden-dim 256 \
  --n-gru-layers 2 \
  --critic-epochs 4 \
  --high-vram \
  --resume

# Maximum capacity training
python src/apps/train.py \
  --hidden-dim 512 \
  --n-gru-layers 3 \
  --n-envs 131072 \
  --batch-size 131072 \
  --critic-epochs 8 \
  --expanded \
  --cache-path data/v2_expanded/gpu_cache_expanded.pt
```

---

---

### 9. VRAM Capping and Auto-Calculation

**Before:** Fixed n_envs/batch_size could cause OOM errors (tried to allocate 312,500 GB!).

**After:** Automatic VRAM-aware settings:
- `--max-vram-gpu0 45`: Cap GPU 0 at 45GB (default)
- `--max-vram-gpu1 20`: Cap GPU 1+ at 20GB (default)
- Auto-calculates safe n_envs, n_steps, batch_size based on actual VRAM
- Warns and caps if settings exceed safe limits

**Memory calculation formula:**
```
bytes_per_sample = obs_dim*4 + 8 + 4 + 4 + 4 + 1 + n_actions*4
                 ≈ 341 bytes for obs_dim=64, n_actions=16

max_samples = (usable_vram_gb * 1e9) / (bytes_per_sample * 1.5 overhead)
n_envs = max_samples // n_steps
```

**Why it's better:**
- Prevents OOM crashes
- Automatically optimizes for available hardware
- Supports multi-GPU with per-GPU caps

**Conservative defaults:**
```
n_envs: 16384 (down from 65536)
n_steps: 256 (down from 512)
batch_size: 8192 (down from 65536)
```

---

---

### 10. Flexible Training Duration (Episodes/Iterations)

**Before:** Only `--timesteps` to specify training duration (default: 10M).

**After:** Three ways to specify how long to train:

| Flag | Description | Conversion |
|------|-------------|------------|
| `--timesteps 1000000` | Direct timestep count | N/A |
| `--episodes 100` | Number of episodes | episodes × episode_length |
| `--iterations 5` | Number of PPO iterations | iterations × n_envs × n_steps |

**Examples:**
```bash
# Run for exactly 100 episodes (25,600 timesteps with default episode_length=256)
python src/apps/train.py --episodes 100

# Run for 5 training iterations (2.6M timesteps with n_envs=2048, n_steps=256)
python src/apps/train.py --iterations 5

# Traditional timestep-based (10M steps)
python src/apps/train.py --timesteps 10000000
```

**Why it's better:**
- More intuitive for quick tests (`--episodes 10`)
- Easier to reason about training duration
- Priority order: `--timesteps` > `--episodes` > `--iterations` > config default

---

### 10. Progress Indicators and Verbosity Control

**Before:** Training appeared to "stall" during PPO updates with no visible progress. Debug logs always printed.

**After:** Three verbosity modes with real-time progress indicators:

| Mode | Flag | Behavior |
|------|------|----------|
| Default | (none) | Progress bars for rollout, PPO, and critic updates |
| Verbose | `--verbose` or `-v` | Progress + debug logs (rollout stats, indexing checks, advantage stats) |
| Quiet | `--quiet` or `-q` | No progress bars, minimal output (production mode) |

**Progress indicators show:**
```
============================================================
  🔄 Iteration 1 | Steps: 0/1,048,576 (0.0%)
============================================================
  🎲 Collecting rollout (256 steps × 2,048 envs)... done (2.2s)
  📈 PPO Update: 500/2048 batches (24%) | Epoch 2/4
  🎯 Critic Update: 1500/2048 batches (73%) | Extra Epoch 2/4
```

**Why it's better:**
- User knows training is progressing during long PPO updates
- Debug output controlled via `--verbose` flag (not cluttering normal runs)
- `--quiet` mode for production/background runs
- Rollout timing visible for performance tuning

---

## Future Updates

<!-- Add new updates below this line -->

