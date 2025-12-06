# Quick Start Guide

**Get training in 5 minutes**

---

## Prerequisites

```bash
# 1. Clone repository
git clone https://github.com/devinadams94/Trading_bot.git
cd Trading_bot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 4. Verify GPU (optional)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Train the Model

**Basic Training (5M timesteps, ~10-20 hours on GPU):**
```bash
python src/apps/train.py \
    --config configs/rl_v2_multi_asset.yaml \
    --timesteps 5000000
```

**Quick Test (100K timesteps, ~5 minutes):**
```bash
python src/apps/train.py \
    --config configs/rl_v2_multi_asset.yaml \
    --timesteps 100000
```

**Monitor Training:**
```bash
# In separate terminal
tensorboard --logdir=runs
# Open http://localhost:6006
```

---

## Evaluate the Model

```bash
python src/apps/eval_with_baselines.py \
    --model checkpoints/clstm_full/model_final_*.pt \
    --cache-path data/v2_test_2020_2024/gpu_cache_test.pt
```

**Expected Output:**
```
Policy Rankings (by Sharpe Ratio):
┌──────┬─────────────┬────────┬──────────┬──────────┐
│ Rank │ Policy      │ Sharpe │ Return % │ Max DD % │
├──────┼─────────────┼────────┼──────────┼──────────┤
│  #1  │ ALL_QQQ     │  5.84  │  +2.56%  │  -12.3%  │
│  #2  │ RL_POLICY   │  2.87  │  +0.84%  │   -8.1%  │
│  #3  │ ALL_SPY     │  0.96  │  +0.32%  │  -15.2%  │
└──────┴─────────────┴────────┴──────────┴──────────┘
```

---

## What to Look For

### During Training

**Good Signs:**
- ✅ Reward increasing over time
- ✅ Sharpe > 1.0 after 1000 iterations
- ✅ Entropy decaying smoothly (2.7 → 0.5)
- ✅ Explained variance > 0.5
- ✅ Clip fraction 0.1-0.3

**Warning Signs:**
- ❌ Entropy drops to 0 in first 100 iterations (policy collapse)
- ❌ KL divergence > 0.1 (unstable training)
- ❌ Losses not decreasing after 500 iterations
- ❌ Sharpe stays negative

### After Evaluation

**Success Criteria:**
- ✅ RL Sharpe > 1.0
- ✅ RL beats SPY on Sharpe
- ✅ Max drawdown < 15%
- ✅ Turnover < 10%

---

## Common Issues

**Out of Memory:**
```bash
python src/apps/train.py --config configs/rl_v2_multi_asset.yaml --n-envs 1024
```

**Training Too Slow:**
```bash
python src/apps/train.py --config configs/rl_v2_multi_asset.yaml --compile
```

**Policy Collapse:**
```bash
python src/apps/train.py --config configs/rl_v2_multi_asset.yaml --entropy-coef 0.05
```

---

## Next Steps

1. **Read Full Guide**: See `TRAINING_GUIDE.md` for detailed explanations
2. **Understand Metrics**: See "Understanding Metrics" section in `TRAINING_GUIDE.md`
3. **Tune Hyperparameters**: Experiment with learning rate, batch size, hidden dim
4. **Add Features**: See "Adding New Features" in `TRAINING_GUIDE.md`

---

## Key Files

- `src/apps/train.py` - Training script
- `src/apps/eval_with_baselines.py` - Evaluation script
- `src/envs/multi_asset_env.py` - Trading environment
- `configs/rl_v2_multi_asset.yaml` - Configuration
- `TRAINING_GUIDE.md` - Complete training guide
- `PROJECT_STATUS.md` - Project status and roadmap

---

**Happy Training! 🚀**

