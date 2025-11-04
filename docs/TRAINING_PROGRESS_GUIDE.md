# Training Progress Indicators Guide

## Overview

The training script now includes comprehensive progress indicators to show exactly what's happening during data loading and training.

## Data Loading Progress

### Phase 1: Initialization
```
================================================================================
📊 DATA LOADING STARTED
================================================================================
  Symbols: 23
  Date range: 2023-01-28 to 2025-11-03 (730 days)
  Estimated time: 15-30 minutes
================================================================================
```

### Phase 2: Stock Data Loading
```
📈 STEP 1/2: Loading stock data...
📊 Loading stock data for 23 symbols...
  [1/23] Loading SPY...
  [1/23] ✅ SPY loaded from cache (12,450 rows)
  [2/23] Loading QQQ...
  [2/23] ✅ QQQ: 12,450 bars (quality: 0.95)
  [3/23] Loading IWM...
  [3/23] ✅ IWM loaded from cache (12,450 rows)
  ...
  [23/23] Loading MA...
  [23/23] ✅ MA: 12,450 bars (quality: 0.92)

✅ Stock data loaded for 23/23 symbols
```

### Phase 3: Options Data Loading
```
📊 STEP 2/2: Loading options data...

📊 Loading historical options data for 23 symbols from 2023-01-28 to 2025-11-03
📊 Loading stock data for 23 symbols...
  [1/23] ✅ SPY loaded from cache (12,450 rows)
  [2/23] ✅ QQQ loaded from cache (12,450 rows)
  ...

📈 Processing options chains for 23 symbols...
  [1/23] Processing options for SPY...
  [1/23] ✅ SPY options loaded from cache (45,230 contracts)
  [2/23] Processing options for QQQ...
  [2/23] 🔄 Generating simulated options for QQQ...
  [2/23] ✅ QQQ: 38,450 options contracts (quality: 0.88)
  ...
  [23/23] Processing options for MA...
  [23/23] ✅ MA: 32,100 options contracts (quality: 0.85)

✅ Completed loading options data for 23/23 symbols
```

### Phase 4: Validation & Completion
```
🔍 Validating data quality...

================================================================================
✅ DATA LOADING COMPLETE
================================================================================
  Successfully loaded: 23/23 symbols
  Total data points: 286,350
  Ready for training!
================================================================================
```

## Training Progress

### Episode Progress
```
Episode 1/100 | Reward: 1234.56 | Win Rate: 0.45 | Sharpe: 1.23 | Loss: 0.0234
Episode 2/100 | Reward: 1456.78 | Win Rate: 0.48 | Sharpe: 1.45 | Loss: 0.0198
...
```

### Checkpoint Saves
```
💾 Saved best model (composite score: 0.8234)
💾 Saved checkpoint at episode 100
```

### GPU Memory Monitoring
```
🖥️ GPU Memory: 45.2% (10.8 GB / 24.0 GB)
```

### Early Stopping
```
⚠️ Early stopping triggered: No improvement for 500 episodes
📊 Best composite score: 0.8234 at episode 1234
```

## Quick Test Mode

For faster testing with progress indicators:

```bash
python train_enhanced_clstm_ppo.py --quick-test
```

This will:
- Use only 3 symbols (SPY, QQQ, AAPL)
- Load 90 days of data (instead of 730)
- Train for 100 episodes
- Complete in 5-10 minutes

## Custom Data Loading

### Control number of days:
```bash
python train_enhanced_clstm_ppo.py --data-days 90
```

### Control number of episodes:
```bash
python train_enhanced_clstm_ppo.py --num_episodes 1000
```

## Progress Indicator Legend

| Symbol | Meaning |
|--------|---------|
| 📊 | Data loading phase |
| 📈 | Stock data processing |
| 📉 | Options data processing |
| ✅ | Success |
| 🔄 | Processing/Generating |
| ⚠️ | Warning (non-critical) |
| ❌ | Error (failed) |
| 💾 | Checkpoint saved |
| 🖥️ | GPU status |
| 🎯 | Training milestone |

## Estimated Times

### First Run (No Cache)
- **3 symbols, 90 days**: 2-5 minutes
- **23 symbols, 90 days**: 5-10 minutes
- **3 symbols, 365 days**: 5-10 minutes
- **23 symbols, 730 days**: 15-30 minutes

### Subsequent Runs (With Cache)
- **Any configuration**: 10-30 seconds (cache loading is fast!)

## Troubleshooting

### If data loading seems stuck:

1. **Check the progress counter**: If you see `[X/Y]` incrementing, it's working
2. **Wait for cache**: First load takes time, but data is cached
3. **Use quick test**: `--quick-test` for faster validation
4. **Check logs**: Look at `logs/training_YYYYMMDD_HHMMSS.log`

### If you need to stop:

1. **Press Ctrl+C once**: Graceful shutdown (saves checkpoint)
2. **Press Ctrl+C twice**: Force quit
3. **Or use**: `pkill -f train_enhanced_clstm_ppo.py`

## Example Full Training Command

```bash
# Production training with progress indicators
python train_enhanced_clstm_ppo.py \
    --num_episodes 5000 \
    --num_gpus 1 \
    --enable-multi-leg \
    --use-ensemble \
    --num-ensemble-models 3 \
    --early-stopping-patience 500 \
    --checkpoint-dir checkpoints/production_run \
    --resume
```

You'll see:
1. ✅ Configuration summary
2. 📊 Data loading with progress (15-30 min first time, 10-30 sec cached)
3. 🎯 Training episodes with metrics
4. 💾 Checkpoint saves every 100 episodes
5. 🖥️ GPU memory monitoring
6. ⚠️ Early stopping if needed

## Tips

1. **First run**: Be patient during data loading (15-30 minutes for full dataset)
2. **Subsequent runs**: Much faster due to caching (10-30 seconds)
3. **Quick testing**: Use `--quick-test` to validate setup
4. **Monitor logs**: Check `logs/` directory for detailed output
5. **GPU usage**: Watch for GPU memory warnings (>90%)

## What's New

✅ **Progress counters** for each symbol: `[5/23]`
✅ **Status indicators**: ✅ 🔄 ⚠️ ❌
✅ **Time estimates** based on data size
✅ **Completion summaries** with statistics
✅ **Phase headers** (STEP 1/2, STEP 2/2)
✅ **Quality scores** for loaded data
✅ **Cache indicators** (loaded from cache vs API)
✅ **Quick test mode** for rapid validation

Now you'll always know exactly what's happening during training! 🚀

