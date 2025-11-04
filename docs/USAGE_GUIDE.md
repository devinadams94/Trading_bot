# Usage Guide - Options Trading Bot

Complete guide for using the Options Trading Bot with CLSTM-PPO.

## 📋 Table of Contents

- [Main Scripts Overview](#main-scripts-overview)
- [Training the Model](#training-the-model)
- [Paper Trading](#paper-trading)
- [What's Excluded from Git](#whats-excluded-from-git)
- [Common Workflows](#common-workflows)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Main Scripts Overview

### **Training Script** (Primary)

**File**: `train_enhanced_clstm_ppo.py`

**Purpose**: Train the CLSTM-PPO agent on historical options data

**Use this for**:
- Training a new model from scratch
- Resuming interrupted training
- Continuing training to improve an existing model
- Experimenting with different hyperparameters

### **Trading Scripts**

#### 1. Paper Trading Bot
**File**: `paper_trading_bot.py`

**Purpose**: Execute trades using Alpaca paper trading API

**Use this for**:
- Testing your trained model with real market data
- Validating strategy performance before live trading
- Debugging trading logic in a safe environment

#### 2. Paper Trading Runner
**File**: `run_paper_trading.py`

**Purpose**: Wrapper script to run paper trading with proper configuration

**Use this for**:
- Quick paper trading execution
- Automated paper trading sessions
- Production-like testing environment

---

## 🏋️ Training the Model

### Quick Start Training

**Start fresh training** (recommended for first time):
```bash
python train_enhanced_clstm_ppo.py --fresh-start --episodes 1000
```

This will:
- ✅ Initialize a new model
- ✅ Train for 1000 episodes (~15-20 hours on GPU)
- ✅ Save checkpoints automatically
- ✅ Track best models by multiple metrics
- ✅ Log training progress in real-time

### Resume Training

**Resume from latest checkpoint**:
```bash
python train_enhanced_clstm_ppo.py --resume --episodes 1000
```

**Resume from best model by specific metric**:
```bash
# Resume from best Sharpe ratio model
python train_enhanced_clstm_ppo.py --resume-from sharpe --episodes 1000

# Resume from best win rate model
python train_enhanced_clstm_ppo.py --resume-from win_rate --episodes 1000

# Resume from best profit rate model
python train_enhanced_clstm_ppo.py --resume-from profit_rate --episodes 1000

# Resume from best composite score model
python train_enhanced_clstm_ppo.py --resume-from composite --episodes 1000
```

### Training Options

**Command Line Arguments**:

| Argument | Description | Default |
|----------|-------------|---------|
| `--fresh-start` | Start fresh training (clears old checkpoints) | False |
| `--resume` | Resume from latest checkpoint | False |
| `--resume-from {metric}` | Resume from best model by metric | None |
| `--episodes N` | Number of episodes to train | 1000 |
| `--checkpoint-dir PATH` | Checkpoint directory | `checkpoints/enhanced_clstm_ppo` |

**Examples**:

```bash
# Quick test (3 episodes)
python train_enhanced_clstm_ppo.py --fresh-start --episodes 3

# Short training session (100 episodes)
python train_enhanced_clstm_ppo.py --fresh-start --episodes 100

# Long training session (2000 episodes)
python train_enhanced_clstm_ppo.py --fresh-start --episodes 2000

# Resume and train for 500 more episodes
python train_enhanced_clstm_ppo.py --resume --episodes 500
```

### Training Output

During training, you'll see:

```
Episode 50/1000:
  Return: -15.23%
  Win Rate: 12.5%
  Profit Rate: 8.0%
  Sharpe Ratio: -0.45
  Composite Score: 5.2%
  Trades: 187
  PPO Loss: 2.3456
  CLSTM Loss: 4.5678
  Time: 3.2s

🎉 New best composite score: 5.2% (episode 50)
💾 Saved checkpoint: checkpoints/enhanced_clstm_ppo/best_model_composite.pth
```

### Monitoring Training

**Watch GPU usage**:
```bash
watch -n 1 nvidia-smi
```

**Check training logs**:
```bash
tail -f logs/training_*.log
```

**Monitor metrics**:
- **Composite Score**: Overall performance (target: 40-60%)
- **Win Rate**: % of profitable trades (target: 50-60%)
- **Profit Rate**: % of profitable episodes (target: 40-50%)
- **Sharpe Ratio**: Risk-adjusted returns (target: 0.5-1.5)
- **PPO Loss**: Should decrease over time
- **CLSTM Loss**: Should decrease over time

### When to Stop Training

**Good signs to stop**:
- ✅ Composite score > 40%
- ✅ Win rate > 50%
- ✅ Sharpe ratio > 0.5
- ✅ Metrics plateauing (not improving for 200+ episodes)

**Warning signs**:
- ❌ Losses becoming NaN
- ❌ Win rate stuck at 0%
- ❌ Returns getting worse over time

---

## 💼 Paper Trading

### Setup Alpaca API

1. **Sign up** at [alpaca.markets](https://alpaca.markets)
2. **Get API keys** from paper trading account
3. **Create `.env` file** in project root:

```bash
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### Run Paper Trading

**Using the runner script** (recommended):
```bash
python run_paper_trading.py
```

**Using the bot directly**:
```bash
python paper_trading_bot.py --checkpoint checkpoints/enhanced_clstm_ppo/best_model_sharpe.pth
```

### Paper Trading Options

**Choose which model to use**:
```bash
# Use best Sharpe ratio model (recommended)
python paper_trading_bot.py --checkpoint checkpoints/enhanced_clstm_ppo/best_model_sharpe.pth

# Use best composite score model
python paper_trading_bot.py --checkpoint checkpoints/enhanced_clstm_ppo/best_model_composite.pth

# Use best win rate model
python paper_trading_bot.py --checkpoint checkpoints/enhanced_clstm_ppo/best_model_win_rate.pth

# Use latest model
python paper_trading_bot.py --checkpoint checkpoints/enhanced_clstm_ppo/latest_model.pth
```

### Paper Trading Output

```
2025-10-04 10:30:15 - INFO - Connected to Alpaca Paper Trading
2025-10-04 10:30:15 - INFO - Account Balance: $100,000.00
2025-10-04 10:30:16 - INFO - Analyzing SPY options...
2025-10-04 10:30:17 - INFO - BUY: SPY Call $450 exp 2025-10-11 (5% position)
2025-10-04 10:30:18 - INFO - Order filled: 2 contracts @ $3.50
2025-10-04 10:30:18 - INFO - Current P&L: +$125.00 (+0.13%)
```

### Monitoring Paper Trading

**Check logs**:
```bash
tail -f paper_trading.log
```

**View Alpaca dashboard**:
- Go to [app.alpaca.markets](https://app.alpaca.markets)
- View positions, orders, and P&L

---

## 🚫 What's Excluded from Git

The `.gitignore` file excludes the following from version control:

### 1. **Virtual Environment**
```
venv/
env/
.venv
```
**Why**: Large, platform-specific, easily recreated with `pip install -r requirements.txt`

### 2. **Training Artifacts**
```
checkpoints/
logs/
wandb/
*.log
```
**Why**: Large files, specific to your training runs, not needed by others

### 3. **Data Cache**
```
data/options_cache/
data/test_cache/
*.cache
```
**Why**: Large cached data files, regenerated automatically when needed

### 4. **Python Cache**
```
__pycache__/
*.pyc
*.pyo
```
**Why**: Compiled Python files, automatically regenerated

### 5. **Archive Directory**
```
archive/
```
**Why**: Old files kept for reference, not needed in clean repository

### 6. **Environment Variables**
```
.env
```
**Why**: Contains API keys and secrets, should never be committed

### 7. **IDE Files**
```
.vscode/
.idea/
*.swp
.DS_Store
```
**Why**: Editor-specific settings, not relevant to other users

### 8. **Model Files**
```
*.pth
*.pt
*.ckpt
*.h5
```
**Why**: Large binary files, should be shared via releases or external storage

### 9. **External Dependencies**
```
qlib/
node_modules/
```
**Why**: External libraries, installed via package managers

### 10. **Research Materials**
```
research_paper.pdf
```
**Why**: Copyrighted material, not distributable

---

## 🔄 Common Workflows

### Workflow 1: Train a New Model

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure symbols
# Edit config/symbols_config.yaml

# 3. Start training
python train_enhanced_clstm_ppo.py --fresh-start --episodes 1000

# 4. Monitor progress
watch -n 1 nvidia-smi

# 5. Wait for training to complete (~15-20 hours)
```

### Workflow 2: Test with Paper Trading

```bash
# 1. Setup Alpaca API
# Create .env file with API keys

# 2. Run paper trading
python run_paper_trading.py

# 3. Monitor trades
tail -f paper_trading.log

# 4. Check Alpaca dashboard for results
```

### Workflow 3: Continue Training

```bash
# 1. Resume from best model
python train_enhanced_clstm_ppo.py --resume-from sharpe --episodes 500

# 2. Monitor improvement
# Watch for composite score increasing

# 3. Test improved model
python run_paper_trading.py
```

### Workflow 4: Experiment with Different Symbols

```bash
# 1. Edit config/symbols_config.yaml
# Add/remove symbols

# 2. Clear old checkpoints
rm -rf checkpoints/enhanced_clstm_ppo/*

# 3. Train fresh model
python train_enhanced_clstm_ppo.py --fresh-start --episodes 1000
```

---

## 🐛 Troubleshooting

### Training Issues

**Problem**: Training not improving (win rate stuck at 0%)

**Solution**:
```bash
# 1. Verify training is happening
grep "PPO Loss" logs/training_*.log | tail -10

# 2. Check that losses are non-zero and changing
# If losses are 0.0000, there's a bug

# 3. Start fresh
rm -rf checkpoints/enhanced_clstm_ppo/*
python train_enhanced_clstm_ppo.py --fresh-start --episodes 100
```

**Problem**: Out of memory error

**Solution**:
- Reduce batch size in `src/options_clstm_ppo.py` (line 340)
- Reduce number of symbols in `config/symbols_config.yaml`
- Use single GPU instead of multi-GPU

**Problem**: Slow training

**Solution**:
```bash
# 1. Verify GPU is being used
nvidia-smi

# 2. Check CUDA version matches PyTorch
python -c "import torch; print(torch.version.cuda)"

# 3. Ensure mixed precision is enabled (default)
```

### Paper Trading Issues

**Problem**: Connection error to Alpaca

**Solution**:
- Verify API keys in `.env` file
- Check internet connection
- Verify Alpaca account is active

**Problem**: No trades being executed

**Solution**:
- Check model is loaded correctly
- Verify symbols have active options
- Check account has sufficient buying power

---

## 📊 Expected Results

### Training Timeline

| Episodes | Win Rate | Profit Rate | Sharpe | Composite | Time |
|----------|----------|-------------|--------|-----------|------|
| 0-50 | 0-15% | 0-10% | -1.0 to -0.5 | 0-10% | 30-60 min |
| 50-200 | 15-30% | 10-25% | -0.5 to 0.0 | 10-25% | 2-4 hours |
| 200-500 | 30-45% | 25-40% | 0.0 to 0.5 | 25-40% | 4-8 hours |
| 500-1000 | 45-60% | 40-50% | 0.5 to 1.5 | 40-60% | 8-15 hours |

### Paper Trading Performance

After 500-1000 episodes of training:
- **Win Rate**: 50-60%
- **Average Return**: 2-8% per day
- **Sharpe Ratio**: 0.5-1.5
- **Max Drawdown**: 10-20%

---

## 📚 Additional Resources

- **[README.md](README.md)**: Project overview
- **[SETUP.md](SETUP.md)**: Detailed setup instructions
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: How to contribute
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**: Repository structure
- **[docs/](docs/)**: Technical documentation

---

## ⚠️ Important Notes

1. **Training takes time**: Expect 15-20 hours for 1000 episodes on GPU
2. **Paper trading first**: Always test with paper trading before live
3. **Monitor closely**: Watch for unusual behavior or losses
4. **Risk management**: Never risk more than you can afford to lose
5. **Educational purpose**: This is for learning, not financial advice

---

## 🎯 Quick Reference

**Train new model**:
```bash
python train_enhanced_clstm_ppo.py --fresh-start --episodes 1000
```

**Resume training**:
```bash
python train_enhanced_clstm_ppo.py --resume --episodes 500
```

**Paper trading**:
```bash
python run_paper_trading.py
```

**Check GPU**:
```bash
nvidia-smi
```

**Monitor logs**:
```bash
tail -f logs/training_*.log
```

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

**Disclaimer**: This software is for educational and research purposes only. Options trading carries significant financial risk. Use at your own risk.

