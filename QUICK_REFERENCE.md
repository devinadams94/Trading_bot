# Quick Reference Card

Fast reference for common commands and workflows.

## 🎯 Main Scripts

| Script | Purpose | Use For |
|--------|---------|---------|
| `train_enhanced_clstm_ppo.py` | **Training** | Train the CLSTM-PPO model |
| `paper_trading_bot.py` | **Trading** | Execute paper trades with Alpaca |
| `run_paper_trading.py` | **Trading** | Quick paper trading runner |

## ⚡ Common Commands

### Training

```bash
# Start fresh training
python train_enhanced_clstm_ppo.py --fresh-start --episodes 1000

# Resume training
python train_enhanced_clstm_ppo.py --resume --episodes 500

# Resume from best Sharpe model
python train_enhanced_clstm_ppo.py --resume-from sharpe --episodes 500

# Quick test (3 episodes)
python train_enhanced_clstm_ppo.py --fresh-start --episodes 3
```

### Paper Trading

```bash
# Run paper trading (recommended)
python run_paper_trading.py

# Run with specific model
python paper_trading_bot.py --checkpoint checkpoints/enhanced_clstm_ppo/best_model_sharpe.pth
```

### Monitoring

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor training logs
tail -f logs/training_*.log

# Monitor paper trading
tail -f paper_trading.log
```

## 📊 Training Metrics

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| Composite Score | 40-60% | >40% | >50% |
| Win Rate | 50-60% | >45% | >55% |
| Profit Rate | 40-50% | >35% | >45% |
| Sharpe Ratio | 0.5-1.5 | >0.5 | >1.0 |

## 🔧 Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Verify setup
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 📁 Key Files

| File | Description |
|------|-------------|
| `src/options_clstm_ppo.py` | CLSTM-PPO agent |
| `src/working_options_env.py` | Trading environment |
| `config/symbols_config.yaml` | Trading symbols |
| `checkpoints/enhanced_clstm_ppo/` | Saved models |
| `.env` | API keys (create this) |

## 🚫 Excluded from Git

- `venv/` - Virtual environment
- `checkpoints/` - Model checkpoints
- `logs/` - Training logs
- `data/` - Data cache
- `.env` - API keys
- `archive/` - Old files

## 📖 Documentation

- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Complete usage guide
- **[SETUP.md](SETUP.md)** - Setup instructions
- **[README.md](README.md)** - Project overview

## 🆘 Troubleshooting

```bash
# Training not improving?
rm -rf checkpoints/enhanced_clstm_ppo/*
python train_enhanced_clstm_ppo.py --fresh-start --episodes 100

# Out of memory?
# Reduce batch_size in src/options_clstm_ppo.py (line 340)

# GPU not detected?
python -c "import torch; print(torch.cuda.is_available())"
```

## 📄 License

MIT License - See [LICENSE](LICENSE) file

