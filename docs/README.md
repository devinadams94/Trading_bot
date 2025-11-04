# Options Trading Bot with CLSTM-PPO

A sophisticated options trading bot using **Cascaded LSTM with Proximal Policy Optimization (CLSTM-PPO)**, implementing research-backed deep reinforcement learning for options trading.

## 🚀 Features

- **CLSTM-PPO Architecture**: Cascaded LSTM encoder with PPO for robust policy learning
- **GPU Acceleration**: Multi-GPU support with mixed precision training (FP16)
- **Research-Based**: Implements paper-compliant reward functions and optimizations
- **Portfolio Management**: Realistic options trading with Greeks, microstructure, and risk management
- **Paper Trading**: Test strategies with Alpaca paper trading before going live
- **Comprehensive Monitoring**: Training metrics, win rates, Sharpe ratios, and composite scores

## 📁 Project Structure

```
Trading_bot/
├── src/                              # Core source code
│   ├── options_clstm_ppo.py          # CLSTM-PPO agent implementation
│   ├── working_options_env.py        # Main trading environment
│   ├── historical_options_data.py    # Historical data loader
│   ├── paper_optimizations.py        # Research paper optimizations
│   ├── gpu_optimizations.py          # GPU and performance optimizations
│   └── checkpoint_manager.py         # Model checkpoint management
├── config/                           # Configuration files
│   ├── symbols_config.yaml           # Trading symbols configuration
│   └── historical_volatility.json    # Historical volatility data
├── docs/                             # Documentation
│   ├── RESEARCH_PAPER_IMPLEMENTATION.md
│   ├── MULTI_GPU_TRAINING.md
│   ├── CHECKPOINT_AND_RESUME.md
│   └── LIVE_TRADING_GUIDE.md
├── scripts/                          # Utility scripts
│   ├── analysis/                     # Performance analysis tools
│   ├── debug/                        # Debugging utilities
│   └── fixes/                        # Environment fixes
├── train_enhanced_clstm_ppo.py       # Main training script
├── paper_trading_bot.py              # Paper trading implementation
└── run_paper_trading.py              # Paper trading runner

## Main Scripts

- **train_enhanced_clstm_ppo.py** - Main training script with GPU optimization
- **paper_trading_bot.py** - Paper trading for strategy validation
- **run_paper_trading.py** - Paper trading execution wrapper
```

## 🚀 Quick Start

> **📖 For detailed usage instructions, see [USAGE_GUIDE.md](USAGE_GUIDE.md)**

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA support (for GPU training)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install -r requirements.txt
```

> **📋 For complete setup instructions, see [SETUP.md](SETUP.md)**

### 2. Configure Trading Symbols

Edit `config/symbols_config.yaml` to set your trading symbols:
```yaml
symbols:
  - SPY
  - AAPL
  - TSLA
```

### 3. Training

> **🏋️ Main Training Script: `train_enhanced_clstm_ppo.py`**

**Fresh training** (recommended):
```bash
python train_enhanced_clstm_ppo.py --fresh-start --episodes 1000
```

**Resume from checkpoint**:
```bash
python train_enhanced_clstm_ppo.py --resume --episodes 1000
```

**Resume from specific metric** (win_rate, profit_rate, sharpe, or composite):
```bash
python train_enhanced_clstm_ppo.py --resume-from sharpe --episodes 1000
```

> **📖 For all training options, see [USAGE_GUIDE.md](USAGE_GUIDE.md#training-the-model)**

### 4. Monitor Training

Training metrics are logged in real-time:
- **Composite Score**: (win_rate + profit_rate + normalized_return) / 3
- **Win Rate**: Percentage of profitable trades
- **Profit Rate**: Percentage of profitable episodes
- **Sharpe Ratio**: Risk-adjusted returns
- **Average Return**: Portfolio return per episode

### 5. Paper Trading (Optional)

> **💼 Main Trading Scripts: `paper_trading_bot.py` and `run_paper_trading.py`**

Test your trained model with Alpaca paper trading:

```bash
# Set up Alpaca credentials in .env file
# ALPACA_API_KEY=your_api_key
# ALPACA_SECRET_KEY=your_secret_key

# Run paper trading
python run_paper_trading.py
```

> **📖 For paper trading setup and options, see [USAGE_GUIDE.md](USAGE_GUIDE.md#paper-trading)**

## 🏗️ Architecture

### CLSTM-PPO Agent

The agent combines:
- **Cascaded LSTM Encoder**: Multi-layer LSTM for temporal feature extraction
- **Actor Network**: Policy network for action selection
- **Critic Network**: Value network for advantage estimation
- **PPO Training**: Clipped objective with GAE for stable learning

### Action Space

11 discrete actions:
- **0**: Hold (no action)
- **1-5**: Buy call options (5 position sizes: 1%, 2%, 5%, 10%, 20% of capital)
- **6-10**: Buy put options (5 position sizes: 1%, 2%, 5%, 10%, 20% of capital)

### Observation Space

Rich market features:
- **Price History**: OHLCV data with technical indicators
- **Options Chain**: Greeks (delta, gamma, theta, vega), implied volatility
- **Market Microstructure**: Bid-ask spreads, volume profiles
- **Portfolio State**: Current positions, P&L, available capital
- **Time Features**: Time of day, day of week, time to expiration

### Reward Function

Portfolio-based rewards aligned with research:
```python
reward = (portfolio_value_new - portfolio_value_old) / portfolio_value_old * 1e-4
```

## 📊 Training Parameters

### Command Line Arguments

- `--fresh-start`: Start fresh training (clears old checkpoints)
- `--resume`: Resume from latest checkpoint
- `--resume-from {metric}`: Resume from best model by metric (win_rate, profit_rate, sharpe, composite)
- `--episodes N`: Number of training episodes (default: 1000)
- `--checkpoint-dir PATH`: Checkpoint directory (default: checkpoints/enhanced_clstm_ppo)

### Hyperparameters

Configured in `src/options_clstm_ppo.py`:
- **Learning rates**: Actor/Critic = 3e-4, CLSTM = 3e-4
- **Batch size**: 256 (optimized for GPU)
- **PPO epochs**: 10 per episode
- **Discount factor (γ)**: 0.99
- **GAE lambda (λ)**: 0.95
- **Clip epsilon (ε)**: 0.2

## 🎯 Performance Metrics

### Composite Score

The primary metric combining three factors:
```
composite_score = (win_rate + profit_rate + normalized_return) / 3.0
```

### Individual Metrics

- **Win Rate**: % of trades that are profitable
- **Profit Rate**: % of episodes with positive returns
- **Sharpe Ratio**: Risk-adjusted returns (return / std_dev)
- **Average Return**: Mean portfolio return per episode

### Expected Performance

After proper training (500-1000 episodes):
- **Win Rate**: 50-60%
- **Profit Rate**: 40-50%
- **Sharpe Ratio**: 0.5-1.5
- **Composite Score**: 40-60%

## 🐛 Troubleshooting

### Training not improving?

1. Check that both PPO and CLSTM losses are non-zero and changing
2. Verify GPU is being used: `nvidia-smi`
3. Ensure data is loaded: Check logs for "Loaded options data for N symbols"
4. Try fresh start: `--fresh-start`

### Out of memory?

1. Reduce batch size in `src/options_clstm_ppo.py`
2. Use single GPU instead of multi-GPU
3. Reduce number of symbols in `config/symbols_config.yaml`

### Slow training?

1. Verify GPU usage: `nvidia-smi`
2. Check CUDA version matches PyTorch: `python -c "import torch; print(torch.version.cuda)"`
3. Enable mixed precision (already enabled by default)

## 🚫 What's Excluded from Git

The following are excluded via `.gitignore` and not included in the repository:

- **Virtual Environment** (`venv/`) - Recreate with `pip install -r requirements.txt`
- **Training Artifacts** (`checkpoints/`, `logs/`, `wandb/`) - Generated during training
- **Data Cache** (`data/`) - Automatically regenerated when needed
- **Python Cache** (`__pycache__/`, `*.pyc`) - Automatically regenerated
- **Environment Variables** (`.env`) - Contains API keys (never commit!)
- **Model Files** (`*.pth`, `*.pt`) - Large binary files (share via releases)
- **Archive** (`archive/`) - Old files for reference only

> **📖 For complete list, see [USAGE_GUIDE.md](USAGE_GUIDE.md#whats-excluded-from-git)**

## 📚 Documentation

### Getting Started
- **[USAGE_GUIDE.md](USAGE_GUIDE.md)**: Complete usage guide (training & trading) ⭐
- **[SETUP.md](SETUP.md)**: Detailed setup instructions
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: How to contribute

### Technical Documentation
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**: Repository structure
- **[Research Paper Implementation](docs/RESEARCH_PAPER_IMPLEMENTATION.md)**: Paper-compliant features
- **[Multi-GPU Training](docs/MULTI_GPU_TRAINING.md)**: Distributed training guide
- **[Checkpoint & Resume](docs/CHECKPOINT_AND_RESUME.md)**: Checkpoint management

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.

- Options trading carries significant risk
- Past performance does not guarantee future results
- Always test thoroughly with paper trading before live deployment
- Use at your own risk

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Contact

For questions or issues, please open a GitHub issue.