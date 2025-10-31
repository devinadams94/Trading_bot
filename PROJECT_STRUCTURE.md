# Project Structure

Complete overview of the Options Trading Bot repository structure.

## 📁 Directory Layout

```
Trading_bot/
├── src/                              # Core source code
│   ├── __init__.py                   # Package initialization
│   ├── options_clstm_ppo.py          # CLSTM-PPO agent (main algorithm)
│   ├── working_options_env.py        # Trading environment
│   ├── historical_options_data.py    # Data loader
│   ├── paper_optimizations.py        # Research paper optimizations
│   ├── gpu_optimizations.py          # GPU acceleration
│   └── checkpoint_manager.py         # Model checkpoint management
│
├── config/                           # Configuration files
│   ├── symbols_config.yaml           # Trading symbols
│   ├── historical_volatility.json    # Historical volatility data
│   └── config.py                     # Configuration loader
│
├── docs/                             # Documentation
│   ├── RESEARCH_PAPER_IMPLEMENTATION.md
│   ├── MULTI_GPU_TRAINING.md
│   ├── CHECKPOINT_AND_RESUME.md
│   ├── LIVE_TRADING_GUIDE.md
│   └── ... (other documentation)
│
├── scripts/                          # Utility scripts
│   ├── analysis/                     # Performance analysis
│   │   ├── analyze_performance.py
│   │   ├── benchmark_training_speed.py
│   │   └── visualize_performance.py
│   ├── debug/                        # Debugging tools
│   │   ├── debug_training.py
│   │   └── diagnose_training.py
│   └── fixes/                        # Environment fixes
│       └── ... (various fix scripts)
│
├── data/                             # Data storage (gitignored)
│   ├── options_cache/                # Cached options data
│   └── test_cache/                   # Test data cache
│
├── checkpoints/                      # Model checkpoints (gitignored)
│   └── enhanced_clstm_ppo/           # Training checkpoints
│       ├── best_model_composite.pth
│       ├── best_model_sharpe.pth
│       ├── best_model_win_rate.pth
│       ├── best_model_profit_rate.pth
│       ├── latest_model.pth
│       └── training_state.json
│
├── logs/                             # Training logs (gitignored)
│   └── training_YYYYMMDD_HHMMSS.log
│
├── wandb/                            # Weights & Biases logs (gitignored)
│   └── ... (experiment tracking)
│
├── archive/                          # Archived old files (gitignored)
│   ├── old_docs/                     # Old documentation
│   ├── old_training_scripts/         # Old training scripts
│   ├── test_scripts/                 # Old test scripts
│   ├── fix_scripts/                  # Old fix scripts
│   ├── diagnostic_scripts/           # Old diagnostic scripts
│   └── old_src/                      # Old source files
│
├── train_enhanced_clstm_ppo.py       # Main training script
├── paper_trading_bot.py              # Paper trading implementation
├── run_paper_trading.py              # Paper trading runner
│
├── README.md                         # Main documentation
├── SETUP.md                          # Setup guide
├── CONTRIBUTING.md                   # Contribution guidelines
├── LICENSE                           # MIT License
├── PROJECT_STRUCTURE.md              # This file
│
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
└── symbols_config.json               # Legacy symbols config
```

## 📄 Key Files

### Core Source Files

#### `src/options_clstm_ppo.py`
**Purpose**: Main CLSTM-PPO agent implementation

**Key Classes**:
- `OptionsClstmPPO`: Main agent class
- `RolloutBuffer`: Experience replay buffer
- `CLSTMEncoder`: Cascaded LSTM encoder
- `ActorCritic`: Actor-critic network

**Key Methods**:
- `select_action()`: Action selection with exploration
- `train()`: PPO training loop
- `save_checkpoint()`: Save model state
- `load_checkpoint()`: Load model state

#### `src/working_options_env.py`
**Purpose**: Options trading environment (Gym-compatible)

**Key Features**:
- Realistic options pricing
- Greeks calculation (delta, gamma, theta, vega)
- Market microstructure simulation
- Portfolio management
- Risk controls

**Key Methods**:
- `reset()`: Reset environment for new episode
- `step(action)`: Execute action and return next state
- `_calculate_reward()`: Portfolio-based reward calculation

#### `src/historical_options_data.py`
**Purpose**: Historical options data loader

**Key Features**:
- Efficient data caching
- Multiple symbol support
- Data validation
- Memory optimization

#### `src/paper_optimizations.py`
**Purpose**: Research paper optimizations

**Key Components**:
- `TurbulenceCalculator`: Market turbulence detection
- `EnhancedRewardFunction`: Paper-compliant rewards
- `CascadedLSTMFeatureExtractor`: Feature extraction
- `TechnicalIndicators`: Technical analysis

#### `src/gpu_optimizations.py`
**Purpose**: GPU acceleration and optimization

**Key Features**:
- Multi-GPU support
- Mixed precision training (FP16)
- Memory optimization
- Batch processing

#### `src/checkpoint_manager.py`
**Purpose**: Model checkpoint management

**Key Features**:
- Save/load checkpoints
- Best model tracking (by multiple metrics)
- Training state persistence
- Resume functionality

### Main Scripts

#### `train_enhanced_clstm_ppo.py`
**Purpose**: Main training script

**Usage**:
```bash
python train_enhanced_clstm_ppo.py --fresh-start --episodes 1000
```

**Key Features**:
- Fresh start or resume training
- Multi-metric best model tracking
- Real-time logging
- GPU optimization
- Checkpoint management

#### `paper_trading_bot.py`
**Purpose**: Paper trading with Alpaca

**Usage**:
```bash
python paper_trading_bot.py --checkpoint checkpoints/enhanced_clstm_ppo/best_model_sharpe.pth
```

**Key Features**:
- Alpaca API integration
- Real-time trading
- Risk management
- Trade logging

#### `run_paper_trading.py`
**Purpose**: Paper trading runner/wrapper

**Usage**:
```bash
python run_paper_trading.py
```

### Configuration Files

#### `config/symbols_config.yaml`
Trading symbols configuration:
```yaml
symbols:
  - SPY
  - AAPL
  - TSLA
```

#### `config/historical_volatility.json`
Historical volatility data for symbols

#### `requirements.txt`
Python package dependencies

## 🔧 Utility Scripts

### Analysis Scripts (`scripts/analysis/`)

- `analyze_performance.py`: Analyze training performance
- `benchmark_training_speed.py`: Benchmark training speed
- `visualize_performance.py`: Visualize training metrics

### Debug Scripts (`scripts/debug/`)

- `debug_training.py`: Debug training issues
- `diagnose_training.py`: Diagnose training problems

### Fix Scripts (`scripts/fixes/`)

Various environment and training fixes

## 📊 Data Flow

```
Historical Data
      ↓
HistoricalOptionsData (loader)
      ↓
WorkingOptionsEnvironment (env)
      ↓
OptionsClstmPPO (agent)
      ↓
Training Loop (train_enhanced_clstm_ppo.py)
      ↓
Checkpoints (saved models)
      ↓
Paper Trading (paper_trading_bot.py)
```

## 🎯 Training Pipeline

1. **Data Loading**: Load historical options data
2. **Environment Setup**: Initialize trading environment
3. **Agent Creation**: Create CLSTM-PPO agent
4. **Training Loop**:
   - Collect episode data
   - Calculate rewards
   - Train agent (PPO + CLSTM)
   - Save checkpoints
   - Log metrics
5. **Best Model Selection**: Track best models by multiple metrics
6. **Deployment**: Use best model for paper/live trading

## 📈 Checkpoint Structure

```
checkpoints/enhanced_clstm_ppo/
├── best_model_composite.pth      # Best by composite score
├── best_model_sharpe.pth         # Best by Sharpe ratio
├── best_model_win_rate.pth       # Best by win rate
├── best_model_profit_rate.pth    # Best by profit rate
├── latest_model.pth              # Most recent model
└── training_state.json           # Training state
```

Each `.pth` file contains:
- Model weights (actor, critic, CLSTM encoder)
- Optimizer states
- Training step
- Episode number
- Performance metrics

## 🗂️ Archive Structure

Old files are moved to `archive/` during cleanup:

```
archive/
├── old_docs/                     # Old documentation files
├── old_training_scripts/         # Old training variants
├── test_scripts/                 # Old test scripts
├── fix_scripts/                  # Old fix scripts
├── diagnostic_scripts/           # Old diagnostic scripts
└── old_src/                      # Old source files
```

These are kept for reference but not needed for the main project.

## 🚀 Getting Started

1. **Setup**: Follow [SETUP.md](SETUP.md)
2. **Training**: Run `train_enhanced_clstm_ppo.py`
3. **Monitoring**: Check logs and metrics
4. **Testing**: Use paper trading
5. **Deployment**: Deploy best model

## 📚 Documentation

- **[README.md](README.md)**: Project overview
- **[SETUP.md](SETUP.md)**: Setup guide
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Contribution guidelines
- **[docs/](docs/)**: Detailed documentation

## 🔍 Finding Things

**Want to modify the reward function?**
→ `src/working_options_env.py` (line ~201-402)

**Want to change hyperparameters?**
→ `src/options_clstm_ppo.py` (line ~328-341)

**Want to add a new symbol?**
→ `config/symbols_config.yaml`

**Want to change training settings?**
→ `train_enhanced_clstm_ppo.py` (command line args)

**Want to analyze performance?**
→ `scripts/analysis/analyze_performance.py`

**Want to debug training?**
→ `scripts/debug/debug_training.py`

## ✅ Clean Repository

After cleanup, the repository contains only:
- ✅ Essential source code
- ✅ Main training script
- ✅ Configuration files
- ✅ Documentation
- ✅ Utility scripts
- ❌ No test files
- ❌ No old documentation
- ❌ No unused code
- ❌ No temporary files

Ready for GitHub! 🚀

