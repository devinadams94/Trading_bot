# Setup Guide

Complete setup guide for the Options Trading Bot with CLSTM-PPO.

## 📋 Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2
- **Python**: 3.9 or higher
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)
  - RTX 3060 or better
  - CUDA 11.8 or 12.x support

### Software Requirements

- Python 3.9+
- pip (Python package manager)
- CUDA Toolkit (for GPU training)
- Git

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Trading_bot.git
cd Trading_bot
```

### Step 2: Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install PyTorch with CUDA

**For GPU (CUDA 12.4):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**For GPU (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Verify Installation

```bash
# Check PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check CUDA availability (if using GPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# Check GPU count
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# Test imports
python -c "from src.options_clstm_ppo import OptionsClstmPPO; print('✅ All imports successful')"
```

Expected output:
```
PyTorch: 2.6.0+cu124
CUDA available: True
CUDA version: 12.4
GPU count: 2
✅ All imports successful
```

## ⚙️ Configuration

### Step 1: Configure Trading Symbols

Edit `config/symbols_config.yaml`:

```yaml
symbols:
  - SPY    # S&P 500 ETF
  - AAPL   # Apple
  - TSLA   # Tesla
  # Add more symbols as needed
```

**Recommendations:**
- Start with 3-5 liquid symbols
- Use high-volume options (SPY, QQQ, AAPL, MSFT, TSLA)
- Avoid low-volume or illiquid options

### Step 2: Configure Alpaca API (Optional - for Paper Trading)

Create a `.env` file in the root directory:

```bash
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

**Get Alpaca API keys:**
1. Sign up at [alpaca.markets](https://alpaca.markets)
2. Go to Paper Trading account
3. Generate API keys
4. Copy keys to `.env` file

### Step 3: Verify Configuration

```bash
# Test data loading
python -c "
from src.historical_options_data import HistoricalOptionsData
data = HistoricalOptionsData()
print(f'✅ Data loader initialized')
"

# Test environment
python -c "
from src.working_options_env import WorkingOptionsEnvironment
env = WorkingOptionsEnvironment()
print(f'✅ Environment initialized')
"
```

## 🏋️ Training

### Quick Start Training

**Fresh training** (recommended for first run):
```bash
python train_enhanced_clstm_ppo.py --fresh-start --episodes 100
```

This will:
- Initialize a new model
- Train for 100 episodes (~6-10 hours on GPU)
- Save checkpoints to `checkpoints/enhanced_clstm_ppo/`
- Log metrics in real-time

### Monitor Training Progress

Watch the logs for:
```
Episode 50/100:
  Return: -15.23%
  Win Rate: 12.5%
  Profit Rate: 8.0%
  Sharpe Ratio: -0.45
  Composite Score: 5.2%
  Trades: 187
  PPO Loss: 2.3456
  CLSTM Loss: 4.5678
```

**Good signs:**
- ✅ Win rate increasing over time
- ✅ Returns improving (becoming less negative, then positive)
- ✅ Both PPO and CLSTM losses are non-zero and changing
- ✅ Trade count is reasonable (100-200 per episode)

**Warning signs:**
- ❌ Win rate stuck at 0%
- ❌ Losses are 0.0000 or NaN
- ❌ No trades being executed
- ❌ GPU utilization is 0%

### Resume Training

If training is interrupted:
```bash
# Resume from latest checkpoint
python train_enhanced_clstm_ppo.py --resume --episodes 1000

# Resume from best model by specific metric
python train_enhanced_clstm_ppo.py --resume-from sharpe --episodes 1000
```

## 🧪 Testing

### Quick Functionality Test

```bash
# Run 3 episodes to verify everything works
python train_enhanced_clstm_ppo.py --fresh-start --episodes 3
```

Expected: Should complete in 10-15 seconds on GPU, no errors.

### Paper Trading Test

```bash
# Test with trained model
python run_paper_trading.py
```

This will:
- Load the best trained model
- Connect to Alpaca paper trading
- Execute trades in simulation
- Log all trades to `paper_trading.log`

## 🐛 Troubleshooting

### CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce batch size in `src/options_clstm_ppo.py`:
   ```python
   batch_size: int = 128  # Reduced from 256
   ```
2. Reduce number of symbols in `config/symbols_config.yaml`
3. Use single GPU instead of multi-GPU

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'torch'`

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Reinstall PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### GPU Not Detected

**Error:** `CUDA available: False`

**Solutions:**
1. Check NVIDIA driver:
   ```bash
   nvidia-smi
   ```
2. Verify CUDA installation:
   ```bash
   nvcc --version
   ```
3. Reinstall PyTorch with correct CUDA version:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

### Training Not Improving

**Symptoms:** Win rate stuck at 0%, returns always negative

**Solutions:**
1. Verify training is actually happening:
   ```bash
   # Check that losses are non-zero
   grep "PPO Loss" logs/training.log | tail -10
   ```
2. Start fresh training:
   ```bash
   rm -rf checkpoints/enhanced_clstm_ppo/*
   python train_enhanced_clstm_ppo.py --fresh-start --episodes 500
   ```
3. Check data quality:
   ```bash
   python -c "
   from src.historical_options_data import HistoricalOptionsData
   data = HistoricalOptionsData()
   print(f'Symbols loaded: {len(data.symbols)}')
   "
   ```

### Slow Training

**Symptoms:** <1 episode per minute on GPU

**Solutions:**
1. Verify GPU is being used:
   ```bash
   watch -n 1 nvidia-smi
   ```
   Should show GPU utilization 70-100% during training.

2. Check batch size (should be 256 for good GPU utilization)
3. Ensure mixed precision is enabled (default in code)

## 📊 Expected Timeline

### Training Progress

**Episodes 1-50** (30-60 minutes):
- Win rate: 0-15%
- Returns: -25% to -15%
- Learning basics of trading

**Episodes 50-200** (2-4 hours):
- Win rate: 15-30%
- Returns: -15% to -5%
- Starting to identify profitable patterns

**Episodes 200-500** (4-8 hours):
- Win rate: 30-45%
- Returns: -5% to +2%
- Becoming profitable

**Episodes 500-1000** (8-15 hours):
- Win rate: 45-60%
- Returns: +2% to +8%
- Consistent profitability

## 🎯 Next Steps

After successful setup:

1. **Run initial training** (100-200 episodes)
2. **Monitor metrics** to ensure learning is happening
3. **Extend training** to 500-1000 episodes
4. **Test with paper trading** before live deployment
5. **Optimize hyperparameters** based on results

## 📚 Additional Resources

- **[README.md](README.md)**: Project overview and quick start
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Contribution guidelines
- **[docs/](docs/)**: Detailed documentation
  - Research paper implementation
  - Multi-GPU training guide
  - Checkpoint management
  - Live trading guide

## ❓ Getting Help

If you encounter issues:

1. Check this troubleshooting section
2. Search existing GitHub issues
3. Create a new issue with:
   - Error message
   - Steps to reproduce
   - System information
   - Relevant logs

## ✅ Setup Checklist

- [ ] Python 3.9+ installed
- [ ] Virtual environment created and activated
- [ ] PyTorch with CUDA installed
- [ ] All dependencies installed
- [ ] Imports verified
- [ ] GPU detected (if using GPU)
- [ ] Symbols configured
- [ ] Quick test run successful (3 episodes)
- [ ] Ready for full training!

Congratulations! You're ready to train your options trading bot! 🚀

