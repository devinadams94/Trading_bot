# CLSTM-PPO Options Trading Bot

A sophisticated reinforcement learning trading bot that uses Cascaded LSTM with Proximal Policy Optimization (CLSTM-PPO) for automated options trading.

---

## 🎯 Features

### Core Capabilities
- **Multi-Leg Options Strategies**: 91 actions including spreads, straddles, strangles, iron condors, and butterflies
- **Advanced RL Algorithm**: CLSTM-PPO with cascaded LSTM encoder and multi-head attention
- **Realistic Transaction Costs**: Commission, slippage, and bid-ask spread modeling
- **Multi-GPU Training**: Data parallelism with PyTorch DataParallel (automatic batch splitting across GPUs)
- **Fast Data Loading**: Flat file data loader (10,800x faster than REST API)
- **Comprehensive Metrics**: Win rate, profit rate, Sharpe ratio, max drawdown tracking

### Advanced Optimizations
- **Sharpe Ratio Reward Shaping**: Risk-adjusted returns optimization
- **Implied Volatility Prediction**: Neural network-based IV forecasting
- **Greeks-Based Position Sizing**: Delta, gamma, theta, vega-aware sizing
- **Expiration Management**: Automatic position management near expiration
- **Ensemble Methods**: Multiple model voting for robust predictions
- **Technical Indicators**: MACD, RSI, CCI, ADX integration

---

## 🏗️ Architecture

### CLSTM-PPO Agent

**Cascaded LSTM Encoder:**
- 3-layer LSTM with residual connections
- Multi-head attention (8 heads) between layers
- Layer normalization for training stability
- Input projection: observation → 256-dim hidden state
- Total parameters: ~2.9M (2.7M encoder, 200K actor/critic)

**Actor-Critic Network:**
- **Actor**: Policy network (hidden → hidden → hidden/2 → actions)
- **Critic**: Value network (hidden → hidden → hidden/2 → 1)
- Separate optimizers for PPO and CLSTM components

**Observation Space:**
```
Total Input Dim = 27 × num_symbols + 65
├── Price History: num_symbols × 20
├── Technical Indicators: num_symbols × 6 (MACD, RSI, CCI, ADX, etc.)
├── Options Chain: max_positions × 8 = 40
├── Portfolio State: 5 (cash, equity, positions, etc.)
├── Greeks: max_positions × 4 = 20 (delta, gamma, theta, vega)
└── Symbol Encoding: num_symbols
```

**Action Space:**
- **Legacy Mode**: 31 actions (buy calls, buy puts, sell positions)
- **Multi-Leg Mode**: 91 actions (8 strategy types, multiple strikes/expirations)

### Training Features
- **PPO Algorithm**: Clipped surrogate objective, GAE advantages
- **Experience Replay**: Rollout buffer with trajectory storage
- **Supervised Pre-training**: Price/volatility/volume prediction heads
- **Gradient Clipping**: Prevents exploding gradients
- **Early Stopping**: Automatic stopping on performance plateau
- **Checkpoint Management**: Best composite, win rate, and profit rate models
- **TensorBoard Integration**: Real-time visualization of training progress, profitability, and risk metrics

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA 12.4+ (for GPU training)
- 16GB+ RAM recommended
- NVIDIA GPU with 8GB+ VRAM (optional but recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/devinadams94/Trading_bot.git
cd Trading_bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt

# Create .env file with API key
echo "POLYGON_API_KEY=your_api_key_here" > .env
```

---

## 🚀 Quick Start

### 1. Download Training Data

```bash
# Download 2 years of historical data for default symbols
python download_data_to_flat_files.py --days 730

# Download data for specific symbols
python download_data_to_flat_files.py --days 730 --symbols SPY AAPL TSLA NVDA

# Download with custom output directory
python download_data_to_flat_files.py --days 730 --output-dir data/my_data
```

### 2. Train the Model

**Basic Training:**
```bash
# Train with flat files (recommended - 10,800x faster)
python train_enhanced_clstm_ppo.py --use-flat-files --episodes 5000

# Quick test (3 episodes)
python train_enhanced_clstm_ppo.py --use-flat-files --quick-test
```

**Advanced Training:**
```bash
# Multi-GPU training (4 GPUs)
python train_enhanced_clstm_ppo.py --use-flat-files --episodes 5000 --num-gpus 4

# Custom configuration
python train_enhanced_clstm_ppo.py \
    --use-flat-files \
    --episodes 10000 \
    --data-days 730 \
    --initial-capital 100000 \
    --max-positions 5 \
    --enable-multi-leg \
    --use-realistic-costs

# Disable realistic costs for faster training
python train_enhanced_clstm_ppo.py --use-flat-files --no-realistic-costs --episodes 5000
```

### 3. Monitor Training

**Logs:**
```bash
# View latest training log
tail -f logs/training_*.log

# Check for errors
grep -i "error\|warning" logs/training_*.log
```

**Checkpoints:**
```bash
# Checkpoints saved to:
checkpoints/enhanced_clstm_ppo/
├── best_composite_model.pt      # Best overall (win rate + profit rate + return)
├── best_win_rate_model.pt       # Highest win rate
└── best_profit_rate_model.pt    # Highest profit rate
```

---

## 📊 Training Configuration

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--episodes` | 5000 | Number of training episodes |
| `--data-days` | 730 | Days of historical data (2 years) |
| `--initial-capital` | 100000 | Starting capital ($100k) |
| `--max-positions` | 5 | Maximum concurrent positions |
| `--use-flat-files` | False | Use flat file data loader (recommended) |
| `--flat-files-dir` | data/flat_files | Flat files directory |
| `--enable-multi-leg` | True | Enable 91-action multi-leg strategies |
| `--use-realistic-costs` | True | Enable realistic transaction costs |
| `--no-realistic-costs` | - | Disable transaction costs (faster) |
| `--num-gpus` | 1 | Number of GPUs for distributed training |
| `--quick-test` | - | Run 3 episodes for testing |

### Environment Variables

Create a `.env` file in the project root:

```bash
# Polygon.io API key (required for downloading data)
POLYGON_API_KEY=your_polygon_api_key_here
```

---

## 📈 Performance Metrics

The training script tracks comprehensive metrics:

### Episode Metrics
- **Total Return**: Cumulative profit/loss
- **Win Rate**: Percentage of profitable trades
- **Profit Rate**: Average profit per winning trade
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Total Trades**: Number of trades executed

### Model Metrics
- **Actor Loss**: Policy network loss
- **Critic Loss**: Value network loss
- **CLSTM Loss**: Encoder loss (supervised pre-training)
- **Entropy**: Policy exploration measure
- **KL Divergence**: Policy update magnitude

### Checkpointing Strategy

Three models are saved during training:

1. **Best Composite Model** (`best_composite_model.pt`)
   - Optimizes: `0.4 × win_rate + 0.3 × profit_rate + 0.3 × normalized_return`
   - Best overall performance

2. **Best Win Rate Model** (`best_win_rate_model.pt`)
   - Highest percentage of winning trades
   - Conservative strategy

3. **Best Profit Rate Model** (`best_profit_rate_model.pt`)
   - Highest average profit per trade
   - Aggressive strategy

---

## 📊 TensorBoard Visualization

### Real-Time Training Monitoring

**Start training:**
```bash
python train_enhanced_clstm_ppo.py --use-flat-files --episodes 5000
```

**Launch TensorBoard (in separate terminal):**
```bash
source venv/bin/activate
tensorboard --logdir=runs
```

**Open browser:** http://localhost:6006

### Metrics Tracked

**Profitability:**
- Episode returns, portfolio value, win rate
- Cumulative profitability rate
- Rolling averages (100 episodes)

**Risk:**
- Maximum drawdown
- Current drawdown
- Return volatility

**Model Performance:**
- PPO, CLSTM, Actor, Critic losses
- Policy entropy (exploration)
- KL divergence (policy updates)

**See [TENSORBOARD_GUIDE.md](TENSORBOARD_GUIDE.md) for detailed visualization guide.**

---

## 🎓 Multi-Leg Strategies

### Supported Strategies (91 Actions)

**Basic Options (Actions 0-60):**
- Hold (1 action)
- Buy Calls (15 strikes: -7% to +7% from current price)
- Buy Puts (15 strikes: -7% to +7% from current price)
- Sell Calls / Covered Calls (15 strikes)
- Sell Puts / Cash-Secured Puts (15 strikes)

**Advanced Spreads (Actions 61-90):**
- **Bull Call Spreads** (5 variations): Defined-risk bullish strategy
- **Bear Put Spreads** (5 variations): Defined-risk bearish strategy
- **Long Straddles** (5 expirations): High volatility play
- **Long Strangles** (5 expirations): Lower-cost volatility play
- **Iron Condors** (5 variations): Range-bound income strategy
- **Butterfly Spreads** (5 variations): Neutral strategy

### Strategy Selection

The agent learns to select strategies based on:
- Market conditions (volatility, trend)
- Portfolio state (cash, positions, Greeks)
- Technical indicators (MACD, RSI, CCI, ADX)
- Options chain data (implied volatility, open interest)

---

## 🔧 Advanced Features

### Flexible Checkpoint Loading

The model supports loading checkpoints even when the observation space changes (e.g., different number of symbols):

- **Automatic dimension detection** from checkpoint metadata
- **Partial weight transfer** for compatible layers (LSTM, attention, actor, critic)
- **Reinitialization** of incompatible layers (input projection)
- **Preserves learned knowledge** while adapting to new observation space

Example:
```python
# Checkpoint saved with 46 symbols (1313 input dims)
# Current model uses 56 symbols (1565 input dims)
# → Automatically transfers LSTM/attention weights, reinitializes input layer
```

### GPU Optimization

**Single GPU:**
```bash
python train_enhanced_clstm_ppo.py --use-flat-files --episodes 5000
```

**Multi-GPU (DataParallel):**
```bash
# Automatically uses all available GPUs
python train_enhanced_clstm_ppo.py --use-flat-files --episodes 5000 --num-gpus -1

# Or specify number of GPUs
python train_enhanced_clstm_ppo.py --use-flat-files --episodes 5000 --num-gpus 2
```

**How It Works:**
- ✅ **Single Process**: One training process manages all GPUs (simpler than DDP)
- ✅ **Automatic Batch Splitting**: PyTorch DataParallel automatically splits batches across GPUs
- ✅ **Gradient Aggregation**: Gradients are automatically averaged across GPUs
- ✅ **Sequential Episodes**: Episodes are trained sequentially (1, 2, 3, ...) not in parallel
- ✅ **Expected Speedup**: ~70-80% efficiency (e.g., 2 GPUs = 1.5-1.6x speedup)

**Features:**
- Automatic GPU detection and allocation
- No complex distributed setup required
- Memory-efficient training with mixed precision
- GPU memory monitoring and logging
- Single checkpoint file (no race conditions)

### Data Loading Options

**Option 1: Flat Files (Recommended)**
- **Speed**: 10,800x faster than REST API
- **Offline**: No API calls during training
- **Setup**: Run `download_data_to_flat_files.py` once
- **Format**: Parquet (default) or CSV

```bash
python download_data_to_flat_files.py --days 730
python train_enhanced_clstm_ppo.py --use-flat-files --episodes 5000
```

**Option 2: REST API (Legacy)**
- **Speed**: Slower (18 minutes vs 0.1 seconds)
- **Online**: Fetches data during training
- **Setup**: Just set `POLYGON_API_KEY` in `.env`

```bash
python train_enhanced_clstm_ppo.py --episodes 5000
```

---

## 📁 Project Structure

```
Trading_bot/
├── train_enhanced_clstm_ppo.py      # Main training script
├── download_data_to_flat_files.py   # Data download utility
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── LICENSE                          # License
├── .env                             # API keys (create this)
│
├── src/                             # Source code
│   ├── options_clstm_ppo.py         # CLSTM-PPO agent
│   ├── working_options_env.py       # Trading environment (31 actions)
│   ├── multi_leg_options_env.py     # Multi-leg environment (91 actions)
│   ├── historical_options_data.py   # REST API data loader
│   ├── flat_file_data_loader.py     # Flat file data loader
│   ├── paper_optimizations.py       # Research paper optimizations
│   ├── gpu_optimizations.py         # GPU utilities
│   ├── advanced_optimizations.py    # Sharpe, Greeks, IV prediction
│   ├── realistic_transaction_costs.py  # Transaction cost modeling
│   └── multi_leg_strategies.py      # Strategy builder
│
├── data/                            # Training data
│   └── flat_files/                  # Downloaded flat files
│       ├── stocks/                  # Stock price data
│       └── options/                 # Options chain data
│
├── checkpoints/                     # Model checkpoints
│   └── enhanced_clstm_ppo/
│       ├── best_composite_model.pt
│       ├── best_win_rate_model.pt
│       └── best_profit_rate_model.pt
│
├── logs/                            # Training logs
│   └── training_YYYYMMDD_HHMMSS.log
│
└── docs/                            # Documentation
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Checkpoint Loading Error (Dimension Mismatch)**
```
Error: size mismatch for input_projection.weight
```
**Solution**: The model automatically handles this with partial weight transfer. Just continue training.

**2. Out of Memory (OOM)**
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size or number of symbols:
```bash
python train_enhanced_clstm_ppo.py --use-flat-files --max-positions 3
```

**3. Data Coverage Warning**
```
⚠️ Data coverage: 499/730 days (68%)
```
**Solution**: This is normal! 499 trading days out of 730 calendar days = 99% coverage (weekends/holidays excluded).

**4. HTTP 429 Rate Limit**
```
HTTP 429: Too Many Requests
```
**Solution**: The script has automatic retry with exponential backoff. Just wait, it will continue.

---

## 📚 References

### Research Papers
- **PPO**: Schulman et al. (2017) - "Proximal Policy Optimization Algorithms"
- **LSTM**: Hochreiter & Schmidhuber (1997) - "Long Short-Term Memory"
- **Attention**: Vaswani et al. (2017) - "Attention Is All You Need"

### Data Source
- **Polygon.io**: Historical stock and options data
- **API Documentation**: https://polygon.io/docs

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading options involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research and consult with a financial advisor before trading.

---

## 📧 Contact

- **GitHub**: [@devinadams94](https://github.com/devinadams94)
- **Repository**: [Trading_bot](https://github.com/devinadams94/Trading_bot)

---

**Happy Trading! 🚀📈**


