# Algorithm 2: PPO with LSTM - Usage Guide

## Overview

The `train_ppo_lstm.py` script implements the exact Algorithm 2 specification for PPO with LSTM. This guide shows how to use it effectively.

## Basic Usage

### 1. Quick Start with Simulated Data

```bash
# Train for 100 episodes with simulated data
python train_ppo_lstm.py --episodes 100 --no-data-load --device cpu

# Faster updates for testing
python train_ppo_lstm.py --episodes 50 --no-data-load --update-interval 32
```

### 2. Training with Historical Data

```bash
# Load last 30 days of data
python train_ppo_lstm.py --episodes 1000

# Specific date range
python train_ppo_lstm.py --episodes 1000 \
  --start-date 2024-01-01 \
  --end-date 2024-03-31

# Custom symbols
python train_ppo_lstm.py --episodes 1000 \
  --symbols AAPL TSLA NVDA \
  --start-date 2024-01-01
```

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--episodes` | 1000 | Number of episodes to train |
| `--learning-rate-actor` | 3e-4 | Actor learning rate (αθ) |
| `--learning-rate-critic` | 1e-3 | Critic learning rate (αV) |
| `--gamma` | 0.99 | Discount factor (γ) |
| `--epsilon` | 0.2 | PPO clipping range (ε) |
| `--update-interval` | 128 | Network update interval (T) |
| `--device` | cuda | Device: cuda or cpu |
| `--checkpoint-interval` | 100 | Save checkpoint every N episodes |
| `--symbols` | SPY QQQ IWM AAPL MSFT | Symbols to trade |
| `--start-date` | 30 days ago | Start date (YYYY-MM-DD) |
| `--end-date` | yesterday | End date (YYYY-MM-DD) |
| `--no-data-load` | False | Skip historical data loading |

## Examples

### Development/Testing

```bash
# Quick test with CPU
python train_ppo_lstm.py \
  --episodes 10 \
  --no-data-load \
  --device cpu \
  --update-interval 16

# Test with specific parameters
python train_ppo_lstm.py \
  --episodes 100 \
  --no-data-load \
  --learning-rate-actor 1e-4 \
  --gamma 0.95 \
  --checkpoint-interval 25
```

### Production Training

```bash
# Full training with historical data
python train_ppo_lstm.py \
  --episodes 10000 \
  --symbols SPY QQQ IWM AAPL MSFT TSLA GOOGL \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --device cuda \
  --checkpoint-interval 500

# Continued training from checkpoint
python train_ppo_lstm.py \
  --episodes 5000 \
  --resume checkpoints/ppo_lstm_ep10000.pt
```

### Hyperparameter Tuning

```bash
# Conservative learning
python train_ppo_lstm.py \
  --episodes 2000 \
  --learning-rate-actor 1e-4 \
  --learning-rate-critic 5e-4 \
  --epsilon 0.1 \
  --no-data-load

# Aggressive learning
python train_ppo_lstm.py \
  --episodes 2000 \
  --learning-rate-actor 1e-3 \
  --learning-rate-critic 3e-3 \
  --epsilon 0.3 \
  --no-data-load
```

## Troubleshooting

### SSL Certificate Errors

If you encounter SSL certificate errors when loading data:

```bash
# Use simulated data instead
python train_ppo_lstm.py --no-data-load

# Or set environment variable
export PYTHONHTTPSVERIFY=0
python train_ppo_lstm.py
```

### Out of Memory

```bash
# Use smaller update interval
python train_ppo_lstm.py --update-interval 64

# Use CPU instead of GPU
python train_ppo_lstm.py --device cpu
```

### No Historical Data

```bash
# Use simulated environment
python train_ppo_lstm.py --no-data-load

# Or use the quick training script
python train_ppo_lstm_quick.py --episodes 100
```

## Monitoring Training

The script logs:
- Episode rewards
- Episode lengths
- Network updates
- Checkpoint saves

Example output:
```
INFO:__main__:Episode 1/1000 - Reward: -20.45, Steps: 15
INFO:__main__:Updated networks - Critic Loss: 0.0234, Actor Loss: 0.0012
INFO:__main__:Saved checkpoint to checkpoints/ppo_lstm_ep100.pt
```

## Using Checkpoints

Checkpoints are saved to `checkpoints/` directory:
- `ppo_lstm_ep100.pt` - After 100 episodes
- `ppo_lstm_ep200.pt` - After 200 episodes
- etc.

Load a checkpoint for inference:
```python
from train_ppo_lstm import PPOLSTMTrainer
import torch

# Load checkpoint
checkpoint = torch.load('checkpoints/ppo_lstm_ep1000.pt')
trainer.network.load_state_dict(checkpoint['network_state_dict'])
```

## Performance Tips

1. **Start with simulated data** for faster iteration
2. **Use smaller update intervals** (32-64) for quicker learning
3. **Monitor reward trends** - if stuck, adjust learning rates
4. **Save checkpoints frequently** during long training runs
5. **Use GPU** for faster training with large networks

## Algorithm 2 Parameters

The implementation follows these specifications:
- Actor optimizer: Adam with learning rate αθ
- Critic optimizer: Adam with learning rate αV  
- Advantage calculation: At = rt + γV(st+1) - V(st)
- PPO objective with clipping parameter ε
- Update every T timesteps
- Clear replay buffer after updates