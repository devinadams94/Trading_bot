# Checkpoint and Resume Features

## Overview

The `train_ppo_lstm.py` script now includes advanced checkpoint management:
- **Automatic checkpoint resumption** by default
- **Best model tracking and saving**
- **Flexible checkpoint management**

## Key Features

### 1. Automatic Resume (Default Behavior)

The script automatically finds and loads the latest checkpoint when you run it:

```bash
python train_ppo_lstm.py --episodes 1000
# Automatically resumes from latest checkpoint if available
```

### 2. Best Model Tracking

The system tracks performance and saves the best model based on:
- Average return (70% weight)
- Win rate (30% weight)

Best model is saved to: `checkpoints/ppo_lstm/best_model.pt`

### 3. Checkpoint Files

| File | Description |
|------|-------------|
| `latest.pt` | Always contains the most recent checkpoint |
| `best_model.pt` | Best performing model based on combined score |
| `ep100.pt`, `ep200.pt`, etc | Periodic checkpoints at intervals |

## Command-Line Options

### Resume Control

```bash
# Auto-resume from latest (default)
python train_ppo_lstm.py --episodes 1000

# Disable auto-resume (start fresh)
python train_ppo_lstm.py --episodes 1000 --no-auto-resume

# Force resume (error if no checkpoint)
python train_ppo_lstm.py --episodes 1000 --resume

# Resume from specific checkpoint
python train_ppo_lstm.py --episodes 1000 --resume-from checkpoints/ppo_lstm/ep500.pt
```

### Checkpoint Directory

```bash
# Use custom checkpoint directory
python train_ppo_lstm.py --checkpoint-dir checkpoints/experiment1

# Different directories for different experiments
python train_ppo_lstm.py --checkpoint-dir checkpoints/high_lr --learning-rate-actor 1e-3
```

## Usage Examples

### Continuous Training

Start training:
```bash
python train_ppo_lstm.py --episodes 1000
# Training interrupted at episode 327...
```

Resume automatically:
```bash
python train_ppo_lstm.py --episodes 1000
# ✅ Resumed from episode 327 (step 42016)
# Training from episode 328 to 1327
```

### Best Model Usage

Load the best model for evaluation:
```python
import torch
from train_ppo_lstm import PPOLSTMTrainer

# Load best model
checkpoint = torch.load('checkpoints/ppo_lstm/best_model.pt')
print(f"Best avg return: ${checkpoint['best_avg_return']:.2f}")
print(f"Best win rate: {checkpoint['best_win_rate']:.1%}")
print(f"From episode: {checkpoint['episode']}")
```

### Multiple Experiments

Run different experiments in parallel:
```bash
# Experiment 1: Conservative learning
python train_ppo_lstm.py \
  --checkpoint-dir checkpoints/conservative \
  --learning-rate-actor 1e-4 \
  --episodes 5000

# Experiment 2: Aggressive learning
python train_ppo_lstm.py \
  --checkpoint-dir checkpoints/aggressive \
  --learning-rate-actor 1e-3 \
  --episodes 5000
```

## Training Output

The enhanced output shows:

```
Episode 150/1000 - Reward: 125.43, Return: $234.56, Win Rate: 65.0% (20 trades), Steps: 234
  🏆 New best model! Avg Return: $156.78, Avg Win Rate: 62.3%
  → 100-Episode Averages - Return: $145.67, Win Rate: 61.5%
  → Best so far - Return: $156.78, Win Rate: 62.3%
```

## Checkpoint Information

Each checkpoint contains:
- Model weights (network_state_dict)
- Optimizer states
- Training step counter
- Episode number
- Full history of returns and win rates
- Best model metrics

## Tips

1. **Don't delete `latest.pt`** - It's needed for auto-resume
2. **Keep `best_model.pt`** - Your best performing model
3. **Periodic checkpoints** can be deleted to save space
4. **Use different directories** for different experiments
5. **Check best model updates** in training output (🏆 icon)

## Troubleshooting

### Resume not working?
```bash
# Check if checkpoints exist
ls -la checkpoints/ppo_lstm/

# Force fresh start if needed
python train_ppo_lstm.py --no-auto-resume
```

### Wrong checkpoint loaded?
```bash
# Specify exact checkpoint
python train_ppo_lstm.py --resume-from checkpoints/ppo_lstm/best_model.pt
```

### Lost track of training progress?
```python
# Check checkpoint info
import torch
ckpt = torch.load('checkpoints/ppo_lstm/latest.pt')
print(f"Episode: {ckpt['episode']}")
print(f"Total steps: {ckpt['t']}")
print(f"Episodes trained: {len(ckpt['episode_returns'])}")
```