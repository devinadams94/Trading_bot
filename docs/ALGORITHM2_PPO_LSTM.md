# Algorithm 2: PPO with LSTM Implementation

This implementation follows the exact Algorithm 2 specification for PPO with LSTM.

## Running Algorithm 2

To use the exact Algorithm 2 implementation:

```bash
python train.py --use-algorithm2
```

## Algorithm Steps

The implementation follows these exact steps:

1. **Initialize Networks** (Step 1)
   - Actor network πθ(a|s) with parameters θ
   - Critic network Vϕ(s) with parameters ϕ
   - Both use Adam optimizers

2. **Initialize Replay Buffer D** (Step 2)
   - Stores (ft, at, At) transitions
   - Cleared after each update interval

3. **Episode Loop** (Step 3)
   - Each episode starts with environment reset

4. **Environment Initialization** (Step 4)
   - Get initial state s0 from environment

5. **Step Loop** (Step 5)
   - Process each timestep in the episode

6. **State Reception** (Step 6)
   - Receive state st from environment

7. **LSTM Processing** (Step 7)
   - Process st with LSTM to get feature vector ft
   - Uses CLSTMEncoder with attention layers

8. **Value Estimation** (Step 8)
   - Compute v̂t = Vϕ(ft) using critic network

9. **Action Sampling** (Step 9)
   - Sample action at ~ πθ(at|ft) from policy

10. **Environment Step** (Step 10)
    - Execute at to get reward rt and next state st+1

11. **Advantage Calculation** (Step 11)
    - At = rt + γv̂t+1 - v̂t
    - Where γ is the discount factor (0.99)

12. **Buffer Addition** (Step 12)
    - Add (ft, at, At) to replay buffer D

13. **Update Check** (Step 13)
    - If t mod T = 0, proceed to update

14. **Critic Update** (Step 14)
    - Minimize MSE: (rt + γv̂t+1 - v̂t)²
    - Uses Adam optimizer with learning rate αV

15. **Actor Update** (Step 15)
    - Maximize PPO objective: L^PPO(θ)
    - Clipped surrogate objective with ε = 0.2
    - Uses Adam optimizer with learning rate αθ

16. **Buffer Clear** (Step 16)
    - Clear replay buffer D after updates

## Key Parameters

- **Learning Rates**:
  - Actor (αθ): 3e-4
  - Critic (αV): 1e-3
- **Discount Factor (γ)**: 0.99
- **Clipping Range (ε)**: 0.2
- **Update Interval (T)**: 128 steps

## Implementation Details

### LSTM Feature Processing

The LSTM encoder:
- Takes raw state components
- Processes through 3 LSTM layers
- Uses multi-head attention
- Outputs feature vector ft

### Advantage Estimation

Standard TD advantage:
```
At = rt + γV(st+1) - V(st)
```

### PPO Objective

Clipped surrogate loss:
```
L^PPO(θ) = min(rt(θ)At, clip(rt(θ), 1-ε, 1+ε)At)
```

Where rt(θ) = πθ(at|st) / πold(at|st)

## Differences from Main Training

1. **Exact Algorithm Following**: Steps are executed in exact order
2. **Simple Advantage**: Uses TD advantage instead of GAE
3. **Synchronous Updates**: Updates every T steps exactly
4. **Clear Separation**: Actor and critic have separate optimizers
5. **No Distributed Training**: Focuses on algorithm correctness

## Usage Examples

Basic training:
```bash
python train.py --use-algorithm2
```

With custom parameters:
```bash
python train_ppo_lstm.py
```

## Monitoring

The algorithm logs:
- Episode rewards
- Critic and actor losses
- Update frequency
- Checkpoint saves every 100 episodes

## Checkpoints

Saved to: `checkpoints/ppo_lstm_ep{episode}.pt`

Contains:
- Network state dict
- Optimizer states
- Global step counter