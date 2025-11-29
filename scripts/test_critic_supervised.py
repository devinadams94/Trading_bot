#!/usr/bin/env python3
"""
Supervised critic test (ChatGPT suggestion):
- Freeze policy
- Collect (obs, returns) buffer
- Train just the value head with plain MSE
- Check if EV improves - if not, either no signal in observations or bug in targets
"""
import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.insert(0, '.')

from train_full_clstm import CLSTMPolicyNetwork
from src.envs.gpu_options_env import GPUOptionsEnvironment as GPUOptionsEnv

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Create environment
    env = GPUOptionsEnv(n_envs=2048, device=device)
    obs = env.reset()
    
    # Create policy
    policy = CLSTMPolicyNetwork(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        hidden_dim=256,
        lstm_layers=3
    ).to(device)
    
    print(f"🤖 Policy: {sum(p.numel() for p in policy.parameters()):,} params")
    
    # Collect rollout data
    print("\n📊 Collecting rollout data...")
    n_steps = 256
    obs_buffer = []
    return_buffer = []

    # Collect multiple rollouts
    for rollout in range(10):
        obs, _ = env.reset()  # Returns (obs, info)
        observations = []
        rewards = []

        for step in range(n_steps):
            with torch.no_grad():
                action, _, _, _ = policy.get_action(obs)
            next_obs, reward, done, truncated, info = env.step(action)  # 5-tuple
            observations.append(obs)
            rewards.append(reward)
            obs = next_obs
        
        # Compute returns (simple discounted sum)
        gamma = 0.99
        returns = torch.zeros(n_steps, env.n_envs, device=device)
        running_return = torch.zeros(env.n_envs, device=device)
        for t in reversed(range(n_steps)):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        
        obs_buffer.append(torch.stack(observations))
        return_buffer.append(returns)
        print(f"  Rollout {rollout+1}/10: returns mean={returns.mean():.2f}, std={returns.std():.2f}")
    
    # Flatten
    all_obs = torch.cat([o.reshape(-1, o.shape[-1]) for o in obs_buffer])
    all_returns = torch.cat([r.reshape(-1) for r in return_buffer])
    
    print(f"\n📊 Dataset: {all_obs.shape[0]:,} samples")
    print(f"   Returns: mean={all_returns.mean():.2f}, std={all_returns.std():.2f}")
    
    # Test 1: Value head only (frozen trunk)
    # Test 2: Value head + trunk (unfrozen) - see if LSTM can learn better features
    TRAIN_TRUNK = True  # Toggle this

    if TRAIN_TRUNK:
        # Train everything except policy head
        for param in policy.encoder.parameters():
            param.requires_grad = True
        for param in policy.lstm.parameters():
            param.requires_grad = True
        for param in policy.attention.parameters():
            param.requires_grad = True
        for param in policy.policy.parameters():
            param.requires_grad = False  # Keep policy frozen
        for param in policy.value.parameters():
            param.requires_grad = True
        mode = "trunk + value head"
    else:
        # Freeze everything except value head
        for param in policy.encoder.parameters():
            param.requires_grad = False
        for param in policy.lstm.parameters():
            param.requires_grad = False
        for param in policy.attention.parameters():
            param.requires_grad = False
        for param in policy.policy.parameters():
            param.requires_grad = False
        for param in policy.value.parameters():
            param.requires_grad = True
        mode = "value head only"

    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"🔧 Trainable params ({mode}): {trainable:,}")
    
    # Train value head (and optionally trunk) with MSE
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=1e-3)
    batch_size = 4096
    
    print("\n🏋️ Training value head (supervised MSE)...")
    for epoch in range(20):
        indices = torch.randperm(all_obs.shape[0], device=device)
        total_loss = 0
        n_batches = 0
        
        for start in range(0, all_obs.shape[0], batch_size):
            end = min(start + batch_size, all_obs.shape[0])
            batch_idx = indices[start:end]
            
            _, values, _ = policy(all_obs[batch_idx])
            loss = nn.MSELoss()(values, all_returns[batch_idx])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        # Compute EV after epoch
        with torch.no_grad():
            all_preds = []
            for start in range(0, all_obs.shape[0], batch_size):
                end = min(start + batch_size, all_obs.shape[0])
                _, v, _ = policy(all_obs[start:end])
                all_preds.append(v)
            all_preds = torch.cat(all_preds)
            
            var_y = torch.var(all_returns)
            ev = 1 - torch.var(all_returns - all_preds) / (var_y + 1e-8)
            corr = torch.corrcoef(torch.stack([all_preds, all_returns]))[0, 1]
            
            print(f"  Epoch {epoch+1:2d}: loss={total_loss/n_batches:.4f}, EV={ev:.4f}, corr={corr:.4f}, pred_std={all_preds.std():.3f}")

    print("\n✅ Done! If EV is still ~0, there's no predictive signal in observations.")

if __name__ == '__main__':
    main()

