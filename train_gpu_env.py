#!/usr/bin/env python3
"""
GPU-Accelerated Training Script

This script uses a fully GPU-based environment for maximum throughput.
All environment stepping happens on GPU, not CPU.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
import numpy as np
import logging
import time
import argparse
import os
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def apply_gpu_optimizations():
    """Apply all GPU optimizations for H200"""
    if torch.cuda.is_available():
        cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        cudnn.allow_tf32 = True
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        logger.info("✅ Applied GPU optimizations (TF32, cuDNN benchmark)")


class SimplePolicyNetwork(nn.Module):
    """Simple MLP policy for GPU environment"""
    def __init__(self, obs_dim: int = 64, n_actions: int = 31, hidden_dim: int = 256):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.policy(x), self.value(x)
    
    def get_action(self, obs):
        logits, value = self.forward(obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1)


def train_gpu(args):
    """Main GPU training loop"""
    apply_gpu_optimizations()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️  Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Import GPU environment
    from src.envs.gpu_options_env import GPUOptionsEnvironment

    # Create GPU environment with direct parquet path
    logger.info(f"🚀 Creating GPU environment with {args.n_envs} parallel instances...")
    env = GPUOptionsEnvironment(
        data_loader='data/flat_files_processed/options_with_greeks_2020-01-02_to_2024-12-31.parquet',
        symbols=['SPY', 'QQQ', 'IWM'],
        n_envs=args.n_envs,
        episode_length=256,
        device='cuda'
    )
    
    # Create policy network
    policy = SimplePolicyNetwork(obs_dim=64, n_actions=31, hidden_dim=512).to(device)
    policy = torch.compile(policy, mode='reduce-overhead')
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    
    logger.info(f"🤖 Policy network: {sum(p.numel() for p in policy.parameters()):,} parameters")
    
    # Training loop
    logger.info(f"🎯 Training for {args.timesteps:,} timesteps...")
    
    obs, _ = env.reset()
    total_steps = 0
    episode_count = 0
    start_time = time.time()
    last_log_time = start_time
    
    # Rollout storage
    n_steps = args.n_steps
    obs_buffer = torch.zeros((n_steps, args.n_envs, 64), device=device)
    action_buffer = torch.zeros((n_steps, args.n_envs), device=device, dtype=torch.int64)
    reward_buffer = torch.zeros((n_steps, args.n_envs), device=device)
    log_prob_buffer = torch.zeros((n_steps, args.n_envs), device=device)
    value_buffer = torch.zeros((n_steps, args.n_envs), device=device)
    done_buffer = torch.zeros((n_steps, args.n_envs), device=device, dtype=torch.bool)
    
    while total_steps < args.timesteps:
        # Collect rollout
        with torch.no_grad():
            for step in range(n_steps):
                obs_buffer[step] = obs
                action, log_prob, value = policy.get_action(obs)
                action_buffer[step] = action
                log_prob_buffer[step] = log_prob
                value_buffer[step] = value
                
                obs, reward, terminated, truncated, _ = env.step(action)
                reward_buffer[step] = reward
                done_buffer[step] = terminated | truncated
                
                episode_count += done_buffer[step].sum().item()
        
        total_steps += n_steps * args.n_envs
        
        # PPO update
        returns = reward_buffer.flip(0).cumsum(0).flip(0)  # Simple returns
        advantages = returns - value_buffer
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Flatten
        flat_obs = obs_buffer.reshape(-1, 64)
        flat_actions = action_buffer.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_old_log_probs = log_prob_buffer.reshape(-1)
        
        # Mini-batch updates
        batch_size = args.batch_size
        indices = torch.randperm(flat_obs.shape[0], device=device)
        
        for start in range(0, flat_obs.shape[0], batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            
            logits, values = policy(flat_obs[batch_idx])
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(flat_actions[batch_idx])
            
            # PPO loss
            ratio = torch.exp(new_log_probs - flat_old_log_probs[batch_idx])
            surr1 = ratio * flat_advantages[batch_idx]
            surr2 = torch.clamp(ratio, 0.8, 1.2) * flat_advantages[batch_idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * ((values.squeeze() - flat_returns[batch_idx]) ** 2).mean()
            entropy = dist.entropy().mean()
            
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Logging
        if time.time() - last_log_time > 5:
            elapsed = time.time() - start_time
            steps_per_sec = total_steps / elapsed
            gpu_mem = torch.cuda.memory_allocated() / 1e9
            logger.info(f"📊 Steps: {total_steps:,} | Episodes: {episode_count:,} | "
                       f"Speed: {steps_per_sec:,.0f} steps/sec | GPU: {gpu_mem:.1f} GB")
            last_log_time = time.time()
    
    elapsed = time.time() - start_time
    logger.info(f"✅ Training complete in {elapsed:.1f}s")
    logger.info(f"   Total steps: {total_steps:,}")
    logger.info(f"   Throughput: {total_steps/elapsed:,.0f} steps/sec")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--n-envs', type=int, default=1024)
    parser.add_argument('--n-steps', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args()
    train_gpu(args)

