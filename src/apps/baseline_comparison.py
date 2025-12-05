#!/usr/bin/env python3
"""
Compare RL policy vs baseline strategies on the multi-asset portfolio environment.

Baselines:
- ALWAYS_CASH: 100% cash (risk-free)
- ALWAYS_SPY: 100% SPY buy-and-hold
- ALWAYS_QQQ: 100% QQQ buy-and-hold
- ALWAYS_IWM: 100% IWM buy-and-hold
- EQUAL_WEIGHT: 25% each (rebalanced)
- DEFENSIVE_50: 50% cash, 25% SPY, 25% IWM

Usage:
    python src/apps/baseline_comparison.py --cache-path data/v1_train_2020_2024/gpu_cache_train.pt
    python src/apps/baseline_comparison.py --cache-path data/v1_train_2020_2024/gpu_cache_train.pt --model checkpoints/clstm_full/model_final_20251204_164205.pt
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.envs.multi_asset_env import MultiAssetEnvironment, PORTFOLIO_REGIMES


class SimplePolicyNetwork(nn.Module):
    """Simplified PPO network (must match train.py)."""
    def __init__(self, obs_dim: int = 64, n_actions: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        encoded = self.encoder(x)
        if hidden is None:
            hidden = torch.zeros(1, x.shape[0], self.hidden_dim, device=x.device)
        gru_out, new_hidden = self.gru(encoded, hidden)
        logits = self.policy_head(gru_out[:, -1, :])
        value = self.value_head(gru_out[:, -1, :]).squeeze(-1)
        return logits, value, new_hidden

    def get_action(self, obs, hidden=None, deterministic=True):
        logits, value, new_hidden = self.forward(obs, hidden)
        probs = torch.softmax(logits, dim=-1)
        action = torch.argmax(probs, dim=-1) if deterministic else torch.distributions.Categorical(probs).sample()
        return action, value, new_hidden


@dataclass
class EvalMetrics:
    """Evaluation metrics for a policy."""
    name: str
    avg_reward: float
    std_reward: float
    sharpe: float
    max_drawdown_pct: float
    total_return_pct: float
    win_rate: float
    turnover: float  # Average actions per episode that changed allocation


def run_baseline(env: MultiAssetEnvironment, action_idx: int, n_episodes: int = 100,
                 episode_length: int = 256) -> EvalMetrics:
    """Run a fixed-action baseline policy."""
    name = PORTFOLIO_REGIMES[action_idx][0]
    all_rewards = []
    all_returns = []
    
    episodes_done = 0
    while episodes_done < n_episodes:
        obs, _ = env.reset()
        episode_rewards = torch.zeros(env.n_envs, device=env.device)
        cumulative_return = torch.zeros(env.n_envs, device=env.device)
        
        for step in range(episode_length):
            actions = torch.full((env.n_envs,), action_idx, dtype=torch.long, device=env.device)
            obs, rewards, terminated, truncated, _ = env.step(actions)
            episode_rewards += rewards
            cumulative_return += rewards / 100.0  # Convert scaled reward to %
            
            done = terminated | truncated
            if done.any():
                done_idx = done.nonzero(as_tuple=True)[0]
                for idx in done_idx:
                    if episodes_done < n_episodes:
                        all_rewards.append(episode_rewards[idx].item())
                        all_returns.append(cumulative_return[idx].item())
                        episodes_done += 1
                        episode_rewards[idx] = 0
                        cumulative_return[idx] = 0
    
    rewards = np.array(all_rewards)
    returns = np.array(all_returns)
    
    # Compute metrics
    sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-8)
    
    # Max drawdown from cumulative returns
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_dd_pct = (drawdowns.max() / (running_max.max() + 1e-8)) * 100
    
    total_return = returns.mean() * 100
    win_rate = (returns > 0).mean() * 100
    
    return EvalMetrics(
        name=name,
        avg_reward=rewards.mean(),
        std_reward=rewards.std(),
        sharpe=sharpe,
        max_drawdown_pct=max_dd_pct,
        total_return_pct=total_return,
        win_rate=win_rate,
        turnover=0.0  # Fixed policy, no turnover
    )


def run_rl_model(env: MultiAssetEnvironment, model_path: str, n_episodes: int = 100,
                  episode_length: int = 256) -> EvalMetrics:
    """Run a trained RL model."""
    # Load model
    checkpoint = torch.load(model_path, map_location=env.device, weights_only=False)
    model = SimplePolicyNetwork(obs_dim=64, n_actions=env.n_actions, hidden_dim=128).to(env.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_rewards = []
    all_returns = []
    all_actions = []

    episodes_done = 0
    with torch.no_grad():
        while episodes_done < n_episodes:
            obs, _ = env.reset()
            hidden = None
            episode_rewards = torch.zeros(env.n_envs, device=env.device)
            cumulative_return = torch.zeros(env.n_envs, device=env.device)

            for step in range(episode_length):
                actions, _, hidden = model.get_action(obs, hidden, deterministic=True)
                all_actions.extend(actions.cpu().tolist())
                obs, rewards, terminated, truncated, _ = env.step(actions)
                episode_rewards += rewards
                cumulative_return += rewards / 100.0

                done = terminated | truncated
                if done.any():
                    done_idx = done.nonzero(as_tuple=True)[0]
                    for idx in done_idx:
                        if episodes_done < n_episodes:
                            all_rewards.append(episode_rewards[idx].item())
                            all_returns.append(cumulative_return[idx].item())
                            episodes_done += 1
                            episode_rewards[idx] = 0
                            cumulative_return[idx] = 0
                            hidden[:, idx, :] = 0  # Reset hidden for this env

    rewards = np.array(all_rewards)
    returns = np.array(all_returns)

    sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-8)
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_dd_pct = (drawdowns.max() / (running_max.max() + 1e-8)) * 100 if running_max.max() > 0 else 0
    total_return = returns.mean() * 100
    win_rate = (returns > 0).mean() * 100

    # Estimate turnover from action changes
    action_changes = sum(1 for i in range(1, len(all_actions)) if all_actions[i] != all_actions[i-1])
    turnover = action_changes / len(all_actions) * 100

    return EvalMetrics(
        name="RL_POLICY",
        avg_reward=rewards.mean(),
        std_reward=rewards.std(),
        sharpe=sharpe,
        max_drawdown_pct=max_dd_pct,
        total_return_pct=total_return,
        win_rate=win_rate,
        turnover=turnover
    )


def main():
    parser = argparse.ArgumentParser(description='Compare baseline policies')
    parser.add_argument('--cache-path', type=str, required=True, help='Path to GPU cache')
    parser.add_argument('--model', type=str, default=None, help='Path to RL model checkpoint')
    parser.add_argument('--episodes', type=int, default=200, help='Episodes per baseline')
    parser.add_argument('--n-envs', type=int, default=256, help='Parallel environments')
    parser.add_argument('--episode-length', type=int, default=256, help='Steps per episode')
    parser.add_argument('--vol-penalty', type=float, default=0.5, help='Volatility penalty')
    parser.add_argument('--dd-penalty', type=float, default=0.2, help='Drawdown penalty')
    parser.add_argument('--trading-cost', type=float, default=0.001, help='Trading cost')
    args = parser.parse_args()

    print("=" * 80)
    print("  BASELINE POLICY COMPARISON")
    print("=" * 80)
    print(f"  Penalties: vol={args.vol_penalty}, dd={args.dd_penalty}, tc={args.trading_cost}")
    print("=" * 80)

    env = MultiAssetEnvironment(
        n_envs=args.n_envs,
        episode_length=args.episode_length,
        device='cuda',
        cache_path=args.cache_path,
        volatility_penalty=args.vol_penalty,
        drawdown_penalty=args.dd_penalty,
        trading_cost=args.trading_cost,
    )

    # Baselines to test (action indices)
    baselines = [
        1,   # ALL_CASH
        2,   # ALL_SPY
        3,   # ALL_QQQ
        4,   # ALL_IWM
        5,   # EQUAL_WEIGHT
        9,   # DEFENSIVE_50
        15,  # BALANCED
    ]

    results: List[EvalMetrics] = []
    for action_idx in baselines:
        print(f"\n📊 Evaluating: {PORTFOLIO_REGIMES[action_idx][0]}...")
        metrics = run_baseline(env, action_idx, args.episodes, args.episode_length)
        results.append(metrics)
        print(f"   Reward: {metrics.avg_reward:.2f} | Sharpe: {metrics.sharpe:.3f} | "
              f"Return: {metrics.total_return_pct:+.2f}%")

    # Evaluate RL model if provided
    if args.model:
        print(f"\n🤖 Evaluating RL model: {args.model}...")
        rl_metrics = run_rl_model(env, args.model, args.episodes, args.episode_length)
        results.append(rl_metrics)
        print(f"   Reward: {rl_metrics.avg_reward:.2f} | Sharpe: {rl_metrics.sharpe:.3f} | "
              f"Return: {rl_metrics.total_return_pct:+.2f}% | Turnover: {rl_metrics.turnover:.1f}%")

    # Sort by Sharpe ratio
    results.sort(key=lambda x: x.sharpe, reverse=True)

    print("\n" + "=" * 80)
    print("  RESULTS (sorted by Sharpe ratio)")
    print("=" * 80)
    print(f"{'Policy':<20} {'Reward':>10} {'Sharpe':>10} {'Return%':>10} {'MaxDD%':>10} {'WinRate':>10} {'Turn%':>8}")
    print("-" * 88)
    for m in results:
        print(f"{m.name:<20} {m.avg_reward:>10.2f} {m.sharpe:>10.3f} "
              f"{m.total_return_pct:>+10.2f} {m.max_drawdown_pct:>10.2f} {m.win_rate:>10.1f}% {m.turnover:>8.1f}")

    print("\n" + "=" * 88)
    best = results[0]
    if args.model and best.name == "RL_POLICY":
        print(f"🏆 RL POLICY is BEST! (Sharpe={best.sharpe:.3f})")
    else:
        print(f"🏆 Best: {best.name} (Sharpe={best.sharpe:.3f})")
        if args.model:
            rl_rank = next(i for i, m in enumerate(results) if m.name == "RL_POLICY") + 1
            print(f"   RL_POLICY ranked #{rl_rank} out of {len(results)}")
    print("=" * 88)


if __name__ == '__main__':
    main()

