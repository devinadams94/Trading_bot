#!/usr/bin/env python3
"""
Standardized RL Policy Evaluation with Baseline Comparison

Evaluates a trained RL model against fixed baseline strategies and produces:
1. JSON report with all metrics
2. Text summary table (stdout)

Usage:
    # Evaluate with default config
    python src/apps/eval_with_baselines.py \
        --model checkpoints/clstm_full/model_final_XXXXXXXX.pt \
        --cache-path data/v2_test_2020_2024/gpu_cache_test.pt

    # With config file
    python src/apps/eval_with_baselines.py \
        --config configs/rl_v2_multi_asset.yaml \
        --model checkpoints/clstm_full/model_final_XXXXXXXX.pt

    # Save JSON report
    python src/apps/eval_with_baselines.py \
        --model checkpoints/... --cache-path data/... \
        --output reports/eval_20241204.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import numpy as np

from src.envs.multi_asset_env import MultiAssetEnvironment, PORTFOLIO_REGIMES


# ==============================================================================
# MODEL DEFINITION (must match train.py)
# ==============================================================================
class SimplePolicyNetwork(nn.Module):
    """Simplified PPO network (v2 baseline architecture)."""
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


# ==============================================================================
# METRICS
# ==============================================================================
@dataclass
class PolicyMetrics:
    """Evaluation metrics for a policy."""
    name: str
    avg_reward: float
    std_reward: float
    sharpe: float
    total_return_pct: float
    max_drawdown_pct: float
    win_rate: float
    turnover_pct: float
    n_episodes: int


def compute_metrics(name: str, rewards: np.ndarray, returns: np.ndarray,
                    turnover: float = 0.0) -> PolicyMetrics:
    """Compute standard metrics from episode data."""
    sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-8)
    
    # Max drawdown
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_dd = (drawdowns.max() / (running_max.max() + 1e-8)) * 100 if running_max.max() > 0 else 0
    
    return PolicyMetrics(
        name=name,
        avg_reward=float(rewards.mean()),
        std_reward=float(rewards.std()),
        sharpe=float(sharpe),
        total_return_pct=float(returns.mean() * 100),
        max_drawdown_pct=float(max_dd),
        win_rate=float((returns > 0).mean() * 100),
        turnover_pct=float(turnover),
        n_episodes=len(rewards)
    )


# ==============================================================================
# EVALUATION FUNCTIONS
# ==============================================================================
def run_baseline(env: MultiAssetEnvironment, action_idx: int,
                 n_episodes: int, episode_length: int) -> PolicyMetrics:
    """Run a fixed-action baseline policy."""
    name = PORTFOLIO_REGIMES[action_idx][0]
    all_rewards, all_returns = [], []
    episodes_done = 0
    
    while episodes_done < n_episodes:
        obs, _ = env.reset()
        ep_rewards = torch.zeros(env.n_envs, device=env.device)
        ep_returns = torch.zeros(env.n_envs, device=env.device)
        
        for _ in range(episode_length):
            actions = torch.full((env.n_envs,), action_idx, dtype=torch.long, device=env.device)
            obs, rewards, terminated, truncated, _ = env.step(actions)
            ep_rewards += rewards
            ep_returns += rewards / 100.0
            
            done = terminated | truncated
            if done.any():
                for idx in done.nonzero(as_tuple=True)[0]:
                    if episodes_done < n_episodes:
                        all_rewards.append(ep_rewards[idx].item())
                        all_returns.append(ep_returns[idx].item())
                        episodes_done += 1
                        ep_rewards[idx] = 0
                        ep_returns[idx] = 0
    
    return compute_metrics(name, np.array(all_rewards), np.array(all_returns), turnover=0.0)


def run_rl_policy(env: MultiAssetEnvironment, model_path: str,
                  n_episodes: int, episode_length: int) -> PolicyMetrics:
    """Run a trained RL model."""
    checkpoint = torch.load(model_path, map_location=env.device, weights_only=False)
    model = SimplePolicyNetwork(obs_dim=64, n_actions=env.n_actions, hidden_dim=128).to(env.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_rewards, all_returns, all_actions = [], [], []
    episodes_done = 0

    with torch.no_grad():
        while episodes_done < n_episodes:
            obs, _ = env.reset()
            hidden = None
            ep_rewards = torch.zeros(env.n_envs, device=env.device)
            ep_returns = torch.zeros(env.n_envs, device=env.device)

            for _ in range(episode_length):
                actions, _, hidden = model.get_action(obs, hidden, deterministic=True)
                all_actions.extend(actions.cpu().tolist())
                obs, rewards, terminated, truncated, _ = env.step(actions)
                ep_rewards += rewards
                ep_returns += rewards / 100.0

                done = terminated | truncated
                if done.any():
                    for idx in done.nonzero(as_tuple=True)[0]:
                        if episodes_done < n_episodes:
                            all_rewards.append(ep_rewards[idx].item())
                            all_returns.append(ep_returns[idx].item())
                            episodes_done += 1
                            ep_rewards[idx] = 0
                            ep_returns[idx] = 0
                            hidden[:, idx, :] = 0

    # Calculate turnover from action changes
    action_changes = sum(1 for i in range(1, len(all_actions)) if all_actions[i] != all_actions[i-1])
    turnover = (action_changes / len(all_actions) * 100) if all_actions else 0.0

    return compute_metrics("RL_POLICY", np.array(all_rewards), np.array(all_returns), turnover)


# ==============================================================================
# MAIN EVALUATION
# ==============================================================================
def evaluate(model_path: str, cache_path: str, n_episodes: int = 500,
             n_envs: int = 256, episode_length: int = 256,
             trading_cost: float = 0.001, output_path: Optional[str] = None) -> Dict:
    """Run full evaluation with baselines and produce report."""

    print("=" * 80)
    print("  MULTI-ASSET RL EVALUATION WITH BASELINES")
    print("=" * 80)
    print(f"  Model:        {model_path}")
    print(f"  Data:         {cache_path}")
    print(f"  Episodes:     {n_episodes}")
    print(f"  Trading cost: {trading_cost*100:.2f}%")
    print("=" * 80)

    env = MultiAssetEnvironment(
        n_envs=n_envs,
        episode_length=episode_length,
        device='cuda',
        cache_path=cache_path,
        trading_cost=trading_cost,
    )

    # Baseline action indices to test
    baseline_actions = {
        "ALL_CASH": 1,
        "ALL_SPY": 2,
        "ALL_QQQ": 3,
        "ALL_IWM": 4,
        "EQUAL_WEIGHT": 5,
        "DEFENSIVE_50": 9,
        "BALANCED": 15,
    }

    results: List[PolicyMetrics] = []

    # Run baselines
    for name, action_idx in baseline_actions.items():
        print(f"\n📊 Evaluating baseline: {name}...")
        metrics = run_baseline(env, action_idx, n_episodes, episode_length)
        results.append(metrics)
        print(f"   Sharpe: {metrics.sharpe:.3f} | Return: {metrics.total_return_pct:+.2f}%")

    # Run RL model
    print(f"\n🤖 Evaluating RL policy: {model_path}...")
    rl_metrics = run_rl_policy(env, model_path, n_episodes, episode_length)
    results.append(rl_metrics)
    print(f"   Sharpe: {rl_metrics.sharpe:.3f} | Return: {rl_metrics.total_return_pct:+.2f}% | Turnover: {rl_metrics.turnover_pct:.1f}%")

    # Sort by Sharpe
    results.sort(key=lambda x: x.sharpe, reverse=True)
    rl_rank = next(i+1 for i, m in enumerate(results) if m.name == "RL_POLICY")

    # Print results table
    print("\n" + "=" * 100)
    print("  RESULTS (sorted by Sharpe ratio)")
    print("=" * 100)
    print(f"{'Rank':<6}{'Policy':<18}{'Reward':>10}{'Sharpe':>10}{'Return%':>10}{'MaxDD%':>10}{'WinRate':>10}{'Turn%':>8}")
    print("-" * 100)
    for rank, m in enumerate(results, 1):
        marker = "🏆" if rank == 1 else ("🤖" if m.name == "RL_POLICY" else "  ")
        print(f"{marker}{rank:<4}{m.name:<18}{m.avg_reward:>10.2f}{m.sharpe:>10.3f}"
              f"{m.total_return_pct:>+10.2f}{m.max_drawdown_pct:>10.2f}{m.win_rate:>10.1f}%{m.turnover_pct:>8.1f}")
    print("=" * 100)
    print(f"\n📌 RL_POLICY ranked #{rl_rank} out of {len(results)}")

    # Build report
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "dataset": cache_path,
        "n_episodes": n_episodes,
        "trading_cost": trading_cost,
        "rl_rank": rl_rank,
        "policies": {m.name: asdict(m) for m in results}
    }

    # Save JSON report
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Report saved to: {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description='Evaluate RL policy vs baselines')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--cache-path', type=str, required=True, help='Path to GPU cache')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config (optional)')
    parser.add_argument('--episodes', type=int, default=500, help='Episodes per policy')
    parser.add_argument('--n-envs', type=int, default=256, help='Parallel environments')
    parser.add_argument('--episode-length', type=int, default=256, help='Steps per episode')
    parser.add_argument('--trading-cost', type=float, default=0.001, help='Trading cost (0.001 = 0.1%)')
    parser.add_argument('--output', '-o', type=str, default=None, help='Path to save JSON report')
    args = parser.parse_args()

    # Load config if provided
    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        env_cfg = config.get('env', {})
        args.trading_cost = env_cfg.get('trading_cost', args.trading_cost)
        args.n_envs = env_cfg.get('n_envs', args.n_envs)
        data_cfg = config.get('data', {})
        if not args.cache_path:
            args.cache_path = data_cfg.get('test_cache_path')

    evaluate(
        model_path=args.model,
        cache_path=args.cache_path,
        n_episodes=args.episodes,
        n_envs=args.n_envs,
        episode_length=args.episode_length,
        trading_cost=args.trading_cost,
        output_path=args.output
    )


if __name__ == '__main__':
    main()

