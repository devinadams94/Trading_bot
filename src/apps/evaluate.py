#!/usr/bin/env python3
"""
Evaluate a frozen CLSTM-PPO model on out-of-sample data.

Usage:
    python src/apps/evaluate.py --model-dir models/v1_clstm_sep_trunks_100M --data-dir data/v1_eval_2015_2019

    # With more episodes
    python src/apps/evaluate.py --model-dir models/v1_clstm_sep_trunks_100M --data-dir data/v1_eval_2015_2019 --episodes 1000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import numpy as np


import numpy as np


class SimplePolicyNetwork(nn.Module):
    """
    Simplified PPO network for stable training (Phase 0).
    Matches the architecture in train.py.
    """
    def __init__(self, obs_dim: int = 64, n_actions: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Single shared GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Simple heads
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch_size = x.shape[0]

        encoded = self.encoder(x)

        if hidden is None:
            hidden = torch.zeros(1, batch_size, self.hidden_dim, device=x.device)

        gru_out, new_hidden = self.gru(encoded, hidden)
        features = gru_out[:, -1, :]

        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)

        return logits, value, new_hidden

    def get_action(self, obs, hidden=None, deterministic=False):
        logits, value, new_hidden = self.forward(obs, hidden)
        probs = torch.softmax(logits, dim=-1)
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = torch.distributions.Categorical(probs).sample()
        return action, value, new_hidden


def load_model(model_dir: str, device: str = 'cuda') -> SimplePolicyNetwork:
    """Load a frozen model from a model directory."""
    config_path = os.path.join(model_dir, "inference_config.json")
    model_path = os.path.join(model_dir, "model.pt")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    model = SimplePolicyNetwork(
        obs_dim=cfg.get("obs_dim", 64),
        n_actions=cfg.get("n_actions", 16),
        hidden_dim=cfg.get("hidden_dim", 128),
    ).to(device)

    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    return model, cfg


def find_cache_file(data_dir: str) -> str:
    """Find the GPU cache file in a data directory."""
    data_path = Path(data_dir)
    cache_files = list(data_path.glob('*.pt'))
    if not cache_files:
        raise FileNotFoundError(f"No .pt cache files found in {data_dir}")
    if len(cache_files) > 1:
        print(f"Warning: Multiple cache files found, using: {cache_files[0].name}")
    return str(cache_files[0])


def evaluate(model_dir: str, data_dir: str, n_episodes: int = 100, n_envs: int = 64,
             episode_length: int = 256, deterministic: bool = True, device: str = 'cuda'):
    """Run evaluation and compute financial metrics."""
    
    print("=" * 70)
    print("  CLSTM-PPO MODEL EVALUATION")
    print("=" * 70)
    print(f"  Model:      {model_dir}")
    print(f"  Data:       {data_dir}")
    print(f"  Episodes:   {n_episodes}")
    print(f"  Envs:       {n_envs}")
    print(f"  Mode:       {'Deterministic' if deterministic else 'Stochastic'}")
    print("=" * 70)
    
    # Load model
    print("\n📦 Loading model...")
    model, cfg = load_model(model_dir, device)
    print(f"   ✅ Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Find cache file
    cache_path = find_cache_file(data_dir)
    print(f"\n📊 Loading data from: {cache_path}")

    # Create Multi-Asset Portfolio environment
    from src.envs.multi_asset_env import MultiAssetEnvironment, print_regime_info
    print_regime_info()

    env = MultiAssetEnvironment(
        n_envs=n_envs,
        episode_length=episode_length,
        device=device,
        cache_path=cache_path,
        volatility_penalty=0.5,
        drawdown_penalty=0.2,
        trading_cost=0.001,
    )
    n_actions = env.n_actions
    print(f"   ✅ Environment ready with {env.n_days} trading days, {n_actions} actions")
    
    # Tracking metrics
    all_episode_returns = []
    all_episode_pnls = []
    all_final_values = []
    all_actions = []  # Track action distribution
    episode_count = 0

    print(f"\n🚀 Running evaluation...")
    start_time = time.time()

    with torch.no_grad():
        while episode_count < n_episodes:
            obs, _ = env.reset()
            hidden = None
            episode_rewards = torch.zeros(n_envs, device=device)
            # Track cumulative return percentage per env (rewards are scaled % returns)
            cumulative_return_pct = torch.zeros(n_envs, device=device)

            for step in range(episode_length):
                actions, _, hidden = model.get_action(obs, hidden, deterministic=deterministic)
                all_actions.extend(actions.cpu().tolist())
                obs, rewards, terminated, truncated, _ = env.step(actions)
                episode_rewards += rewards
                # rewards are raw_returns * 100, so divide by 100 to get actual %
                cumulative_return_pct += rewards / 100.0

                done = terminated | truncated
                if done.any():
                    done_idx = done.nonzero(as_tuple=True)[0]
                    for idx in done_idx:
                        if episode_count < n_episodes:
                            ret = episode_rewards[idx].item()
                            # Calculate PnL from cumulative return percentage
                            pct_return = cumulative_return_pct[idx].item()
                            pnl = env.initial_capital * pct_return
                            final_val = env.initial_capital + pnl

                            all_episode_returns.append(ret)
                            all_episode_pnls.append(pnl)
                            all_final_values.append(final_val)
                            episode_count += 1

                            # Reset tracking for this env
                            episode_rewards[idx] = 0
                            cumulative_return_pct[idx] = 0

                            if episode_count % 50 == 0:
                                print(f"   Episodes: {episode_count}/{n_episodes}")
    
    elapsed = time.time() - start_time
    
    # Compute metrics
    returns = np.array(all_episode_returns)
    pnls = np.array(all_episode_pnls)
    final_values = np.array(all_final_values)
    
    # Financial metrics
    initial_capital = env.initial_capital
    total_return_pct = ((final_values.mean() - initial_capital) / initial_capital) * 100
    
    # Sharpe ratio (annualized, assuming 252 trading days)
    daily_returns = pnls / initial_capital
    sharpe = np.sqrt(252) * daily_returns.mean() / (daily_returns.std() + 1e-8)
    
    # Max drawdown
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = drawdowns.max()
    max_drawdown_pct = (max_drawdown / initial_capital) * 100
    
    # Win rate
    win_rate = (pnls > 0).mean() * 100
    
    # Profit factor
    gross_profit = pnls[pnls > 0].sum() if (pnls > 0).any() else 0
    gross_loss = abs(pnls[pnls < 0].sum()) if (pnls < 0).any() else 1e-8
    profit_factor = gross_profit / gross_loss
    
    # Action distribution analysis
    action_counts = np.bincount(all_actions, minlength=n_actions)
    action_probs = action_counts / action_counts.sum()
    top_actions = action_probs.argsort()[::-1][:5]  # Top 5 most used

    # Print results
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)
    print(f"\n📊 Episode Statistics:")
    print(f"   Episodes evaluated:     {len(returns)}")
    print(f"   Avg episode reward:     {returns.mean():.2f} ± {returns.std():.2f}")
    print(f"   Min/Max reward:         {returns.min():.2f} / {returns.max():.2f}")

    print(f"\n💰 Financial Metrics:")
    print(f"   Initial capital:        ${initial_capital:,.0f}")
    print(f"   Avg final value:        ${final_values.mean():,.0f}")
    print(f"   Total return:           {total_return_pct:+.2f}%")
    print(f"   Avg PnL per episode:    ${pnls.mean():,.2f}")

    print(f"\n📈 Risk Metrics:")
    print(f"   Sharpe ratio (ann.):    {sharpe:.3f}")
    print(f"   Max drawdown:           ${max_drawdown:,.0f} ({max_drawdown_pct:.2f}%)")
    print(f"   Win rate:               {win_rate:.1f}%")
    print(f"   Profit factor:          {profit_factor:.2f}")

    print(f"\n🎯 Action Distribution (Top 5):")
    for action_idx in top_actions:
        regime_name = env.portfolio_regimes[action_idx][0]
        pct = action_probs[action_idx] * 100
        print(f"   {regime_name:20s}: {pct:5.1f}%")

    # Check for policy collapse
    if action_probs.max() > 0.8:
        print(f"\n⚠️  WARNING: Policy collapsed to single action ({action_probs.max()*100:.1f}% {env.portfolio_regimes[action_probs.argmax()][0]})")

    print(f"\n⏱️  Evaluation time:        {elapsed:.1f}s ({len(returns)/elapsed:.1f} episodes/sec)")
    print("=" * 70)
    
    # Return metrics dict
    return {
        'n_episodes': len(returns),
        'avg_reward': float(returns.mean()),
        'std_reward': float(returns.std()),
        'avg_pnl': float(pnls.mean()),
        'total_return_pct': float(total_return_pct),
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(max_drawdown),
        'max_drawdown_pct': float(max_drawdown_pct),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate a frozen CLSTM-PPO model')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Path to model directory (e.g., models/v1_clstm_sep_trunks_100M)')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to data directory with GPU cache (e.g., data/v1_eval_2015_2019)')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to evaluate (default: 100)')
    parser.add_argument('--n-envs', type=int, default=64,
                        help='Number of parallel environments (default: 64)')
    parser.add_argument('--episode-length', type=int, default=256,
                        help='Steps per episode (default: 256)')
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic actions instead of deterministic')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--save-results', type=str, default=None,
                        help='Path to save results JSON')
    args = parser.parse_args()
    
    metrics = evaluate(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        n_episodes=args.episodes,
        n_envs=args.n_envs,
        episode_length=args.episode_length,
        deterministic=not args.stochastic,
        device=args.device
    )
    
    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n💾 Results saved to: {args.save_results}")


if __name__ == '__main__':
    main()

