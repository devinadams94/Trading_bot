#!/usr/bin/env python3
"""
v2 Multi-Asset Baseline Training Script

================================================================================
CONFIGURATION SUMMARY (v2 Multi-Asset Baseline)
================================================================================
- Env: MultiAssetEnvironment over [CASH, SPY, QQQ, IWM]
- Actions: 16 discrete portfolio regimes (HOLD, ALL_SPY, ALL_QQQ, etc.)
- Model: SimplePolicyNetwork (GRU, ~110k params)
  - Encoder: Linear(64→128) + ReLU + LayerNorm
  - GRU: 1 layer, hidden_size=128 (shared actor+critic)
  - Policy Head: Linear(128→n_actions)
  - Value Head: Linear(128→1)
- Reward: alpha_vs_equal_weight_equity - trading_cost
  - alpha = portfolio_return - benchmark_return (EW of SPY/QQQ/IWM)
  - No vol/dd penalties in training (evaluated at test time)
- Train data: data/v2_train_2015_2019/gpu_cache_train.pt
- Test data:  data/v2_test_2020_2024/gpu_cache_test.pt

Out-of-sample results (model trained on 2015-2019, tested on 2020-2024):
- RL policy ranks #2 (after ALL_QQQ), beats ALL_SPY on Sharpe
- Very low turnover (~0.3%) - learned to hold optimal regimes
================================================================================

Production training script with comprehensive logging and monitoring.
Uses the GPU-accelerated environment with pre-cached historical data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
import time
import argparse
import os
import sys
from datetime import datetime
from collections import deque

# Rich console for pretty logging
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Setup logging
log_dir = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)
os.makedirs("checkpoints/clstm_full", exist_ok=True)

# File + console logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(f"{log_dir}/training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

console = Console() if RICH_AVAILABLE else None


def apply_gpu_optimizations():
    """Apply all GPU optimizations for H200"""
    if torch.cuda.is_available():
        cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        logger.info("✅ GPU optimizations applied (TF32, cuDNN benchmark)")


class TrainingMetrics:
    """Track and log training metrics with PPO diagnostics"""
    def __init__(self, window_size: int = 100):
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.policy_losses = deque(maxlen=window_size)
        self.value_losses = deque(maxlen=window_size)
        self.entropies = deque(maxlen=window_size)
        self.learning_rates = deque(maxlen=window_size)
        self.steps_per_sec = deque(maxlen=window_size)

        # PPO Diagnostic metrics
        self.kl_divergences = deque(maxlen=window_size)
        self.clip_fractions = deque(maxlen=window_size)
        self.explained_variances = deque(maxlen=window_size)
        self.policy_grad_norms = deque(maxlen=window_size)
        self.value_grad_norms = deque(maxlen=window_size)
        self.approx_kls = deque(maxlen=window_size)

        # Track entropy decay for early warning
        self.initial_entropy = None
        self.entropy_history = deque(maxlen=1000)

        self.total_steps = 0
        self.total_episodes = 0
        self.best_reward = float('-inf')
        self.start_time = time.time()

        # Warning flags
        self.warnings = []

    def update(self, rewards, lengths, policy_loss, value_loss, entropy, lr, sps,
               kl_div=0, clip_frac=0, explained_var=0, policy_grad_norm=0,
               value_grad_norm=0, approx_kl=0):
        self.episode_rewards.extend(rewards)
        self.episode_lengths.extend(lengths)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropies.append(entropy)
        self.learning_rates.append(lr)
        self.steps_per_sec.append(sps)

        # PPO diagnostics
        self.kl_divergences.append(kl_div)
        self.clip_fractions.append(clip_frac)
        self.explained_variances.append(explained_var)
        self.policy_grad_norms.append(policy_grad_norm)
        self.value_grad_norms.append(value_grad_norm)
        self.approx_kls.append(approx_kl)

        self.total_episodes += len(rewards)

        # Track initial entropy
        if self.initial_entropy is None:
            self.initial_entropy = entropy
        self.entropy_history.append(entropy)

        avg_reward = np.mean(rewards) if rewards else 0
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward

        # Check for warning conditions
        self._check_warnings()

    def _check_warnings(self):
        """Check for training issues"""
        self.warnings = []

        # KL divergence too high (policy changing too fast)
        if self.approx_kls and np.mean(list(self.approx_kls)[-10:]) > 0.05:
            self.warnings.append("⚠️  HIGH KL: Policy changing too fast, reduce LR")

        # Entropy collapsed (premature convergence)
        if self.initial_entropy and self.entropies:
            entropy_ratio = np.mean(list(self.entropies)[-10:]) / self.initial_entropy
            if entropy_ratio < 0.1:
                self.warnings.append("⚠️  LOW ENTROPY: Premature convergence, increase entropy_coef")

        # Clip fraction too high
        if self.clip_fractions and np.mean(list(self.clip_fractions)[-10:]) > 0.3:
            self.warnings.append("⚠️  HIGH CLIP: Too many clipped updates, reduce LR")

        # Explained variance too low (value function not learning)
        if self.explained_variances and np.mean(list(self.explained_variances)[-10:]) < 0:
            self.warnings.append("⚠️  NEG EXP_VAR: Value function worse than baseline")

        # Gradient explosion
        if self.policy_grad_norms and np.mean(list(self.policy_grad_norms)[-10:]) > 10:
            self.warnings.append("⚠️  HIGH GRAD: Gradient explosion, reduce LR")

    def get_stats(self):
        return {
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'avg_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'policy_loss': np.mean(self.policy_losses) if self.policy_losses else 0,
            'value_loss': np.mean(self.value_losses) if self.value_losses else 0,
            'entropy': np.mean(self.entropies) if self.entropies else 0,
            'lr': self.learning_rates[-1] if self.learning_rates else 0,
            'sps': np.mean(self.steps_per_sec) if self.steps_per_sec else 0,
            'best_reward': self.best_reward,
            'total_episodes': self.total_episodes,
            'elapsed': time.time() - self.start_time,
            # PPO diagnostics
            'kl_div': np.mean(self.kl_divergences) if self.kl_divergences else 0,
            'approx_kl': np.mean(self.approx_kls) if self.approx_kls else 0,
            'clip_frac': np.mean(self.clip_fractions) if self.clip_fractions else 0,
            'explained_var': np.mean(self.explained_variances) if self.explained_variances else 0,
            'policy_grad_norm': np.mean(self.policy_grad_norms) if self.policy_grad_norms else 0,
            'value_grad_norm': np.mean(self.value_grad_norms) if self.value_grad_norms else 0,
            'entropy_ratio': (np.mean(list(self.entropies)[-10:]) / self.initial_entropy) if self.initial_entropy and self.entropies else 1.0,
            'warnings': self.warnings
        }


class SimplePolicyNetwork(nn.Module):
    """
    Simplified PPO network for stable training.

    Architecture (ChatGPT Phase 0 recommendation):
        Shared Encoder: Linear(64 → 128) + ReLU + LayerNorm
        Shared GRU: 1 layer, hidden=128
        Policy Head: 128 → n_actions
        Value Head: 128 → 1

    Much smaller than CLSTM (~50K params vs 3.8M) - easier to train.
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

        # Single shared GRU (simpler than LSTM, 1 layer only)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Simple policy head (no extra hidden layer)
        self.policy_head = nn.Linear(hidden_dim, n_actions)

        # Simple value head (no extra hidden layer)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization (standard for PPO)"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.orthogonal_(param, gain=np.sqrt(2))
            elif 'bias' in name:
                nn.init.zeros_(param)
        # Policy head: small init for exploration
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        # Value head: standard init
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, x, hidden=None):
        """
        Args:
            x: [batch, obs_dim] or [batch, seq, obs_dim]
            hidden: GRU hidden state [1, batch, hidden_dim] or None
        Returns:
            logits: [batch, n_actions]
            value: [batch]
            new_hidden: [1, batch, hidden_dim]
        """
        # Handle 2D input (no sequence dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, obs_dim]

        batch_size = x.shape[0]

        # Encode
        encoded = self.encoder(x)  # [batch, seq, hidden]

        # Initialize hidden if needed
        if hidden is None:
            hidden = torch.zeros(1, batch_size, self.hidden_dim, device=x.device)

        # GRU forward
        gru_out, new_hidden = self.gru(encoded, hidden)

        # Use last timestep
        features = gru_out[:, -1, :]  # [batch, hidden]

        # Heads
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)

        return logits, value, new_hidden

    def get_action(self, obs, hidden=None):
        """Sample action from policy"""
        logits, value, new_hidden = self.forward(obs, hidden)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, new_hidden, logits


def print_header():
    """Print training header"""
    header = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    CLSTM-PPO OPTIONS TRADING - FULL TRAINING                 ║
║                         Using Massive.io Historical Data                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(header)


def print_status(metrics: TrainingMetrics, total_steps: int, target_steps: int, gpu_mem: float):
    """Print live training status with PPO diagnostics (console only)"""
    stats = metrics.get_stats()
    progress = 100 * total_steps / target_steps

    # Color coding for diagnostics
    def color_kl(kl):
        if kl > 0.05: return f"\033[91m{kl:.4f}\033[0m"  # Red - too high
        elif kl > 0.02: return f"\033[93m{kl:.4f}\033[0m"  # Yellow - warning
        else: return f"\033[92m{kl:.4f}\033[0m"  # Green - good

    def color_clip(clip):
        if clip > 0.3: return f"\033[91m{clip:.1%}\033[0m"  # Red
        elif clip > 0.2: return f"\033[93m{clip:.1%}\033[0m"  # Yellow
        else: return f"\033[92m{clip:.1%}\033[0m"  # Green

    def color_expvar(ev):
        if ev < 0: return f"\033[91m{ev:.3f}\033[0m"  # Red - very bad
        elif ev < 0.5: return f"\033[93m{ev:.3f}\033[0m"  # Yellow
        else: return f"\033[92m{ev:.3f}\033[0m"  # Green

    def color_entropy(ratio):
        if ratio < 0.1: return f"\033[91m{ratio:.1%}\033[0m"  # Red - collapsed
        elif ratio < 0.3: return f"\033[93m{ratio:.1%}\033[0m"  # Yellow
        else: return f"\033[92m{ratio:.1%}\033[0m"  # Green

    status = f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│ TRAINING PROGRESS: {progress:5.1f}% │ Steps: {total_steps:,}/{target_steps:,}
├──────────────────────────────────────────────────────────────────────────────┤
│ 📊 PERFORMANCE                                                                │
│    Avg Reward:    {stats['avg_reward']:+8.2f}   │  Best Reward: {stats['best_reward']:+8.2f}          │
│    Episodes:      {stats['total_episodes']:8,}   │  Avg Length:  {stats['avg_length']:8.1f}          │
├──────────────────────────────────────────────────────────────────────────────┤
│ 🔧 PPO LOSSES                                                                 │
│    Policy Loss:   {stats['policy_loss']:8.4f}   │  Value Loss:  {stats['value_loss']:8.4f}          │
│    Entropy:       {stats['entropy']:8.4f}   │  Learn Rate:  {stats['lr']:8.6f}          │
├──────────────────────────────────────────────────────────────────────────────┤
│ 🔬 PPO DIAGNOSTICS (health indicators)                                        │
│    Approx KL:     {color_kl(stats['approx_kl']):>17}   │  Target: < 0.02                   │
│    Clip Frac:     {color_clip(stats['clip_frac']):>17}   │  Target: 10-20%                   │
│    Expl Var:      {color_expvar(stats['explained_var']):>17}   │  Target: > 0.5                    │
│    Entropy %:     {color_entropy(stats['entropy_ratio']):>17}   │  (of initial)                     │
│    Policy Grad:   {stats['policy_grad_norm']:8.4f}   │  Value Grad:  {stats['value_grad_norm']:8.4f}          │
├──────────────────────────────────────────────────────────────────────────────┤
│ ⚡ SPEED                                                                       │
│    Steps/sec:     {stats['sps']:8,.0f}   │  GPU Memory:  {gpu_mem:8.2f} GB         │
│    Elapsed:       {stats['elapsed']/60:8.1f} min  │  ETA: {(target_steps-total_steps)/(stats['sps']+1)/60:8.1f} min         │
└──────────────────────────────────────────────────────────────────────────────┘
"""

    # Add warnings if any
    if stats['warnings']:
        status += "\n┌── ⚠️  WARNINGS ──────────────────────────────────────────────────────────────┐\n"
        for w in stats['warnings']:
            status += f"│ {w:<76} │\n"
        status += "└──────────────────────────────────────────────────────────────────────────────┘\n"

    # Clear and print
    print("\033[2J\033[H", end="")  # Clear screen
    print_header()
    print(status)


def train(args):
    """Main training function"""
    print_header()
    apply_gpu_optimizations()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️  Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # TensorBoard
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"📊 TensorBoard logs: {log_dir}")

    # Import Multi-Asset Portfolio Environment
    from src.envs.multi_asset_env import MultiAssetEnvironment, print_regime_info

    # Create Multi-Asset GPU environment
    cache_path = args.cache_path if hasattr(args, 'cache_path') and args.cache_path else None
    logger.info(f"🚀 Creating Multi-Asset Portfolio environment with {args.n_envs} parallel instances...")
    if cache_path:
        logger.info(f"   Using cache: {cache_path}")

    # Print regime info
    print_regime_info()

    env = MultiAssetEnvironment(
        n_envs=args.n_envs,
        episode_length=256,
        device='cuda',
        cache_path=cache_path,
        volatility_penalty=0.5,   # Penalize volatile returns
        drawdown_penalty=0.2,     # Penalize drawdowns
        trading_cost=0.001,       # 0.1% per rebalance
    )

    # Get number of actions from environment
    n_actions = env.n_actions
    logger.info(f"   Actions: {n_actions} portfolio regimes")

    # Create SIMPLIFIED policy network (Phase 0: small, stable, boring)
    policy = SimplePolicyNetwork(
        obs_dim=64,
        n_actions=n_actions,
        hidden_dim=args.hidden_dim,  # 128 by default now
    ).to(device)

    # Compile for speed
    if args.compile:
        logger.info("⚡ Compiling model with torch.compile...")
        policy = torch.compile(policy, mode='reduce-overhead')

    # Single optimizer with standard PPO LR
    all_params = list(policy.parameters())
    lr = args.lr  # 3e-4 (standard PPO LR)

    optimizer = optim.AdamW(all_params, lr=lr, weight_decay=1e-4)

    n_iterations = args.timesteps // (args.n_envs * args.n_steps)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iterations)

    logger.info(f"📈 Single optimizer: LR={lr:.0e} for all params")
    logger.info(f"🤖 Simple GRU Policy Network: {sum(p.numel() for p in policy.parameters()):,} parameters")

    # ========== NO CRITIC PRE-TRAINING ==========
    # ChatGPT recommendation: Skip pretraining so advantages are larger initially
    # This lets the policy learn before the critic becomes too good
    logger.info("⚡ Skipping critic pre-training (allows larger advantages initially)")

    # Metrics tracking
    metrics = TrainingMetrics(window_size=100)

    # Training loop
    logger.info(f"🎯 Training for {args.timesteps:,} timesteps...")
    logger.info(f"   Batch size: {args.batch_size:,}")
    logger.info(f"   N steps: {args.n_steps}")
    logger.info(f"   N envs: {args.n_envs}")

    # Pre-allocate rollout buffers on GPU
    obs_buffer = torch.zeros((args.n_steps, args.n_envs, 64), device=device)
    action_buffer = torch.zeros((args.n_steps, args.n_envs), device=device, dtype=torch.int64)
    reward_buffer = torch.zeros((args.n_steps, args.n_envs), device=device)
    log_prob_buffer = torch.zeros((args.n_steps, args.n_envs), device=device)
    value_buffer = torch.zeros((args.n_steps, args.n_envs), device=device)
    done_buffer = torch.zeros((args.n_steps, args.n_envs), device=device, dtype=torch.bool)
    logits_buffer = torch.zeros((args.n_steps, args.n_envs, n_actions), device=device)

    # Initialize
    obs, _ = env.reset()
    hidden = None
    total_steps = 0
    iteration = 0
    last_save_time = time.time()
    last_log_time = time.time()

    episode_rewards_batch = []
    episode_lengths_batch = []

    # Entropy coefficient with linear decay
    entropy_coef_start = args.entropy_coef
    entropy_coef_end = getattr(args, 'entropy_coef_final', 0.005)

    while total_steps < args.timesteps:
        iteration += 1
        iter_start = time.time()

        # Collect rollout
        with torch.no_grad():
            for step in range(args.n_steps):
                obs_buffer[step] = obs
                action, log_prob, value, hidden, logits = policy.get_action(obs, hidden)
                action_buffer[step] = action
                log_prob_buffer[step] = log_prob
                value_buffer[step] = value
                logits_buffer[step] = logits

                obs, reward, terminated, truncated, _ = env.step(action)
                reward_buffer[step] = reward * args.reward_scale  # Scale rewards to O(1-10)
                done_buffer[step] = terminated | truncated

                # Track episode completions (approximate)
                done_count = done_buffer[step].sum().item()
                if done_count > 0:
                    episode_rewards_batch.extend(
                        [reward_buffer[:step+1, i].sum().item() for i in range(int(done_count))]
                    )
                    episode_lengths_batch.extend([step + 1] * int(done_count))

                # Reset hidden state for done environments (GRU uses single tensor)
                if done_buffer[step].any():
                    done_idx = done_buffer[step].nonzero(as_tuple=True)[0]
                    if hidden is not None:
                        hidden[:, done_idx, :] = 0

        steps_collected = args.n_steps * args.n_envs
        total_steps += steps_collected
        metrics.total_steps = total_steps

        # Compute returns and advantages (MC)
        with torch.no_grad():
            mc_returns = torch.zeros_like(reward_buffer)
            for t in reversed(range(args.n_steps)):
                if t == args.n_steps - 1:
                    mc_returns[t] = reward_buffer[t]
                else:
                    not_done = (~done_buffer[t]).float()
                    mc_returns[t] = reward_buffer[t] + args.gamma * mc_returns[t + 1] * not_done

        # Flatten for training
        flat_obs = obs_buffer.reshape(-1, 64)
        flat_actions = action_buffer.reshape(-1)
        flat_log_probs = log_prob_buffer.reshape(-1)
        flat_returns = mc_returns.reshape(-1)
        flat_old_logits = logits_buffer.reshape(-1, n_actions)

        # MC advantages: A_t = R_t - V(s_t) using CURRENT policy values
        with torch.no_grad():
            flat_values = []
            batch_size_val = 8192
            for start in range(0, flat_obs.shape[0], batch_size_val):
                end = min(start + batch_size_val, flat_obs.shape[0])
                _, v, _ = policy(flat_obs[start:end])
                flat_values.append(v)
            flat_values = torch.cat(flat_values, dim=0)

            flat_advantages = flat_returns - flat_values
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        # DEBUG: Sanity checks — throttled
        if iteration <= 3 or iteration % 50 == 0 or total_steps >= args.timesteps:
            logger.info(
                "DEBUG rollout: rewards(min=%.4f, mean=%.4f, std=%.4f, max=%.4f) | "
                "mc_ret(min=%.4f, mean=%.4f, std=%.4f, max=%.4f) | "
                "values(min=%.4f, mean=%.4f, std=%.4f, max=%.4f)",
                reward_buffer.min().item(), reward_buffer.mean().item(), reward_buffer.std().item(), reward_buffer.max().item(),
                flat_returns.min().item(), flat_returns.mean().item(), flat_returns.std().item(), flat_returns.max().item(),
                flat_values.min().item(), flat_values.mean().item(), flat_values.std().item(), flat_values.max().item()
            )

        # CHATGPT VERIFICATION: Check indexing alignment
        if iteration == 1:
            t_test, e_test = 10, 7
            obs_te = obs_buffer[t_test, e_test]
            ret_te = mc_returns[t_test, e_test]
            idx = t_test * args.n_envs + e_test
            flat_obs_te = flat_obs[idx]
            flat_ret_te = flat_returns[idx]
            logger.info(
                "INDEXING CHECK: obs match=%s, ret match=%.4f vs %.4f",
                str(torch.allclose(obs_te, flat_obs_te)),
                ret_te.item(),
                flat_ret_te.item()
            )

        # Log advantage stats (pre-normalization for debugging)
        adv_raw = flat_returns - flat_values
        adv_mean_raw = adv_raw.mean().item()
        adv_std_raw = adv_raw.std().item()
        adv_min_raw = adv_raw.min().item()
        adv_max_raw = adv_raw.max().item()

        if iteration <= 5 or iteration % 10 == 0:
            logger.info(
                "ADV STATS (raw): mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
                adv_mean_raw, adv_std_raw, adv_min_raw, adv_max_raw
            )

        # PPO update epochs with full diagnostics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_approx_kl = 0
        total_clip_frac = 0
        total_grad_norm = 0
        n_updates = 0

        # Standard PPO: use n_epochs, NO extra critic epochs
        ppo_epochs = args.n_epochs  # e.g. 4

        # ===== STANDARD PPO: Joint actor-critic updates for n_epochs =====
        for epoch in range(ppo_epochs):
            indices = torch.randperm(flat_obs.shape[0], device=device)

            for start in range(0, flat_obs.shape[0], args.batch_size):
                end = min(start + args.batch_size, flat_obs.shape[0])
                batch_idx = indices[start:end]

                logits, values, _ = policy(flat_obs[batch_idx])
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(flat_actions[batch_idx])
                entropy = dist.entropy().mean()

                log_ratio = new_log_probs - flat_log_probs[batch_idx]
                ratio = torch.exp(log_ratio)

                # Approx KL divergence
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > args.clip_epsilon).float().mean().item()

                surr1 = ratio * flat_advantages[batch_idx]
                surr2 = torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon) * flat_advantages[batch_idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE to MC returns)
                value_loss = 0.5 * ((values - flat_returns[batch_idx]) ** 2).mean()

                # Entropy coefficient with decay
                progress = min(1.0, total_steps / args.timesteps)
                current_entropy_coef = entropy_coef_start + (entropy_coef_end - entropy_coef_start) * progress

                # Standard PPO loss
                loss = policy_loss + args.value_coef * value_loss - current_entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()

                # Gradient norm
                grad_norm = sum(
                    (p.grad.norm(2).item() ** 2) for p in all_params if p.grad is not None
                ) ** 0.5

                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_approx_kl += approx_kl
                total_clip_frac += clip_frac
                total_grad_norm += grad_norm
                n_updates += 1

        scheduler.step()

        # Calculate explained variance using NEW values (after PPO update)
        with torch.no_grad():
            new_values = []
            batch_size_eval = 4096
            for start in range(0, flat_obs.shape[0], batch_size_eval):
                end = min(start + batch_size_eval, flat_obs.shape[0])
                _, v, _ = policy(flat_obs[start:end])
                new_values.append(v)
            new_values = torch.cat(new_values, dim=0)

            y_pred = new_values
            y_true = flat_returns
            var_y = torch.var(y_true)
            explained_var = 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)
            explained_var = explained_var.item()

            # Critic diag periodically
            if iteration == 1 or iteration % 5 == 0 or total_steps >= args.timesteps:
                pred_mean = y_pred.mean().item()
                pred_std = y_pred.std().item()
                true_mean = y_true.mean().item()
                true_std = y_true.std().item()
                corr = torch.corrcoef(torch.stack([y_pred, y_true]))[0, 1].item()
                logger.info(
                    "Critic diag: pred(mu=%.3f, std=%.3f) vs true(mu=%.3f, std=%.3f), corr=%.3f, EV=%.3f",
                    pred_mean, pred_std, true_mean, true_std, corr, explained_var
                )

        # Calculate metrics
        iter_time = time.time() - iter_start
        steps_per_sec = steps_collected / iter_time
        gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

        avg_grad_norm = total_grad_norm / max(n_updates, 1)
        metrics.update(
            rewards=episode_rewards_batch[-100:] if episode_rewards_batch else [0],
            lengths=episode_lengths_batch[-100:] if episode_lengths_batch else [0],
            policy_loss=total_policy_loss / max(n_updates, 1),
            value_loss=total_value_loss / max(n_updates, 1),
            entropy=total_entropy / max(n_updates, 1),
            lr=scheduler.get_last_lr()[0],
            sps=steps_per_sec,
            kl_div=total_approx_kl / max(n_updates, 1),
            clip_frac=total_clip_frac / max(n_updates, 1),
            explained_var=explained_var,
            policy_grad_norm=avg_grad_norm,
            value_grad_norm=avg_grad_norm,  # Same optimizer now
            approx_kl=total_approx_kl / max(n_updates, 1)
        )

        # Log to TensorBoard
        stats = metrics.get_stats()
        writer.add_scalar('train/reward', stats['avg_reward'], total_steps)
        writer.add_scalar('train/policy_loss', stats['policy_loss'], total_steps)
        writer.add_scalar('train/value_loss', stats['value_loss'], total_steps)
        writer.add_scalar('train/entropy', stats['entropy'], total_steps)
        writer.add_scalar('train/steps_per_sec', steps_per_sec, total_steps)
        writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], total_steps)
        writer.add_scalar('diagnostics/approx_kl', stats['approx_kl'], total_steps)
        writer.add_scalar('diagnostics/clip_fraction', stats['clip_frac'], total_steps)
        writer.add_scalar('diagnostics/explained_variance', stats['explained_var'], total_steps)
        writer.add_scalar('diagnostics/entropy_ratio', stats['entropy_ratio'], total_steps)
        writer.add_scalar('diagnostics/grad_norm', avg_grad_norm, total_steps)
        # NEW: compact per-iteration summary into training.log
        if (
            iteration == 1 or
            iteration % args.log_interval == 0 or
            total_steps >= args.timesteps
        ):
            # Calculate action distribution for this iteration
            action_counts = torch.bincount(flat_actions, minlength=n_actions).float()
            action_probs = action_counts / action_counts.sum()
            top_action = action_probs.argmax().item()
            top_action_name = env.portfolio_regimes[top_action][0]
            top_action_pct = action_probs[top_action].item() * 100

            logger.info(
                "ITER %05d | steps=%9d | avg_reward=%+6.3f | best=%+6.3f | "
                "EV=%.3f | KL=%.4f | clip=%.3f | entropy=%.4f (%.1f%% init) | sps=%7.0f",
                iteration,
                total_steps,
                stats['avg_reward'],
                stats['best_reward'],
                stats['explained_var'],
                stats['approx_kl'],
                stats['clip_frac'],
                stats['entropy'],
                stats['entropy_ratio'] * 100.0,
                stats['sps']
            )
            logger.info(
                "         | Top action: %s (%.1f%%) | Top 3: %s",
                top_action_name,
                top_action_pct,
                ', '.join([f"{env.portfolio_regimes[i][0]}:{action_probs[i].item()*100:.1f}%"
                           for i in action_probs.argsort(descending=True)[:3].tolist()])
            )

        # Console pretty print
        if time.time() - last_log_time > 2:  # Every ~2 seconds
            print_status(metrics, total_steps, args.timesteps, gpu_mem)
            last_log_time = time.time()

        # Save checkpoint
        if time.time() - last_save_time > args.save_interval:
            checkpoint_path = f"checkpoints/clstm_full/model_step_{total_steps}.pt"
            torch.save({
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'total_steps': total_steps,
                'metrics': metrics.get_stats()
            }, checkpoint_path)
            logger.info(f"💾 Saved checkpoint: {checkpoint_path}")
            last_save_time = time.time()

    # Final save
    final_path = f"checkpoints/clstm_full/model_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save({
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'total_steps': total_steps,
        'metrics': metrics.get_stats()
    }, final_path)

    writer.close()

    # Final stats
    stats = metrics.get_stats()
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print(f"   Total Steps:     {total_steps:,}")
    print(f"   Total Episodes:  {stats['total_episodes']:,}")
    print(f"   Best Reward:     {stats['best_reward']:.2f}")
    print(f"   Final Avg Reward: {stats['avg_reward']:.2f}")
    print(f"   Training Time:   {stats['elapsed']/60:.1f} minutes")
    print(f"   Avg Speed:       {total_steps/stats['elapsed']:,.0f} steps/sec")
    print(f"   Model saved:     {final_path}")
    print("=" * 80)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_config_with_args(args, config: dict):
    """Merge YAML config with command-line args. CLI args take precedence."""
    # PPO config
    ppo = config.get('ppo', {})
    if args.lr is None:
        args.lr = ppo.get('learning_rate', 3e-4)
    if args.batch_size is None:
        args.batch_size = ppo.get('batch_size', 2048)
    if args.n_steps is None:
        args.n_steps = ppo.get('n_steps', 256)
    if args.n_epochs is None:
        args.n_epochs = ppo.get('n_epochs', 4)
    if args.gamma is None:
        args.gamma = ppo.get('gamma', 0.99)
    if args.clip_epsilon is None:
        args.clip_epsilon = ppo.get('clip_epsilon', 0.2)
    if args.entropy_coef is None:
        args.entropy_coef = ppo.get('entropy_coef', 0.01)
    if args.entropy_coef_final is None:
        args.entropy_coef_final = ppo.get('entropy_coef_final', 0.005)
    if args.value_coef is None:
        args.value_coef = ppo.get('value_coef', 0.5)
    if args.max_grad_norm is None:
        args.max_grad_norm = ppo.get('max_grad_norm', 0.5)
    if args.reward_scale is None:
        args.reward_scale = ppo.get('reward_scale', 0.1)

    # Model config
    model = config.get('model', {})
    if args.hidden_dim is None:
        args.hidden_dim = model.get('hidden_dim', 128)

    # Env config
    env = config.get('env', {})
    if args.n_envs is None:
        args.n_envs = env.get('n_envs', 2048)

    # Training config
    training = config.get('training', {})
    if args.timesteps is None:
        args.timesteps = training.get('total_timesteps', 10_000_000)
    if args.save_interval is None:
        args.save_interval = training.get('save_interval', 300)
    if args.log_interval is None:
        args.log_interval = training.get('log_interval', 10)

    # Data config
    data = config.get('data', {})
    if args.cache_path is None:
        args.cache_path = data.get('train_cache_path', None)

    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='v2 Multi-Asset GRU-PPO Training')

    # Config file (optional - can override all defaults)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (e.g., configs/rl_v2_multi_asset.yaml)')

    # Core training args
    parser.add_argument('--timesteps', type=int, default=None, help='Total training timesteps')
    parser.add_argument('--n-envs', type=int, default=None, help='Number of parallel environments')
    parser.add_argument('--n-steps', type=int, default=None, help='Steps per rollout')
    parser.add_argument('--batch-size', type=int, default=None, help='Mini-batch size')
    parser.add_argument('--hidden-dim', type=int, default=None, help='GRU hidden dimension')

    # PPO hyperparameters
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=None, help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda (unused with MC advantage)')
    parser.add_argument('--clip-epsilon', type=float, default=None, help='PPO clip epsilon')
    parser.add_argument('--value-coef', type=float, default=None, help='Value loss coefficient')
    parser.add_argument('--entropy-coef', type=float, default=None, help='Initial entropy coefficient')
    parser.add_argument('--entropy-coef-final', type=float, default=None, help='Final entropy coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=None, help='Max gradient norm')
    parser.add_argument('--reward-scale', type=float, default=None, help='Reward scale')
    parser.add_argument('--target-kl', type=float, default=0.02, help='Target KL for logging')
    parser.add_argument('--n-epochs', type=int, default=None, help='PPO epochs per iteration')

    # Other args
    parser.add_argument('--save-interval', type=int, default=None, help='Checkpoint save interval (seconds)')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')
    parser.add_argument('--log-interval', type=int, default=None, help='Iterations between summary log lines')
    parser.add_argument('--cache-path', type=str, default=None, help='Path to GPU cache file')

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        logger.info(f"📄 Loading config from: {args.config}")
        config = load_config(args.config)
        args = merge_config_with_args(args, config)
    else:
        # Apply defaults if no config and no CLI override
        if args.timesteps is None: args.timesteps = 10_000_000
        if args.n_envs is None: args.n_envs = 2048
        if args.n_steps is None: args.n_steps = 256
        if args.batch_size is None: args.batch_size = 2048
        if args.hidden_dim is None: args.hidden_dim = 128
        if args.lr is None: args.lr = 3e-4
        if args.gamma is None: args.gamma = 0.99
        if args.clip_epsilon is None: args.clip_epsilon = 0.2
        if args.value_coef is None: args.value_coef = 0.5
        if args.entropy_coef is None: args.entropy_coef = 0.01
        if args.entropy_coef_final is None: args.entropy_coef_final = 0.005
        if args.max_grad_norm is None: args.max_grad_norm = 0.5
        if args.reward_scale is None: args.reward_scale = 0.1
        if args.n_epochs is None: args.n_epochs = 4
        if args.save_interval is None: args.save_interval = 300
        if args.log_interval is None: args.log_interval = 10

    train(args)
