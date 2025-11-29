#!/usr/bin/env python3
"""
Full CLSTM-PPO Training with GPU Environment

Production training script with comprehensive logging and monitoring.
Uses the GPU-accelerated environment with pre-cached Massive.io data.
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


class CLSTMPolicyNetwork(nn.Module):
    """
    CLSTM-PPO network with SEPARATE LSTM trunks for actor and critic.

    Architecture (ChatGPT recommendation to fix EV=0 issue):
        Encoder (shared)
        ├── LSTM_actor  → Attention_actor  → Policy Head
        └── LSTM_critic → Attention_critic → Value Head

    This prevents actor gradients from dominating the shared representation
    and starving the critic of useful features.
    """
    def __init__(self, obs_dim: int = 64, n_actions: int = 31, hidden_dim: int = 256, lstm_layers: int = 3):
        super().__init__()

        # Feature encoder (SHARED between actor and critic)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # SEPARATE LSTM for Actor
        self.lstm_actor = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.1 if lstm_layers > 1 else 0
        )

        # SEPARATE LSTM for Critic
        self.lstm_critic = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.1 if lstm_layers > 1 else 0
        )

        # SEPARATE Attention for Actor
        self.attention_actor = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True, dropout=0.1)

        # SEPARATE Attention for Critic
        self.attention_critic = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True, dropout=0.1)

        # Policy head (actor)
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

        # Value head (critic)
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize LSTM hidden states
        self.hidden_actor = None
        self.hidden_critic = None
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers

    def shared_parameters(self):
        """Parameters shared between actor and critic (encoder only now)"""
        return list(self.encoder.parameters())

    def actor_parameters(self):
        """Actor-only parameters (LSTM, attention, policy head)"""
        return (list(self.lstm_actor.parameters()) +
                list(self.attention_actor.parameters()) +
                list(self.policy.parameters()))

    def critic_parameters(self):
        """Critic-only parameters (LSTM, attention, value head)"""
        return (list(self.lstm_critic.parameters()) +
                list(self.attention_critic.parameters()) +
                list(self.value.parameters()))

    def reset_hidden(self, batch_size: int, device: torch.device):
        self.hidden_actor = (
            torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device),
            torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device)
        )
        self.hidden_critic = (
            torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device),
            torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device)
        )

    def forward(self, x, hidden=None):
        # x: [batch, obs_dim] or [batch, seq, obs_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        batch_size = x.shape[0]

        # Shared encoder
        encoded = self.encoder(x)  # [batch, seq, hidden]

        # Separate hidden states for actor and critic
        if hidden is None:
            hidden_actor = (
                torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=x.device),
                torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=x.device)
            )
            hidden_critic = (
                torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=x.device),
                torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=x.device)
            )
        else:
            # hidden is a tuple of (actor_hidden, critic_hidden)
            hidden_actor, hidden_critic = hidden

        # Actor path: LSTM → Attention → Policy
        lstm_actor_out, new_hidden_actor = self.lstm_actor(encoded, hidden_actor)
        attn_actor_out, _ = self.attention_actor(lstm_actor_out, lstm_actor_out, lstm_actor_out)
        actor_features = attn_actor_out[:, -1, :]  # [batch, hidden]
        logits = self.policy(actor_features)

        # Critic path: LSTM → Attention → Value
        lstm_critic_out, new_hidden_critic = self.lstm_critic(encoded, hidden_critic)
        attn_critic_out, _ = self.attention_critic(lstm_critic_out, lstm_critic_out, lstm_critic_out)
        critic_features = attn_critic_out[:, -1, :]  # [batch, hidden]
        value = self.value(critic_features)

        # Return combined hidden state
        new_hidden = (new_hidden_actor, new_hidden_critic)

        return logits, value.squeeze(-1), new_hidden

    def get_action(self, obs, hidden=None):
        logits, value, new_hidden = self.forward(obs, hidden)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, new_hidden


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
    """Print live training status with PPO diagnostics"""
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

    # Import GPU environment
    from src.envs.gpu_options_env import GPUOptionsEnvironment

    # Create GPU environment
    logger.info(f"🚀 Creating GPU environment with {args.n_envs} parallel instances...")
    env = GPUOptionsEnvironment(
        data_loader='cache',  # Use pre-built cache
        symbols=['SPY', 'QQQ', 'IWM'],
        n_envs=args.n_envs,
        episode_length=256,
        device='cuda'
    )

    # Create CLSTM policy network
    policy = CLSTMPolicyNetwork(
        obs_dim=64,
        n_actions=31,
        hidden_dim=args.hidden_dim,
        lstm_layers=3
    ).to(device)

    # Compile for speed
    if args.compile:
        logger.info("⚡ Compiling model with torch.compile...")
        policy = torch.compile(policy, mode='reduce-overhead')

    # SEPARATE OPTIMIZERS: Critic gets higher LR to track moving targets better
    actor_params = list(policy.shared_parameters()) + list(policy.actor_parameters())
    critic_params = list(policy.critic_parameters())
    actor_lr = args.lr  # 1e-4 (reduced from 2e-4 to lower KL/clip)
    critic_lr = args.lr * 4  # 4e-4 (keep critic strong for tracking)

    optimizer_actor = optim.AdamW(actor_params, lr=actor_lr, weight_decay=1e-4)
    optimizer_critic = optim.AdamW(critic_params, lr=critic_lr, weight_decay=1e-4)

    n_iterations = args.timesteps // (args.n_envs * args.n_steps)
    scheduler_actor = optim.lr_scheduler.CosineAnnealingLR(optimizer_actor, T_max=n_iterations)
    scheduler_critic = optim.lr_scheduler.CosineAnnealingLR(optimizer_critic, T_max=n_iterations)

    logger.info(f"📈 Separate optimizers: actor LR={actor_lr:.0e}, critic LR={critic_lr:.0e}")

    # Count parameters
    n_params = sum(p.numel() for p in policy.parameters())
    logger.info(f"🤖 CLSTM Policy Network: {n_params:,} parameters")

    # ========== CRITIC PRE-TRAINING (ChatGPT suggestion) ==========
    # Collect data and pre-train critic before PPO to give it a head start
    # Using fewer epochs and smaller LR to avoid over-fitting to random policy values
    logger.info("🏋️ Pre-training critic (light warmup)...")

    # Freeze actor, only train critic
    for param in policy.shared_parameters():
        param.requires_grad = True  # Encoder is shared
    for param in policy.actor_parameters():
        param.requires_grad = False
    for param in policy.critic_parameters():
        param.requires_grad = True

    critic_optimizer = optim.Adam(
        list(policy.shared_parameters()) + list(policy.critic_parameters()),
        lr=5e-4  # Lower LR for gentle warmup
    )

    # Collect rollouts for pre-training (just 2 rollouts for light warmup)
    pretrain_obs = []
    pretrain_returns = []
    obs_pt, _ = env.reset()

    for _ in range(2):  # 2 rollouts (was 5)
        observations = []
        rewards = []
        for step in range(256):
            with torch.no_grad():
                action, _, _, _ = policy.get_action(obs_pt)
            next_obs, reward, done, truncated, info = env.step(action)
            observations.append(obs_pt)
            rewards.append(reward * args.reward_scale)  # Apply same scaling
            obs_pt = next_obs

        # Compute returns
        returns = torch.zeros(256, env.n_envs, device=device)
        running_return = torch.zeros(env.n_envs, device=device)
        for t in reversed(range(256)):
            running_return = rewards[t] + 0.99 * running_return
            returns[t] = running_return

        pretrain_obs.append(torch.stack(observations))
        pretrain_returns.append(returns)

    all_obs = torch.cat([o.reshape(-1, o.shape[-1]) for o in pretrain_obs])
    all_returns = torch.cat([r.reshape(-1) for r in pretrain_returns])

    # Pre-train critic for just 3 epochs (light warmup)
    batch_size_pt = 4096
    for epoch in range(3):  # Was 10
        indices = torch.randperm(all_obs.shape[0], device=device)
        total_loss = 0
        n_batches = 0

        for start in range(0, all_obs.shape[0], batch_size_pt):
            end = min(start + batch_size_pt, all_obs.shape[0])
            batch_idx = indices[start:end]

            _, values, _ = policy(all_obs[batch_idx])
            loss = nn.MSELoss()(values, all_returns[batch_idx])

            critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            critic_optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        with torch.no_grad():
            _, preds, _ = policy(all_obs[:10000])
            corr = torch.corrcoef(torch.stack([preds, all_returns[:10000]]))[0, 1].item()
            ev = 1 - torch.var(all_returns[:10000] - preds) / (torch.var(all_returns[:10000]) + 1e-8)
        logger.info(f"   Epoch {epoch+1}: loss={total_loss/n_batches:.4f}, EV={ev:.3f}, corr={corr:.3f}")

    # Unfreeze everything for PPO
    for param in policy.parameters():
        param.requires_grad = True

    logger.info("✅ Critic pre-training complete")

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
                action, log_prob, value, hidden = policy.get_action(obs, hidden)
                action_buffer[step] = action
                log_prob_buffer[step] = log_prob
                value_buffer[step] = value

                obs, reward, terminated, truncated, _ = env.step(action)
                reward_buffer[step] = reward * args.reward_scale  # Scale rewards to O(1-10)
                done_buffer[step] = terminated | truncated

                # Track episode completions
                done_count = done_buffer[step].sum().item()
                if done_count > 0:
                    # Estimate episode stats
                    episode_rewards_batch.extend([reward_buffer[:step+1, i].sum().item()
                                                   for i in range(int(done_count))])
                    episode_lengths_batch.extend([step + 1] * int(done_count))

                # Reset hidden state for done environments
                if done_buffer[step].any():
                    done_idx = done_buffer[step].nonzero(as_tuple=True)[0]
                    if hidden is not None:
                        # hidden is now ((actor_h, actor_c), (critic_h, critic_c))
                        hidden_actor, hidden_critic = hidden
                        hidden_actor[0][:, done_idx, :] = 0
                        hidden_actor[1][:, done_idx, :] = 0
                        hidden_critic[0][:, done_idx, :] = 0
                        hidden_critic[1][:, done_idx, :] = 0

        steps_collected = args.n_steps * args.n_envs
        total_steps += steps_collected
        metrics.total_steps = total_steps

        # Compute returns and advantages
        # SIMPLIFIED: MC returns for value, MC advantages for policy (no GAE recursion)
        with torch.no_grad():
            # 1) Monte Carlo returns for value targets
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

        # 2) MC advantages: A_t = R_t - V(s_t) using CURRENT policy values
        with torch.no_grad():
            flat_values = []
            batch_size_val = 8192
            for start in range(0, flat_obs.shape[0], batch_size_val):
                end = min(start + batch_size_val, flat_obs.shape[0])
                _, v, _ = policy(flat_obs[start:end])
                flat_values.append(v)
            flat_values = torch.cat(flat_values, dim=0)

            # Simple MC advantage: how much better was actual return vs predicted value
            flat_advantages = flat_returns - flat_values
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        # DEBUG: Sanity checks
        logger.info(f"DEBUG rewards: min={reward_buffer.min().item():.4f}, mean={reward_buffer.mean().item():.4f}, std={reward_buffer.std().item():.4f}, max={reward_buffer.max().item():.4f}")
        logger.info(f"DEBUG mc_ret:  min={flat_returns.min().item():.4f}, mean={flat_returns.mean().item():.4f}, std={flat_returns.std().item():.4f}, max={flat_returns.max().item():.4f}")
        logger.info(f"DEBUG values:  min={flat_values.min().item():.4f}, mean={flat_values.mean().item():.4f}, std={flat_values.std().item():.4f}, max={flat_values.max().item():.4f}")

        # CHATGPT VERIFICATION: Check indexing alignment
        if iteration == 1:
            t_test, e_test = 10, 7
            obs_te = obs_buffer[t_test, e_test]
            ret_te = mc_returns[t_test, e_test]
            idx = t_test * args.n_envs + e_test
            flat_obs_te = flat_obs[idx]
            flat_ret_te = flat_returns[idx]
            logger.info(f"INDEXING CHECK: obs match={torch.allclose(obs_te, flat_obs_te)}, ret match={ret_te.item():.4f} vs {flat_ret_te.item():.4f}")

        # CHATGPT TEST: Supervised-in-the-loop (first iteration only)
        # If critic CAN learn with enough epochs, the wiring is correct
        if iteration == 1:
            logger.info("🔬 SUPERVISED-IN-LOOP TEST: 20 critic-only epochs on this rollout...")
            policy.train()
            test_optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4)
            for test_epoch in range(20):
                indices = torch.randperm(flat_obs.shape[0], device=device)
                epoch_loss = 0
                n_batches = 0
                for start in range(0, flat_obs.shape[0], args.batch_size):
                    end = min(start + args.batch_size, flat_obs.shape[0])
                    batch_idx = indices[start:end]
                    _, values, _ = policy(flat_obs[batch_idx])
                    loss = F.mse_loss(values, flat_returns[batch_idx])
                    test_optimizer.zero_grad()
                    loss.backward()
                    test_optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1
                if test_epoch in [0, 9, 19]:
                    # Compute EV on full data after this epoch
                    with torch.no_grad():
                        all_vals = []
                        for s in range(0, flat_obs.shape[0], args.batch_size):
                            e = min(s + args.batch_size, flat_obs.shape[0])
                            _, v, _ = policy(flat_obs[s:e])
                            all_vals.append(v)
                        all_vals = torch.cat(all_vals, dim=0)
                        var_y = flat_returns.var()
                        ev = 1 - (flat_returns - all_vals).var() / (var_y + 1e-8)
                        corr = torch.corrcoef(torch.stack([flat_returns, all_vals]))[0, 1]
                        logger.info(f"   Epoch {test_epoch+1}: loss={epoch_loss/n_batches:.4f}, EV={ev.item():.3f}, corr={corr.item():.3f}")
            logger.info("🔬 If EV is high here, wiring is correct. Resetting optimizer state...")
            # Reset to original optimizer (don't keep test optimizer state)

        # PPO update epochs with full diagnostics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_approx_kl = 0
        total_clip_frac = 0
        total_policy_grad_norm = 0
        total_value_grad_norm = 0
        n_updates = 0

        target_kl = 0.02  # Target KL for early stopping
        policy_epochs = 1  # Single policy epoch (was 2) - conservative for lower clip%
        extra_critic_epochs = 4  # Extra critic-only epochs

        # ===== PHASE 1: Joint actor-critic updates (fewer epochs) =====
        for epoch in range(policy_epochs):
            indices = torch.randperm(flat_obs.shape[0], device=device)
            early_stop = False

            for start in range(0, flat_obs.shape[0], args.batch_size):
                end = min(start + args.batch_size, flat_obs.shape[0])
                batch_idx = indices[start:end]

                # Forward pass
                logits, values, _ = policy(flat_obs[batch_idx])
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(flat_actions[batch_idx])
                entropy = dist.entropy().mean()

                # PPO clipped objective
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

                # Total loss
                loss = policy_loss + args.value_coef * value_loss - current_entropy_coef * entropy

                # Update both actor and critic
                optimizer_actor.zero_grad()
                optimizer_critic.zero_grad()
                loss.backward()

                # Gradient norms
                policy_grad_norm = sum(p.grad.norm(2).item()**2 for p in actor_params if p.grad is not None)**0.5
                value_grad_norm = sum(p.grad.norm(2).item()**2 for p in critic_params if p.grad is not None)**0.5

                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer_actor.step()
                optimizer_critic.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_approx_kl += approx_kl
                total_clip_frac += clip_frac
                total_policy_grad_norm += policy_grad_norm
                total_value_grad_norm += value_grad_norm
                n_updates += 1

                if approx_kl > target_kl * 1.5:
                    early_stop = True
                    break

            if early_stop:
                break

        # ===== PHASE 2: Extra critic-only epochs (no policy updates) =====
        for _ in range(extra_critic_epochs):
            indices = torch.randperm(flat_obs.shape[0], device=device)
            for start in range(0, flat_obs.shape[0], args.batch_size):
                end = min(start + args.batch_size, flat_obs.shape[0])
                batch_idx = indices[start:end]

                _, values, _ = policy(flat_obs[batch_idx])
                value_loss = 0.5 * ((values - flat_returns[batch_idx]) ** 2).mean()

                optimizer_critic.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic_params, args.max_grad_norm)
                optimizer_critic.step()

        scheduler_actor.step()
        scheduler_critic.step()

        # Calculate explained variance using NEW values (after PPO update)
        with torch.no_grad():
            # Recompute values with updated policy
            new_values = []
            batch_size_eval = 4096
            for start in range(0, flat_obs.shape[0], batch_size_eval):
                end = min(start + batch_size_eval, flat_obs.shape[0])
                _, v, _ = policy(flat_obs[start:end])
                new_values.append(v)
            new_values = torch.cat(new_values, dim=0)

            # Use raw MC returns for EV calculation (no normalization anymore!)
            y_pred = new_values
            y_true = flat_returns
            var_y = torch.var(y_true)
            explained_var = 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)
            explained_var = explained_var.item()

            # Diagnostic: Check if critic predictions match returns
            if iteration == 1 or iteration % 5 == 0:
                pred_mean = y_pred.mean().item()
                pred_std = y_pred.std().item()
                true_mean = y_true.mean().item()
                true_std = y_true.std().item()
                corr = torch.corrcoef(torch.stack([y_pred, y_true]))[0, 1].item()
                logger.info(f"Critic: pred(mu={pred_mean:.3f}, std={pred_std:.3f}) vs true(mu={true_mean:.3f}, std={true_std:.3f}), corr={corr:.3f}")

        # Calculate metrics
        iter_time = time.time() - iter_start
        steps_per_sec = steps_collected / iter_time
        gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

        # Update metrics with full diagnostics
        metrics.update(
            rewards=episode_rewards_batch[-100:] if episode_rewards_batch else [0],
            lengths=episode_lengths_batch[-100:] if episode_lengths_batch else [0],
            policy_loss=total_policy_loss / max(n_updates, 1),
            value_loss=total_value_loss / max(n_updates, 1),
            entropy=total_entropy / max(n_updates, 1),
            lr=scheduler_actor.get_last_lr()[0],
            sps=steps_per_sec,
            kl_div=total_approx_kl / max(n_updates, 1),
            clip_frac=total_clip_frac / max(n_updates, 1),
            explained_var=explained_var,
            policy_grad_norm=total_policy_grad_norm / max(n_updates, 1),
            value_grad_norm=total_value_grad_norm / max(n_updates, 1),
            approx_kl=total_approx_kl / max(n_updates, 1)
        )

        # Log to TensorBoard (including diagnostics)
        stats = metrics.get_stats()
        writer.add_scalar('train/reward', stats['avg_reward'], total_steps)
        writer.add_scalar('train/policy_loss', stats['policy_loss'], total_steps)
        writer.add_scalar('train/value_loss', stats['value_loss'], total_steps)
        writer.add_scalar('train/entropy', stats['entropy'], total_steps)
        writer.add_scalar('train/steps_per_sec', steps_per_sec, total_steps)
        writer.add_scalar('train/learning_rate', scheduler_actor.get_last_lr()[0], total_steps)
        # PPO diagnostics
        writer.add_scalar('diagnostics/approx_kl', stats['approx_kl'], total_steps)
        writer.add_scalar('diagnostics/clip_fraction', stats['clip_frac'], total_steps)
        writer.add_scalar('diagnostics/explained_variance', stats['explained_var'], total_steps)
        writer.add_scalar('diagnostics/entropy_ratio', stats['entropy_ratio'], total_steps)
        writer.add_scalar('diagnostics/policy_grad_norm', stats['policy_grad_norm'], total_steps)
        writer.add_scalar('diagnostics/value_grad_norm', stats['value_grad_norm'], total_steps)

        # Print status
        if time.time() - last_log_time > 2:  # Every 2 seconds
            print_status(metrics, total_steps, args.timesteps, gpu_mem)
            last_log_time = time.time()

        # Save checkpoint
        if time.time() - last_save_time > args.save_interval:
            checkpoint_path = f"checkpoints/clstm_full/model_step_{total_steps}.pt"
            torch.save({
                'model_state_dict': policy.state_dict(),
                'optimizer_actor_state_dict': optimizer_actor.state_dict(),
                'optimizer_critic_state_dict': optimizer_critic.state_dict(),
                'total_steps': total_steps,
                'metrics': metrics.get_stats()
            }, checkpoint_path)
            logger.info(f"💾 Saved checkpoint: {checkpoint_path}")
            last_save_time = time.time()

    # Final save
    final_path = f"checkpoints/clstm_full/model_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save({
        'model_state_dict': policy.state_dict(),
        'optimizer_actor_state_dict': optimizer_actor.state_dict(),
        'optimizer_critic_state_dict': optimizer_critic.state_dict(),
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Full CLSTM-PPO Training')
    parser.add_argument('--timesteps', type=int, default=10_000_000, help='Total training timesteps')
    parser.add_argument('--n-envs', type=int, default=2048, help='Number of parallel environments')
    parser.add_argument('--n-steps', type=int, default=256, help='Steps per rollout')
    parser.add_argument('--batch-size', type=int, default=8192, help='Mini-batch size')
    parser.add_argument('--hidden-dim', type=int, default=256, help='LSTM hidden dimension')
    # Standard PPO hyperparameters (SB3/SpinningUp defaults)
    # LR tuned down to reduce KL/clip fraction (was 2e-4, now 1e-4)
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (actor)')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda')
    # Clip epsilon reduced from 0.2 -> 0.1 for tighter trust region
    parser.add_argument('--clip-epsilon', type=float, default=0.10, help='PPO clip epsilon')
    parser.add_argument('--value-coef', type=float, default=1.0, help='Value loss coefficient (increased to push critic harder)')
    # Slightly higher entropy coef with decay (0.02 -> 0.005)
    parser.add_argument('--entropy-coef', type=float, default=0.02, help='Initial entropy coefficient')
    parser.add_argument('--entropy-coef-final', type=float, default=0.005, help='Final entropy coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='Max gradient norm')
    parser.add_argument('--reward-scale', type=float, default=0.02, help='Fixed reward scale (1/50) to get returns O(1-10)')
    # Standard PPO epochs (3-5)
    parser.add_argument('--n-epochs', type=int, default=4, help='PPO epochs per iteration')
    parser.add_argument('--save-interval', type=int, default=300, help='Checkpoint save interval (seconds)')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')
    args = parser.parse_args()

    train(args)

