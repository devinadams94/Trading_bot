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
import signal
from datetime import datetime
from collections import deque

# ============================================================================
# GRACEFUL SHUTDOWN HANDLER
# ============================================================================
class GracefulShutdown:
    """Handle Ctrl+C gracefully by saving checkpoint before exit."""
    def __init__(self):
        self.shutdown_requested = False
        self.policy = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.total_steps = 0
        self.metrics = None

    def register(self, policy, actor_optimizer, critic_optimizer, metrics):
        """Register components needed for checkpoint saving."""
        self.policy = policy
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.metrics = metrics

    def update_steps(self, total_steps):
        """Update current step count."""
        self.total_steps = total_steps

    def request_shutdown(self, signum, frame):
        """Signal handler for Ctrl+C."""
        if self.shutdown_requested:
            print("\n⚠️  Force quit requested. Exiting without saving...")
            sys.exit(1)

        self.shutdown_requested = True
        print("\n")
        print("=" * 60)
        print("🛑 GRACEFUL SHUTDOWN REQUESTED (Ctrl+C)")
        print("   Finishing current iteration and saving checkpoint...")
        print("   Press Ctrl+C again to force quit (no save)")
        print("=" * 60)

    def save_checkpoint_on_exit(self):
        """Save checkpoint when shutting down."""
        if self.policy is None:
            return

        os.makedirs("checkpoints/clstm_full", exist_ok=True)
        checkpoint_path = f"checkpoints/clstm_full/model_interrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"

        checkpoint_data = {
            'model_state_dict': self.policy.state_dict(),
            'total_steps': self.total_steps,
        }

        if self.actor_optimizer:
            checkpoint_data['actor_optimizer_state_dict'] = self.actor_optimizer.state_dict()
        if self.critic_optimizer:
            checkpoint_data['critic_optimizer_state_dict'] = self.critic_optimizer.state_dict()
        if self.metrics:
            checkpoint_data['metrics'] = self.metrics.get_stats()

        torch.save(checkpoint_data, checkpoint_path)

        print("\n" + "=" * 60)
        print("💾 CHECKPOINT SAVED ON INTERRUPT")
        print("=" * 60)
        print(f"   Steps completed: {self.total_steps:,}")
        print(f"   Saved to: {checkpoint_path}")
        print("   Resume with: --resume or --checkpoint <path>")
        print("=" * 60)

# Global shutdown handler
shutdown_handler = GracefulShutdown()

# Add repository root to Python path for imports
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

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


def find_latest_checkpoint(checkpoint_dir: str = "checkpoints/clstm_full") -> str:
    """Find the latest checkpoint file in the checkpoint directory."""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_files = []
    for f in os.listdir(checkpoint_dir):
        if f.endswith('.pt'):
            path = os.path.join(checkpoint_dir, f)
            checkpoint_files.append((path, os.path.getmtime(path)))

    if not checkpoint_files:
        return None

    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: x[1], reverse=True)
    return checkpoint_files[0][0]


def load_checkpoint(checkpoint_path: str, policy, actor_optimizer, device, critic_optimizer=None):
    """Load checkpoint and return the starting step count."""
    logger.info(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Check if model architecture matches
    try:
        policy.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        if "size mismatch" in str(e):
            logger.warning(f"   ⚠️  Checkpoint architecture mismatch - starting fresh")
            logger.warning(f"   {e}")
            return 0, {}
        else:
            raise

    # Load optimizer states (supports both old single optimizer and new dual optimizer format)
    if 'actor_optimizer_state_dict' in checkpoint:
        try:
            actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            if critic_optimizer and 'critic_optimizer_state_dict' in checkpoint:
                critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        except Exception as e:
            logger.warning(f"   ⚠️  Could not load optimizer state: {e}")
    elif 'optimizer_state_dict' in checkpoint:
        # Legacy single optimizer format
        try:
            actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            logger.warning(f"   ⚠️  Could not load optimizer state (legacy format): {e}")

    total_steps = checkpoint.get('total_steps', 0)
    metrics_data = checkpoint.get('metrics', {})

    logger.info(f"   ✅ Resumed from step {total_steps:,}")
    if metrics_data:
        logger.info(f"   Best reward: {metrics_data.get('best_reward', 'N/A')}")
        logger.info(f"   Win rate: {metrics_data.get('win_rate', 0)*100:.1f}%")

    return total_steps, metrics_data

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
        # Enable CUDA graphs for reduced kernel launch overhead (experimental)
        # torch.cuda.set_device(0)
        logger.info("✅ GPU optimizations applied (TF32, cuDNN benchmark)")


def create_amp_context(use_amp: bool, device_type: str = 'cuda'):
    """Create AMP autocast context for mixed precision training"""
    if use_amp and device_type == 'cuda':
        return torch.amp.autocast(device_type='cuda', dtype=torch.float16)
    else:
        # Return a dummy context manager that does nothing
        return torch.amp.autocast(device_type='cuda', enabled=False)


def get_gpu_info():
    """Get information about available GPUs"""
    if not torch.cuda.is_available():
        return {'available': False, 'count': 0, 'devices': []}

    n_gpus = torch.cuda.device_count()
    devices = []
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        devices.append({
            'id': i,
            'name': props.name,
            'memory_gb': props.total_memory / 1e9,
            'compute_capability': f"{props.major}.{props.minor}"
        })

    return {'available': True, 'count': n_gpus, 'devices': devices}


def calculate_safe_vram_settings(gpu_info: dict, obs_dim: int = 64, n_actions: int = 16,
                                  max_vram_gpu0: float = 45.0, max_vram_gpu1: float = 20.0):
    """
    Calculate safe n_envs and n_steps based on available VRAM.

    Memory estimation for rollout buffers (per GPU):
        obs_buffer:     n_steps × n_envs × obs_dim × 4 bytes
        action_buffer:  n_steps × n_envs × 8 bytes (int64)
        reward_buffer:  n_steps × n_envs × 4 bytes
        log_prob_buffer: n_steps × n_envs × 4 bytes
        value_buffer:   n_steps × n_envs × 4 bytes
        done_buffer:    n_steps × n_envs × 1 byte
        logits_buffer:  n_steps × n_envs × n_actions × 4 bytes

    Total per sample ≈ obs_dim*4 + 8 + 4 + 4 + 4 + 1 + n_actions*4
                     ≈ 64*4 + 8 + 4 + 4 + 4 + 1 + 16*4 = 341 bytes

    Args:
        gpu_info: Output from get_gpu_info()
        obs_dim: Observation dimension
        n_actions: Number of actions
        max_vram_gpu0: Maximum VRAM to use on GPU 0 (GB)
        max_vram_gpu1: Maximum VRAM to use on GPU 1+ (GB)

    Returns:
        dict with recommended n_envs, n_steps, batch_size
    """
    if not gpu_info['available']:
        return {'n_envs': 2048, 'n_steps': 128, 'batch_size': 2048, 'total_vram_gb': 0}

    # Calculate bytes per sample in rollout buffer
    bytes_per_sample = (
        obs_dim * 4 +      # obs (float32)
        8 +                 # action (int64)
        4 +                 # reward (float32)
        4 +                 # log_prob (float32)
        4 +                 # value (float32)
        1 +                 # done (bool)
        n_actions * 4       # logits (float32)
    )

    # Add ~50% overhead for model, gradients, optimizer states, etc.
    overhead_factor = 1.5

    # Calculate usable VRAM per GPU
    usable_vram = []
    for i, gpu in enumerate(gpu_info['devices']):
        if i == 0:
            max_gb = min(gpu['memory_gb'] * 0.9, max_vram_gpu0)  # 90% of available or cap
        else:
            max_gb = min(gpu['memory_gb'] * 0.9, max_vram_gpu1)
        usable_vram.append(max_gb)

    total_vram_gb = sum(usable_vram)

    # Calculate max samples we can hold (n_steps × n_envs)
    # Leave 2GB for model and other tensors
    buffer_vram_gb = total_vram_gb - 2.0
    max_samples = int((buffer_vram_gb * 1e9) / (bytes_per_sample * overhead_factor))

    # Choose n_steps and n_envs
    # Prefer larger n_envs over longer n_steps for parallelism
    n_steps = 256  # Good balance for advantage estimation
    n_envs = max_samples // n_steps

    # Round n_envs to power of 2 for efficiency
    n_envs = 2 ** int(np.log2(max(n_envs, 1024)))

    # Cap at reasonable maximums
    n_envs = min(n_envs, 32768)  # 32K envs max

    # Batch size = n_envs for single update per rollout
    batch_size = min(n_envs, 16384)  # Cap batch size for memory during backward pass

    return {
        'n_envs': n_envs,
        'n_steps': n_steps,
        'batch_size': batch_size,
        'total_vram_gb': total_vram_gb,
        'per_gpu_vram': usable_vram,
        'max_samples': max_samples,
    }


class MultiGPUPolicyWrapper(nn.Module):
    """
    Wrapper for multi-GPU training using DataParallel.
    Handles hidden state management across GPUs.
    """
    def __init__(self, policy: nn.Module, device_ids: list = None):
        super().__init__()
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self.n_gpus = len(self.device_ids)
        self.primary_device = torch.device(f'cuda:{self.device_ids[0]}')

        # Move policy to primary GPU and wrap with DataParallel
        self.policy = policy.to(self.primary_device)
        if self.n_gpus > 1:
            self.policy = nn.DataParallel(self.policy, device_ids=self.device_ids)
            logger.info(f"🔥 Multi-GPU: Using {self.n_gpus} GPUs with DataParallel")
            for i in self.device_ids:
                logger.info(f"   GPU {i}: {torch.cuda.get_device_name(i)}")

    def forward(self, x, hidden=None):
        """Forward pass with hidden state management"""
        return self.policy(x, hidden)

    def get_action(self, obs, hidden=None):
        """Sample action from policy (delegates to base policy)"""
        base_policy = self.get_base_policy()
        return base_policy.get_action(obs, hidden)

    def get_base_policy(self):
        """Get the underlying policy (unwrap DataParallel if needed)"""
        if isinstance(self.policy, nn.DataParallel):
            return self.policy.module
        return self.policy

    def parameters(self):
        """Return parameters for optimizer"""
        return self.policy.parameters()

    def state_dict(self):
        """Get state dict (unwrap DataParallel if needed)"""
        return self.get_base_policy().state_dict()

    def load_state_dict(self, state_dict):
        """Load state dict"""
        self.get_base_policy().load_state_dict(state_dict)


class MultiGPUEnvironmentManager:
    """
    Manages multiple environments across GPUs for parallel training.
    Each GPU runs a portion of the environments.
    """
    def __init__(self, env_class, n_envs: int, device_ids: list = None, **env_kwargs):
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self.n_gpus = len(self.device_ids)
        self.n_envs = n_envs
        self.envs_per_gpu = n_envs // self.n_gpus

        # Create environments on each GPU
        self.envs = []
        for i, device_id in enumerate(self.device_ids):
            device = f'cuda:{device_id}'
            # Last GPU gets any remainder
            env_count = self.envs_per_gpu if i < self.n_gpus - 1 else n_envs - i * self.envs_per_gpu
            env = env_class(n_envs=env_count, device=device, **env_kwargs)
            self.envs.append(env)
            logger.info(f"   GPU {device_id}: {env_count} environments")

        # Get properties from first env
        self.n_actions = self.envs[0].n_actions
        self.portfolio_regimes = self.envs[0].portfolio_regimes
        self.primary_device = torch.device(f'cuda:{self.device_ids[0]}')

    def reset(self):
        """Reset all environments and gather results"""
        obs_list = []
        info_list = []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs.to(self.primary_device))
            info_list.append(info)
        return torch.cat(obs_list, dim=0), info_list

    def step(self, actions: torch.Tensor):
        """Step all environments with given actions"""
        # Split actions across GPUs
        action_splits = torch.split(actions, [env.n_envs for env in self.envs])

        obs_list = []
        reward_list = []
        terminated_list = []
        truncated_list = []
        info_list = []

        for i, (env, act) in enumerate(zip(self.envs, action_splits)):
            # Move actions to correct device
            env_device = torch.device(f'cuda:{self.device_ids[i]}')
            act_gpu = act.to(env_device)
            obs, reward, terminated, truncated, info = env.step(act_gpu)

            # Gather to primary device
            obs_list.append(obs.to(self.primary_device))
            reward_list.append(reward.to(self.primary_device))
            terminated_list.append(terminated.to(self.primary_device))
            truncated_list.append(truncated.to(self.primary_device))
            info_list.append(info)

        return (
            torch.cat(obs_list, dim=0),
            torch.cat(reward_list, dim=0),
            torch.cat(terminated_list, dim=0),
            torch.cat(truncated_list, dim=0),
            info_list
        )


class TrainingMetrics:
    """Track and log training metrics with PPO diagnostics and trade metrics"""
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

        # Trade metrics (new)
        self.trades_completed = deque(maxlen=window_size)
        self.winning_trades = deque(maxlen=window_size)
        self.avg_trade_returns = deque(maxlen=window_size)
        self.total_trades = 0
        self.total_wins = 0
        self.cumulative_trade_return = 0.0

        # Warning flags
        self.warnings = []

    def update(self, rewards, lengths, policy_loss, value_loss, entropy, lr, sps,
               kl_div=0, clip_frac=0, explained_var=0, policy_grad_norm=0,
               value_grad_norm=0, approx_kl=0, trades=0, wins=0, avg_trade_ret=0):
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

        # Trade metrics
        self.trades_completed.append(trades)
        self.winning_trades.append(wins)
        self.avg_trade_returns.append(avg_trade_ret)
        self.total_trades += trades
        self.total_wins += wins
        if trades > 0:
            self.cumulative_trade_return += avg_trade_ret * trades

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
        elapsed = time.time() - self.start_time

        # Calculate trade metrics
        win_rate = self.total_wins / max(self.total_trades, 1)
        avg_return_per_trade = self.cumulative_trade_return / max(self.total_trades, 1)
        trades_per_sec = self.total_trades / max(elapsed, 1)

        # PPO health metrics
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        explained_var = np.mean(self.explained_variances) if self.explained_variances else 0
        entropy_ratio = (np.mean(list(self.entropies)[-10:]) / self.initial_entropy) if self.initial_entropy and self.entropies else 1.0
        approx_kl = np.mean(self.approx_kls) if self.approx_kls else 0
        clip_frac = np.mean(self.clip_fractions) if self.clip_fractions else 0

        # ===== COMPOSITE MODEL SCORE =====
        # Combines trading performance + model health into a single score
        # Range: -100 to +100 (approximately)
        #
        # Components (weighted average):
        # 1. Profitability (40%): avg_return * win_rate - captures trading performance
        # 2. Reward Signal (20%): normalized avg_reward - captures RL learning
        # 3. Value Accuracy (20%): explained_variance - captures critic quality
        # 4. Exploration Health (10%): entropy ratio - prevents premature convergence
        # 5. Stability (10%): penalty for high KL or clip fraction

        # 1. Profitability component: scale to ~[-50, 50]
        profitability = avg_return_per_trade * 100 * win_rate * 100  # e.g., 0.01 * 0.6 -> 60
        profitability = np.clip(profitability, -50, 50)

        # 2. Reward component: normalize to [-25, 25]
        reward_component = np.clip(avg_reward * 5, -25, 25)

        # 3. Value accuracy: scale explained_var [-1, 1] to [0, 25]
        value_component = np.clip((explained_var + 1) * 12.5, 0, 25)

        # 4. Exploration health: entropy_ratio [0, 1] -> [0, 12.5]
        exploration_component = np.clip(entropy_ratio * 12.5, 0, 12.5)

        # 5. Stability: penalize high KL (>0.02) and high clip (>0.2)
        kl_penalty = max(0, (approx_kl - 0.02) * 250)  # 0 if KL < 0.02
        clip_penalty = max(0, (clip_frac - 0.2) * 50)  # 0 if clip < 0.2
        stability_penalty = np.clip(kl_penalty + clip_penalty, 0, 12.5)

        # Final composite: weighted sum
        model_score = (
            0.40 * profitability +
            0.20 * reward_component +
            0.20 * value_component +
            0.10 * exploration_component -
            0.10 * stability_penalty
        )

        return {
            'avg_reward': avg_reward,
            'avg_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'policy_loss': np.mean(self.policy_losses) if self.policy_losses else 0,
            'value_loss': np.mean(self.value_losses) if self.value_losses else 0,
            'entropy': np.mean(self.entropies) if self.entropies else 0,
            'lr': self.learning_rates[-1] if self.learning_rates else 0,
            'sps': np.mean(self.steps_per_sec) if self.steps_per_sec else 0,
            'best_reward': self.best_reward,
            'total_episodes': self.total_episodes,
            'elapsed': elapsed,
            # PPO diagnostics
            'kl_div': np.mean(self.kl_divergences) if self.kl_divergences else 0,
            'approx_kl': approx_kl,
            'clip_frac': clip_frac,
            'explained_var': explained_var,
            'policy_grad_norm': np.mean(self.policy_grad_norms) if self.policy_grad_norms else 0,
            'value_grad_norm': np.mean(self.value_grad_norms) if self.value_grad_norms else 0,
            'entropy_ratio': entropy_ratio,
            'warnings': self.warnings,
            # Trade metrics
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'avg_return_per_trade': avg_return_per_trade,
            'trades_per_sec': trades_per_sec,
            # Composite scores
            'composite_score': avg_return_per_trade * win_rate * 100,  # Legacy trade score
            'model_score': model_score,  # New comprehensive model score
        }


class SimplePolicyNetwork(nn.Module):
    """
    Enhanced PPO network with increased capacity for better learning.

    Architecture (improved for 50GB VRAM):
        Shared Encoder: Linear(obs → hidden) + ReLU + LayerNorm + Linear(hidden → hidden) + ReLU + LayerNorm
        Shared GRU: 2 layers, hidden=256/512 (with dropout)
        Policy Head: hidden → hidden//2 → n_actions (with LayerNorm)
        Value Head: hidden → hidden//2 → 1 (separate capacity for critic)

    Increased capacity helps with:
        - Better feature extraction from market data
        - Improved temporal modeling with 2-layer GRU
        - Separate hidden layers for policy/value reduce interference
    """
    def __init__(self, obs_dim: int = 64, n_actions: int = 16, hidden_dim: int = 256,
                 n_gru_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_gru_layers = n_gru_layers

        # Enhanced encoder (2 layers for better feature extraction)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # 2-layer GRU for better temporal modeling
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_gru_layers,
            batch_first=True,
            dropout=dropout if n_gru_layers > 1 else 0,
        )

        # Enhanced policy head with hidden layer
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, n_actions),
        )

        # Enhanced value head with hidden layer (separate capacity for critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization (standard for PPO)"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.orthogonal_(param, gain=np.sqrt(2))
            elif 'bias' in name:
                nn.init.zeros_(param)
        # Policy head final layer: small init for exploration
        # Find the last Linear layer in policy_head
        for module in reversed(list(self.policy_head.modules())):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                break
        # Value head final layer: standard init
        for module in reversed(list(self.value_head.modules())):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                break

    def forward(self, x, hidden=None):
        """
        Args:
            x: [batch, obs_dim] or [batch, seq, obs_dim]
            hidden: GRU hidden state [n_layers, batch, hidden_dim] or None
        Returns:
            logits: [batch, n_actions]
            value: [batch]
            new_hidden: [n_layers, batch, hidden_dim]
        """
        # Handle 2D input (no sequence dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, obs_dim]

        batch_size = x.shape[0]

        # Encode
        encoded = self.encoder(x)  # [batch, seq, hidden]

        # Initialize hidden if needed (now handles n_gru_layers)
        if hidden is None:
            hidden = torch.zeros(self.n_gru_layers, batch_size, self.hidden_dim, device=x.device)

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

    # Color coding for trade metrics
    def color_winrate(wr):
        if wr >= 0.55: return f"\033[92m{wr:.1%}\033[0m"  # Green - good
        elif wr >= 0.45: return f"\033[93m{wr:.1%}\033[0m"  # Yellow - ok
        else: return f"\033[91m{wr:.1%}\033[0m"  # Red - bad

    def color_composite(cs):
        if cs > 0.5: return f"\033[92m{cs:+.3f}\033[0m"  # Green
        elif cs > 0: return f"\033[93m{cs:+.3f}\033[0m"  # Yellow
        else: return f"\033[91m{cs:+.3f}\033[0m"  # Red

    def color_model_score(ms):
        """Color the model score based on quality thresholds"""
        if ms > 20: return f"\033[92m{ms:+.1f}\033[0m"   # Green - excellent
        elif ms > 10: return f"\033[96m{ms:+.1f}\033[0m" # Cyan - good
        elif ms > 0: return f"\033[93m{ms:+.1f}\033[0m"  # Yellow - learning
        elif ms > -10: return f"\033[91m{ms:+.1f}\033[0m" # Red - poor
        else: return f"\033[95m{ms:+.1f}\033[0m"         # Magenta - very poor

    status = f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│ TRAINING PROGRESS: {progress:5.1f}% │ Steps: {total_steps:,}/{target_steps:,}
├──────────────────────────────────────────────────────────────────────────────┤
│ 📊 PERFORMANCE                                              MODEL SCORE      │
│    Avg Reward:    {stats['avg_reward']:+8.2f}   │  Best Reward: {stats['best_reward']:+8.2f}   ┌─────────────┐│
│    Episodes:      {stats['total_episodes']:8,}   │  Avg Length:  {stats['avg_length']:8.1f}   │ {color_model_score(stats['model_score']):^16}││
│                                                                └─────────────┘│
├──────────────────────────────────────────────────────────────────────────────┤
│ 💰 TRADE METRICS                                                              │
│    Total Trades:  {stats['total_trades']:8,}   │  Win Rate:    {color_winrate(stats['win_rate']):>17}          │
│    Avg Ret/Trade: {stats['avg_return_per_trade']*100:+7.3f}%   │  Trades/sec:  {stats['trades_per_sec']:8.1f}          │
│    Trade Score:   {color_composite(stats['composite_score']):>17}   │  (ret × win% × 100)               │
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

    # Clear screen and print status (no header spam)
    print("\033[2J\033[H", end="")  # Clear screen and move cursor to top
    print(status)


def train(args):
    """Main training function with multi-GPU support"""
    print_header()
    apply_gpu_optimizations()

    # ========== GPU DETECTION AND SETUP ==========
    gpu_info = get_gpu_info()
    use_multi_gpu = gpu_info['available'] and gpu_info['count'] > 1 and args.multi_gpu

    if gpu_info['available']:
        device = torch.device('cuda:0')
        logger.info(f"🖥️  Device: cuda ({gpu_info['count']} GPU(s) available)")
        for gpu in gpu_info['devices']:
            logger.info(f"   GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.1f} GB)")

        # ========== VRAM-AWARE SETTINGS ==========
        # Cap VRAM usage: GPU 0 = 45GB max, GPU 1+ = 20GB max
        max_vram_gpu0 = getattr(args, 'max_vram_gpu0', 45.0)
        max_vram_gpu1 = getattr(args, 'max_vram_gpu1', 20.0)

        # Calculate safe settings based on available VRAM
        safe_settings = calculate_safe_vram_settings(
            gpu_info,
            obs_dim=64,  # Will be updated after env creation
            n_actions=16,
            max_vram_gpu0=max_vram_gpu0,
            max_vram_gpu1=max_vram_gpu1,
        )

        # Apply VRAM caps if current settings exceed safe limits
        if args.n_envs > safe_settings['n_envs']:
            logger.warning(f"⚠️  n_envs={args.n_envs} exceeds safe limit. Capping to {safe_settings['n_envs']}")
            args.n_envs = safe_settings['n_envs']
        if args.batch_size > safe_settings['batch_size']:
            logger.warning(f"⚠️  batch_size={args.batch_size} exceeds safe limit. Capping to {safe_settings['batch_size']}")
            args.batch_size = safe_settings['batch_size']

        logger.info(f"💾 VRAM allocation: {safe_settings['total_vram_gb']:.1f} GB total (GPU0: {max_vram_gpu0}GB cap, GPU1+: {max_vram_gpu1}GB cap)")
        logger.info(f"   Safe limits: n_envs={safe_settings['n_envs']}, batch={safe_settings['batch_size']}, n_steps={safe_settings['n_steps']}")

        if use_multi_gpu:
            device_ids = list(range(gpu_info['count']))
            logger.info(f"🔥 Multi-GPU training ENABLED on {len(device_ids)} GPUs")
        else:
            device_ids = [0]
            if gpu_info['count'] > 1:
                logger.info(f"💡 Tip: Use --multi-gpu to utilize all {gpu_info['count']} GPUs")
    else:
        device = torch.device('cpu')
        device_ids = None
        logger.info(f"🖥️  Device: cpu (no GPU available)")

    # TensorBoard
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"📊 TensorBoard logs: {log_dir}")

    # Import Multi-Asset Portfolio Environment
    use_expanded = getattr(args, 'expanded', False)
    if use_expanded:
        from src.envs.expanded_multi_asset_env import ExpandedMultiAssetEnvironment as MultiAssetEnvironment, print_regime_info
        logger.info("📈 Using EXPANDED Multi-Asset Environment (12 symbols)")
    else:
        from src.envs.multi_asset_env import MultiAssetEnvironment, print_regime_info
        logger.info("📊 Using Standard Multi-Asset Environment (3 symbols)")

    # Create Multi-Asset environment(s)
    cache_path = args.cache_path if hasattr(args, 'cache_path') and args.cache_path else None
    logger.info(f"🚀 Creating Multi-Asset Portfolio environment with {args.n_envs} parallel instances...")
    if cache_path:
        logger.info(f"   Using cache: {cache_path}")

    # Print regime info
    if use_expanded:
        from src.envs.expanded_multi_asset_env import generate_portfolio_regimes, ExpandedMultiAssetEnvironment
        symbols = ExpandedMultiAssetEnvironment.DEFAULT_SYMBOLS
        regimes = generate_portfolio_regimes(symbols)
        print_regime_info(regimes, symbols)
    else:
        print_regime_info()

    # ========== ENVIRONMENT CREATION ==========
    if use_multi_gpu:
        # Multi-GPU: Split environments across GPUs
        logger.info(f"📊 Distributing {args.n_envs} environments across {len(device_ids)} GPUs...")
        env = MultiGPUEnvironmentManager(
            env_class=MultiAssetEnvironment,
            n_envs=args.n_envs,
            device_ids=device_ids,
            episode_length=256,
            cache_path=cache_path,
            volatility_penalty=0.5,
            drawdown_penalty=0.2,
            trading_cost=0.001,
        )
        n_actions = env.n_actions
    else:
        # Single GPU or CPU
        env_device = 'cuda' if gpu_info['available'] else 'cpu'
        env = MultiAssetEnvironment(
            n_envs=args.n_envs,
            episode_length=256,
            device=env_device,
            cache_path=cache_path,
            volatility_penalty=0.5,
            drawdown_penalty=0.2,
            trading_cost=0.001,
        )
        n_actions = env.n_actions

    logger.info(f"   Actions: {n_actions} portfolio regimes")

    # Get observation dimension from environment
    obs_dim = env.obs_dim if hasattr(env, 'obs_dim') else 64
    logger.info(f"   Obs dim: {obs_dim}")

    # ========== POLICY NETWORK CREATION ==========
    # Create base policy with increased capacity
    n_gru_layers = getattr(args, 'n_gru_layers', 2)
    dropout = getattr(args, 'dropout', 0.1)
    base_policy = SimplePolicyNetwork(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_dim=args.hidden_dim,
        n_gru_layers=n_gru_layers,
        dropout=dropout,
    )

    if use_multi_gpu:
        # Wrap with multi-GPU support
        policy = MultiGPUPolicyWrapper(base_policy, device_ids=device_ids)
        device = policy.primary_device
    else:
        policy = base_policy.to(device)

    # Compile for speed (after DataParallel wrapping)
    if args.compile and not use_multi_gpu:  # torch.compile doesn't work well with DataParallel
        logger.info("⚡ Compiling model with torch.compile...")
        policy = torch.compile(policy, mode='reduce-overhead')
    elif args.compile and use_multi_gpu:
        logger.info("⚠️  Skipping torch.compile (not compatible with DataParallel)")

    # ========== SEPARATE OPTIMIZERS FOR ACTOR AND CRITIC ==========
    # Critic needs higher LR to improve explained variance (target: > 0.5)
    # Actor: encoder + GRU + policy_head (base LR)
    # Critic: value_head only (3x LR for faster value learning)

    # Get the base policy (unwrap MultiGPUPolicyWrapper and/or DataParallel)
    if hasattr(policy, 'get_base_policy'):
        # MultiGPUPolicyWrapper
        base_policy = policy.get_base_policy()
    elif hasattr(policy, 'module'):
        # DataParallel
        base_policy = policy.module
    else:
        base_policy = policy

    # Separate parameters
    actor_params = list(base_policy.encoder.parameters()) + \
                   list(base_policy.gru.parameters()) + \
                   list(base_policy.policy_head.parameters())
    critic_params = list(base_policy.value_head.parameters())
    all_params = actor_params + critic_params

    lr = args.lr  # 3e-4 (standard PPO LR)
    critic_lr = lr * 3.0  # 3x higher LR for critic (9e-4)

    actor_optimizer = optim.AdamW(actor_params, lr=lr, weight_decay=1e-4)
    critic_optimizer = optim.AdamW(critic_params, lr=critic_lr, weight_decay=1e-4)

    n_iterations = args.timesteps // (args.n_envs * args.n_steps)
    # Ensure T_max is at least 1 to avoid division by zero
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(actor_optimizer, T_max=max(n_iterations, 1))
    critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(critic_optimizer, T_max=max(n_iterations, 1))

    logger.info(f"📈 Separate optimizers: Actor LR={lr:.0e}, Critic LR={critic_lr:.0e} (3x faster)")
    logger.info(f"🤖 Enhanced GRU Policy Network: {sum(p.numel() for p in policy.parameters()):,} parameters")
    logger.info(f"   Architecture: {n_gru_layers}-layer GRU, hidden_dim={args.hidden_dim}, dropout={dropout}")
    logger.info(f"   Actor params: {sum(p.numel() for p in actor_params):,}, Critic params: {sum(p.numel() for p in critic_params):,}")

    # ========== NO CRITIC PRE-TRAINING ==========
    # ChatGPT recommendation: Skip pretraining so advantages are larger initially
    # This lets the policy learn before the critic becomes too good
    logger.info("⚡ Skipping critic pre-training (allows larger advantages initially)")

    # Metrics tracking
    metrics = TrainingMetrics(window_size=100)

    # ========== CHECKPOINT RESUME ==========
    resume_steps = 0
    if getattr(args, 'resume', False) or getattr(args, 'checkpoint', None):
        checkpoint_path = args.checkpoint if args.checkpoint else find_latest_checkpoint()
        if checkpoint_path and os.path.exists(checkpoint_path):
            # Load with both optimizers
            resume_steps, saved_metrics = load_checkpoint(
                checkpoint_path, policy, actor_optimizer, device, critic_optimizer
            )
            # Restore best reward if available
            if saved_metrics:
                metrics.best_reward = saved_metrics.get('best_reward', float('-inf'))
                metrics.total_trades = saved_metrics.get('total_trades', 0)
                metrics.total_wins = int(saved_metrics.get('win_rate', 0) * saved_metrics.get('total_trades', 0))
        elif getattr(args, 'resume', False):
            logger.info("📂 No checkpoint found, starting fresh training")

    # ========== AUTOMATIC MIXED PRECISION (AMP) ==========
    use_amp = getattr(args, 'amp', False) and gpu_info['available']
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')
        logger.info("⚡ Mixed Precision Training ENABLED (FP16 forward/backward)")
    else:
        scaler = None

    # ========== GRACEFUL SHUTDOWN SETUP ==========
    shutdown_handler.register(policy, actor_optimizer, critic_optimizer, metrics)
    signal.signal(signal.SIGINT, shutdown_handler.request_shutdown)
    signal.signal(signal.SIGTERM, shutdown_handler.request_shutdown)

    # Training loop
    logger.info(f"🎯 Training for {args.timesteps:,} timesteps...")
    if resume_steps > 0:
        logger.info(f"   Resuming from step {resume_steps:,}")
    logger.info(f"   Batch size: {args.batch_size:,}")
    logger.info(f"   N steps: {args.n_steps}")
    logger.info(f"   N envs: {args.n_envs}")

    # Pre-allocate rollout buffers on GPU
    obs_buffer = torch.zeros((args.n_steps, args.n_envs, obs_dim), device=device)
    action_buffer = torch.zeros((args.n_steps, args.n_envs), device=device, dtype=torch.int64)
    reward_buffer = torch.zeros((args.n_steps, args.n_envs), device=device)
    log_prob_buffer = torch.zeros((args.n_steps, args.n_envs), device=device)
    value_buffer = torch.zeros((args.n_steps, args.n_envs), device=device)
    done_buffer = torch.zeros((args.n_steps, args.n_envs), device=device, dtype=torch.bool)
    logits_buffer = torch.zeros((args.n_steps, args.n_envs, n_actions), device=device)

    # Initialize
    obs, _ = env.reset()
    hidden = None
    total_steps = resume_steps  # Start from resume point if resuming
    iteration = resume_steps // (args.n_envs * args.n_steps)  # Resume iteration count
    last_save_time = time.time()
    last_log_time = time.time()

    episode_rewards_batch = []
    episode_lengths_batch = []

    # Entropy coefficient with linear decay
    entropy_coef_start = args.entropy_coef
    entropy_coef_end = getattr(args, 'entropy_coef_final', 0.005)

    # Verbosity flags (check once before loop)
    verbose = getattr(args, 'verbose', False)
    quiet = getattr(args, 'quiet', False)

    # Calculate total iterations for progress display
    total_iterations = (args.timesteps - total_steps) // (args.n_envs * args.n_steps) + 1

    while total_steps < args.timesteps:
        iteration += 1
        iter_start = time.time()

        # Iteration progress (not quiet mode)
        if not quiet:
            pct_complete = 100 * total_steps / args.timesteps
            print(f"\n{'='*60}")
            print(f"  🔄 Iteration {iteration} | Steps: {total_steps:,}/{args.timesteps:,} ({pct_complete:.1f}%)")
            print(f"{'='*60}")

        # Collect rollout
        iter_trades = 0
        iter_wins = 0
        iter_trade_ret = 0.0

        # Rollout progress indicator
        if not quiet:
            print(f"  🎲 Collecting rollout ({args.n_steps} steps × {args.n_envs:,} envs)...", end="", flush=True)

        rollout_start = time.time()
        # Use no_grad for rollout collection (inference_mode causes issues with DataParallel)
        amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16) if use_amp else torch.amp.autocast(device_type='cuda', enabled=False)
        with torch.no_grad(), amp_ctx:
            for step in range(args.n_steps):
                obs_buffer[step] = obs
                action, log_prob, value, hidden, logits = policy.get_action(obs, hidden)
                action_buffer[step] = action
                log_prob_buffer[step] = log_prob
                value_buffer[step] = value
                logits_buffer[step] = logits

                obs, reward, terminated, truncated, info = env.step(action)
                reward_buffer[step] = reward * args.reward_scale  # Scale rewards to O(1-10)
                done_buffer[step] = terminated | truncated

                # Collect trade metrics from info dict
                if isinstance(info, dict):
                    iter_trades += info.get('trades_completed', 0)
                    iter_wins += info.get('winning_trades', 0)
                    iter_trade_ret += info.get('avg_trade_return', 0) * info.get('trades_completed', 1)
                elif isinstance(info, list) and len(info) > 0:
                    # Multi-GPU: aggregate from list of infos
                    for inf in info:
                        if isinstance(inf, dict):
                            iter_trades += inf.get('trades_completed', 0)
                            iter_wins += inf.get('winning_trades', 0)
                            iter_trade_ret += inf.get('avg_trade_return', 0) * inf.get('trades_completed', 1)

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

        rollout_time = time.time() - rollout_start
        if not quiet:
            print(f" done ({rollout_time:.1f}s)")

        steps_collected = args.n_steps * args.n_envs
        total_steps += steps_collected
        metrics.total_steps = total_steps
        shutdown_handler.update_steps(total_steps)

        # ===== COMPUTE GAE (Generalized Advantage Estimation) =====
        # GAE provides lower variance than MC returns while maintaining low bias
        # A_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
        # where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        gae_lambda = getattr(args, 'gae_lambda', 0.95)

        with torch.no_grad():
            # First compute values for all observations
            values_buffer = torch.zeros((args.n_steps, args.n_envs), device=device)
            batch_size_val = 8192
            flat_obs_temp = obs_buffer.reshape(-1, obs_dim)

            flat_values_temp = []
            for start in range(0, flat_obs_temp.shape[0], batch_size_val):
                end = min(start + batch_size_val, flat_obs_temp.shape[0])
                _, v, _ = policy(flat_obs_temp[start:end])
                flat_values_temp.append(v)
            flat_values_temp = torch.cat(flat_values_temp, dim=0)
            values_buffer = flat_values_temp.reshape(args.n_steps, args.n_envs)

            # Bootstrap value for last step (use value of final obs, or 0 if done)
            # Get value estimate for the state AFTER the last step
            _, next_value, _ = policy(obs)  # obs is the current observation after rollout
            next_value = next_value.detach()

            # Compute GAE
            advantages = torch.zeros_like(reward_buffer)
            gae = torch.zeros(args.n_envs, device=device)

            for t in reversed(range(args.n_steps)):
                if t == args.n_steps - 1:
                    next_non_terminal = (~done_buffer[t]).float()
                    next_values = next_value
                else:
                    next_non_terminal = (~done_buffer[t]).float()
                    next_values = values_buffer[t + 1]

                # TD error: delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
                delta = reward_buffer[t] + args.gamma * next_values * next_non_terminal - values_buffer[t]

                # GAE: A_t = delta_t + gamma * lambda * (1 - done) * A_{t+1}
                gae = delta + args.gamma * gae_lambda * next_non_terminal * gae
                advantages[t] = gae

            # Returns = advantages + values (for value function training)
            returns_buffer = advantages + values_buffer

        # Flatten for training
        flat_obs = obs_buffer.reshape(-1, obs_dim)
        flat_actions = action_buffer.reshape(-1)
        flat_log_probs = log_prob_buffer.reshape(-1)
        flat_returns = returns_buffer.reshape(-1)
        flat_old_logits = logits_buffer.reshape(-1, n_actions)
        flat_values = values_buffer.reshape(-1)

        # Normalize advantages (important for stable training)
        with torch.no_grad():
            flat_advantages = advantages.reshape(-1)
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        # DEBUG: Sanity checks — only in verbose mode
        if verbose and (iteration <= 3 or iteration % 50 == 0 or total_steps >= args.timesteps):
            logger.info(
                "DEBUG rollout: rewards(min=%.4f, mean=%.4f, std=%.4f, max=%.4f) | "
                "returns(min=%.4f, mean=%.4f, std=%.4f, max=%.4f) | "
                "values(min=%.4f, mean=%.4f, std=%.4f, max=%.4f)",
                reward_buffer.min().item(), reward_buffer.mean().item(), reward_buffer.std().item(), reward_buffer.max().item(),
                flat_returns.min().item(), flat_returns.mean().item(), flat_returns.std().item(), flat_returns.max().item(),
                flat_values.min().item(), flat_values.mean().item(), flat_values.std().item(), flat_values.max().item()
            )

        # CHATGPT VERIFICATION: Check indexing alignment (only in verbose mode)
        if verbose and iteration == 1:
            t_test, e_test = 10, 7
            obs_te = obs_buffer[t_test, e_test]
            ret_te = returns_buffer[t_test, e_test]  # GAE returns
            idx = t_test * args.n_envs + e_test
            flat_obs_te = flat_obs[idx]
            flat_ret_te = flat_returns[idx]
            logger.info(
                "INDEXING CHECK: obs match=%s, ret match=%.4f vs %.4f",
                str(torch.allclose(obs_te, flat_obs_te)),
                ret_te.item(),
                flat_ret_te.item()
            )

        # Log advantage stats (only in verbose mode)
        adv_raw = flat_returns - flat_values
        adv_mean_raw = adv_raw.mean().item()
        adv_std_raw = adv_raw.std().item()
        adv_min_raw = adv_raw.min().item()
        adv_max_raw = adv_raw.max().item()

        if verbose and (iteration <= 5 or iteration % 10 == 0):
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
        critic_extra_epochs = getattr(args, 'critic_epochs', 4)
        total_epochs = ppo_epochs + critic_extra_epochs
        n_batches_per_epoch = (flat_obs.shape[0] + args.batch_size - 1) // args.batch_size
        total_batches = total_epochs * n_batches_per_epoch
        current_batch = 0

        # ===== STANDARD PPO: Joint actor-critic updates for n_epochs =====
        for epoch in range(ppo_epochs):
            indices = torch.randperm(flat_obs.shape[0], device=device)

            for start in range(0, flat_obs.shape[0], args.batch_size):
                current_batch += 1
                # Progress indicator (only if not quiet)
                if not quiet and current_batch % 10 == 0:
                    pct = 100 * current_batch / total_batches
                    print(f"\r  📈 PPO Update: {current_batch}/{total_batches} batches ({pct:.0f}%) | Epoch {epoch+1}/{ppo_epochs}", end="", flush=True)
                end = min(start + args.batch_size, flat_obs.shape[0])
                batch_idx = indices[start:end]

                # Zero both optimizers
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()

                # Forward pass with optional AMP
                if use_amp:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        logits, values, _ = policy(flat_obs[batch_idx])
                        probs = torch.softmax(logits, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        new_log_probs = dist.log_prob(flat_actions[batch_idx])
                        entropy = dist.entropy().mean()

                        log_ratio = new_log_probs - flat_log_probs[batch_idx]
                        ratio = torch.exp(log_ratio)

                        surr1 = ratio * flat_advantages[batch_idx]
                        surr2 = torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon) * flat_advantages[batch_idx]
                        policy_loss = -torch.min(surr1, surr2).mean()
                        value_loss = 0.5 * ((values - flat_returns[batch_idx]) ** 2).mean()

                        progress = min(1.0, total_steps / args.timesteps)
                        current_entropy_coef = entropy_coef_start + (entropy_coef_end - entropy_coef_start) * progress
                        loss = policy_loss + args.value_coef * value_loss - current_entropy_coef * entropy

                    # Backward with scaler
                    scaler.scale(loss).backward()
                    scaler.unscale_(actor_optimizer)
                    scaler.unscale_(critic_optimizer)
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                    scaler.step(actor_optimizer)
                    scaler.step(critic_optimizer)
                    scaler.update()
                else:
                    # Standard FP32 path
                    logits, values, _ = policy(flat_obs[batch_idx])
                    probs = torch.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    new_log_probs = dist.log_prob(flat_actions[batch_idx])
                    entropy = dist.entropy().mean()

                    log_ratio = new_log_probs - flat_log_probs[batch_idx]
                    ratio = torch.exp(log_ratio)

                    surr1 = ratio * flat_advantages[batch_idx]
                    surr2 = torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon) * flat_advantages[batch_idx]
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = 0.5 * ((values - flat_returns[batch_idx]) ** 2).mean()

                    progress = min(1.0, total_steps / args.timesteps)
                    current_entropy_coef = entropy_coef_start + (entropy_coef_end - entropy_coef_start) * progress
                    loss = policy_loss + args.value_coef * value_loss - current_entropy_coef * entropy

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                    actor_optimizer.step()
                    critic_optimizer.step()

                # Approx KL divergence (always FP32)
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > args.clip_epsilon).float().mean().item()

                # Gradient norm
                grad_norm = sum(
                    (p.grad.norm(2).item() ** 2) for p in all_params if p.grad is not None
                ) ** 0.5
                critic_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_approx_kl += approx_kl
                total_clip_frac += clip_frac
                total_grad_norm += grad_norm
                n_updates += 1

        # ===== EXTRA CRITIC TRAINING EPOCHS =====
        # Train critic more epochs to improve explained variance (target: > 0.5)
        # Critic already has 3x LR via separate optimizer, so no additional boost needed
        # Default: 4 extra epochs (total 8 with 4 PPO epochs = critic sees 8x more updates)
        # Note: critic_extra_epochs already defined above

        for critic_epoch in range(critic_extra_epochs):
            indices = torch.randperm(flat_obs.shape[0], device=device)
            for start in range(0, flat_obs.shape[0], args.batch_size):
                current_batch += 1
                # Progress indicator for critic epochs
                if not quiet and current_batch % 10 == 0:
                    pct = 100 * current_batch / total_batches
                    print(f"\r  🎯 Critic Update: {current_batch}/{total_batches} batches ({pct:.0f}%) | Extra Epoch {critic_epoch+1}/{critic_extra_epochs}", end="", flush=True)

                end = min(start + args.batch_size, flat_obs.shape[0])
                batch_idx = indices[start:end]

                critic_optimizer.zero_grad()

                if use_amp:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        _, values, _ = policy(flat_obs[batch_idx])
                        critic_loss = 0.5 * ((values - flat_returns[batch_idx]) ** 2).mean()
                    scaler.scale(critic_loss).backward()
                    scaler.unscale_(critic_optimizer)
                    torch.nn.utils.clip_grad_norm_(base_policy.value_head.parameters(), args.max_grad_norm)
                    scaler.step(critic_optimizer)
                    scaler.update()
                else:
                    _, values, _ = policy(flat_obs[batch_idx])
                    critic_loss = 0.5 * ((values - flat_returns[batch_idx]) ** 2).mean()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(base_policy.value_head.parameters(), args.max_grad_norm)
                    critic_optimizer.step()

                total_value_loss += critic_loss.item()
                n_updates += 1

        # Clear progress line
        if not quiet:
            print("\r" + " " * 80 + "\r", end="", flush=True)

        # Step both schedulers
        actor_scheduler.step()
        critic_scheduler.step()

        # Calculate explained variance using NEW values (after PPO + extra critic training)
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

        # GPU memory: sum across all GPUs if multi-GPU
        # Use memory_reserved (total GPU memory held by PyTorch) not memory_allocated (active tensors only)
        if torch.cuda.is_available():
            gpu_mem = sum(torch.cuda.memory_reserved(i) for i in range(torch.cuda.device_count())) / 1e9
        else:
            gpu_mem = 0

        avg_grad_norm = total_grad_norm / max(n_updates, 1)
        avg_iter_trade_ret = iter_trade_ret / max(iter_trades, 1)
        metrics.update(
            rewards=episode_rewards_batch[-100:] if episode_rewards_batch else [0],
            lengths=episode_lengths_batch[-100:] if episode_lengths_batch else [0],
            policy_loss=total_policy_loss / max(n_updates, 1),
            value_loss=total_value_loss / max(n_updates, 1),
            entropy=total_entropy / max(n_updates, 1),
            lr=actor_scheduler.get_last_lr()[0],
            sps=steps_per_sec,
            kl_div=total_approx_kl / max(n_updates, 1),
            clip_frac=total_clip_frac / max(n_updates, 1),
            explained_var=explained_var,
            policy_grad_norm=avg_grad_norm,
            value_grad_norm=avg_grad_norm,  # Same optimizer now
            approx_kl=total_approx_kl / max(n_updates, 1),
            trades=iter_trades,
            wins=iter_wins,
            avg_trade_ret=avg_iter_trade_ret
        )

        # Log to TensorBoard
        stats = metrics.get_stats()
        writer.add_scalar('train/reward', stats['avg_reward'], total_steps)
        writer.add_scalar('train/policy_loss', stats['policy_loss'], total_steps)
        writer.add_scalar('train/value_loss', stats['value_loss'], total_steps)
        writer.add_scalar('train/entropy', stats['entropy'], total_steps)
        writer.add_scalar('train/steps_per_sec', steps_per_sec, total_steps)
        writer.add_scalar('train/learning_rate', actor_scheduler.get_last_lr()[0], total_steps)
        writer.add_scalar('train/critic_learning_rate', critic_scheduler.get_last_lr()[0], total_steps)
        writer.add_scalar('diagnostics/approx_kl', stats['approx_kl'], total_steps)
        writer.add_scalar('diagnostics/clip_fraction', stats['clip_frac'], total_steps)
        writer.add_scalar('diagnostics/explained_variance', stats['explained_var'], total_steps)
        writer.add_scalar('diagnostics/entropy_ratio', stats['entropy_ratio'], total_steps)
        writer.add_scalar('diagnostics/grad_norm', avg_grad_norm, total_steps)
        # Trade metrics to TensorBoard
        writer.add_scalar('trades/total_trades', stats['total_trades'], total_steps)
        writer.add_scalar('trades/win_rate', stats['win_rate'], total_steps)
        writer.add_scalar('trades/avg_return_per_trade', stats['avg_return_per_trade'], total_steps)
        writer.add_scalar('trades/trades_per_sec', stats['trades_per_sec'], total_steps)
        writer.add_scalar('trades/composite_score', stats['composite_score'], total_steps)
        # Model score - comprehensive quality metric
        writer.add_scalar('model/score', stats['model_score'], total_steps)
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
                'actor_optimizer_state_dict': actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': critic_optimizer.state_dict(),
                'total_steps': total_steps,
                'metrics': metrics.get_stats()
            }, checkpoint_path)
            logger.info(f"💾 Saved checkpoint: {checkpoint_path}")
            last_save_time = time.time()

        # Check for graceful shutdown request (Ctrl+C)
        if shutdown_handler.shutdown_requested:
            shutdown_handler.save_checkpoint_on_exit()
            writer.close()
            return

    # Final save
    final_path = f"checkpoints/clstm_full/model_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save({
        'model_state_dict': policy.state_dict(),
        'actor_optimizer_state_dict': actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': critic_optimizer.state_dict(),
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
    print("   ─────────────────────────────────────")
    print(f"   Win Rate:        {stats['win_rate']*100:.1f}%")
    print(f"   Avg Ret/Trade:   {stats['avg_return_per_trade']*100:+.3f}%")
    print(f"   Trade Score:     {stats['composite_score']:+.3f}")
    print(f"   ⭐ MODEL SCORE:   {stats['model_score']:+.1f} / 100")
    print("   ─────────────────────────────────────")
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

    # Core training args - multiple ways to specify duration
    parser.add_argument('--timesteps', type=int, default=None, help='Total training timesteps (default: 10M)')
    parser.add_argument('--episodes', type=int, default=None, help='Number of episodes to run (alternative to --timesteps)')
    parser.add_argument('--iterations', type=int, default=None, help='Number of training iterations (alternative to --timesteps)')
    parser.add_argument('--n-envs', type=int, default=None, help='Number of parallel environments')
    parser.add_argument('--n-steps', type=int, default=None, help='Steps per rollout')
    parser.add_argument('--batch-size', type=int, default=None, help='Mini-batch size')
    parser.add_argument('--hidden-dim', type=int, default=None, help='GRU hidden dimension (256 or 512 recommended)')
    parser.add_argument('--n-gru-layers', type=int, default=2, help='Number of GRU layers (1-3)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for regularization')

    # PPO hyperparameters
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=None, help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda for advantage estimation (0.95 default)')
    parser.add_argument('--clip-epsilon', type=float, default=None, help='PPO clip epsilon')
    parser.add_argument('--value-coef', type=float, default=None, help='Value loss coefficient')
    parser.add_argument('--entropy-coef', type=float, default=None, help='Initial entropy coefficient')
    parser.add_argument('--entropy-coef-final', type=float, default=None, help='Final entropy coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=None, help='Max gradient norm')
    parser.add_argument('--reward-scale', type=float, default=None, help='Reward scale')
    parser.add_argument('--critic-epochs', type=int, default=4, help='Extra critic-only training epochs per iteration (improves explained variance)')
    parser.add_argument('--target-kl', type=float, default=0.02, help='Target KL for logging')
    parser.add_argument('--n-epochs', type=int, default=None, help='PPO epochs per iteration')

    # Other args
    parser.add_argument('--save-interval', type=int, default=None, help='Checkpoint save interval (seconds)')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')
    parser.add_argument('--amp', action='store_true',
                        help='Use Automatic Mixed Precision (FP16) for ~1.5-2x speedup')
    parser.add_argument('--log-interval', type=int, default=None, help='Iterations between summary log lines')
    parser.add_argument('--cache-path', type=str, default=None, help='Path to GPU cache file')

    # Verbosity control
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed debug output (rollout stats, advantage stats, etc.)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Production mode: minimal output, no progress bars')

    # Multi-GPU args
    parser.add_argument('--multi-gpu', action='store_true',
                        help='Enable multi-GPU training (uses all available GPUs)')
    parser.add_argument('--gpu-ids', type=str, default=None,
                        help='Comma-separated GPU IDs to use (e.g., "0,1"). Default: all GPUs')

    # Expanded environment
    parser.add_argument('--expanded', action='store_true',
                        help='Use ExpandedMultiAssetEnvironment with 12+ symbols')

    # Resume training
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from the latest checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Specific checkpoint file to resume from')

    # VRAM presets and caps
    parser.add_argument('--low-vram', action='store_true',
                        help='Use conservative settings for 8GB VRAM (n_envs=2048, batch=2048)')
    parser.add_argument('--high-vram', action='store_true',
                        help='Use aggressive settings for 50GB+ VRAM (auto-calculated safe limits)')
    parser.add_argument('--max-vram-gpu0', type=float, default=45.0,
                        help='Maximum VRAM to use on GPU 0 in GB (default: 45)')
    parser.add_argument('--max-vram-gpu1', type=float, default=20.0,
                        help='Maximum VRAM to use on GPU 1+ in GB (default: 20)')

    args = parser.parse_args()

    # Apply VRAM presets (before config loading)
    if args.low_vram:
        if args.n_envs is None: args.n_envs = 2048
        if args.batch_size is None: args.batch_size = 2048
        if args.n_steps is None: args.n_steps = 128
    elif args.high_vram:
        # High VRAM: Use auto-calculated safe limits (will be capped in train())
        if args.n_envs is None: args.n_envs = 32768  # Will be capped to safe limit
        if args.batch_size is None: args.batch_size = 16384
        if args.n_steps is None: args.n_steps = 256

    # Parse GPU IDs if provided
    if args.gpu_ids:
        args.gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    else:
        args.gpu_ids = None

    # Load config if provided
    if args.config:
        logger.info(f"📄 Loading config from: {args.config}")
        config = load_config(args.config)
        args = merge_config_with_args(args, config)
    else:
        # Apply defaults if no config and no CLI override
        # ========== CONSERVATIVE DEFAULTS (will be auto-adjusted in train()) ==========
        # These are safe starting points that will be capped based on actual VRAM
        # The train() function calculates safe limits using calculate_safe_vram_settings()
        # ======================================================
        if args.timesteps is None: args.timesteps = 10_000_000
        if args.n_envs is None: args.n_envs = 16384  # Safe default, will be capped if needed
        if args.n_steps is None: args.n_steps = 256   # Good balance for advantage estimation
        if args.batch_size is None: args.batch_size = 8192  # Safe batch size
        if args.hidden_dim is None: args.hidden_dim = 256
        if args.lr is None: args.lr = 3e-4
        if args.gamma is None: args.gamma = 0.99
        if args.clip_epsilon is None: args.clip_epsilon = 0.2
        if args.value_coef is None: args.value_coef = 0.5
        # OPTIMAL ENTROPY: 0.02-0.05 for discrete action spaces with 16 actions
        # Higher entropy encourages exploration and prevents premature convergence
        if args.entropy_coef is None: args.entropy_coef = 0.03
        if args.entropy_coef_final is None: args.entropy_coef_final = 0.01
        if args.max_grad_norm is None: args.max_grad_norm = 0.5
        if args.reward_scale is None: args.reward_scale = 0.1
        if args.n_epochs is None: args.n_epochs = 4
        if args.save_interval is None: args.save_interval = 300
        if args.log_interval is None: args.log_interval = 10

    # Set multi_gpu attribute if not set
    if not hasattr(args, 'multi_gpu'):
        args.multi_gpu = False

    # ========== CONVERT EPISODES/ITERATIONS TO TIMESTEPS ==========
    # Priority: --timesteps > --episodes > --iterations > config/default
    if args.episodes is not None:
        # episodes = number of iterations to run (1 iteration = 1 episode per env)
        # Each iteration: all n_envs run for n_steps (= 1 episode each)
        # Total timesteps = episodes × n_envs × episode_length
        # This way, --episodes 1000 means "run 1000 iterations" = 1000 episodes per env
        episode_length = args.n_steps if args.n_steps else 256
        n_envs = args.n_envs if args.n_envs else 2048
        args.timesteps = args.episodes * n_envs * episode_length
        total_episodes = args.episodes * n_envs
        logger.info(f"📊 Converting {args.episodes:,} episodes → {args.timesteps:,} timesteps")
        logger.info(f"   ({args.episodes:,} iterations × {n_envs:,} envs × {episode_length} steps = {total_episodes:,} total episode completions)")
    elif args.iterations is not None:
        # iterations × n_envs × n_steps = total timesteps
        n_envs = args.n_envs if args.n_envs else 2048
        n_steps = args.n_steps if args.n_steps else 256
        args.timesteps = args.iterations * n_envs * n_steps
        logger.info(f"📊 Converting {args.iterations:,} iterations → {args.timesteps:,} timesteps (n_envs={n_envs}, n_steps={n_steps})")

    train(args)
