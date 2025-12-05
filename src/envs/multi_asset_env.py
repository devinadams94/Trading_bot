#!/usr/bin/env python3
"""
Multi-Asset Portfolio Environment for RL Trading

Key changes from single-asset environment:
1. Actions represent discrete portfolio REGIMES (target weight allocations)
2. Tracks positions in multiple assets: SPY, QQQ, IWM
3. Reward includes risk penalty (volatility or drawdown)
4. Agent can rotate between assets, go defensive, etc.

This gives the RL agent meaningful decisions beyond simple timing.
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# PORTFOLIO REGIME DEFINITIONS
# Each action maps to a target weight vector: [CASH, SPY, QQQ, IWM]
# ============================================================================
PORTFOLIO_REGIMES = {
    # Index: (name, [CASH, SPY, QQQ, IWM])
    0:  ("HOLD",           None),  # Special: don't rebalance, keep current allocation
    1:  ("ALL_CASH",       [1.00, 0.00, 0.00, 0.00]),  # 100% cash (risk-off)
    2:  ("ALL_SPY",        [0.00, 1.00, 0.00, 0.00]),  # 100% S&P 500
    3:  ("ALL_QQQ",        [0.00, 0.00, 1.00, 0.00]),  # 100% Nasdaq
    4:  ("ALL_IWM",        [0.00, 0.00, 0.00, 1.00]),  # 100% Russell 2000
    5:  ("EQUAL_WEIGHT",   [0.25, 0.25, 0.25, 0.25]),  # Equal weight all 4
    6:  ("SPY_QQQ_50",     [0.00, 0.50, 0.50, 0.00]),  # 50/50 SPY-QQQ
    7:  ("SPY_IWM_50",     [0.00, 0.50, 0.00, 0.50]),  # 50/50 SPY-IWM
    8:  ("QQQ_IWM_50",     [0.00, 0.00, 0.50, 0.50]),  # 50/50 QQQ-IWM
    9:  ("DEFENSIVE_50",   [0.50, 0.25, 0.00, 0.25]),  # 50% cash + SPY/IWM
    10: ("GROWTH_TILT",    [0.10, 0.30, 0.60, 0.00]),  # Growth: heavy QQQ
    11: ("VALUE_TILT",     [0.10, 0.60, 0.00, 0.30]),  # Value: heavy SPY + IWM
    12: ("SMALLCAP_TILT",  [0.10, 0.20, 0.00, 0.70]),  # Small cap tilt
    13: ("RISK_OFF_75",    [0.75, 0.25, 0.00, 0.00]),  # 75% cash (risk-off)
    14: ("QQQ_HEAVY",      [0.10, 0.10, 0.70, 0.10]),  # QQQ-heavy with diversification
    15: ("BALANCED",       [0.20, 0.40, 0.20, 0.20]),  # Balanced: SPY core
}

N_ACTIONS = len(PORTFOLIO_REGIMES)


class MultiAssetEnvironment:
    """
    Multi-asset portfolio environment with discrete regime actions.

    ================================================================================
    v2 Multi-Asset Baseline Environment
    ================================================================================

    Assets (weight order):
        Index 0: CASH (risk-free, no position)
        Index 1: SPY  (S&P 500 ETF)
        Index 2: QQQ  (Nasdaq 100 ETF)
        Index 3: IWM  (Russell 2000 ETF)

    Action Space:
        16 discrete portfolio regimes mapping to target weight vectors.
        See PORTFOLIO_REGIMES dict for full list.
        Action 0 (HOLD) keeps current allocation; others trigger rebalance.

    Observation Space (dim=64):
        The observation vector contains the following features (in order):

        [0:3]   price_normalized   - Normalized prices for [SPY, QQQ, IWM]
                                     Computed as: (current_price / base_price) - 1.0
                                     Base is day 20's closing price
        [3:6]   returns_scaled     - Daily returns for [SPY, QQQ, IWM] * 10
        [6:9]   volatility_scaled  - 20-day rolling volatility * 10
        [9:13]  weights            - Current portfolio weights [CASH, SPY, QQQ, IWM]
        [13]    portfolio_return   - (portfolio_value - initial) / initial
        [14]    drawdown           - (peak_value - current_value) / peak_value
        [15:64] padding            - Zero padding to reach obs_dim=64

        Total: 15 meaningful features + 49 zeros = 64

    Reward (simplified, v2 baseline):
        reward_t = alpha_t - trading_cost_t

        Where:
        - alpha_t = portfolio_return_t - benchmark_return_t
        - benchmark = equal weight of SPY/QQQ/IWM returns
        - trading_cost_t = trading_cost * turnover (sum of |weight_changes|)
        - Reward is scaled by 100x and clipped to [-5, 5]

        Note: Volatility and drawdown penalties are NOT in training reward.
              They are tracked for evaluation only.

    ================================================================================
    TODO: Future Feature Enhancements (Phase B)
    ================================================================================
    Potential additions to observation space for regime detection:

    1. Rolling volatility features (could replace static 20-day):
       - rolling_vol_5d, rolling_vol_20d, rolling_vol_60d for each asset
       - vol_regime flag: 1 if current_vol > 1.5 * avg_vol else 0

    2. Cross-asset features:
       - QQQ/SPY spread (tech vs broad market)
       - IWM/SPY spread (small cap vs large cap)
       - correlation_spy_qqq_20d, correlation_spy_iwm_20d

    3. Momentum indicators:
       - 5-day momentum, 20-day momentum for each asset
       - RSI_14 for each asset

    4. Market regime features:
       - VIX level (if available)
       - 50-day SMA cross indicator
       - trend strength (e.g., ADX)

    Where to compute: _compute_indicators() for static features,
                      _get_observations() for dynamic features.
    ================================================================================
    """

    def __init__(
        self,
        n_envs: int = 256,
        initial_capital: float = 100000,
        episode_length: int = 256,
        device: str = 'cuda',
        cache_path: str = None,
        # Reward parameters
        volatility_penalty: float = 0.5,  # Lambda for volatility penalty
        drawdown_penalty: float = 0.0,    # Lambda for drawdown penalty (optional)
        trading_cost: float = 0.001,      # 0.1% per rebalance
    ):
        self.n_envs = n_envs
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.initial_capital = initial_capital
        self.episode_length = episode_length
        self.cache_path = cache_path

        # Weight order: [CASH, SPY, QQQ, IWM]
        # Index:         0     1    2    3
        self.symbols = ['SPY', 'QQQ', 'IWM']  # Tradeable symbols
        self.n_assets = 4  # CASH + 3 equities
        self.n_symbols = 3  # Number of tradeable symbols

        # Action space
        self.n_actions = N_ACTIONS
        self.portfolio_regimes = PORTFOLIO_REGIMES

        # Reward parameters - SIMPLIFIED per ChatGPT Phase 1 guidance
        # Option A: reward = alpha - trading_cost only (vol/dd moved to evaluation)
        self.trading_cost = trading_cost              # 0.001 = 0.1% per rebalance
        self.reward_scale = 100.0  # Scale rewards to O(1-10)
        self.reward_clip = 5.0     # Clip to [-5, 5]

        # Legacy penalty params (kept for backward compat, but NOT used in reward)
        self.volatility_penalty = volatility_penalty  # Only for logging/eval
        self.drawdown_penalty = drawdown_penalty      # Only for logging/eval

        # Observation dimensions: per-asset features + portfolio state
        # Per asset: price, return, volatility (3 * 3 = 9)
        # Portfolio: current weights (4), portfolio value (1), recent return (1)
        self.obs_dim = 64

        # Pre-allocate GPU tensors
        self._init_gpu_tensors()

        # Load data
        if cache_path:
            self._load_data_to_gpu()
        else:
            self._create_synthetic_data()

        logger.info(f"✅ Multi-Asset Environment initialized on {self.device}")
        logger.info(f"   Actions: {N_ACTIONS} portfolio regimes")
        logger.info(f"   Assets: SPY, QQQ, IWM, CASH")
        logger.info(f"   Parallel envs: {n_envs}")

    def _init_gpu_tensors(self):
        """Pre-allocate state tensors on GPU"""
        # Portfolio state
        self.capital = torch.full((self.n_envs,), self.initial_capital,
                                   device=self.device, dtype=torch.float32)
        self.portfolio_value = self.capital.clone()

        # Position tracking: [n_envs, n_assets] = shares held per asset
        # Index 0=SPY, 1=QQQ, 2=IWM, 3=CASH (cash is just stored in capital)
        self.positions = torch.zeros((self.n_envs, self.n_symbols),
                                      device=self.device, dtype=torch.float32)

        # Current weights: [n_envs, n_assets] in order [CASH, SPY, QQQ, IWM]
        self.weights = torch.zeros((self.n_envs, self.n_assets),
                                    device=self.device, dtype=torch.float32)
        self.weights[:, 0] = 1.0  # Start 100% cash (index 0)

        # Time tracking
        self.current_step = torch.zeros(self.n_envs, device=self.device, dtype=torch.int32)
        self.current_day_idx = torch.zeros(self.n_envs, device=self.device, dtype=torch.int64)

        # Episode tracking
        self.episode_returns = torch.zeros(self.n_envs, device=self.device, dtype=torch.float32)
        self.done = torch.zeros(self.n_envs, device=self.device, dtype=torch.bool)

        # For reward calculation
        self.prev_portfolio_value = self.capital.clone()
        self.peak_value = self.capital.clone()  # For drawdown calculation
        self.recent_returns = torch.zeros((self.n_envs, 20), device=self.device)  # Rolling window
        self.return_idx = torch.zeros(self.n_envs, device=self.device, dtype=torch.int64)

    def _load_data_to_gpu(self):
        """Load market data from cache"""
        import os

        if not os.path.exists(self.cache_path):
            raise ValueError(f"Cache not found: {self.cache_path}")

        logger.info(f"⚡ Loading cache: {self.cache_path}")
        cache = torch.load(self.cache_path, map_location='cpu', weights_only=False)

        self.n_days = cache['n_days']

        # Prices: [n_days, n_symbols, 5] (OHLCV)
        self.prices = torch.zeros((self.n_days, self.n_symbols, 5),
                                   device=self.device, dtype=torch.float32)

        for i, symbol in enumerate(self.symbols):
            if symbol in cache['stock_prices']:
                stock_data = cache['stock_prices'][symbol].to(self.device)
                n = min(len(stock_data), self.n_days)
                self.prices[:n, i, :] = stock_data[:n]

        # Compute returns and volatility
        self._compute_indicators()

        logger.info(f"✅ Loaded {self.n_days} days, GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    def _create_synthetic_data(self):
        """Create synthetic data for testing"""
        self.n_days = 1000
        self.prices = torch.zeros((self.n_days, self.n_symbols, 5),
                                   device=self.device, dtype=torch.float32)

        # Simulate correlated random walks
        base_prices = [400.0, 350.0, 200.0]  # SPY, QQQ, IWM starting prices
        for i, base in enumerate(base_prices):
            returns = torch.randn(self.n_days, device=self.device) * 0.01
            prices = base * torch.cumprod(1 + returns, dim=0)
            self.prices[:, i, 3] = prices  # Close prices
            self.prices[:, i, 0] = prices * 0.995  # Open
            self.prices[:, i, 1] = prices * 1.01   # High
            self.prices[:, i, 2] = prices * 0.99   # Low

        self._compute_indicators()

    def _compute_indicators(self):
        """Compute returns and volatility for all assets"""
        # Daily returns
        close = self.prices[:, :, 3]  # [n_days, n_symbols]
        self.returns = torch.zeros_like(close)
        self.returns[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-8)

        # Rolling volatility (20-day)
        self.volatility = torch.zeros_like(close)
        if self.n_days > 20:
            for i in range(20, self.n_days):
                self.volatility[i] = self.returns[i-20:i].std(dim=0)

    def reset(self, seed: int = None) -> Tuple[torch.Tensor, Dict]:
        """Reset all environments"""
        if seed is not None:
            torch.manual_seed(seed)

        # Reset portfolio
        self.capital[:] = self.initial_capital
        self.portfolio_value[:] = self.initial_capital
        self.positions[:] = 0
        self.weights[:] = 0
        self.weights[:, 0] = 1.0  # 100% cash (index 0 = CASH)

        # Reset tracking
        self.current_step[:] = 0
        self.episode_returns[:] = 0
        self.done[:] = False
        self.prev_portfolio_value[:] = self.initial_capital
        self.peak_value[:] = self.initial_capital
        self.recent_returns[:] = 0
        self.return_idx[:] = 0

        # Random starting days
        max_start = max(0, self.n_days - self.episode_length - 1)
        self.current_day_idx = torch.randint(20, max_start, (self.n_envs,),
                                              device=self.device, dtype=torch.int64)

        return self._get_observations(), {}

    def _get_observations(self) -> torch.Tensor:
        """Build observation tensor"""
        day_idx = self.current_day_idx

        # Per-asset features: normalized price, return, volatility
        current_prices = self.prices[day_idx, :, 3]  # [n_envs, n_symbols]
        current_returns = self.returns[day_idx]       # [n_envs, n_symbols]
        current_vol = self.volatility[day_idx]        # [n_envs, n_symbols]

        # Normalize prices relative to first day
        price_normalized = current_prices / (self.prices[20, :, 3] + 1e-8) - 1.0

        # Portfolio features
        portfolio_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        drawdown = (self.peak_value - self.portfolio_value) / (self.peak_value + 1e-8)

        obs = torch.cat([
            price_normalized,           # [n_envs, 3]
            current_returns * 10,       # [n_envs, 3] scaled
            current_vol * 10,           # [n_envs, 3] scaled
            self.weights,               # [n_envs, 4] current allocation
            portfolio_return.unsqueeze(1),  # [n_envs, 1]
            drawdown.unsqueeze(1),      # [n_envs, 1]
        ], dim=1)

        # Pad to obs_dim
        if obs.shape[1] < self.obs_dim:
            padding = torch.zeros((self.n_envs, self.obs_dim - obs.shape[1]), device=self.device)
            obs = torch.cat([obs, padding], dim=1)

        return obs[:, :self.obs_dim]

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Step all environments with portfolio regime actions.

        Args:
            actions: [n_envs] tensor of regime indices (0-15)

        Returns:
            observations, rewards, terminated, truncated, infos
        """
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, device=self.device, dtype=torch.int64)

        # Store previous value for reward calculation
        old_portfolio_value = self.portfolio_value.clone()
        old_weights = self.weights.clone()

        # Get current prices
        current_prices = self.prices[self.current_day_idx, :, 3]  # [n_envs, n_symbols]

        # Execute regime changes (rebalance to target weights)
        trading_costs = self._execute_regime_actions(actions, current_prices)

        # Advance time
        self.current_step += 1
        self.current_day_idx += 1

        # Get new prices and update portfolio value
        new_prices = self.prices[self.current_day_idx, :, 3]
        self._update_portfolio_value(new_prices)

        # ===== REWARD CALCULATION (Simplified: Option A) =====
        # reward_t = alpha_t - trading_cost
        # Vol/DD moved to evaluation only (not in training signal)

        portfolio_return = (self.portfolio_value - old_portfolio_value) / (old_portfolio_value + 1e-8)

        # Benchmark return (equal weight of all 3 assets)
        benchmark_return = self.returns[self.current_day_idx].mean(dim=1)

        # Alpha = portfolio return minus benchmark return
        alpha = portfolio_return - benchmark_return

        # Update rolling return buffer (for evaluation metrics, not reward)
        idx = self.return_idx % 20
        self.recent_returns[torch.arange(self.n_envs, device=self.device), idx] = portfolio_return
        self.return_idx += 1

        # Update peak value for drawdown tracking (for evaluation, not reward)
        self.peak_value = torch.maximum(self.peak_value, self.portfolio_value)

        # SIMPLIFIED REWARD: alpha - trading_cost only
        # No vol/dd penalties in the training signal
        rewards = alpha - trading_costs

        # Scale and clip rewards to O(1-10) range for PPO stability
        rewards = torch.clamp(rewards * self.reward_scale, -self.reward_clip, self.reward_clip)

        # Episode tracking
        self.episode_returns += rewards

        # Check termination
        terminated = self.current_step >= self.episode_length
        truncated = self.current_day_idx >= (self.n_days - 2)
        done = terminated | truncated

        # Reset done environments
        self._reset_done_envs(done)

        obs = self._get_observations()

        return obs, rewards, terminated, truncated, {}

    def _execute_regime_actions(self, actions: torch.Tensor, prices: torch.Tensor) -> torch.Tensor:
        """
        Execute portfolio regime changes.

        Returns trading costs incurred.
        """
        trading_costs = torch.zeros(self.n_envs, device=self.device)

        for action_idx in range(self.n_actions):
            mask = actions == action_idx
            if not mask.any():
                continue

            regime_name, target_weights = self.portfolio_regimes[action_idx]

            if target_weights is None:  # HOLD action
                continue

            # Convert to tensor
            target = torch.tensor(target_weights, device=self.device, dtype=torch.float32)

            # Get current weights for these envs
            envs = mask.nonzero(as_tuple=True)[0]
            current_w = self.weights[envs]  # [n_selected, 4]

            # Calculate weight changes
            weight_delta = target.unsqueeze(0) - current_w  # [n_selected, 4]

            # Trading cost = sum of absolute weight changes * cost rate
            turnover = weight_delta.abs().sum(dim=1)
            trading_costs[envs] = self.trading_cost * turnover

            # Execute rebalance
            self._rebalance_to_target(envs, target, prices[envs])

        return trading_costs

    def _rebalance_to_target(self, envs: torch.Tensor, target_weights: torch.Tensor, prices: torch.Tensor):
        """
        Rebalance selected environments to target weights.

        Args:
            envs: indices of environments to rebalance
            target_weights: [4] tensor of target weights [CASH, SPY, QQQ, IWM]
            prices: [n_selected, n_symbols] current prices (SPY, QQQ, IWM)
        """
        # Get current portfolio value for these envs
        pv = self.portfolio_value[envs]  # [n_selected]

        # Calculate target dollar amounts per asset
        # target_weights: [CASH, SPY, QQQ, IWM] = indices [0, 1, 2, 3]
        target_values = pv.unsqueeze(1) * target_weights.unsqueeze(0)  # [n_selected, 4]

        # Calculate target shares for each equity
        # positions[:, 0] = SPY, prices[:, 0] = SPY -> target_weights[1]
        # positions[:, 1] = QQQ, prices[:, 1] = QQQ -> target_weights[2]
        # positions[:, 2] = IWM, prices[:, 2] = IWM -> target_weights[3]
        for i in range(self.n_symbols):  # 0=SPY, 1=QQQ, 2=IWM
            weight_idx = i + 1  # Map: 0->1 (SPY), 1->2 (QQQ), 2->3 (IWM)
            target_shares = target_values[:, weight_idx] / (prices[:, i] + 1e-8)

            # Update position (simplified: instant rebalance)
            old_shares = self.positions[envs, i]
            delta_shares = target_shares - old_shares

            # Update cash: sell gives cash, buy takes cash
            self.capital[envs] -= delta_shares * prices[:, i]

            # Update positions
            self.positions[envs, i] = target_shares

        # Update weights
        self.weights[envs] = target_weights.unsqueeze(0).expand(len(envs), -1)

    def _update_portfolio_value(self, prices: torch.Tensor):
        """Update portfolio value based on current prices"""
        # Position value: sum of shares * prices
        position_values = (self.positions * prices).sum(dim=1)

        # Total value = cash + positions
        self.portfolio_value = self.capital + position_values

        # Update weights: [CASH, SPY, QQQ, IWM]
        # weights[:, 0] = CASH, weights[:, 1] = SPY, etc.
        self.weights[:, 0] = self.capital / (self.portfolio_value + 1e-8)  # CASH
        for i in range(self.n_symbols):
            self.weights[:, i + 1] = (self.positions[:, i] * prices[:, i]) / (self.portfolio_value + 1e-8)

    def _reset_done_envs(self, done_mask: torch.Tensor):
        """Reset environments that are done"""
        n_done = done_mask.sum().item()
        if n_done == 0:
            return

        done_idx = done_mask.nonzero(as_tuple=True)[0]

        self.capital[done_idx] = self.initial_capital
        self.portfolio_value[done_idx] = self.initial_capital
        self.positions[done_idx] = 0
        self.weights[done_idx] = 0
        self.weights[done_idx, 0] = 1.0  # 100% cash (index 0 = CASH)
        self.current_step[done_idx] = 0
        self.episode_returns[done_idx] = 0
        self.done[done_idx] = False
        self.prev_portfolio_value[done_idx] = self.initial_capital
        self.peak_value[done_idx] = self.initial_capital
        self.recent_returns[done_idx] = 0
        self.return_idx[done_idx] = 0

        # New random starting days
        max_start = max(0, self.n_days - self.episode_length - 1)
        self.current_day_idx[done_idx] = torch.randint(
            20, max_start, (n_done,), device=self.device, dtype=torch.int64
        )

    def close(self):
        """Cleanup"""
        pass


# ============================================================================
# Helper function to print regime info
# ============================================================================
def print_regime_info():
    """Print all available portfolio regimes"""
    print("\n" + "="*60)
    print("PORTFOLIO REGIMES (Actions)")
    print("Weights: [CASH, SPY, QQQ, IWM]")
    print("="*60)
    for idx, (name, weights) in PORTFOLIO_REGIMES.items():
        if weights is None:
            print(f"  {idx:2d}: {name:20s} -> Keep current allocation")
        else:
            w_str = f"CASH={weights[0]:.0%} SPY={weights[1]:.0%} QQQ={weights[2]:.0%} IWM={weights[3]:.0%}"
            print(f"  {idx:2d}: {name:20s} -> {w_str}")
    print("="*60)


if __name__ == "__main__":
    # Test the environment
    print_regime_info()

    env = MultiAssetEnvironment(n_envs=16, cache_path=None)
    obs, _ = env.reset()
    print(f"\nObs shape: {obs.shape}")

    # Test a few steps
    for step in range(5):
        actions = torch.randint(0, N_ACTIONS, (16,), device=env.device)
        obs, rewards, term, trunc, _ = env.step(actions)
        print(f"Step {step+1}: reward={rewards.mean():.4f}, weights[0]={env.weights[0].tolist()}")

