#!/usr/bin/env python3
"""
Expanded Multi-Asset Portfolio Environment for RL Trading

Supports 12+ tradeable assets plus CASH.
Uses dynamically generated portfolio regimes based on available symbols.

Default symbols (most traded options stocks):
- ETFs: SPY, QQQ, IWM
- Tech: AAPL, MSFT, NVDA, TSLA, AMZN, META, GOOGL, AMD, NFLX
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import logging
import itertools

logger = logging.getLogger(__name__)


def generate_portfolio_regimes(symbols: List[str]) -> Dict[int, Tuple[str, Optional[List[float]]]]:
    """
    Dynamically generate portfolio regimes based on available symbols.

    Generates regimes for:
    - HOLD (keep current)
    - ALL_CASH (100% cash)
    - Single asset (100% in one stock)
    - Pairs (50/50 splits for key combinations)
    - Sector tilts (tech heavy, diversified, defensive, etc.)

    Returns dict mapping action_id -> (name, weights) where weights is [CASH, sym1, sym2, ...]
    """
    n_symbols = len(symbols)
    n_assets = n_symbols + 1  # +1 for CASH
    regimes = {}
    action_id = 0

    # 0: HOLD - keep current allocation
    regimes[action_id] = ("HOLD", None)
    action_id += 1

    # 1: ALL_CASH - 100% cash (risk-off)
    weights = [0.0] * n_assets
    weights[0] = 1.0
    regimes[action_id] = ("ALL_CASH", weights)
    action_id += 1

    # Single asset allocations (100% in one stock)
    for i, symbol in enumerate(symbols):
        weights = [0.0] * n_assets
        weights[i + 1] = 1.0  # +1 because index 0 is CASH
        regimes[action_id] = (f"ALL_{symbol}", weights)
        action_id += 1

    # Equal weight all assets (including some cash for safety)
    weights = [0.1] + [0.9 / n_symbols] * n_symbols
    regimes[action_id] = ("EQUAL_WEIGHT", weights)
    action_id += 1

    # Tech heavy (if we have tech stocks)
    tech_stocks = ['AAPL', 'MSFT', 'NVDA', 'AMD', 'GOOGL', 'META', 'AMZN', 'NFLX', 'TSLA', 'QQQ']
    tech_in_symbols = [s for s in symbols if s in tech_stocks]
    if len(tech_in_symbols) >= 3:
        weights = [0.1] + [0.0] * n_symbols
        weight_per_tech = 0.9 / len(tech_in_symbols)
        for s in tech_in_symbols:
            if s in symbols:
                weights[symbols.index(s) + 1] = weight_per_tech
        regimes[action_id] = ("TECH_HEAVY", weights)
        action_id += 1

    # Defensive (high cash + stable stocks like SPY)
    weights = [0.5] + [0.0] * n_symbols
    if 'SPY' in symbols:
        weights[symbols.index('SPY') + 1] = 0.3
    if 'QQQ' in symbols:
        weights[symbols.index('QQQ') + 1] = 0.2
    else:
        weights[0] = 0.7  # More cash if no QQQ
        if 'SPY' in symbols:
            weights[symbols.index('SPY') + 1] = 0.3
    regimes[action_id] = ("DEFENSIVE", weights)
    action_id += 1

    # Risk-off 75% cash
    weights = [0.75] + [0.0] * n_symbols
    if 'SPY' in symbols:
        weights[symbols.index('SPY') + 1] = 0.25
    regimes[action_id] = ("RISK_OFF_75", weights)
    action_id += 1

    # Growth tilt (QQQ + growth stocks)
    if 'QQQ' in symbols:
        weights = [0.1] + [0.0] * n_symbols
        weights[symbols.index('QQQ') + 1] = 0.4
        growth = ['NVDA', 'TSLA', 'AMD']
        available_growth = [g for g in growth if g in symbols]
        if available_growth:
            per_stock = 0.5 / len(available_growth)
            for g in available_growth:
                weights[symbols.index(g) + 1] = per_stock
        regimes[action_id] = ("GROWTH_TILT", weights)
        action_id += 1

    # Value/Large cap tilt (SPY heavy)
    if 'SPY' in symbols:
        weights = [0.1] + [0.0] * n_symbols
        weights[symbols.index('SPY') + 1] = 0.5
        large_cap = ['AAPL', 'MSFT', 'GOOGL']
        available_large = [l for l in large_cap if l in symbols]
        if available_large:
            per_stock = 0.4 / len(available_large)
            for l in available_large:
                weights[symbols.index(l) + 1] = per_stock
        regimes[action_id] = ("LARGE_CAP_TILT", weights)
        action_id += 1

    # Momentum plays (high beta stocks)
    momentum = ['NVDA', 'TSLA', 'AMD', 'META']
    avail_momentum = [m for m in momentum if m in symbols]
    if len(avail_momentum) >= 2:
        weights = [0.1] + [0.0] * n_symbols
        per_stock = 0.9 / len(avail_momentum)
        for m in avail_momentum:
            weights[symbols.index(m) + 1] = per_stock
        regimes[action_id] = ("MOMENTUM", weights)
        action_id += 1

    # FAANG-like (big tech)
    faang = ['META', 'AAPL', 'AMZN', 'NFLX', 'GOOGL']
    avail_faang = [f for f in faang if f in symbols]
    if len(avail_faang) >= 3:
        weights = [0.1] + [0.0] * n_symbols
        per_stock = 0.9 / len(avail_faang)
        for f in avail_faang:
            weights[symbols.index(f) + 1] = per_stock
        regimes[action_id] = ("FAANG", weights)
        action_id += 1

    # Semiconductor focus
    semis = ['NVDA', 'AMD']
    avail_semis = [s for s in semis if s in symbols]
    if len(avail_semis) == 2:
        weights = [0.1] + [0.0] * n_symbols
        weights[symbols.index('NVDA') + 1] = 0.5
        weights[symbols.index('AMD') + 1] = 0.4
        regimes[action_id] = ("SEMICONDUCTORS", weights)
        action_id += 1

    # Balanced portfolio
    weights = [0.2] + [0.8 / n_symbols] * n_symbols
    regimes[action_id] = ("BALANCED", weights)
    action_id += 1

    return regimes


def print_regime_info(regimes: Dict, symbols: List[str]):
    """Print portfolio regime information"""
    print("\n" + "=" * 80)
    print("PORTFOLIO REGIMES (Actions)")
    header = "Weights: [CASH, " + ", ".join(symbols) + "]"
    print(header)
    print("=" * 80)

    for action_id, (name, weights) in regimes.items():
        if weights is None:
            print(f"  {action_id:3d}: {name:20s} -> Keep current allocation")
        else:
            weight_str = " ".join([f"{w*100:.0f}%" for w in weights[:min(6, len(weights))]])
            if len(weights) > 6:
                weight_str += " ..."
            print(f"  {action_id:3d}: {name:20s} -> {weight_str}")
    print("=" * 80)


class ExpandedMultiAssetEnvironment:
    """
    Expanded multi-asset portfolio environment supporting 12+ assets.

    Assets (weight order): [CASH, symbol_1, symbol_2, ..., symbol_n]

    Default symbols: SPY, QQQ, IWM, AAPL, MSFT, NVDA, TSLA, AMZN, META, GOOGL, AMD, NFLX
    """

    # Default symbols - most traded options stocks
    DEFAULT_SYMBOLS = [
        "SPY", "QQQ", "IWM",  # ETFs
        "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "AMD", "NFLX"
    ]

    def __init__(
        self,
        n_envs: int = 256,
        initial_capital: float = 100000,
        episode_length: int = 256,
        device: str = 'cuda',
        cache_path: str = None,
        symbols: List[str] = None,
        trading_cost: float = 0.001,
        volatility_penalty: float = 0.5,  # For backward compat, not used in reward
        drawdown_penalty: float = 0.0,    # For backward compat, not used in reward
    ):
        self.n_envs = n_envs
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.initial_capital = initial_capital
        self.episode_length = episode_length
        self.cache_path = cache_path
        self.trading_cost = trading_cost
        # Legacy parameters (kept for backward compat, not used in reward)
        self.volatility_penalty = volatility_penalty
        self.drawdown_penalty = drawdown_penalty

        # Symbols configuration
        self.symbols = symbols or self.DEFAULT_SYMBOLS
        self.n_symbols = len(self.symbols)
        self.n_assets = self.n_symbols + 1  # +1 for CASH

        # Generate portfolio regimes dynamically
        self.portfolio_regimes = generate_portfolio_regimes(self.symbols)
        self.n_actions = len(self.portfolio_regimes)

        # Reward parameters
        self.reward_scale = 100.0
        self.reward_clip = 5.0

        # Observation dimensions
        # Per asset: price, return, volatility (3 features each)
        # Portfolio: weights (n_assets), portfolio value (1), drawdown (1)
        # Padding to reach obs_dim
        self.obs_dim = 128  # Larger for more assets

        # Pre-allocate GPU tensors
        self._init_gpu_tensors()

        # Load data
        if cache_path:
            self._load_data_to_gpu()
        else:
            self._create_synthetic_data()

        logger.info(f"✅ Expanded Multi-Asset Environment initialized on {self.device}")
        logger.info(f"   Actions: {self.n_actions} portfolio regimes")
        logger.info(f"   Assets: CASH + {', '.join(self.symbols)}")
        logger.info(f"   Parallel envs: {n_envs}")

    def _init_gpu_tensors(self):
        """Pre-allocate state tensors on GPU"""
        # Portfolio state
        self.capital = torch.full((self.n_envs,), self.initial_capital,
                                   device=self.device, dtype=torch.float32)
        self.portfolio_value = self.capital.clone()

        # Positions: [n_envs, n_symbols] = shares held
        self.positions = torch.zeros((self.n_envs, self.n_symbols),
                                      device=self.device, dtype=torch.float32)

        # Weights: [n_envs, n_assets] = [CASH, sym1, sym2, ...]
        self.weights = torch.zeros((self.n_envs, self.n_assets),
                                    device=self.device, dtype=torch.float32)
        self.weights[:, 0] = 1.0  # Start 100% cash

        # Time tracking
        self.current_step = torch.zeros(self.n_envs, device=self.device, dtype=torch.int32)
        self.current_day_idx = torch.zeros(self.n_envs, device=self.device, dtype=torch.int64)

        # Episode tracking
        self.episode_returns = torch.zeros(self.n_envs, device=self.device, dtype=torch.float32)
        self.done = torch.zeros(self.n_envs, device=self.device, dtype=torch.bool)

        # For reward calculation
        self.prev_portfolio_value = self.capital.clone()
        self.peak_value = self.capital.clone()
        self.recent_returns = torch.zeros((self.n_envs, 20), device=self.device)
        self.return_idx = torch.zeros(self.n_envs, device=self.device, dtype=torch.int64)

        # Trade tracking
        self.trade_count = torch.zeros(self.n_envs, device=self.device, dtype=torch.int64)
        self.winning_trades = torch.zeros(self.n_envs, device=self.device, dtype=torch.int64)
        self.total_trade_return = torch.zeros(self.n_envs, device=self.device, dtype=torch.float32)
        self.pre_trade_value = self.capital.clone()
        self.in_position = torch.zeros(self.n_envs, device=self.device, dtype=torch.bool)

    def _load_data_to_gpu(self):
        """Load market data from cache"""
        import os

        if not os.path.exists(self.cache_path):
            raise ValueError(f"Cache not found: {self.cache_path}")

        logger.info(f"⚡ Loading cache: {self.cache_path}")
        cache = torch.load(self.cache_path, map_location='cpu', weights_only=False)

        self.n_days = cache['n_days']

        # Update symbols list based on what's in cache
        cached_symbols = cache.get('symbols', self.symbols)
        available_symbols = [s for s in self.symbols if s in cache['stock_prices']]

        if len(available_symbols) < len(self.symbols):
            missing = set(self.symbols) - set(available_symbols)
            logger.warning(f"⚠️  Missing symbols in cache: {missing}")
            self.symbols = available_symbols
            self.n_symbols = len(self.symbols)
            self.n_assets = self.n_symbols + 1
            # Regenerate regimes for available symbols
            self.portfolio_regimes = generate_portfolio_regimes(self.symbols)
            self.n_actions = len(self.portfolio_regimes)
            # Reinitialize tensors with correct sizes
            self._init_gpu_tensors()

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

        logger.info(f"✅ Loaded {self.n_days} days, {self.n_symbols} symbols")
        logger.info(f"   GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    def _create_synthetic_data(self):
        """Create synthetic data for testing"""
        self.n_days = 1000
        self.prices = torch.zeros((self.n_days, self.n_symbols, 5),
                                   device=self.device, dtype=torch.float32)

        # Base prices for different asset types
        base_prices = {
            'SPY': 450.0, 'QQQ': 380.0, 'IWM': 200.0,
            'AAPL': 180.0, 'MSFT': 380.0, 'NVDA': 500.0,
            'TSLA': 250.0, 'AMZN': 180.0, 'META': 350.0,
            'GOOGL': 140.0, 'AMD': 150.0, 'NFLX': 450.0
        }

        for i, symbol in enumerate(self.symbols):
            base = base_prices.get(symbol, 100.0)
            returns = torch.randn(self.n_days, device=self.device) * 0.015
            prices = base * torch.cumprod(1 + returns, dim=0)
            self.prices[:, i, 3] = prices  # Close
            self.prices[:, i, 0] = prices * 0.995  # Open
            self.prices[:, i, 1] = prices * 1.01   # High
            self.prices[:, i, 2] = prices * 0.99   # Low

        self._compute_indicators()

    def _compute_indicators(self):
        """Compute returns and volatility for all assets"""
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

        # Reset portfolio state
        self.capital.fill_(self.initial_capital)
        self.portfolio_value.copy_(self.capital)
        self.positions.zero_()
        self.weights.zero_()
        self.weights[:, 0] = 1.0  # 100% cash

        # Reset time
        self.current_step.zero_()
        max_start = max(21, self.n_days - self.episode_length - 10)
        self.current_day_idx = torch.randint(21, max_start, (self.n_envs,),
                                              device=self.device, dtype=torch.int64)

        # Reset tracking
        self.episode_returns.zero_()
        self.done.zero_()
        self.prev_portfolio_value.copy_(self.capital)
        self.peak_value.copy_(self.capital)
        self.recent_returns.zero_()
        self.return_idx.zero_()

        # Reset trade tracking
        self.trade_count.zero_()
        self.winning_trades.zero_()
        self.total_trade_return.zero_()
        self.pre_trade_value.copy_(self.capital)
        self.in_position.zero_()

        obs = self._get_observations()
        return obs, {}

    def _get_observations(self) -> torch.Tensor:
        """Build observation tensor"""
        obs = torch.zeros((self.n_envs, self.obs_dim), device=self.device)

        day_idx = self.current_day_idx.long()

        # Get current prices and compute features
        current_prices = self.prices[day_idx, :, 3]  # [n_envs, n_symbols]
        base_prices = self.prices[20, :, 3].unsqueeze(0)  # [1, n_symbols]

        # Normalized prices
        price_norm = (current_prices / (base_prices + 1e-8)) - 1.0

        # Returns
        returns = self.returns[day_idx]  # [n_envs, n_symbols]

        # Volatility
        vol = self.volatility[day_idx]  # [n_envs, n_symbols]

        # Build observation
        idx = 0
        # Per-asset features (3 per asset)
        for i in range(self.n_symbols):
            obs[:, idx] = price_norm[:, i]
            obs[:, idx + 1] = returns[:, i] * 10
            obs[:, idx + 2] = vol[:, i] * 10
            idx += 3

        # Portfolio weights
        obs[:, idx:idx + self.n_assets] = self.weights
        idx += self.n_assets

        # Portfolio return
        obs[:, idx] = (self.portfolio_value - self.initial_capital) / self.initial_capital
        idx += 1

        # Drawdown
        obs[:, idx] = (self.peak_value - self.portfolio_value) / (self.peak_value + 1e-8)

        return obs

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Step all environments with portfolio regime actions.

        Args:
            actions: [n_envs] tensor of regime indices

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

        # ===== REWARD CALCULATION =====
        portfolio_return = (self.portfolio_value - old_portfolio_value) / (old_portfolio_value + 1e-8)

        # Benchmark return (equal weight of all symbols)
        benchmark_return = self.returns[self.current_day_idx].mean(dim=1)

        # Alpha = portfolio return minus benchmark return
        alpha = portfolio_return - benchmark_return

        # Update rolling return buffer
        idx = self.return_idx % 20
        self.recent_returns[torch.arange(self.n_envs, device=self.device), idx] = portfolio_return
        self.return_idx += 1

        # Update peak value for drawdown tracking
        self.peak_value = torch.maximum(self.peak_value, self.portfolio_value)

        # SIMPLIFIED REWARD: alpha - trading_cost
        rewards = alpha - trading_costs

        # Scale and clip rewards
        rewards = torch.clamp(rewards * self.reward_scale, -self.reward_clip, self.reward_clip)

        # Episode tracking
        self.episode_returns += rewards

        # ===== TRADE TRACKING =====
        was_in_position = self.in_position.clone()
        is_now_in_position = (self.weights[:, 0] < 0.5)  # Less than 50% cash = in position

        position_exit = was_in_position & ~is_now_in_position
        position_change = (self.weights[:, 1:].argmax(dim=1) != old_weights[:, 1:].argmax(dim=1)) & was_in_position & is_now_in_position
        trade_completed = position_exit | position_change

        if trade_completed.any():
            trade_return = (self.portfolio_value[trade_completed] - self.pre_trade_value[trade_completed]) / (self.pre_trade_value[trade_completed] + 1e-8)
            self.total_trade_return[trade_completed] += trade_return
            self.trade_count[trade_completed] += 1
            self.winning_trades[trade_completed] += (trade_return > 0).long()

        position_entry = ~was_in_position & is_now_in_position
        self.pre_trade_value[position_entry] = old_portfolio_value[position_entry]
        self.in_position = is_now_in_position

        # Check termination
        terminated = self.current_step >= self.episode_length
        truncated = self.current_day_idx >= (self.n_days - 2)
        done = terminated | truncated

        # Reset done environments
        self._reset_done_envs(done)

        obs = self._get_observations()

        info = {
            'trades_completed': self.trade_count.sum().item(),
            'winning_trades': self.winning_trades.sum().item(),
            'avg_trade_return': (self.total_trade_return.sum() / max(self.trade_count.sum(), 1)).item(),
        }

        return obs, rewards, terminated, truncated, info

    def _execute_regime_actions(self, actions: torch.Tensor, prices: torch.Tensor) -> torch.Tensor:
        """Execute portfolio regime changes. Returns trading costs."""
        trading_costs = torch.zeros(self.n_envs, device=self.device)

        for action_idx in range(self.n_actions):
            mask = actions == action_idx
            if not mask.any():
                continue

            regime_name, target_weights = self.portfolio_regimes[action_idx]

            if target_weights is None:  # HOLD action
                continue

            target = torch.tensor(target_weights, device=self.device, dtype=torch.float32)
            envs = mask.nonzero(as_tuple=True)[0]
            current_w = self.weights[envs]

            weight_delta = target.unsqueeze(0) - current_w
            turnover = weight_delta.abs().sum(dim=1)
            trading_costs[envs] = self.trading_cost * turnover

            self._rebalance_to_target(envs, target, prices[envs])

        return trading_costs

    def _rebalance_to_target(self, envs: torch.Tensor, target_weights: torch.Tensor, prices: torch.Tensor):
        """Rebalance selected environments to target weights."""
        pv = self.portfolio_value[envs]
        target_values = pv.unsqueeze(1) * target_weights.unsqueeze(0)

        for i in range(self.n_symbols):
            weight_idx = i + 1
            target_shares = target_values[:, weight_idx] / (prices[:, i] + 1e-8)
            old_shares = self.positions[envs, i]
            delta_shares = target_shares - old_shares
            self.capital[envs] -= delta_shares * prices[:, i]
            self.positions[envs, i] = target_shares

        self.weights[envs] = target_weights.unsqueeze(0).expand(len(envs), -1)

    def _update_portfolio_value(self, prices: torch.Tensor):
        """Update portfolio value based on current prices"""
        position_values = (self.positions * prices).sum(dim=1)
        self.portfolio_value = self.capital + position_values

        self.weights[:, 0] = self.capital / (self.portfolio_value + 1e-8)
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
        self.weights[done_idx, 0] = 1.0
        self.current_step[done_idx] = 0
        self.episode_returns[done_idx] = 0
        self.done[done_idx] = False
        self.prev_portfolio_value[done_idx] = self.initial_capital
        self.peak_value[done_idx] = self.initial_capital
        self.recent_returns[done_idx] = 0
        self.return_idx[done_idx] = 0

        # Reset trade tracking for done envs
        self.trade_count[done_idx] = 0
        self.winning_trades[done_idx] = 0
        self.total_trade_return[done_idx] = 0
        self.pre_trade_value[done_idx] = self.initial_capital
        self.in_position[done_idx] = False

        max_start = max(21, self.n_days - self.episode_length - 1)
        self.current_day_idx[done_idx] = torch.randint(
            21, max_start, (n_done,), device=self.device, dtype=torch.int64
        )

    def close(self):
        """Cleanup"""
        pass


if __name__ == "__main__":
    # Test the environment
    symbols = ExpandedMultiAssetEnvironment.DEFAULT_SYMBOLS
    regimes = generate_portfolio_regimes(symbols)
    print_regime_info(regimes, symbols)

    env = ExpandedMultiAssetEnvironment(n_envs=16, cache_path=None)
    obs, _ = env.reset()
    print(f"\nObs shape: {obs.shape}")
    print(f"Actions: {env.n_actions}")
    print(f"Assets: {env.n_assets} (CASH + {env.n_symbols} symbols)")

    # Test a few steps
    for step in range(5):
        actions = torch.randint(0, env.n_actions, (16,), device=env.device)
        obs, rewards, term, trunc, info = env.step(actions)
        print(f"Step {step+1}: reward={rewards.mean():.4f}")

