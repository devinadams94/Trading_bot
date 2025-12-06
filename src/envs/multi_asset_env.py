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
# SCALABLE PORTFOLIO REGIME DEFINITIONS
# ============================================================================
# Design Philosophy: Factor-based regimes that scale to ANY number of stocks
#
# Instead of per-stock regimes (ALL_SPY, ALL_QQQ), we use FACTOR TILTS that
# work with 3 stocks or 300 stocks. The weights are computed dynamically
# based on factor scores.
#
# Action Space (12 regimes):
#   - 3 Cash levels: FULL_CASH, DEFENSIVE, RISK_ON
#   - 5 Factor tilts: EQUAL, MOMENTUM, LOW_VOL, QUALITY, SIZE
#   - 3 Mixed strategies: combinations
#   - 1 HOLD action
# ============================================================================

# Regime type constants
REGIME_TYPE_HOLD = "hold"
REGIME_TYPE_CASH = "cash"
REGIME_TYPE_FACTOR = "factor"
REGIME_TYPE_MIXED = "mixed"

PORTFOLIO_REGIMES = {
    # Index: (name, type, params)
    # params dict keys:
    #   - cash_pct: fixed cash allocation (0.0 to 1.0)
    #   - factor: "equal", "momentum", "low_vol", "quality", "size"
    #   - top_k_pct: what % of stocks to overweight (0.3 = top 30%)

    # === HOLD (reduces turnover) ===
    0:  ("HOLD",         REGIME_TYPE_HOLD,   {}),

    # === CASH LEVELS (risk management) ===
    1:  ("FULL_CASH",    REGIME_TYPE_CASH,   {"cash_pct": 1.0}),   # 100% cash
    2:  ("DEFENSIVE",    REGIME_TYPE_CASH,   {"cash_pct": 0.5}),   # 50% cash + 50% equal
    3:  ("LIGHT_CASH",   REGIME_TYPE_CASH,   {"cash_pct": 0.25}),  # 25% cash + 75% equal

    # === FACTOR TILTS (scale to any universe) ===
    4:  ("EQUAL_WEIGHT", REGIME_TYPE_FACTOR, {"factor": "equal"}),
    5:  ("MOMENTUM",     REGIME_TYPE_FACTOR, {"factor": "momentum", "top_k_pct": 0.5}),
    6:  ("LOW_VOL",      REGIME_TYPE_FACTOR, {"factor": "low_vol", "top_k_pct": 0.5}),
    7:  ("QUALITY",      REGIME_TYPE_FACTOR, {"factor": "quality", "top_k_pct": 0.5}),
    8:  ("SIZE_SMALL",   REGIME_TYPE_FACTOR, {"factor": "size_small", "top_k_pct": 0.5}),

    # === MIXED STRATEGIES ===
    9:  ("SAFE_MOMENTUM", REGIME_TYPE_MIXED, {"cash_pct": 0.3, "factor": "momentum", "top_k_pct": 0.5}),
    10: ("SAFE_QUALITY",  REGIME_TYPE_MIXED, {"cash_pct": 0.3, "factor": "quality", "top_k_pct": 0.5}),
    11: ("RISK_ON",       REGIME_TYPE_MIXED, {"cash_pct": 0.0, "factor": "momentum", "top_k_pct": 0.3}),
}

N_ACTIONS = len(PORTFOLIO_REGIMES)

# Legacy regime format for backward compatibility (will be auto-generated)
# This maps factor regimes to concrete weights at runtime based on factor scores
LEGACY_REGIME_WEIGHTS = None  # Computed dynamically per step


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
        # ===== REALISM PARAMETERS =====
        slippage: float = 0.0005,         # Max slippage: 0.05% random deviation
        bid_ask_spread: float = 0.0002,   # 0.02% bid-ask spread cost (ETFs are tight)
        market_impact: float = 0.01,      # Market impact coefficient (low for liquid ETFs)
        use_t1_prices: bool = True,       # Use T-1 prices for decisions (realistic)
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

        # ===== REALISM PARAMETERS =====
        self.slippage = slippage                      # Max random slippage
        self.bid_ask_spread = bid_ask_spread          # Bid-ask spread cost
        self.market_impact = market_impact            # Market impact coefficient
        self.use_t1_prices = use_t1_prices            # Use T-1 prices for observations

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

        # Trade tracking metrics (new)
        self.trade_count = torch.zeros(self.n_envs, device=self.device, dtype=torch.int64)
        self.winning_trades = torch.zeros(self.n_envs, device=self.device, dtype=torch.int64)
        self.total_trade_return = torch.zeros(self.n_envs, device=self.device, dtype=torch.float32)
        self.pre_trade_value = self.capital.clone()  # Value before last trade
        self.in_position = torch.zeros(self.n_envs, device=self.device, dtype=torch.bool)  # Currently in non-cash position

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
        """Compute returns, volatility, and factor scores for all assets"""
        # Daily returns
        close = self.prices[:, :, 3]  # [n_days, n_symbols]
        self.returns = torch.zeros_like(close)
        self.returns[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-8)

        # Rolling volatility (20-day)
        self.volatility = torch.zeros_like(close)
        if self.n_days > 20:
            for i in range(20, self.n_days):
                self.volatility[i] = self.returns[i-20:i].std(dim=0)

        # ========== FACTOR SCORES (for factor-based regimes) ==========
        # These are computed once and stored for fast lookup during training

        # Momentum: 20-day cumulative return (higher = better momentum)
        self.momentum_score = torch.zeros_like(close)
        if self.n_days > 20:
            for i in range(20, self.n_days):
                self.momentum_score[i] = (close[i] / (close[i-20] + 1e-8)) - 1.0

        # Low Volatility: inverse of 20-day volatility (higher = lower vol = better)
        self.low_vol_score = torch.zeros_like(close)
        if self.n_days > 20:
            self.low_vol_score[20:] = 1.0 / (self.volatility[20:] + 1e-4)

        # Quality: Sharpe-like ratio = momentum / volatility (higher = better risk-adj return)
        self.quality_score = torch.zeros_like(close)
        if self.n_days > 20:
            self.quality_score[20:] = self.momentum_score[20:] / (self.volatility[20:] + 1e-4)

        # Size: For ETFs, use inverse of price as proxy (smaller price = "smaller")
        # In production with individual stocks, use market cap
        self.size_score = torch.zeros_like(close)
        self.size_score = 1.0 / (close + 1e-8)  # Inverse price as size proxy

        logger.info(f"   Factor scores computed: momentum, low_vol, quality, size")

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

        # Reset trade tracking
        self.trade_count[:] = 0
        self.winning_trades[:] = 0
        self.total_trade_return[:] = 0
        self.pre_trade_value[:] = self.initial_capital
        self.in_position[:] = False

        # Random starting days
        max_start = max(0, self.n_days - self.episode_length - 1)
        self.current_day_idx = torch.randint(20, max_start, (self.n_envs,),
                                              device=self.device, dtype=torch.int64)

        return self._get_observations(), {}

    def _get_observations(self) -> torch.Tensor:
        """
        Build observation tensor.

        REALISM: Uses T-1 prices for observations (agent can't see today's close
        when making decisions). This prevents lookahead bias.
        """
        day_idx = self.current_day_idx

        # ===== T-1 PRICE FIX: Use yesterday's data for observations =====
        # In real trading, you make decisions at market open based on yesterday's close
        if self.use_t1_prices:
            obs_day_idx = torch.clamp(day_idx - 1, min=20)  # Use T-1, min day 20
        else:
            obs_day_idx = day_idx  # Legacy: use current day (unrealistic)

        # Per-asset features: normalized price, return, volatility (from T-1)
        obs_prices = self.prices[obs_day_idx, :, 3]   # [n_envs, n_symbols]
        obs_returns = self.returns[obs_day_idx]       # [n_envs, n_symbols]
        obs_vol = self.volatility[obs_day_idx]        # [n_envs, n_symbols]

        # Normalize prices relative to base day
        price_normalized = obs_prices / (self.prices[20, :, 3] + 1e-8) - 1.0

        # Portfolio features (current, not T-1)
        portfolio_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        drawdown = (self.peak_value - self.portfolio_value) / (self.peak_value + 1e-8)

        obs = torch.cat([
            price_normalized,           # [n_envs, 3] - uses T-1 prices if enabled
            obs_returns * 10,           # [n_envs, 3] scaled - uses T-1
            obs_vol * 10,               # [n_envs, 3] scaled - uses T-1
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

        # ===== TRADE TRACKING (for metrics) =====
        # Detect position changes (entering or exiting positions)
        was_in_position = self.in_position.clone()
        is_now_in_position = (self.weights[:, 0] < 0.5)  # Less than 50% cash = in position

        # Trade completed: was in position, now mostly cash OR changed position significantly
        position_exit = was_in_position & ~is_now_in_position
        position_change = (self.weights[:, 1:].argmax(dim=1) != old_weights[:, 1:].argmax(dim=1)) & was_in_position & is_now_in_position
        trade_completed = position_exit | position_change

        # Calculate trade return for completed trades
        if trade_completed.any():
            trade_return = (self.portfolio_value[trade_completed] - self.pre_trade_value[trade_completed]) / (self.pre_trade_value[trade_completed] + 1e-8)
            self.total_trade_return[trade_completed] += trade_return
            self.trade_count[trade_completed] += 1
            self.winning_trades[trade_completed] += (trade_return > 0).long()

        # Mark new position entries
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

        # Build info dict with trade metrics
        info = {
            'trades_completed': self.trade_count.sum().item(),
            'winning_trades': self.winning_trades.sum().item(),
            'avg_trade_return': (self.total_trade_return.sum() / max(self.trade_count.sum(), 1)).item(),
        }

        return obs, rewards, terminated, truncated, info

    def _compute_factor_weights(self, day_indices: torch.Tensor, regime_params: dict) -> torch.Tensor:
        """
        Compute target weights based on factor scores for given day indices.

        Args:
            day_indices: [n_envs] tensor of current day indices
            regime_params: dict with 'cash_pct', 'factor', 'top_k_pct'

        Returns:
            [n_envs, n_assets] tensor of target weights [CASH, SPY, QQQ, IWM]
        """
        n_selected = len(day_indices)
        cash_pct = regime_params.get("cash_pct", 0.0)
        factor = regime_params.get("factor", "equal")
        top_k_pct = regime_params.get("top_k_pct", 1.0)  # 1.0 = use all stocks

        # Start with cash allocation
        weights = torch.zeros((n_selected, self.n_assets), device=self.device)
        weights[:, 0] = cash_pct  # Cash weight

        equity_pct = 1.0 - cash_pct
        if equity_pct <= 0:
            return weights

        # Get factor scores for the current days
        if factor == "equal":
            # Equal weight across all equities
            equity_weight = equity_pct / self.n_symbols
            weights[:, 1:] = equity_weight
        else:
            # Get factor scores: [n_selected, n_symbols]
            if factor == "momentum":
                scores = self.momentum_score[day_indices]  # Higher = better
            elif factor == "low_vol":
                scores = self.low_vol_score[day_indices]   # Higher = lower vol
            elif factor == "quality":
                scores = self.quality_score[day_indices]   # Higher = better risk-adj
            elif factor == "size_small":
                scores = self.size_score[day_indices]      # Higher = smaller
            else:
                # Default to equal weight
                scores = torch.ones((n_selected, self.n_symbols), device=self.device)

            # Compute weights based on scores
            # Option 1: Top-K weighting (overweight top performers)
            if top_k_pct < 1.0:
                k = max(1, int(self.n_symbols * top_k_pct))
                # Get top-k mask
                _, top_indices = scores.topk(k, dim=1)
                mask = torch.zeros_like(scores, dtype=torch.bool)
                mask.scatter_(1, top_indices, True)
                # Zero out non-top-k
                scores = scores * mask.float()

            # Normalize scores to weights (softmax-like but simpler)
            scores = scores.clamp(min=0)  # Ensure non-negative
            score_sum = scores.sum(dim=1, keepdim=True) + 1e-8
            equity_weights = (scores / score_sum) * equity_pct

            weights[:, 1:] = equity_weights

        return weights

    def _execute_regime_actions(self, actions: torch.Tensor, prices: torch.Tensor) -> torch.Tensor:
        """
        Execute portfolio regime changes using factor-based regimes.

        Returns trading costs incurred.
        """
        trading_costs = torch.zeros(self.n_envs, device=self.device)

        for action_idx in range(self.n_actions):
            mask = actions == action_idx
            if not mask.any():
                continue

            regime_name, regime_type, regime_params = self.portfolio_regimes[action_idx]

            if regime_type == REGIME_TYPE_HOLD:
                continue  # HOLD = don't rebalance

            envs = mask.nonzero(as_tuple=True)[0]
            current_w = self.weights[envs]  # [n_selected, 4]

            # Compute target weights based on regime type
            if regime_type == REGIME_TYPE_CASH:
                # Pure cash regime
                cash_pct = regime_params.get("cash_pct", 1.0)
                target = torch.zeros((len(envs), self.n_assets), device=self.device)
                target[:, 0] = cash_pct
                # Distribute remaining to equal weight
                if cash_pct < 1.0:
                    equity_pct = (1.0 - cash_pct) / self.n_symbols
                    target[:, 1:] = equity_pct
            elif regime_type in (REGIME_TYPE_FACTOR, REGIME_TYPE_MIXED):
                # Factor-based weighting
                target = self._compute_factor_weights(self.current_day_idx[envs], regime_params)
            else:
                continue

            # Calculate weight changes and trading costs
            weight_delta = target - current_w
            turnover = weight_delta.abs().sum(dim=1)
            trading_costs[envs] = self.trading_cost * turnover

            # Execute rebalance for each env (target is now per-env)
            self._rebalance_to_target_batch(envs, target, prices[envs])

        return trading_costs

    def _rebalance_to_target_batch(self, envs: torch.Tensor, target_weights: torch.Tensor, prices: torch.Tensor):
        """
        Rebalance selected environments to target weights (batch version).

        REALISM FEATURES:
        1. Slippage: Random price deviation (0-0.1% default)
        2. Bid-ask spread: Fixed cost per trade (0.05% default)
        3. Market impact: sqrt(order_size) impact (simplified)

        Args:
            envs: [n_selected] indices of environments to rebalance
            target_weights: [n_selected, 4] tensor of target weights [CASH, SPY, QQQ, IWM]
            prices: [n_selected, n_symbols] current prices (SPY, QQQ, IWM)
        """
        n_selected = len(envs)

        # Get current portfolio value for these envs
        pv = self.portfolio_value[envs]  # [n_selected]

        # Calculate target dollar amounts per asset
        target_values = pv.unsqueeze(1) * target_weights  # [n_selected, 4]

        total_slippage_cost = torch.zeros(n_selected, device=self.device)
        total_spread_cost = torch.zeros(n_selected, device=self.device)
        total_impact_cost = torch.zeros(n_selected, device=self.device)

        # Calculate target shares for each equity
        for i in range(self.n_symbols):  # 0=SPY, 1=QQQ, 2=IWM
            weight_idx = i + 1  # Map: 0->1 (SPY), 1->2 (QQQ), 2->3 (IWM)
            base_price = prices[:, i]

            # ===== 1. SLIPPAGE: Random price deviation =====
            # Buying gets worse price (higher), selling gets worse price (lower)
            slippage_pct = torch.rand(n_selected, device=self.device) * self.slippage

            old_shares = self.positions[envs, i]
            target_shares_approx = target_values[:, weight_idx] / (base_price + 1e-8)
            delta_shares = target_shares_approx - old_shares

            # Buying: price goes up (1 + slip), Selling: price goes down (1 - slip)
            is_buying = delta_shares > 0
            slippage_factor = torch.where(is_buying, 1 + slippage_pct, 1 - slippage_pct)
            executed_price = base_price * slippage_factor

            # ===== 2. BID-ASK SPREAD: Fixed cost per trade =====
            # Always pay half the spread when crossing
            spread_cost = self.bid_ask_spread * delta_shares.abs() * base_price
            total_spread_cost += spread_cost

            # ===== 3. MARKET IMPACT: sqrt(order_size) impact =====
            # Impact is proportional to sqrt of trade size relative to typical volume
            # Simplified: impact = coefficient * sqrt(|delta_value| / portfolio_value)
            delta_value = delta_shares.abs() * base_price
            relative_size = delta_value / (pv + 1e-8)
            impact_pct = self.market_impact * torch.sqrt(relative_size + 1e-8)
            impact_cost = impact_pct * delta_value
            total_impact_cost += impact_cost

            # Recalculate target shares with executed price
            target_shares = target_values[:, weight_idx] / (executed_price + 1e-8)
            delta_shares = target_shares - old_shares

            # Update cash: includes slippage (executed_price != base_price)
            self.capital[envs] -= delta_shares * executed_price

            # Deduct spread and impact costs from cash
            self.capital[envs] -= spread_cost + impact_cost

            # Update positions
            self.positions[envs, i] = target_shares

        # Update weights
        self.weights[envs] = target_weights

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
    """Print all available portfolio regimes (factor-based)"""
    print("\n" + "="*70)
    print("FACTOR-BASED PORTFOLIO REGIMES (Scalable to any stock universe)")
    print("="*70)
    print(f"{'Idx':<4} {'Name':<16} {'Type':<8} {'Description':<40}")
    print("-"*70)

    descriptions = {
        "HOLD": "Keep current allocation (reduce turnover)",
        "FULL_CASH": "100% cash - full risk-off",
        "DEFENSIVE": "50% cash + 50% equal weight equities",
        "LIGHT_CASH": "25% cash + 75% equal weight equities",
        "EQUAL_WEIGHT": "Equal weight all stocks in universe",
        "MOMENTUM": "Overweight top 50% by 20-day momentum",
        "LOW_VOL": "Overweight top 50% by lowest volatility",
        "QUALITY": "Overweight top 50% by Sharpe ratio",
        "SIZE_SMALL": "Overweight smaller stocks (by price proxy)",
        "SAFE_MOMENTUM": "30% cash + 70% momentum tilt",
        "SAFE_QUALITY": "30% cash + 70% quality tilt",
        "RISK_ON": "0% cash + aggressive momentum (top 30%)",
    }

    for idx, (name, regime_type, params) in PORTFOLIO_REGIMES.items():
        desc = descriptions.get(name, str(params))
        print(f"  {idx:<2}  {name:<16} {regime_type:<8} {desc:<40}")

    print("="*70)
    print("Note: Factor weights are computed DYNAMICALLY based on current scores.")
    print("      This design scales to 3 stocks or 300 stocks without changes.")
    print("="*70)


if __name__ == "__main__":
    # Test the environment
    print_regime_info()

    print("\n" + "="*70)
    print("REALISM PARAMETERS")
    print("="*70)

    env = MultiAssetEnvironment(n_envs=16, cache_path=None)
    print(f"  Slippage:       {env.slippage*100:.2f}% (max random price deviation)")
    print(f"  Bid-ask spread: {env.bid_ask_spread*100:.3f}% (per trade)")
    print(f"  Market impact:  {env.market_impact:.1f} × sqrt(trade_size)")
    print(f"  T-1 prices:     {env.use_t1_prices} (realistic lookahead prevention)")

    obs, _ = env.reset()
    print(f"\nObs shape: {obs.shape}")
    print(f"N actions: {N_ACTIONS}")

    # Test each regime with fresh reset
    print("\nTesting each regime (with reset between):")
    for action_idx in range(N_ACTIONS):
        env.reset()  # Fresh start each time
        actions = torch.full((16,), action_idx, device=env.device, dtype=torch.int64)
        obs, rewards, term, trunc, _ = env.step(actions)
        regime_name = PORTFOLIO_REGIMES[action_idx][0]
        weights = env.weights[0].cpu().numpy()
        pv = env.portfolio_value[0].item()
        cost_pct = (100000 - pv) / 100000 * 100  # Trading cost as % of initial
        print(f"  {action_idx:2d} {regime_name:16s}: w=[{weights[0]:.2f}, {weights[1]:.2f}, {weights[2]:.2f}, {weights[3]:.2f}] cost={cost_pct:+.3f}%")

