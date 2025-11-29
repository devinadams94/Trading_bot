#!/usr/bin/env python3
"""
GPU-Accelerated Options Environment for H200

This environment runs the simulation logic on GPU using PyTorch tensors,
eliminating the CPU bottleneck in reinforcement learning training.

Key optimizations:
1. All state is stored as GPU tensors
2. Vectorized operations replace Python loops
3. Options data pre-loaded to GPU memory
4. Batched environment stepping
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class GPUOptionsEnvironment:
    """
    GPU-accelerated options trading environment.
    
    All data and computations happen on GPU for maximum throughput.
    """
    
    def __init__(
        self,
        data_loader=None,
        symbols: List[str] = None,
        n_envs: int = 20,
        initial_capital: float = 100000,
        max_positions: int = 5,
        episode_length: int = 256,
        device: str = 'cuda',
    ):
        self.n_envs = n_envs
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.episode_length = episode_length
        self.symbols = symbols or ['SPY', 'QQQ', 'IWM']
        self.n_symbols = len(self.symbols)
        
        # Action space: 31 actions (hold + 10 calls + 10 puts per symbol direction)
        self.n_actions = 31
        
        # Observation dimensions
        self.obs_dim = 64  # Market features + portfolio state
        
        # Pre-allocate GPU tensors for all environments
        self._init_gpu_tensors()
        
        # Load and preprocess data to GPU
        if data_loader:
            self._load_data_to_gpu(data_loader)
        else:
            self._create_synthetic_gpu_data()
        
        logger.info(f"✅ GPU Options Environment initialized on {self.device}")
        logger.info(f"   Parallel envs: {n_envs}")
        logger.info(f"   GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    def _init_gpu_tensors(self):
        """Pre-allocate all state tensors on GPU"""
        # Portfolio state [n_envs]
        self.capital = torch.full((self.n_envs,), self.initial_capital,
                                   device=self.device, dtype=torch.float32)
        self.portfolio_value = self.capital.clone()

        # Position tracking [n_envs, max_positions, features]
        # Features: symbol_idx, strike, option_type, quantity, entry_price, current_price
        self.positions = torch.zeros((self.n_envs, self.max_positions, 6),
                                      device=self.device, dtype=torch.float32)
        self.n_positions = torch.zeros(self.n_envs, device=self.device, dtype=torch.int32)
        
        # Time tracking
        self.current_step = torch.zeros(self.n_envs, device=self.device, dtype=torch.int32)
        self.current_day_idx = torch.zeros(self.n_envs, device=self.device, dtype=torch.int64)
        
        # Episode tracking
        self.episode_returns = torch.zeros(self.n_envs, device=self.device, dtype=torch.float32)
        self.done = torch.zeros(self.n_envs, device=self.device, dtype=torch.bool)
    
    def _load_data_to_gpu(self, data_source):
        """Load market data to GPU tensors

        Args:
            data_source: Either a data loader object, path to parquet file, or 'cache'
        """
        import os

        # Check for pre-built cache first (instant loading!)
        cache_path = 'data/gpu_cache.pt'
        if os.path.exists(cache_path):
            logger.info("⚡ Loading pre-built GPU cache (instant load)...")
            try:
                cache = torch.load(cache_path, map_location='cpu')

                # Load stock prices
                self.n_days = cache['n_days']
                self.prices = torch.zeros((self.n_days, self.n_symbols, 5),
                                          device=self.device, dtype=torch.float32)

                for i, symbol in enumerate(self.symbols):
                    if symbol in cache['stock_prices']:
                        stock_data = cache['stock_prices'][symbol].to(self.device)
                        n = min(len(stock_data), self.n_days)
                        self.prices[:n, i, :] = stock_data[:n]

                # Load options with Greeks
                self.options_data = {}
                for symbol in self.symbols:
                    if symbol in cache['options']:
                        self.options_data[symbol] = cache['options'][symbol].to(self.device)

                # Compute indicators
                self._compute_gpu_indicators()

                logger.info(f"✅ Loaded {self.n_days} days from cache in <1 second!")
                logger.info(f"   GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                return
            except Exception as e:
                logger.warning(f"Cache load failed: {e}, falling back to parquet")

        logger.info("📊 Loading market data to GPU...")

        import pandas as pd

        try:
            # Load data - either from path or loader
            if isinstance(data_source, str) and data_source.endswith('.parquet'):
                df = pd.read_parquet(data_source)
            elif hasattr(data_source, 'load_all_data'):
                df = data_source.load_all_data()
            else:
                # Try to find parquet file directly
                import glob
                parquet_files = glob.glob('data/flat_files_processed/*.parquet')
                if parquet_files:
                    df = pd.read_parquet(parquet_files[0])
                    logger.info(f"📂 Loaded {parquet_files[0]}")
                else:
                    raise ValueError("No data source found")

            if df.empty:
                raise ValueError("Empty dataframe")

            logger.info(f"📊 Loaded {len(df):,} options records")

            # Get unique dates and create daily OHLCV per symbol
            stock_data = {}
            for symbol in self.symbols:
                symbol_df = df[df['underlying'] == symbol].copy()
                if len(symbol_df) == 0:
                    logger.warning(f"No data for {symbol}")
                    continue

                # Aggregate to daily data using underlying price
                symbol_df['date'] = pd.to_datetime(symbol_df['trade_date_dt']).dt.date
                daily = symbol_df.groupby('date').agg(
                    open=('underlying_price', 'first'),
                    high=('underlying_price', 'max'),
                    low=('underlying_price', 'min'),
                    close=('underlying_price', 'last'),
                    volume=('volume', 'sum')
                ).reset_index()
                stock_data[symbol] = daily.sort_values('date')
                logger.info(f"   {symbol}: {len(daily)} trading days")

            if not stock_data:
                raise ValueError("No stock data extracted")

            # Find common dates
            first_symbol = list(stock_data.keys())[0]
            self.n_days = len(stock_data[first_symbol])

            # Create price tensors [n_days, n_symbols, features]
            self.prices = torch.zeros((self.n_days, self.n_symbols, 5),
                                       device=self.device, dtype=torch.float32)

            for i, symbol in enumerate(self.symbols):
                if symbol in stock_data:
                    sdf = stock_data[symbol]
                    n = min(len(sdf), self.n_days)
                    self.prices[:n, i, 0] = torch.tensor(sdf['open'].values[:n], device=self.device, dtype=torch.float32)
                    self.prices[:n, i, 1] = torch.tensor(sdf['high'].values[:n], device=self.device, dtype=torch.float32)
                    self.prices[:n, i, 2] = torch.tensor(sdf['low'].values[:n], device=self.device, dtype=torch.float32)
                    self.prices[:n, i, 3] = torch.tensor(sdf['close'].values[:n], device=self.device, dtype=torch.float32)
                    self.prices[:n, i, 4] = torch.tensor(sdf['volume'].values[:n], device=self.device, dtype=torch.float32)

            # Also load options data for more realistic simulation
            self._load_options_to_gpu(df)

            # Pre-compute technical indicators on GPU
            self._compute_gpu_indicators()

            logger.info(f"✅ Loaded {self.n_days} days of REAL Massive.io data to GPU")
            logger.info(f"   GPU Memory used: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        except Exception as e:
            import traceback
            logger.error(f"Failed to load data: {e}")
            logger.error(traceback.format_exc())
            logger.warning("Falling back to synthetic data")
            self._create_synthetic_gpu_data()

    def _load_options_to_gpu(self, df):
        """Load options chain data with Greeks to GPU for realistic pricing"""
        import pandas as pd

        # Store options data as GPU tensors for fast lookup
        # Group by date and create tensors
        # Features: strike, option_type (0=put, 1=call), delta, gamma, theta, vega, iv, price

        logger.info("📊 Loading options Greeks to GPU...")

        # For each trading day, store top N options per symbol
        max_options_per_day = 100  # Top 100 most liquid options per symbol per day

        self.options_data = {}
        for symbol in self.symbols:
            symbol_df = df[df['underlying'] == symbol].copy()
            if len(symbol_df) == 0:
                continue

            # Get unique dates
            symbol_df['date'] = pd.to_datetime(symbol_df['trade_date_dt']).dt.date
            dates = sorted(symbol_df['date'].unique())

            # Pre-allocate tensor [n_days, max_options, features]
            # Features: strike, type, delta, gamma, theta, vega, iv, mid_price, underlying_price
            n_features = 9
            options_tensor = torch.zeros(
                (len(dates), max_options_per_day, n_features),
                device=self.device, dtype=torch.float32
            )

            for day_idx, date in enumerate(dates):
                day_df = symbol_df[symbol_df['date'] == date].nlargest(max_options_per_day, 'volume')
                n_opts = min(len(day_df), max_options_per_day)

                if n_opts > 0:
                    options_tensor[day_idx, :n_opts, 0] = torch.tensor(day_df['strike'].values[:n_opts], device=self.device)
                    options_tensor[day_idx, :n_opts, 1] = torch.tensor((day_df['option_type'] == 'call').astype(float).values[:n_opts], device=self.device)
                    options_tensor[day_idx, :n_opts, 2] = torch.tensor(day_df['delta'].values[:n_opts], device=self.device)
                    options_tensor[day_idx, :n_opts, 3] = torch.tensor(day_df['gamma'].values[:n_opts], device=self.device)
                    options_tensor[day_idx, :n_opts, 4] = torch.tensor(day_df['theta'].values[:n_opts], device=self.device)
                    options_tensor[day_idx, :n_opts, 5] = torch.tensor(day_df['vega'].values[:n_opts], device=self.device)
                    options_tensor[day_idx, :n_opts, 6] = torch.tensor(day_df['iv'].values[:n_opts], device=self.device)
                    options_tensor[day_idx, :n_opts, 7] = torch.tensor(day_df['option_price'].values[:n_opts], device=self.device)
                    options_tensor[day_idx, :n_opts, 8] = torch.tensor(day_df['underlying_price'].values[:n_opts], device=self.device)

            self.options_data[symbol] = options_tensor
            logger.info(f"   {symbol}: {options_tensor.shape} options tensor")
    
    def _create_synthetic_gpu_data(self):
        """Create synthetic data for testing"""
        self.n_days = 1000
        self.prices = torch.zeros((self.n_days, self.n_symbols, 5), 
                                   device=self.device, dtype=torch.float32)
        
        # Generate random walk prices
        for i in range(self.n_symbols):
            base_price = 100 + i * 50
            returns = torch.randn(self.n_days, device=self.device) * 0.02
            prices = base_price * torch.exp(torch.cumsum(returns, dim=0))
            self.prices[:, i, 3] = prices  # Close
            self.prices[:, i, 0] = prices * (1 + torch.rand(self.n_days, device=self.device) * 0.01)
            self.prices[:, i, 1] = prices * (1 + torch.rand(self.n_days, device=self.device) * 0.02)
            self.prices[:, i, 2] = prices * (1 - torch.rand(self.n_days, device=self.device) * 0.02)
            self.prices[:, i, 4] = torch.rand(self.n_days, device=self.device) * 1e6
        
        self._compute_gpu_indicators()
    
    def _compute_gpu_indicators(self):
        """Compute technical indicators on GPU"""
        # RSI, MACD, etc. computed vectorized on GPU
        closes = self.prices[:, :, 3]  # [n_days, n_symbols]
        
        # Simple returns
        self.returns = torch.zeros_like(closes)
        self.returns[1:] = (closes[1:] - closes[:-1]) / closes[:-1]
        
        # Volatility (20-day rolling std) - vectorized
        self.volatility = torch.zeros_like(closes)
        if self.n_days > 20:
            # Use unfold for vectorized rolling window
            returns_unfolded = self.returns.unfold(0, 20, 1)  # [n_days-19, n_symbols, 20]
            self.volatility[20:] = returns_unfolded.std(dim=2)[:-1] if returns_unfolded.shape[0] > 1 else returns_unfolded.std(dim=2)

    def reset(self, seed: int = None) -> Tuple[torch.Tensor, Dict]:
        """Reset all environments - fully vectorized on GPU"""
        if seed is not None:
            torch.manual_seed(seed)

        # Reset portfolio state
        self.capital.fill_(self.initial_capital)
        self.portfolio_value.copy_(self.capital)
        self.positions.zero_()
        self.n_positions.zero_()
        self.current_step.zero_()
        self.episode_returns.zero_()
        self.done.zero_()

        # Random starting days for each environment
        max_start = max(0, self.n_days - self.episode_length - 1)
        self.current_day_idx = torch.randint(20, max_start + 20, (self.n_envs,),
                                              device=self.device, dtype=torch.int64)

        obs = self._get_observations()
        return obs, {}

    def _get_observations(self) -> torch.Tensor:
        """Get observations for all envs - vectorized on GPU with normalization"""
        # Gather current prices for each env's day
        batch_idx = torch.arange(self.n_envs, device=self.device)
        current_prices = self.prices[self.current_day_idx]  # [n_envs, n_symbols, 5]
        current_returns = self.returns[self.current_day_idx]  # [n_envs, n_symbols]
        current_vol = self.volatility[self.current_day_idx]  # [n_envs, n_symbols]

        # Normalize returns (typically [-0.1, 0.1] range)
        current_returns = current_returns * 10.0  # Scale to [-1, 1] approx

        # Normalize volatility (typically [0.1, 0.5] range)
        current_vol = (current_vol - 0.2) / 0.15  # Center and scale

        # Normalize prices (log-scale relative to ~400 baseline)
        normalized_prices = torch.log(current_prices / 400.0 + 1e-6)  # Log-scale, centered near 0

        # Flatten market features
        market_features = torch.cat([
            normalized_prices.reshape(self.n_envs, -1),  # 15 features (3 symbols * 5)
            current_returns,  # 3 features (already normalized above)
            current_vol,  # 3 features (already normalized above)
        ], dim=1)  # [n_envs, 21]

        # Portfolio features (already normalized to [0, 1] or [-1, 1] range)
        portfolio_features = torch.stack([
            (self.capital / self.initial_capital - 1.0),  # Center at 0
            (self.portfolio_value / self.initial_capital - 1.0),  # Center at 0
            self.n_positions.float() / self.max_positions,
            self.current_step.float() / self.episode_length,
        ], dim=1)  # [n_envs, 4]

        # Position features (flattened)
        position_features = self.positions.reshape(self.n_envs, -1)  # [n_envs, 30]

        # Pad to obs_dim
        obs = torch.zeros((self.n_envs, self.obs_dim), device=self.device, dtype=torch.float32)
        total_features = market_features.shape[1] + portfolio_features.shape[1] + position_features.shape[1]
        obs[:, :min(total_features, self.obs_dim)] = torch.cat([
            market_features, portfolio_features, position_features
        ], dim=1)[:, :self.obs_dim]

        return obs

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Step all environments in parallel on GPU.

        Args:
            actions: [n_envs] tensor of action indices

        Returns:
            observations, rewards, terminated, truncated, infos
        """
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, device=self.device, dtype=torch.int64)

        # Get current prices
        current_prices = self.prices[self.current_day_idx, :, 3]  # [n_envs, n_symbols]

        # Calculate rewards based on portfolio change
        old_value = self.portfolio_value.clone()

        # Execute actions (vectorized)
        self._execute_actions_gpu(actions, current_prices)

        # Update portfolio value
        self._update_portfolio_value_gpu(current_prices)

        # Calculate rewards (percentage returns, scaled and clipped)
        # Simple fixed transform: scale by 100 then clip to [-5, 5]
        # This is stable and doesn't change distribution during training
        raw_returns = (self.portfolio_value - old_value) / (old_value + 1e-8)
        rewards = torch.clamp(raw_returns * 100.0, -5.0, 5.0)

        # Update episode tracking
        self.episode_returns += rewards
        self.current_step += 1
        self.current_day_idx += 1

        # Check termination
        terminated = self.current_day_idx >= self.n_days - 1
        truncated = self.current_step >= self.episode_length
        self.done = terminated | truncated

        # Auto-reset done environments
        done_mask = self.done
        if done_mask.any():
            self._reset_done_envs(done_mask)

        obs = self._get_observations()

        return obs, rewards, terminated, truncated, {}

    def _execute_actions_gpu(self, actions: torch.Tensor, prices: torch.Tensor):
        """Execute trading actions on GPU"""
        # Action 0 = hold
        # Actions 1-15 = buy options
        # Actions 16-30 = sell options

        buy_mask = (actions >= 1) & (actions <= 15)
        sell_mask = (actions >= 16) & (actions <= 30)

        # Simple position updates (can be expanded)
        # Buy: decrease capital, add position
        if buy_mask.any():
            buy_envs = buy_mask.nonzero(as_tuple=True)[0]
            option_cost = prices[buy_envs, 0] * 0.02  # Simplified option pricing
            self.capital[buy_envs] -= option_cost * 100  # 1 contract = 100 shares

        # Sell: increase capital
        if sell_mask.any():
            sell_envs = sell_mask.nonzero(as_tuple=True)[0]
            self.capital[sell_envs] += prices[sell_envs, 0] * 0.01 * 100

    def _update_portfolio_value_gpu(self, prices: torch.Tensor):
        """Update portfolio value on GPU"""
        # Simplified: value = capital + position values
        self.portfolio_value = self.capital.clone()

    def _reset_done_envs(self, done_mask: torch.Tensor):
        """Reset specific environments that are done"""
        n_done = done_mask.sum().item()
        if n_done == 0:
            return

        done_idx = done_mask.nonzero(as_tuple=True)[0]

        self.capital[done_idx] = self.initial_capital
        self.portfolio_value[done_idx] = self.initial_capital
        self.positions[done_idx] = 0
        self.n_positions[done_idx] = 0
        self.current_step[done_idx] = 0
        self.episode_returns[done_idx] = 0
        self.done[done_idx] = False

        # New random starting days
        max_start = max(0, self.n_days - self.episode_length - 1)
        self.current_day_idx[done_idx] = torch.randint(
            20, max_start + 20, (n_done,), device=self.device, dtype=torch.int64
        )

    def close(self):
        """Cleanup GPU memory"""
        pass

