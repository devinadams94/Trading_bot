#!/usr/bin/env python3
"""
Working Options Environment - Guaranteed to Execute Trades
This replaces the broken enhanced_options_env.py

UPDATED: Now includes realistic transaction costs based on Alpaca fee structure
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
import logging
from datetime import datetime, timedelta

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

# Import from new module structure
try:
    from src.utils.indicators import TechnicalIndicators
except ImportError:
    # Fallback if import fails
    class TechnicalIndicators:
        @staticmethod
        def calculate_macd(prices, fast=12, slow=26, signal=9):
            return 0.0
        @staticmethod
        def calculate_rsi(prices, period=14):
            return 50.0
        @staticmethod
        def calculate_cci(high, low, close, period=20):
            return 0.0
        @staticmethod
        def calculate_adx(high, low, close, period=14):
            return 25.0

# Setup logger first
logger = logging.getLogger(__name__)

# Import realistic transaction costs
try:
    from src.trading.transaction_costs import RealisticTransactionCostCalculator
except ImportError:
    # Fallback if import fails
    RealisticTransactionCostCalculator = None
    logger.warning("⚠️ Realistic transaction costs not available, using legacy commission model")

# Import unified Greeks calculator
try:
    from src.utils.greeks import GreeksCalculator, get_greeks_calculator
    GREEKS_CALCULATOR = get_greeks_calculator(risk_free_rate=0.05)
except ImportError:
    GREEKS_CALCULATOR = None
    logger.warning("⚠️ Greeks calculator not available, using fallback values")


class WorkingOptionsEnvironment(gym.Env):
    """
    Simplified but working options environment that GUARANTEES trades
    """
    
    def __init__(
        self,
        data_loader=None,
        symbols: List[str] = None,
        initial_capital: float = 100000,
        max_positions: int = 5,
        commission: float = 0.65,  # DEPRECATED: Use realistic_costs instead
        lookback_window: int = 20,
        episode_length: int = 195,
        min_data_quality: float = 0.2,
        # Enhanced parameters for CLSTM-PPO compatibility
        include_technical_indicators: bool = True,
        include_market_microstructure: bool = True,
        volatility_window: int = 20,
        rsi_window: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        # NEW: Realistic transaction costs
        use_realistic_costs: bool = True,
        enable_slippage: bool = True,
        slippage_model: str = 'volume_based',
        # Position sizing parameters
        position_size_pct: float = 0.02,  # Risk 2% of portfolio per trade
        max_contracts_per_trade: int = 10,  # Cap at 10 contracts max
        **kwargs
    ):
        super().__init__()

        self.data_loader = data_loader
        self.position_size_pct = position_size_pct
        self.max_contracts_per_trade = max_contracts_per_trade
        # Enhanced symbol list with major tech stocks
        self.symbols = symbols or [
            'SPY', 'QQQ', 'IWM',  # ETFs
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',  # Mega cap tech
            'TSLA', 'META', 'NFLX', 'AMD', 'CRM',  # High volatility tech
            'PLTR', 'SNOW', 'COIN', 'RBLX', 'ZM'   # Growth/meme stocks
        ]
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.commission = commission  # Legacy fallback
        self.lookback_window = lookback_window
        self.episode_length = episode_length
        self.min_data_quality = min_data_quality

        # Enhanced features for CLSTM
        self.include_technical_indicators = include_technical_indicators
        self.include_market_microstructure = include_market_microstructure
        self.volatility_window = volatility_window
        self.rsi_window = rsi_window
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow

        # NEW: Realistic transaction costs
        self.use_realistic_costs = use_realistic_costs and RealisticTransactionCostCalculator is not None
        if self.use_realistic_costs:
            self.cost_calculator = RealisticTransactionCostCalculator(
                enable_slippage=enable_slippage,
                slippage_model=slippage_model,
                log_costs=False  # Disable per-trade logging for performance
            )
            logger.info("✅ Using realistic transaction costs (Alpaca fee structure)")
        else:
            self.cost_calculator = None
            logger.info(f"⚠️ Using legacy commission model (${commission} per trade)")

        # OPTIMIZATION: Cache for Greeks lookups and technical indicators
        self._greeks_cache = {}  # Key: (symbol, strike, option_type) -> Greeks dict
        self._technical_indicators_cache = {}  # Key: (symbol, step) -> indicators dict
        self._cache_hits = 0
        self._cache_misses = 0

        # Action space: 0=hold, 1-10=buy calls, 11-20=buy puts, 21-30=sell positions
        self.action_space = spaces.Discrete(31)

        # CLSTM-PPO compatible observation space
        num_symbols = len(self.symbols)

        # Calculate dimensions for CLSTM-PPO expected format
        price_history_shape = (num_symbols, lookback_window)  # Price history for each symbol
        technical_indicators_shape = (num_symbols * 6,)  # RSI, MACD, volatility, etc. per symbol
        options_chain_shape = (max_positions, 8)  # Position data: type, strike, price, etc.
        portfolio_state_shape = (5,)  # Capital, value, drawdown, positions, time
        greeks_summary_shape = (max_positions * 4,)  # Delta, gamma, theta, vega per position
        symbol_encoding_shape = (num_symbols,)  # One-hot encoding of symbols

        self.observation_space = spaces.Dict({
            # CLSTM-PPO expected format
            'price_history': spaces.Box(low=0, high=10000, shape=price_history_shape, dtype=np.float32),
            'technical_indicators': spaces.Box(low=-10, high=10, shape=technical_indicators_shape, dtype=np.float32),
            'options_chain': spaces.Box(low=-1000, high=1000, shape=options_chain_shape, dtype=np.float32),
            'portfolio_state': spaces.Box(low=0, high=1e8, shape=portfolio_state_shape, dtype=np.float32),
            'greeks_summary': spaces.Box(low=-10, high=10, shape=greeks_summary_shape, dtype=np.float32),
            'symbol_encoding': spaces.Box(low=0, high=1, shape=symbol_encoding_shape, dtype=np.float32),

            # Additional features for compatibility
            'market_microstructure': spaces.Box(low=0, high=1e12, shape=(num_symbols * 3,), dtype=np.float32),
            'time_features': spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        })
        
        # Market data (will be loaded)
        self.market_data = {}
        self.current_data_index = 0
        self.data_loaded = False

        # State variables
        self.reset()
        
        logger.info(f"WorkingOptionsEnvironment initialized:")
        logger.info(f"  Symbols: {self.symbols}")
        logger.info(f"  Actions: {self.action_space.n}")
        logger.info(f"  Episode length: {episode_length}")
        logger.info(f"  Max positions: {max_positions}")
    
    async def load_data(self, start_date: datetime, end_date: datetime):
        """Load market data and options data with Greeks"""
        logger.info(f"Loading data from {start_date.date()} to {end_date.date()}")

        # Create simple synthetic data if no data loader
        if self.data_loader is None:
            self._create_synthetic_data(start_date, end_date)
            self.options_data = {}  # No options data for synthetic
        else:
            try:
                # Try to load real stock data
                self.market_data = await self.data_loader.load_historical_stock_data(
                    self.symbols, start_date, end_date
                )

                # Try to load real options data with Greeks
                logger.info("📊 Loading options data with Greeks...")
                try:
                    self.options_data = await self.data_loader.load_historical_options_data(
                        self.symbols, start_date, end_date
                    )

                    # Log Greeks availability
                    total_contracts = sum(len(opts) for opts in self.options_data.values())
                    logger.info(f"✅ Loaded {total_contracts} options contracts with Greeks")

                    # Check if Greeks are present
                    if total_contracts > 0:
                        sample_symbol = list(self.options_data.keys())[0]
                        sample_contract = self.options_data[sample_symbol][0]
                        has_greeks = any(k in sample_contract for k in ['delta', 'gamma', 'theta', 'vega'])
                        if has_greeks:
                            logger.info("✅ Greeks (delta, gamma, theta, vega) available in options data")
                        else:
                            logger.warning("⚠️  Greeks not found in options data")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to load options data: {e}")
                    self.options_data = {}

                if not self.market_data:
                    logger.warning("No real data loaded, using synthetic data")
                    self._create_synthetic_data(start_date, end_date)
                    self.options_data = {}
            except Exception as e:
                logger.warning(f"Failed to load real data: {e}, using synthetic data")
                self._create_synthetic_data(start_date, end_date)
                self.options_data = {}

        self.data_loaded = True
        logger.info(f"Data loaded for {len(self.market_data)} symbols")
    
    def _find_option_contract(self, symbol: str, strike: float, option_type: str, timestamp: datetime = None, tolerance: float = 2.0) -> Optional[Dict]:
        """
        Find matching option contract from loaded options data at a specific timestamp

        Args:
            symbol: Underlying symbol
            strike: Strike price
            option_type: 'call' or 'put'
            timestamp: Timestamp to find option price at (matches by DATE, not exact time)
            tolerance: Strike price tolerance (default: $2.00 to handle strike rounding)

        Returns:
            Option contract dict with Greeks, or None if not found
        """
        # Timestamp is required for accurate pricing
        if timestamp is None:
            return None

        # Extract date for matching (options data has intraday timestamps)
        if hasattr(timestamp, 'date'):
            target_date = timestamp.date()
        else:
            target_date = pd.Timestamp(timestamp).date()

        # OPTIMIZATION: Check cache first (use date, not full timestamp)
        cache_key = (symbol, round(strike, 0), option_type.lower(), target_date)
        if cache_key in self._greeks_cache:
            self._cache_hits += 1
            return self._greeks_cache[cache_key]

        self._cache_misses += 1

        if not hasattr(self, 'options_data') or symbol not in self.options_data:
            self._greeks_cache[cache_key] = None
            return None

        # Find options with matching strike, type, AND date
        matching_options = []
        for opt in self.options_data[symbol]:
            opt_strike = opt.get('strike', 0)
            opt_type = opt.get('option_type', '').lower()
            opt_ts = opt.get('timestamp')

            # Check strike and type
            if abs(opt_strike - strike) > tolerance:
                continue
            if opt_type != option_type.lower():
                continue

            # Check date match
            if opt_ts is not None:
                if hasattr(opt_ts, 'date'):
                    opt_date = opt_ts.date()
                else:
                    opt_date = pd.Timestamp(opt_ts).date()
                if opt_date == target_date:
                    matching_options.append(opt)

        if not matching_options:
            self._greeks_cache[cache_key] = None
            return None

        # Return the closest match by strike
        result = min(matching_options, key=lambda x: abs(x.get('strike', 0) - strike))
        self._greeks_cache[cache_key] = result
        return result

    def _calculate_greeks(
        self,
        underlying_price: float,
        strike: float,
        option_type: str,
        time_to_expiry_days: float = 30,
        iv: float = 0.30,
        option_price: float = None
    ) -> Dict[str, float]:
        """
        Calculate Greeks using unified calculator.

        This ensures consistency with historical data processing.

        Args:
            underlying_price: Current underlying price
            strike: Strike price
            option_type: 'call' or 'put'
            time_to_expiry_days: Days to expiration
            iv: Implied volatility (if known)
            option_price: Option mid price (used to calculate IV if iv not provided)

        Returns:
            Dict with delta, gamma, theta, vega
        """
        if GREEKS_CALCULATOR is None:
            # Fallback: approximate Greeks based on moneyness
            moneyness = underlying_price / strike
            is_call = option_type.lower() == 'call'

            if is_call:
                if moneyness > 1.05:  # ITM
                    delta = 0.7 + 0.2 * min(1, (moneyness - 1) * 5)
                elif moneyness < 0.95:  # OTM
                    delta = 0.3 - 0.2 * min(1, (1 - moneyness) * 5)
                else:  # ATM
                    delta = 0.5
            else:
                if moneyness > 1.05:  # OTM put
                    delta = -0.3 + 0.2 * min(1, (moneyness - 1) * 5)
                elif moneyness < 0.95:  # ITM put
                    delta = -0.7 - 0.2 * min(1, (1 - moneyness) * 5)
                else:  # ATM
                    delta = -0.5

            # Rough estimates for other Greeks
            gamma = 0.05 * (1 - abs(1 - moneyness) * 2)  # Highest ATM
            theta = -iv * underlying_price * 0.01 / max(time_to_expiry_days, 1)
            vega = underlying_price * 0.01 * np.sqrt(time_to_expiry_days / 252)

            return {'delta': delta, 'gamma': max(0, gamma), 'theta': theta, 'vega': vega}

        # Use unified calculator
        time_to_expiry = time_to_expiry_days / 252  # Convert to years

        if option_price is not None and iv <= 0:
            # Calculate from price
            result = GREEKS_CALCULATOR.calculate_greeks_from_price(
                option_price=option_price,
                underlying_price=underlying_price,
                strike=strike,
                time_to_expiry=time_to_expiry,
                option_type=option_type
            )
        else:
            # Calculate from IV
            result = GREEKS_CALCULATOR.calculate_greeks(
                underlying_price=underlying_price,
                strike=strike,
                time_to_expiry=time_to_expiry,
                iv=iv,
                option_type=option_type
            )

        return result.to_dict()

    def _create_synthetic_data(self, start_date: datetime, end_date: datetime):
        """Create synthetic market data"""
        days = (end_date - start_date).days
        dates = [start_date + timedelta(days=i) for i in range(days)]
        
        for symbol in self.symbols:
            # Create realistic price data
            initial_price = 100.0 if symbol == 'SPY' else 300.0
            prices = [initial_price]
            
            for i in range(1, len(dates)):
                # Random walk with slight upward bias
                change = np.random.normal(0.001, 0.02)  # 0.1% drift, 2% volatility
                new_price = prices[-1] * (1 + change)
                prices.append(max(10.0, new_price))  # Minimum price
            
            # Create DataFrame
            data = pd.DataFrame({
                'timestamp': dates,
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                'close': prices,
                'volume': [np.random.randint(1000000, 10000000) for _ in prices],
                'underlying_price': prices
            })
            
            self.market_data[symbol] = data
    
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment"""
        self.capital = self.initial_capital
        self.positions = []
        self.current_step = 0
        self.episode_trades = 0
        self.trade_history = []

        # Randomize starting point for diverse training
        # Leave room for episode_length steps
        if self.data_loaded and self.symbols:
            self.current_symbol = np.random.choice(self.symbols)
            data_len = len(self.market_data.get(self.current_symbol, []))
            max_start = max(0, data_len - self.episode_length - 1)
            self.current_data_index = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        else:
            self.current_symbol = self.symbols[0] if self.symbols else 'SPY'
            self.current_data_index = 0

        # Portfolio tracking
        self.portfolio_value_history = [self.initial_capital]
        self.peak_portfolio_value = self.initial_capital

        # OPTIMIZATION: Clear caches to prevent memory leaks
        # Keep Greeks cache (static data) but clear technical indicators cache (step-dependent)
        self._technical_indicators_cache.clear()

        # Log cache statistics periodically (every 100 episodes)
        if hasattr(self, '_reset_count'):
            self._reset_count += 1
            if self._reset_count % 100 == 0:
                total_lookups = self._cache_hits + self._cache_misses
                hit_rate = (self._cache_hits / total_lookups * 100) if total_lookups > 0 else 0
                logger.info(f"📊 Cache stats: {self._cache_hits} hits, {self._cache_misses} misses ({hit_rate:.1f}% hit rate)")
        else:
            self._reset_count = 0

        return self._get_observation()

    def _calculate_position_size(self, option_price: float, portfolio_value: float) -> int:
        """
        Calculate number of contracts to trade based on position sizing rules

        Args:
            option_price: Price per contract (will be multiplied by 100 for shares)
            portfolio_value: Current total portfolio value

        Returns:
            Number of contracts to trade (minimum 1, maximum max_contracts_per_trade)
        """
        # Calculate risk amount (e.g., 2% of portfolio)
        risk_amount = portfolio_value * self.position_size_pct

        # Calculate how many contracts this allows
        # Each contract costs: option_price * 100 shares
        contract_cost = option_price * 100

        if contract_cost <= 0:
            return 1  # Minimum 1 contract

        # Number of contracts we can afford with our risk budget
        num_contracts = int(risk_amount / contract_cost)

        # Apply constraints
        num_contracts = max(1, num_contracts)  # Minimum 1 contract
        num_contracts = min(num_contracts, self.max_contracts_per_trade)  # Cap at max

        return num_contracts

    def _calculate_transaction_cost(self, option_data: Dict, quantity: int, side: str = 'buy') -> Tuple[float, Dict]:
        """
        Calculate transaction cost using realistic model or legacy commission

        Returns:
            (total_cost, cost_breakdown_dict)
        """
        if self.use_realistic_costs and self.cost_calculator:
            # Use realistic transaction cost model
            breakdown = self.cost_calculator.calculate_transaction_cost(
                option_data, quantity, side
            )
            return breakdown.total_cost, breakdown.to_dict()
        else:
            # Legacy commission model
            return self.commission, {'commission': self.commission, 'total_cost': self.commission}

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Execute one step - GUARANTEED to work"""
        self.current_step += 1

        # Get current market data
        current_data = self._get_current_market_data()
        if current_data is None:
            # End episode if no data
            return self._get_observation(), 0.0, True, {'episode_trades': self.episode_trades}

        # Calculate portfolio value BEFORE action (for reward calculation)
        portfolio_value_before = self._calculate_portfolio_value(current_data)

        # Initialize previous portfolio value if first step
        if not hasattr(self, 'previous_portfolio_value'):
            self.previous_portfolio_value = self.initial_capital

        # Track transaction costs for this step
        step_transaction_costs = 0.0
        transaction_cost_breakdown = {}

        # Execute action - THIS WILL ALWAYS WORK
        trade_executed = False
        action_name = "HOLD"
        
        if action == 0:
            # Hold action
            action_name = "HOLD"
            
        elif 1 <= action <= 10:
            # Buy call options
            trade_executed = True
            self.episode_trades += 1

            strike_offset = (action - 1) * 0.01  # 0% to 9% OTM
            current_price = current_data['close']
            strike_price = current_price * (1 + strike_offset)

            # Get current timestamp for option lookup
            current_timestamp = current_data.get('timestamp')

            # Try to find real option contract with Greeks FIRST
            option_contract = self._find_option_contract(
                self.current_symbol, strike_price, 'call', timestamp=current_timestamp
            )

            # Use real option data if available, otherwise fall back to synthetic
            if option_contract:
                option_price_mid = (option_contract.get('bid', 0) + option_contract.get('ask', 0)) / 2
                if option_price_mid <= 0:
                    option_price_mid = option_contract.get('last', 0)
                option_data = {
                    'bid': option_contract.get('bid', option_price_mid * 0.98),
                    'ask': option_contract.get('ask', option_price_mid * 1.02),
                    'last': option_contract.get('last', option_price_mid),
                    'volume': option_contract.get('volume', 100),
                    'open_interest': option_contract.get('open_interest', 1000),
                    'moneyness': strike_price / current_price,
                    'implied_volatility': option_contract.get('implied_volatility', 0.3)
                }
                # Use Greeks from data, or recalculate if missing/zero
                iv = option_contract.get('implied_volatility', 0.3)
                stored_delta = option_contract.get('delta', 0)

                if abs(stored_delta) > 0.001:
                    # Use stored Greeks
                    delta = stored_delta
                    gamma = option_contract.get('gamma', 0.05)
                    theta = option_contract.get('theta', -0.5)
                    vega = option_contract.get('vega', 0.02)
                else:
                    # Recalculate using unified calculator
                    greeks = self._calculate_greeks(
                        underlying_price=current_price,
                        strike=strike_price,
                        option_type='call',
                        time_to_expiry_days=30,  # Assume 30 DTE
                        iv=iv,
                        option_price=option_price_mid
                    )
                    delta = greeks['delta']
                    gamma = greeks['gamma']
                    theta = greeks['theta']
                    vega = greeks['vega']
            else:
                # Fallback: synthetic pricing with calculated Greeks
                moneyness = strike_price / current_price
                spread_pct = 0.02 + 0.03 * abs(1 - moneyness)
                option_price_mid = max(0.5, current_price * 0.05 * (1 - strike_offset))
                iv = current_data.get('volatility', 0.3)

                option_data = {
                    'bid': option_price_mid * (1 - spread_pct / 2),
                    'ask': option_price_mid * (1 + spread_pct / 2),
                    'last': option_price_mid,
                    'volume': current_data.get('volume', 1000) / 100,
                    'open_interest': 1000,
                    'moneyness': moneyness,
                    'implied_volatility': iv
                }

                # Calculate Greeks using unified calculator
                greeks = self._calculate_greeks(
                    underlying_price=current_price,
                    strike=strike_price,
                    option_type='call',
                    time_to_expiry_days=30,
                    iv=iv,
                    option_price=option_price_mid
                )
                delta = greeks['delta']
                gamma = greeks['gamma']
                theta = greeks['theta']
                vega = greeks['vega']

            # Calculate position size based on portfolio value
            portfolio_value = self.capital + sum(
                p.get('entry_price', 0) * p.get('quantity', 0) * 100
                for p in self.positions
            )
            quantity = self._calculate_position_size(option_price_mid, portfolio_value)

            # Calculate transaction cost
            transaction_cost, cost_breakdown = self._calculate_transaction_cost(
                option_data, quantity, side='buy'
            )

            # Execution price (buy at ask for realistic costs, mid for legacy)
            if self.use_realistic_costs:
                execution_price = option_data['ask']
            else:
                execution_price = option_price_mid

            # Total cost including transaction costs (quantity * price * 100 shares)
            total_cost = execution_price * 100 * quantity + transaction_cost

            if self.capital >= total_cost:
                # Execute trade
                self.capital -= total_cost
                step_transaction_costs += transaction_cost
                transaction_cost_breakdown = cost_breakdown

                position = {
                    'type': 'call',
                    'strike': strike_price,
                    'entry_price': execution_price,
                    'quantity': quantity,  # FIX: Use calculated quantity, not hardcoded 100
                    'symbol': self.current_symbol,
                    'entry_step': self.current_step,
                    'cost': total_cost,
                    'transaction_cost': transaction_cost,
                    # Store Greeks
                    'delta': delta,
                    'gamma': gamma,
                    'theta': theta,
                    'vega': vega
                }
                self.positions.append(position)

                action_name = f"BUY_CALL_{strike_price:.0f}"

                logger.debug(f"Executed: {action_name}, qty={quantity}, cost=${total_cost:.2f}, txn_cost=${transaction_cost:.2f}, delta={delta:.3f}")

        elif 11 <= action <= 20:
            # Buy put options
            trade_executed = True
            self.episode_trades += 1

            strike_offset = (action - 11) * 0.01  # 0% to 9% OTM
            current_price = current_data['close']
            strike_price = current_price * (1 - strike_offset)

            # Get current timestamp for option lookup
            current_timestamp = current_data.get('timestamp')

            # Try to find real option contract with Greeks FIRST
            option_contract = self._find_option_contract(
                self.current_symbol, strike_price, 'put', timestamp=current_timestamp
            )

            # Use real option data if available, otherwise fall back to synthetic
            if option_contract:
                option_price_mid = (option_contract.get('bid', 0) + option_contract.get('ask', 0)) / 2
                if option_price_mid <= 0:
                    option_price_mid = option_contract.get('last', 0)
                option_data = {
                    'bid': option_contract.get('bid', option_price_mid * 0.98),
                    'ask': option_contract.get('ask', option_price_mid * 1.02),
                    'last': option_contract.get('last', option_price_mid),
                    'volume': option_contract.get('volume', 100),
                    'open_interest': option_contract.get('open_interest', 1000),
                    'moneyness': strike_price / current_price,
                    'implied_volatility': option_contract.get('implied_volatility', 0.3)
                }
                # Use Greeks from data, or recalculate if missing/zero
                iv = option_contract.get('implied_volatility', 0.3)
                stored_delta = option_contract.get('delta', 0)

                if abs(stored_delta) > 0.001:
                    # Use stored Greeks
                    delta = stored_delta
                    gamma = option_contract.get('gamma', 0.05)
                    theta = option_contract.get('theta', -0.5)
                    vega = option_contract.get('vega', 0.02)
                else:
                    # Recalculate using unified calculator
                    greeks = self._calculate_greeks(
                        underlying_price=current_price,
                        strike=strike_price,
                        option_type='put',
                        time_to_expiry_days=30,
                        iv=iv,
                        option_price=option_price_mid
                    )
                    delta = greeks['delta']
                    gamma = greeks['gamma']
                    theta = greeks['theta']
                    vega = greeks['vega']
            else:
                # Fallback: synthetic pricing with calculated Greeks
                moneyness = strike_price / current_price
                spread_pct = 0.02 + 0.03 * abs(1 - moneyness)
                option_price_mid = max(0.5, current_price * 0.05 * (1 - strike_offset))
                iv = current_data.get('volatility', 0.3)

                option_data = {
                    'bid': option_price_mid * (1 - spread_pct / 2),
                    'ask': option_price_mid * (1 + spread_pct / 2),
                    'last': option_price_mid,
                    'volume': current_data.get('volume', 1000) / 100,
                    'open_interest': 1000,
                    'moneyness': moneyness,
                    'implied_volatility': iv
                }

                # Calculate Greeks using unified calculator
                greeks = self._calculate_greeks(
                    underlying_price=current_price,
                    strike=strike_price,
                    option_type='put',
                    time_to_expiry_days=30,
                    iv=iv,
                    option_price=option_price_mid
                )
                delta = greeks['delta']
                gamma = greeks['gamma']
                theta = greeks['theta']
                vega = greeks['vega']

            # Calculate position size based on portfolio value
            portfolio_value = self.capital + sum(
                p.get('entry_price', 0) * p.get('quantity', 0) * 100
                for p in self.positions
            )
            quantity = self._calculate_position_size(option_price_mid, portfolio_value)

            # Calculate transaction cost
            transaction_cost, cost_breakdown = self._calculate_transaction_cost(
                option_data, quantity, side='buy'
            )

            # Execution price (buy at ask for realistic costs)
            if self.use_realistic_costs:
                execution_price = option_data['ask']
            else:
                execution_price = option_price_mid

            # Total cost including transaction costs (quantity * price * 100 shares)
            total_cost = execution_price * 100 * quantity + transaction_cost

            if self.capital >= total_cost:
                # Execute trade
                self.capital -= total_cost
                step_transaction_costs += transaction_cost
                transaction_cost_breakdown = cost_breakdown

                position = {
                    'type': 'put',
                    'strike': strike_price,
                    'entry_price': execution_price,
                    'quantity': quantity,  # FIX: Use calculated quantity, not hardcoded 100
                    'symbol': self.current_symbol,
                    'entry_step': self.current_step,
                    'cost': total_cost,
                    'transaction_cost': transaction_cost,
                    # Store Greeks
                    'delta': delta,
                    'gamma': gamma,
                    'theta': theta,
                    'vega': vega
                }
                self.positions.append(position)

                action_name = f"BUY_PUT_{strike_price:.0f}"

                logger.debug(f"Executed: {action_name}, qty={quantity}, cost=${total_cost:.2f}, txn_cost=${transaction_cost:.2f}, delta={delta:.3f}")
            
        elif 21 <= action <= 30:
            # Sell positions (or buy if no positions)
            trade_executed = True
            self.episode_trades += 1

            if len(self.positions) > 0:
                # Sell oldest position
                position = self.positions.pop(0)
                current_price = current_data['close']
                current_timestamp = current_data.get('timestamp')
                position_type = position['type']

                # Determine option type for lookup
                if position_type in ['call', 'covered_call']:
                    lookup_type = 'call'
                else:
                    lookup_type = 'put'

                # Try to find real option contract for current price
                option_contract = self._find_option_contract(
                    position['symbol'], position['strike'], lookup_type, timestamp=current_timestamp
                )

                if option_contract:
                    # Use real market data
                    option_value_mid = (option_contract.get('bid', 0) + option_contract.get('ask', 0)) / 2
                    if option_value_mid <= 0:
                        option_value_mid = option_contract.get('last', 0)
                    option_data = {
                        'bid': option_contract.get('bid', option_value_mid * 0.98),
                        'ask': option_contract.get('ask', option_value_mid * 1.02),
                        'last': option_contract.get('last', option_value_mid),
                        'volume': option_contract.get('volume', 100),
                        'open_interest': option_contract.get('open_interest', 1000),
                        'moneyness': position['strike'] / current_price,
                        'implied_volatility': option_contract.get('implied_volatility', 0.3)
                    }
                else:
                    # Fallback: synthetic pricing based on intrinsic + time value
                    if position_type == 'call':
                        intrinsic_value = max(0, current_price - position['strike'])
                    elif position_type == 'put':
                        intrinsic_value = max(0, position['strike'] - current_price)
                    elif position_type == 'covered_call':
                        intrinsic_value = max(0, current_price - position['strike'])
                    elif position_type == 'cash_secured_put':
                        intrinsic_value = max(0, position['strike'] - current_price)
                    else:
                        logger.warning(f"Unknown position type: {position_type}")
                        intrinsic_value = 0

                    time_value = max(0.1, position['entry_price'] * 0.3)  # Reduced time value decay
                    option_value_mid = intrinsic_value + time_value

                    moneyness = position['strike'] / current_price
                    spread_pct = 0.02 + 0.03 * abs(1 - moneyness)
                    option_data = {
                        'bid': option_value_mid * (1 - spread_pct / 2),
                        'ask': option_value_mid * (1 + spread_pct / 2),
                        'last': option_value_mid,
                        'volume': current_data.get('volume', 1000) / 100,
                        'open_interest': 1000,
                        'moneyness': moneyness,
                        'implied_volatility': current_data.get('volatility', 0.3)
                    }

                # Determine if this is a long or short position
                is_short_position = position_type in ['covered_call', 'cash_secured_put']

                # Use the quantity from the position (not recalculated)
                quantity = position.get('quantity', 1)
                side = 'buy' if is_short_position else 'sell'  # Short positions: buy to close
                transaction_cost, cost_breakdown = self._calculate_transaction_cost(
                    option_data, quantity, side=side
                )

                # Execution price
                if self.use_realistic_costs:
                    # Long positions: sell at bid
                    # Short positions: buy at ask (to close)
                    execution_price = option_data['ask'] if is_short_position else option_data['bid']
                else:
                    execution_price = option_value_mid

                # Execute closing trade
                if is_short_position:
                    # Short position: we PAY to buy it back (negative proceeds)
                    # We originally received premium, now we pay to close
                    gross_cost = execution_price * position['quantity'] * 100  # Cost to buy back
                    net_cost = gross_cost + transaction_cost  # Total cost including fees
                    self.capital -= net_cost  # Deduct cost from capital

                    # P&L = premium received - cost to close
                    premium_received = position.get('premium_received', 0)
                    pnl = premium_received - net_cost

                    # Release reserved capital if any
                    if 'capital_reserved' in position:
                        self.capital += position['capital_reserved']
                else:
                    # Long position: we RECEIVE proceeds from selling
                    gross_proceeds = execution_price * position['quantity'] * 100
                    net_proceeds = gross_proceeds - transaction_cost
                    self.capital += net_proceeds

                    # P&L = proceeds - original cost
                    position_cost = position.get('cost', position['entry_price'] * position['quantity'] * 100)
                    pnl = net_proceeds - position_cost

                step_transaction_costs += transaction_cost
                transaction_cost_breakdown = cost_breakdown

                action_name = f"SELL_{position['type'].upper()}"

                # Record trade
                trade_record = {
                    'action': action_name,
                    'pnl': pnl,
                    'entry_price': position['entry_price'],
                    'exit_price': execution_price,
                    'step': self.current_step,
                    'transaction_costs': transaction_cost
                }
                self.trade_history.append(trade_record)

                logger.debug(f"Executed: {action_name}, P&L=${pnl:.2f}, txn_cost=${transaction_cost:.2f}")

            else:
                # No positions to sell, buy a random option instead
                current_price = current_data['close']
                option_type = 'call' if action % 2 == 1 else 'put'

                if option_type == 'call':
                    strike_price = current_price * 1.02  # 2% OTM call
                else:
                    strike_price = current_price * 0.98  # 2% OTM put

                option_price = current_price * 0.03  # 3% of stock price

                if self.capital >= option_price * 100 + self.commission:
                    cost = option_price * 100 + self.commission
                    self.capital -= cost

                    position = {
                        'type': option_type,
                        'strike': strike_price,
                        'entry_price': option_price,
                        'quantity': 100,
                        'symbol': self.current_symbol,
                        'entry_step': self.current_step,
                        'cost': cost
                    }
                    self.positions.append(position)

                    action_name = f"BUY_{option_type.upper()}_{strike_price:.0f}"

                    logger.debug(f"Executed: {action_name} (no positions to sell)")
                else:
                    action_name = "INSUFFICIENT_CAPITAL"
        
        # Update portfolio value AFTER action
        portfolio_value_after = self._calculate_portfolio_value(current_data)
        self.portfolio_value_history.append(portfolio_value_after)
        self.peak_portfolio_value = max(self.peak_portfolio_value, portfolio_value_after)

        # Calculate reward based on portfolio return (aligned with research paper)
        # Return_t = (portfolio_value_after) - (portfolio_value_before) - transaction_costs
        # This directly optimizes for profitability while penalizing high transaction costs
        raw_return = portfolio_value_after - self.previous_portfolio_value

        # Subtract transaction costs from reward (agent learns to minimize costs)
        net_return = raw_return - step_transaction_costs

        # Apply scaling factor for stable training
        # INCREASED from 1e-4 to 1e-3 for stronger learning signal
        reward = net_return * 1e-3

        # Penalize excessive trading to prevent overtrading
        if trade_executed:
            reward -= 0.02  # Small penalty for each trade

        # Update previous portfolio value for next step
        self.previous_portfolio_value = portfolio_value_after

        # Check if episode is done
        done = (self.current_step >= self.episode_length or
                self.current_data_index >= len(self.market_data[self.current_symbol]) - 1)

        # Move to next data point
        self.current_data_index += 1

        # Create info dictionary
        info = {
            'episode_trades': self.episode_trades,
            'portfolio_value': portfolio_value_after,
            'portfolio_return': (portfolio_value_after - self.initial_capital) / self.initial_capital,
            'num_positions': len(self.positions),
            'capital': self.capital,
            'trade_executed': trade_executed,
            'action_name': action_name,
            'data_quality': 0.8,  # Always good quality for synthetic data
            'raw_return': raw_return,  # For debugging
            'transaction_costs': step_transaction_costs,  # NEW: Track transaction costs
            'transaction_cost_breakdown': transaction_cost_breakdown,  # NEW: Detailed breakdown
            'net_return': net_return,  # NEW: Return after transaction costs
            'reward': reward  # For debugging
        }

        return self._get_observation(), reward, done, info
    
    def _get_current_market_data(self) -> Optional[Dict]:
        """Get current market data"""
        if not self.data_loaded or self.current_symbol not in self.market_data:
            return None
        
        data = self.market_data[self.current_symbol]
        if self.current_data_index >= len(data):
            return None
        
        return data.iloc[self.current_data_index].to_dict()
    
    def _calculate_portfolio_value(self, current_data: Dict) -> float:
        """Calculate total portfolio value using REAL option prices from data at current timestamp"""
        total_value = self.capital

        # Get current timestamp for option price lookup
        current_timestamp = current_data.get('timestamp')

        # Add value of open positions
        current_price = current_data['close']
        for position in self.positions:
            position_type = position['type']
            strike = position['strike']
            symbol = position.get('symbol', self.current_symbol)
            quantity = position.get('quantity', 1)

            # Try to find real option contract with current market price AT CURRENT TIMESTAMP
            option_contract = None
            if position_type in ['call', 'covered_call']:
                option_contract = self._find_option_contract(symbol, strike, 'call', timestamp=current_timestamp)
            elif position_type in ['put', 'cash_secured_put']:
                option_contract = self._find_option_contract(symbol, strike, 'put', timestamp=current_timestamp)

            # Use real option price if available, otherwise fall back to intrinsic value only
            if option_contract and 'close' in option_contract:
                option_value = option_contract['close']
            else:
                # Fallback: Use only intrinsic value (no fake time value!)
                if position_type in ['call', 'covered_call']:
                    option_value = max(0, current_price - strike)
                else:  # put or cash_secured_put
                    option_value = max(0, strike - current_price)

            # Handle standard call/put positions (long)
            if position_type in ['call', 'put']:
                total_value += option_value * quantity * 100  # 100 shares per contract

            # Handle covered call positions (short call)
            elif position_type == 'covered_call':
                # For short calls, we owe the option value (liability)
                total_value -= option_value * quantity * 100

            # Handle cash-secured put positions (short put)
            elif position_type == 'cash_secured_put':
                # For short puts, we owe the option value (liability)
                total_value -= option_value * quantity * 100
                # Note: capital_reserved is already deducted from self.capital

        return total_value

    def _calculate_technical_indicators(self, price_series: np.ndarray) -> Dict[str, float]:
        """Calculate technical indicators for a price series"""
        if len(price_series) < max(self.rsi_window, self.macd_slow, self.volatility_window):
            # Not enough data, return neutral values
            return {
                'rsi': 50.0,
                'macd': 0.0,
                'macd_signal': 0.0,
                'volatility': 0.2,
                'bollinger_upper': price_series[-1] * 1.02,
                'bollinger_lower': price_series[-1] * 0.98
            }

        # RSI calculation
        deltas = np.diff(price_series)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-self.rsi_window:])
        avg_loss = np.mean(losses[-self.rsi_window:])

        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        # MACD calculation
        if len(price_series) >= self.macd_slow:
            ema_fast = self._calculate_ema(price_series, self.macd_fast)
            ema_slow = self._calculate_ema(price_series, self.macd_slow)
            macd = ema_fast - ema_slow
            macd_signal = macd * 0.9  # Simplified signal line
        else:
            macd = 0.0
            macd_signal = 0.0

        # Volatility (rolling standard deviation of returns)
        if len(price_series) >= self.volatility_window:
            returns = np.diff(price_series) / price_series[:-1]
            volatility = np.std(returns[-self.volatility_window:]) * np.sqrt(252)  # Annualized
        else:
            volatility = 0.2

        # Bollinger Bands
        if len(price_series) >= 20:
            sma = np.mean(price_series[-20:])
            std = np.std(price_series[-20:])
            bollinger_upper = sma + (2 * std)
            bollinger_lower = sma - (2 * std)
        else:
            current_price = price_series[-1]
            bollinger_upper = current_price * 1.02
            bollinger_lower = current_price * 0.98

        return {
            'rsi': rsi,
            'macd': macd,
            'macd_signal': macd_signal,
            'volatility': volatility,
            'bollinger_upper': bollinger_upper,
            'bollinger_lower': bollinger_lower
        }

    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.mean(prices)

        multiplier = 2 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return ema

    def _create_dummy_observation(self, num_symbols: int) -> Dict[str, np.ndarray]:
        """Create dummy observation in CLSTM-PPO format"""
        return {
            'price_history': np.full((num_symbols, self.lookback_window), 100.0, dtype=np.float32),
            'technical_indicators': np.zeros(num_symbols * 6, dtype=np.float32),
            'options_chain': np.zeros((self.max_positions, 8), dtype=np.float32),
            'portfolio_state': np.array([self.capital, self.capital, 0.0, 0.0, 0.0], dtype=np.float32),
            'greeks_summary': np.zeros(self.max_positions * 4, dtype=np.float32),
            'symbol_encoding': np.eye(num_symbols)[0] if num_symbols > 0 else np.array([1.0], dtype=np.float32),
            'market_microstructure': np.tile([1000000.0, 1e12, 0.01], num_symbols).astype(np.float32),
            'time_features': np.array([0.5, 2.0, 0.0], dtype=np.float32)
        }

    def _create_clstm_observation(self, prices, volumes, technical_data, market_data) -> Dict[str, np.ndarray]:
        """Create observation in CLSTM-PPO format from market data"""
        num_symbols = len(self.symbols)

        # Price history matrix (symbols x time)
        price_history = np.zeros((num_symbols, self.lookback_window), dtype=np.float32)
        for i, symbol in enumerate(self.symbols):
            if symbol in self.market_data:
                data = self.market_data[symbol]
                current_idx = min(self.current_data_index, len(data) - 1)
                start_idx = max(0, current_idx - self.lookback_window)
                end_idx = current_idx + 1

                price_series = data.iloc[start_idx:end_idx]['close'].values
                # Pad or truncate to lookback_window
                if len(price_series) < self.lookback_window:
                    padded = np.full(self.lookback_window, price_series[0] if len(price_series) > 0 else 100.0)
                    padded[-len(price_series):] = price_series
                    price_history[i] = padded
                else:
                    price_history[i] = price_series[-self.lookback_window:]
            else:
                price_history[i] = np.full(self.lookback_window, 100.0)

        # Technical indicators (flattened: RSI, MACD, volatility, etc. for each symbol)
        technical_indicators = []
        for i, symbol in enumerate(self.symbols):
            if i < len(technical_data):
                indicators = technical_data[i]
                technical_indicators.extend([
                    indicators.get('rsi', 50.0) / 100.0,  # Normalize to 0-1
                    indicators.get('macd', 0.0) / 10.0,   # Normalize
                    indicators.get('volatility', 0.2) / 2.0,  # Normalize
                    (indicators.get('bollinger_upper', 102.0) - 100.0) / 100.0,
                    (indicators.get('bollinger_lower', 98.0) - 100.0) / 100.0,
                    0.0  # Reserved for additional indicator
                ])
            else:
                technical_indicators.extend([0.5, 0.0, 0.1, 0.02, -0.02, 0.0])

        # Options chain (current positions)
        options_chain = np.zeros((self.max_positions, 8), dtype=np.float32)
        for i, position in enumerate(self.positions[:self.max_positions]):
            options_chain[i] = [
                1.0 if position['type'] == 'call' else -1.0,  # Type
                position['strike'] / 1000.0,  # Normalized strike
                position['entry_price'] / 100.0,  # Normalized price
                position['quantity'] / 100.0,  # Normalized quantity
                (self.current_step - position['entry_step']) / 100.0,  # Time held
                0.0, 0.0, 0.0  # Reserved for additional data
            ]

        # Portfolio state
        portfolio_value = self._calculate_portfolio_value({'close': prices[0]}) if prices else self.capital
        drawdown = max(0, (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value)

        portfolio_state = np.array([
            self.capital / 100000.0,  # Normalized capital
            portfolio_value / 100000.0,  # Normalized portfolio value
            drawdown,  # Drawdown ratio
            len(self.positions) / self.max_positions,  # Position utilization
            self.current_step / self.episode_length  # Episode progress
        ], dtype=np.float32)

        # OPTIMIZATION: Vectorized Greeks summary extraction
        greeks_summary = np.zeros(self.max_positions * 4, dtype=np.float32)

        if self.positions:
            # Extract Greeks from positions in vectorized manner
            num_positions = min(len(self.positions), self.max_positions)
            for i in range(num_positions):
                position = self.positions[i]
                # Use array slicing for faster assignment
                greeks_summary[i*4:(i+1)*4] = [
                    position.get('delta', 0.0),
                    position.get('gamma', 0.0),
                    position.get('theta', 0.0),
                    position.get('vega', 0.0)
                ]

        # Symbol encoding (one-hot for current primary symbol)
        symbol_encoding = np.zeros(num_symbols, dtype=np.float32)
        if num_symbols > 0:
            symbol_encoding[0] = 1.0  # Primary symbol

        # Market microstructure (volume, market cap, spread for each symbol)
        market_microstructure = []
        for i, symbol in enumerate(self.symbols):
            if i < len(volumes) and i < len(market_data):
                market_microstructure.extend([
                    volumes[i] / 1e6,  # Normalized volume
                    market_data[i].get('market_cap', 1e12) / 1e12,  # Normalized market cap
                    market_data[i].get('bid_ask_spread', 0.01)  # Spread
                ])
            else:
                market_microstructure.extend([1.0, 1.0, 0.01])

        # Time features
        time_of_day = (self.current_step % 390) / 390
        day_of_week = (self.current_step // 390) % 5 / 5.0
        time_features = np.array([time_of_day, day_of_week, self.current_step / self.episode_length], dtype=np.float32)

        return {
            'price_history': price_history,
            'technical_indicators': np.array(technical_indicators, dtype=np.float32),
            'options_chain': options_chain,
            'portfolio_state': portfolio_state,
            'greeks_summary': greeks_summary,
            'symbol_encoding': symbol_encoding,
            'market_microstructure': np.array(market_microstructure, dtype=np.float32),
            'time_features': time_features
        }

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation in CLSTM-PPO compatible format"""
        if not self.data_loaded:
            # Return dummy observation in CLSTM-PPO format
            num_symbols = len(self.symbols)
            return self._create_dummy_observation(num_symbols)
        
        # Collect market data for CLSTM-PPO format
        prices = []
        volumes = []
        technical_data = []
        market_data = []

        for symbol in self.symbols:
            if symbol in self.market_data:
                data = self.market_data[symbol]
                current_idx = min(self.current_data_index, len(data) - 1)

                current_price = data.iloc[current_idx]['close']
                current_volume = data.iloc[current_idx]['volume']

                prices.append(current_price)
                volumes.append(current_volume)

                # Calculate technical indicators
                if self.include_technical_indicators:
                    indicator_start = max(0, current_idx - 50)
                    indicator_prices = data.iloc[indicator_start:current_idx + 1]['close'].values
                    indicators = self._calculate_technical_indicators(indicator_prices)
                    technical_data.append(indicators)
                else:
                    technical_data.append({
                        'rsi': 50.0, 'macd': 0.0, 'volatility': 0.2,
                        'bollinger_upper': current_price * 1.02,
                        'bollinger_lower': current_price * 0.98
                    })

                # Market microstructure
                if self.include_market_microstructure:
                    spread = min(0.1, technical_data[-1]['volatility'] * 0.1)

                    # Market cap estimation
                    if 'SPY' in symbol or 'QQQ' in symbol or 'IWM' in symbol:
                        market_cap = 1e12
                    elif symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']:
                        market_cap = 2e12
                    elif symbol in ['TSLA', 'META', 'NFLX']:
                        market_cap = 8e11
                    else:
                        market_cap = 5e10

                    market_data.append({
                        'bid_ask_spread': spread,
                        'market_cap': market_cap
                    })
                else:
                    market_data.append({
                        'bid_ask_spread': 0.01,
                        'market_cap': 1e12
                    })
            else:
                # Default values
                prices.append(100.0)
                volumes.append(1000000.0)
                technical_data.append({
                    'rsi': 50.0, 'macd': 0.0, 'volatility': 0.2,
                    'bollinger_upper': 102.0, 'bollinger_lower': 98.0
                })
                market_data.append({
                    'bid_ask_spread': 0.01,
                    'market_cap': 1e12
                })

        # Create CLSTM-PPO compatible observation
        return self._create_clstm_observation(prices, volumes, technical_data, market_data)

    def get_current_returns(self) -> np.ndarray:
        """
        Get current period returns for turbulence calculation
        Paper optimization: needed for risk management
        """
        try:
            if not hasattr(self, 'historical_returns') or len(self.historical_returns) < 2:
                return np.zeros(len(self.symbols))

            # Calculate returns from last two periods
            current_prices = []
            prev_prices = []

            for symbol in self.symbols:
                if symbol in self.market_data and len(self.market_data[symbol]) >= 2:
                    data = self.market_data[symbol]
                    current_prices.append(data.iloc[-1]['close'])
                    prev_prices.append(data.iloc[-2]['close'])
                else:
                    current_prices.append(100.0)
                    prev_prices.append(100.0)

            current_prices = np.array(current_prices)
            prev_prices = np.array(prev_prices)

            # Calculate returns, avoiding division by zero
            returns = np.where(prev_prices > 0,
                             (current_prices - prev_prices) / prev_prices,
                             0.0)

            return returns

        except Exception as e:
            logger.debug(f"Error calculating current returns: {e}")
            return np.zeros(len(self.symbols))

    @property
    def historical_returns(self) -> np.ndarray:
        """
        Get historical returns matrix for turbulence calculation
        Paper optimization: needed for covariance matrix
        """
        try:
            if not hasattr(self, '_historical_returns_cache'):
                self._calculate_historical_returns()
            return self._historical_returns_cache
        except:
            return np.zeros((30, len(self.symbols)))  # Default 30 periods

    def _calculate_historical_returns(self):
        """Calculate and cache historical returns matrix"""
        try:
            returns_data = []
            min_length = float('inf')

            # Get returns for each symbol
            for symbol in self.symbols:
                if symbol in self.market_data and len(self.market_data[symbol]) > 1:
                    data = self.market_data[symbol]
                    prices = data['close'].values
                    returns = np.diff(prices) / prices[:-1]
                    returns_data.append(returns)
                    min_length = min(min_length, len(returns))
                else:
                    # Generate synthetic returns for missing data
                    returns = np.random.normal(0, 0.02, 30)  # 2% volatility
                    returns_data.append(returns)
                    min_length = min(min_length, len(returns))

            # Ensure all return series have same length
            if min_length == float('inf') or min_length < 10:
                min_length = 30

            # Truncate to same length and stack
            aligned_returns = []
            for returns in returns_data:
                if len(returns) >= min_length:
                    aligned_returns.append(returns[-min_length:])
                else:
                    # Pad with zeros if needed
                    padded = np.zeros(min_length)
                    padded[-len(returns):] = returns
                    aligned_returns.append(padded)

            self._historical_returns_cache = np.column_stack(aligned_returns)

        except Exception as e:
            logger.debug(f"Error calculating historical returns: {e}")
            # Fallback to synthetic data
            self._historical_returns_cache = np.random.normal(0, 0.02, (30, len(self.symbols)))

    def get_enhanced_technical_indicators(self, symbol: str) -> Dict[str, float]:
        """
        Get enhanced technical indicators as used in the paper (OPTIMIZED with caching)
        Includes MACD, RSI, CCI, ADX
        """
        # OPTIMIZATION: Check cache first (cache per step to avoid recalculation)
        cache_key = (symbol, self.current_step)
        if cache_key in self._technical_indicators_cache:
            return self._technical_indicators_cache[cache_key]

        try:
            if symbol not in self.market_data or len(self.market_data[symbol]) < 30:
                result = {
                    'macd': 0.0,
                    'rsi': 50.0,
                    'cci': 0.0,
                    'adx': 25.0
                }
                self._technical_indicators_cache[cache_key] = result
                return result

            data = self.market_data[symbol]
            close_prices = data['close']
            high_prices = data['high'] if 'high' in data.columns else close_prices
            low_prices = data['low'] if 'low' in data.columns else close_prices

            # Calculate paper's technical indicators
            macd = TechnicalIndicators.calculate_macd(close_prices)
            rsi = TechnicalIndicators.calculate_rsi(close_prices)
            cci = TechnicalIndicators.calculate_cci(high_prices, low_prices, close_prices)
            adx = TechnicalIndicators.calculate_adx(high_prices, low_prices, close_prices)

            result = {
                'macd': macd,
                'rsi': rsi,
                'cci': cci,
                'adx': adx
            }
            self._technical_indicators_cache[cache_key] = result
            return result

        except Exception as e:
            logger.debug(f"Error calculating enhanced technical indicators for {symbol}: {e}")
            result = {
                'macd': 0.0,
                'rsi': 50.0,
                'cci': 0.0,
                'adx': 25.0
            }
            self._technical_indicators_cache[cache_key] = result
            return result
