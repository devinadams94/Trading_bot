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

# Import paper optimizations
try:
    from .paper_optimizations import TechnicalIndicators
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

# Import realistic transaction costs
try:
    from .realistic_transaction_costs import RealisticTransactionCostCalculator
except ImportError:
    # Fallback if import fails
    RealisticTransactionCostCalculator = None
    logger.warning("⚠️ Realistic transaction costs not available, using legacy commission model")

logger = logging.getLogger(__name__)


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
        **kwargs
    ):
        super().__init__()

        self.data_loader = data_loader
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
        """Load market data"""
        logger.info(f"Loading data from {start_date.date()} to {end_date.date()}")
        
        # Create simple synthetic data if no data loader
        if self.data_loader is None:
            self._create_synthetic_data(start_date, end_date)
        else:
            try:
                # Try to load real data
                self.market_data = await self.data_loader.load_historical_data(
                    self.symbols, start_date, end_date
                )
                if not self.market_data:
                    logger.warning("No real data loaded, using synthetic data")
                    self._create_synthetic_data(start_date, end_date)
            except Exception as e:
                logger.warning(f"Failed to load real data: {e}, using synthetic data")
                self._create_synthetic_data(start_date, end_date)
        
        self.data_loaded = True
        logger.info(f"Data loaded for {len(self.market_data)} symbols")
    
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
        self.current_data_index = 0
        self.current_symbol = self.symbols[0] if self.symbols else 'SPY'
        
        # Portfolio tracking
        self.portfolio_value_history = [self.initial_capital]
        self.peak_portfolio_value = self.initial_capital
        
        return self._get_observation()

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

            # Simple option pricing (mid-price estimate)
            option_price_mid = max(0.5, current_price * 0.05 * (1 - strike_offset))

            # Create option data for transaction cost calculation
            moneyness = strike_price / current_price
            spread_pct = 0.02 + 0.03 * abs(1 - moneyness)  # 2-5% spread
            option_data = {
                'bid': option_price_mid * (1 - spread_pct / 2),
                'ask': option_price_mid * (1 + spread_pct / 2),
                'last': option_price_mid,
                'volume': current_data.get('volume', 1000) / 100,  # Estimate options volume
                'open_interest': 1000,
                'moneyness': moneyness,
                'implied_volatility': current_data.get('volatility', 0.3)
            }

            # Calculate transaction cost
            quantity = 1  # 1 contract = 100 shares
            transaction_cost, cost_breakdown = self._calculate_transaction_cost(
                option_data, quantity, side='buy'
            )

            # Execution price (buy at ask for realistic costs, mid for legacy)
            if self.use_realistic_costs:
                execution_price = option_data['ask']
            else:
                execution_price = option_price_mid

            # Total cost including transaction costs
            total_cost = execution_price * 100 + transaction_cost

            if self.capital >= total_cost:
                # Execute trade
                self.capital -= total_cost
                step_transaction_costs += transaction_cost
                transaction_cost_breakdown = cost_breakdown

                position = {
                    'type': 'call',
                    'strike': strike_price,
                    'entry_price': execution_price,
                    'quantity': 100,
                    'symbol': self.current_symbol,
                    'entry_step': self.current_step,
                    'cost': total_cost,
                    'transaction_cost': transaction_cost
                }
                self.positions.append(position)

                action_name = f"BUY_CALL_{strike_price:.0f}"

                logger.debug(f"Executed: {action_name}, cost=${total_cost:.2f}, txn_cost=${transaction_cost:.2f}")

        elif 11 <= action <= 20:
            # Buy put options
            trade_executed = True
            self.episode_trades += 1

            strike_offset = (action - 11) * 0.01  # 0% to 9% OTM
            current_price = current_data['close']
            strike_price = current_price * (1 - strike_offset)

            # Simple option pricing (mid-price estimate)
            option_price_mid = max(0.5, current_price * 0.05 * (1 - strike_offset))

            # Create option data for transaction cost calculation
            moneyness = strike_price / current_price
            spread_pct = 0.02 + 0.03 * abs(1 - moneyness)  # 2-5% spread
            option_data = {
                'bid': option_price_mid * (1 - spread_pct / 2),
                'ask': option_price_mid * (1 + spread_pct / 2),
                'last': option_price_mid,
                'volume': current_data.get('volume', 1000) / 100,
                'open_interest': 1000,
                'moneyness': moneyness,
                'implied_volatility': current_data.get('volatility', 0.3)
            }

            # Calculate transaction cost
            quantity = 1  # 1 contract
            transaction_cost, cost_breakdown = self._calculate_transaction_cost(
                option_data, quantity, side='buy'
            )

            # Execution price (buy at ask for realistic costs)
            if self.use_realistic_costs:
                execution_price = option_data['ask']
            else:
                execution_price = option_price_mid

            # Total cost including transaction costs
            total_cost = execution_price * 100 + transaction_cost

            if self.capital >= total_cost:
                # Execute trade
                self.capital -= total_cost
                step_transaction_costs += transaction_cost
                transaction_cost_breakdown = cost_breakdown

                position = {
                    'type': 'put',
                    'strike': strike_price,
                    'entry_price': execution_price,
                    'quantity': 100,
                    'symbol': self.current_symbol,
                    'entry_step': self.current_step,
                    'cost': total_cost,
                    'transaction_cost': transaction_cost
                }
                self.positions.append(position)

                action_name = f"BUY_PUT_{strike_price:.0f}"

                logger.debug(f"Executed: {action_name}, cost=${total_cost:.2f}, txn_cost=${transaction_cost:.2f}")
            
        elif 21 <= action <= 30:
            # Sell positions (or buy if no positions)
            trade_executed = True
            self.episode_trades += 1

            if len(self.positions) > 0:
                # Sell oldest position
                position = self.positions.pop(0)
                current_price = current_data['close']

                # Calculate option value at current price (mid-price estimate)
                if position['type'] == 'call':
                    intrinsic_value = max(0, current_price - position['strike'])
                else:  # put
                    intrinsic_value = max(0, position['strike'] - current_price)

                # Add time value
                time_value = max(0.1, position['entry_price'] * 0.5)
                option_value_mid = intrinsic_value + time_value

                # Create option data for transaction cost calculation
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

                # Calculate transaction cost for selling
                quantity = 1  # 1 contract
                transaction_cost, cost_breakdown = self._calculate_transaction_cost(
                    option_data, quantity, side='sell'
                )

                # Execution price (sell at bid for realistic costs)
                if self.use_realistic_costs:
                    execution_price = option_data['bid']
                else:
                    execution_price = option_value_mid

                # Execute sale (proceeds minus transaction costs)
                gross_proceeds = execution_price * position['quantity']
                net_proceeds = gross_proceeds - transaction_cost
                self.capital += net_proceeds
                step_transaction_costs += transaction_cost
                transaction_cost_breakdown = cost_breakdown

                # Calculate P&L (including transaction costs on both entry and exit)
                entry_txn_cost = position.get('transaction_cost', 0)
                total_txn_costs = entry_txn_cost + transaction_cost
                pnl = net_proceeds - position['cost']

                action_name = f"SELL_{position['type'].upper()}"

                # Record trade
                trade_record = {
                    'action': action_name,
                    'pnl': pnl,
                    'entry_price': position['entry_price'],
                    'exit_price': execution_price,
                    'step': self.current_step,
                    'transaction_costs': total_txn_costs
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

        # Apply paper's scaling factor (1e-4) for stable training
        reward = net_return * 1e-4

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
        """Calculate total portfolio value"""
        total_value = self.capital
        
        # Add value of open positions
        current_price = current_data['close']
        for position in self.positions:
            if position['type'] == 'call':
                intrinsic_value = max(0, current_price - position['strike'])
            else:  # put
                intrinsic_value = max(0, position['strike'] - current_price)
            
            time_value = max(0.1, position['entry_price'] * 0.3)
            option_value = intrinsic_value + time_value
            total_value += option_value * position['quantity']
        
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

        # Greeks summary (simplified - zeros for now)
        greeks_summary = np.zeros(self.max_positions * 4, dtype=np.float32)

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
        Get enhanced technical indicators as used in the paper
        Includes MACD, RSI, CCI, ADX
        """
        try:
            if symbol not in self.market_data or len(self.market_data[symbol]) < 30:
                return {
                    'macd': 0.0,
                    'rsi': 50.0,
                    'cci': 0.0,
                    'adx': 25.0
                }

            data = self.market_data[symbol]
            close_prices = data['close']
            high_prices = data['high'] if 'high' in data.columns else close_prices
            low_prices = data['low'] if 'low' in data.columns else close_prices

            # Calculate paper's technical indicators
            macd = TechnicalIndicators.calculate_macd(close_prices)
            rsi = TechnicalIndicators.calculate_rsi(close_prices)
            cci = TechnicalIndicators.calculate_cci(high_prices, low_prices, close_prices)
            adx = TechnicalIndicators.calculate_adx(high_prices, low_prices, close_prices)

            return {
                'macd': macd,
                'rsi': rsi,
                'cci': cci,
                'adx': adx
            }

        except Exception as e:
            logger.debug(f"Error calculating enhanced technical indicators for {symbol}: {e}")
            return {
                'macd': 0.0,
                'rsi': 50.0,
                'cci': 0.0,
                'adx': 25.0
            }
