#!/usr/bin/env python3
"""
Paper Trading Environment for live market simulation
Executes trades in real-time without real money
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict

from src.data.realtime_stream import MassiveRealtimeStream, fetch_options_chain_snapshot

# Import unified Greeks calculator for consistency with historical training
try:
    from src.utils.greeks import GreeksCalculator, get_greeks_calculator
    GREEKS_CALCULATOR = get_greeks_calculator(risk_free_rate=0.05)
except ImportError:
    GREEKS_CALCULATOR = None

logger = logging.getLogger(__name__)


class Portfolio:
    """Manages paper trading portfolio state"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # {ticker: {'quantity': int, 'entry_price': float, 'entry_time': datetime}}
        self.trade_history = []
        self.equity_curve = []
        
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        positions_value = 0
        for ticker, position in self.positions.items():
            current_price = current_prices.get(ticker, position['entry_price'])
            positions_value += position['quantity'] * current_price * 100  # Options are per 100 shares
        
        return self.cash + positions_value
    
    def execute_trade(self, ticker: str, action: str, quantity: int, price: float, 
                     commission: float = 0.65, timestamp: datetime = None) -> Dict:
        """
        Execute a paper trade
        
        Args:
            ticker: Option ticker (e.g., 'O:SPY251124C00500000')
            action: 'buy' or 'sell'
            quantity: Number of contracts
            price: Price per contract
            commission: Commission per contract
            timestamp: Trade timestamp
        
        Returns:
            Trade result dict
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        contract_value = quantity * price * 100  # Options are per 100 shares
        total_cost = contract_value + (commission * quantity)
        
        if action == 'buy':
            if self.cash < total_cost:
                return {
                    'success': False,
                    'reason': 'insufficient_funds',
                    'required': total_cost,
                    'available': self.cash
                }
            
            # Execute buy
            self.cash -= total_cost
            
            if ticker in self.positions:
                # Average up position
                old_qty = self.positions[ticker]['quantity']
                old_price = self.positions[ticker]['entry_price']
                new_qty = old_qty + quantity
                new_avg_price = ((old_qty * old_price) + (quantity * price)) / new_qty
                
                self.positions[ticker]['quantity'] = new_qty
                self.positions[ticker]['entry_price'] = new_avg_price
            else:
                # New position
                self.positions[ticker] = {
                    'quantity': quantity,
                    'entry_price': price,
                    'entry_time': timestamp
                }
            
            trade_record = {
                'timestamp': timestamp,
                'ticker': ticker,
                'action': 'buy',
                'quantity': quantity,
                'price': price,
                'commission': commission * quantity,
                'total_cost': total_cost,
                'cash_after': self.cash
            }
            
        elif action == 'sell':
            if ticker not in self.positions or self.positions[ticker]['quantity'] < quantity:
                return {
                    'success': False,
                    'reason': 'insufficient_position',
                    'required': quantity,
                    'available': self.positions.get(ticker, {}).get('quantity', 0)
                }
            
            # Execute sell
            proceeds = contract_value - (commission * quantity)
            self.cash += proceeds
            
            # Calculate P&L
            entry_price = self.positions[ticker]['entry_price']
            pnl = (price - entry_price) * quantity * 100 - (commission * quantity * 2)  # Round-trip commission
            
            # Update position
            self.positions[ticker]['quantity'] -= quantity
            if self.positions[ticker]['quantity'] == 0:
                del self.positions[ticker]
            
            trade_record = {
                'timestamp': timestamp,
                'ticker': ticker,
                'action': 'sell',
                'quantity': quantity,
                'price': price,
                'entry_price': entry_price,
                'commission': commission * quantity,
                'proceeds': proceeds,
                'pnl': pnl,
                'cash_after': self.cash
            }
        
        else:
            return {'success': False, 'reason': 'invalid_action'}
        
        # Record trade
        self.trade_history.append(trade_record)
        
        return {
            'success': True,
            'trade': trade_record
        }
    
    def get_position(self, ticker: str) -> Optional[Dict]:
        """Get current position for a ticker"""
        return self.positions.get(ticker)
    
    def get_all_positions(self) -> Dict:
        """Get all current positions"""
        return self.positions.copy()
    
    def get_metrics(self, current_prices: Dict[str, float]) -> Dict:
        """Calculate portfolio metrics"""
        portfolio_value = self.get_portfolio_value(current_prices)
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital
        
        # Calculate win rate from closed trades
        closed_trades = [t for t in self.trade_history if t['action'] == 'sell']
        winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
        
        return {
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'num_positions': len(self.positions),
            'total_trades': len(self.trade_history),
            'closed_trades': len(closed_trades),
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100
        }


class PaperTradingEnvironment:
    """
    Live paper trading environment with real-time market data
    Compatible with RL agent interface
    """

    def __init__(self, api_key: str, symbols: List[str], initial_capital: float = 100000,
                 max_positions: int = 5, commission: float = 0.65):
        self.api_key = api_key
        self.symbols = symbols
        self.max_positions = max_positions
        self.commission = commission

        # Portfolio
        self.portfolio = Portfolio(initial_capital)

        # Real-time data stream
        self.data_stream = MassiveRealtimeStream(api_key, symbols)

        # Current market state
        self.current_state = None
        self.last_update_time = None

        # Action space (same as training environment)
        # 0: hold, 1-30: various option strategies
        self.action_space_n = 31  # Will be updated based on multi-leg support

        # Episode tracking
        self.step_count = 0
        self.episode_start_time = None

    async def initialize(self):
        """Initialize the paper trading environment"""
        logger.info("🚀 Initializing paper trading environment...")

        # Start real-time data stream
        asyncio.create_task(self.data_stream.start())

        # Wait for connection
        for _ in range(30):  # Wait up to 30 seconds
            if self.data_stream.connected:
                break
            await asyncio.sleep(1)

        if not self.data_stream.connected:
            raise RuntimeError("Failed to connect to real-time data stream")

        # Fetch initial options chain
        logger.info("📊 Fetching initial options chain...")
        for symbol in self.symbols:
            options_chain = await fetch_options_chain_snapshot(self.api_key, symbol)
            logger.info(f"  {symbol}: {len(options_chain)} options available")

        # Get initial state
        self.current_state = self.data_stream.get_current_state()
        self.last_update_time = datetime.utcnow()
        self.episode_start_time = datetime.utcnow()

        logger.info("✅ Paper trading environment initialized")

    def get_observation(self) -> np.ndarray:
        """
        Get current observation for RL agent
        Format matches training environment
        """
        # Get current market state
        state = self.data_stream.get_current_state()

        # Build observation vector (simplified - expand based on your training env)
        obs_components = []

        # Portfolio state
        current_prices = self._extract_current_prices(state)
        portfolio_value = self.portfolio.get_portfolio_value(current_prices)
        cash_ratio = self.portfolio.cash / portfolio_value if portfolio_value > 0 else 1.0

        obs_components.extend([
            cash_ratio,
            len(self.portfolio.positions) / self.max_positions,  # Position utilization
            portfolio_value / self.portfolio.initial_capital - 1.0  # Return
        ])

        # Market state for each symbol
        for symbol in self.symbols:
            stock_quote = state['stocks'].get(symbol, {})

            if stock_quote:
                bid = stock_quote.get('bid', 0)
                ask = stock_quote.get('ask', 0)
                mid = (bid + ask) / 2 if bid and ask else 0
                spread = (ask - bid) / mid if mid > 0 else 0

                obs_components.extend([
                    mid / 100.0,  # Normalize price
                    spread,
                    stock_quote.get('bid_size', 0) / 1000.0,  # Normalize size
                    stock_quote.get('ask_size', 0) / 1000.0
                ])
            else:
                obs_components.extend([0, 0, 0, 0])

        return np.array(obs_components, dtype=np.float32)

    def _extract_current_prices(self, state: Dict) -> Dict[str, float]:
        """Extract current prices from market state"""
        prices = {}

        # Stock prices
        for symbol, quote in state.get('stocks', {}).items():
            if quote.get('bid') and quote.get('ask'):
                prices[symbol] = (quote['bid'] + quote['ask']) / 2

        # Option prices
        for ticker, quote in state.get('options', {}).items():
            if quote.get('bid') and quote.get('ask'):
                prices[ticker] = (quote['bid'] + quote['ask']) / 2

        return prices

    async def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment

        Args:
            action: Action to take (0=hold, 1-30=various strategies)

        Returns:
            (observation, reward, done, info)
        """
        self.step_count += 1

        # Decode action and execute trade
        trade_result = await self._execute_action(action)

        # Calculate reward
        current_prices = self._extract_current_prices(self.data_stream.get_current_state())
        portfolio_value = self.portfolio.get_portfolio_value(current_prices)
        reward = self._calculate_reward(trade_result, portfolio_value)

        # Get next observation
        next_obs = self.get_observation()

        # Check if done (for now, never done - continuous trading)
        done = False

        # Info dict
        info = {
            'step': self.step_count,
            'portfolio_value': portfolio_value,
            'cash': self.portfolio.cash,
            'num_positions': len(self.portfolio.positions),
            'trade_executed': trade_result.get('success', False),
            'action': action
        }

        if trade_result.get('success'):
            info['trade'] = trade_result['trade']

        return next_obs, reward, done, info

    async def _execute_action(self, action: int) -> Dict:
        """
        Execute trading action

        Action space (simplified):
        0: Hold
        1-10: Buy calls at different strikes
        11-20: Buy puts at different strikes
        21-30: Sell/close positions
        """
        if action == 0:
            return {'success': False, 'reason': 'hold'}

        # Get current market state
        state = self.data_stream.get_current_state()

        # For now, implement simple call/put buying
        # TODO: Expand to multi-leg strategies

        if 1 <= action <= 10:
            # Buy call
            return await self._buy_option(state, 'call', action - 1)
        elif 11 <= action <= 20:
            # Buy put
            return await self._buy_option(state, 'put', action - 11)
        elif 21 <= action <= 30:
            # Close position
            return await self._close_position(action - 21)

        return {'success': False, 'reason': 'invalid_action'}

    async def _buy_option(self, state: Dict, option_type: str, strike_index: int) -> Dict:
        """Buy an option contract"""
        # Select symbol (for now, use first symbol)
        symbol = self.symbols[0]

        # Get stock price
        stock_quote = state['stocks'].get(symbol)
        if not stock_quote:
            return {'success': False, 'reason': 'no_stock_quote'}

        stock_price = (stock_quote['bid'] + stock_quote['ask']) / 2

        # Find suitable option (ATM + strike_index)
        # This is simplified - in production, use proper option chain filtering
        target_strike = stock_price + (strike_index - 5) * 5  # -25 to +25 in $5 increments

        # Find closest option in chain
        # TODO: Implement proper option selection from real-time chain

        return {'success': False, 'reason': 'not_implemented'}

    async def _close_position(self, position_index: int) -> Dict:
        """Close an existing position"""
        positions = list(self.portfolio.positions.items())

        if position_index >= len(positions):
            return {'success': False, 'reason': 'no_position'}

        ticker, position = positions[position_index]

        # Get current price
        state = self.data_stream.get_current_state()
        option_quote = state['options'].get(ticker)

        if not option_quote:
            return {'success': False, 'reason': 'no_quote'}

        # Sell at bid price
        sell_price = option_quote.get('bid', 0)
        if sell_price <= 0:
            return {'success': False, 'reason': 'invalid_price'}

        # Execute sell
        return self.portfolio.execute_trade(
            ticker=ticker,
            action='sell',
            quantity=position['quantity'],
            price=sell_price,
            commission=self.commission
        )

    def _calculate_reward(self, trade_result: Dict, portfolio_value: float) -> float:
        """
        Calculate reward for RL agent
        Based on portfolio return and trade quality
        """
        # Base reward: portfolio return
        portfolio_return = (portfolio_value - self.portfolio.initial_capital) / self.portfolio.initial_capital
        reward = portfolio_return * 100  # Scale up

        # Penalty for failed trades
        if not trade_result.get('success', False):
            reward -= 0.01

        # Bonus for successful profitable trades
        if trade_result.get('success') and trade_result.get('trade', {}).get('pnl', 0) > 0:
            reward += 0.1

        return reward

    def reset(self) -> np.ndarray:
        """Reset environment (for episodic training)"""
        self.step_count = 0
        self.episode_start_time = datetime.utcnow()
        return self.get_observation()

    async def close(self):
        """Close the environment and cleanup"""
        await self.data_stream.stop()
        logger.info("🛑 Paper trading environment closed")

