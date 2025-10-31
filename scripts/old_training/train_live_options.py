#!/usr/bin/env python3
"""
Training script for options trading using Gymnasium environment with CLSTM-PPO architecture.
This script uses real options data from Alpaca for training.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from collections import deque
import random
from pathlib import Path

# Ensure we're in the correct directory
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

# Load environment variables
from dotenv import load_dotenv
env_path = script_dir / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded .env from {env_path}")
else:
    print(f"Warning: .env file not found at {env_path}")

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.options_clstm_ppo import OptionsCLSTMPPOAgent
from src.options_data_collector import AlpacaOptionsDataCollector
from src.options_trading_env import OptionContract, OptionsPosition
from config.config import TradingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Set specific loggers to debug if needed
# logging.getLogger('__main__').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


class AlpacaOptionsGymEnvironment(gym.Env):
    """Gymnasium environment for options trading with real Alpaca data"""
    
    def __init__(
        self,
        config: TradingConfig,
        symbols: List[str],
        initial_capital: float = 100000,
        max_positions: int = 10,
        lookback_days: int = 30,
        episode_length: int = 252,  # Trading days in a year
        training_mode: bool = True
    ):
        super().__init__()
        
        self.config = config
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.lookback_days = lookback_days
        self.episode_length = episode_length
        self.training_mode = training_mode
        
        # Initialize Alpaca data collector
        base_url = 'https://paper-api.alpaca.markets' if config.alpaca_paper_trading else 'https://api.alpaca.markets'
        logger.info(f"Initializing Alpaca with base URL: {base_url}")
        self.data_collector = AlpacaOptionsDataCollector(
            api_key=config.alpaca_api_key,
            api_secret=config.alpaca_secret_key,
            base_url=base_url
        )
        
        # State components
        self.current_step = 0
        self.current_capital = initial_capital
        self.positions: List[OptionsPosition] = []
        self.portfolio_value_history = []
        self.trade_history = []
        
        # Historical data cache
        self.price_history = {}
        self.options_chains = {}
        self.technical_indicators = {}
        
        # Define action and observation spaces
        self._define_spaces()
        
        # Pre-load historical data
        if training_mode:
            self._preload_training_data()
    
    def _define_spaces(self):
        """Define the action and observation spaces"""
        # Action space: 0=hold, 1-5=buy call/put, 6-10=sell call/put
        self.action_space = spaces.Discrete(11)
        
        # Observation space components
        self.observation_space = spaces.Dict({
            # Price history: last 130 hours (~20 trading days * 6.5 hours/day) of OHLCV for each symbol
            'price_history': spaces.Box(
                low=0, high=np.inf, shape=(len(self.symbols), 130, 5), dtype=np.float32
            ),
            # Technical indicators - using 14 to match pre-trained network
            # but we'll select the most important from our 20 calculated indicators
            'technical_indicators': spaces.Box(
                low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
            ),
            # Options chain summary (top 5 strikes for calls and puts)
            'options_chain': spaces.Box(
                low=-np.inf, high=np.inf, shape=(10, 8), dtype=np.float32
            ),
            # Portfolio state
            'portfolio_state': spaces.Box(
                low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
            ),
            # Greeks summary
            'greeks_summary': spaces.Box(
                low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
            )
        })
    
    def _preload_training_data(self):
        """Pre-load historical data for training"""
        logger.info("Pre-loading historical data for training...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 2)  # 2 years of data
        
        for symbol in self.symbols:
            try:
                # Get stock price history using Alpaca API with IEX feed (free)
                # Using 1Hour bars for more granular data (6.5 hours * 252 days = ~1638 data points per year)
                logger.info(f"Fetching hourly data for {symbol} from {start_date} to {end_date}")
                bars = self.data_collector.api.get_bars(
                    symbol, 
                    '1Hour',  # Changed from '1Day' to '1Hour' for more data points
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    feed='iex'  # Use IEX feed instead of SIP
                ).df
                
                if not bars.empty:
                    df = bars.reset_index()
                    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
                    logger.info(f"Loaded {len(df)} days of data for {symbol}, first date: {df.iloc[0]['timestamp']}, last date: {df.iloc[-1]['timestamp']}")
                else:
                    logger.error(f"No data returned for {symbol}")
                    df = pd.DataFrame()
                self.price_history[symbol] = df
                
                # Calculate technical indicators
                if not df.empty:
                    self.technical_indicators[symbol] = self._calculate_technical_indicators(df)
                
                logger.info(f"Loaded {len(df)} days of data for {symbol}")
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
                import traceback
                traceback.print_exc()
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the given price data"""
        indicators = pd.DataFrame(index=df.index)
        
        # Price data
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # Simple Moving Averages
        indicators['SMA_10'] = close.rolling(window=10).mean()
        indicators['SMA_20'] = close.rolling(window=20).mean()
        indicators['SMA_50'] = close.rolling(window=50).mean()
        
        # Exponential Moving Averages for MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        
        # MACD - Moving Average Convergence Divergence
        indicators['MACD'] = ema_12 - ema_26
        indicators['MACD_signal'] = indicators['MACD'].ewm(span=9, adjust=False).mean()
        indicators['MACD_histogram'] = indicators['MACD'] - indicators['MACD_signal']
        
        # RSI - Relative Strength Index
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)  # Avoid division by zero
        indicators['RSI'] = 100 - (100 / (1 + rs))
        
        # CCI - Commodity Channel Index
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
        indicators['CCI'] = (typical_price - sma_tp) / (0.015 * mad + 1e-10)
        
        # ADX - Average Directional Index
        # Calculate True Range
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        
        # Calculate directional movements
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        # Smooth the directional movements
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / (atr + 1e-10))
        
        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        indicators['ADX'] = dx.rolling(window=14).mean()
        
        # Additional indicators for context
        indicators['Plus_DI'] = plus_di
        indicators['Minus_DI'] = minus_di
        
        # Bollinger Bands
        indicators['BB_middle'] = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        indicators['BB_upper'] = indicators['BB_middle'] + 2 * bb_std
        indicators['BB_lower'] = indicators['BB_middle'] - 2 * bb_std
        indicators['BB_width'] = indicators['BB_upper'] - indicators['BB_lower']
        
        # Volume indicators
        indicators['Volume_SMA'] = volume.rolling(window=20).mean()
        indicators['Volume_ratio'] = volume / (indicators['Volume_SMA'] + 1e-10)
        
        # Price position indicators
        indicators['Price_to_SMA20'] = close / (indicators['SMA_20'] + 1e-10)
        indicators['High_Low_Ratio'] = high / (low + 1e-10)
        
        return indicators
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_capital = self.initial_capital
        self.positions = []
        self.portfolio_value_history = [self.initial_capital]
        self.trade_history = []
        
        # Select random starting point in historical data
        if self.training_mode and len(self.price_history) > 0:
            symbol = self.symbols[0]
            data_length = len(self.price_history[symbol])
            max_start = data_length - self.episode_length - self.lookback_days
            
            if max_start > 0:
                self.start_index = random.randint(self.lookback_days, max_start)
            else:
                self.start_index = self.lookback_days
        else:
            self.start_index = 0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """Execute one step in the environment"""
        # Store previous portfolio value
        prev_portfolio_value = self._calculate_portfolio_value()
        
        # Execute action
        trade_info = self._execute_action(action)
        
        # Update positions (check expirations, update prices)
        self._update_positions()
        
        # Calculate reward
        current_portfolio_value = self._calculate_portfolio_value()
        self.portfolio_value_history.append(current_portfolio_value)
        
        # Calculate reward based on profit/loss and risk-adjusted returns
        pnl = current_portfolio_value - prev_portfolio_value
        pnl_pct = pnl / prev_portfolio_value
        
        # Risk-adjusted reward
        if len(self.portfolio_value_history) > 20:
            # Calculate returns from the last 20 portfolio values
            recent_values = np.array(self.portfolio_value_history[-21:])  # Get last 21 values
            returns = np.diff(recent_values) / recent_values[:-1]  # Calculate returns
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            reward = pnl_pct + 0.1 * sharpe
        else:
            reward = pnl_pct
        
        # Add trade cost penalty
        if trade_info and trade_info['executed']:
            reward -= 0.001  # Transaction cost penalty
        
        # Add small penalty for not trading to encourage exploration
        if action == 0 and len(self.positions) == 0:
            reward -= 0.0001
        
        # Update step
        self.current_step += 1
        
        # Check if episode is done
        terminated = (
            self.current_capital <= self.initial_capital * 0.2 or  # 80% loss
            self.current_step >= self.episode_length
        )
        truncated = False
        
        # Get new observation
        observation = self._get_observation()
        info = self._get_info()
        info['trade_info'] = trade_info
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation"""
        obs = {}
        
        # Debug: Log dimensions on first call
        if self.current_step == 0:
            logger.debug("Observation space dimensions:")
        
        # Get price history
        price_data = []
        for symbol in self.symbols:
            if symbol in self.price_history:
                df = self.price_history[symbol]
                current_idx = self.start_index + self.current_step
                
                if current_idx >= 130 and current_idx < len(df):
                    # Get last 130 hours of OHLCV data
                    hist = df.iloc[current_idx-130:current_idx][['open', 'high', 'low', 'close', 'volume']].values
                    price_data.append(hist)
                else:
                    price_data.append(np.zeros((130, 5)))
            else:
                price_data.append(np.zeros((130, 5)))
        
        # Ensure no NaN values in price history
        price_array = np.array(price_data, dtype=np.float32)
        price_array = np.nan_to_num(price_array, nan=0.0, posinf=1e6, neginf=0.0)
        obs['price_history'] = price_array
        
        # Get technical indicators (normalized)
        tech_indicators = []
        if self.symbols[0] in self.technical_indicators:
            df = self.technical_indicators[self.symbols[0]]
            current_idx = self.start_index + self.current_step
            
            if current_idx < len(df) and self.symbols[0] in self.price_history:
                row = df.iloc[current_idx]
                price_df = self.price_history[self.symbols[0]]
                current_price = price_df.iloc[current_idx]['close'] if current_idx < len(price_df) else 100
                
                # Normalize all indicators for neural network input
                tech_indicators = [
                    # 1. RSI (0-100) -> normalize to 0-1
                    np.clip(row.get('RSI', 50) / 100, 0, 1),
                    
                    # 2-4. MACD components -> normalize with tanh
                    np.tanh(row.get('MACD', 0) / (current_price * 0.02)),  # MACD relative to price
                    np.tanh(row.get('MACD_signal', 0) / (current_price * 0.02)),
                    np.tanh(row.get('MACD_histogram', 0) / (current_price * 0.01)),
                    
                    # 5. CCI (-100 to 100 typically) -> normalize with tanh
                    np.tanh(row.get('CCI', 0) / 100),
                    
                    # 6. ADX (0-100) -> normalize to 0-1
                    np.clip(row.get('ADX', 25) / 100, 0, 1),
                    
                    # 7-8. Directional indicators
                    np.clip(row.get('Plus_DI', 0) / 50, 0, 1),
                    np.clip(row.get('Minus_DI', 0) / 50, 0, 1),
                    
                    # 9-11. Moving average relationships
                    np.tanh((row.get('SMA_10', current_price) - row.get('SMA_20', current_price)) / current_price),
                    np.tanh((row.get('SMA_20', current_price) - row.get('SMA_50', current_price)) / current_price),
                    row.get('Price_to_SMA20', 1),  # Already normalized
                    
                    # 12-14. Bollinger Bands position
                    (current_price - row.get('BB_lower', current_price)) / (row.get('BB_width', 1) + 1e-10),
                    row.get('BB_width', 0) / current_price,  # BB width relative to price
                    
                    # 14. Volume indicators
                    np.clip(row.get('Volume_ratio', 1), 0, 3),  # Cap at 3x average
                    
                    # 15. Price volatility (High/Low ratio)
                    row.get('High_Low_Ratio', 1),
                    
                    # 16-18. Trend strength indicators
                    # 16. Is RSI oversold/overbought
                    1.0 if row.get('RSI', 50) < 30 else (-1.0 if row.get('RSI', 50) > 70 else 0.0),
                    # 17. MACD crossover signal
                    1.0 if row.get('MACD', 0) > row.get('MACD_signal', 0) else -1.0,
                    # 18. ADX trend strength
                    1.0 if row.get('ADX', 0) > 25 else 0.0,
                ]
                
                # Select the 14 most important indicators
                # Keep: RSI, MACD (3 components), CCI, ADX, Directional indicators, 
                # MA relationships, BB position, Volume ratio, Trend signals
                selected_indicators = [
                    tech_indicators[0],   # RSI
                    tech_indicators[1],   # MACD
                    tech_indicators[2],   # MACD signal
                    tech_indicators[3],   # MACD histogram
                    tech_indicators[4],   # CCI
                    tech_indicators[5],   # ADX
                    tech_indicators[6],   # Plus_DI
                    tech_indicators[7],   # Minus_DI
                    tech_indicators[8],   # SMA 10-20 diff
                    tech_indicators[9],   # SMA 20-50 diff
                    tech_indicators[11],  # BB position
                    tech_indicators[13],  # Volume ratio (index 13)
                    tech_indicators[15] if len(tech_indicators) > 15 else 0.0,  # RSI signal
                    tech_indicators[16] if len(tech_indicators) > 16 else 0.0,  # MACD crossover
                ]
                tech_indicators = selected_indicators
            else:
                tech_indicators = [0.5] * 14  # Neutral values
        else:
            tech_indicators = [0.5] * 14  # Neutral values
        
        # Ensure no NaN values in technical indicators
        tech_indicators_array = np.array(tech_indicators[:14], dtype=np.float32)
        tech_indicators_array = np.nan_to_num(tech_indicators_array, nan=0.5, posinf=1.0, neginf=-1.0)
        obs['technical_indicators'] = tech_indicators_array
        
        # Get options chain summary
        # TODO: Implement real options chain fetching
        # For now, using dummy data - replace with:
        # options_chain = await self.data_collector.get_options_chain(symbol)
        options_data = np.random.randn(10, 8).astype(np.float32) * 0.1
        obs['options_chain'] = np.nan_to_num(options_data, nan=0.0)
        
        # Portfolio state
        portfolio_state = [
            self.current_capital,
            len(self.positions),
            self._calculate_portfolio_value(),
            self._calculate_total_pnl(),
            self.current_step
        ]
        obs['portfolio_state'] = np.nan_to_num(np.array(portfolio_state, dtype=np.float32), nan=0.0)
        
        # Greeks summary
        total_greeks = self._calculate_portfolio_greeks()
        greeks_summary = [
            total_greeks.get('delta', 0),
            total_greeks.get('gamma', 0),
            total_greeks.get('theta', 0),
            total_greeks.get('vega', 0),
            total_greeks.get('rho', 0)
        ]
        obs['greeks_summary'] = np.nan_to_num(np.array(greeks_summary, dtype=np.float32), nan=0.0)
        
        return obs
    
    def _execute_action(self, action: int) -> Dict:
        """Execute the given action"""
        trade_info = {
            'executed': False,
            'action': action,
            'details': None
        }
        
        if action == 0:  # Hold
            return trade_info
        
        # Log action for debugging (only log trades, not every attempt)
        # logger.info(f"Step {self.current_step}: Attempting action {action}, Positions: {len(self.positions)}/{self.max_positions}, Capital: ${self.current_capital:.2f}")
        
        # Determine action type
        if action <= 5:  # Buy actions
            position_type = 'long'
            if action <= 3:
                option_type = 'call'
                strike_offset = action - 1  # 0, 1, 2 for ATM, ITM, OTM
            else:
                option_type = 'put'
                strike_offset = action - 4  # 0, 1, 2 for ATM, ITM, OTM
        else:  # Sell actions
            position_type = 'short'
            if action <= 8:
                option_type = 'call'
                strike_offset = action - 6  # 0, 1, 2 for ATM, ITM, OTM
            else:
                option_type = 'put'
                strike_offset = action - 9  # 0, 1 for ATM, ITM
        
        # Check if we can open a new position
        if len(self.positions) >= self.max_positions:
            return trade_info
        
        # Get current market conditions
        symbol = self.symbols[0]
        current_price = self._get_current_price(symbol)
        
        # Get technical indicators for decision making
        tech_indicators = self._get_current_indicators(symbol)
        
        if current_price is None:
            logger.error(f"No price data for {symbol} at step {self.current_step}")
            return trade_info
        
        # logger.info(f"Current price for {symbol}: ${current_price:.2f}")
        
        # Calculate strike price
        if option_type == 'call':
            if strike_offset == 0:  # ATM
                strike = round(current_price)
            elif strike_offset == 1:  # ITM
                strike = round(current_price * 0.95)
            else:  # OTM
                strike = round(current_price * 1.05)
        else:  # put
            if strike_offset == 0:  # ATM
                strike = round(current_price)
            elif strike_offset == 1:  # ITM
                strike = round(current_price * 1.05)
            else:  # OTM
                strike = round(current_price * 0.95)
        
        # Create option contract
        contract = OptionContract(
            symbol=symbol,
            strike=strike,
            expiration=datetime.now() + timedelta(days=30),
            option_type=option_type,
            bid=2.0,  # Dummy values
            ask=2.1,
            last_price=2.05,
            volume=1000,
            open_interest=5000,
            implied_volatility=0.25
        )
        
        # Calculate position size (risk management)
        position_cost = contract.ask * 100  # Cost per contract
        max_contracts_by_capital = int(self.current_capital * 0.05 / position_cost) if position_cost > 0 else 0
        position_size = min(5, max(1, max_contracts_by_capital))
        
        # logger.info(f"Position sizing: Contract ask=${contract.ask}, Cost per contract=${position_cost}, Max by capital={max_contracts_by_capital}, Final size={position_size}")
        
        if position_size <= 0:
            logger.warning("Position size is 0, skipping trade")
            return trade_info
        
        # Execute trade
        entry_price = contract.ask if position_type == 'long' else contract.bid
        cost = position_size * entry_price * 100
        
        if position_type == 'long' and cost > self.current_capital:
            logger.warning(f"Insufficient capital: Need ${cost:.2f}, Have ${self.current_capital:.2f}")
            return trade_info
        
        # Update capital
        if position_type == 'long':
            self.current_capital -= cost
        else:
            self.current_capital += cost  # Receive premium for short positions
        
        # Create position
        position = OptionsPosition(
            contract=contract,
            quantity=position_size,
            entry_price=entry_price,
            entry_time=datetime.now(),
            position_type=position_type
        )
        
        self.positions.append(position)
        
        trade_info['executed'] = True
        trade_info['details'] = {
            'symbol': symbol,
            'option_type': option_type,
            'strike': strike,
            'position_type': position_type,
            'quantity': position_size,
            'entry_price': entry_price,
            'cost': cost
        }
        
        self.trade_history.append(trade_info)
        
        # logger.info(f"TRADE EXECUTED: {position_type} {position_size} {option_type} contracts, Strike=${strike}, Entry=${entry_price}, Cost=${cost:.2f}, Remaining Capital=${self.current_capital:.2f}")
        
        return trade_info
    
    def _update_positions(self):
        """Update positions and check for expirations"""
        current_time = datetime.now()
        positions_to_close = []
        
        for i, position in enumerate(self.positions):
            # Check if option expired
            if position.contract.expiration <= current_time:
                positions_to_close.append(i)
            else:
                # Update option prices (simplified - in reality would fetch from Alpaca)
                # For now, just add some random walk
                position.contract.bid *= (1 + np.random.normal(0, 0.02))
                position.contract.ask *= (1 + np.random.normal(0, 0.02))
                position.contract.last_price = (position.contract.bid + position.contract.ask) / 2
        
        # Close expired positions
        for i in reversed(positions_to_close):
            position = self.positions.pop(i)
            # Handle expiration based on moneyness
            # Simplified logic - in reality would check against current stock price
            final_value = position.current_value
            self.current_capital += final_value
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(position.current_value for position in self.positions)
        return self.current_capital + positions_value
    
    def _calculate_total_pnl(self) -> float:
        """Calculate total P&L"""
        return self._calculate_portfolio_value() - self.initial_capital
    
    def _calculate_portfolio_greeks(self) -> Dict[str, float]:
        """Calculate portfolio Greeks"""
        total_greeks = {
            'delta': 0,
            'gamma': 0,
            'theta': 0,
            'vega': 0,
            'rho': 0
        }
        
        for position in self.positions:
            multiplier = position.quantity if position.position_type == 'long' else -position.quantity
            total_greeks['delta'] += position.contract.delta * multiplier
            total_greeks['gamma'] += position.contract.gamma * multiplier
            total_greeks['theta'] += position.contract.theta * multiplier
            total_greeks['vega'] += position.contract.vega * multiplier
            total_greeks['rho'] += position.contract.rho * multiplier
        
        return total_greeks
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        if symbol not in self.price_history:
            logger.error(f"Symbol {symbol} not in price history")
            return None
            
        df = self.price_history[symbol]
        if df.empty:
            logger.error(f"Price history for {symbol} is empty")
            return None
            
        current_idx = self.start_index + self.current_step
        logger.debug(f"Getting price: start_index={self.start_index}, current_step={self.current_step}, current_idx={current_idx}, df_len={len(df)}")
        
        if current_idx >= len(df):
            logger.error(f"Index {current_idx} out of bounds for {symbol} (max: {len(df)-1})")
            return None
            
        return df.iloc[current_idx]['close']
    
    def _get_current_indicators(self, symbol: str) -> Dict[str, float]:
        """Get current technical indicators for symbol"""
        indicators = {}
        
        if symbol in self.technical_indicators:
            df = self.technical_indicators[symbol]
            current_idx = self.start_index + self.current_step
            
            if current_idx < len(df):
                row = df.iloc[current_idx]
                indicators = {
                    'RSI': row.get('RSI', 50),
                    'MACD': row.get('MACD', 0),
                    'MACD_signal': row.get('MACD_signal', 0),
                    'MACD_histogram': row.get('MACD_histogram', 0),
                    'CCI': row.get('CCI', 0),
                    'ADX': row.get('ADX', 0),
                    'Plus_DI': row.get('Plus_DI', 0),
                    'Minus_DI': row.get('Minus_DI', 0)
                }
        
        return indicators
    
    def _get_info(self) -> Dict:
        """Get additional info about the environment state"""
        return {
            'portfolio_value': self._calculate_portfolio_value(),
            'total_pnl': self._calculate_total_pnl(),
            'num_positions': len(self.positions),
            'current_step': self.current_step
        }


def train(args):
    """Main training function"""
    # Load configuration
    config = TradingConfig()
    
    # Verify Alpaca credentials are loaded
    if not config.alpaca_api_key or not config.alpaca_secret_key:
        logger.error("Alpaca API credentials not found! Please check your .env file.")
        logger.error("Expected environment variables: ALPACA_API_KEY, ALPACA_SECRET_KEY")
        return
    else:
        logger.info(f"Alpaca API key loaded: {config.alpaca_api_key[:4]}...{config.alpaca_api_key[-4:]}")
    
    # Set up logging
    log_dir = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment
    logger.info(f"Creating Gymnasium environment with symbols: {args.symbols}")
    env = AlpacaOptionsGymEnvironment(
        config=config,
        symbols=args.symbols,
        initial_capital=args.initial_capital,
        max_positions=args.max_positions,
        lookback_days=args.lookback_days,
        episode_length=args.episode_length,
        training_mode=True
    )
    
    # Verify data was loaded
    if not env.price_history:
        logger.error("No price history loaded! Check Alpaca API credentials.")
        return
    
    # Create CLSTM-PPO agent
    logger.info("Creating CLSTM-PPO agent...")
    
    # Log observation space dimensions for debugging
    total_obs_dim = 0
    for key, space in env.observation_space.spaces.items():
        obs_dim = np.prod(space.shape)
        logger.info(f"Observation '{key}': shape={space.shape}, dim={obs_dim}")
        total_obs_dim += obs_dim
    logger.info(f"Total observation dimension: {total_obs_dim}")
    
    agent = OptionsCLSTMPPOAgent(
        observation_space=env.observation_space,
        action_space=env.action_space.n,
        learning_rate_actor_critic=args.lr_actor_critic,
        learning_rate_clstm=args.lr_clstm,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        batch_size=args.batch_size,
        n_epochs=args.ppo_epochs,
        device=args.device
    )
    
    # Load checkpoint if resuming
    if args.resume_from:
        logger.info(f"Loading checkpoint from {args.resume_from}")
        agent.load(args.resume_from)
    
    # Check for NaN in model weights
    has_nan = False
    for name, param in agent.network.named_parameters():
        if torch.isnan(param).any():
            logger.error(f"NaN found in parameter: {name}")
            has_nan = True
    
    if has_nan:
        logger.error("Model contains NaN values! Consider reinitializing or using a different checkpoint.")
        return
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    episode_trade_stats = []  # Track trade statistics per episode
    best_reward = -np.inf
    
    # Training loop
    logger.info(f"Starting training for {args.num_episodes} episodes...")
    
    for episode in range(args.num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_start_value = env._calculate_portfolio_value()
        episode_trades = []  # Track trades in this episode
        
        # Debug first observation
        if episode == 0:
            logger.info("First observation shapes:")
            for key, value in obs.items():
                logger.info(f"  {key}: {value.shape}, min={np.min(value):.2f}, max={np.max(value):.2f}, has_nan={np.isnan(value).any()}")
        
        while True:
            # Get action from agent
            action, action_info = agent.act(obs, deterministic=False)
            
            # Force exploration in early episodes
            if episode < 50:
                # 50% random actions for first 50 episodes
                if np.random.random() < 0.5:
                    action = env.action_space.sample()
            elif episode < 100:
                # 30% random actions for episodes 50-100
                if np.random.random() < 0.3:
                    action = env.action_space.sample()
            
            # Log first few actions per episode
            if episode_length < 5:
                logger.info(f"Episode {episode}, Step {episode_length}: Action = {action}")
            
            # Step environment
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            
            # Track if a trade was executed
            if step_info.get('trade_info', {}).get('executed', False):
                episode_trades.append(step_info['trade_info'])
            
            # Store transition
            agent.store_transition(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                done=terminated or truncated,
                info=action_info
            )
            
            episode_reward += reward
            episode_length += 1
            
            obs = next_obs
            
            if terminated or truncated:
                break
            
            # Train agent every n steps
            if len(agent.buffer) >= args.train_freq:
                train_metrics = agent.train()
                
                if episode % args.log_interval == 0 and train_metrics:
                    logger.info(
                        f"Episode {episode}, Step {episode_length}: "
                        f"Policy Loss: {train_metrics.get('policy_loss', 0):.4f}, "
                        f"Value Loss: {train_metrics.get('value_loss', 0):.4f}, "
                        f"Entropy: {train_metrics.get('entropy', 0):.4f}"
                    )
        
        # Record episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Calculate episode trade statistics
        episode_end_value = env._calculate_portfolio_value()
        episode_return = (episode_end_value - episode_start_value) / episode_start_value
        episode_trade_stats.append({
            'trades': len(episode_trades),
            'return': episode_return,
            'final_value': episode_end_value
        })
        
        # Log episode summary
        if episode % args.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            # Count trades executed in this episode
            trades_executed = len([t for t in env.trade_history if t['executed']])
            logger.info(
                f"Episode {episode}/{args.num_episodes}: "
                f"Reward: {episode_reward:.4f}, "
                f"Avg Reward (100 eps): {avg_reward:.4f}, "
                f"Length: {episode_length}, "
                f"Portfolio Value: {info['portfolio_value']:.2f}, "
                f"PnL: {info['total_pnl']:.2f}, "
                f"Trades: {trades_executed}"
            )
        
        # Every 100 episodes, calculate and log detailed statistics
        if episode > 0 and episode % 100 == 0:
            # Calculate 100-episode average return
            if len(episode_rewards) >= 100:
                last_100_rewards = episode_rewards[-100:]
                avg_return = np.mean(last_100_rewards)
                
                # Calculate win rate and returns from last 100 episodes
                last_100_stats = episode_trade_stats[-100:]
                
                # Episode win rate (episodes with positive returns)
                winning_episodes = sum(1 for stat in last_100_stats if stat['return'] > 0)
                episode_win_rate = winning_episodes / 100 * 100
                
                # Average portfolio return
                avg_portfolio_return = np.mean([stat['return'] for stat in last_100_stats]) * 100
                
                # Total trades in last 100 episodes
                total_trades = sum(stat['trades'] for stat in last_100_stats)
                avg_trades_per_episode = total_trades / 100
                
                logger.info("="*60)
                logger.info(f"100-EPISODE SUMMARY (Episodes {episode-99} to {episode}):")
                logger.info(f"  Average Return: {avg_return:.4f}")
                logger.info(f"  Average Portfolio Return: {avg_portfolio_return:.2f}%")
                logger.info(f"  Episode Win Rate: {episode_win_rate:.1f}% ({winning_episodes}/100)")
                logger.info(f"  Total Trades: {total_trades} (avg {avg_trades_per_episode:.1f}/episode)")
                logger.info(f"  Best Episode Return: {max(last_100_rewards):.4f}")
                logger.info(f"  Worst Episode Return: {min(last_100_rewards):.4f}")
                logger.info(f"  Return Std Dev: {np.std(last_100_rewards):.4f}")
                logger.info("="*60)
        
        # Save checkpoint
        if episode % args.save_interval == 0:
            checkpoint_path = os.path.join(log_dir, f"checkpoint_episode_{episode}.pt")
            agent.save(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_model_path = os.path.join(log_dir, "best_model.pt")
            agent.save(best_model_path)
            logger.info(f"New best model saved with reward: {best_reward:.4f}")
    
    # Final save
    final_model_path = os.path.join(log_dir, "final_model.pt")
    agent.save(final_model_path)
    logger.info(f"Training completed. Final model saved to {final_model_path}")
    
    # Save training history
    history = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'config': vars(args)
    }
    
    with open(os.path.join(log_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train CLSTM-PPO agent for options trading')
    
    # Environment parameters
    parser.add_argument('--symbols', nargs='+', default=['SPY', 'QQQ', 'TSLA', 'AAPL', 'META', 'NVDA', 'AMD', 'PLTR'], help='Stock symbols to trade')
    parser.add_argument('--initial-capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--max-positions', type=int, default=10, help='Maximum number of positions')
    parser.add_argument('--lookback-days', type=int, default=30, help='Days of historical data to use')
    parser.add_argument('--episode-length', type=int, default=252, help='Length of each episode (trading days)')
    
    # Agent parameters
    parser.add_argument('--lr-actor-critic', type=float, default=1e-4, help='Learning rate for actor-critic')
    parser.add_argument('--lr-clstm', type=float, default=1e-3, help='Learning rate for CLSTM')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip-epsilon', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--value-coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--entropy-coef', type=float, default=0.1, help='Entropy coefficient')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--ppo-epochs', type=int, default=10, help='PPO epochs per update')
    
    # Training parameters
    parser.add_argument('--num-episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--train-freq', type=int, default=256, help='Train every n steps')
    parser.add_argument('--log-interval', type=int, default=10, help='Log every n episodes')
    parser.add_argument('--save-interval', type=int, default=100, help='Save checkpoint every n episodes')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    parser.add_argument('--resume-from', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Run training
    train(args)


if __name__ == "__main__":
    main()