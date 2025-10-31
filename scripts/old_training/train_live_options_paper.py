#!/usr/bin/env python3
"""
Advanced Live Paper Trading with Options Simulation
Uses real market data to simulate options trading while training PPO-CLSTM
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import argparse
import traceback
import asyncio
from dotenv import load_dotenv
import time
from collections import deque
import json
import threading
import queue
from typing import Dict, List, Tuple, Optional
from scipy.stats import norm
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
from alpaca_trade_api.stream import Stream
import multiprocessing as mp
from multiprocessing import Process, Queue as MPQueue, Value, Array
import torch.multiprocessing as torch_mp

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from options_clstm_ppo import OptionsCLSTMPPOAgent


class TechnicalIndicators:
    """Calculate technical indicators for trading decisions"""
    
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """
        Calculate MACD (Moving Average Convergence Divergence)
        Returns: (macd_line, signal_line, histogram)
        """
        if len(prices) < slow:
            return 0.0, 0.0, 0.0
        
        prices_array = np.array(prices)
        
        # Calculate exponential moving averages
        ema_fast = TechnicalIndicators._calculate_ema(prices_array, fast)
        ema_slow = TechnicalIndicators._calculate_ema(prices_array, slow)
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line (EMA of MACD)
        signal_line = TechnicalIndicators._calculate_ema(macd_line, signal)
        
        # MACD histogram
        histogram = macd_line - signal_line
        
        return float(macd_line[-1]), float(signal_line[-1]), float(histogram[-1])
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        """
        Calculate RSI (Relative Strength Index)
        Returns: RSI value (0-100)
        """
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        prices_array = np.array(prices)
        deltas = np.diff(prices_array)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0  # Maximum RSI
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    @staticmethod
    def calculate_cci(prices, highs=None, lows=None, period=20):
        """
        Calculate CCI (Commodity Channel Index)
        Returns: CCI value (typically -100 to +100)
        """
        if len(prices) < period:
            return 0.0
        
        # Use prices for high/low if not provided
        if highs is None:
            highs = prices
        if lows is None:
            lows = prices
        
        # Calculate typical price
        typical_price = (np.array(highs) + np.array(lows) + np.array(prices)) / 3
        
        # Moving average of typical price
        sma = np.mean(typical_price[-period:])
        
        # Mean deviation
        mean_dev = np.mean(np.abs(typical_price[-period:] - sma))
        
        if mean_dev == 0:
            return 0.0
        
        # CCI calculation
        cci = (typical_price[-1] - sma) / (0.015 * mean_dev)
        
        return float(cci)
    
    @staticmethod
    def calculate_adx(prices, highs=None, lows=None, period=14):
        """
        Calculate ADX (Average Directional Index)
        Returns: ADX value (0-100, trend strength)
        """
        if len(prices) < period * 2:
            return 0.0
        
        # Use prices for high/low if not provided
        if highs is None:
            highs = [p * 1.001 for p in prices]  # Approximate
        if lows is None:
            lows = [p * 0.999 for p in prices]  # Approximate
        
        highs = np.array(highs)
        lows = np.array(lows)
        closes = np.array(prices)
        
        # Calculate True Range
        high_low = highs - lows
        high_close = np.abs(highs[1:] - closes[:-1])
        low_close = np.abs(lows[1:] - closes[:-1])
        
        # Pad to maintain array size
        high_close = np.concatenate([[0], high_close])
        low_close = np.concatenate([[0], low_close])
        
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        
        # Calculate directional movements
        up_move = highs[1:] - highs[:-1]
        down_move = lows[:-1] - lows[1:]
        
        # Pad arrays
        up_move = np.concatenate([[0], up_move])
        down_move = np.concatenate([[0], down_move])
        
        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Calculate smoothed averages
        atr = TechnicalIndicators._calculate_ema(tr, period)
        # Avoid division by zero
        atr = np.where(atr == 0, 1e-8, atr)
        pos_di = 100 * TechnicalIndicators._calculate_ema(pos_dm, period) / atr
        neg_di = 100 * TechnicalIndicators._calculate_ema(neg_dm, period) / atr
        
        # Calculate DX
        di_diff = np.abs(pos_di - neg_di)
        di_sum = pos_di + neg_di
        dx = np.where(di_sum > 0, 100 * di_diff / di_sum, 0)
        
        # Calculate ADX
        adx = TechnicalIndicators._calculate_ema(dx, period)
        
        return float(adx[-1])
    
    @staticmethod
    def _calculate_ema(data, period):
        """Helper function to calculate Exponential Moving Average"""
        if len(data) < period:
            return np.array([np.mean(data[:i+1]) for i in range(len(data))])
        
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    @staticmethod
    def calculate_all_indicators(prices, high_prices=None, low_prices=None):
        """Calculate all technical indicators at once"""
        if len(prices) < 26:  # Minimum for MACD
            return {
                'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0,
                'rsi': 50.0, 'cci': 0.0, 'adx': 0.0
            }
        
        # Calculate indicators
        macd, signal, histogram = TechnicalIndicators.calculate_macd(prices)
        rsi = TechnicalIndicators.calculate_rsi(prices)
        cci = TechnicalIndicators.calculate_cci(prices, high_prices, low_prices)
        adx = TechnicalIndicators.calculate_adx(prices, high_prices, low_prices)
        
        return {
            'macd': macd,
            'macd_signal': signal,
            'macd_histogram': histogram,
            'rsi': rsi,
            'cci': cci,
            'adx': adx
        }

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/live_options_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BlackScholesOptionPricer:
    """Black-Scholes option pricing for simulated options"""
    
    @staticmethod
    def calculate_option_price(S, K, T, r, sigma, option_type='call'):
        """
        Calculate option price using Black-Scholes formula
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility
        """
        if T <= 0:
            # Option expired
            if option_type == 'call':
                return max(0, S - K)
            else:
                return max(0, K - S)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return price
    
    @staticmethod
    def calculate_greeks(S, K, T, r, sigma, option_type='call'):
        """Calculate option Greeks"""
        if T <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        
        # Gamma
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        if option_type == 'call':
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                    r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                    r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
        # Vega
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega
        }


class LiveOptionsEnvironment:
    """Live options trading environment with real-time data"""
    
    def __init__(self, symbols, initial_capital=100000, use_websocket=False, simulation_mode=None):
        self.symbols = symbols
        self.use_websocket = use_websocket
        
        # Initialize Alpaca API for market data and trading
        self.api = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            base_url=os.getenv('ALPACA_API_BASE_URL', 'https://paper-api.alpaca.markets')
        )
        
        # Check if market is open
        clock = self.api.get_clock()
        self.market_open = clock.is_open
        
        # Determine simulation mode
        if simulation_mode is None:
            # Auto-detect: use simulation if market is closed
            self.simulation_mode = not self.market_open
        else:
            self.simulation_mode = simulation_mode
        
        if self.simulation_mode:
            logger.info("Running in SIMULATION MODE - will use historical data for training")
            # In simulation mode, use initial capital parameter
            self.initial_capital = initial_capital
            self.capital = initial_capital
        else:
            # Get actual account info for live trading
            account = self.api.get_account()
            self.initial_capital = float(account.portfolio_value)
            self.capital = float(account.cash)
            logger.info(f"Loaded actual account - Portfolio Value: ${self.initial_capital:,.2f}, Cash: ${self.capital:,.2f}")
        
        # Position tracking
        self.options_positions = []  # List of option positions (simulated or real)
        self.stock_positions = {}    # Stock positions
        self.alpaca_orders = {}      # Track Alpaca order IDs
        
        # Market data
        self.current_prices = {symbol: 100.0 for symbol in symbols}
        self.price_history = {symbol: deque(maxlen=100) for symbol in symbols}
        self.volume_history = {symbol: deque(maxlen=100) for symbol in symbols}
        self.volatility = {symbol: 0.25 for symbol in symbols}  # Implied volatility
        self.intraday_highs = {symbol: 0.0 for symbol in symbols}
        self.intraday_lows = {symbol: float('inf') for symbol in symbols}
        self.vwap = {symbol: 0.0 for symbol in symbols}  # Volume weighted average price
        
        # Technical indicators
        self.technical_indicators = {symbol: {} for symbol in symbols}
        
        # Performance tracking
        self.trade_history = []
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0
        
        # Real-time data
        self.market_data_queue = queue.Queue()
        self.ws_thread = None
        self.data_thread = None
        self.is_running = False
        
        # Risk-free rate (3-month Treasury)
        self.risk_free_rate = 0.05
        
        # Historical data for simulation mode
        self.historical_data = {symbol: None for symbol in symbols}
        self.historical_index = 0
        self.historical_start_date = None
        self.simulation_speed = 300  # 1 second = 5 minutes (300 seconds)
        self.last_update_time = time.time()
        
        logger.info(f"Initialized live options environment for symbols: {symbols}")
        
        # Sync initial positions
        self._sync_positions()
    
    def start(self):
        """Start real-time data collection"""
        self.is_running = True
        
        # Only start data collection threads in live mode
        if not self.simulation_mode:
            # Start REST API data collection thread
            self.data_thread = threading.Thread(target=self._collect_market_data)
            self.data_thread.daemon = True
            self.data_thread.start()
            
            # Start WebSocket for real-time updates only if enabled
            if self.use_websocket:
                self._start_websocket()
            
            logger.info("Started live data collection")
        else:
            logger.info("Simulation mode - using historical data, no live collection")
    
    def stop(self):
        """Stop data collection"""
        self.is_running = False
        
        if self.data_thread:
            self.data_thread.join()
        
        if hasattr(self, 'stream'):
            try:
                self.stream.stop()
            except:
                pass
        
        logger.info("Stopped data collection")
    
    def _start_websocket(self):
        """Start Alpaca Stream for real-time data"""
        try:
            # Initialize Alpaca Stream
            self.stream = Stream(
                os.getenv('ALPACA_API_KEY'),
                os.getenv('ALPACA_SECRET_KEY'),
                base_url=os.getenv('ALPACA_API_BASE_URL', 'https://paper-api.alpaca.markets'),
                data_feed='iex'  # Use IEX feed which has less restrictions
            )
            
            # Define handlers
            async def on_trade(trade):
                """Handle trade updates"""
                symbol = trade.symbol
                if symbol in self.symbols:
                    self.current_prices[symbol] = float(trade.price)
                    self.market_data_queue.put({
                        'symbol': symbol,
                        'price': float(trade.price),
                        'size': float(trade.size),
                        'timestamp': datetime.now(),
                        'type': 'trade'
                    })
            
            async def on_quote(quote):
                """Handle quote updates"""
                symbol = quote.symbol
                if symbol in self.symbols:
                    # Update bid/ask info
                    self.market_data_queue.put({
                        'symbol': symbol,
                        'bid': float(quote.bid_price),
                        'ask': float(quote.ask_price),
                        'bid_size': float(quote.bid_size),
                        'ask_size': float(quote.ask_size),
                        'timestamp': datetime.now(),
                        'type': 'quote'
                    })
            
            # Subscribe to only trades for the first symbol to avoid subscription limits
            # We'll rely on REST API for quotes and other symbols
            if self.symbols:
                self.stream.subscribe_trades(on_trade, self.symbols[0])
                logger.info(f"Alpaca Stream subscribed to trades for: {self.symbols[0]}")
            
            # Run stream in separate thread
            self.ws_thread = threading.Thread(target=self._run_stream)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            logger.info(f"Alpaca Stream started with limited subscription")
            
        except Exception as e:
            logger.error(f"Failed to start Alpaca Stream: {e}")
    
    def _run_stream(self):
        """Run the Alpaca stream"""
        try:
            self.stream.run()
        except Exception as e:
            logger.error(f"Stream error: {e}")
    
    def _collect_market_data(self):
        """Collect market data using Alpaca REST API"""
        symbol_index = 0
        while self.is_running:
            try:
                # Process one symbol at a time to avoid rate limits
                symbol = self.symbols[symbol_index % len(self.symbols)]
                
                try:
                    if self.simulation_mode or not self.market_open:
                        # In simulation mode or after hours, use historical data
                        # Get the most recent bar data
                        end_time = datetime.now()
                        start_time = end_time - timedelta(days=1)
                        
                        bars = self.api.get_bars(
                            symbol,
                            TimeFrame.Minute,
                            start=start_time.strftime('%Y-%m-%d'),
                            end=end_time.strftime('%Y-%m-%d'),
                            feed='iex',  # Use IEX feed to avoid subscription issues
                            limit=1
                        ).df
                        
                        if not bars.empty:
                            current_price = float(bars['close'].iloc[-1])
                            self.current_prices[symbol] = current_price
                            
                            # Simulate quote from historical data
                            quote = type('Quote', (), {
                                'bid_price': current_price * 0.9995,  # 0.05% spread
                                'ask_price': current_price * 1.0005,
                                'bid_size': 100,
                                'ask_size': 100
                            })()
                            trade = type('Trade', (), {
                                'price': current_price,
                                'size': 100
                            })()
                        else:
                            # If no bars, skip this symbol
                            logger.warning(f"No historical data available for {symbol}")
                            continue
                    else:
                        # During market hours, get real-time data
                        quote = self.api.get_latest_quote(symbol, feed='iex')
                        trade = self.api.get_latest_trade(symbol, feed='iex')
                        
                        # Update current price
                        current_price = float(trade.price)
                        self.current_prices[symbol] = current_price
                    
                    # Update price history
                    self.price_history[symbol].append(current_price)
                    
                    # Update intraday highs/lows
                    self.intraday_highs[symbol] = max(self.intraday_highs[symbol], current_price)
                    self.intraday_lows[symbol] = min(self.intraday_lows[symbol], current_price)
                    
                    # Only update volatility once per 10 minutes for each symbol
                    if symbol_index % (len(self.symbols) * 300) < len(self.symbols):
                        # Get historical bars for volatility calculation
                        end_time = datetime.now()
                        start_time = end_time - timedelta(days=30)
                        
                        bars = self.api.get_bars(
                            symbol,
                            TimeFrame.Day,
                            start=start_time.strftime('%Y-%m-%d'),
                            end=end_time.strftime('%Y-%m-%d'),
                            feed='iex',  # Explicitly use IEX feed
                            limit=30
                        ).df
                        
                        if len(bars) > 10:
                            # Calculate volatility from daily returns
                            bars['returns'] = bars['close'].pct_change()
                            returns = bars['returns'].dropna()
                            self.volatility[symbol] = returns.std() * np.sqrt(252)
                    
                    # Update volume history
                    self.volume_history[symbol].append(float(trade.size))
                    
                    # Only update VWAP once per 5 minutes for each symbol
                    if symbol_index % (len(self.symbols) * 150) < len(self.symbols):
                        try:
                            today_start = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
                            end_time = datetime.now()
                            intraday_bars = self.api.get_bars(
                                symbol,
                                TimeFrame.Minute,
                                start=today_start.strftime('%Y-%m-%dT%H:%M:%S-04:00'),  # RFC3339 format with timezone
                                end=end_time.strftime('%Y-%m-%dT%H:%M:%S-04:00'),
                                feed='iex',  # Use IEX feed
                                limit=100  # Reduced from 390 to save API calls
                            ).df
                            
                            if len(intraday_bars) > 0:
                                # Calculate VWAP
                                intraday_bars['typical_price'] = (intraday_bars['high'] + intraday_bars['low'] + intraday_bars['close']) / 3
                                intraday_bars['pv'] = intraday_bars['typical_price'] * intraday_bars['volume']
                                cumulative_pv = intraday_bars['pv'].cumsum()
                                cumulative_volume = intraday_bars['volume'].cumsum()
                                self.vwap[symbol] = cumulative_pv.iloc[-1] / cumulative_volume.iloc[-1] if cumulative_volume.iloc[-1] > 0 else current_price
                        except:
                            self.vwap[symbol] = current_price
                    
                    # Update technical indicators
                    if len(self.price_history[symbol]) >= 26:
                        self._update_technical_indicators(symbol)
                    
                    # Add to market data queue
                    self.market_data_queue.put({
                        'symbol': symbol,
                        'price': current_price,
                        'bid': float(quote.bid_price),
                        'ask': float(quote.ask_price),
                        'bid_size': float(quote.bid_size),
                        'ask_size': float(quote.ask_size),
                        'volume': float(trade.size),
                        'timestamp': datetime.now()
                    })
                    
                except Exception as e:
                    logger.error(f"Error getting data for {symbol}: {e}")
                
                # Move to next symbol
                symbol_index += 1
                
                # Process queued market data
                while not self.market_data_queue.empty():
                    try:
                        data = self.market_data_queue.get_nowait()
                        symbol = data['symbol']
                        if symbol in self.price_history:
                            self.price_history[symbol].append(data['price'])
                    except queue.Empty:
                        break
                
                # Rate limit: Process one symbol every 2 seconds
                # With 5 symbols, each symbol gets updated every 10 seconds
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error in data collection: {e}")
                time.sleep(10)
    
    def _update_technical_indicators(self, symbol):
        """Update technical indicators for symbol"""
        prices = list(self.price_history[symbol])
        
        self.technical_indicators[symbol] = TechnicalIndicators.calculate_all_indicators(
            prices=prices,
            high_prices=prices,
            low_prices=prices
        )
    
    def _sync_positions(self):
        """Sync positions with actual Alpaca account"""
        if self.simulation_mode:
            # In simulation mode, positions are tracked internally
            return
            
        try:
            # Get current positions from Alpaca
            positions = self.api.list_positions()
            
            # Clear and update stock positions
            self.stock_positions = {}
            for position in positions:
                symbol = position.symbol
                if symbol in self.symbols:
                    self.stock_positions[symbol] = int(position.qty)
                    logger.info(f"Synced position: {symbol} = {position.qty} shares")
            
            # Update account info
            account = self.api.get_account()
            self.capital = float(account.cash)
            
        except Exception as e:
            logger.error(f"Error syncing positions: {e}")
    
    def _execute_stock_trade(self, symbol, quantity, side='buy'):
        """Execute actual stock trade through Alpaca"""
        try:
            # Place market order
            order = self.api.submit_order(
                symbol=symbol,
                qty=abs(quantity),
                side=side,
                type='market',
                time_in_force='day'
            )
            
            logger.info(f"Placed {side} order for {abs(quantity)} shares of {symbol}")
            
            # Store order ID
            self.alpaca_orders[order.id] = {
                'symbol': symbol,
                'qty': quantity,
                'side': side,
                'status': order.status,
                'submitted_at': order.submitted_at
            }
            
            # Wait for order to fill (with timeout)
            timeout = 30  # seconds
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                order = self.api.get_order(order.id)
                if order.status == 'filled':
                    filled_price = float(order.filled_avg_price)
                    logger.info(f"Order filled: {side} {abs(quantity)} shares of {symbol} at ${filled_price:.2f}")
                    
                    # Update positions
                    if symbol not in self.stock_positions:
                        self.stock_positions[symbol] = 0
                    
                    if side == 'buy':
                        self.stock_positions[symbol] += abs(quantity)
                        self.capital -= filled_price * abs(quantity)
                    else:  # sell
                        self.stock_positions[symbol] -= abs(quantity)
                        self.capital += filled_price * abs(quantity)
                    
                    return filled_price
                elif order.status in ['cancelled', 'rejected']:
                    logger.error(f"Order {order.status}: {order.id}")
                    return None
                
                time.sleep(1)
            
            logger.warning(f"Order timeout: {order.id}")
            return None
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None
    
    def _format_option_symbol(self, underlying, expiry, option_type, strike):
        """Format option symbol for Alpaca (OCC format)
        Example: AAPL240119C00100000 for AAPL Jan 19, 2024 Call at $100"""
        # Format expiry as YYMMDD
        expiry_str = expiry.strftime('%y%m%d')
        
        # Format strike price (multiply by 1000, pad to 8 digits)
        strike_str = str(int(strike * 1000)).zfill(8)
        
        # Option type: C for call, P for put
        type_char = 'C' if option_type == 'call' else 'P'
        
        return f"{underlying}{expiry_str}{type_char}{strike_str}"
    
    def _get_options_chain(self, symbol, expiry_date=None):
        """Get options chain from Alpaca"""
        # In simulation mode, don't make API calls - use synthetic options
        if self.simulation_mode:
            return []
        
        try:
            # If no expiry specified, get options expiring in next 15-45 days
            params = {
                'underlying_symbols': symbol,  # Note: 'underlying_symbols' not 'underlying_symbol'
                'limit': 100,
                'status': 'active'
            }
            
            if expiry_date:
                params['expiration_date'] = expiry_date
            else:
                # Get options expiring between 1-30 days from now
                params['expiration_date_gte'] = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
                params['expiration_date_lte'] = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
            
            # Use the proper Alpaca API method to get options contracts
            response = self.api._request('GET', '/options/contracts', params)
            
            options = []
            if isinstance(response, dict):
                if 'contracts' in response:
                    contracts = response['contracts']
                elif 'option_contracts' in response:
                    contracts = response['option_contracts']
                else:
                    contracts = []
            elif isinstance(response, list):
                contracts = response  # Sometimes the response is directly the list
            else:
                contracts = []
            
            for contract in contracts:
                try:
                    options.append({
                        'symbol': contract['symbol'],
                        'underlying': contract['underlying_symbol'],
                        'strike': float(contract['strike_price']),
                        'expiry': datetime.strptime(contract['expiration_date'], '%Y-%m-%d'),
                        'type': 'call' if contract['type'] == 'call' else 'put',
                        'size': int(contract.get('size', 100)),
                        'style': contract.get('style', 'american')
                    })
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping malformed contract: {e}")
                    continue
            
            logger.debug(f"Found {len(options)} options for {symbol}")
            return options
            
        except Exception as e:
            logger.error(f"Error fetching options chain: {e}")
            return []
    
    def _execute_option_trade(self, option_symbol, quantity, side='buy'):
        """Execute option trade (real or simulated)"""
        try:
            if self.simulation_mode:
                # In simulation mode, simulate the trade execution
                # Use Black-Scholes for realistic pricing
                
                # Parse option symbol to get details
                # Format: AAPL250119C00100000 or PLTR250711P00131000
                # Find where the date starts (after ticker symbol)
                date_idx = 0
                for i, char in enumerate(option_symbol):
                    if char.isdigit():
                        date_idx = i
                        break
                
                underlying = option_symbol[:date_idx]
                
                # Parse expiry date (YYMMDD format)
                date_str = option_symbol[date_idx:date_idx+6]
                year = 2000 + int(date_str[:2])
                month = int(date_str[2:4])
                day = int(date_str[4:6])
                expiry = datetime(year, month, day)
                
                # Parse option type and strike
                type_char = option_symbol[date_idx+6]  # 'C' or 'P'
                option_type = 'call' if type_char == 'C' else 'put'
                strike_str = option_symbol[date_idx+7:]
                strike = float(strike_str) / 1000
                
                # Get current underlying price and volatility
                current_price = self.current_prices.get(underlying, 100)
                volatility = self.volatility.get(underlying, 0.25)
                
                # Calculate time to expiry
                time_to_expiry = max(0.001, (expiry - datetime.now()).days / 365)
                
                # Calculate option price using Black-Scholes
                fill_price = BlackScholesOptionPricer.calculate_option_price(
                    current_price, strike, time_to_expiry,
                    self.risk_free_rate, volatility, option_type
                )
                
                # Add realistic bid-ask spread
                spread_pct = 0.02 + 0.03 * abs(current_price - strike) / current_price
                if side == 'buy':
                    # Pay the ask price
                    fill_price = fill_price * (1 + spread_pct)
                else:
                    # Receive the bid price
                    fill_price = fill_price * (1 - spread_pct)
                
                # Ensure minimum price
                fill_price = max(0.01, fill_price)
                
                # logger.info(f"SIMULATED {side} order for {abs(quantity)} contracts of {option_symbol} at ${fill_price:.2f}")
                return fill_price
            
            else:
                # Real trading mode
                # Check if market is open
                clock = self.api.get_clock()
                if not clock.is_open:
                    logger.warning(f"Market is closed. Cannot execute option trade.")
                    return None
                
                # Place limit order for option
                # Use a reasonable limit price based on expected fill
                limit_price = 10.0  # In production, calculate from option chain data
                
                order = self.api.submit_order(
                    symbol=option_symbol,
                    qty=abs(quantity),
                    side=side,
                    type='limit',
                    limit_price=limit_price,
                    time_in_force='day'
                )
            
            logger.info(f"Placed {side} order for {abs(quantity)} contracts of {option_symbol}")
            
            # Store order ID
            self.alpaca_orders[order.id] = {
                'symbol': option_symbol,
                'qty': quantity,
                'side': side,
                'status': order.status,
                'submitted_at': order.submitted_at,
                'is_option': True
            }
            
            # Wait for order to fill
            timeout = 30
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                order = self.api.get_order(order.id)
                if order.status == 'filled':
                    filled_price = float(order.filled_avg_price)
                    logger.info(f"Option order filled: {side} {abs(quantity)} contracts at ${filled_price:.2f}")
                    
                    # Update capital (options are quoted per share, 1 contract = 100 shares)
                    if side == 'buy':
                        self.capital -= filled_price * abs(quantity) * 100
                    else:
                        self.capital += filled_price * abs(quantity) * 100
                    
                    return filled_price
                elif order.status in ['cancelled', 'rejected']:
                    logger.error(f"Option order {order.status}: {order.id}")
                    return None
                
                time.sleep(1)
            
            logger.warning(f"Option order timeout: {order.id}")
            return None
            
        except Exception as e:
            logger.error(f"Error executing option trade: {e}")
            return None
    
    def get_observation(self):
        """Get current market observation"""
        obs = {
            'price_history': np.zeros((len(self.symbols), 50, 5)),
            'technical_indicators': np.zeros((20,)),
            'options_chain': np.zeros((10, 20, 8)),
            'portfolio_state': np.zeros((5,)),
            'greeks_summary': np.zeros((5,))
        }
        
        # Fill price history
        for i, symbol in enumerate(self.symbols):
            prices = list(self.price_history[symbol])[-50:] if self.price_history[symbol] else [100]
            for j, price in enumerate(prices):
                obs['price_history'][i, j, :] = [price, price, price, price, 1000]
        
        # Fill technical indicators
        indicators_list = []
        for symbol in self.symbols:
            if symbol in self.technical_indicators and self.technical_indicators[symbol]:
                ind = self.technical_indicators[symbol]
                current_price = self.current_prices[symbol]
                
                # Calculate price position relative to VWAP
                vwap_deviation = (current_price - self.vwap[symbol]) / current_price if self.vwap[symbol] > 0 else 0
                
                # Calculate intraday range position
                intraday_range = self.intraday_highs[symbol] - self.intraday_lows[symbol]
                range_position = (current_price - self.intraday_lows[symbol]) / intraday_range if intraday_range > 0 else 0.5
                
                indicators_list.append([
                    ind.get('macd_histogram', 0) / 10,
                    (ind.get('rsi', 50) - 50) / 50,
                    ind.get('cci', 0) / 200,
                    ind.get('adx', 0) / 50,
                    self.volatility.get(symbol, 0.25) * 4,
                    vwap_deviation * 10,  # VWAP deviation
                    range_position,  # Position in daily range (0-1)
                    intraday_range / current_price * 100  # Intraday range as % of price
                ])
        
        if indicators_list:
            avg_indicators = np.mean(indicators_list, axis=0)
            obs['technical_indicators'][:len(avg_indicators)] = avg_indicators
        
        # Use simulated options chain for speed
        for i, symbol in enumerate(self.symbols[:10]):
            current_price = self.current_prices[symbol]
            volatility = self.volatility[symbol]
            
            # In simulation mode, always use simulated options for speed
            options_chain = []
            
            if options_chain and not self.simulation_mode:
                # Use real options data
                strikes_processed = 0
                for option in sorted(options_chain, key=lambda x: x['strike'])[:20]:
                    if strikes_processed >= 20:
                        break
                    
                    # Get real-time quote for this option (if available)
                    # Note: Option quotes might not be available without OPRA subscription
                    # Fallback to Black-Scholes pricing for now
                    try:
                        # Skip trying to get option quotes for now
                        raise Exception("Skip option quotes")
                    except:
                        # Fallback to Black-Scholes pricing
                        T = (option['expiry'] - datetime.now()).days / 365
                        option_price = BlackScholesOptionPricer.calculate_option_price(
                            current_price, option['strike'], T, self.risk_free_rate, volatility, option['type']
                        )
                        spread_pct = 0.02 + 0.03 * abs(current_price - option['strike']) / current_price
                        bid = option_price * (1 - spread_pct)
                        ask = option_price * (1 + spread_pct)
                        mid = option_price
                    
                    # Calculate Greeks
                    T = max(0.001, (option['expiry'] - datetime.now()).days / 365)
                    greeks = BlackScholesOptionPricer.calculate_greeks(
                        current_price, option['strike'], T, self.risk_free_rate, volatility, option['type']
                    )
                    
                    obs['options_chain'][i, strikes_processed, :] = [
                        option['strike'], bid, ask, mid,
                        1000,  # Volume (placeholder)
                        greeks['delta'],
                        volatility,
                        current_price / option['strike']  # Moneyness
                    ]
                    strikes_processed += 1
            else:
                # Fallback to simulated options chain
                strikes = np.linspace(current_price * 0.9, current_price * 1.1, 20)
                
                for j, strike in enumerate(strikes):
                    T = 30 / 365
                    
                    # Calculate option prices
                    call_price = BlackScholesOptionPricer.calculate_option_price(
                        current_price, strike, T, self.risk_free_rate, volatility, 'call'
                    )
                    put_price = BlackScholesOptionPricer.calculate_option_price(
                        current_price, strike, T, self.risk_free_rate, volatility, 'put'
                    )
                    
                    # Simulate bid-ask spread
                    spread_pct = 0.02 + 0.03 * abs(current_price - strike) / current_price
                    
                    if j < 10:  # Calls
                        bid = call_price * (1 - spread_pct)
                        ask = call_price * (1 + spread_pct)
                        greeks = BlackScholesOptionPricer.calculate_greeks(
                            current_price, strike, T, self.risk_free_rate, volatility, 'call'
                        )
                    else:  # Puts
                        bid = put_price * (1 - spread_pct)
                        ask = put_price * (1 + spread_pct)
                        greeks = BlackScholesOptionPricer.calculate_greeks(
                            current_price, strike, T, self.risk_free_rate, volatility, 'put'
                        )
                    
                    obs['options_chain'][i, j, :] = [
                        strike, bid, ask, (bid + ask) / 2,
                        1000,  # Volume
                        greeks['delta'],
                        volatility,
                        current_price / strike  # Moneyness
                    ]
        
        # Portfolio state
        portfolio_value = self._calculate_portfolio_value()
        obs['portfolio_state'] = np.array([
            self.capital / self.initial_capital,
            len(self.options_positions) / 10,
            portfolio_value / self.initial_capital,
            self.total_pnl / self.initial_capital,
            self.winning_trades / max(1, self.winning_trades + self.losing_trades)
        ])
        
        # Greeks summary
        total_delta, total_gamma, total_theta, total_vega = self._calculate_portfolio_greeks()
        obs['greeks_summary'] = np.array([
            total_delta / 10,
            total_gamma / 10,
            total_theta / 100,
            total_vega / 100,
            0
        ])
        
        return obs
    
    def step(self, action):
        """Execute action and return reward"""
        action_mapping = {
            0: 'hold',
            1: 'buy_call',
            2: 'buy_put',
            3: 'sell_call',
            4: 'sell_put',
            5: 'close_all_positions',
            6: 'delta_hedge'
        }
        
        action_name = action_mapping.get(action, 'hold')
        reward = 0
        
        # Execute action
        if action_name == 'hold':
            # Just update positions
            self._update_positions()
            # Small negative reward to encourage action
            reward = -0.01
            
        elif action_name in ['buy_call', 'buy_put'] and len(self.options_positions) < 10:
            # Select best option to buy
            option = self._select_best_option(action_name)
            if option:
                # Use the Alpaca symbol if available, otherwise format our own
                if 'alpaca_symbol' in option:
                    option_symbol = option['alpaca_symbol']
                else:
                    option_symbol = self._format_option_symbol(
                        option['symbol'],
                        option['expiry'],
                        option['type'],
                        option['strike']
                    )
                
                # Execute real option trade
                filled_price = self._execute_option_trade(option_symbol, 1, 'buy')
                
                if filled_price:
                    option['actual_order'] = True
                    option['filled_price'] = filled_price
                    option['entry_price'] = filled_price  # Make sure entry_price is set to filled_price
                    option['alpaca_symbol'] = option_symbol
                    self.options_positions.append(option)
                    # No immediate reward for opening position
                    reward = 0
                    logger.info(f"Bought {option['type']} option: {option_symbol}, "
                              f"Strike=${option['strike']:.2f}, Filled at ${filled_price:.2f}, "
                              f"Delta={option['greeks']['delta']:.3f}")
        
        elif action_name in ['sell_call', 'sell_put'] and len(self.options_positions) < 10:
            # Sell options (covered)
            option = self._select_best_option(action_name, selling=True)
            if option:
                # Use the Alpaca symbol if available, otherwise format our own
                if 'alpaca_symbol' in option:
                    option_symbol = option['alpaca_symbol']
                else:
                    option_symbol = self._format_option_symbol(
                        option['symbol'],
                        option['expiry'],
                        option['type'],
                        option['strike']
                    )
                
                # Execute real option trade (sell)
                filled_price = self._execute_option_trade(option_symbol, 1, 'sell')
                
                if filled_price:
                    option['quantity'] = -option['quantity']  # Negative for sold options
                    option['actual_order'] = True
                    option['filled_price'] = filled_price
                    option['entry_price'] = filled_price  # Make sure entry_price is set to filled_price
                    option['alpaca_symbol'] = option_symbol
                    self.options_positions.append(option)
                    # No immediate reward for opening position
                    reward = 0
                    logger.info(f"Sold {option['type']} option: {option_symbol}, "
                              f"Strike=${option['strike']:.2f}, Premium received ${filled_price:.2f}")
        
        elif action_name == 'close_all_positions':
            # Close all positions
            total_pnl = 0
            positions_closed = 0
            
            # Close all actual options positions
            for position in list(self.options_positions):
                pnl = self._close_position(position)
                total_pnl += pnl
                positions_closed += 1
                
                if 'alpaca_symbol' in position:
                    # Close real option position
                    side = 'sell' if position['quantity'] > 0 else 'buy'
                    filled_price = self._execute_option_trade(
                        position['alpaca_symbol'],
                        abs(position['quantity']),
                        side
                    )
                    
                    if filled_price:
                        # Calculate return percentage
                        if position['quantity'] > 0:  # Long position
                            trade_return = (filled_price - position.get('filled_price', position['entry_price'])) / position.get('filled_price', position['entry_price']) * 100
                        else:  # Short position
                            trade_return = (position.get('filled_price', position['entry_price']) - filled_price) / position.get('filled_price', position['entry_price']) * 100
                        
                        # logger.info(f"CLOSED option position: {position['alpaca_symbol']}, "
                        #           f"Entry=${position.get('filled_price', position['entry_price']):.2f}, "
                        #           f"Exit=${filled_price:.2f}, Return={trade_return:.1f}%, "
                        #           f"PnL=${pnl:.2f}, Portfolio=${self.capital:.2f}")
            
            # Close all actual stock positions
            for symbol, shares in list(self.stock_positions.items()):
                if shares != 0:
                    filled_price = self._execute_stock_trade(
                        symbol,
                        abs(shares),
                        'sell' if shares > 0 else 'buy'
                    )
                    if filled_price:
                        logger.info(f"Closed stock position: {symbol} {shares} shares at ${filled_price:.2f}")
            
            self.options_positions = []
            self._sync_positions()  # Sync with Alpaca
            
            # Reward based on actual PnL from closed positions
            if positions_closed > 0:
                # Scale based on portfolio size
                reward = (total_pnl / self.initial_capital) * 50
            else:
                reward = -0.1  # Small penalty for closing with no positions
            
        elif action_name == 'delta_hedge':
            # Delta neutral hedging
            total_delta, _, _, _ = self._calculate_portfolio_greeks()
            if abs(total_delta) > 0.1:
                # Hedge with stock
                hedge_shares = -int(total_delta * 100)
                if hedge_shares != 0:
                    symbol = self.symbols[0]  # Use first symbol
                    cost = abs(hedge_shares) * self.current_prices[symbol]
                    if cost < self.capital * 0.2:
                        # Execute real hedge trade
                        filled_price = self._execute_stock_trade(
                            symbol,
                            abs(hedge_shares),
                            'buy' if hedge_shares > 0 else 'sell'
                        )
                        
                        if filled_price:
                            # Small reward for risk management
                            reward = 0.02
                            logger.info(f"Delta hedge executed: {'Bought' if hedge_shares > 0 else 'Sold'} "
                                      f"{abs(hedge_shares)} shares of {symbol} at ${filled_price:.2f}")
        
        # Update prices from historical data (in simulation mode)
        if self.simulation_mode:
            end_of_data = self._update_from_historical_data()
            if end_of_data:
                done = True
                reward -= 1  # Small penalty for running out of data
        
        # Update all positions and calculate PnL from closed positions
        closed_pnl = self._update_positions()
        
        # Add reward from positions that were closed during update (stop loss, take profit, expiry)
        if closed_pnl != 0:
            # Scale reward based on portfolio size for proportional impact
            reward += (closed_pnl / self.initial_capital) * 100
        
        # Calculate current portfolio metrics
        portfolio_value = self._calculate_portfolio_value()
        portfolio_return = (portfolio_value - self.initial_capital) / self.initial_capital
        
        # CRITICAL: Add reward based on portfolio return
        # This ensures negative returns lead to negative rewards
        reward += portfolio_return * 50  # Increased scale for stronger impact
        
        # Additional penalty if portfolio is losing money
        if portfolio_return < 0:
            reward -= abs(portfolio_return) * 30  # Stronger penalty for losses
        
        # Win rate bonus (only if we have enough closed trades AND positive returns)
        if self.winning_trades + self.losing_trades >= 10 and portfolio_return > 0:
            win_rate = self.winning_trades / (self.winning_trades + self.losing_trades)
            if win_rate >= 0.6:
                reward += 0.5
            elif win_rate >= 0.5:
                reward += 0.2
            elif win_rate < 0.3:
                reward -= 0.5
        
        # Risk penalty for excessive positions
        if len(self.options_positions) > 7:
            reward -= 0.1
        
        # Penalty for not making trades when portfolio is flat
        if len(self.options_positions) == 0 and abs(portfolio_return) < 0.001:
            reward -= 0.1
        
        # Get next observation
        obs = self.get_observation()
        
        # Episode done conditions
        done = False
        if portfolio_value < self.initial_capital * 0.8:  # 20% loss
            done = True
            reward -= 20  # Severe penalty for large losses
        elif portfolio_value > self.initial_capital * 1.5:  # 50% gain
            done = True
            reward += 50  # Big reward for achieving 50% returns
        elif portfolio_value > self.initial_capital * 1.2:  # 20% gain
            done = True
            reward += 20  # Good reward for solid returns
        
        info = {
            'portfolio_value': portfolio_value,
            'positions': len(self.options_positions),
            'total_pnl': self.total_pnl,
            'win_rate': self.winning_trades / max(1, self.winning_trades + self.losing_trades)
        }
        
        return obs, reward, done, info
    
    def _select_best_option(self, action_type, selling=False):
        """Select best option based on current market conditions"""
        best_score = -float('inf')
        best_option = None
        
        # Determine option type
        option_type = 'call' if 'call' in action_type else 'put'
        
        # Analyze each symbol
        for symbol in self.symbols:
            current_price = self.current_prices[symbol]
            volatility = self.volatility[symbol]
            
            # Skip if no technical indicators
            if symbol not in self.technical_indicators:
                continue
            
            indicators = self.technical_indicators[symbol]
            
            # Score based on market conditions
            base_score = 0
            
            if option_type == 'call':
                # Bullish indicators for calls
                if indicators.get('rsi', 50) < 40:
                    base_score += 1
                if indicators.get('macd_histogram', 0) > 0:
                    base_score += 1
            else:
                # Bearish indicators for puts
                if indicators.get('rsi', 50) > 60:
                    base_score += 1
                if indicators.get('macd_histogram', 0) < 0:
                    base_score += 1
            
            # High volatility is good for buying options
            if not selling and volatility > 0.3:
                base_score += 1
            # Low volatility is good for selling options
            elif selling and volatility < 0.2:
                base_score += 1
            
            # Try to get real options chain first
            options_chain = self._get_options_chain(symbol)
            
            if options_chain:
                # Filter for desired option type and reasonable strikes
                filtered_options = [
                    opt for opt in options_chain 
                    if opt['type'] == option_type and 
                    current_price * 0.95 <= opt['strike'] <= current_price * 1.05
                ]
                
                for option in filtered_options:
                    # Get real-time quote if available
                    # Note: Option quotes might not be available without OPRA subscription
                    # Fallback to Black-Scholes pricing for now
                    try:
                        # Skip trying to get option quotes for now
                        raise Exception("Skip option quotes")
                    except:
                        # Fallback to Black-Scholes
                        T = max(0.001, (option['expiry'] - datetime.now()).days / 365)
                        option_price = BlackScholesOptionPricer.calculate_option_price(
                            current_price, option['strike'], T, self.risk_free_rate, volatility, option_type
                        )
                    
                    # Calculate Greeks
                    T = max(0.001, (option['expiry'] - datetime.now()).days / 365)
                    greeks = BlackScholesOptionPricer.calculate_greeks(
                        current_price, option['strike'], T, self.risk_free_rate, volatility, option_type
                    )
                    
                    # Score based on Greeks
                    score = base_score
                    
                    # Prefer high delta for directional plays
                    if not selling:
                        score += abs(greeks['delta']) * 2
                    # Prefer low delta for selling
                    else:
                        score += (1 - abs(greeks['delta'])) * 2
                    
                    # Prefer positive theta when selling
                    if selling:
                        score += max(0, -greeks['theta']) * 10
                    
                    # Adjust for price
                    if option_price > 0.5 and option_price < 5:
                        score += 1  # Reasonable price range
                    
                    if score > best_score:
                        best_score = score
                        best_option = {
                            'symbol': symbol,
                            'type': option_type,
                            'strike': option['strike'],
                            'expiry': option['expiry'],
                            'entry_price': option_price,
                            'cost': option_price * 100,  # 1 contract = 100 shares
                            'quantity': 1,
                            'greeks': greeks,
                            'entry_time': datetime.now(),
                            'alpaca_symbol': option['symbol']  # Store actual option symbol
                        }
            else:
                # Fallback to simulated options
                if option_type == 'call':
                    strikes = [current_price * (1 + i * 0.01) for i in range(1, 6)]  # OTM calls
                else:
                    strikes = [current_price * (1 - i * 0.01) for i in range(1, 6)]  # OTM puts
                
                for strike in strikes:
                    T = 30 / 365  # 30-day options
                    
                    # Calculate option price
                    option_price = BlackScholesOptionPricer.calculate_option_price(
                        current_price, strike, T, self.risk_free_rate, volatility, option_type
                    )
                    
                    # Calculate Greeks
                    greeks = BlackScholesOptionPricer.calculate_greeks(
                        current_price, strike, T, self.risk_free_rate, volatility, option_type
                    )
                    
                    # Score based on Greeks
                    score = base_score
                    
                    # Prefer high delta for directional plays
                    if not selling:
                        score += abs(greeks['delta']) * 2
                    # Prefer low delta for selling
                    else:
                        score += (1 - abs(greeks['delta'])) * 2
                    
                    # Prefer positive theta when selling
                    if selling:
                        score += max(0, -greeks['theta']) * 10
                    
                    # Adjust for price
                    if option_price > 0.5 and option_price < 5:
                        score += 1  # Reasonable price range
                    
                    if score > best_score:
                        best_score = score
                        best_option = {
                            'symbol': symbol,
                            'type': option_type,
                            'strike': strike,
                            'expiry': datetime.now() + timedelta(days=30),
                            'entry_price': option_price,
                            'cost': option_price * 100,  # 1 contract = 100 shares
                            'quantity': 1,
                            'greeks': greeks,
                            'entry_time': datetime.now()
                        }
        
        return best_option
    
    def _update_positions(self):
        """Update all option positions and return PnL from closed positions"""
        positions_to_close = []
        total_closed_pnl = 0
        
        for i, position in enumerate(self.options_positions):
            # Calculate current value
            symbol = position['symbol']
            current_price = self.current_prices[symbol]
            volatility = self.volatility[symbol]
            
            # Time to expiry
            time_to_expiry = (position['expiry'] - datetime.now()).days / 365
            
            if time_to_expiry <= 0:
                # Option expired
                positions_to_close.append(i)
                continue
            
            # Calculate current option value
            current_value = BlackScholesOptionPricer.calculate_option_price(
                current_price, position['strike'], time_to_expiry,
                self.risk_free_rate, volatility, position['type']
            )
            
            # Update Greeks
            position['greeks'] = BlackScholesOptionPricer.calculate_greeks(
                current_price, position['strike'], time_to_expiry,
                self.risk_free_rate, volatility, position['type']
            )
            
            # Calculate P&L
            if position['quantity'] > 0:  # Long position
                pnl = (current_value - position['entry_price']) * 100 * position['quantity']
            else:  # Short position
                pnl = (position['entry_price'] - current_value) * 100 * abs(position['quantity'])
            
            position['current_value'] = current_value
            position['pnl'] = pnl
            position['pnl_pct'] = pnl / abs(position['cost'])
            
            # Exit conditions
            if position['quantity'] > 0:  # Long positions
                # Take profit at 30% (more realistic)
                if position['pnl_pct'] >= 0.3:
                    positions_to_close.append(i)
                    self.winning_trades += 1
                # Stop loss at -30%
                elif position['pnl_pct'] <= -0.3:
                    positions_to_close.append(i)
                    self.losing_trades += 1
                # Close if near expiry (< 3 days)
                elif time_to_expiry < 3/365:
                    positions_to_close.append(i)
                    if pnl > 0:
                        self.winning_trades += 1
                    else:
                        self.losing_trades += 1
            else:  # Short positions
                # Take profit at 30% of premium
                if position['pnl_pct'] >= 0.3:
                    positions_to_close.append(i)
                    self.winning_trades += 1
                # Stop loss at -50% (half premium loss)
                elif position['pnl_pct'] <= -0.5:
                    positions_to_close.append(i)
                    self.losing_trades += 1
        
        # Close positions
        for i in reversed(positions_to_close):
            position = self.options_positions.pop(i)
            self.capital += position['current_value'] * 100 * position['quantity']
            self.total_pnl += position['pnl']
            total_closed_pnl += position['pnl']
            
            # Calculate return percentage for this trade
            if position['quantity'] > 0:  # Long position
                trade_return = (position['current_value'] - position['entry_price']) / position['entry_price'] * 100
            else:  # Short position
                trade_return = (position['entry_price'] - position['current_value']) / position['entry_price'] * 100
            
            # Determine if it's a win or loss
            trade_result = "WIN" if position['pnl'] > 0 else "LOSS" if position['pnl'] < 0 else "EVEN"
            
            # Log with return percentage and portfolio balance
            logger.info(f"CLOSED {position['type']} option ({trade_result}): {position.get('alpaca_symbol', 'N/A')}, "
                      f"Strike=${position['strike']:.2f}, Entry=${position['entry_price']:.2f}, "
                      f"Exit=${position['current_value']:.2f}, Return={trade_return:.1f}%, "
                      f"PnL=${position['pnl']:.2f}, Portfolio=${self.capital + position['current_value'] * 100 * position['quantity']:.2f}")
        
        return total_closed_pnl
    
    def _close_position(self, position):
        """Close a specific position"""
        symbol = position['symbol']
        current_price = self.current_prices[symbol]
        volatility = self.volatility[symbol]
        time_to_expiry = max(0, (position['expiry'] - datetime.now()).days / 365)
        
        # Calculate current value
        current_value = BlackScholesOptionPricer.calculate_option_price(
            current_price, position['strike'], time_to_expiry,
            self.risk_free_rate, volatility, position['type']
        )
        
        # Calculate P&L
        if position['quantity'] > 0:
            pnl = (current_value - position['entry_price']) * 100 * position['quantity']
        else:
            pnl = (position['entry_price'] - current_value) * 100 * abs(position['quantity'])
        
        # Update capital
        self.capital += current_value * 100 * position['quantity']
        self.total_pnl += pnl
        
        # Update win/loss count
        if pnl > 0:
            self.winning_trades += 1
        elif pnl < 0:
            self.losing_trades += 1
        # Don't count break-even trades
        
        return pnl
    
    def _calculate_portfolio_value(self):
        """Calculate total portfolio value"""
        if self.simulation_mode:
            # In simulation mode, calculate from tracked positions
            options_value = 0
            for position in self.options_positions:
                symbol = position['symbol']
                current_price = self.current_prices[symbol]
                volatility = self.volatility[symbol]
                time_to_expiry = max(0, (position['expiry'] - datetime.now()).days / 365)
                
                current_value = BlackScholesOptionPricer.calculate_option_price(
                    current_price, position['strike'], time_to_expiry,
                    self.risk_free_rate, volatility, position['type']
                )
                
                options_value += current_value * 100 * position['quantity']
            
            # Stock value
            stock_value = 0
            for symbol, shares in self.stock_positions.items():
                stock_value += shares * self.current_prices.get(symbol, 0)
            
            portfolio_value = self.capital + options_value + stock_value
            logger.debug(f"SIMULATED Portfolio value: ${portfolio_value:.2f}")
            return portfolio_value
        else:
            # Real trading mode - get from Alpaca
            try:
                account = self.api.get_account()
                portfolio_value = float(account.portfolio_value)
                
                # Also calculate simulated options value for tracking
                options_value = 0
                for position in self.options_positions:
                    symbol = position['symbol']
                    current_price = self.current_prices[symbol]
                    volatility = self.volatility[symbol]
                    time_to_expiry = max(0, (position['expiry'] - datetime.now()).days / 365)
                    
                    current_value = BlackScholesOptionPricer.calculate_option_price(
                        current_price, position['strike'], time_to_expiry,
                        self.risk_free_rate, volatility, position['type']
                    )
                    
                    options_value += current_value * 100 * position['quantity']
                
                logger.debug(f"Portfolio value: ${portfolio_value:.2f} (Options tracking: ${options_value:.2f})")
                return portfolio_value
                
            except Exception as e:
                logger.error(f"Error getting portfolio value: {e}")
                # Fallback to calculated value
                stock_value = 0
                for symbol, shares in self.stock_positions.items():
                    stock_value += shares * self.current_prices[symbol]
                
                return self.capital + stock_value
    
    def _calculate_portfolio_greeks(self):
        """Calculate total portfolio Greeks"""
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0
        
        for position in self.options_positions:
            greeks = position.get('greeks', {})
            quantity = position['quantity']
            
            total_delta += greeks.get('delta', 0) * quantity * 100
            total_gamma += greeks.get('gamma', 0) * quantity * 100
            total_theta += greeks.get('theta', 0) * quantity * 100
            total_vega += greeks.get('vega', 0) * quantity * 100
        
        # Add stock deltas
        for symbol, shares in self.stock_positions.items():
            total_delta += shares
        
        return total_delta, total_gamma, total_theta, total_vega
    
    def _load_random_historical_period(self):
        """Load a random historical period for training"""
        if not self.simulation_mode:
            return
            
        # Pick a random starting date from the past year
        days_back = np.random.randint(30, 365)  # Random period from 30-365 days ago
        end_date = datetime.now() - timedelta(days=np.random.randint(1, 30))  # End 1-30 days ago
        start_date = end_date - timedelta(days=days_back)
        
        logger.info(f"Loading historical data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Load historical data for each symbol
        for symbol in self.symbols:
            try:
                # Get minute bars for the period
                bars = self.api.get_bars(
                    symbol,
                    TimeFrame.Minute,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    feed='iex',
                    limit=10000
                ).df
                
                if not bars.empty:
                    self.historical_data[symbol] = bars
                    logger.info(f"Loaded {len(bars)} bars for {symbol}")
                else:
                    logger.warning(f"No historical data available for {symbol}")
                    # Generate synthetic data as fallback
                    self.historical_data[symbol] = self._generate_synthetic_data(symbol, start_date, end_date)
                    
            except Exception as e:
                logger.error(f"Error loading historical data for {symbol}: {e}")
                # Generate synthetic data as fallback
                self.historical_data[symbol] = self._generate_synthetic_data(symbol, start_date, end_date)
        
        # Reset historical index
        self.historical_index = 0
        self.historical_start_date = start_date
    
    def _generate_synthetic_data(self, symbol, start_date, end_date):
        """Generate synthetic price data for testing"""
        # Generate minute-level data
        date_range = pd.date_range(start=start_date, end=end_date, freq='1min')
        # Filter for market hours only (9:30 AM - 4:00 PM EST)
        date_range = [d for d in date_range if 9 <= d.hour < 16 or (d.hour == 9 and d.minute >= 30)]
        
        # Start with a base price
        base_price = 100.0 if symbol not in ['SPY', 'QQQ'] else 400.0
        
        # Generate random walk with drift
        returns = np.random.normal(0.0001, 0.001, len(date_range))  # Small positive drift
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Add some intraday patterns
        for i, d in enumerate(date_range):
            # Morning volatility
            if d.hour == 9 and d.minute < 45:
                prices[i] *= np.random.uniform(0.995, 1.005)
            # Lunch time calm
            elif 12 <= d.hour <= 13:
                prices[i] *= np.random.uniform(0.999, 1.001)
            # End of day activity
            elif d.hour == 15 and d.minute > 30:
                prices[i] *= np.random.uniform(0.995, 1.005)
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': prices * np.random.uniform(0.999, 1.001, len(prices)),
            'high': prices * np.random.uniform(1.001, 1.005, len(prices)),
            'low': prices * np.random.uniform(0.995, 0.999, len(prices)),
            'close': prices,
            'volume': np.random.randint(10000, 1000000, len(prices))
        }, index=date_range)
        
        return df
    
    def _update_from_historical_data(self):
        """Update prices from historical data with time acceleration"""
        if not self.simulation_mode or not any(v is not None and len(v) > 0 for v in self.historical_data.values()):
            return
            
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        
        # Calculate how many 5-minute periods have passed (1 second = 5 minutes)
        # But advance at least 1 period per call to ensure progress
        periods_to_advance = max(1, int(elapsed * self.simulation_speed / 60))  # At least 1 period
        
        if periods_to_advance > 0:
            self.last_update_time = current_time
            
            # Advance through historical data
            for _ in range(periods_to_advance):
                self.historical_index += 1
                
                # Update prices for each symbol
                for symbol in self.symbols:
                    if self.historical_data[symbol] is not None and len(self.historical_data[symbol]) > self.historical_index:
                        bar = self.historical_data[symbol].iloc[self.historical_index]
                        
                        # Update current price
                        self.current_prices[symbol] = float(bar['close'])
                        
                        # Update price history
                        self.price_history[symbol].append(float(bar['close']))
                        
                        # Update volume history
                        self.volume_history[symbol].append(float(bar['volume']))
                        
                        # Update intraday highs/lows
                        self.intraday_highs[symbol] = max(self.intraday_highs[symbol], float(bar['high']))
                        self.intraday_lows[symbol] = min(self.intraday_lows[symbol], float(bar['low']))
                        
                        # Update technical indicators if we have enough data
                        if len(self.price_history[symbol]) >= 26:
                            self._update_technical_indicators(symbol)
                
                # Check if we've reached the end of historical data
                if all(self.historical_data[s] is None or self.historical_index >= len(self.historical_data[s]) - 1 
                       for s in self.symbols):
                    logger.info("Reached end of historical data, episode will end")
                    return True  # Signal end of episode
        
        return False
    
    def reset(self):
        """Reset environment"""
        if self.simulation_mode:
            # In simulation mode, reset to initial capital
            self.capital = self.initial_capital
            self.options_positions = []
            self.stock_positions = {}
            
            # Load a new random historical period
            self._load_random_historical_period()
            
            # Initialize prices from first historical data point
            for symbol in self.symbols:
                if self.historical_data[symbol] is not None and len(self.historical_data[symbol]) > 0:
                    first_bar = self.historical_data[symbol].iloc[0]
                    self.current_prices[symbol] = float(first_bar['close'])
                    self.price_history[symbol].clear()
                    self.price_history[symbol].append(float(first_bar['close']))
                    self.intraday_highs[symbol] = float(first_bar['high'])
                    self.intraday_lows[symbol] = float(first_bar['low'])
                    
            self.last_update_time = time.time()
        else:
            # Close all actual stock positions
            try:
                positions = self.api.list_positions()
                for position in positions:
                    if position.symbol in self.symbols:
                        self.api.submit_order(
                            symbol=position.symbol,
                            qty=abs(int(position.qty)),
                            side='sell' if int(position.qty) > 0 else 'buy',
                            type='market',
                            time_in_force='day'
                        )
                        logger.info(f"Closing position: {position.symbol} {position.qty} shares")
                
                # Wait for orders to fill
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error closing positions: {e}")
            
            # Clear simulated positions
            self.options_positions = []
            
            # Sync with actual account
            self._sync_positions()
            account = self.api.get_account()
            self.capital = float(account.cash)
        
        # Reset stats
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0
        
        # Reset intraday stats
        self._reset_intraday_stats()
        
        return self.get_observation()
    
    def _reset_intraday_stats(self):
        """Reset intraday statistics (call at market open)"""
        for symbol in self.symbols:
            current_price = self.current_prices.get(symbol, 100)
            self.intraday_highs[symbol] = current_price
            self.intraday_lows[symbol] = current_price
            self.vwap[symbol] = current_price


def episode_worker(worker_id, gpu_id, args, shared_queue, episode_counter, total_episodes, stop_signal):
    """Worker process for collecting episodes in parallel"""
    # Set GPU for this worker
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    torch.cuda.set_device(0)  # Since we only see one GPU per process
    
    # Create logger for this worker
    worker_logger = logging.getLogger(f'worker_{worker_id}')
    worker_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(f'[Worker {worker_id}/GPU {gpu_id}] %(asctime)s - %(message)s'))
    worker_logger.addHandler(handler)
    
    worker_logger.info(f"Worker {worker_id} started on GPU {gpu_id}")
    
    try:
        # Create environment for this worker
        env = LiveOptionsEnvironment(
            symbols=args.symbols.split(','),
            initial_capital=args.initial_capital,
            use_websocket=False,
            simulation_mode=True  # Always simulation for parallel workers
        )
        env.start()
        
        # Create a local agent for action selection (inference only)
        obs = env.get_observation()
        observation_space = {
            'price_history': type('', (), {'shape': obs['price_history'].shape})(),
            'technical_indicators': type('', (), {'shape': obs['technical_indicators'].shape})(),
            'options_chain': type('', (), {'shape': obs['options_chain'].shape})(),
            'portfolio_state': type('', (), {'shape': obs['portfolio_state'].shape})(),
            'greeks_summary': type('', (), {'shape': obs['greeks_summary'].shape})()
        }
        
        # Import agent class
        from options_clstm_ppo import OptionsCLSTMPPOAgent
        
        agent = OptionsCLSTMPPOAgent(
            observation_space=observation_space,
            action_space=7,
            learning_rate_actor_critic=args.lr_actor_critic,
            learning_rate_clstm=args.lr_clstm,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            batch_size=args.batch_size,
            n_epochs=args.ppo_epochs,
            device=f'cuda:0'  # Local GPU 0 (which is actually gpu_id)
        )
        
        # Main worker loop
        while not stop_signal.value:
            # Get next episode number
            with episode_counter.get_lock():
                if episode_counter.value >= total_episodes:
                    break
                current_episode = episode_counter.value
                episode_counter.value += 1
            
            worker_logger.info(f"Starting episode {current_episode + 1}/{total_episodes}")
            
            # Run episode
            obs = env.reset()
            episode_data = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'values': [],
                'log_probs': [],
                'dones': [],
                'episode_reward': 0,
                'portfolio_return': 0,
                'win_rate': 0,
                'total_trades': 0
            }
            
            for step in range(args.max_steps_per_episode):
                # Get action from agent
                action, action_info = agent.act(obs, deterministic=False)
                
                # Store experience
                episode_data['observations'].append(obs)
                episode_data['actions'].append(action)
                episode_data['values'].append(action_info['value'])
                episode_data['log_probs'].append(action_info['log_prob'])
                
                # Execute action
                next_obs, reward, done, info = env.step(action)
                
                episode_data['rewards'].append(reward)
                episode_data['dones'].append(done)
                episode_data['episode_reward'] += reward
                
                obs = next_obs
                
                if done:
                    break
            
            # Calculate final metrics
            final_value = env._calculate_portfolio_value()
            episode_data['portfolio_return'] = (final_value - args.initial_capital) / args.initial_capital
            episode_data['win_rate'] = info['win_rate']
            episode_data['total_trades'] = env.winning_trades + env.losing_trades
            
            worker_logger.info(f"Episode {current_episode + 1} complete: "
                             f"Return={episode_data['portfolio_return']:.1%}, "
                             f"Win Rate={episode_data['win_rate']:.1%}")
            
            # Send episode data to main process
            shared_queue.put({
                'worker_id': worker_id,
                'episode': current_episode,
                'data': episode_data
            })
        
    except Exception as e:
        worker_logger.error(f"Worker {worker_id} error: {e}")
        traceback.print_exc()
    finally:
        if 'env' in locals():
            env.stop()
        worker_logger.info(f"Worker {worker_id} stopped")


def episode_worker(worker_id, gpu_id, args, shared_queue, episode_counter, total_episodes, stop_signal):
    """Worker process for collecting episodes in parallel"""
    try:
        # Set CUDA device for this worker
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        torch.cuda.set_device(0)  # After setting CUDA_VISIBLE_DEVICES, use device 0
        
        # Create worker logger
        worker_logger = logging.getLogger(f'worker_{worker_id}')
        
        # Create environment for this worker
        env = LiveOptionsEnvironment(
            symbols=args.symbols.split(','),
            initial_capital=args.initial_capital,
            use_websocket=False,
            simulation_mode=True  # Always simulation for workers
        )
        env.start()
        
        # Get observation shape
        obs = env.get_observation()
        observation_space = {
            'price_history': type('', (), {'shape': obs['price_history'].shape})(),
            'technical_indicators': type('', (), {'shape': obs['technical_indicators'].shape})(),
            'options_chain': type('', (), {'shape': obs['options_chain'].shape})(),
            'portfolio_state': type('', (), {'shape': obs['portfolio_state'].shape})(),
            'greeks_summary': type('', (), {'shape': obs['greeks_summary'].shape})()
        }
        
        # Create agent for this worker
        device = torch.device('cuda:0')  # Local device 0 after CUDA_VISIBLE_DEVICES
        agent = OptionsCLSTMPPOAgent(
            observation_space=observation_space,
            action_space=7,
            learning_rate_actor_critic=args.lr_actor_critic,
            learning_rate_clstm=args.lr_clstm,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            batch_size=args.batch_size,
            n_epochs=args.ppo_epochs
        )
        
        worker_logger.info(f"Worker {worker_id} started on GPU {gpu_id}")
        
        while not stop_signal.value:
            # Get episode number
            with episode_counter.get_lock():
                if episode_counter.value >= total_episodes:
                    break
                episode_num = episode_counter.value
                episode_counter.value += 1
            
            # Collect episode
            obs = env.reset()
            episode_data = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'dones': [],
                'action_infos': [],
                'episode_reward': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_trades': 0
            }
            
            for step in range(args.max_steps_per_episode):
                # Get action
                action, action_info = agent.act(obs, deterministic=False)
                
                # Execute action
                next_obs, reward, done, info = env.step(action)
                
                # Store data
                episode_data['observations'].append(obs)
                episode_data['actions'].append(action)
                episode_data['rewards'].append(reward)
                episode_data['dones'].append(done)
                episode_data['action_infos'].append(action_info)
                episode_data['episode_reward'] += reward
                
                obs = next_obs
                
                if done:
                    break
            
            # Get final trade stats
            episode_data['winning_trades'] = env.winning_trades
            episode_data['losing_trades'] = env.losing_trades
            episode_data['total_trades'] = env.winning_trades + env.losing_trades
            
            # Send episode data to main process
            shared_queue.put({
                'worker_id': worker_id,
                'episode_num': episode_num,
                'data': episode_data
            })
            
            worker_logger.info(f"Worker {worker_id} completed episode {episode_num}, reward: {episode_data['episode_reward']:.2f}")
        
        env.stop()
        worker_logger.info(f"Worker {worker_id} finished")
        
    except Exception as e:
        logger.error(f"Worker {worker_id} error: {e}")
        traceback.print_exc()


def save_checkpoint(agent, episode, episode_rewards, win_rates, best_win_rate, final=False):
    """Save checkpoint with full state"""
    if final:
        checkpoint_path = "checkpoints/live_options/final_model.pt"
    else:
        checkpoint_path = f"checkpoints/live_options/checkpoint_ep{episode}.pt"
    
    checkpoint = {
        'episode': episode,
        'best_win_rate': best_win_rate,
        'episode_rewards': episode_rewards,
        'win_rates': win_rates,
        'final': final
    }
    
    # Handle DataParallel
    if hasattr(agent.network, 'module'):
        checkpoint['network_state_dict'] = agent.network.module.state_dict()
    else:
        checkpoint['network_state_dict'] = agent.network.state_dict()
    
    # Save optimizer states
    checkpoint['clstm_optimizer_state_dict'] = agent.clstm_optimizer.state_dict()
    checkpoint['ppo_optimizer_state_dict'] = agent.ppo_optimizer.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"💾 Saved checkpoint at {checkpoint_path}")


def load_checkpoint_if_exists(agent, args):
    """Load checkpoint and return starting episode"""
    start_episode = 0
    
    # Find latest checkpoint
    checkpoint_dir = "checkpoints/live_options"
    latest_checkpoint = None
    
    if os.path.exists(checkpoint_dir) and args.resume and not args.no_resume:
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_ep") and f.endswith(".pt")]
        
        if checkpoint_files:
            # Sort by episode number
            checkpoint_episodes = []
            for f in checkpoint_files:
                try:
                    ep_num = int(f.replace("checkpoint_ep", "").replace(".pt", ""))
                    checkpoint_episodes.append((ep_num, f))
                except:
                    pass
            
            if checkpoint_episodes:
                checkpoint_episodes.sort(key=lambda x: x[0], reverse=True)
                latest_ep, latest_file = checkpoint_episodes[0]
                latest_checkpoint = os.path.join(checkpoint_dir, latest_file)
                start_episode = latest_ep
    
    # Load checkpoint if found
    if latest_checkpoint and os.path.exists(latest_checkpoint):
        try:
            checkpoint = torch.load(latest_checkpoint, weights_only=False)
            
            # Check for dimension mismatch
            if 'network_state_dict' in checkpoint:
                old_input_dim = checkpoint['network_state_dict'].get(
                    'clstm_encoder.input_projection.weight', 
                    torch.zeros(1, 1)
                ).shape[1]
                
                current_input_dim = agent.network.clstm_encoder.input_projection.weight.shape[1]
                
                if old_input_dim != current_input_dim:
                    logger.warning(f"⚠️ Checkpoint dimension mismatch: old={old_input_dim}, current={current_input_dim}")
                    logger.warning("This checkpoint is incompatible with the current observation space.")
                    logger.warning("Starting fresh training instead.")
                    start_episode = 0
                else:
                    # Load model state
                    if hasattr(agent.network, 'module'):
                        agent.network.module.load_state_dict(checkpoint['network_state_dict'])
                    else:
                        agent.network.load_state_dict(checkpoint['network_state_dict'])
                    
                    # Load optimizer states
                    if 'clstm_optimizer_state_dict' in checkpoint:
                        agent.clstm_optimizer.load_state_dict(checkpoint['clstm_optimizer_state_dict'])
                    if 'ppo_optimizer_state_dict' in checkpoint:
                        agent.ppo_optimizer.load_state_dict(checkpoint['ppo_optimizer_state_dict'])
                    
                    logger.info(f"✅ Loaded checkpoint from episode {start_episode}")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            logger.error("Starting fresh training instead.")
            start_episode = 0
    
    return start_episode


def train_parallel(args):
    """Train using parallel episode collection on multiple GPUs"""
    # Set up multiprocessing
    torch_mp.set_start_method('spawn', force=True)
    
    # Create dummy environment to get observation space
    env = LiveOptionsEnvironment(
        symbols=args.symbols.split(','),
        initial_capital=args.initial_capital,
        use_websocket=False,
        simulation_mode=True
    )
    env.start()
    obs = env.get_observation()
    observation_space = {
        'price_history': type('', (), {'shape': obs['price_history'].shape})(),
        'technical_indicators': type('', (), {'shape': obs['technical_indicators'].shape})(),
        'options_chain': type('', (), {'shape': obs['options_chain'].shape})(),
        'portfolio_state': type('', (), {'shape': obs['portfolio_state'].shape})(),
        'greeks_summary': type('', (), {'shape': obs['greeks_summary'].shape})()
    }
    env.stop()
    
    # Create main agent for training
    device = torch.device('cuda:0')  # Main model on GPU 0
    agent = OptionsCLSTMPPOAgent(
        observation_space=observation_space,
        action_space=7,
        learning_rate_actor_critic=args.lr_actor_critic,
        learning_rate_clstm=args.lr_clstm,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        batch_size=args.batch_size * torch.cuda.device_count(),  # Scale batch size
        n_epochs=args.ppo_epochs
    )
    
    # Load checkpoint if needed
    start_episode = load_checkpoint_if_exists(agent, args)
    
    # Create worker processes
    num_workers = torch.cuda.device_count()
    episode_queue = MPQueue(maxsize=num_workers * 2)
    episode_counter = Value('i', start_episode)
    stop_signal = Value('i', 0)
    
    workers = []
    for i in range(num_workers):
        gpu_id = i
        p = Process(
            target=episode_worker,
            args=(i, gpu_id, args, episode_queue, episode_counter, args.episodes, stop_signal)
        )
        p.start()
        workers.append(p)
    
    logger.info(f"Started {num_workers} worker processes")
    
    # Training metrics
    episode_rewards = []
    win_rates = []
    best_win_rate = 0
    
    # Collect episodes and train
    episodes_collected = 0
    
    try:
        while episodes_collected < args.episodes - start_episode:
            # Collect episodes from workers
            try:
                result = episode_queue.get(timeout=60)  # 1 minute timeout
                episode_data = result['data']
                
                # Add to training buffer
                for i in range(len(episode_data['observations'])):
                    next_obs = episode_data['observations'][i+1] if i+1 < len(episode_data['observations']) else episode_data['observations'][i]
                    
                    agent.store_transition(
                        observation=episode_data['observations'][i],
                        action=episode_data['actions'][i],
                        reward=episode_data['rewards'][i],
                        next_observation=next_obs,
                        done=episode_data['dones'][i],
                        info=episode_data['action_infos'][i]
                    )
                
                # Track metrics
                episode_rewards.append(episode_data['episode_reward'])
                if episode_data['total_trades'] > 0:
                    win_rate = episode_data['winning_trades'] / episode_data['total_trades']
                    win_rates.append(win_rate)
                    if win_rate > best_win_rate:
                        best_win_rate = win_rate
                
                episodes_collected += 1
                logger.info(f"Collected episode {result['episode_num']} from worker {result['worker_id']}, reward: {episode_data['episode_reward']:.2f}")
                
                # Train when we have enough data
                if len(agent.buffer) >= agent.batch_size * 2:
                    logger.info(f"Training on {len(agent.buffer)} transitions")
                    metrics = agent.train()
                    
                    if metrics:
                        if 'clstm_loss' in metrics:
                            logger.info(f"CLSTM Loss: {metrics['clstm_loss']:.4f}")
                        if 'policy_loss' in metrics:
                            logger.info(f"Policy Loss: {metrics['policy_loss']:.4f}")
                        if 'value_loss' in metrics:
                            logger.info(f"Value Loss: {metrics['value_loss']:.4f}")
                        if 'total_loss' in metrics:
                            logger.info(f"Total Loss: {metrics['total_loss']:.4f}")
                
                # Save checkpoint periodically
                if episodes_collected % args.save_interval == 0:
                    save_checkpoint(agent, start_episode + episodes_collected, episode_rewards, win_rates, best_win_rate)
                
            except Exception as e:
                if 'Empty' in str(type(e).__name__):
                    logger.warning("No episodes received from workers for 60 seconds")
                    continue
                else:
                    logger.error(f"Error processing episode: {e}")
                    traceback.print_exc()
        
    except KeyboardInterrupt:
        logger.info("Training interrupted")
    finally:
        # Stop workers
        stop_signal.value = 1
        for p in workers:
            p.join(timeout=30)
        
        # Save final model
        save_checkpoint(agent, start_episode + episodes_collected, episode_rewards, win_rates, best_win_rate, final=True)
        logger.info("Parallel training complete")


def train_live_options(args):
    """Main training loop for live options trading"""
    logger.info("Starting live options paper trading training")
    
    # Check if parallel processing is requested and available
    use_parallel = torch.cuda.device_count() > 1 and not args.no_parallel
    
    if use_parallel:
        # Parallel episode collection
        logger.info(f"Starting parallel episode collection on {torch.cuda.device_count()} GPUs")
        train_parallel(args)
        return
    
    # Initialize environment
    env = LiveOptionsEnvironment(
        symbols=args.symbols.split(','),
        initial_capital=args.initial_capital,
        use_websocket=False,  # Disable websocket to avoid subscription limits
        simulation_mode=args.simulation_mode if hasattr(args, 'simulation_mode') else None
    )
    
    # Start live data
    env.start()
    
    # Wait for initial data only in live mode
    if not env.simulation_mode:
        logger.info("Waiting for market data initialization...")
        time.sleep(20)
    else:
        logger.info("Simulation mode - no wait needed")
    
    # Initialize agent
    obs = env.get_observation()
    observation_space = {
        'price_history': type('', (), {'shape': obs['price_history'].shape})(),
        'technical_indicators': type('', (), {'shape': obs['technical_indicators'].shape})(),
        'options_chain': type('', (), {'shape': obs['options_chain'].shape})(),
        'portfolio_state': type('', (), {'shape': obs['portfolio_state'].shape})(),
        'greeks_summary': type('', (), {'shape': obs['greeks_summary'].shape})()
    }
    
    # Check available GPUs
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"Found {gpu_count} GPU(s) available")
        for i in range(gpu_count):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    agent = OptionsCLSTMPPOAgent(
        observation_space=observation_space,
        action_space=7,  # hold, buy_call, buy_put, sell_call, sell_put, close_all, delta_hedge
        learning_rate_actor_critic=args.lr_actor_critic,
        learning_rate_clstm=args.lr_clstm,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        batch_size=args.batch_size,
        n_epochs=args.ppo_epochs
    )
    
    # Enable multi-GPU training if available
    if torch.cuda.device_count() > 1:
        logger.info(f"Enabling DataParallel training on {torch.cuda.device_count()} GPUs")
        
        # Set balanced memory allocation for heterogeneous GPUs
        # GPU 0 has 48GB, GPU 1 has 24GB, so we can balance accordingly
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of available memory
        
        # Wrap the network in DataParallel
        agent.network = torch.nn.DataParallel(agent.network, device_ids=list(range(torch.cuda.device_count())))
        agent.base_network = agent.network.module  # Access the underlying module
        
        # Increase batch size for multi-GPU efficiency
        original_batch_size = agent.batch_size
        agent.batch_size = agent.batch_size * torch.cuda.device_count()
        logger.info(f"Increased batch size from {original_batch_size} to {agent.batch_size} for multi-GPU training")
        
        # Enable cudnn benchmarking for better performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    
    # Find and load the latest checkpoint
    latest_checkpoint = None
    start_episode = 0
    
    # Check for existing checkpoints
    checkpoint_dir = "checkpoints/live_options"
    if os.path.exists(checkpoint_dir):
        # Find all checkpoint files
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_ep") and f.endswith(".pt")]
        
        if checkpoint_files:
            # Sort by episode number
            checkpoint_episodes = []
            for f in checkpoint_files:
                try:
                    ep_num = int(f.replace("checkpoint_ep", "").replace(".pt", ""))
                    checkpoint_episodes.append((ep_num, f))
                except:
                    pass
            
            if checkpoint_episodes:
                # Get the latest checkpoint
                checkpoint_episodes.sort(key=lambda x: x[0], reverse=True)
                latest_ep, latest_file = checkpoint_episodes[0]
                latest_checkpoint = os.path.join(checkpoint_dir, latest_file)
                start_episode = latest_ep
                
    # Also check for best model if no checkpoint found
    if not latest_checkpoint:
        best_models = [f for f in os.listdir(checkpoint_dir) if f.startswith("best_model_wr") and f.endswith(".pt")]
        if best_models:
            # Sort by win rate
            best_models.sort(reverse=True)
            latest_checkpoint = os.path.join(checkpoint_dir, best_models[0])
            logger.info(f"No checkpoint found, using best model: {best_models[0]}")
    
    # Load checkpoint if found or explicitly specified
    if args.checkpoint:
        # User specified checkpoint takes precedence
        latest_checkpoint = args.checkpoint
        logger.info(f"Loading user-specified checkpoint: {latest_checkpoint}")
    # Handle --no-resume flag
    if args.no_resume:
        args.resume = False
        latest_checkpoint = None
        logger.info("--no-resume specified, starting fresh training")
    elif latest_checkpoint and args.resume:  # Resume if checkpoint exists and not disabled
        logger.info(f"Found latest checkpoint: {latest_checkpoint} (Episode {start_episode})")
    
    # Training metrics - initialize before loading checkpoint
    episode_rewards = []
    win_rates = []
    best_win_rate = 0
    
    # Load checkpoint if found
    if latest_checkpoint and os.path.exists(latest_checkpoint):
        try:
            checkpoint = torch.load(latest_checkpoint, weights_only=False)
            
            # Check for dimension mismatch
            if 'network_state_dict' in checkpoint:
                old_input_dim = checkpoint['network_state_dict'].get(
                    'clstm_encoder.input_projection.weight', 
                    torch.zeros(1, 1)
                ).shape[1]
                
                if torch.cuda.device_count() > 1:
                    current_input_dim = agent.network.module.clstm_encoder.input_projection.weight.shape[1]
                else:
                    current_input_dim = agent.network.clstm_encoder.input_projection.weight.shape[1]
                
                if old_input_dim != current_input_dim:
                    logger.warning(f"⚠️ Checkpoint dimension mismatch: old={old_input_dim}, current={current_input_dim}")
                    logger.warning("This checkpoint is incompatible with the current observation space.")
                    logger.warning("Starting fresh training instead.")
                    start_episode = 0
                else:
                    # Load the model state
                    if torch.cuda.device_count() > 1:
                        agent.network.module.load_state_dict(checkpoint['network_state_dict'])
                    else:
                        agent.network.load_state_dict(checkpoint['network_state_dict'])
                    
                    # Load optimizer states if available
                    if 'clstm_optimizer_state_dict' in checkpoint:
                        agent.clstm_optimizer.load_state_dict(checkpoint['clstm_optimizer_state_dict'])
                    if 'ppo_optimizer_state_dict' in checkpoint:
                        agent.ppo_optimizer.load_state_dict(checkpoint['ppo_optimizer_state_dict'])
                    
                    # Get starting episode
                    if 'episode' in checkpoint:
                        start_episode = checkpoint['episode']
                    
                    # Restore training history if available
                    if 'episode_rewards' in checkpoint:
                        episode_rewards = checkpoint['episode_rewards']
                        logger.info(f"Restored {len(episode_rewards)} episodes of reward history")
                    if 'win_rates' in checkpoint:
                        win_rates = checkpoint['win_rates']
                        logger.info(f"Restored {len(win_rates)} episodes of win rate history")
                    if 'best_win_rate' in checkpoint:
                        best_win_rate = checkpoint['best_win_rate']
                        logger.info(f"Best win rate so far: {best_win_rate:.1%}")
                    
                    logger.info(f"✅ Successfully loaded checkpoint from episode {start_episode}")
                    logger.info(f"Resuming training from episode {start_episode + 1}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Starting fresh training")
            start_episode = 0
    else:
        logger.info("No checkpoint found, starting fresh training")
    
    try:
        for episode in range(start_episode, args.episodes):
            logger.info(f"\n{'='*80}")
            logger.info(f"Episode {episode + 1}/{args.episodes}")
            logger.info(f"{'='*80}")
            
            # Reset environment
            obs = env.reset()
            episode_reward = 0
            episode_steps = 0
            
            # Episode loop
            for step in range(args.max_steps_per_episode):
                # Get action from agent
                action, action_info = agent.act(obs, deterministic=False)
                
                # Log CLSTM processing info periodically
                if step == 0 and episode % 10 == 0:
                    if torch.cuda.device_count() > 1:
                        logger.info(f"CLSTM processing historical sequence of {len(agent.network.module.feature_memory)} timesteps")
                    else:
                        logger.info(f"CLSTM processing historical sequence of {len(agent.base_network.feature_memory)} timesteps")
                
                # Execute action in environment
                next_obs, reward, done, info = env.step(action)
                
                # Store transition
                agent.store_transition(
                    observation=obs,
                    action=action,
                    reward=reward,
                    next_observation=next_obs,
                    done=done,
                    info=action_info
                )
                
                # Update
                obs = next_obs
                episode_reward += reward
                episode_steps += 1
                
                # Train when buffer is full
                if len(agent.buffer) >= agent.batch_size * 2:
                    # For multi-GPU, ensure we're using both GPUs efficiently
                    if torch.cuda.device_count() > 1:
                        # Clear GPU cache to prevent memory fragmentation
                        torch.cuda.empty_cache()
                    
                    metrics = agent.train()
                    if step == 0 and episode % 10 == 0:
                        logger.info(f"Training metrics: {metrics}")
                        if torch.cuda.is_available():
                            for i in range(torch.cuda.device_count()):
                                logger.info(f"GPU {i} memory: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB / {torch.cuda.max_memory_allocated(i) / 1024**3:.2f} GB peak")
                
                # Log progress less frequently to speed up training
                if step % 50 == 0 and episode % 5 == 0:
                    portfolio_value = env._calculate_portfolio_value()
                    logger.info(f"Step {step}: Portfolio=${portfolio_value:.2f}, "
                              f"Positions={len(env.options_positions)}, "
                              f"Win Rate={info['win_rate']:.1%}")
                
                if done:
                    break
                
                # Check market hours and reset intraday stats at open (only in live mode)
                if not env.simulation_mode:
                    now = datetime.now()
                    if now.hour == 9 and now.minute == 30 and step == 0:
                        env._reset_intraday_stats()
                        logger.info("Market open - reset intraday statistics")
                    
                    if now.hour >= 16 or now.hour < 9 or (now.hour == 9 and now.minute < 30):
                        logger.info("Outside market hours, pausing...")
                        time.sleep(60)
            
            # Episode summary
            final_value = env._calculate_portfolio_value()
            episode_return = (final_value - args.initial_capital) / args.initial_capital
            win_rate = info['win_rate']
            
            episode_rewards.append(episode_reward)
            win_rates.append(win_rate)
            
            logger.info(f"\nEpisode {episode + 1} Complete:")
            logger.info(f"Total Reward: {episode_reward:.2f}")
            logger.info(f"Portfolio Return: {episode_return:.1%}")
            logger.info(f"Win Rate: {win_rate:.1%}")
            logger.info(f"Total Trades: {env.winning_trades + env.losing_trades}")
            
            # Save best model
            if win_rate > best_win_rate and env.winning_trades + env.losing_trades >= 10:
                best_win_rate = win_rate
                # Handle multi-GPU saving
                if torch.cuda.device_count() > 1:
                    # Save the underlying module, not the DataParallel wrapper
                    save_path = f"checkpoints/live_options/best_model_wr{int(win_rate*100)}.pt"
                    torch.save({
                        'network_state_dict': agent.network.module.state_dict(),
                        'clstm_optimizer_state_dict': agent.clstm_optimizer.state_dict(),
                        'ppo_optimizer_state_dict': agent.ppo_optimizer.state_dict(),
                        'episode': episode + 1,
                        'win_rate': win_rate
                    }, save_path)
                else:
                    agent.save(f"checkpoints/live_options/best_model_wr{int(win_rate*100)}.pt")
                logger.info(f"🏆 New best model saved! Win rate: {win_rate:.1%}")
            
            # Regular checkpoints
            if (episode + 1) % args.save_interval == 0:
                checkpoint_path = f"checkpoints/live_options/checkpoint_ep{episode+1}.pt"
                
                # Handle multi-GPU saving
                if torch.cuda.device_count() > 1:
                    torch.save({
                        'network_state_dict': agent.network.module.state_dict(),
                        'clstm_optimizer_state_dict': agent.clstm_optimizer.state_dict(),
                        'ppo_optimizer_state_dict': agent.ppo_optimizer.state_dict(),
                        'episode': episode + 1,
                        'best_win_rate': best_win_rate,
                        'episode_rewards': episode_rewards,
                        'win_rates': win_rates
                    }, checkpoint_path)
                else:
                    # Add episode info to save
                    agent.save(checkpoint_path)
                    # Update with additional info
                    checkpoint = torch.load(checkpoint_path, weights_only=False)
                    checkpoint.update({
                        'episode': episode + 1,
                        'best_win_rate': best_win_rate,
                        'episode_rewards': episode_rewards,
                        'win_rates': win_rates
                    })
                    torch.save(checkpoint, checkpoint_path)
                
                logger.info(f"💾 Saved checkpoint at episode {episode + 1}")
                
                # Save metrics
                metrics_data = {
                    'episode': episode + 1,
                    'avg_reward': np.mean(episode_rewards[-args.save_interval:]) if episode_rewards else 0,
                    'avg_win_rate': np.mean(win_rates[-args.save_interval:]) if win_rates else 0,
                    'best_win_rate': best_win_rate,
                    'total_episodes': len(episode_rewards),
                    'total_winning_trades': env.winning_trades,
                    'total_losing_trades': env.losing_trades
                }
                
                with open(f"checkpoints/live_options/metrics_ep{episode+1}.json", 'w') as f:
                    json.dump(metrics_data, f, indent=2)
            
            # Check convergence
            if len(win_rates) >= 20:
                recent_wr = np.mean(win_rates[-20:])
                if recent_wr >= 0.6:
                    logger.info(f"🎉 Achieved target win rate: {recent_wr:.1%}")
                    break
    
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted")
    except Exception as e:
        logger.error(f"Training error: {e}")
        traceback.print_exc()
    finally:
        env.stop()
        
        # Save final model with full state
        final_path = "checkpoints/live_options/final_model.pt"
        if torch.cuda.device_count() > 1:
            torch.save({
                'network_state_dict': agent.network.module.state_dict(),
                'clstm_optimizer_state_dict': agent.clstm_optimizer.state_dict(),
                'ppo_optimizer_state_dict': agent.ppo_optimizer.state_dict(),
                'episode': episode + 1 if 'episode' in locals() else start_episode,
                'best_win_rate': best_win_rate,
                'episode_rewards': episode_rewards,
                'win_rates': win_rates,
                'final': True
            }, final_path)
        else:
            agent.save(final_path)
            # Update with additional info
            checkpoint = torch.load(final_path, weights_only=False)
            checkpoint.update({
                'episode': episode + 1 if 'episode' in locals() else start_episode,
                'best_win_rate': best_win_rate,
                'episode_rewards': episode_rewards,
                'win_rates': win_rates,
                'final': True
            })
            torch.save(checkpoint, final_path)
        
        logger.info(f"💾 Saved final model at {final_path}")
        logger.info("Training complete")


def main():
    parser = argparse.ArgumentParser(description='Live Options Paper Trading with PPO-CLSTM')
    
    # Environment
    parser.add_argument('--symbols', type=str, default='SPY',
                       help='Symbols to trade (comma-separated, default: SPY only to avoid rate limits)')
    parser.add_argument('--initial-capital', type=float, default=100000,
                       help='Starting capital')
    
    # GPU settings
    parser.add_argument('--gpus', type=str, default=None,
                       help='GPUs to use (e.g., "0,1" for both GPUs, "0" for first GPU only)')
    
    # Training
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of episodes')
    parser.add_argument('--max-steps-per-episode', type=int, default=78,
                       help='Max steps per episode (6.5 hours * 12 five-minute bars)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='PPO batch size')
    parser.add_argument('--ppo-epochs', type=int, default=4,
                       help='PPO epochs per update')
    
    # Learning rates
    parser.add_argument('--lr-actor-critic', type=float, default=1e-4,
                       help='Actor-critic learning rate')
    parser.add_argument('--lr-clstm', type=float, default=3e-4,
                       help='CLSTM learning rate')
    
    # Checkpointing
    parser.add_argument('--save-interval', type=int, default=5,
                       help='Save every N episodes')
    parser.add_argument('--resume', action='store_true', default=True,
                       help='Resume training from latest checkpoint (default: True)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh training, ignore existing checkpoints')
    parser.add_argument('--checkpoint', type=str,
                       help='Specific checkpoint path to load')
    
    # Simulation mode
    parser.add_argument('--simulation-mode', action='store_true',
                       help='Force simulation mode even during market hours')
    parser.add_argument('--live-mode', action='store_true',
                       help='Force live trading mode (requires market to be open)')
    
    # Parallel processing
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel episode collection even with multiple GPUs')
    
    args = parser.parse_args()
    
    # Handle simulation mode
    if args.simulation_mode and args.live_mode:
        parser.error("Cannot specify both --simulation-mode and --live-mode")
    
    if args.live_mode:
        args.simulation_mode = False
    elif args.simulation_mode:
        args.simulation_mode = True
    else:
        args.simulation_mode = None  # Auto-detect based on market hours
    
    # Set GPU configuration
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        logger.info(f"Set CUDA_VISIBLE_DEVICES={args.gpus}")
    
    # Create directories
    os.makedirs('checkpoints/live_options', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run training
    train_live_options(args)


if __name__ == '__main__':
    main()