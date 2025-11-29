#!/usr/bin/env python3
"""
Real-time market data stream from Massive.com WebSocket API
Provides live options and stock data for paper trading
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Callable
import websockets
import aiohttp
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class MassiveRealtimeStream:
    """
    Real-time WebSocket stream for options and stock data from Massive.com
    
    Features:
    - WebSocket connection to Massive.com streaming API
    - Automatic reconnection on disconnect
    - Real-time quotes, trades, and Greeks updates
    - Buffered state management for latest market data
    """
    
    def __init__(self, api_key: str, symbols: List[str]):
        self.api_key = api_key
        self.symbols = symbols
        self.ws = None
        self.running = False
        
        # Latest market state
        self.stock_quotes = {}  # {symbol: {bid, ask, last, timestamp}}
        self.option_quotes = {}  # {ticker: {bid, ask, greeks, iv, timestamp}}
        self.option_trades = {}  # {ticker: {price, size, timestamp}}
        
        # Callbacks for real-time updates
        self.quote_callbacks = []
        self.trade_callbacks = []
        
        # Connection state
        self.connected = False
        self.reconnect_delay = 5
        self.max_reconnect_delay = 60
        
    async def connect(self):
        """Connect to Massive.com WebSocket API"""
        ws_url = "wss://socket.massive.com/options"
        
        try:
            self.ws = await websockets.connect(
                ws_url,
                extra_headers={"Authorization": f"Bearer {self.api_key}"}
            )
            self.connected = True
            logger.info(f"✅ Connected to Massive.com WebSocket")
            
            # Subscribe to symbols
            await self._subscribe()
            
            # Reset reconnect delay on successful connection
            self.reconnect_delay = 5
            
        except Exception as e:
            logger.error(f"❌ WebSocket connection failed: {e}")
            self.connected = False
            raise
    
    async def _subscribe(self):
        """Subscribe to stock and options streams"""
        # Subscribe to stock quotes
        for symbol in self.symbols:
            subscribe_msg = {
                "action": "subscribe",
                "params": f"Q.{symbol}"  # Quote stream
            }
            await self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"📡 Subscribed to {symbol} quotes")
        
        # Subscribe to options for each underlying
        for symbol in self.symbols:
            subscribe_msg = {
                "action": "subscribe",
                "params": f"O.{symbol}.*"  # All options for this underlying
            }
            await self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"📡 Subscribed to {symbol} options")
    
    async def start(self):
        """Start the real-time data stream"""
        self.running = True
        
        while self.running:
            try:
                if not self.connected:
                    await self.connect()
                
                # Listen for messages
                async for message in self.ws:
                    await self._handle_message(message)
                    
            except websockets.exceptions.ConnectionClosed:
                logger.warning("⚠️ WebSocket connection closed, reconnecting...")
                self.connected = False
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
                
            except Exception as e:
                logger.error(f"❌ WebSocket error: {e}")
                self.connected = False
                await asyncio.sleep(self.reconnect_delay)
    
    async def _handle_message(self, message: str):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            
            # Handle different message types
            msg_type = data.get('ev')  # Event type
            
            if msg_type == 'Q':  # Stock quote
                await self._handle_stock_quote(data)
            elif msg_type == 'T':  # Trade
                await self._handle_trade(data)
            elif msg_type == 'A':  # Aggregate (OHLCV bar)
                await self._handle_aggregate(data)
            elif msg_type == 'status':
                logger.debug(f"Status: {data.get('message')}")
                
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON message: {message[:100]}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _handle_stock_quote(self, data: Dict):
        """Handle stock quote update"""
        symbol = data.get('sym')
        
        self.stock_quotes[symbol] = {
            'bid': data.get('bp'),
            'ask': data.get('ap'),
            'bid_size': data.get('bs'),
            'ask_size': data.get('as'),
            'last': data.get('lp'),
            'timestamp': data.get('t'),
            'exchange': data.get('x')
        }
        
        # Trigger callbacks
        for callback in self.quote_callbacks:
            await callback('stock', symbol, self.stock_quotes[symbol])

    async def _handle_trade(self, data: Dict):
        """Handle trade update"""
        ticker = data.get('sym')

        trade_data = {
            'price': data.get('p'),
            'size': data.get('s'),
            'timestamp': data.get('t'),
            'exchange': data.get('x'),
            'conditions': data.get('c', [])
        }

        self.option_trades[ticker] = trade_data

        # Trigger callbacks
        for callback in self.trade_callbacks:
            await callback('trade', ticker, trade_data)

    async def _handle_aggregate(self, data: Dict):
        """Handle aggregate bar (OHLCV)"""
        # Can be used for minute/second bars if needed
        pass

    def get_stock_quote(self, symbol: str) -> Optional[Dict]:
        """Get latest stock quote"""
        return self.stock_quotes.get(symbol)

    def get_option_quote(self, ticker: str) -> Optional[Dict]:
        """Get latest option quote with Greeks"""
        return self.option_quotes.get(ticker)

    def get_current_state(self, symbols: Optional[List[str]] = None) -> Dict:
        """
        Get current market state for all symbols

        Returns:
            Dict with stock quotes, option quotes, and metadata
        """
        if symbols is None:
            symbols = self.symbols

        state = {
            'timestamp': datetime.utcnow().isoformat(),
            'stocks': {},
            'options': {},
            'connected': self.connected
        }

        for symbol in symbols:
            if symbol in self.stock_quotes:
                state['stocks'][symbol] = self.stock_quotes[symbol]

        # Get all options for requested symbols
        for ticker, quote in self.option_quotes.items():
            # Parse underlying from option ticker (O:SPY251124C00500000)
            if ticker.startswith('O:'):
                underlying = ticker.split('O:')[1][:3]  # Simplified parsing
                if underlying in symbols:
                    state['options'][ticker] = quote

        return state

    def register_quote_callback(self, callback: Callable):
        """Register callback for quote updates"""
        self.quote_callbacks.append(callback)

    def register_trade_callback(self, callback: Callable):
        """Register callback for trade updates"""
        self.trade_callbacks.append(callback)

    async def stop(self):
        """Stop the data stream"""
        self.running = False
        if self.ws:
            await self.ws.close()
        logger.info("🛑 Real-time data stream stopped")


async def fetch_options_chain_snapshot(api_key: str, symbol: str) -> Dict:
    """
    Fetch current options chain snapshot via REST API
    Used for initial state and Greeks data
    """
    url = f"https://api.massive.com/v3/snapshot/options/{symbol}"
    headers = {"Authorization": f"Bearer {api_key}"}

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return data.get('results', [])
            else:
                logger.error(f"Failed to fetch options chain: {response.status}")
                return []

