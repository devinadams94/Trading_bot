import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging
import asyncio
import json
import pickle
import os
import sys
from collections import defaultdict
import pytz
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
import math

# WebSocket imports for Massive.com API
try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    import warnings
    warnings.warn("websockets package not available. Install with: pip install websockets", ImportWarning)

# aiohttp for async HTTP requests (legacy Alpaca methods)
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    # Create a dummy class to avoid NameError
    class aiohttp:
        class ClientSession:
            pass

try:
    from scipy.stats import norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    import warnings
    warnings.warn("scipy not available, using simplified option pricing", ImportWarning)

logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """Metrics for tracking data quality"""
    total_records: int = 0
    missing_values: int = 0
    outliers: int = 0
    data_gaps: int = 0
    last_updated: datetime = None

    @property
    def quality_score(self) -> float:
        """Calculate overall data quality score (0-1)"""
        if self.total_records == 0:
            return 0.0

        # More reasonable penalties for real market data
        missing_penalty = min(0.3, self.missing_values / self.total_records)  # Cap at 30%
        outlier_penalty = min(0.2, self.outliers / max(1, self.total_records * 10))  # Much more lenient
        gap_penalty = min(0.1, self.data_gaps / max(1, self.total_records / 10))  # More lenient

        # Base score starts higher for real data
        base_score = 0.8 if self.total_records > 10 else 0.5

        return max(0.0, base_score - missing_penalty - outlier_penalty - gap_penalty)


class OptimizedHistoricalOptionsDataLoader:
    """Enhanced data loader using Massive.com WebSocket API for options data"""

    def __init__(
        self,
        api_key: str,
        api_secret: str = None,  # Not used for Massive.com, kept for compatibility
        base_url: str = None,  # Not used for Massive.com, kept for compatibility
        data_url: str = None,  # Not used for Massive.com, kept for compatibility
        cache_dir: str = 'data/options_cache',
        max_workers: int = 4,
        rate_limit_delay: float = 0.1,
        cache_ttl_hours: int = 24
    ):
        """
        Initialize data loader with Massive.com API

        Args:
            api_key: Massive.com API key (e.g., 'O_182Z1cNv_y6zMpPwjLZ_pwIH8W9lWF')
            api_secret: Not used (kept for backward compatibility)
            base_url: Not used (kept for backward compatibility)
            data_url: Not used (kept for backward compatibility)
            cache_dir: Directory for caching data
            max_workers: Number of worker threads
            rate_limit_delay: Delay between API requests
            cache_ttl_hours: Cache time-to-live in hours

        WebSocket URLs:
            - Real-time: wss://socket.massive.com/options
            - Delayed (15-min): wss://delayed.massive.com/options
        """
        if not HAS_WEBSOCKETS:
            raise ImportError("websockets package required. Install with: pip install websockets")

        # Massive.com API configuration
        self.api_key = api_key

        # REST API URLs (Polygon.io - Massive.com uses Polygon infrastructure)
        self.rest_api_base_url = "https://api.polygon.io"

        # WebSocket URLs for different data types
        self.ws_url_stocks_realtime = "wss://socket.massive.com/stocks"
        self.ws_url_stocks_delayed = "wss://delayed.massive.com/stocks"
        self.ws_url_options_realtime = "wss://socket.massive.com/options"
        self.ws_url_options_delayed = "wss://delayed.massive.com/options"

        self.use_realtime = False  # Default to delayed data (15-min delay)

        # Cache configuration
        self.cache_dir = cache_dir
        self.cache_ttl_hours = cache_ttl_hours

        # Create cache directory structure
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.join(cache_dir, 'stocks'), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, 'options'), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, 'metadata'), exist_ok=True)

        # Eastern timezone for market hours
        self.market_tz = pytz.timezone('US/Eastern')

        # Data storage with thread safety
        self.historical_data = defaultdict(list)
        self.options_chains = defaultdict(dict)
        self.data_quality_metrics = defaultdict(DataQualityMetrics)
        self._data_lock = threading.RLock()

        # Enhanced rate limiting
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit_lock = threading.Lock()

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Cache management
        self.cache_index = self._load_cache_index()
        self.min_request_interval = 0.2  # 200ms between requests

        # WebSocket connection state (separate connections for stocks and options)
        self.ws_connection_stocks = None
        self.ws_connection_options = None
        self.ws_authenticated_stocks = False
        self.ws_authenticated_options = False
        self.ws_data_buffer = defaultdict(list)

        # Options data availability flag
        # Set to True since REST API is now implemented
        self.has_options_data = True

        logger.info(f"🚀 Initialized Massive.com data loader with API key: {api_key[:8]}...")
        logger.info(f"   REST API Base URL: {self.rest_api_base_url}")
        logger.info(f"   Stock WebSocket URL: {self.ws_url_stocks_realtime if self.use_realtime else self.ws_url_stocks_delayed}")
        logger.info(f"   Options WebSocket URL: {self.ws_url_options_realtime if self.use_realtime else self.ws_url_options_delayed}")
        logger.info(f"   Cache directory: {cache_dir}")
        logger.info(f"✅ REST API enabled for historical data loading")

    def _load_cache_index(self) -> Dict:
        """Load cache index for tracking cached data"""
        index_path = os.path.join(self.cache_dir, 'metadata', 'cache_index.json')
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
        return {}

    def _save_cache_index(self):
        """Save cache index"""
        index_path = os.path.join(self.cache_dir, 'metadata', 'cache_index.json')
        try:
            with open(index_path, 'w') as f:
                json.dump(self.cache_index, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")

    def _get_cache_key(self, symbol: str, start_date: datetime, end_date: datetime, data_type: str) -> str:
        """Generate cache key for data"""
        date_str = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        return hashlib.md5(f"{symbol}_{date_str}_{data_type}".encode()).hexdigest()

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache_index:
            return False

        cache_time = datetime.fromisoformat(self.cache_index[cache_key]['timestamp'])
        return (datetime.now() - cache_time).total_seconds() < (self.cache_ttl_hours * 3600)

    def _rate_limit(self):
        """Enhanced rate limiting with thread safety"""
        with self.rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time

            if time_since_last < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - time_since_last
                time.sleep(sleep_time)

            self.last_request_time = time.time()
            self.request_count += 1

            # Log every 100 requests
            if self.request_count % 100 == 0:
                logger.info(f"Made {self.request_count} API requests")

    async def _rate_limit_async(self):
        """Async rate limiting that doesn't block the event loop"""
        with self.rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time

            if time_since_last < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - time_since_last
                # Use asyncio.sleep instead of time.sleep to not block event loop
                await asyncio.sleep(sleep_time)

            self.last_request_time = time.time()
            self.request_count += 1

            # Log every 10 requests for better progress feedback
            if self.request_count % 10 == 0:
                logger.info(f"📡 Made {self.request_count} API requests")
                sys.stdout.flush()
                sys.stderr.flush()

    def _validate_data_quality(self, data: pd.DataFrame, symbol: str) -> DataQualityMetrics:
        """Validate data quality and return metrics"""
        metrics = DataQualityMetrics()
        metrics.total_records = len(data)
        metrics.last_updated = datetime.now()

        if data.empty:
            return metrics

        # Check for missing values
        metrics.missing_values = data.isnull().sum().sum()

        # Check for outliers (simple z-score method)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in data.columns and not data[col].empty:
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                metrics.outliers += (z_scores > 3).sum()

        # Check for data gaps (missing time periods)
        if 'timestamp' in data.columns:
            data_sorted = data.sort_values('timestamp')
            time_diffs = data_sorted['timestamp'].diff()
            expected_interval = time_diffs.median()
            large_gaps = time_diffs > (expected_interval * 2)
            metrics.data_gaps = large_gaps.sum()

        # Store metrics
        self.data_quality_metrics[symbol] = metrics

        logger.info(f"Data quality for {symbol}: Score={metrics.quality_score:.2f}, "
                   f"Records={metrics.total_records}, Missing={metrics.missing_values}, "
                   f"Outliers={metrics.outliers}, Gaps={metrics.data_gaps}")

        return metrics

    def _get_cache_path(self, symbol: str, date: datetime) -> str:
        """Get cache file path for a specific symbol and date"""
        date_str = date.strftime('%Y%m%d')
        return os.path.join(self.cache_dir, f"{symbol}_{date_str}_options.pkl")

    async def _connect_websocket(self, data_type: str = "options", verify_ssl: bool = True) -> bool:
        """
        Connect to Massive.com WebSocket API and authenticate

        Args:
            data_type: Type of data to connect to ("stocks" or "options")
            verify_ssl: Whether to verify SSL certificates (default: True)

        Returns:
            bool: True if connection and authentication successful
        """
        try:
            import ssl

            # Select the appropriate WebSocket URL based on data type
            if data_type == "stocks":
                ws_url = self.ws_url_stocks_realtime if self.use_realtime else self.ws_url_stocks_delayed
                connection_attr = "ws_connection_stocks"
                auth_attr = "ws_authenticated_stocks"
            else:  # options
                ws_url = self.ws_url_options_realtime if self.use_realtime else self.ws_url_options_delayed
                connection_attr = "ws_connection_options"
                auth_attr = "ws_authenticated_options"

            logger.info(f"🔌 Connecting to Massive.com {data_type.upper()} WebSocket: {ws_url}")

            # Create SSL context
            if verify_ssl:
                # Use certifi's CA bundle for SSL verification
                try:
                    import certifi
                    ssl_context = ssl.create_default_context(cafile=certifi.where())
                    logger.info("✅ Using certifi CA bundle for SSL verification")
                except ImportError:
                    ssl_context = ssl.create_default_context()
                    logger.warning("⚠️ certifi not available, using system CA bundle")
            else:
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                logger.warning("⚠️ SSL certificate verification disabled")

            # Connect to WebSocket
            ws_connection = await websockets.connect(ws_url, ssl=ssl_context)
            setattr(self, connection_attr, ws_connection)
            logger.info(f"✅ {data_type.upper()} WebSocket connected")

            # Wait for connection confirmation
            response = await ws_connection.recv()
            logger.info(f"📨 Connection response: {response}")

            # Authenticate with API key
            auth_message = json.dumps({"action": "auth", "params": self.api_key})
            await ws_connection.send(auth_message)
            logger.info(f"🔐 Sent authentication with API key: {self.api_key[:8]}...")

            # Wait for authentication response
            auth_response = await ws_connection.recv()
            auth_data = json.loads(auth_response)
            logger.info(f"📨 Auth response: {auth_data}")

            # Check if authentication was successful
            if isinstance(auth_data, list) and len(auth_data) > 0:
                if auth_data[0].get('ev') == 'status' and auth_data[0].get('status') == 'auth_success':
                    setattr(self, auth_attr, True)
                    logger.info(f"✅ {data_type.upper()} WebSocket authenticated successfully")
                    return True

            logger.error(f"❌ {data_type.upper()} authentication failed: {auth_data}")
            return False

        except Exception as e:
            logger.error(f"❌ {data_type.upper()} WebSocket connection failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    async def _subscribe_to_options(self, option_symbols: List[str]) -> bool:
        """
        Subscribe to options data feeds

        Args:
            option_symbols: List of option contract symbols (e.g., ['O:SPY240119C00450000'])

        Returns:
            bool: True if subscription successful
        """
        if not self.ws_authenticated_options:
            logger.error("❌ Cannot subscribe: OPTIONS WebSocket not authenticated")
            return False

        try:
            # Massive.com supports up to 1,000 simultaneous subscriptions per connection
            # Subscribe to minute aggregates (AM) for the option contracts
            subscription_params = ",".join([f"AM.{symbol}" for symbol in option_symbols])
            subscribe_message = json.dumps({
                "action": "subscribe",
                "params": subscription_params
            })

            await self.ws_connection_options.send(subscribe_message)
            logger.info(f"📡 Subscribed to {len(option_symbols)} option contracts")

            return True

        except Exception as e:
            logger.error(f"❌ Subscription failed: {e}")
            return False

    async def _collect_websocket_data(self, duration_seconds: int = 60, data_type: str = "options") -> Dict[str, List[Dict]]:
        """
        Collect data from WebSocket for a specified duration

        Args:
            duration_seconds: How long to collect data (default: 60 seconds)
            data_type: Type of data to collect ("stocks" or "options")

        Returns:
            Dict mapping symbols to list of data points
        """
        collected_data = defaultdict(list)
        start_time = time.time()

        # Select the appropriate WebSocket connection
        ws_connection = self.ws_connection_options if data_type == "options" else self.ws_connection_stocks

        try:
            while (time.time() - start_time) < duration_seconds:
                # Set a timeout to avoid blocking forever
                try:
                    message = await asyncio.wait_for(
                        ws_connection.recv(),
                        timeout=5.0
                    )

                    data = json.loads(message)

                    # Process each message in the response
                    if isinstance(data, list):
                        for item in data:
                            if item.get('ev') == 'AM':  # Aggregate Minute event
                                symbol = item.get('sym')
                                if symbol:
                                    collected_data[symbol].append(item)

                except asyncio.TimeoutError:
                    # No data received in timeout period, continue
                    continue

        except Exception as e:
            logger.error(f"❌ Error collecting {data_type.upper()} WebSocket data: {e}")

        logger.info(f"📊 Collected {data_type.upper()} data for {len(collected_data)} symbols")
        return dict(collected_data)

    async def _disconnect_websocket(self, data_type: str = "both"):
        """
        Disconnect from WebSocket

        Args:
            data_type: Which connection to disconnect ("stocks", "options", or "both")
        """
        if data_type in ["options", "both"] and self.ws_connection_options:
            try:
                await self.ws_connection_options.close()
                logger.info("🔌 OPTIONS WebSocket disconnected")
            except Exception as e:
                logger.error(f"❌ Error disconnecting OPTIONS WebSocket: {e}")
            finally:
                self.ws_connection_options = None
                self.ws_authenticated_options = False

        if data_type in ["stocks", "both"] and self.ws_connection_stocks:
            try:
                await self.ws_connection_stocks.close()
                logger.info("🔌 STOCKS WebSocket disconnected")
            except Exception as e:
                logger.error(f"❌ Error disconnecting STOCKS WebSocket: {e}")
            finally:
                self.ws_connection_stocks = None
                self.ws_authenticated_stocks = False

    async def load_historical_stock_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1Hour"
    ) -> Dict[str, pd.DataFrame]:
        """Load historical stock data using the new Alpaca data client"""
        result = {}
        total_symbols = len(symbols)

        msg = f"📊 Loading stock data for {total_symbols} symbols...\n   Date range: {start_date.date()} to {end_date.date()}\n   Timeframe: {timeframe}\n"
        print(msg, flush=True)
        logger.info(msg)
        sys.stdout.flush()
        sys.stderr.flush()

        for idx, symbol in enumerate(symbols, 1):
            msg = f"  [{idx}/{total_symbols}] 📥 Downloading {symbol}..."
            print(msg, flush=True)
            logger.info(msg)
            sys.stdout.flush()
            sys.stderr.flush()
            try:
                cache_key = self._get_cache_key(symbol, start_date, end_date, f"stock_{timeframe}")
                cache_path = os.path.join(self.cache_dir, 'stocks', f"{cache_key}.pkl")

                # Check cache first
                if self._is_cache_valid(cache_key) and os.path.exists(cache_path):
                    msg = f"  [{idx}/{total_symbols}] 💾 Loading {symbol} from cache..."
                    print(msg, flush=True)
                    logger.info(msg)
                    sys.stdout.flush()
                    sys.stderr.flush()

                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)

                    msg = f"  [{idx}/{total_symbols}] ✅ {symbol}: {len(data)} bars (cached)"
                    print(msg, flush=True)
                    logger.info(msg)
                    sys.stdout.flush()
                    sys.stderr.flush()
                    result[symbol] = data
                    continue

                # Use Massive.com REST API for historical stock data
                msg = f"  [{idx}/{total_symbols}] 🌐 Fetching historical stock data from REST API..."
                print(msg, flush=True)
                logger.info(msg)
                sys.stdout.flush()
                sys.stderr.flush()

                await self._rate_limit_async()

                try:
                    # Fetch real historical stock data using REST API
                    data = await self._fetch_stock_data_rest_api(symbol, start_date, end_date)

                    if data is not None and len(data) > 0:
                        msg = f"  [{idx}/{total_symbols}] ✅ Fetched {len(data)} real bars for {symbol}"
                        print(msg, flush=True)
                        logger.info(msg)
                        sys.stdout.flush()
                        sys.stderr.flush()
                    else:
                        msg = f"  [{idx}/{total_symbols}] ⚠️ No data returned from REST API for {symbol}"
                        print(msg, flush=True)
                        logger.warning(msg)
                        sys.stdout.flush()
                        sys.stderr.flush()
                        continue

                except Exception as e:
                    msg = f"  [{idx}/{total_symbols}] ❌ {symbol}: {type(e).__name__}: {e}"
                    print(msg, flush=True)
                    logger.error(msg)
                    sys.stdout.flush()
                    sys.stderr.flush()
                    continue

                # Validate data quality
                quality_metrics = self._validate_data_quality(data, f"{symbol}_stock")

                if quality_metrics.quality_score < 0.3:
                    msg = f"  [{idx}/{total_symbols}] ⚠️ {symbol}: Low quality ({quality_metrics.quality_score:.2f})"
                    print(msg, flush=True)
                    logger.warning(msg)
                else:
                    msg = f"  [{idx}/{total_symbols}] ✅ {symbol}: {len(data)} bars (quality: {quality_metrics.quality_score:.2f})"
                    print(msg, flush=True)
                    logger.info(msg)

                sys.stdout.flush()
                sys.stderr.flush()

                # Cache the data
                msg = f"  [{idx}/{total_symbols}] 💾 Caching {symbol} data..."
                print(msg, flush=True)
                logger.info(msg)
                sys.stdout.flush()
                sys.stderr.flush()
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)

                # Update cache index
                self.cache_index[cache_key] = {
                    'symbol': symbol,
                    'data_type': f'stock_{timeframe}',
                    'timestamp': datetime.now().isoformat(),
                    'records': len(data),
                    'quality_score': quality_metrics.quality_score
                }

                result[symbol] = data

            except Exception as e:
                error_msg = str(e)
                msg = f"  [{idx}/{total_symbols}] ❌ {symbol}: {type(e).__name__}: {error_msg}"
                print(msg, flush=True)
                logger.error(msg)

                # Provide helpful context for common errors
                if "401" in error_msg or "Unauthorized" in error_msg:
                    msg = f"  [{idx}/{total_symbols}] 🔑 API authentication failed - check your API keys"
                    print(msg, flush=True)
                    logger.error(msg)
                elif "403" in error_msg or "Forbidden" in error_msg:
                    msg = f"  [{idx}/{total_symbols}] 🚫 API access forbidden - check your subscription/permissions"
                    print(msg, flush=True)
                    logger.error(msg)
                elif "429" in error_msg or "rate limit" in error_msg.lower():
                    msg = f"  [{idx}/{total_symbols}] ⏱️ Rate limit exceeded - waiting before retry..."
                    print(msg, flush=True)
                    logger.error(msg)
                elif "timeout" in error_msg.lower():
                    msg = f"  [{idx}/{total_symbols}] ⏰ API request timed out"
                    print(msg, flush=True)
                    logger.error(msg)
                elif "connection" in error_msg.lower():
                    msg = f"  [{idx}/{total_symbols}] 🌐 Network connection error"
                    print(msg, flush=True)
                    logger.error(msg)

                sys.stdout.flush()
                sys.stderr.flush()
                continue

        self._save_cache_index()
        return result

    async def load_historical_options_data(
        self,
        symbol: str = None,
        symbols: List[str] = None,
        start_date: datetime = None,
        end_date: datetime = None,
        use_cache: bool = True
    ) -> Dict:
        """Enhanced options data loading with better caching and validation"""

        # Handle both single symbol and multiple symbols
        if symbol and not symbols:
            symbols = [symbol]
        elif not symbols:
            raise ValueError("Either symbol or symbols must be provided")

        # Default dates if not provided
        if not end_date:
            end_date = datetime.now() - timedelta(days=1)
        if not start_date:
            start_date = end_date - timedelta(days=7)

        total_symbols = len(symbols)
        msg = f"📊 Loading historical options data for {total_symbols} symbols from {start_date.date()} to {end_date.date()}\n"
        print(msg, flush=True)
        logger.info(msg)
        sys.stdout.flush()
        sys.stderr.flush()

        result = {}

        # First, load stock data for underlying prices
        msg = f"📈 Loading underlying stock prices first..."
        print(msg, flush=True)
        logger.info(msg)
        sys.stdout.flush()
        sys.stderr.flush()

        stock_data = await self.load_historical_stock_data(symbols, start_date, end_date)

        msg = ""
        print(msg, flush=True)
        logger.info(msg)

        # Process each symbol
        msg = f"📊 Processing options chains for {total_symbols} symbols...\n"
        print(msg, flush=True)
        logger.info(msg)
        sys.stdout.flush()
        sys.stderr.flush()

        for idx, sym in enumerate(symbols, 1):
            msg = f"  [{idx}/{total_symbols}] 📥 Fetching options chain for {sym}..."
            print(msg, flush=True)
            logger.info(msg)
            sys.stdout.flush()
            sys.stderr.flush()
            try:
                cache_key = self._get_cache_key(sym, start_date, end_date, "options")
                cache_path = os.path.join(self.cache_dir, 'options', f"{cache_key}.pkl")

                # Check cache first
                if use_cache and self._is_cache_valid(cache_key) and os.path.exists(cache_path):
                    msg = f"  [{idx}/{total_symbols}] 💾 Loading {sym} options from cache..."
                    print(msg, flush=True)
                    logger.info(msg)
                    sys.stdout.flush()
                    sys.stderr.flush()

                    with open(cache_path, 'rb') as f:
                        symbol_data = pickle.load(f)

                    msg = f"  [{idx}/{total_symbols}] ✅ {sym} options loaded from cache ({len(symbol_data)} contracts)"
                    print(msg, flush=True)
                    logger.info(msg)
                    sys.stdout.flush()
                    sys.stderr.flush()

                    result[sym] = symbol_data
                    continue

                # Generate options data (real or simulated)
                symbol_data = []

                if self.has_options_data and sym in stock_data:
                    # Try to fetch real options data
                    msg = f"  [{idx}/{total_symbols}] 🌐 Attempting to fetch real options data for {sym}..."
                    print(msg, flush=True)
                    logger.info(msg)
                    sys.stdout.flush()
                    sys.stderr.flush()

                    try:
                        symbol_data = await self._fetch_real_options_data(
                            sym, start_date, end_date, stock_data[sym]
                        )

                        msg = f"  [{idx}/{total_symbols}] ✅ Fetched {len(symbol_data)} real options contracts for {sym}"
                        print(msg, flush=True)
                        logger.info(msg)
                        sys.stdout.flush()
                        sys.stderr.flush()
                    except Exception as e:
                        msg = f"  [{idx}/{total_symbols}] ⚠️ Failed to fetch real options data for {sym}: {e}"
                        print(msg, flush=True)
                        logger.warning(msg)
                        sys.stdout.flush()
                        sys.stderr.flush()
                        symbol_data = []

                # Fall back to simulated data if real data unavailable
                if not symbol_data and sym in stock_data:
                    msg = f"  [{idx}/{total_symbols}] 🔄 Generating simulated options for {sym}..."
                    print(msg, flush=True)
                    logger.info(msg)
                    sys.stdout.flush()
                    sys.stderr.flush()

                    symbol_data = await self._generate_simulated_options_data(
                        sym, start_date, end_date, stock_data[sym]
                    )

                    msg = f"  [{idx}/{total_symbols}] ✅ Generated {len(symbol_data)} simulated options for {sym}"
                    print(msg, flush=True)
                    logger.info(msg)
                    sys.stdout.flush()
                    sys.stderr.flush()

                if symbol_data:
                    # Validate data quality
                    df = pd.DataFrame(symbol_data)
                    quality_metrics = self._validate_data_quality(df, f"{sym}_options")

                    # Cache the data if quality is acceptable
                    if quality_metrics.quality_score > 0.3:  # Minimum quality threshold
                        with open(cache_path, 'wb') as f:
                            pickle.dump(symbol_data, f)

                        # Update cache index
                        self.cache_index[cache_key] = {
                            'symbol': sym,
                            'data_type': 'options',
                            'timestamp': datetime.now().isoformat(),
                            'records': len(symbol_data),
                            'quality_score': quality_metrics.quality_score
                        }

                    result[sym] = symbol_data
                    logger.info(f"  [{idx}/{total_symbols}] ✅ {sym}: {len(symbol_data)} options contracts (quality: {quality_metrics.quality_score:.2f})")
                else:
                    logger.warning(f"  [{idx}/{total_symbols}] ⚠️ {sym}: No options data available")

            except Exception as e:
                logger.error(f"  [{idx}/{total_symbols}] ❌ {sym}: {e}")
                continue

        # Store in instance variable for later use
        with self._data_lock:
            for sym, data in result.items():
                self.historical_data[sym] = data

        self._save_cache_index()
        logger.info(f"✅ Completed loading options data for {len(result)}/{total_symbols} symbols")
        return result

    async def _fetch_real_options_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        stock_data: pd.DataFrame
    ) -> List[Dict]:
        """
        Fetch real options data using Massive.com (Polygon.io) REST API

        This method fetches historical options data using the Polygon.io REST API.
        It gets options snapshots and aggregates for the specified date range.
        """
        msg = f"      🌐 Fetching real options data from REST API for {symbol}..."
        print(msg, flush=True)
        logger.info(msg)
        sys.stdout.flush()
        sys.stderr.flush()

        try:
            options_data = []

            # Iterate through each day in the date range
            current_date = start_date
            days_processed = 0
            total_days = (end_date - start_date).days

            while current_date <= end_date:
                # Skip weekends
                if current_date.weekday() >= 5:
                    current_date += timedelta(days=1)
                    continue

                # Rate limiting - don't overwhelm the API
                if days_processed > 0 and days_processed % 5 == 0:
                    await asyncio.sleep(0.1)  # Small delay every 5 days

                # Get stock price for this date
                stock_price = self._get_stock_price_for_date(stock_data, current_date)
                if stock_price is None:
                    current_date += timedelta(days=1)
                    continue

                # Fetch options snapshot for this underlying
                daily_options = await self._fetch_options_snapshot_rest_api(symbol, current_date, stock_price)

                if daily_options:
                    options_data.extend(daily_options)

                days_processed += 1
                current_date += timedelta(days=1)

            msg = f"      ✅ Fetched {len(options_data)} real options contracts for {symbol}"
            print(msg, flush=True)
            logger.info(msg)
            sys.stdout.flush()
            sys.stderr.flush()

            return options_data

        except Exception as e:
            msg = f"      ❌ Error fetching real options data for {symbol}: {e}"
            print(msg, flush=True)
            logger.error(msg)
            sys.stdout.flush()
            sys.stderr.flush()
            import traceback
            traceback.print_exc()
            return []

    async def _fetch_options_snapshot_rest_api(
        self,
        symbol: str,
        date: datetime,
        stock_price: float
    ) -> List[Dict]:
        """
        Fetch options snapshot for a specific underlying symbol and date

        Args:
            symbol: Underlying stock symbol (e.g., 'SPY')
            date: Date for the snapshot
            stock_price: Current stock price for filtering

        Returns:
            List of option contracts with pricing and Greeks
        """
        try:
            # Polygon.io REST API endpoint for options snapshot
            # Format: /v3/snapshot/options/{underlyingAsset}
            url = f"{self.rest_api_base_url}/v3/snapshot/options/{symbol}"

            # Filter options by strike price (within ±20% of current price)
            min_strike = stock_price * 0.8
            max_strike = stock_price * 1.2

            params = {
                'apiKey': self.api_key,
                'limit': 250,  # Get up to 250 contracts
                'strike_price.gte': min_strike,
                'strike_price.lte': max_strike,
            }

            try:
                import aiohttp
                import ssl
                import certifi

                # Create SSL context with certifi CA bundle
                ssl_context = ssl.create_default_context(cafile=certifi.where())

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, ssl=ssl_context, timeout=aiohttp.ClientTimeout(total=30)) as response:
                        if response.status == 200:
                            data = await response.json()

                            if data.get('status') == 'OK' and 'results' in data:
                                results = data['results']

                                if len(results) == 0:
                                    return []

                                # Convert to our format
                                options_list = []
                                for opt in results:
                                    try:
                                        details = opt.get('details', {})
                                        day_data = opt.get('day', {})
                                        greeks = opt.get('greeks', {})

                                        option_dict = {
                                            'timestamp': date,
                                            'symbol': symbol,
                                            'option_symbol': details.get('ticker', ''),
                                            'option_type': details.get('contract_type', 'call'),
                                            'strike': details.get('strike_price', 0),
                                            'expiration': details.get('expiration_date', ''),
                                            'bid': day_data.get('close', 0) * 0.98,  # Approximate bid
                                            'ask': day_data.get('close', 0) * 1.02,  # Approximate ask
                                            'last': day_data.get('close', 0),
                                            'volume': day_data.get('volume', 0),
                                            'open_interest': opt.get('open_interest', 0),
                                            'underlying_price': stock_price,
                                            'delta': greeks.get('delta', 0),
                                            'gamma': greeks.get('gamma', 0),
                                            'theta': greeks.get('theta', 0),
                                            'vega': greeks.get('vega', 0),
                                            'rho': greeks.get('rho', 0),
                                            'implied_volatility': opt.get('implied_volatility', 0),
                                        }
                                        options_list.append(option_dict)
                                    except Exception as e:
                                        logger.debug(f"Error parsing option contract: {e}")
                                        continue

                                return options_list
                            else:
                                return []
                        elif response.status == 429:
                            # Rate limit hit - wait and retry
                            logger.warning(f"Rate limit hit for {symbol}, waiting 1 second...")
                            await asyncio.sleep(1)
                            return []
                        else:
                            return []

            except ImportError:
                logger.error("aiohttp not installed. Install with: pip install aiohttp")
                return []

        except Exception as e:
            logger.debug(f"Error fetching options snapshot for {symbol}: {e}")
            return []

    async def _fetch_stock_data_rest_api(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch historical stock data using Massive.com (Polygon.io) REST API

        Args:
            symbol: Stock symbol (e.g., 'SPY')
            start_date: Start date for historical data
            end_date: End date for historical data

        Returns:
            DataFrame with columns: timestamp, symbol, open, high, low, close, volume
        """
        try:
            # Polygon.io REST API endpoint for aggregate bars
            # Format: /v2/aggs/ticker/{stocksTicker}/range/{multiplier}/{timespan}/{from}/{to}
            url = f"{self.rest_api_base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"

            params = {
                'apiKey': self.api_key,
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000  # Maximum results per request
            }

            # Use aiohttp for async HTTP requests
            try:
                import aiohttp
                import ssl
                import certifi

                # Create SSL context with certifi CA bundle
                ssl_context = ssl.create_default_context(cafile=certifi.where())

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, ssl=ssl_context, timeout=aiohttp.ClientTimeout(total=30)) as response:
                        if response.status == 200:
                            data = await response.json()

                            if data.get('status') == 'OK' and 'results' in data:
                                results = data['results']

                                if len(results) == 0:
                                    logger.warning(f"No stock data returned for {symbol}")
                                    return None

                                # Convert to DataFrame
                                df = pd.DataFrame(results)

                                # Rename columns to match our format
                                df = df.rename(columns={
                                    't': 'timestamp',
                                    'o': 'open',
                                    'h': 'high',
                                    'l': 'low',
                                    'c': 'close',
                                    'v': 'volume',
                                    'vw': 'vwap',
                                    'n': 'transactions'
                                })

                                # Convert timestamp from milliseconds to datetime
                                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                                df['symbol'] = symbol

                                # Select only the columns we need
                                df = df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]

                                logger.info(f"✅ Fetched {len(df)} bars for {symbol} from REST API")
                                return df
                            else:
                                logger.error(f"API returned status: {data.get('status')}, message: {data.get('message', 'No message')}")
                                return None
                        else:
                            error_text = await response.text()
                            logger.error(f"HTTP {response.status} error fetching stock data for {symbol}: {error_text}")
                            return None

            except ImportError:
                logger.error("aiohttp not installed. Install with: pip install aiohttp")
                return None

        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None


    async def _generate_simulated_options_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        stock_data: pd.DataFrame
    ) -> List[Dict]:
        """Generate realistic simulated options data based on stock data"""
        msg = f"      🎲 Generating simulated options data for {symbol}..."
        print(msg, flush=True)
        logger.debug(msg)
        sys.stdout.flush()
        sys.stderr.flush()

        options_data = []

        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() >= 5:  # Skip weekends
                current_date += timedelta(days=1)
                continue

            # Get stock price for this date
            stock_price = self._get_stock_price_for_date(stock_data, current_date)
            if stock_price is None:
                current_date += timedelta(days=1)
                continue

            # Generate options for multiple strikes and expirations
            for days_to_expiry in [7, 14, 21, 30, 45]:
                expiration = current_date + timedelta(days=days_to_expiry)

                # Generate strikes around current price
                for strike_offset in [-0.1, -0.05, 0, 0.05, 0.1]:  # ±10% from current price
                    strike = round(stock_price * (1 + strike_offset), 2)

                    for option_type in ['call', 'put']:
                        option_data = self._simulate_option_pricing(
                            symbol, current_date, stock_price, strike,
                            expiration, option_type
                        )
                        options_data.append(option_data)

            current_date += timedelta(days=1)

        msg = f"      ✅ Generated {len(options_data)} simulated options for {symbol}"
        print(msg, flush=True)
        logger.debug(msg)
        sys.stdout.flush()
        sys.stderr.flush()

        return options_data

    def _get_stock_price_for_date(self, stock_data: pd.DataFrame, date: datetime) -> Optional[float]:
        """Get stock price for a specific date from stock data"""
        try:
            # Convert date to the same timezone as stock data
            if 'timestamp' in stock_data.columns:
                # Find the closest timestamp
                stock_data['timestamp'] = pd.to_datetime(stock_data['timestamp'])
                date_normalized = pd.to_datetime(date.date())

                # Get data for the specific date
                day_data = stock_data[stock_data['timestamp'].dt.date == date.date()]

                if not day_data.empty:
                    return float(day_data['close'].iloc[-1])  # Use last close price of the day

                # If no exact match, find the closest previous date
                previous_data = stock_data[stock_data['timestamp'] <= date_normalized]
                if not previous_data.empty:
                    return float(previous_data['close'].iloc[-1])

        except Exception as e:
            logger.debug(f"Error getting stock price for {date}: {e}")

        return None

    def _extract_strike_from_symbol(self, option_symbol: str) -> float:
        """
        Extract strike price from option symbol
        Format: SPY251104C00635000 -> 635.00
        Last 8 digits represent strike * 1000
        """
        try:
            # Option symbol format: SYMBOL + YYMMDD + C/P + 8-digit strike
            # Example: SPY251104C00635000 -> strike = 635.00
            strike_str = option_symbol[-8:]  # Last 8 digits
            strike = float(strike_str) / 1000.0
            return strike
        except Exception as e:
            logger.debug(f"Error extracting strike from {option_symbol}: {e}")
            return 0.0

    def _extract_expiration_from_symbol(self, option_symbol: str) -> datetime:
        """
        Extract expiration date from option symbol
        Format: SPY251104C00635000 -> 2025-11-04
        Characters after underlying symbol: YYMMDD
        """
        try:
            # Find where the date starts (after the underlying symbol)
            # Look for 6 consecutive digits followed by C or P
            import re
            match = re.search(r'(\d{6})[CP]', option_symbol)
            if match:
                date_str = match.group(1)
                year = 2000 + int(date_str[0:2])
                month = int(date_str[2:4])
                day = int(date_str[4:6])
                return datetime(year, month, day)
        except Exception as e:
            logger.debug(f"Error extracting expiration from {option_symbol}: {e}")

        # Default to 30 days from now
        return datetime.now() + timedelta(days=30)

    def _filter_options_chain(self, options_chain, stock_price: float, current_date: datetime) -> List[Dict]:
        """Filter options chain for relevant strikes and expirations"""
        filtered_options = []

        try:
            # Filter by strike price (within ±10% of current stock price)
            min_strike = stock_price * 0.9
            max_strike = stock_price * 1.1

            # Filter by expiration (7-60 days from current date)
            min_expiry = current_date + timedelta(days=7)
            max_expiry = current_date + timedelta(days=60)

            # Handle both list and DataFrame formats
            if isinstance(options_chain, list):
                # List of option objects
                for option in options_chain:
                    # Extract fields from option object
                    if hasattr(option, 'strike_price'):
                        strike = option.strike_price
                        expiry = option.expiration_date
                        symbol = option.symbol
                        opt_type = option.type
                    elif isinstance(option, dict):
                        strike = option.get('strike_price')
                        expiry = option.get('expiration_date')
                        symbol = option.get('symbol')
                        opt_type = option.get('type')
                    else:
                        continue

                    # Convert expiry to date if needed
                    if isinstance(expiry, str):
                        expiry = pd.to_datetime(expiry).date()
                    elif isinstance(expiry, datetime):
                        expiry = expiry.date()

                    if (strike and expiry and
                        min_strike <= strike <= max_strike and
                        min_expiry.date() <= expiry <= max_expiry.date()):
                        filtered_options.append({
                            'symbol': symbol,
                            'strike_price': strike,
                            'expiration_date': expiry,
                            'type': opt_type
                        })
            else:
                # DataFrame format (legacy)
                for _, option in options_chain.iterrows():
                    if (min_strike <= option['strike_price'] <= max_strike and
                        min_expiry.date() <= option['expiration_date'] <= max_expiry.date()):
                        filtered_options.append({
                            'symbol': option['symbol'],
                            'strike_price': option['strike_price'],
                            'expiration_date': option['expiration_date'],
                            'type': option['type']
                        })

        except Exception as e:
            logger.debug(f"Error filtering options chain: {e}")

        return filtered_options

    def _simulate_option_pricing(
        self,
        symbol: str,
        date: datetime,
        stock_price: float,
        strike: float,
        expiration: datetime,
        option_type: str
    ) -> Dict:
        """Simulate realistic option pricing using Black-Scholes approximation"""
        import math

        # Calculate time to expiry
        days_to_expiry = max(1, (expiration - date).days)
        time_to_expiry = days_to_expiry / 365.0

        # Estimate implied volatility based on moneyness
        moneyness = stock_price / strike
        base_vol = 0.25  # 25% base volatility
        vol_adjustment = 0.1 * abs(1 - moneyness)  # Higher vol for OTM options
        implied_vol = base_vol + vol_adjustment

        # Risk-free rate
        risk_free_rate = 0.05

        # Calculate option price using simplified Black-Scholes
        d1 = (math.log(stock_price / strike) + (risk_free_rate + 0.5 * implied_vol**2) * time_to_expiry) / (implied_vol * math.sqrt(time_to_expiry))
        d2 = d1 - implied_vol * math.sqrt(time_to_expiry)

        if HAS_SCIPY:
            # Use proper Black-Scholes with scipy
            if option_type == 'call':
                option_price = (stock_price * norm.cdf(d1) -
                              strike * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2))
                delta = norm.cdf(d1)
            else:  # put
                option_price = (strike * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) -
                              stock_price * norm.cdf(-d1))
                delta = -norm.cdf(-d1)

            # Calculate other Greeks
            gamma = norm.pdf(d1) / (stock_price * implied_vol * math.sqrt(time_to_expiry))
            theta = (-stock_price * norm.pdf(d1) * implied_vol / (2 * math.sqrt(time_to_expiry)) -
                    risk_free_rate * strike * math.exp(-risk_free_rate * time_to_expiry) *
                    (norm.cdf(d2) if option_type == 'call' else norm.cdf(-d2))) / 365
            vega = stock_price * norm.pdf(d1) * math.sqrt(time_to_expiry) / 100
            rho = (strike * time_to_expiry * math.exp(-risk_free_rate * time_to_expiry) *
                   (norm.cdf(d2) if option_type == 'call' else norm.cdf(-d2))) / 100
        else:
            # Simplified approximation without scipy
            moneyness = stock_price / strike
            if option_type == 'call':
                intrinsic = max(0, stock_price - strike)
                delta = 0.5 + 0.3 * (moneyness - 1)  # Simplified delta
            else:  # put
                intrinsic = max(0, strike - stock_price)
                delta = -0.5 - 0.3 * (1 - moneyness)  # Simplified delta

            time_value = implied_vol * math.sqrt(time_to_expiry) * stock_price * 0.4
            option_price = intrinsic + time_value

            # Simplified Greeks
            gamma = 0.02 * math.exp(-2 * abs(moneyness - 1))
            theta = -option_price / (days_to_expiry * 365) if days_to_expiry > 0 else 0
            vega = option_price * 0.3
            rho = option_price * 0.01 * time_to_expiry

        # Add some realistic bid-ask spread
        spread_pct = 0.02 + 0.01 * abs(1 - moneyness)  # Wider spreads for OTM
        bid = option_price * (1 - spread_pct)
        ask = option_price * (1 + spread_pct)

        # Simulate volume based on moneyness and time to expiry
        base_volume = 100
        volume_factor = max(0.1, 1 - abs(1 - moneyness)) * max(0.1, time_to_expiry)
        volume = int(base_volume * volume_factor * np.random.uniform(0.5, 2.0))

        return {
            'timestamp': date,
            'symbol': symbol,
            'option_symbol': f"{symbol}{expiration.strftime('%y%m%d')}{'C' if option_type == 'call' else 'P'}{int(strike*1000):08d}",
            'strike': strike,
            'expiration': expiration,
            'option_type': option_type,
            'underlying_price': stock_price,
            'open': option_price * np.random.uniform(0.98, 1.02),
            'high': option_price * np.random.uniform(1.0, 1.05),
            'low': option_price * np.random.uniform(0.95, 1.0),
            'close': option_price,
            'volume': volume,
            'bid': max(0.01, bid),
            'ask': ask,
            'implied_volatility': implied_vol,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

    async def _fetch_daily_options_data(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        date: datetime
    ) -> List[Dict]:
        """Fetch options data for a specific day"""
        
        daily_data = []
        
        # Get stock bars for the day
        stock_bars = await self._fetch_stock_bars(session, symbol, date)
        if not stock_bars:
            return []
        
        # Get current stock price
        stock_price = stock_bars[-1]['close'] if stock_bars else 100
        
        # Get option contracts
        contracts = await self._fetch_option_contracts(session, symbol, stock_price, date)
        
        # Instead of fetching market data for each contract individually,
        # batch the requests or use simulated data for training
        if contracts:
            # Filter to most liquid contracts (near the money)
            filtered_contracts = [
                c for c in contracts
                if abs(float(c['strike_price']) - stock_price) / stock_price < 0.10  # Within 10% of stock price (expanded filter)
            ][:150]  # Limit to 150 most relevant contracts per day
            
            # For training purposes, use simulated market data based on contract properties
            # This avoids hundreds of API calls and rate limiting issues
            for contract in filtered_contracts:
                market_data = self._simulate_option_market_data(
                    contract, stock_price, date
                )
                
                option_data = {
                    'timestamp': date,
                    'symbol': symbol,
                    'option_symbol': contract['symbol'],
                    'strike': float(contract['strike_price']),
                    'expiration': contract['expiration_date'],
                    'option_type': contract['type'],
                    'underlying_price': stock_price,
                    **market_data
                }
                daily_data.append(option_data)
        
        return daily_data
    
    async def _async_rate_limit(self):
        """Ensure we don't exceed rate limits (async version)"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    async def _fetch_stock_bars(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        date: datetime
    ) -> List[Dict]:
        """Fetch stock bars for a specific day"""
        
        await self._async_rate_limit()
        
        url = f"{self.data_url}/v2/stocks/{symbol}/bars"
        headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret
        }
        
        params = {
            'start': date.strftime('%Y-%m-%dT00:00:00Z'),
            'end': date.strftime('%Y-%m-%dT23:59:59Z'),
            'timeframe': '1Hour',
            'limit': 1000
        }
        
        try:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    bars = data.get('bars')
                    if bars:
                        return [{
                            'timestamp': bar['t'],
                            'open': bar['o'],
                            'high': bar['h'],
                            'low': bar['l'],
                            'close': bar['c'],
                            'volume': bar['v']
                        } for bar in bars]
                    else:
                        return []
                else:
                    error_text = await response.text()
                    logger.error(f"Error fetching stock bars: {response.status} - {error_text}")
        except Exception as e:
            logger.error(f"Error fetching stock bars: {e}")
            
        return []
    
    async def _fetch_option_contracts(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        stock_price: float,
        date: datetime
    ) -> List[Dict]:
        """Fetch available option contracts"""
        
        await self._async_rate_limit()
        
        # Options contracts are on the trading API, not data API
        url = f"{self.base_url}/v2/options/contracts"
        headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret
        }
        
        # Get contracts expiring in next 60 days
        expiration_start = date
        expiration_end = date + timedelta(days=60)
        
        params = {
            'underlying_symbols': symbol,
            'expiration_date_gte': expiration_start.strftime('%Y-%m-%d'),
            'expiration_date_lte': expiration_end.strftime('%Y-%m-%d'),
            'strike_price_gte': stock_price * 0.90,  # Expanded range for 10% moneyness
            'strike_price_lte': stock_price * 1.10,  # Expanded range for 10% moneyness
            'limit': 500,  # Increased to fetch more contracts
            'status': 'active'
        }
        
        contracts = []
        
        try:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    contracts = data.get('option_contracts', [])
                    logger.debug(f"Found {len(contracts)} option contracts for {symbol}")
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to fetch option contracts: {response.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"Error fetching option contracts: {e}")
        
        return contracts
    
    def _simulate_option_market_data(
        self,
        contract: Dict,
        stock_price: float,
        date: datetime
    ) -> Dict:
        """Simulate realistic option market data for training"""
        import math
        
        strike = float(contract['strike_price'])
        expiration = datetime.strptime(contract['expiration_date'], '%Y-%m-%d')
        days_to_expiry = max(1, (expiration - date).days)
        
        # Calculate moneyness
        moneyness = stock_price / strike
        is_call = contract['type'].lower() == 'call'
        
        # Simple Black-Scholes approximation for option pricing
        # These are realistic approximations for training purposes
        volatility = 0.2 + 0.1 * abs(1 - moneyness)  # Higher vol for OTM options
        time_value = volatility * math.sqrt(days_to_expiry / 365) * stock_price * 0.4
        
        if is_call:
            intrinsic_value = max(0, stock_price - strike)
        else:
            intrinsic_value = max(0, strike - stock_price)
        
        option_price = intrinsic_value + time_value
        
        # Simulate bid-ask spread (wider for less liquid options)
        spread_pct = 0.02 + 0.03 * abs(1 - moneyness)
        bid = option_price * (1 - spread_pct/2)
        ask = option_price * (1 + spread_pct/2)
        
        # Simulate Greeks using simplified formulas
        delta = 0.5 if is_call else -0.5
        if moneyness > 1.1 and is_call:  # ITM call
            delta = 0.7 + 0.3 * min(0.2, moneyness - 1)
        elif moneyness < 0.9 and is_call:  # OTM call
            delta = 0.3 * moneyness
        elif moneyness < 0.9 and not is_call:  # ITM put
            delta = -0.7 - 0.3 * min(0.2, 1 - moneyness)
        elif moneyness > 1.1 and not is_call:  # OTM put
            delta = -0.3 * (2 - moneyness)
        
        # Other Greeks (simplified)
        gamma = 0.02 * math.exp(-2 * abs(1 - moneyness))
        theta = -option_price / days_to_expiry * 0.5
        vega = option_price * 0.3
        rho = option_price * 0.01 * days_to_expiry / 365
        
        return {
            'bid': round(bid, 2),
            'ask': round(ask, 2),
            'last': round((bid + ask) / 2, 2),
            'volume': int(1000 * math.exp(-abs(1 - moneyness) * 5)),  # Higher volume ATM
            'open_interest': int(10000 * math.exp(-abs(1 - moneyness) * 3)),
            'implied_volatility': volatility,
            'delta': round(delta, 4),
            'gamma': round(gamma, 4),
            'theta': round(theta, 4),
            'vega': round(vega, 4),
            'rho': round(rho, 4)
        }
    
    async def _fetch_option_market_data(
        self,
        session: aiohttp.ClientSession,
        option_symbol: str,
        date: datetime
    ) -> Optional[Dict]:
        """Fetch market data for a specific option contract"""
        
        await self._async_rate_limit()
        
        # Get latest quote from v1beta1 API
        quote_url = f"{self.data_url}/v1beta1/options/quotes/latest"
        headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret
        }
        
        params = {
            'symbols': option_symbol
        }
        
        try:
            async with session.get(quote_url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    quotes = data.get('quotes', {})
                    
                    if option_symbol in quotes:
                        quote = quotes[option_symbol]
                        
                        # Calculate Greeks (simplified - in production use proper model)
                        # This is placeholder - real Greeks should come from API or be calculated
                        
                        return {
                            'bid': quote.get('bp', 0),
                            'ask': quote.get('ap', 0),
                            'last': (quote.get('bp', 0) + quote.get('ap', 0)) / 2,
                            'volume': quote.get('bs', 0) + quote.get('as', 0),
                            'open_interest': 0,  # Not available in quotes
                            'implied_volatility': 0.25,  # Placeholder
                            'delta': 0.5,  # Placeholder
                            'gamma': 0.01,  # Placeholder
                            'theta': -0.05,  # Placeholder
                            'vega': 0.1,  # Placeholder
                            'rho': 0.01  # Placeholder
                        }
                else:
                    logger.debug(f"No market data for {option_symbol}: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error fetching option market data: {e}")
        
        return None
    
    def get_training_data(self, symbol: str) -> pd.DataFrame:
        """Get processed training data for a symbol"""
        
        if symbol not in self.historical_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.historical_data[symbol])
        
        if df.empty:
            return df
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Add derived features
        df['moneyness'] = df['underlying_price'] / df['strike']
        df['time_to_expiry'] = (pd.to_datetime(df['expiration']) - df['timestamp']).dt.days / 365
        df['bid_ask_spread'] = df['ask'] - df['bid']
        df['mid_price'] = (df['bid'] + df['ask']) / 2
        
        return df
    
    def get_options_chain(self, symbol: str, date: datetime) -> List[Dict]:
        """Get options chain for a specific date"""
        
        if symbol in self.options_chains and date in self.options_chains[symbol]:
            return self.options_chains[symbol][date]
        
        return []


    async def load_historical_data(
        self, 
        symbols: List[str], 
        start_date: datetime, 
        end_date: datetime,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Main data loading method called by training environment.
        Combines stock and options data for comprehensive market information.
        """
        total_symbols = len(symbols)
        days = (end_date - start_date).days

        # Use print for immediate unbuffered output
        msg = f"\n{'='*80}\n📊 DATA LOADING STARTED\n{'='*80}\n  Symbols: {total_symbols}\n  Date range: {start_date.date()} to {end_date.date()} ({days} days)\n  Estimated time: {2 if days < 180 else 5 if days < 365 else 15}-{5 if days < 180 else 15 if days < 365 else 30} minutes\n{'='*80}\n"
        print(msg, flush=True)
        logger.info(msg)
        sys.stdout.flush()
        sys.stderr.flush()

        result = {}

        try:
            # First, try to load stock data (more reliable and always available)
            msg = "📈 STEP 1/2: Loading stock data..."
            print(msg, flush=True)
            logger.info(msg)
            sys.stdout.flush()

            stock_data = await self.load_historical_stock_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                timeframe="1Hour"
            )

            if stock_data:
                msg = f"\n✅ Stock data loaded for {len(stock_data)}/{total_symbols} symbols\n"
                print(msg, flush=True)
                logger.info(msg)
                sys.stdout.flush()
                sys.stderr.flush()
                result.update(stock_data)
            else:
                msg = "⚠️ No stock data loaded"
                print(msg, flush=True)
                logger.warning(msg)
                sys.stdout.flush()
                sys.stderr.flush()

            # Try to load options data if we have stock data
            if result:
                msg = "📊 STEP 2/2: Loading options data..."
                print(msg, flush=True)
                logger.info(msg)
                sys.stdout.flush()
                sys.stderr.flush()

                try:
                    options_data = await self.load_historical_options_data(
                        symbols=symbols,
                        start_date=start_date,
                        end_date=end_date,
                        use_cache=use_cache
                    )

                    if options_data:
                        msg = f"\n✅ Options data loaded for {len(options_data)}/{total_symbols} symbols\n"
                        print(msg, flush=True)
                        logger.info(msg)
                        sys.stdout.flush()
                        sys.stderr.flush()

                        # Merge options data with stock data
                        for symbol in symbols:
                            if symbol in result and symbol in options_data:
                                # Add options-specific columns to stock data
                                stock_df = result[symbol]
                                if isinstance(stock_df, pd.DataFrame) and not stock_df.empty:
                                    # Add basic options indicators
                                    stock_df['has_options_data'] = True
                                    stock_df['options_volume'] = len(options_data[symbol]) if options_data[symbol] else 0
                                    result[symbol] = stock_df
                    else:
                        msg = "ℹ️ No options data available, using stock data only"
                        print(msg, flush=True)
                        logger.info(msg)
                        sys.stdout.flush()
                        sys.stderr.flush()

                except Exception as e:
                    msg = f"⚠️ Options data loading failed: {e}"
                    print(msg, flush=True)
                    logger.warning(msg)

                    msg = "ℹ️ Continuing with stock data only"
                    print(msg, flush=True)
                    logger.info(msg)
                    sys.stdout.flush()
                    sys.stderr.flush()

            # Validate and clean the data
            msg = "🔍 Validating data quality..."
            print(msg, flush=True)
            logger.info(msg)
            sys.stdout.flush()
            sys.stderr.flush()

            validated_result = {}
            for symbol, data in result.items():
                if isinstance(data, pd.DataFrame) and not data.empty:
                    # Basic data validation
                    if len(data) > 10:  # Minimum data points
                        validated_result[symbol] = data
                        msg = f"  ✅ Validated {len(data)} data points for {symbol}"
                        print(msg, flush=True)
                        logger.debug(msg)
                    else:
                        msg = f"  ⚠️ Insufficient data for {symbol}: {len(data)} points"
                        print(msg, flush=True)
                        logger.warning(msg)
                else:
                    msg = f"  ⚠️ Invalid data format for {symbol}"
                    print(msg, flush=True)
                    logger.warning(msg)

            if validated_result:
                total_rows = sum(len(df) for df in validated_result.values())
                msg = f"\n{'='*80}\n✅ DATA LOADING COMPLETE\n{'='*80}\n  Successfully loaded: {len(validated_result)}/{total_symbols} symbols\n  Total data points: {total_rows:,}\n  Ready for training!\n{'='*80}\n"
                print(msg, flush=True)
                logger.info(msg)
                sys.stdout.flush()
                sys.stderr.flush()
                return validated_result
            else:
                msg = "❌ No valid data loaded for any symbol"
                print(msg, flush=True)
                logger.error(msg)
                sys.stdout.flush()
                sys.stderr.flush()
                return {}

        except Exception as e:
            msg = f"❌ Critical error in data loading: {e}"
            print(msg, flush=True)
            logger.error(msg)
            sys.stdout.flush()
            sys.stderr.flush()
            return {}

class HistoricalOptionsEnvironment:
    """Trading environment that uses real historical options data"""

    def __init__(
        self,
        historical_data: Dict[str, pd.DataFrame] = None,
        data_loader: OptimizedHistoricalOptionsDataLoader = None,
        symbols: List[str] = None,
        initial_capital: float = 100000,
        commission: float = 0.65,
        max_positions: int = 10,
        lookback_window: int = 20,
        episode_length: int = 100
    ):
        self.historical_data = historical_data or {}
        self.data_loader = data_loader
        self.symbols = symbols or list(self.historical_data.keys())
        self.initial_capital = initial_capital
        self.commission = commission
        self.max_positions = max_positions
        self.lookback_window = lookback_window
        self.episode_length = episode_length
        
        # Define action and observation spaces (Gym-style)
        import gymnasium as gym
        from gymnasium import spaces
        
        self.action_space = spaces.Discrete(11)  # 11 different actions
        self.observation_space = spaces.Dict({
            'price_history': spaces.Box(low=0, high=np.inf, shape=(lookback_window, 5), dtype=np.float32),
            'technical_indicators': spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32),
            'options_chain': spaces.Box(low=0, high=np.inf, shape=(20, 15), dtype=np.float32),
            'portfolio_state': spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
            'greeks_summary': spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
            'symbol_encoding': spaces.Box(low=0, high=1, shape=(len(self.symbols),), dtype=np.float32)
        })
        
        # Trading state
        self.reset()
        
    def reset(self):
        """Reset the environment for a new episode"""
        self.capital = self.initial_capital
        self.positions = []
        self.current_symbol = np.random.choice(self.symbols)
        self.done = False
        
        # Get training data for current symbol
        if self.current_symbol in self.historical_data:
            self.training_data = self.historical_data[self.current_symbol].copy()
        elif self.data_loader:
            self.training_data = self.data_loader.get_training_data(self.current_symbol)
        else:
            self.training_data = pd.DataFrame()
        
        # Randomize starting position within the data
        if not self.training_data.empty:
            max_start = max(0, len(self.training_data) - self.episode_length - self.lookback_window)
            if max_start > 0:
                self.current_step = np.random.randint(0, max_start)
            else:
                self.current_step = 0
        else:
            self.current_step = 0
            logger.warning(f"No training data available for {self.current_symbol}")
            self.done = True
            
        return self._get_observation()
    
    def _get_observation(self):
        """Get current market observation"""
        if self.done or self.current_step >= len(self.training_data):
            return None
            
        # Get current underlying price from the current step
        current_row = self.training_data.iloc[self.current_step]
        current_price = current_row.get('underlying_price', 600)
        
        # Generate realistic price history with some volatility
        price_history = []
        base_price = current_price
        
        # Create price history going backwards from current price
        for i in range(self.lookback_window):
            # Add some random walk to create realistic price movement
            volatility = 0.002  # 0.2% daily volatility
            price_change = np.random.normal(0, base_price * volatility)
            base_price = base_price - price_change  # Going backwards in time
            
            # OHLCV format with realistic volume
            high = base_price * (1 + volatility/2)
            low = base_price * (1 - volatility/2)
            volume = np.random.randint(1000000, 5000000)
            
            price_history.insert(0, [base_price, high, low, base_price, volume])
        
        # Make sure the last price matches current price
        if len(price_history) > 0:
            price_history[-1] = [current_price, current_price * 1.001, current_price * 0.999, current_price, 3000000]
            
        # Get current options chain (up to 20 contracts)
        current_idx = self.current_step
        options_chain = []
        
        # Find all options at current timestamp
        current_time = self.training_data.iloc[current_idx]['timestamp']
        current_options = self.training_data[self.training_data['timestamp'] == current_time]
        
        for _, opt in current_options.iterrows():
            option_features = [
                opt.get('strike', 0),
                opt.get('bid', 0),
                opt.get('ask', 0),
                opt.get('last', 0),
                opt.get('volume', 0),
                opt.get('open_interest', 0),
                opt.get('implied_volatility', 0.25),
                opt.get('delta', 0),
                opt.get('gamma', 0),
                opt.get('theta', 0),
                opt.get('vega', 0),
                opt.get('rho', 0),
                1.0 if opt.get('option_type') == 'call' else 0.0,
                30,  # Days to expiry (simplified)
                opt.get('mid_price', 0)
            ]
            options_chain.append(option_features)
            
        # Pad to 20 options
        while len(options_chain) < 20:
            options_chain.append([0] * 15)
            
        # Only keep first 20
        options_chain = options_chain[:20]
        
        # Create symbol encoding
        symbol_encoding = np.zeros(len(self.symbols), dtype=np.float32)
        if self.current_symbol in self.symbols:
            symbol_idx = self.symbols.index(self.current_symbol)
            symbol_encoding[symbol_idx] = 1.0
        
        # Build observation dict
        obs = {
            'price_history': np.array(price_history, dtype=np.float32),
            'technical_indicators': np.zeros(20, dtype=np.float32),  # Placeholder
            'options_chain': np.array(options_chain, dtype=np.float32),
            'portfolio_state': np.array([
                self.capital,
                len(self.positions),
                self._calculate_portfolio_value(),
                0,  # total_pnl
                self.current_step
            ], dtype=np.float32),
            'greeks_summary': np.zeros(5, dtype=np.float32),  # Placeholder
            'symbol_encoding': symbol_encoding
        }
        
        return obs
    
    def step(self, action: int):
        """Execute a trading action"""
        if self.done:
            return None, 0, True, {}
            
        # Get current market data
        current_data = self.training_data.iloc[self.current_step]
        
        # Execute action (simplified for now)
        reward = 0
        if action == 0:  # Hold
            pass
        elif action == 1:  # Buy
            if self.capital > current_data['ask'] * 100 + self.commission:
                # Open position
                self.positions.append({
                    'option_symbol': current_data['option_symbol'],
                    'entry_price': current_data['ask'],
                    'quantity': 1,
                    'entry_step': self.current_step
                })
                self.capital -= current_data['ask'] * 100 + self.commission
                reward = -0.1  # Small penalty for transaction cost
        elif action == 2:  # Sell/Close
            if self.positions:
                # Close oldest position
                pos = self.positions.pop(0)
                exit_price = current_data['bid']
                pnl = (exit_price - pos['entry_price']) * 100 - self.commission
                self.capital += exit_price * 100 - self.commission
                reward = pnl / 100  # Normalize reward
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        if self.current_step >= len(self.training_data) - 1:
            self.done = True
            
        obs = self._get_observation()
        info = {
            'portfolio_value': self._calculate_portfolio_value(),
            'positions': len(self.positions),
            'symbol': self.current_symbol,
            'date': self.training_data.iloc[min(self.current_step, len(self.training_data)-1)]['timestamp']
        }
        
        return obs, reward, self.done, info
    
    def _calculate_portfolio_value(self):
        """Calculate total portfolio value"""
        positions_value = 0
        if self.positions and self.current_step < len(self.training_data):
            current_price = self.training_data.iloc[self.current_step]['mid_price']
            positions_value = len(self.positions) * current_price * 100
            
        return self.capital + positions_value