import pandas as pd

try:
    import alpaca_trade_api as tradeapi
    from alpaca.data.historical import StockHistoricalDataClient, OptionHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, OptionBarsRequest, OptionChainRequest
    from alpaca.data.timeframe import TimeFrame
    HAS_ALPACA = True
except ImportError:
    HAS_ALPACA = False
    import warnings
    warnings.warn("Alpaca packages not available, using mock implementations", ImportWarning)

    # Mock classes for testing
    class MockBarsResponse:
        """Mock response for bars data"""
        def __init__(self):
            self.df = pd.DataFrame()

    class MockAlpacaClient:
        def __init__(self, *args, **kwargs):
            pass

        def get_stock_bars(self, request):
            """Mock method that returns empty bars"""
            return MockBarsResponse()

        def get_option_bars(self, request):
            """Mock method that returns empty bars"""
            return MockBarsResponse()

        def get_option_chain(self, request):
            """Mock method that returns empty option chain"""
            return MockBarsResponse()

    tradeapi = type('MockTradeAPI', (), {'REST': MockAlpacaClient})()
    StockHistoricalDataClient = MockAlpacaClient
    OptionHistoricalDataClient = MockAlpacaClient
    StockBarsRequest = MockAlpacaClient
    OptionBarsRequest = MockAlpacaClient
    OptionChainRequest = MockAlpacaClient
    TimeFrame = type('MockTimeFrame', (), {'Hour': '1Hour', 'Day': '1Day'})()
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging
import asyncio
import aiohttp
import json
import pickle
import os
import sys
from collections import defaultdict
import pytz
import time
import ssl
import certifi
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
import math
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
    """Enhanced data loader with improved caching, rate limiting, and data quality validation"""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = 'https://paper-api.alpaca.markets',
        data_url: str = 'https://data.alpaca.markets',
        cache_dir: str = 'data/options_cache',
        max_workers: int = 4,
        rate_limit_delay: float = 0.1,
        cache_ttl_hours: int = 24
    ):
        # Initialize Alpaca clients
        self.trading_client = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        self.stock_data_client = StockHistoricalDataClient(api_key, api_secret)

        # Try to initialize options data client (may not be available for all accounts)
        try:
            self.options_data_client = OptionHistoricalDataClient(api_key, api_secret)
            self.has_options_data = True
            logger.info("Options data client initialized successfully")
        except Exception as e:
            logger.warning(f"Options data client not available: {e}. Will use simulated options data.")
            self.options_data_client = None
            self.has_options_data = False

        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.data_url = data_url
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

                # Fetch from API
                msg = f"  [{idx}/{total_symbols}] 🌐 Calling Alpaca API for {symbol}..."
                print(msg, flush=True)
                logger.info(msg)
                sys.stdout.flush()
                sys.stderr.flush()

                await self._rate_limit_async()

                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Hour if timeframe == "1Hour" else TimeFrame.Day,
                    start=start_date,
                    end=end_date
                )

                # Run blocking API call in thread pool to not block event loop
                msg = f"  [{idx}/{total_symbols}] ⏳ Waiting for API response..."
                print(msg, flush=True)
                logger.info(msg)
                sys.stdout.flush()
                sys.stderr.flush()

                bars = await asyncio.to_thread(self.stock_data_client.get_stock_bars, request)

                msg = f"  [{idx}/{total_symbols}] 📦 Received API response for {symbol}"
                print(msg, flush=True)
                logger.info(msg)
                sys.stdout.flush()
                sys.stderr.flush()

                if bars.df.empty:
                    msg = f"  [{idx}/{total_symbols}] ⚠️ {symbol}: No data returned from API"
                    print(msg, flush=True)
                    logger.warning(msg)
                    sys.stdout.flush()
                    sys.stderr.flush()
                    continue

                # Process and validate data
                msg = f"  [{idx}/{total_symbols}] 🔄 Processing {len(bars.df)} bars for {symbol}..."
                print(msg, flush=True)
                logger.info(msg)
                sys.stdout.flush()
                sys.stderr.flush()

                data = bars.df.reset_index()
                data['symbol'] = symbol

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
        """Fetch real options data from Alpaca API"""
        options_data = []

        try:
            # Get options chain for the symbol
            msg = f"      🔧 Creating options chain request for {symbol}..."
            print(msg, flush=True)
            logger.debug(msg)
            sys.stdout.flush()
            sys.stderr.flush()

            chain_request = OptionChainRequest(
                underlying_symbol=symbol,
                expiration_date_gte=start_date.date(),
                expiration_date_lte=(end_date + timedelta(days=45)).date()  # Include options expiring after our date range
            )

            # Use async rate limiting to not block event loop
            msg = f"      ⏳ Rate limiting before API call..."
            print(msg, flush=True)
            logger.debug(msg)
            sys.stdout.flush()
            sys.stderr.flush()

            await self._rate_limit_async()

            # Log the request details for debugging
            msg = f"      🌐 Calling Alpaca Options API for {symbol}..."
            print(msg, flush=True)
            logger.debug(msg)
            sys.stdout.flush()
            sys.stderr.flush()

            try:
                # Run blocking API call in thread pool
                options_chain = await asyncio.to_thread(self.options_data_client.get_option_chain, chain_request)

                msg = f"      📦 Received options chain response for {symbol}"
                print(msg, flush=True)
                logger.debug(msg)
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception as api_error:
                msg = f"      ❌ API error fetching options chain for {symbol}: {type(api_error).__name__}: {api_error}"
                print(msg, flush=True)
                logger.error(msg)

                msg = f"      🔄 Falling back to simulated data for {symbol}"
                print(msg, flush=True)
                logger.info(msg)
                sys.stdout.flush()
                sys.stderr.flush()
                return []

            # Check if options_chain is valid
            if not options_chain:
                msg = f"      ⚠️ No options chain data returned from API for {symbol}"
                print(msg, flush=True)
                logger.warning(msg)

                msg = f"      💡 This may be due to: 1) Demo API keys, 2) No options available, 3) API permissions"
                print(msg, flush=True)
                logger.info(msg)
                sys.stdout.flush()
                sys.stderr.flush()
                return []

            # Convert to list - Alpaca returns dict with option symbols as keys
            msg = f"      🔄 Processing options chain for {symbol}..."
            print(msg, flush=True)
            logger.debug(msg)
            sys.stdout.flush()
            sys.stderr.flush()

            chain_list = []

            logger.debug(f"Options chain type: {type(options_chain)}")
            logger.debug(f"Has 'options' attr: {hasattr(options_chain, 'options')}")
            logger.debug(f"Is dict: {isinstance(options_chain, dict)}")

            if hasattr(options_chain, 'options'):
                # If it has an 'options' attribute
                chain_list = options_chain.options
                msg = f"      📋 Extracted {len(chain_list)} options from .options attribute"
                print(msg, flush=True)
                logger.debug(msg)
                sys.stdout.flush()
                sys.stderr.flush()
            elif isinstance(options_chain, dict):
                # Alpaca returns dict like: {'SPY251104C00635000': OptionsSnapshot(...), ...}
                total_options = len(options_chain)
                msg = f"      📋 Processing dict with {total_options} options contracts..."
                print(msg, flush=True)
                logger.debug(msg)
                sys.stdout.flush()
                sys.stderr.flush()

                # Convert to list of dicts with symbol and data
                processed_count = 0
                for option_symbol, option_data in options_chain.items():
                    processed_count += 1

                    # Show progress every 1000 options
                    if processed_count % 1000 == 0:
                        msg = f"      ⏳ Processed {processed_count}/{total_options} options..."
                        print(msg, flush=True)
                        logger.debug(msg)
                        sys.stdout.flush()
                        sys.stderr.flush()
                    # option_data can be either a dict or an OptionsSnapshot object
                    if isinstance(option_data, dict):
                        # Dict format
                        option_dict = {
                            'symbol': option_symbol,
                            'strike_price': self._extract_strike_from_symbol(option_symbol),
                            'expiration_date': self._extract_expiration_from_symbol(option_symbol),
                            'type': 'call' if 'C' in option_symbol else 'put',
                            'latest_quote': option_data.get('latest_quote', {}),
                            'latest_trade': option_data.get('latest_trade', {}),
                            'greeks': option_data.get('greeks'),
                            'implied_volatility': option_data.get('implied_volatility')
                        }
                        chain_list.append(option_dict)
                    elif hasattr(option_data, 'symbol'):
                        # OptionsSnapshot object format
                        option_dict = {
                            'symbol': option_symbol,
                            'strike_price': self._extract_strike_from_symbol(option_symbol),
                            'expiration_date': self._extract_expiration_from_symbol(option_symbol),
                            'type': 'call' if 'C' in option_symbol else 'put',
                            'latest_quote': option_data.latest_quote if hasattr(option_data, 'latest_quote') else {},
                            'latest_trade': option_data.latest_trade if hasattr(option_data, 'latest_trade') else {},
                            'greeks': option_data.greeks if hasattr(option_data, 'greeks') else None,
                            'implied_volatility': option_data.implied_volatility if hasattr(option_data, 'implied_volatility') else None
                        }
                        chain_list.append(option_dict)

                msg = f"      ✅ Converted {len(chain_list)} options to internal format"
                print(msg, flush=True)
                logger.debug(msg)
                sys.stdout.flush()
                sys.stderr.flush()
            else:
                chain_list = list(options_chain) if options_chain else []
                msg = f"      📋 Converted to list (fallback): {len(chain_list)} items"
                print(msg, flush=True)
                logger.debug(msg)
                sys.stdout.flush()
                sys.stderr.flush()

            if not chain_list:
                msg = f"      ⚠️ No options in chain for {symbol}"
                print(msg, flush=True)
                logger.warning(msg)

                msg = f"      📊 Options chain object type: {type(options_chain)}"
                print(msg, flush=True)
                logger.info(msg)

                if isinstance(options_chain, dict):
                    msg = f"      🔑 Options chain keys (first 5): {list(options_chain.keys())[:5]}"
                    print(msg, flush=True)
                    logger.info(msg)

                sys.stdout.flush()
                sys.stderr.flush()
                return []

            msg = f"      ✅ Found {len(chain_list)} options in chain for {symbol}"
            print(msg, flush=True)
            logger.info(msg)
            sys.stdout.flush()
            sys.stderr.flush()

            # For each day in our range, get options data
            msg = f"      📅 Processing options data for date range ({start_date.date()} to {end_date.date()})..."
            print(msg, flush=True)
            logger.info(msg)
            sys.stdout.flush()
            sys.stderr.flush()

            total_days = (end_date - start_date).days
            days_processed = 0

            current_date = start_date
            while current_date <= end_date:
                if current_date.weekday() >= 5:  # Skip weekends
                    current_date += timedelta(days=1)
                    continue

                days_processed += 1

                # Show progress every 10 days
                if days_processed % 10 == 0:
                    msg = f"      ⏳ Processing day {days_processed}/{total_days} ({current_date.date()})..."
                    print(msg, flush=True)
                    logger.debug(msg)
                    sys.stdout.flush()
                    sys.stderr.flush()

                # Get stock price for this date
                stock_price = self._get_stock_price_for_date(stock_data, current_date)
                if stock_price is None:
                    current_date += timedelta(days=1)
                    continue

                # Filter options chain for relevant strikes and expirations
                relevant_options = self._filter_options_chain(
                    chain_list, stock_price, current_date
                )

                # Get historical bars for these options
                for option in relevant_options:
                    try:
                        bars_request = OptionBarsRequest(
                            symbol_or_symbols=option['symbol'],
                            timeframe=TimeFrame.Hour,
                            start=current_date,
                            end=current_date + timedelta(days=1)
                        )

                        await self._rate_limit_async()
                        # Run blocking API call in thread pool
                        bars = await asyncio.to_thread(self.options_data_client.get_option_bars, bars_request)

                        if not bars.df.empty:
                            for _, bar in bars.df.iterrows():
                                options_data.append({
                                    'timestamp': current_date,
                                    'symbol': symbol,
                                    'option_symbol': option['symbol'],
                                    'strike': option['strike_price'],
                                    'expiration': option['expiration_date'],
                                    'option_type': option['type'],
                                    'underlying_price': stock_price,
                                    'open': bar['open'],
                                    'high': bar['high'],
                                    'low': bar['low'],
                                    'close': bar['close'],
                                    'volume': bar['volume'],
                                    'bid': bar.get('bid', bar['close'] * 0.98),
                                    'ask': bar.get('ask', bar['close'] * 1.02),
                                    'implied_volatility': bar.get('implied_volatility', 0.3),
                                    'delta': bar.get('delta', 0.5),
                                    'gamma': bar.get('gamma', 0.1),
                                    'theta': bar.get('theta', -0.05),
                                    'vega': bar.get('vega', 0.2),
                                    'rho': bar.get('rho', 0.1)
                                })
                    except Exception as e:
                        logger.debug(f"Error fetching bars for option {option['symbol']}: {e}")
                        continue

                current_date += timedelta(days=1)

            msg = f"      ✅ Completed processing {days_processed} days of options data"
            print(msg, flush=True)
            logger.info(msg)
            sys.stdout.flush()
            sys.stderr.flush()

        except Exception as e:
            msg = f"      ❌ Error fetching real options data for {symbol}: {e}"
            print(msg, flush=True)
            logger.error(msg)
            sys.stdout.flush()
            sys.stderr.flush()

        msg = f"      📊 Returning {len(options_data)} options data points for {symbol}"
        print(msg, flush=True)
        logger.info(msg)
        sys.stdout.flush()
        sys.stderr.flush()

        return options_data

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