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
    class MockAlpacaClient:
        def __init__(self, *args, **kwargs):
            pass

    tradeapi = type('MockTradeAPI', (), {'REST': MockAlpacaClient})()
    StockHistoricalDataClient = MockAlpacaClient
    OptionHistoricalDataClient = MockAlpacaClient
    StockBarsRequest = MockAlpacaClient
    OptionBarsRequest = MockAlpacaClient
    OptionChainRequest = MockAlpacaClient
    TimeFrame = type('MockTimeFrame', (), {'Hour': '1Hour', 'Day': '1Day'})()
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging
import asyncio
import aiohttp
import json
import pickle
import os
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

        for symbol in symbols:
            try:
                cache_key = self._get_cache_key(symbol, start_date, end_date, f"stock_{timeframe}")
                cache_path = os.path.join(self.cache_dir, 'stocks', f"{cache_key}.pkl")

                # Check cache first
                if self._is_cache_valid(cache_key) and os.path.exists(cache_path):
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                    logger.info(f"Loaded {symbol} stock data from cache")
                    result[symbol] = data
                    continue

                # Fetch from API
                self._rate_limit()

                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Hour if timeframe == "1Hour" else TimeFrame.Day,
                    start=start_date,
                    end=end_date
                )

                bars = self.stock_data_client.get_stock_bars(request)

                if bars.df.empty:
                    logger.warning(f"No stock data returned for {symbol}")
                    continue

                # Process and validate data
                data = bars.df.reset_index()
                data['symbol'] = symbol

                # Validate data quality
                quality_metrics = self._validate_data_quality(data, f"{symbol}_stock")

                if quality_metrics.quality_score < 0.3:
                    logger.warning(f"Low quality stock data for {symbol}: {quality_metrics.quality_score:.2f}")
                else:
                    logger.info(f"Good quality stock data for {symbol}: {quality_metrics.quality_score:.2f}")

                # Cache the data
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
                logger.info(f"Loaded {len(data)} stock bars for {symbol}")

            except Exception as e:
                logger.error(f"Error loading stock data for {symbol}: {e}")
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

        logger.info(f"Loading historical options data for {symbols} from {start_date.date()} to {end_date.date()}")

        result = {}

        # First, load stock data for underlying prices
        stock_data = await self.load_historical_stock_data(symbols, start_date, end_date)

        # Process each symbol
        for sym in symbols:
            try:
                cache_key = self._get_cache_key(sym, start_date, end_date, "options")
                cache_path = os.path.join(self.cache_dir, 'options', f"{cache_key}.pkl")

                # Check cache first
                if use_cache and self._is_cache_valid(cache_key) and os.path.exists(cache_path):
                    with open(cache_path, 'rb') as f:
                        symbol_data = pickle.load(f)
                    logger.info(f"Loaded {len(symbol_data)} options records from cache for {sym}")
                    result[sym] = symbol_data
                    continue

                # Generate options data (real or simulated)
                symbol_data = []

                if self.has_options_data and sym in stock_data:
                    # Try to fetch real options data
                    try:
                        symbol_data = await self._fetch_real_options_data(
                            sym, start_date, end_date, stock_data[sym]
                        )
                    except Exception as e:
                        logger.warning(f"Failed to fetch real options data for {sym}: {e}")
                        symbol_data = []

                # Fall back to simulated data if real data unavailable
                if not symbol_data and sym in stock_data:
                    logger.info(f"Generating simulated options data for {sym}")
                    symbol_data = await self._generate_simulated_options_data(
                        sym, start_date, end_date, stock_data[sym]
                    )

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
                    logger.info(f"Loaded {len(symbol_data)} options records for {sym}")
                else:
                    logger.warning(f"No options data available for {sym}")

            except Exception as e:
                logger.error(f"Error processing options data for {sym}: {e}")
                continue

        # Store in instance variable for later use
        with self._data_lock:
            for sym, data in result.items():
                self.historical_data[sym] = data

        self._save_cache_index()
        logger.info(f"Loaded historical options data for {len(result)} symbols")
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
            chain_request = OptionChainRequest(
                underlying_symbol=symbol,
                expiration_date_gte=start_date.date(),
                expiration_date_lte=(end_date + timedelta(days=45)).date()  # Include options expiring after our date range
            )

            # Use sync rate limiting since this is calling sync API
            self._rate_limit()
            options_chain = self.options_data_client.get_option_chain(chain_request)

            if not options_chain or options_chain.empty:
                logger.warning(f"No options chain data for {symbol}")
                return []

            # For each day in our range, get options data
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

                # Filter options chain for relevant strikes and expirations
                relevant_options = self._filter_options_chain(
                    options_chain, stock_price, current_date
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

                        self._rate_limit()
                        bars = self.options_data_client.get_option_bars(bars_request)

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

        except Exception as e:
            logger.error(f"Error fetching real options data for {symbol}: {e}")

        return options_data

    async def _generate_simulated_options_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        stock_data: pd.DataFrame
    ) -> List[Dict]:
        """Generate realistic simulated options data based on stock data"""
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

    def _filter_options_chain(self, options_chain, stock_price: float, current_date: datetime) -> List[Dict]:
        """Filter options chain for relevant strikes and expirations"""
        filtered_options = []

        try:
            # Filter by strike price (within ±20% of current stock price)
            min_strike = stock_price * 0.8
            max_strike = stock_price * 1.2

            # Filter by expiration (7-45 days from current date)
            min_expiry = current_date + timedelta(days=7)
            max_expiry = current_date + timedelta(days=45)

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
                if abs(float(c['strike_price']) - stock_price) / stock_price < 0.07  # Within 7% of stock price (tighter filter)
            ][:100]  # Limit to 100 most relevant contracts per day
            
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
            'strike_price_gte': stock_price * 0.93,  # Tighter range for 7% moneyness
            'strike_price_lte': stock_price * 1.07,  # Tighter range for 7% moneyness
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
        logger.info(f"Loading historical data for {len(symbols)} symbols from {start_date.date()} to {end_date.date()}")
        
        result = {}
        
        try:
            # First, try to load stock data (more reliable and always available)
            logger.info("Loading stock data...")
            stock_data = await self.load_historical_stock_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                timeframe="1Hour"
            )
            
            if stock_data:
                logger.info(f"✅ Loaded stock data for {len(stock_data)} symbols")
                result.update(stock_data)
            else:
                logger.warning("⚠️ No stock data loaded")
            
            # Try to load options data if we have stock data
            if result:
                logger.info("Attempting to load options data...")
                try:
                    options_data = await self.load_historical_options_data(
                        symbols=symbols,
                        start_date=start_date,
                        end_date=end_date,
                        use_cache=use_cache
                    )
                    
                    if options_data:
                        logger.info(f"✅ Loaded options data for {len(options_data)} symbols")
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
                        logger.info("ℹ️ No options data available, using stock data only")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Options data loading failed: {e}")
                    logger.info("ℹ️ Continuing with stock data only")
            
            # Validate and clean the data
            validated_result = {}
            for symbol, data in result.items():
                if isinstance(data, pd.DataFrame) and not data.empty:
                    # Basic data validation
                    if len(data) > 10:  # Minimum data points
                        validated_result[symbol] = data
                        logger.debug(f"✅ Validated {len(data)} data points for {symbol}")
                    else:
                        logger.warning(f"⚠️ Insufficient data for {symbol}: {len(data)} points")
                else:
                    logger.warning(f"⚠️ Invalid data format for {symbol}")
            
            if validated_result:
                logger.info(f"✅ Successfully loaded data for {len(validated_result)} symbols")
                return validated_result
            else:
                logger.error("❌ No valid data loaded for any symbol")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Critical error in data loading: {e}")
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