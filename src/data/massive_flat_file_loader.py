"""
Massive Flat Files Data Loader

Loads preprocessed options data downloaded from Massive.com flat files.
The data has already been processed to include Greeks calculated from option prices.

This loader is designed to work with the OptionsEnv training environment and
ensures consistency with live/paper trading by using the same Greeks calculation.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MassiveFlatFileLoader:
    """
    Load preprocessed options data from Massive flat files.
    
    The data structure matches what's expected by OptionsEnv:
    - Options have Greeks (delta, gamma, theta, vega) calculated using GreeksCalculator
    - Underlying prices are from yfinance
    - Time to expiry and IV are properly calculated
    """
    
    def __init__(
        self,
        data_dir: str = 'data/flat_files_processed',
        cache_in_memory: bool = True
    ):
        """
        Initialize the loader
        
        Args:
            data_dir: Directory containing processed parquet files
            cache_in_memory: Whether to cache loaded data
        """
        self.data_dir = Path(data_dir)
        self.cache_in_memory = cache_in_memory
        self._cache = {}
        
        logger.info(f"📁 MassiveFlatFileLoader initialized")
        logger.info(f"   Data directory: {self.data_dir}")
    
    def get_available_files(self) -> List[Path]:
        """List all available parquet files"""
        if not self.data_dir.exists():
            return []
        return sorted(self.data_dir.glob("*.parquet"))
    
    def load_all_data(self) -> pd.DataFrame:
        """
        Load all available parquet files and combine into single DataFrame
        
        Returns:
            DataFrame with all options data
        """
        if 'all_data' in self._cache:
            return self._cache['all_data']
        
        files = self.get_available_files()
        if not files:
            logger.warning(f"No parquet files found in {self.data_dir}")
            return pd.DataFrame()
        
        logger.info(f"Loading {len(files)} parquet files...")
        
        dfs = []
        for f in files:
            df = pd.read_parquet(f)
            dfs.append(df)
            logger.info(f"  Loaded {f.name}: {len(df):,} records")
        
        combined = pd.concat(dfs, ignore_index=True)
        
        # Ensure proper column names for compatibility
        combined = self._standardize_columns(combined)
        
        if self.cache_in_memory:
            self._cache['all_data'] = combined
        
        logger.info(f"✅ Loaded total {len(combined):,} records")
        return combined
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names for compatibility with OptionsEnv"""
        # Map from Massive flat file format to internal format
        column_mapping = {
            'trade_date': 'timestamp',
            'underlying': 'symbol',
            'expiry': 'expiration',
            'option_price': 'mid_price',
            'iv': 'implied_volatility'
        }
        
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Convert dates to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        if 'expiration' in df.columns:
            df['expiration'] = pd.to_datetime(df['expiration'])
        
        return df
    
    def load_data_for_symbol(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Load options data for a specific underlying symbol
        
        Args:
            symbol: Underlying symbol (e.g., 'SPY')
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame with options data for the symbol
        """
        df = self.load_all_data()
        
        if df.empty:
            return df
        
        # Filter by symbol
        df = df[df['symbol'] == symbol].copy()
        
        # Filter by date range
        if start_date is not None:
            df = df[df['timestamp'] >= pd.Timestamp(start_date)]
        if end_date is not None:
            df = df[df['timestamp'] <= pd.Timestamp(end_date)]
        
        logger.info(f"✅ Loaded {len(df):,} options for {symbol}")
        return df
    
    def get_unique_dates(self) -> List[datetime]:
        """Get list of unique trading dates in the data"""
        df = self.load_all_data()
        if df.empty:
            return []
        return sorted(df['timestamp'].dt.date.unique())
    
    def get_unique_symbols(self) -> List[str]:
        """Get list of unique underlying symbols"""
        df = self.load_all_data()
        if df.empty:
            return []
        return sorted(df['symbol'].unique())

    def get_options_for_date(
        self,
        date: datetime,
        symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get all options contracts available on a specific date

        Args:
            date: Trading date
            symbol: Optional symbol filter

        Returns:
            DataFrame with options for that date
        """
        df = self.load_all_data()

        if df.empty:
            return df

        # Filter by date
        date_ts = pd.Timestamp(date).normalize()
        df = df[df['timestamp'].dt.normalize() == date_ts]

        # Filter by symbol if specified
        if symbol:
            df = df[df['symbol'] == symbol]

        return df.copy()

    def get_options_chain(
        self,
        date: datetime,
        symbol: str,
        expiration: Optional[datetime] = None,
        min_delta: float = -1.0,
        max_delta: float = 1.0
    ) -> pd.DataFrame:
        """
        Get options chain for a symbol on a date

        Args:
            date: Trading date
            symbol: Underlying symbol
            expiration: Optional expiration filter
            min_delta: Minimum delta filter
            max_delta: Maximum delta filter

        Returns:
            DataFrame with options chain
        """
        df = self.get_options_for_date(date, symbol)

        if df.empty:
            return df

        # Filter by expiration
        if expiration:
            exp_ts = pd.Timestamp(expiration).normalize()
            df = df[df['expiration'].dt.normalize() == exp_ts]

        # Filter by delta
        df = df[(df['delta'] >= min_delta) & (df['delta'] <= max_delta)]

        return df.sort_values(['expiration', 'strike'])

    def load_options_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Load options data in list format (compatible with FlatFileDataLoader)

        Args:
            symbol: Underlying symbol
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            List of option contract dicts
        """
        df = self.load_data_for_symbol(symbol, start_date, end_date)
        return df.to_dict('records')

    async def load_historical_options_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True
    ) -> Dict[str, List[Dict]]:
        """
        Async loader for compatibility with existing code

        Args:
            symbols: List of underlying symbols
            start_date: Start date
            end_date: End date
            use_cache: Whether to use cache (ignored, always uses cache)

        Returns:
            Dict mapping symbol to list of options
        """
        result = {}

        for symbol in symbols:
            options = self.load_options_data(symbol, start_date, end_date)
            if options:
                result[symbol] = options

        return result

    def get_data_summary(self) -> Dict:
        """Get summary statistics about the loaded data"""
        df = self.load_all_data()

        if df.empty:
            return {"error": "No data loaded"}

        return {
            "total_records": len(df),
            "symbols": self.get_unique_symbols(),
            "date_range": {
                "start": df['timestamp'].min().strftime('%Y-%m-%d'),
                "end": df['timestamp'].max().strftime('%Y-%m-%d')
            },
            "unique_dates": len(self.get_unique_dates()),
            "iv_stats": {
                "mean": df['implied_volatility'].mean(),
                "std": df['implied_volatility'].std(),
                "min": df['implied_volatility'].min(),
                "max": df['implied_volatility'].max()
            },
            "delta_stats": {
                "calls": df[df['option_type'] == 'call']['delta'].describe().to_dict(),
                "puts": df[df['option_type'] == 'put']['delta'].describe().to_dict()
            }
        }

    def clear_cache(self):
        """Clear the in-memory cache"""
        self._cache.clear()
        logger.info("🗑️ Cleared cache")

    async def load_historical_stock_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """
        Load historical stock data (underlying prices) from the options data.

        Since our options data includes underlying_price for each trade date,
        we can extract daily stock prices from it.

        Note: If requested date range doesn't overlap with available data,
        we use all available data instead to maximize training data.

        Args:
            symbols: List of stock symbols
            start_date: Start date (may be ignored if no overlap with data)
            end_date: End date (may be ignored if no overlap with data)

        Returns:
            Dict mapping symbol to DataFrame with stock OHLCV data
        """
        df = self.load_all_data()

        if df.empty:
            logger.warning("No options data loaded, returning empty stock data")
            return {}

        # Get actual data range
        data_start = df['timestamp'].min()
        data_end = df['timestamp'].max()

        # Check if requested range overlaps with data
        req_start = pd.Timestamp(start_date)
        req_end = pd.Timestamp(end_date)

        # Calculate overlap
        if req_end < data_start or req_start > data_end:
            # No overlap at all - use all available data
            overlap_days = 0
        else:
            overlap_start = max(req_start, data_start)
            overlap_end = min(req_end, data_end)
            overlap_days = (overlap_end - overlap_start).days

        # Calculate requested days
        requested_days = (req_end - req_start).days
        available_days = (data_end - data_start).days

        # If overlap is less than 50% of requested OR less than 50% of available, use all data
        if overlap_days < requested_days * 0.5 or overlap_days < available_days * 0.5:
            logger.info(f"📅 Requested dates ({start_date.date()} to {end_date.date()}) have limited overlap with data ({data_start.date()} to {data_end.date()})")
            logger.info(f"   Overlap: {overlap_days} days, Available: {available_days} days")
            logger.info(f"   Using ALL available data to maximize training data")
            use_start = data_start
            use_end = data_end
        else:
            # Good overlap - use intersection
            use_start = max(req_start, data_start)
            use_end = min(req_end, data_end)
            logger.info(f"📅 Using data from {use_start.date()} to {use_end.date()}")

        result = {}

        for symbol in symbols:
            symbol_df = df[df['symbol'] == symbol].copy()

            if symbol_df.empty:
                logger.warning(f"No data for symbol {symbol}")
                continue

            # Filter by adjusted date range
            symbol_df = symbol_df[
                (symbol_df['timestamp'] >= use_start) &
                (symbol_df['timestamp'] <= use_end)
            ]

            if symbol_df.empty:
                logger.warning(f"No data for {symbol} in date range")
                continue

            # Extract daily stock data from options records
            # Group by date and get the underlying_price (should be same for all options on same day)
            stock_df = symbol_df.groupby(symbol_df['timestamp'].dt.date).agg({
                'underlying_price': 'first',  # Same for all options on the day
                'timestamp': 'first'
            }).reset_index(drop=True)

            # Create OHLCV-like structure (we only have close price, so use it for all)
            stock_df = stock_df.rename(columns={'underlying_price': 'close'})
            stock_df['open'] = stock_df['close']
            stock_df['high'] = stock_df['close']
            stock_df['low'] = stock_df['close']
            stock_df['volume'] = 1000000  # Placeholder volume
            stock_df['symbol'] = symbol

            # Reorder columns
            stock_df = stock_df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
            stock_df = stock_df.sort_values('timestamp').reset_index(drop=True)

            result[symbol] = stock_df
            logger.info(f"✅ Loaded {len(stock_df)} days of stock data for {symbol}")

        return result

    async def load_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1Day'
    ) -> Dict[str, pd.DataFrame]:
        """
        Main data loading method (compatible with OptimizedHistoricalOptionsDataLoader)

        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            timeframe: Timeframe (ignored for flat files)

        Returns:
            Dict mapping symbol to DataFrame
        """
        return await self.load_historical_stock_data(symbols, start_date, end_date)

