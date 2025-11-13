"""
Flat File Data Loader for Historical Stock and Options Data

This module provides fast data loading from pre-downloaded flat files (CSV/Parquet)
instead of making REST API calls. This is much faster for training and allows offline work.

File Structure:
    data/flat_files/
        stocks/
            SPY.parquet (or SPY.csv)
            QQQ.parquet
            ...
        options/
            SPY_options.parquet (or SPY_options.csv)
            QQQ_options.parquet
            ...

Data Format:
    Stock data columns: timestamp, symbol, open, high, low, close, volume
    Options data columns: timestamp, symbol, strike, expiration, option_type, bid, ask, 
                          last, volume, open_interest, underlying_price, delta, gamma, 
                          theta, vega, rho, implied_volatility
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pickle
import json

logger = logging.getLogger(__name__)

# Try to import pyarrow for parquet support
try:
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False
    logger.warning("pyarrow not installed. Parquet files not supported. Install with: pip install pyarrow")


class FlatFileDataLoader:
    """Load historical stock and options data from flat files"""
    
    def __init__(
        self,
        data_dir: str = 'data/flat_files',
        file_format: str = 'parquet',  # 'parquet' or 'csv'
        cache_in_memory: bool = True
    ):
        """
        Initialize flat file data loader
        
        Args:
            data_dir: Root directory containing flat files
            file_format: File format ('parquet' or 'csv')
            cache_in_memory: Whether to cache loaded data in memory
        """
        self.data_dir = Path(data_dir)
        self.stocks_dir = self.data_dir / 'stocks'
        self.options_dir = self.data_dir / 'options'
        self.file_format = file_format
        self.cache_in_memory = cache_in_memory
        
        # In-memory cache
        self.stock_cache = {}
        self.options_cache = {}
        
        # Create directories if they don't exist
        self.stocks_dir.mkdir(parents=True, exist_ok=True)
        self.options_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate file format
        if file_format == 'parquet' and not HAS_PARQUET:
            logger.warning("Parquet format requested but pyarrow not installed. Falling back to CSV.")
            self.file_format = 'csv'
        
        logger.info(f"📁 Initialized FlatFileDataLoader")
        logger.info(f"   Data directory: {self.data_dir}")
        logger.info(f"   File format: {self.file_format}")
        logger.info(f"   Cache in memory: {self.cache_in_memory}")
    
    def _get_stock_file_path(self, symbol: str) -> Path:
        """Get file path for stock data"""
        extension = 'parquet' if self.file_format == 'parquet' else 'csv'
        return self.stocks_dir / f"{symbol}.{extension}"
    
    def _get_options_file_path(self, symbol: str) -> Path:
        """Get file path for options data"""
        extension = 'parquet' if self.file_format == 'parquet' else 'csv'
        return self.options_dir / f"{symbol}_options.{extension}"
    
    def load_stock_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Load stock data from flat file
        
        Args:
            symbol: Stock symbol (e.g., 'SPY')
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame with columns: timestamp, symbol, open, high, low, close, volume
        """
        # Check cache first
        if self.cache_in_memory and symbol in self.stock_cache:
            df = self.stock_cache[symbol]
        else:
            file_path = self._get_stock_file_path(symbol)
            
            if not file_path.exists():
                logger.warning(f"Stock data file not found: {file_path}")
                return pd.DataFrame()
            
            # Load from file
            if self.file_format == 'parquet':
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path, parse_dates=['timestamp'])
            
            # Cache if enabled
            if self.cache_in_memory:
                self.stock_cache[symbol] = df
        
        # Filter by date range if specified
        if start_date is not None or end_date is not None:
            df = df.copy()  # Don't modify cached data
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                if start_date is not None:
                    df = df[df['timestamp'] >= start_date]
                if end_date is not None:
                    df = df[df['timestamp'] <= end_date]
        
        logger.info(f"✅ Loaded {len(df)} stock bars for {symbol}")
        return df

    def load_options_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Load options data from flat file

        Args:
            symbol: Underlying symbol (e.g., 'SPY')
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            List of option contracts with pricing and Greeks
        """
        # Check cache first
        if self.cache_in_memory and symbol in self.options_cache:
            df = self.options_cache[symbol]
        else:
            file_path = self._get_options_file_path(symbol)

            if not file_path.exists():
                logger.warning(f"Options data file not found: {file_path}")
                return []

            # Load from file
            if self.file_format == 'parquet':
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path, parse_dates=['timestamp', 'expiration'])

            # Cache if enabled
            if self.cache_in_memory:
                self.options_cache[symbol] = df

        # Filter by date range if specified
        if start_date is not None or end_date is not None:
            df = df.copy()  # Don't modify cached data
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                if start_date is not None:
                    df = df[df['timestamp'] >= start_date]
                if end_date is not None:
                    df = df[df['timestamp'] <= end_date]

        # Convert to list of dicts (compatible with existing code)
        options_list = df.to_dict('records')

        logger.info(f"✅ Loaded {len(options_list)} options contracts for {symbol}")
        return options_list

    async def load_historical_stock_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """
        Load historical stock data for multiple symbols

        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date

        Returns:
            Dict mapping symbol to DataFrame
        """
        result = {}

        logger.info(f"📊 Loading stock data for {len(symbols)} symbols from flat files...")

        for idx, symbol in enumerate(symbols, 1):
            try:
                df = self.load_stock_data(symbol, start_date, end_date)
                if not df.empty:
                    result[symbol] = df
                    logger.info(f"  [{idx}/{len(symbols)}] ✅ {symbol}: {len(df)} bars")
                else:
                    logger.warning(f"  [{idx}/{len(symbols)}] ⚠️ {symbol}: No data")
            except Exception as e:
                logger.error(f"  [{idx}/{len(symbols)}] ❌ {symbol}: {e}")

        logger.info(f"✅ Loaded stock data for {len(result)}/{len(symbols)} symbols")
        return result

    async def load_historical_options_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True
    ) -> Dict[str, List[Dict]]:
        """
        Load historical options data for multiple symbols

        Args:
            symbols: List of underlying symbols
            start_date: Start date
            end_date: End date
            use_cache: Whether to use cache (ignored for flat files, always uses cache)

        Returns:
            Dict mapping symbol to list of option contracts
        """
        result = {}

        logger.info(f"📊 Loading options data for {len(symbols)} symbols from flat files...")

        for idx, symbol in enumerate(symbols, 1):
            try:
                options = self.load_options_data(symbol, start_date, end_date)
                if options:
                    result[symbol] = options
                    logger.info(f"  [{idx}/{len(symbols)}] ✅ {symbol}: {len(options)} contracts")
                else:
                    logger.warning(f"  [{idx}/{len(symbols)}] ⚠️ {symbol}: No data")
            except Exception as e:
                logger.error(f"  [{idx}/{len(symbols)}] ❌ {symbol}: {e}")

        logger.info(f"✅ Loaded options data for {len(result)}/{len(symbols)} symbols")
        return result

    async def load_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1Hour'
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

    def clear_cache(self):
        """Clear in-memory cache"""
        self.stock_cache.clear()
        self.options_cache.clear()
        logger.info("🗑️ Cleared in-memory cache")

    def get_available_symbols(self) -> Tuple[List[str], List[str]]:
        """
        Get list of available symbols in flat files

        Returns:
            Tuple of (stock_symbols, options_symbols)
        """
        stock_symbols = []
        options_symbols = []

        # Find stock files
        extension = 'parquet' if self.file_format == 'parquet' else 'csv'
        for file_path in self.stocks_dir.glob(f"*.{extension}"):
            symbol = file_path.stem
            stock_symbols.append(symbol)

        # Find options files
        for file_path in self.options_dir.glob(f"*_options.{extension}"):
            symbol = file_path.stem.replace('_options', '')
            options_symbols.append(symbol)

        return sorted(stock_symbols), sorted(options_symbols)

