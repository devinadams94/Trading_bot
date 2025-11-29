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
        cache_in_memory: bool = True,
        use_fixed_options: bool = True  # Use corrected options data
    ):
        """
        Initialize flat file data loader

        Args:
            data_dir: Root directory containing flat files
            file_format: File format ('parquet' or 'csv')
            cache_in_memory: Whether to cache loaded data in memory
            use_fixed_options: Use options_fixed/ directory with corrected data
        """
        self.data_dir = Path(data_dir)
        self.stocks_dir = self.data_dir / 'stocks'
        # Use fixed options data by default (corrected timestamps and underlying_price)
        if use_fixed_options and (self.data_dir / 'options_fixed').exists():
            self.options_dir = self.data_dir / 'options_fixed'
            logger.info("✅ Using fixed options data from options_fixed/")
        else:
            self.options_dir = self.data_dir / 'options'
            if use_fixed_options:
                logger.warning("⚠️ options_fixed/ not found, using original options/")
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

    def recalculate_greeks(
        self,
        symbol: str,
        save_to_file: bool = True,
        output_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Recalculate all Greeks for options using our unified Greeks calculator.

        This ensures consistency between historical data and live trading.
        Uses Black-Scholes with IV calculated from option mid price when IV is missing.

        Args:
            symbol: Underlying symbol
            save_to_file: Whether to save the recalculated data
            output_dir: Optional output directory (defaults to options_recalculated/)

        Returns:
            DataFrame with recalculated Greeks
        """
        from src.utils.greeks import GreeksCalculator

        # Load raw options data
        file_path = self._get_options_file_path(symbol)
        if not file_path.exists():
            logger.error(f"Options file not found: {file_path}")
            return pd.DataFrame()

        logger.info(f"📊 Loading options data for {symbol}...")
        if self.file_format == 'parquet':
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path, parse_dates=['timestamp', 'expiration'])

        logger.info(f"   Loaded {len(df)} option records")

        # Initialize calculator
        calc = GreeksCalculator(risk_free_rate=0.05)

        # Calculate mid price if not present
        if 'mid_price' not in df.columns:
            df['mid_price'] = (df['bid'].fillna(0) + df['ask'].fillna(0)) / 2

        # Ensure we have required columns
        required_cols = ['underlying_price', 'strike', 'expiration', 'option_type']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return df

        # Convert timestamps for time-to-expiry calculation
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        if not pd.api.types.is_datetime64_any_dtype(df['expiration']):
            df['expiration'] = pd.to_datetime(df['expiration'])

        # Calculate time to expiry in years
        df['time_to_expiry'] = (df['expiration'] - df['timestamp']).dt.total_seconds() / (252 * 6.5 * 3600)
        df['time_to_expiry'] = df['time_to_expiry'].clip(lower=1e-6)

        logger.info(f"   Recalculating Greeks for {len(df)} options...")

        # Check if we have IV - if not, need to calculate from price
        has_iv = 'implied_volatility' in df.columns and df['implied_volatility'].notna().mean() > 0.5

        if has_iv:
            logger.info("   Using existing IV for Greeks calculation")
            # Fill missing IV with default
            df['implied_volatility'] = df['implied_volatility'].fillna(0.30)

            # Batch calculation
            batch_result = calc.calculate_greeks_batch(
                underlying_prices=df['underlying_price'].values,
                strikes=df['strike'].values,
                times_to_expiry=df['time_to_expiry'].values,
                ivs=df['implied_volatility'].values,
                option_types=df['option_type'].values
            )
        else:
            logger.info("   Calculating IV from option prices, then Greeks")
            # Need to calculate IV from prices first
            n = len(df)
            deltas = np.zeros(n)
            gammas = np.zeros(n)
            thetas = np.zeros(n)
            vegas = np.zeros(n)
            ivs = np.zeros(n)

            for i in range(n):
                if i % 10000 == 0:
                    logger.info(f"   Progress: {i}/{n} ({100*i/n:.1f}%)")

                row = df.iloc[i]
                mid = row['mid_price']

                if mid > 0 and row['underlying_price'] > 0:
                    result = calc.calculate_greeks_from_price(
                        option_price=mid,
                        underlying_price=row['underlying_price'],
                        strike=row['strike'],
                        time_to_expiry=row['time_to_expiry'],
                        option_type=row['option_type']
                    )
                    deltas[i] = result.delta
                    gammas[i] = result.gamma
                    thetas[i] = result.theta
                    vegas[i] = result.vega
                    ivs[i] = result.iv

            batch_result = {
                'delta': deltas,
                'gamma': gammas,
                'theta': thetas,
                'vega': vegas,
                'iv': ivs
            }

        # Update DataFrame with recalculated values
        df['delta_recalc'] = batch_result['delta']
        df['gamma_recalc'] = batch_result['gamma']
        df['theta_recalc'] = batch_result['theta']
        df['vega_recalc'] = batch_result['vega']
        df['iv_recalc'] = batch_result['iv']

        # Replace original columns
        df['delta'] = batch_result['delta']
        df['gamma'] = batch_result['gamma']
        df['theta'] = batch_result['theta']
        df['vega'] = batch_result['vega']
        if not has_iv:
            df['implied_volatility'] = batch_result['iv']

        # Validate results
        valid_greeks = (df['delta'].abs() <= 1) & (df['gamma'] >= 0)
        logger.info(f"   Valid Greeks: {valid_greeks.mean()*100:.1f}%")

        # Save if requested
        if save_to_file:
            out_dir = Path(output_dir) if output_dir else self.data_dir / 'options_recalculated'
            out_dir.mkdir(parents=True, exist_ok=True)

            out_path = out_dir / f"{symbol}_options.{self.file_format}"
            if self.file_format == 'parquet':
                df.to_parquet(out_path)
            else:
                df.to_csv(out_path, index=False)

            logger.info(f"   ✅ Saved to {out_path}")

        return df

