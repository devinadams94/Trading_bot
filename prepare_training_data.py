#!/usr/bin/env python3
"""
Prepare training data from S3 flat files with calculated Greeks

Flow:
1. Pull options data from S3 using Massive flat file keys
2. Calculate Greeks (delta, gamma, theta, vega, rho) from options data
3. Join Greeks into the options data
4. Save to Apache Parquet format
5. Ready for training

Usage:
    python3 prepare_training_data.py --symbols SPY QQQ AAPL
    python3 prepare_training_data.py --all  # Process all symbols in S3
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import math
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Dict, List, Tuple

# Load environment variables
load_dotenv()

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("❌ ERROR: boto3 not installed")
    print("   Install with: pip install boto3")
    sys.exit(1)

try:
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    print("❌ ERROR: pyarrow not installed")
    print("   Install with: pip install pyarrow")
    sys.exit(1)

try:
    from scipy.stats import norm
    HAS_SCIPY = True
except ImportError:
    print("⚠️  WARNING: scipy not installed. Using approximations for Greeks.")
    print("   Install with: pip install scipy")
    HAS_SCIPY = False


def calculate_greeks_for_row(row_data: Tuple[int, dict], risk_free_rate: float = 0.05) -> dict:
    """
    Calculate Greeks for a single row (used in multiprocessing)

    Args:
        row_data: Tuple of (index, row_dict)
        risk_free_rate: Risk-free rate for Black-Scholes

    Returns:
        dict with index and calculated Greeks
    """
    idx, row = row_data

    try:
        # Extract required fields
        stock_price = row.get('stock_price', 0)
        strike = row.get('strike', row.get('strike_price', 0))
        implied_vol = row.get('implied_volatility', row.get('iv', 0))
        option_type = row.get('option_type', row.get('type', 'call'))

        # Calculate time to expiry
        if 'expiration' in row and 'timestamp' in row:
            expiration = pd.to_datetime(row['expiration'])
            timestamp = pd.to_datetime(row['timestamp'])
            days_to_expiry = (expiration - timestamp).days
            time_to_expiry = max(days_to_expiry / 365.0, 0.001)  # Minimum 0.001 years
        else:
            time_to_expiry = 0.1  # Default 36.5 days

        # Validate inputs
        if any(x <= 0 for x in [stock_price, strike, time_to_expiry, implied_vol]):
            return {'index': idx, **_default_greeks()}

        # Calculate d1 and d2
        d1 = (math.log(stock_price / strike) +
              (risk_free_rate + 0.5 * implied_vol**2) * time_to_expiry) / \
             (implied_vol * math.sqrt(time_to_expiry))
        d2 = d1 - implied_vol * math.sqrt(time_to_expiry)

        if HAS_SCIPY:
            greeks = _calculate_greeks_scipy(
                stock_price, strike, time_to_expiry, implied_vol,
                option_type, d1, d2, risk_free_rate
            )
        else:
            greeks = _calculate_greeks_approx(
                stock_price, strike, time_to_expiry, implied_vol,
                option_type, d1, d2, risk_free_rate
            )

        return {'index': idx, **greeks}

    except Exception as e:
        return {'index': idx, **_default_greeks()}


def _default_greeks() -> dict:
    """Return default Greeks when calculation fails"""
    return {
        'delta': 0.0,
        'gamma': 0.0,
        'theta': 0.0,
        'vega': 0.0,
        'rho': 0.0
    }


def _calculate_greeks_scipy(
    stock_price, strike, time_to_expiry, implied_vol,
    option_type, d1, d2, risk_free_rate
) -> dict:
    """Calculate Greeks using scipy (accurate)"""
    from scipy.stats import norm

    is_call = option_type.lower() == 'call'

    # Delta
    if is_call:
        delta = norm.cdf(d1)
    else:
        delta = -norm.cdf(-d1)

    # Gamma (same for calls and puts)
    gamma = norm.pdf(d1) / (stock_price * implied_vol * math.sqrt(time_to_expiry))

    # Theta
    term1 = -(stock_price * norm.pdf(d1) * implied_vol) / (2 * math.sqrt(time_to_expiry))
    if is_call:
        term2 = -risk_free_rate * strike * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
    else:
        term2 = risk_free_rate * strike * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)
    theta = (term1 + term2) / 365  # Daily theta

    # Vega (same for calls and puts)
    vega = stock_price * norm.pdf(d1) * math.sqrt(time_to_expiry) / 100

    # Rho
    if is_call:
        rho = strike * time_to_expiry * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2) / 100
    else:
        rho = -strike * time_to_expiry * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) / 100

    return {
        'delta': round(delta, 6),
        'gamma': round(gamma, 6),
        'theta': round(theta, 6),
        'vega': round(vega, 6),
        'rho': round(rho, 6)
    }


def _calculate_greeks_approx(
    stock_price, strike, time_to_expiry, implied_vol,
    option_type, d1, d2, risk_free_rate
) -> dict:
    """Calculate Greeks using approximations (no scipy)"""
    is_call = option_type.lower() == 'call'

    # Approximate CDF using tanh
    def approx_cdf(x):
        return 0.5 * (1 + math.tanh(0.7978845608 * x))

    # Approximate PDF
    def approx_pdf(x):
        return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

    # Delta
    if is_call:
        delta = approx_cdf(d1)
    else:
        delta = -approx_cdf(-d1)

    # Gamma
    gamma = approx_pdf(d1) / (stock_price * implied_vol * math.sqrt(time_to_expiry))

    # Theta (simplified)
    theta = -stock_price * approx_pdf(d1) * implied_vol / (2 * math.sqrt(time_to_expiry)) / 365

    # Vega
    vega = stock_price * approx_pdf(d1) * math.sqrt(time_to_expiry) / 100

    # Rho (simplified)
    rho = strike * time_to_expiry * 0.01 if is_call else -strike * time_to_expiry * 0.01

    return {
        'delta': round(delta, 6),
        'gamma': round(gamma, 6),
        'theta': round(theta, 6),
        'vega': round(vega, 6),
        'rho': round(rho, 6)
    }


class GreeksCalculator:
    """Calculate option Greeks using Black-Scholes model"""

    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
    
    def calculate_greeks(
        self,
        stock_price: float,
        strike: float,
        time_to_expiry: float,  # in years
        implied_vol: float,
        option_type: str
    ) -> dict:
        """
        Calculate all Greeks using Black-Scholes
        
        Returns:
            dict with keys: delta, gamma, theta, vega, rho
        """
        try:
            # Validate inputs
            if any(x <= 0 for x in [stock_price, strike, time_to_expiry, implied_vol]):
                return self._default_greeks()
            
            # Calculate d1 and d2
            d1 = (math.log(stock_price / strike) + 
                  (self.risk_free_rate + 0.5 * implied_vol**2) * time_to_expiry) / \
                 (implied_vol * math.sqrt(time_to_expiry))
            d2 = d1 - implied_vol * math.sqrt(time_to_expiry)
            
            if HAS_SCIPY:
                return self._calculate_greeks_scipy(
                    stock_price, strike, time_to_expiry, implied_vol, 
                    option_type, d1, d2
                )
            else:
                return self._calculate_greeks_approx(
                    stock_price, strike, time_to_expiry, implied_vol,
                    option_type, d1, d2
                )
        
        except Exception as e:
            print(f"⚠️  Error calculating Greeks: {e}")
            return self._default_greeks()
    
    def _calculate_greeks_scipy(
        self, stock_price, strike, time_to_expiry, implied_vol,
        option_type, d1, d2
    ) -> dict:
        """Calculate Greeks using scipy (accurate)"""
        is_call = option_type.lower() == 'call'
        
        # Delta
        if is_call:
            delta = norm.cdf(d1)
        else:
            delta = -norm.cdf(-d1)
        
        # Gamma (same for calls and puts)
        gamma = norm.pdf(d1) / (stock_price * implied_vol * math.sqrt(time_to_expiry))
        
        # Theta
        term1 = -(stock_price * norm.pdf(d1) * implied_vol) / (2 * math.sqrt(time_to_expiry))
        if is_call:
            term2 = -self.risk_free_rate * strike * math.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2)
        else:
            term2 = self.risk_free_rate * strike * math.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(-d2)
        theta = (term1 + term2) / 365  # Daily theta
        
        # Vega (same for calls and puts)
        vega = stock_price * norm.pdf(d1) * math.sqrt(time_to_expiry) / 100
        
        # Rho
        if is_call:
            rho = strike * time_to_expiry * math.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2) / 100
        else:
            rho = -strike * time_to_expiry * math.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(-d2) / 100
        
        return {
            'delta': round(delta, 6),
            'gamma': round(gamma, 6),
            'theta': round(theta, 6),
            'vega': round(vega, 6),
            'rho': round(rho, 6)
        }
    
    def _calculate_greeks_approx(
        self, stock_price, strike, time_to_expiry, implied_vol,
        option_type, d1, d2
    ) -> dict:
        """Calculate Greeks using approximations (no scipy)"""
        is_call = option_type.lower() == 'call'
        moneyness = stock_price / strike
        
        # Approximate CDF using tanh
        def approx_cdf(x):
            return 0.5 * (1 + math.tanh(0.7978845608 * x))
        
        # Approximate PDF
        def approx_pdf(x):
            return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
        
        # Delta
        if is_call:
            delta = approx_cdf(d1)
        else:
            delta = -approx_cdf(-d1)
        
        # Gamma
        gamma = approx_pdf(d1) / (stock_price * implied_vol * math.sqrt(time_to_expiry))
        
        # Theta (simplified)
        theta = -stock_price * approx_pdf(d1) * implied_vol / (2 * math.sqrt(time_to_expiry)) / 365
        
        # Vega
        vega = stock_price * approx_pdf(d1) * math.sqrt(time_to_expiry) / 100
        
        # Rho (simplified)
        rho = strike * time_to_expiry * 0.01 if is_call else -strike * time_to_expiry * 0.01
        
        return {
            'delta': round(delta, 6),
            'gamma': round(gamma, 6),
            'theta': round(theta, 6),
            'vega': round(vega, 6),
            'rho': round(rho, 6)
        }
    
    def _default_greeks(self) -> dict:
        """Return default Greeks when calculation fails"""
        return {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }


class S3DataProcessor:
    """Process options data from S3 with Greeks calculation"""

    def __init__(self):
        # Get S3 credentials from environment
        self.access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.endpoint_url = os.getenv('S3_ENDPOINT_URL')
        self.bucket_name = os.getenv('S3_BUCKET')

        if not all([self.access_key, self.secret_key, self.endpoint_url, self.bucket_name]):
            print("❌ ERROR: Missing S3 credentials in .env file")
            print("   Required: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_ENDPOINT_URL, S3_BUCKET")
            sys.exit(1)

        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        )

        # Initialize Greeks calculator
        self.greeks_calc = GreeksCalculator()

        # Output directory
        self.output_dir = Path('data/processed')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"✅ S3 Data Processor initialized")
        print(f"   Endpoint: {self.endpoint_url}")
        print(f"   Bucket: {self.bucket_name}")
        print(f"   Output: {self.output_dir}")
        print()

    def list_available_symbols(self) -> list:
        """List all symbols available in S3"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix='options/'
            )

            if 'Contents' not in response:
                return []

            symbols = set()
            for obj in response['Contents']:
                key = obj['Key']
                # Extract symbol from filename like "options/SPY_options.parquet"
                if '_options.parquet' in key:
                    symbol = key.split('/')[-1].replace('_options.parquet', '')
                    symbols.add(symbol)

            return sorted(list(symbols))

        except ClientError as e:
            print(f"❌ Error listing S3 objects: {e}")
            return []

    def download_options_data(self, symbol: str) -> pd.DataFrame:
        """Download options data for a symbol from S3"""
        s3_key = f'options/{symbol}_options.parquet'

        try:
            # Download to temporary file
            temp_file = f'/tmp/{symbol}_options.parquet'
            self.s3_client.download_file(
                self.bucket_name,
                s3_key,
                temp_file
            )

            # Read parquet file
            df = pd.read_parquet(temp_file)

            # Clean up temp file
            os.remove(temp_file)

            return df

        except ClientError as e:
            print(f"❌ Error downloading {s3_key}: {e}")
            return pd.DataFrame()

    def download_stock_data(self, symbol: str) -> pd.DataFrame:
        """Download stock data for a symbol from S3"""
        s3_key = f'stocks/{symbol}.parquet'

        try:
            # Download to temporary file
            temp_file = f'/tmp/{symbol}_stock.parquet'
            self.s3_client.download_file(
                self.bucket_name,
                s3_key,
                temp_file
            )

            # Read parquet file
            df = pd.read_parquet(temp_file)

            # Clean up temp file
            os.remove(temp_file)

            return df

        except ClientError as e:
            print(f"❌ Error downloading {s3_key}: {e}")
            return pd.DataFrame()

    def calculate_and_join_greeks(
        self,
        options_df: pd.DataFrame,
        stock_df: pd.DataFrame,
        symbol: str,
        n_processes: int = None
    ) -> pd.DataFrame:
        """
        Calculate Greeks for options data and join with stock prices (using multiprocessing)

        Expected columns in options_df:
        - timestamp, strike, expiration, option_type, implied_volatility, bid, ask, etc.

        Expected columns in stock_df:
        - timestamp, close (stock price)

        Args:
            n_processes: Number of processes to use (default: CPU count)
        """
        print(f"  📊 Calculating Greeks for {symbol}...")

        if options_df.empty or stock_df.empty:
            print(f"  ⚠️  Empty dataframe for {symbol}")
            return pd.DataFrame()

        # Ensure timestamp columns are datetime
        if 'timestamp' in options_df.columns:
            options_df['timestamp'] = pd.to_datetime(options_df['timestamp'])
        if 'timestamp' in stock_df.columns:
            stock_df['timestamp'] = pd.to_datetime(stock_df['timestamp'])

        # Merge options with stock prices on timestamp
        merged_df = options_df.merge(
            stock_df[['timestamp', 'close']].rename(columns={'close': 'stock_price'}),
            on='timestamp',
            how='left'
        )

        total_rows = len(merged_df)

        # Determine number of processes
        if n_processes is None:
            n_processes = max(1, cpu_count() - 1)  # Leave 1 CPU free

        print(f"  🚀 Using {n_processes} processes for parallel Greeks calculation")
        print(f"  📊 Processing {total_rows:,} options contracts...")

        # Prepare data for multiprocessing
        # Convert dataframe rows to list of (index, dict) tuples
        row_data = [(idx, row.to_dict()) for idx, row in merged_df.iterrows()]

        # Create partial function with risk_free_rate
        calc_func = partial(calculate_greeks_for_row, risk_free_rate=self.greeks_calc.risk_free_rate)

        # Calculate Greeks in parallel
        greeks_list = []

        # Use multiprocessing Pool
        with Pool(processes=n_processes) as pool:
            # Process in chunks for progress reporting
            chunk_size = max(1000, total_rows // (n_processes * 10))

            results = []
            for i, result in enumerate(pool.imap(calc_func, row_data, chunksize=chunk_size)):
                results.append(result)

                # Progress reporting every 10,000 rows
                if (i + 1) % 10000 == 0:
                    progress_pct = (i + 1) / total_rows * 100
                    print(f"    Progress: {i+1:,}/{total_rows:,} ({progress_pct:.1f}%)")

        # Sort results by index to maintain order
        results.sort(key=lambda x: x['index'])

        # Extract Greeks (remove index from dict)
        greeks_list = [{k: v for k, v in r.items() if k != 'index'} for r in results]

        # Add Greeks columns to dataframe
        greeks_df = pd.DataFrame(greeks_list)
        result_df = pd.concat([merged_df.reset_index(drop=True), greeks_df], axis=1)

        print(f"  ✅ Calculated Greeks for {len(result_df):,} options contracts")

        return result_df

    def process_symbol(self, symbol: str, n_processes: int = None) -> bool:
        """Process a single symbol: download, calculate Greeks, save to Parquet"""
        print(f"\n{'='*80}")
        print(f"Processing {symbol}")
        print(f"{'='*80}")

        # Step 1: Download options data from S3
        print(f"  📥 Downloading options data from S3...")
        options_df = self.download_options_data(symbol)

        if options_df.empty:
            print(f"  ❌ No options data found for {symbol}")
            return False

        print(f"  ✅ Downloaded {len(options_df):,} options contracts")

        # Step 2: Download stock data from S3
        print(f"  📥 Downloading stock data from S3...")
        stock_df = self.download_stock_data(symbol)

        if stock_df.empty:
            print(f"  ❌ No stock data found for {symbol}")
            return False

        print(f"  ✅ Downloaded {len(stock_df):,} stock bars")

        # Step 3: Calculate Greeks and join (with multiprocessing)
        enriched_df = self.calculate_and_join_greeks(
            options_df, stock_df, symbol, n_processes=n_processes
        )

        if enriched_df.empty:
            print(f"  ❌ Failed to calculate Greeks for {symbol}")
            return False

        # Step 4: Save to Parquet
        output_file = self.output_dir / f'{symbol}_options_with_greeks.parquet'
        enriched_df.to_parquet(output_file, index=False, compression='snappy')

        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"  💾 Saved to {output_file} ({file_size_mb:.2f} MB)")
        print(f"  ✅ {symbol} processing complete!")

        return True

    def process_all_symbols(self, symbols: list = None, n_processes: int = None):
        """Process multiple symbols"""
        if symbols is None:
            print("🔍 Discovering symbols in S3...")
            symbols = self.list_available_symbols()
            print(f"✅ Found {len(symbols)} symbols: {', '.join(symbols)}")

        if not symbols:
            print("❌ No symbols to process")
            return

        # Determine number of processes
        if n_processes is None:
            n_processes = max(1, cpu_count() - 1)

        print(f"\n{'='*80}")
        print(f"PROCESSING {len(symbols)} SYMBOLS")
        print(f"{'='*80}")
        print(f"💻 System: {cpu_count()} CPUs available")
        print(f"🚀 Using: {n_processes} processes per symbol\n")

        success_count = 0
        failed_symbols = []

        for idx, symbol in enumerate(symbols, 1):
            print(f"\n[{idx}/{len(symbols)}] Processing {symbol}...")

            if self.process_symbol(symbol, n_processes=n_processes):
                success_count += 1
            else:
                failed_symbols.append(symbol)

        # Summary
        print(f"\n{'='*80}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"✅ Success: {success_count}/{len(symbols)}")

        if failed_symbols:
            print(f"❌ Failed: {', '.join(failed_symbols)}")

        print(f"\n📁 Output directory: {self.output_dir}")
        print(f"🎯 Ready for training!")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare training data from S3 with calculated Greeks (using multiprocessing)'
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Symbols to process (e.g., SPY QQQ AAPL)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all symbols found in S3'
    )
    parser.add_argument(
        '--processes',
        type=int,
        default=None,
        help=f'Number of processes to use (default: {max(1, cpu_count() - 1)} = CPU count - 1)'
    )

    args = parser.parse_args()

    # Show system info
    print(f"{'='*80}")
    print(f"GREEKS CALCULATOR - MULTIPROCESSING MODE")
    print(f"{'='*80}")
    print(f"💻 System CPUs: {cpu_count()}")
    print(f"🚀 Processes: {args.processes if args.processes else max(1, cpu_count() - 1)}")
    print(f"📊 Using: {'scipy (accurate)' if HAS_SCIPY else 'approximations (scipy not installed)'}")
    print(f"{'='*80}\n")

    # Initialize processor
    processor = S3DataProcessor()

    # Process symbols
    if args.all:
        processor.process_all_symbols(n_processes=args.processes)
    elif args.symbols:
        processor.process_all_symbols(args.symbols, n_processes=args.processes)
    else:
        print("❌ ERROR: Please specify --symbols or --all")
        print("\nExamples:")
        print("  python3 prepare_training_data.py --symbols SPY QQQ AAPL")
        print("  python3 prepare_training_data.py --all")
        print("  python3 prepare_training_data.py --all --processes 16")
        sys.exit(1)


if __name__ == '__main__':
    main()

