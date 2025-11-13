#!/usr/bin/env python3
"""
Download historical stock and options data from REST API and save as flat files

This script downloads data once and saves it to disk for fast offline training.
Run this script periodically to update your data.

Usage:
    # Download 3 years of data for default symbols
    python3 download_data_to_flat_files.py
    
    # Download specific date range
    python3 download_data_to_flat_files.py --days 730
    
    # Download specific symbols
    python3 download_data_to_flat_files.py --symbols SPY QQQ AAPL
    
    # Use CSV instead of Parquet
    python3 download_data_to_flat_files.py --format csv
"""

import os
import sys
import asyncio
import argparse
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.historical_options_data import OptimizedHistoricalOptionsDataLoader

# Try to import pyarrow for parquet support
try:
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False
    print("⚠️  pyarrow not installed. Parquet format not available.")
    print("   Install with: pip install pyarrow")


async def download_and_save_data(
    symbols: list,
    days: int,
    output_dir: str,
    file_format: str
):
    """Download data from REST API and save to flat files"""
    
    print("=" * 80)
    print("📥 DOWNLOADING DATA TO FLAT FILES")
    print("=" * 80)
    print()
    
    # Initialize data loader
    api_key = os.getenv('MASSIVE_API_KEY')
    if not api_key:
        print("❌ ERROR: MASSIVE_API_KEY not found in .env file")
        print("   Please set MASSIVE_API_KEY in your .env file")
        return
    
    print(f"✅ Using Massive.com API key: {api_key[:8]}...")
    print()
    
    loader = OptimizedHistoricalOptionsDataLoader(
        api_key=api_key,
        api_secret=None,
        base_url=None,
        data_url=None
    )
    
    # Calculate date range
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=days)
    
    print(f"📅 Date range: {start_date.date()} to {end_date.date()} ({days} days)")
    print(f"📊 Symbols: {len(symbols)}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 File format: {file_format}")
    print()
    
    # Create output directories
    output_path = Path(output_dir)
    stocks_dir = output_path / 'stocks'
    options_dir = output_path / 'options'
    stocks_dir.mkdir(parents=True, exist_ok=True)
    options_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("📈 DOWNLOADING STOCK DATA")
    print("=" * 80)
    print()
    
    # Download stock data
    stock_data = await loader.load_historical_stock_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date
    )
    
    # Save stock data
    print()
    print("💾 Saving stock data to flat files...")
    print()
    
    for symbol, df in stock_data.items():
        if df is not None and not df.empty:
            if file_format == 'parquet':
                file_path = stocks_dir / f"{symbol}.parquet"
                df.to_parquet(file_path, index=False)
            else:
                file_path = stocks_dir / f"{symbol}.csv"
                df.to_csv(file_path, index=False)
            
            print(f"  ✅ {symbol}: {len(df)} bars → {file_path}")
        else:
            print(f"  ⚠️  {symbol}: No data")
    
    print()
    print("=" * 80)
    print("📊 DOWNLOADING OPTIONS DATA")
    print("=" * 80)
    print()
    
    # Download options data
    options_data = await loader.load_historical_options_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        use_cache=False
    )
    
    # Save options data
    print()
    print("💾 Saving options data to flat files...")
    print()
    
    for symbol, contracts in options_data.items():
        if contracts:
            # Convert list of dicts to DataFrame
            df = pd.DataFrame(contracts)
            
            if file_format == 'parquet':
                file_path = options_dir / f"{symbol}_options.parquet"
                df.to_parquet(file_path, index=False)
            else:
                file_path = options_dir / f"{symbol}_options.csv"
                df.to_csv(file_path, index=False)
            
            print(f"  ✅ {symbol}: {len(contracts)} contracts → {file_path}")
        else:
            print(f"  ⚠️  {symbol}: No data")
    
    print()
    print("=" * 80)
    print("✅ DOWNLOAD COMPLETE")
    print("=" * 80)
    print()
    print(f"📁 Data saved to: {output_dir}")
    print(f"📈 Stock files: {len(stock_data)}")
    print(f"📊 Options files: {len(options_data)}")
    print()
    print("To use flat files for training, update your training script:")
    print("  from src.flat_file_data_loader import FlatFileDataLoader")
    print(f"  data_loader = FlatFileDataLoader(data_dir='{output_dir}', file_format='{file_format}')")
    print()


def main():
    parser = argparse.ArgumentParser(description='Download data to flat files')
    parser.add_argument('--symbols', nargs='+', default=None,
                        help='Symbols to download (default: standard list)')
    parser.add_argument('--days', type=int, default=1095,
                        help='Number of days of historical data (default: 1095 = 3 years)')
    parser.add_argument('--output-dir', type=str, default='data/flat_files',
                        help='Output directory for flat files (default: data/flat_files)')
    parser.add_argument('--format', type=str, choices=['parquet', 'csv'], default='parquet',
                        help='File format (default: parquet)')
    
    args = parser.parse_args()
    
    # Default symbols if not specified
    if args.symbols is None:
        args.symbols = [
            'SPY', 'QQQ', 'IWM',  # ETFs
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',  # Mega cap
            'TSLA', 'META', 'NFLX', 'AMD', 'CRM',  # High vol tech
            'PLTR', 'SNOW', 'COIN', 'RBLX', 'ZM',  # Growth stocks
            'JPM', 'BAC', 'GS', 'V', 'MA'  # Financials
        ]
    
    # Validate format
    if args.format == 'parquet' and not HAS_PARQUET:
        print("❌ ERROR: Parquet format requested but pyarrow not installed")
        print("   Install with: pip install pyarrow")
        print("   Or use --format csv")
        sys.exit(1)
    
    # Run download
    asyncio.run(download_and_save_data(
        symbols=args.symbols,
        days=args.days,
        output_dir=args.output_dir,
        file_format=args.format
    ))


if __name__ == '__main__':
    main()

