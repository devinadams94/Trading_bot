#!/usr/bin/env python3
"""
Fix corrupted options data:
1. Fix timestamps (some stored as ms, should be ns)
2. Fix underlying_price by joining with stock data
3. Filter to only dates where we have valid stock data
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path('data/flat_files')
STOCKS_DIR = DATA_DIR / 'stocks'
OPTIONS_DIR = DATA_DIR / 'options'
FIXED_DIR = DATA_DIR / 'options_fixed'


def fix_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Fix timestamps that were stored as milliseconds instead of nanoseconds"""
    ts_raw = df['timestamp'].astype('int64')
    bad_timestamps = ts_raw < 1e15  # These look like ms not ns
    ts_fixed = ts_raw.copy()
    ts_fixed[bad_timestamps] = ts_raw[bad_timestamps] * 1000000  # ms to ns
    df['timestamp'] = pd.to_datetime(ts_fixed, unit='ns')
    return df


def fix_symbol_options(symbol: str) -> dict:
    """Fix options data for a single symbol"""
    stock_file = STOCKS_DIR / f'{symbol}.parquet'
    options_file = OPTIONS_DIR / f'{symbol}_options.parquet'
    
    if not stock_file.exists():
        print(f'  ⚠️  No stock data for {symbol}')
        return {'symbol': symbol, 'status': 'no_stock_data'}
    
    if not options_file.exists():
        print(f'  ⚠️  No options data for {symbol}')
        return {'symbol': symbol, 'status': 'no_options_data'}
    
    # Load data
    stock_df = pd.read_parquet(stock_file)
    options_df = pd.read_parquet(options_file)
    
    original_rows = len(options_df)
    
    # Fix timestamps
    options_df = fix_timestamps(options_df)
    
    # Get overlapping dates
    stock_df['date'] = stock_df['timestamp'].dt.date
    stock_dates = set(stock_df['date'].unique())
    
    options_df['date'] = options_df['timestamp'].dt.date
    options_df = options_df[options_df['date'].isin(stock_dates)]
    
    if len(options_df) == 0:
        print(f'  ⚠️  No overlapping dates for {symbol}')
        return {'symbol': symbol, 'status': 'no_overlap', 'original': original_rows}
    
    # Join with stock prices to fix underlying_price
    stock_prices = stock_df[['date', 'close']].rename(columns={'close': 'stock_price'})
    options_df = options_df.merge(stock_prices, on='date', how='left')
    
    # Replace underlying_price with correct stock_price
    options_df['underlying_price'] = options_df['stock_price']
    options_df = options_df.drop(columns=['stock_price', 'date'])
    
    # Save fixed data
    output_file = FIXED_DIR / f'{symbol}_options.parquet'
    options_df.to_parquet(output_file, index=False)
    
    print(f'  ✅ {symbol}: {len(options_df):,} rows (was {original_rows:,})')
    
    return {
        'symbol': symbol,
        'status': 'fixed',
        'original': original_rows,
        'fixed': len(options_df),
        'date_range': f"{options_df['timestamp'].min()} to {options_df['timestamp'].max()}"
    }


def main():
    print('=' * 60)
    print('FIXING OPTIONS DATA')
    print('=' * 60)
    print()
    
    # Create output directory
    FIXED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all symbols
    symbols = [f.stem for f in STOCKS_DIR.glob('*.parquet')]
    print(f'Found {len(symbols)} symbols with stock data')
    print()
    
    results = []
    for symbol in sorted(symbols):
        result = fix_symbol_options(symbol)
        results.append(result)
    
    print()
    print('=' * 60)
    print('SUMMARY')
    print('=' * 60)
    
    fixed = [r for r in results if r['status'] == 'fixed']
    total_original = sum(r.get('original', 0) for r in fixed)
    total_fixed = sum(r.get('fixed', 0) for r in fixed)
    
    print(f'Symbols fixed: {len(fixed)}/{len(symbols)}')
    print(f'Total rows: {total_fixed:,} (from {total_original:,})')
    print(f'Output directory: {FIXED_DIR}')


if __name__ == '__main__':
    main()

