#!/usr/bin/env python3
"""
Create GPU-ready cache file from versioned options data.

Usage:
    # Create training cache
    python scripts/create_versioned_gpu_cache.py --data-dir data/v1_train_2020_2024

    # Create evaluation cache
    python scripts/create_versioned_gpu_cache.py --data-dir data/v1_eval_2015_2019

    # Custom output path
    python scripts/create_versioned_gpu_cache.py --data-dir data/v1_eval_2015_2019 --output my_cache.pt
"""

import torch
import pandas as pd
import time
import sys
import argparse
from pathlib import Path


def create_cache(input_path: str, output_path: str, symbols: list = None):
    """Create GPU cache from parquet file"""
    if symbols is None:
        symbols = ['SPY', 'QQQ', 'IWM']
    
    print("=" * 60)
    print("🚀 Creating GPU-Ready Data Cache")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Symbols: {symbols}")
    
    total_start = time.time()
    
    # Load original data
    print("\n📂 Loading parquet file...")
    sys.stdout.flush()
    start = time.time()
    df = pd.read_parquet(input_path)
    print(f"   ✅ Loaded {len(df):,} records in {time.time()-start:.1f}s")
    
    # Get date range
    df['trade_date_dt'] = pd.to_datetime(df['trade_date_dt'])
    date_min = df['trade_date_dt'].min()
    date_max = df['trade_date_dt'].max()
    print(f"   Date range: {date_min.date()} to {date_max.date()}")
    sys.stdout.flush()
    
    stock_tensors = {}
    options_tensors = {}
    
    for symbol in symbols:
        print(f"\n📊 Processing {symbol}...")
        sys.stdout.flush()
        symbol_start = time.time()
        
        symbol_df = df[df['underlying'] == symbol].copy()
        if len(symbol_df) == 0:
            print(f"   ⚠️ No data for {symbol}, skipping")
            continue
            
        print(f"   Records: {len(symbol_df):,}")
        sys.stdout.flush()
        
        symbol_df['date'] = pd.to_datetime(symbol_df['trade_date_dt']).dt.date
        
        # Daily prices
        print(f"   Creating daily price tensor...")
        sys.stdout.flush()
        daily = symbol_df.groupby('date').agg(
            open=('underlying_price', 'first'),
            high=('underlying_price', 'max'),
            low=('underlying_price', 'min'),
            close=('underlying_price', 'last'),
            volume=('volume', 'sum')
        ).reset_index().sort_values('date')
        
        stock_tensors[symbol] = torch.tensor(
            daily[['open', 'high', 'low', 'close', 'volume']].values,
            dtype=torch.float32
        )
        print(f"   ✅ Stock prices: {stock_tensors[symbol].shape}")
        sys.stdout.flush()
        
        # Options by date
        print(f"   Creating options tensor with Greeks...")
        sys.stdout.flush()
        dates = sorted(symbol_df['date'].unique())
        max_opts = 100
        options_tensor = torch.zeros((len(dates), max_opts, 9), dtype=torch.float32)
        
        for day_idx, date in enumerate(dates):
            if day_idx % 200 == 0:
                print(f"      Day {day_idx}/{len(dates)} ({100*day_idx/len(dates):.0f}%)")
                sys.stdout.flush()
            
            day_df = symbol_df[symbol_df['date'] == date].nlargest(max_opts, 'volume')
            n = min(len(day_df), max_opts)
            if n > 0:
                options_tensor[day_idx, :n, 0] = torch.tensor(day_df['strike'].values[:n])
                options_tensor[day_idx, :n, 1] = torch.tensor((day_df['option_type'] == 'call').astype(float).values[:n])
                options_tensor[day_idx, :n, 2] = torch.tensor(day_df['delta'].values[:n])
                options_tensor[day_idx, :n, 3] = torch.tensor(day_df['gamma'].values[:n])
                options_tensor[day_idx, :n, 4] = torch.tensor(day_df['theta'].values[:n])
                options_tensor[day_idx, :n, 5] = torch.tensor(day_df['vega'].values[:n])
                options_tensor[day_idx, :n, 6] = torch.tensor(day_df['iv'].values[:n])
                options_tensor[day_idx, :n, 7] = torch.tensor(day_df['option_price'].values[:n])
                options_tensor[day_idx, :n, 8] = torch.tensor(day_df['underlying_price'].values[:n])
        
        options_tensors[symbol] = options_tensor
        print(f"   ✅ Options: {options_tensor.shape} ({options_tensor.numel() * 4 / 1e6:.1f} MB)")
        print(f"   ⏱️  {symbol} completed in {time.time()-symbol_start:.1f}s")
        sys.stdout.flush()
    
    # Save cache
    print(f"\n💾 Saving cache file...")
    sys.stdout.flush()
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    cache = {
        'stock_prices': stock_tensors,
        'options': options_tensors,
        'symbols': list(stock_tensors.keys()),
        'n_days': len(dates) if dates else 0,
        'date_range': (str(date_min.date()), str(date_max.date())),
        'source_file': str(input_path)
    }
    torch.save(cache, output_path)
    
    # Verify
    import os
    file_size = os.path.getsize(output_path) / 1e6
    
    print("\n" + "=" * 60)
    print("✅ GPU CACHE CREATED SUCCESSFULLY")
    print("=" * 60)
    print(f"   File: {output_path}")
    print(f"   Size: {file_size:.1f} MB")
    print(f"   Symbols: {cache['symbols']}")
    print(f"   Trading days: {cache['n_days']}")
    print(f"   Date range: {cache['date_range'][0]} to {cache['date_range'][1]}")
    print(f"   Total time: {time.time()-total_start:.1f}s")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Create versioned GPU cache')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to data directory containing parquet file (e.g., data/v1_eval_2015_2019)')
    parser.add_argument('--output', type=str, help='Custom output cache path (default: gpu_cache.pt in data-dir)')
    parser.add_argument('--symbols', nargs='+', default=['SPY', 'QQQ', 'IWM'],
                        help='Symbols to include')
    args = parser.parse_args()

    # Find parquet file in data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    parquet_files = list(data_dir.glob('*.parquet'))
    if len(parquet_files) == 0:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    elif len(parquet_files) > 1:
        print(f"Warning: Multiple parquet files found, using first: {parquet_files[0].name}")

    input_path = str(parquet_files[0])
    output_path = args.output or str(data_dir / 'gpu_cache.pt')

    create_cache(input_path, output_path, args.symbols)


if __name__ == '__main__':
    main()

