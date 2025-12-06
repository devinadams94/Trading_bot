#!/usr/bin/env python3
"""
Create GPU cache for Multi-Asset Environment from stock parquet files.

Usage:
    # Create training cache (2015-2019)
    python scripts/create_multi_asset_cache.py --output data/v2_train_2015_2019/gpu_cache_train.pt --start 2015-01-01 --end 2019-12-31

    # Create test cache (2020-2024)
    python scripts/create_multi_asset_cache.py --output data/v2_test_2020_2024/gpu_cache_test.pt --start 2020-01-01 --end 2024-12-31

    # Create full cache (all available data)
    python scripts/create_multi_asset_cache.py --output data/gpu_cache_full.pt
"""

import torch
import pandas as pd
import argparse
import os
from pathlib import Path
from datetime import datetime


def create_cache(data_dir: str, output_path: str, symbols: list, start_date: str = None, end_date: str = None):
    """Create GPU cache from parquet files"""
    print("=" * 60)
    print("🚀 Creating Multi-Asset GPU Cache")
    print("=" * 60)
    print(f"Data dir: {data_dir}")
    print(f"Output:   {output_path}")
    print(f"Symbols:  {symbols}")
    if start_date:
        print(f"Start:    {start_date}")
    if end_date:
        print(f"End:      {end_date}")
    print()

    stock_prices = {}
    all_dates = set()

    for symbol in symbols:
        parquet_path = os.path.join(data_dir, f"{symbol}.parquet")
        if not os.path.exists(parquet_path):
            print(f"⚠️  {symbol}.parquet not found, skipping")
            continue

        print(f"📊 Loading {symbol}...")
        df = pd.read_parquet(parquet_path)

        # Standardize column names
        df.columns = df.columns.str.lower()

        # Find date column
        date_col = None
        for col in ['date', 'trade_date', 'timestamp', 'time']:
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            print(f"   ⚠️  No date column found in {symbol}, columns: {list(df.columns)}")
            continue

        df['date'] = pd.to_datetime(df[date_col])

        # Filter by date range
        if start_date:
            df = df[df['date'] >= start_date]
        if end_date:
            df = df[df['date'] <= end_date]

        if len(df) == 0:
            print(f"   ⚠️  No data for {symbol} in date range")
            continue

        df = df.sort_values('date')
        all_dates.update(df['date'].dt.date.tolist())

        # Get OHLCV columns
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in ohlcv_cols if c not in df.columns]
        if missing:
            print(f"   ⚠️  Missing columns: {missing}")
            # Try to fill missing with close price
            if 'close' in df.columns:
                for col in missing:
                    if col != 'volume':
                        df[col] = df['close']
                    else:
                        df[col] = 0

        # Create tensor [n_days, 5] for OHLCV
        data = df[ohlcv_cols].values
        stock_prices[symbol] = {
            'dates': df['date'].dt.date.tolist(),
            'data': torch.tensor(data, dtype=torch.float32)
        }
        print(f"   ✅ {len(df)} days: {df['date'].min().date()} to {df['date'].max().date()}")

    if not stock_prices:
        raise ValueError("No stock data loaded!")

    # Create aligned tensors
    all_dates = sorted(all_dates)
    n_days = len(all_dates)
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    print(f"\n📅 Total trading days: {n_days}")
    print(f"   Date range: {all_dates[0]} to {all_dates[-1]}")

    # Create aligned price tensors
    aligned_prices = {}
    for symbol, data in stock_prices.items():
        tensor = torch.zeros((n_days, 5), dtype=torch.float32)
        for date, row_idx in zip(data['dates'], range(len(data['data']))):
            if date in date_to_idx:
                tensor[date_to_idx[date]] = data['data'][row_idx]

        # Forward fill missing values
        for i in range(1, n_days):
            if tensor[i].sum() == 0:
                tensor[i] = tensor[i-1]

        aligned_prices[symbol] = tensor
        print(f"   {symbol}: {(tensor[:, 3] > 0).sum().item()} valid days")

    # Create cache
    cache = {
        'n_days': n_days,
        'symbols': symbols,
        'dates': [str(d) for d in all_dates],
        'stock_prices': aligned_prices,
        'created': datetime.now().isoformat(),
        'start_date': str(all_dates[0]),
        'end_date': str(all_dates[-1]),
    }

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(cache, output_path)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"\n✅ Saved cache: {output_path} ({size_mb:.1f} MB)")
    print(f"   {n_days} days, {len(aligned_prices)} symbols")

    return cache


def main():
    parser = argparse.ArgumentParser(description="Create Multi-Asset GPU Cache")
    parser.add_argument("--data-dir", default="data/flat_files/stocks", help="Directory with parquet files")
    parser.add_argument("--output", required=True, help="Output cache path")
    parser.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "IWM"], help="Symbols to include")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    create_cache(args.data_dir, args.output, args.symbols, args.start, args.end)


if __name__ == "__main__":
    main()

