#!/usr/bin/env python3
"""
Create GPU-ready cache file from Massive.io options data.
This pre-processes all data so training starts instantly.
"""

import torch
import pandas as pd
import time
import sys

def create_cache():
    print("=" * 60)
    print("🚀 Creating GPU-Ready Data Cache")
    print("=" * 60)
    
    total_start = time.time()
    
    # Load original data
    print("\n📂 Loading parquet file...")
    sys.stdout.flush()
    start = time.time()
    df = pd.read_parquet('data/flat_files_processed/options_with_greeks_2020-01-02_to_2024-12-31.parquet')
    print(f"   ✅ Loaded {len(df):,} records in {time.time()-start:.1f}s")
    sys.stdout.flush()
    
    symbols = ['SPY', 'QQQ', 'IWM']
    stock_tensors = {}
    options_tensors = {}
    
    for symbol in symbols:
        print(f"\n📊 Processing {symbol}...")
        sys.stdout.flush()
        symbol_start = time.time()
        
        symbol_df = df[df['underlying'] == symbol].copy()
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
    cache = {
        'stock_prices': stock_tensors,
        'options': options_tensors,
        'symbols': symbols,
        'n_days': len(dates)
    }
    torch.save(cache, 'data/gpu_cache.pt')
    
    # Verify
    import os
    file_size = os.path.getsize('data/gpu_cache.pt') / 1e6
    
    print("\n" + "=" * 60)
    print("✅ GPU CACHE CREATED SUCCESSFULLY")
    print("=" * 60)
    print(f"   File: data/gpu_cache.pt")
    print(f"   Size: {file_size:.1f} MB")
    print(f"   Trading days: {len(dates)}")
    print(f"   Total time: {time.time()-total_start:.1f}s")
    print("=" * 60)

if __name__ == '__main__':
    create_cache()

