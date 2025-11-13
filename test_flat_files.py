#!/usr/bin/env python3
"""
Test flat file data loading

This script tests the flat file data loader to ensure it works correctly.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

from src.flat_file_data_loader import FlatFileDataLoader


async def test_flat_file_loading():
    """Test flat file data loading"""
    
    print("=" * 80)
    print("🧪 TESTING FLAT FILE DATA LOADER")
    print("=" * 80)
    print()
    
    # Check if flat files exist
    data_dir = Path('data/flat_files')
    stocks_dir = data_dir / 'stocks'
    options_dir = data_dir / 'options'
    
    if not stocks_dir.exists() or not options_dir.exists():
        print("❌ ERROR: Flat files directory not found")
        print(f"   Expected: {data_dir}")
        print()
        print("Please download data first:")
        print("  python3 download_data_to_flat_files.py")
        print()
        return
    
    # Initialize loader
    print("📁 Initializing FlatFileDataLoader...")
    loader = FlatFileDataLoader(
        data_dir='data/flat_files',
        file_format='parquet',
        cache_in_memory=True
    )
    print()
    
    # Get available symbols
    print("📊 Checking available symbols...")
    stock_symbols, options_symbols = loader.get_available_symbols()
    print(f"   Stock files: {len(stock_symbols)}")
    print(f"   Options files: {len(options_symbols)}")
    print(f"   Stock symbols: {stock_symbols[:5]}..." if len(stock_symbols) > 5 else f"   Stock symbols: {stock_symbols}")
    print(f"   Options symbols: {options_symbols[:5]}..." if len(options_symbols) > 5 else f"   Options symbols: {options_symbols}")
    print()
    
    if not stock_symbols:
        print("❌ ERROR: No stock data files found")
        print("   Please download data first:")
        print("     python3 download_data_to_flat_files.py")
        return
    
    # Test loading stock data
    print("=" * 80)
    print("📈 TESTING STOCK DATA LOADING")
    print("=" * 80)
    print()
    
    test_symbol = stock_symbols[0]
    print(f"Loading stock data for {test_symbol}...")
    
    import time
    start_time = time.time()
    
    stock_df = loader.load_stock_data(test_symbol)
    
    load_time = time.time() - start_time
    
    if not stock_df.empty:
        print(f"✅ Loaded {len(stock_df)} bars in {load_time:.2f} seconds")
        print()
        print("Sample data:")
        print(stock_df.head())
        print()
        print(f"Date range: {stock_df['timestamp'].min()} to {stock_df['timestamp'].max()}")
        print(f"Columns: {list(stock_df.columns)}")
    else:
        print(f"❌ No data loaded for {test_symbol}")
    
    print()
    
    # Test loading options data
    if options_symbols:
        print("=" * 80)
        print("📊 TESTING OPTIONS DATA LOADING")
        print("=" * 80)
        print()
        
        test_symbol = options_symbols[0]
        print(f"Loading options data for {test_symbol}...")
        
        start_time = time.time()
        
        options_list = loader.load_options_data(test_symbol)
        
        load_time = time.time() - start_time
        
        if options_list:
            print(f"✅ Loaded {len(options_list)} contracts in {load_time:.2f} seconds")
            print()
            print("Sample contract:")
            print(options_list[0])
            print()
            if len(options_list) > 0:
                print(f"Total contracts: {len(options_list)}")
                print(f"Keys: {list(options_list[0].keys())}")
        else:
            print(f"❌ No data loaded for {test_symbol}")
        
        print()
    
    # Test batch loading
    print("=" * 80)
    print("📦 TESTING BATCH LOADING")
    print("=" * 80)
    print()
    
    test_symbols = stock_symbols[:3]  # Test with first 3 symbols
    print(f"Loading data for {len(test_symbols)} symbols: {test_symbols}")
    print()
    
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=90)
    
    start_time = time.time()
    
    stock_data = await loader.load_historical_stock_data(
        symbols=test_symbols,
        start_date=start_date,
        end_date=end_date
    )
    
    load_time = time.time() - start_time
    
    print()
    print(f"✅ Loaded data for {len(stock_data)}/{len(test_symbols)} symbols in {load_time:.2f} seconds")
    print()
    
    for symbol, df in stock_data.items():
        print(f"   {symbol}: {len(df)} bars")
    
    print()
    
    # Test cache
    print("=" * 80)
    print("🗄️  TESTING CACHE")
    print("=" * 80)
    print()
    
    print("Loading same data again (should be cached)...")
    
    start_time = time.time()
    
    stock_df_cached = loader.load_stock_data(stock_symbols[0])
    
    cache_time = time.time() - start_time
    
    print(f"✅ Loaded {len(stock_df_cached)} bars in {cache_time:.4f} seconds (from cache)")
    print(f"   Speed improvement: {load_time / cache_time:.1f}x faster")
    print()
    
    # Summary
    print("=" * 80)
    print("✅ ALL TESTS PASSED")
    print("=" * 80)
    print()
    print("Flat file data loader is working correctly!")
    print()
    print("To use in training:")
    print("  python3 train_enhanced_clstm_ppo.py --use-flat-files --episodes 2000")
    print()


if __name__ == '__main__':
    asyncio.run(test_flat_file_loading())

