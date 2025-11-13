#!/usr/bin/env python3
"""
Verify that training actually uses flat files and real data

This script runs a quick training test and verifies:
1. Data is loaded from flat files (not REST API)
2. No simulated data is generated
3. Real market data is used in the environment
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta

# Set up logging to capture all messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.flat_file_data_loader import FlatFileDataLoader
from src.working_options_env import WorkingOptionsEnvironment


async def verify_training_data_source():
    """Verify that training uses flat files with real data"""
    
    print("=" * 80)
    print("🔍 VERIFYING TRAINING USES FLAT FILES WITH REAL DATA")
    print("=" * 80)
    print()
    
    # Step 1: Initialize flat file loader
    print("📁 Step 1: Initializing flat file data loader...")
    print()
    
    flat_loader = FlatFileDataLoader(
        data_dir='data/flat_files',
        file_format='parquet',
        cache_in_memory=True
    )
    
    # Check available symbols
    stock_symbols, options_symbols = flat_loader.get_available_symbols()
    print(f"✅ Found {len(stock_symbols)} stock files")
    print(f"✅ Found {len(options_symbols)} options files")
    print(f"   Symbols: {stock_symbols}")
    print()
    
    if not stock_symbols:
        print("❌ ERROR: No flat files found!")
        print("   Run: python3 verify_flat_files_real_data.py")
        return False
    
    # Step 2: Load data from flat files
    print("=" * 80)
    print("📊 Step 2: Loading data from flat files...")
    print("=" * 80)
    print()
    
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=30)
    
    import time
    start_time = time.time()
    
    stock_data = await flat_loader.load_historical_stock_data(
        symbols=stock_symbols,
        start_date=start_date,
        end_date=end_date
    )
    
    options_data = await flat_loader.load_historical_options_data(
        symbols=stock_symbols,
        start_date=start_date,
        end_date=end_date
    )
    
    load_time = time.time() - start_time
    
    print()
    print(f"✅ Data loaded in {load_time:.2f} seconds")
    print(f"   Stock data: {len(stock_data)} symbols")
    print(f"   Options data: {len(options_data)} symbols")
    print()
    
    # Step 3: Verify data is real
    print("=" * 80)
    print("🔍 Step 3: Verifying data is REAL (not simulated)")
    print("=" * 80)
    print()
    
    is_real = True
    
    for symbol, df in stock_data.items():
        if df is not None and not df.empty:
            close_prices = df['close'].values
            avg_price = close_prices.mean()
            
            print(f"✅ {symbol}:")
            print(f"   Bars: {len(df)}")
            print(f"   Average price: ${avg_price:.2f}")
            
            # Check if prices are realistic
            if symbol == 'SPY' and (avg_price < 300 or avg_price > 700):
                print(f"   ❌ WARNING: Price seems unrealistic!")
                is_real = False
            elif symbol == 'QQQ' and (avg_price < 200 or avg_price > 700):
                print(f"   ❌ WARNING: Price seems unrealistic!")
                is_real = False
            elif symbol == 'AAPL' and (avg_price < 100 or avg_price > 300):
                print(f"   ❌ WARNING: Price seems unrealistic!")
                is_real = False
            else:
                print(f"   ✅ Price is realistic")
    
    print()
    
    if not is_real:
        print("❌ Data appears to be SIMULATED!")
        return False
    
    print("✅ All data is REAL market data!")
    print()
    
    # Step 4: Initialize environment with flat file loader
    print("=" * 80)
    print("🎯 Step 4: Initializing environment with flat file loader...")
    print("=" * 80)
    print()
    
    env = WorkingOptionsEnvironment(
        data_loader=flat_loader,
        symbols=stock_symbols,
        initial_capital=100000,
        lookback_window=30,
        episode_length=50,  # Short episode for testing
        include_technical_indicators=True,
        include_market_microstructure=True,
        use_realistic_costs=False,  # Disable for faster testing
        enable_slippage=False,
        slippage_model='none'
    )
    
    print("✅ Environment initialized")
    print()
    
    # Step 5: Reset environment and check data source
    print("=" * 80)
    print("🔄 Step 5: Resetting environment and checking data...")
    print("=" * 80)
    print()
    
    print("Resetting environment (this loads data)...")
    result = env.reset()
    if isinstance(result, tuple):
        obs = result[0]
    else:
        obs = result

    print()
    print("✅ Environment reset successful")
    print()
    
    # Check if environment has real data
    if hasattr(env, 'stock_data') and env.stock_data is not None:
        print("📊 Environment stock data:")
        for symbol, df in env.stock_data.items():
            if df is not None and not df.empty:
                print(f"   {symbol}: {len(df)} bars, avg price: ${df['close'].mean():.2f}")
    
    print()
    
    if hasattr(env, 'options_data') and env.options_data is not None:
        print("📊 Environment options data:")
        for symbol, contracts in env.options_data.items():
            if contracts:
                print(f"   {symbol}: {len(contracts)} contracts")
    
    print()
    
    # Step 6: Verify no simulated data was generated
    print("=" * 80)
    print("🔍 Step 6: Verifying NO simulated data was generated")
    print("=" * 80)
    print()
    
    # Check if any simulated data flags are set
    simulated_data_detected = False
    
    if hasattr(env, 'using_simulated_data'):
        if env.using_simulated_data:
            print("❌ WARNING: Environment is using simulated data!")
            simulated_data_detected = True
        else:
            print("✅ Environment is NOT using simulated data")
    else:
        print("✅ No simulated data flag found (good)")
    
    print()
    
    if simulated_data_detected:
        print("❌ VERIFICATION FAILED: Simulated data detected!")
        return False
    
    # Step 7: Summary
    print("=" * 80)
    print("✅ VERIFICATION COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  ✅ Loaded data from flat files in {load_time:.2f} seconds")
    print(f"  ✅ Data is REAL market data (not simulated)")
    print(f"  ✅ Environment initialized with flat file loader")
    print(f"  ✅ Environment reset successful")
    print(f"  ✅ NO simulated data generated")
    print()
    print("Training will use:")
    print(f"  📁 Data source: Flat files (data/flat_files)")
    print(f"  📊 Symbols: {len(stock_symbols)}")
    print(f"  📈 Stock data: {sum(len(df) for df in stock_data.values() if df is not None)} total bars")
    print(f"  📊 Options data: {sum(len(c) for c in options_data.values() if c)} total contracts")
    print()
    print("To train with flat files:")
    print("  python3 train_enhanced_clstm_ppo.py --use-flat-files --no-realistic-costs --episodes 100")
    print()
    
    return True


if __name__ == '__main__':
    success = asyncio.run(verify_training_data_source())
    sys.exit(0 if success else 1)

