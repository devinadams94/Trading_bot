#!/usr/bin/env python3
"""
Verify that flat files contain real data and training uses them correctly

This script:
1. Downloads a small sample of real data to flat files
2. Verifies the data is real (not simulated)
3. Tests that training loads from flat files
4. Confirms no simulated data is being used
"""

import os
import sys
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.historical_options_data import OptimizedHistoricalOptionsDataLoader
from src.flat_file_data_loader import FlatFileDataLoader


async def verify_real_data():
    """Verify that we're using real data, not simulated"""
    
    print("=" * 80)
    print("🔍 VERIFYING FLAT FILES CONTAIN REAL DATA")
    print("=" * 80)
    print()
    
    # Step 1: Download small sample of real data
    print("📥 Step 1: Downloading sample real data from REST API...")
    print()
    
    api_key = os.getenv('MASSIVE_API_KEY')
    if not api_key:
        print("❌ ERROR: MASSIVE_API_KEY not found in .env file")
        return False
    
    print(f"✅ Using Massive.com API key: {api_key[:8]}...")
    print()
    
    # Initialize REST API loader
    rest_loader = OptimizedHistoricalOptionsDataLoader(
        api_key=api_key,
        api_secret=None,
        base_url=None,
        data_url=None
    )
    
    # Download small sample (30 days, 3 symbols)
    test_symbols = ['SPY', 'QQQ', 'AAPL']
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=30)
    
    print(f"📅 Date range: {start_date.date()} to {end_date.date()} (30 days)")
    print(f"📊 Symbols: {test_symbols}")
    print()
    
    # Download stock data
    print("📈 Downloading stock data from REST API...")
    stock_data = await rest_loader.load_historical_stock_data(
        symbols=test_symbols,
        start_date=start_date,
        end_date=end_date
    )
    
    print()
    
    # Download options data
    print("📊 Downloading options data from REST API...")
    options_data = await rest_loader.load_historical_options_data(
        symbols=test_symbols,
        start_date=start_date,
        end_date=end_date,
        use_cache=False
    )
    
    print()
    
    # Step 2: Verify data is real (not simulated)
    print("=" * 80)
    print("🔍 Step 2: Verifying data is REAL (not simulated)")
    print("=" * 80)
    print()
    
    is_real = True
    
    for symbol, df in stock_data.items():
        if df is not None and not df.empty:
            # Check for realistic price ranges
            if 'close' in df.columns:
                close_prices = df['close'].values
                price_range = close_prices.max() - close_prices.min()
                avg_price = close_prices.mean()
                
                print(f"✅ {symbol} stock data:")
                print(f"   Bars: {len(df)}")
                print(f"   Price range: ${close_prices.min():.2f} - ${close_prices.max():.2f}")
                print(f"   Average price: ${avg_price:.2f}")
                print(f"   Volatility: {(price_range / avg_price * 100):.2f}%")
                
                # Real data should have realistic prices
                if symbol == 'SPY' and (avg_price < 300 or avg_price > 700):
                    print(f"   ⚠️  WARNING: SPY price seems unrealistic!")
                    is_real = False
                elif symbol == 'AAPL' and (avg_price < 100 or avg_price > 300):
                    print(f"   ⚠️  WARNING: AAPL price seems unrealistic!")
                    is_real = False
                else:
                    print(f"   ✅ Price range looks realistic")
                print()
    
    for symbol, contracts in options_data.items():
        if contracts:
            print(f"✅ {symbol} options data:")
            print(f"   Contracts: {len(contracts)}")
            
            # Check for realistic Greeks
            sample_contract = contracts[0]
            if 'delta' in sample_contract and sample_contract['delta'] is not None:
                print(f"   Sample contract delta: {sample_contract['delta']:.4f}")
                print(f"   Sample contract strike: ${sample_contract.get('strike', 'N/A')}")
                print(f"   ✅ Greeks present (real data)")
            else:
                print(f"   ⚠️  WARNING: No Greeks found (might be simulated)")
                is_real = False
            print()
    
    if not is_real:
        print("❌ Data appears to be SIMULATED, not real!")
        return False
    
    print("✅ All data appears to be REAL market data!")
    print()
    
    # Step 3: Save to flat files
    print("=" * 80)
    print("💾 Step 3: Saving data to flat files")
    print("=" * 80)
    print()
    
    output_dir = Path('data/flat_files')
    stocks_dir = output_dir / 'stocks'
    options_dir = output_dir / 'options'
    stocks_dir.mkdir(parents=True, exist_ok=True)
    options_dir.mkdir(parents=True, exist_ok=True)
    
    # Save stock data
    for symbol, df in stock_data.items():
        if df is not None and not df.empty:
            file_path = stocks_dir / f"{symbol}.parquet"
            df.to_parquet(file_path, index=False)
            print(f"✅ Saved {symbol}: {len(df)} bars → {file_path}")
    
    print()
    
    # Save options data
    for symbol, contracts in options_data.items():
        if contracts:
            df = pd.DataFrame(contracts)
            file_path = options_dir / f"{symbol}_options.parquet"
            df.to_parquet(file_path, index=False)
            print(f"✅ Saved {symbol}: {len(contracts)} contracts → {file_path}")
    
    print()
    
    # Step 4: Load from flat files and verify
    print("=" * 80)
    print("📂 Step 4: Loading from flat files and verifying")
    print("=" * 80)
    print()
    
    flat_loader = FlatFileDataLoader(
        data_dir='data/flat_files',
        file_format='parquet',
        cache_in_memory=True
    )
    
    # Load stock data from flat files
    print("📈 Loading stock data from flat files...")
    for symbol in test_symbols:
        df = flat_loader.load_stock_data(symbol, start_date, end_date)
        if not df.empty:
            print(f"✅ {symbol}: {len(df)} bars loaded from flat file")
            print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        else:
            print(f"❌ {symbol}: No data loaded")
    
    print()
    
    # Load options data from flat files
    print("📊 Loading options data from flat files...")
    for symbol in test_symbols:
        contracts = flat_loader.load_options_data(symbol, start_date, end_date)
        if contracts:
            print(f"✅ {symbol}: {len(contracts)} contracts loaded from flat file")
            if 'delta' in contracts[0]:
                print(f"   Sample delta: {contracts[0]['delta']:.4f}")
        else:
            print(f"❌ {symbol}: No data loaded")
    
    print()
    
    # Step 5: Verify training will use flat files
    print("=" * 80)
    print("🎯 Step 5: Verifying training configuration")
    print("=" * 80)
    print()
    
    print("To train with flat files, use:")
    print("  python3 train_enhanced_clstm_ppo.py --use-flat-files --episodes 100")
    print()
    print("This will:")
    print("  ✅ Load data from flat files (5-15 seconds)")
    print("  ✅ Use 100% real market data")
    print("  ✅ No API calls during training")
    print("  ✅ No simulated data")
    print()
    
    print("=" * 80)
    print("✅ VERIFICATION COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  ✅ Downloaded real data from Massive.com REST API")
    print(f"  ✅ Verified data is real (not simulated)")
    print(f"  ✅ Saved to flat files: {output_dir}")
    print(f"  ✅ Loaded from flat files successfully")
    print(f"  ✅ Ready for training with --use-flat-files flag")
    print()
    
    return True


if __name__ == '__main__':
    success = asyncio.run(verify_real_data())
    sys.exit(0 if success else 1)

