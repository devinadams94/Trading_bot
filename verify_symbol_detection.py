#!/usr/bin/env python3
"""
Verify that the training script correctly detects available symbols
"""

import os
import sys

# Simulate the symbol detection logic from train_enhanced_clstm_ppo.py

def detect_available_symbols(flat_files_dir='data/flat_files', file_format='parquet'):
    """Detect which symbols have both stock and options data"""
    
    # All symbols the training script might try to use
    all_symbols = [
        'SPY', 'QQQ', 'IWM',  # ETFs
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',  # Mega cap
        'TSLA', 'META', 'NFLX', 'AMD', 'CRM',  # High vol tech
        'PLTR', 'SNOW', 'COIN', 'RBLX', 'ZM',  # Growth stocks
        'JPM', 'BAC', 'GS', 'V', 'MA'  # Financials
    ]
    
    available_symbols = []
    stocks_dir = os.path.join(flat_files_dir, 'stocks')
    options_dir = os.path.join(flat_files_dir, 'options')
    
    print("=" * 80)
    print("SYMBOL AVAILABILITY CHECK")
    print("=" * 80)
    print(f"Data directory: {flat_files_dir}")
    print(f"File format: {file_format}")
    print()
    
    for symbol in all_symbols:
        # Check if both stock and options data exist
        stock_file = os.path.join(stocks_dir, f"{symbol}.{file_format}")
        options_file = os.path.join(options_dir, f"{symbol}_options.{file_format}")
        
        stock_exists = os.path.exists(stock_file)
        options_exists = os.path.exists(options_file)
        
        if stock_exists and options_exists:
            available_symbols.append(symbol)
            print(f"✅ {symbol:6s} - Both stock and options data available")
        elif stock_exists:
            print(f"⚠️  {symbol:6s} - Stock data only (missing options)")
        elif options_exists:
            print(f"⚠️  {symbol:6s} - Options data only (missing stock)")
        else:
            print(f"❌ {symbol:6s} - No data available")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total symbols checked: {len(all_symbols)}")
    print(f"Available symbols: {len(available_symbols)}")
    print(f"Missing symbols: {len(all_symbols) - len(available_symbols)}")
    print()
    
    if available_symbols:
        print(f"✅ Training will use: {available_symbols}")
        print()
        print("To train with these symbols:")
        print("  python3 train_enhanced_clstm_ppo.py --use-flat-files --no-realistic-costs --episodes 2000")
    else:
        print("❌ No symbols have complete data!")
        print()
        print("To download data:")
        print("  python3 download_data_to_flat_files.py")
    
    print()
    
    return available_symbols


if __name__ == '__main__':
    # Check default location
    available = detect_available_symbols()
    
    if not available:
        print("⚠️  WARNING: No training data available!")
        print("   Please run: python3 download_data_to_flat_files.py")
        sys.exit(1)
    else:
        print(f"✅ Ready to train with {len(available)} symbols!")
        sys.exit(0)

