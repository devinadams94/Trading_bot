#!/usr/bin/env python3
"""
Test script for Massive.com REST API data loading
"""

import asyncio
import sys
from datetime import datetime, timedelta
from src.historical_options_data import OptimizedHistoricalOptionsDataLoader

async def test_rest_api_loading():
    """Test REST API data loading for stocks and options"""
    
    print("=" * 80)
    print("🧪 Testing Massive.com REST API Data Loading")
    print("=" * 80)
    print()
    
    # Initialize data loader with Massive.com API key
    api_key = "O_182Z1cNv_y6zMpPwjLZ_pwIH8W9lWF"
    
    print(f"✅ Initializing data loader with API key: {api_key[:8]}...")
    loader = OptimizedHistoricalOptionsDataLoader(
        api_key=api_key,
        api_secret=None,
        base_url=None,
        data_url=None
    )
    print()
    
    # Test 1: Fetch stock data for a short period
    print("=" * 80)
    print("📊 Test 1: Fetch Stock Data (SPY, 10 days)")
    print("=" * 80)
    print()
    
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=10)
    
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print()
    
    try:
        stock_data = await loader._fetch_stock_data_rest_api('SPY', start_date, end_date)
        
        if stock_data is not None and len(stock_data) > 0:
            print(f"✅ SUCCESS: Fetched {len(stock_data)} bars for SPY")
            print()
            print("Sample data (first 3 rows):")
            print(stock_data.head(3))
            print()
            print("Sample data (last 3 rows):")
            print(stock_data.tail(3))
            print()
        else:
            print("❌ FAILED: No stock data returned")
            print()
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print()
    
    # Test 2: Fetch options snapshot
    print("=" * 80)
    print("📊 Test 2: Fetch Options Snapshot (SPY, current)")
    print("=" * 80)
    print()
    
    try:
        # Get current stock price first
        recent_stock = await loader._fetch_stock_data_rest_api('SPY', end_date - timedelta(days=1), end_date)
        
        if recent_stock is not None and len(recent_stock) > 0:
            stock_price = recent_stock['close'].iloc[-1]
            print(f"Current SPY price: ${stock_price:.2f}")
            print()
            
            options_snapshot = await loader._fetch_options_snapshot_rest_api('SPY', end_date, stock_price)
            
            if options_snapshot and len(options_snapshot) > 0:
                print(f"✅ SUCCESS: Fetched {len(options_snapshot)} option contracts")
                print()
                print("Sample options (first 5):")
                for i, opt in enumerate(options_snapshot[:5]):
                    print(f"  {i+1}. {opt['option_symbol']}: Strike=${opt['strike']:.2f}, "
                          f"Type={opt['option_type']}, Last=${opt['last']:.2f}, "
                          f"Delta={opt['delta']:.4f}, IV={opt['implied_volatility']:.2%}")
                print()
            else:
                print("❌ FAILED: No options data returned")
                print()
        else:
            print("❌ FAILED: Could not get current stock price")
            print()
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print()
    
    # Test 3: Full integration test - load stock and options data
    print("=" * 80)
    print("📊 Test 3: Full Integration Test (SPY, 7 days)")
    print("=" * 80)
    print()
    
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=7)
    
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print()
    
    try:
        # Load stock data
        print("Loading stock data...")
        stock_data = await loader.load_historical_stock_data(['SPY'], start_date, end_date)
        
        if 'SPY' in stock_data and len(stock_data['SPY']) > 0:
            print(f"✅ Stock data loaded: {len(stock_data['SPY'])} bars")
            print()
        else:
            print("❌ Stock data loading failed")
            print()
            return
        
        # Load options data
        print("Loading options data...")
        options_data = await loader.load_historical_options_data(
            symbols=['SPY'],
            start_date=start_date,
            end_date=end_date,
            use_cache=False  # Don't use cache for testing
        )
        
        if 'SPY' in options_data and len(options_data['SPY']) > 0:
            print(f"✅ Options data loaded: {len(options_data['SPY'])} contracts")
            print()
            
            # Show sample option
            sample_opt = options_data['SPY'][0]
            print("Sample option contract:")
            for key, value in sample_opt.items():
                print(f"  {key}: {value}")
            print()
        else:
            print("❌ Options data loading failed")
            print()
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print()
    
    print("=" * 80)
    print("✅ All tests completed!")
    print("=" * 80)

if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_rest_api_loading())

