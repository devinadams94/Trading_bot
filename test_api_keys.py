#!/usr/bin/env python3
"""
Test Alpaca API keys and data access
"""

import os
import sys
from dotenv import load_dotenv
from datetime import datetime, timedelta

def test_api_keys():
    """Test if Alpaca API keys work"""
    print("=" * 80)
    print("🔑 Testing Alpaca API Keys")
    print("=" * 80)
    print()
    
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    base_url = os.getenv('ALPACA_API_BASE_URL')
    
    # Check if keys are loaded
    print("📋 Environment Variables:")
    print(f"   API Key: {api_key[:15] if api_key else 'NOT FOUND'}...")
    print(f"   Secret Key: {secret_key[:15] if secret_key else 'NOT FOUND'}...")
    print(f"   Base URL: {base_url if base_url else 'NOT FOUND'}")
    print()
    
    if not api_key or not secret_key:
        print("❌ API keys not found in .env file!")
        print()
        print("💡 Create a .env file with:")
        print("   ALPACA_API_KEY=your_key_here")
        print("   ALPACA_SECRET_KEY=your_secret_here")
        print("   ALPACA_API_BASE_URL=https://paper-api.alpaca.markets")
        return False
    
    # Test stock data API
    print("🧪 Testing Stock Data API...")
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        
        client = StockHistoricalDataClient(api_key, secret_key)
        
        request = StockBarsRequest(
            symbol_or_symbols='SPY',
            timeframe=TimeFrame.Day,
            start=datetime.now() - timedelta(days=7),
            end=datetime.now()
        )
        
        print(f"   Fetching 7 days of SPY data...")
        bars = client.get_stock_bars(request)
        
        if bars.df.empty:
            print("   ⚠️ API returned empty data")
            return False
        
        print(f"   ✅ Stock API works! Got {len(bars.df)} bars for SPY")
        print(f"   Latest close: ${bars.df['close'].iloc[-1]:.2f}")
        print()
        
    except Exception as e:
        print(f"   ❌ Stock API error: {type(e).__name__}: {e}")
        print()
        
        if "401" in str(e) or "Unauthorized" in str(e):
            print("💡 401 Error means your API keys are invalid or expired.")
            print("   Get new keys from: https://alpaca.markets/")
            print()
        
        return False
    
    # Test options data API
    print("🧪 Testing Options Data API...")
    try:
        from alpaca.data.historical import OptionHistoricalDataClient
        from alpaca.data.requests import OptionChainRequest
        
        options_client = OptionHistoricalDataClient(api_key, secret_key)
        
        chain_request = OptionChainRequest(
            underlying_symbol='SPY',
            expiration_date_gte=datetime.now().date(),
            expiration_date_lte=(datetime.now() + timedelta(days=30)).date()
        )
        
        print(f"   Fetching SPY options chain...")
        options_chain = options_client.get_option_chain(chain_request)
        
        if not options_chain:
            print("   ⚠️ Options API returned no data")
            print("   This is normal for paper trading accounts - options data may not be available")
            print()
        else:
            print(f"   ✅ Options API works! Got options chain data")
            print()
        
    except Exception as e:
        print(f"   ⚠️ Options API error: {type(e).__name__}: {e}")
        print("   This is normal for paper trading accounts - options data may not be available")
        print()
    
    print("=" * 80)
    print("✅ API Key Test Complete")
    print("=" * 80)
    print()
    print("📝 Summary:")
    print("   - Stock data API: ✅ Working")
    print("   - Options data API: ⚠️ May not be available for paper trading")
    print()
    print("💡 You can proceed with training:")
    print("   python train_enhanced_clstm_ppo.py --quick-test --num_gpus 1 --checkpoint-dir checkpoints/test --fresh-start")
    print()
    
    return True

if __name__ == "__main__":
    try:
        success = test_api_keys()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

