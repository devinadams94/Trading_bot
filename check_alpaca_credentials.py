#!/usr/bin/env python3
"""
Check Alpaca API Credentials and Test Options Data Access
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 80)
print("ALPACA API CREDENTIALS CHECK")
print("=" * 80)

# Check environment variables
alpaca_api_key = os.getenv('ALPACA_API_KEY')
alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')

print("\n1. Environment Variables:")
print("-" * 80)

if alpaca_api_key:
    print(f"✅ ALPACA_API_KEY is set")
    print(f"   Value: {alpaca_api_key[:8]}...{alpaca_api_key[-4:]} (length: {len(alpaca_api_key)})")
else:
    print("❌ ALPACA_API_KEY is NOT set")
    print("   Set it with: export ALPACA_API_KEY='your_key_here'")

if alpaca_secret_key:
    print(f"✅ ALPACA_SECRET_KEY is set")
    print(f"   Value: {alpaca_secret_key[:8]}...{alpaca_secret_key[-4:]} (length: {len(alpaca_secret_key)})")
else:
    print("❌ ALPACA_SECRET_KEY is NOT set")
    print("   Set it with: export ALPACA_SECRET_KEY='your_secret_here'")

# Check .env file
print("\n2. .env File:")
print("-" * 80)

env_file = Path('.env')
if env_file.exists():
    print(f"✅ .env file exists at: {env_file.absolute()}")
    
    # Read and check contents (without exposing secrets)
    with open(env_file, 'r') as f:
        lines = f.readlines()
    
    has_api_key = any('ALPACA_API_KEY' in line and not line.strip().startswith('#') for line in lines)
    has_secret_key = any('ALPACA_SECRET_KEY' in line and not line.strip().startswith('#') for line in lines)
    
    if has_api_key:
        print("   ✅ Contains ALPACA_API_KEY")
    else:
        print("   ❌ Does NOT contain ALPACA_API_KEY")
    
    if has_secret_key:
        print("   ✅ Contains ALPACA_SECRET_KEY")
    else:
        print("   ❌ Does NOT contain ALPACA_SECRET_KEY")
else:
    print(f"❌ .env file does NOT exist at: {env_file.absolute()}")
    print("   Create it with:")
    print("   echo 'ALPACA_API_KEY=your_key_here' > .env")
    print("   echo 'ALPACA_SECRET_KEY=your_secret_here' >> .env")

# Test API connection
print("\n3. API Connection Test:")
print("-" * 80)

if not alpaca_api_key or not alpaca_secret_key:
    print("⚠️ Cannot test API connection - credentials not set")
    print("\nTo get Alpaca API credentials:")
    print("1. Go to https://alpaca.markets/")
    print("2. Sign up for a free account")
    print("3. Go to Paper Trading dashboard")
    print("4. Generate API keys")
    print("5. Set them as environment variables or in .env file")
else:
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        
        print("Testing stock data API...")
        
        # Create client
        stock_client = StockHistoricalDataClient(
            api_key=alpaca_api_key,
            secret_key=alpaca_secret_key
        )
        
        # Test request
        request = StockBarsRequest(
            symbol_or_symbols=['SPY'],
            timeframe=TimeFrame.Day,
            start=datetime.now() - timedelta(days=7),
            end=datetime.now()
        )
        
        bars = stock_client.get_stock_bars(request)
        
        if bars and hasattr(bars, 'df') and not bars.df.empty:
            print(f"✅ Stock data API working! Retrieved {len(bars.df)} bars for SPY")
        else:
            print("⚠️ Stock data API returned empty response")
        
    except Exception as e:
        print(f"❌ Stock data API error: {type(e).__name__}: {e}")
    
    # Test options data API
    try:
        from alpaca.data.historical import OptionHistoricalDataClient
        from alpaca.data.requests import OptionChainRequest
        
        print("\nTesting options data API...")
        
        # Create client
        options_client = OptionHistoricalDataClient(
            api_key=alpaca_api_key,
            secret_key=alpaca_secret_key
        )
        
        # Test request
        chain_request = OptionChainRequest(
            underlying_symbol='SPY',
            expiration_date_gte=datetime.now().date(),
            expiration_date_lte=(datetime.now() + timedelta(days=30)).date()
        )
        
        options_chain = options_client.get_option_chain(chain_request)
        
        # Check response
        if options_chain:
            # Try to get chain list
            if hasattr(options_chain, 'options'):
                chain_list = options_chain.options
            elif isinstance(options_chain, dict):
                chain_list = options_chain.get('options', [])
            else:
                chain_list = list(options_chain) if options_chain else []
            
            if chain_list:
                print(f"✅ Options data API working! Retrieved {len(chain_list)} options for SPY")
                print(f"   Sample option: {chain_list[0] if chain_list else 'N/A'}")
            else:
                print("⚠️ Options data API returned empty chain")
                print(f"   Response type: {type(options_chain)}")
                print(f"   Response: {options_chain}")
        else:
            print("⚠️ Options data API returned None")
        
    except Exception as e:
        print(f"❌ Options data API error: {type(e).__name__}: {e}")
        print("\nPossible reasons:")
        print("1. Options data requires a paid Alpaca subscription")
        print("2. API keys don't have options data permissions")
        print("3. Options data API endpoint is different")
        print("\nNote: If options data is not available, training will use simulated data")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if alpaca_api_key and alpaca_secret_key:
    print("✅ Alpaca API credentials are configured")
    print("\nNext steps:")
    print("1. Run the training script to test real data loading")
    print("2. Check logs for 'Using real Alpaca API keys' message")
    print("3. If options data fails, training will use simulated data (this is OK)")
else:
    print("❌ Alpaca API credentials are NOT configured")
    print("\nTo fix:")
    print("1. Get API keys from https://alpaca.markets/")
    print("2. Create .env file with:")
    print("   ALPACA_API_KEY=your_key_here")
    print("   ALPACA_SECRET_KEY=your_secret_here")
    print("3. Or export as environment variables")
    print("\nNote: Training will work with simulated data even without API keys")

print("=" * 80)

