#!/usr/bin/env python3
"""
Script to check if Alpaca API credentials are valid
"""

import os
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_credentials():
    """Check if Alpaca credentials are valid"""
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_SECRET_KEY')
    
    print("🔍 Checking Alpaca API credentials...")
    
    if not api_key or not api_secret:
        print("❌ API credentials not found in environment!")
        print("\nTo fix this:")
        print("1. Create a .env file in the project root")
        print("2. Add the following lines:")
        print("   ALPACA_API_KEY=your_api_key_here")
        print("   ALPACA_SECRET_KEY=your_secret_key_here")
        print("\nOr export them in your shell:")
        print("   export ALPACA_API_KEY='your_api_key'")
        print("   export ALPACA_SECRET_KEY='your_secret_key'")
        return False
    
    print(f"✅ Found API key: {api_key[:10]}...")
    print(f"✅ Found API secret: {api_secret[:10]}...")
    
    # Try paper trading endpoint
    print("\n📋 Testing paper trading connection...")
    try:
        paper_api = tradeapi.REST(api_key, api_secret, 'https://paper-api.alpaca.markets')
        account = paper_api.get_account()
        print(f"✅ Paper trading account connected!")
        print(f"   Status: {account.status}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Cash: ${float(account.cash):,.2f}")
        paper_works = True
    except Exception as e:
        print(f"❌ Paper trading connection failed: {e}")
        paper_works = False
    
    # Try live trading endpoint
    print("\n💰 Testing live trading connection...")
    try:
        live_api = tradeapi.REST(api_key, api_secret, 'https://api.alpaca.markets')
        account = live_api.get_account()
        print(f"✅ Live trading account connected!")
        print(f"   Status: {account.status}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Cash: ${float(account.cash):,.2f}")
        live_works = True
    except Exception as e:
        print(f"❌ Live trading connection failed: {e}")
        live_works = False
    
    print("\n📊 Summary:")
    if paper_works or live_works:
        print("✅ Credentials are valid!")
        if paper_works:
            print("   - Paper trading: Available")
        if live_works:
            print("   - Live trading: Available")
        return True
    else:
        print("❌ Credentials are invalid or unauthorized")
        print("\nPossible issues:")
        print("1. API key/secret are incorrect")
        print("2. Account is not activated")
        print("3. API access is not enabled for your account")
        print("\nTo get valid credentials:")
        print("1. Go to https://alpaca.markets/")
        print("2. Sign up for a free account")
        print("3. Go to your dashboard")
        print("4. Generate API keys for paper trading")
        return False

if __name__ == "__main__":
    check_credentials()