#!/usr/bin/env python3
"""
Setup script for Alpaca live trading credentials
"""

import os
import sys
from dotenv import load_dotenv, set_key
import alpaca_trade_api as tradeapi

def setup_alpaca_credentials():
    """Interactive setup for Alpaca API credentials"""
    
    print("🚀 Alpaca Live Trading Setup")
    print("=" * 50)
    
    # Check if .env exists
    env_path = '.env'
    if not os.path.exists(env_path):
        print("Creating .env file...")
        with open(env_path, 'w') as f:
            f.write("# Alpaca API Credentials\n")
    
    # Load existing environment
    load_dotenv()
    
    # Check existing credentials
    existing_key = os.getenv('ALPACA_API_KEY')
    existing_secret = os.getenv('ALPACA_SECRET_KEY')
    
    if existing_key and existing_secret:
        print("\n✅ Found existing credentials")
        print(f"   API Key: {existing_key[:10]}...")
        print(f"   Secret: {existing_secret[:10]}...")
        
        update = input("\nDo you want to update these credentials? (y/n): ").lower()
        if update != 'y':
            # Test existing credentials
            test_credentials(existing_key, existing_secret)
            return
    
    # Get new credentials
    print("\n📝 Enter your Alpaca API credentials")
    print("   (Get them from https://alpaca.markets/)")
    print("   Note: Use paper trading keys for testing!\n")
    
    api_key = input("API Key: ").strip()
    api_secret = input("API Secret: ").strip()
    
    if not api_key or not api_secret:
        print("❌ Invalid credentials - both key and secret are required")
        return
    
    # Save to .env
    set_key(env_path, 'ALPACA_API_KEY', api_key)
    set_key(env_path, 'ALPACA_SECRET_KEY', api_secret)
    print("\n✅ Credentials saved to .env file")
    
    # Test the credentials
    test_credentials(api_key, api_secret)

def test_credentials(api_key, api_secret):
    """Test Alpaca API credentials"""
    print("\n🔍 Testing credentials...")
    
    # Test paper trading
    print("\n1️⃣ Testing paper trading connection...")
    try:
        paper_api = tradeapi.REST(api_key, api_secret, 'https://paper-api.alpaca.markets')
        account = paper_api.get_account()
        print(f"✅ Paper trading connected!")
        print(f"   Account status: {account.status}")
        print(f"   Buying power: ${float(account.buying_power):,.2f}")
        print(f"   Cash: ${float(account.cash):,.2f}")
        paper_works = True
    except Exception as e:
        print(f"❌ Paper trading failed: {e}")
        paper_works = False
    
    # Test live trading
    print("\n2️⃣ Testing live trading connection...")
    try:
        live_api = tradeapi.REST(api_key, api_secret, 'https://api.alpaca.markets')
        account = live_api.get_account()
        print(f"✅ Live trading connected!")
        print(f"   Account status: {account.status}")
        print(f"   Buying power: ${float(account.buying_power):,.2f}")
        print(f"   Cash: ${float(account.cash):,.2f}")
        live_works = True
    except Exception as e:
        print(f"❌ Live trading failed: {e}")
        live_works = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Summary")
    if paper_works or live_works:
        print("✅ Credentials are valid!")
        print("\nYou can now run live trading with:")
        if paper_works:
            print("   python train.py --live-mode --paper-trading")
        if live_works:
            print("   python train.py --live-mode")
        print("\nOptions:")
        print("   --live-capital 10000    # Trading capital")
        print("   --live-symbols SPY QQQ  # Symbols to trade")
        print("   --position-size 0.05    # 5% per position")
    else:
        print("❌ Credentials are invalid")
        print("\nPossible issues:")
        print("1. API key/secret are incorrect")
        print("2. Account is not activated")
        print("3. API access is not enabled")
        print("\nGet valid credentials from:")
        print("https://alpaca.markets/")

if __name__ == "__main__":
    setup_alpaca_credentials()