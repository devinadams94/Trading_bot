#!/usr/bin/env python3
"""Diagnose Alpaca API access and options data availability"""

import os
import sys
import alpaca_trade_api as tradeapi
import requests
from datetime import datetime, timedelta
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_alpaca_connection():
    """Test basic Alpaca API connection"""
    logger.info("=== Testing Alpaca API Connection ===")
    
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not api_secret:
        logger.error("❌ Missing API credentials in .env file")
        return False
    
    try:
        # Test with paper trading endpoint
        api = tradeapi.REST(
            api_key, 
            api_secret, 
            'https://paper-api.alpaca.markets',
            api_version='v2'
        )
        
        # Get account info
        account = api.get_account()
        logger.success(f"✓ Connected to Alpaca API")
        logger.info(f"  Account Status: {account.status}")
        logger.info(f"  Buying Power: ${account.buying_power}")
        logger.info(f"  Account Type: {'Paper' if 'paper' in str(api._base_url) else 'Live'}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to Alpaca API: {e}")
        return False

def check_market_data_access():
    """Test market data access"""
    logger.info("\n=== Testing Market Data Access ===")
    
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_SECRET_KEY')
    
    try:
        api = tradeapi.REST(api_key, api_secret, 'https://paper-api.alpaca.markets')
        
        # Test stock data
        bars = api.get_bars('SPY', '1Day', limit=5).df
        logger.success(f"✓ Stock data access working - Retrieved {len(bars)} bars for SPY")
        
        # Test if we have real-time data subscription
        latest_trade = api.get_latest_trade('SPY')
        logger.success(f"✓ Latest trade data accessible")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Market data access failed: {e}")
        return False

def check_options_data_access():
    """Test options data access"""
    logger.info("\n=== Testing Options Data Access ===")
    
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_SECRET_KEY')
    
    # Test direct API call
    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': api_secret
    }
    
    # Try different endpoints
    endpoints = [
        'https://data.alpaca.markets/v1beta1/options/snapshots/SPY',
        'https://data.alpaca.markets/v1beta2/options/snapshots/SPY',  
        'https://data.alpaca.markets/v1/options/contracts'
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, headers=headers)
            if response.status_code == 200:
                logger.success(f"✓ Options endpoint accessible: {endpoint}")
                return True
            elif response.status_code == 403:
                logger.warning(f"⚠️  Options data requires subscription: {endpoint} (403)")
            elif response.status_code == 404:
                logger.info(f"  Endpoint not found: {endpoint} (404)")
            else:
                logger.error(f"  Endpoint returned {response.status_code}: {endpoint}")
                logger.debug(f"  Response: {response.text[:200]}")
        except Exception as e:
            logger.error(f"  Failed to access {endpoint}: {e}")
    
    logger.error("❌ No working options data endpoint found")
    logger.info("\n💡 Options data typically requires:")
    logger.info("   1. A funded live account (not paper trading)")
    logger.info("   2. An active market data subscription")
    logger.info("   3. Specific options data subscription ($45+/month)")
    
    return False

def suggest_alternatives():
    """Suggest alternatives for getting options data"""
    logger.info("\n=== Alternative Solutions ===")
    
    logger.info("\n📊 Option 1: Use Free Data Sources")
    logger.info("   - Yahoo Finance (yfinance) for basic options chains")
    logger.info("   - CBOE for delayed options data")
    logger.info("   - TD Ameritrade API (free with account)")
    
    logger.info("\n🔧 Option 2: Use Simulated Data for Training")
    logger.info("   - Train with synthetic data first")
    logger.info("   - Use train_gpu_optimized.py for initial development")
    logger.info("   - Fine-tune with real data later if needed")
    
    logger.info("\n💰 Option 3: Upgrade Alpaca Subscription")
    logger.info("   - Opra (Options Price Reporting Authority) feed")
    logger.info("   - Real-time options quotes and trades")
    logger.info("   - Historical options data")
    
    logger.info("\n🚀 Option 4: Paper Trade with Stocks First")
    logger.info("   - Alpaca provides free stock data")
    logger.info("   - Test your strategies on stocks")
    logger.info("   - Move to options when you have data access")

def test_alternative_data():
    """Test if we can get options data from Yahoo Finance"""
    logger.info("\n=== Testing Yahoo Finance Alternative ===")
    
    try:
        import yfinance as yf
        spy = yf.Ticker("SPY")
        
        # Get options expiration dates
        expirations = spy.options
        if expirations:
            logger.success(f"✓ Yahoo Finance accessible - Found {len(expirations)} expiration dates")
            
            # Get options chain for first expiration
            opt_chain = spy.option_chain(expirations[0])
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            logger.info(f"  Sample options chain: {len(calls)} calls, {len(puts)} puts")
            logger.info(f"  First call: Strike={calls.iloc[0]['strike']}, Bid={calls.iloc[0]['bid']}, Ask={calls.iloc[0]['ask']}")
            
            return True
    except ImportError:
        logger.warning("⚠️  yfinance not installed. Install with: pip install yfinance")
    except Exception as e:
        logger.error(f"❌ Yahoo Finance test failed: {e}")
    
    return False

def main():
    """Run all diagnostics"""
    logger.info("🔍 Alpaca API Diagnostics")
    logger.info("=" * 50)
    
    # Run tests
    connection_ok = check_alpaca_connection()
    if connection_ok:
        market_data_ok = check_market_data_access()
        options_ok = check_options_data_access()
        
        if not options_ok:
            suggest_alternatives()
            test_alternative_data()
    
    logger.info("\n" + "=" * 50)
    logger.info("Diagnostics complete!")

if __name__ == "__main__":
    main()