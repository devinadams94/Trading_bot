#!/usr/bin/env python3
"""
Fix SSL issues and test historical data loading from Alpaca
"""

import ssl
import certifi
import aiohttp
import asyncio
import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


async def test_alpaca_connection():
    """Test connection to Alpaca with SSL fix"""
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not api_secret:
        logger.error("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env file")
        return False
    
    # Create SSL context with certificate bundle
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    
    # Create connector with SSL context
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    
    # Test the connection
    async with aiohttp.ClientSession(connector=connector) as session:
        headers = {
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': api_secret
        }
        
        try:
            # Test account endpoint
            async with session.get(
                'https://paper-api.alpaca.markets/v2/account',
                headers=headers
            ) as response:
                if response.status == 200:
                    logger.info("✓ Successfully connected to Alpaca API")
                    account = await response.json()
                    logger.info(f"✓ Account status: {account.get('status')}")
                    return True
                else:
                    logger.error(f"✗ Failed to connect: HTTP {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"✗ Connection error: {e}")
            return False


async def test_historical_data_fetch():
    """Test fetching historical data with SSL fix"""
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_SECRET_KEY')
    
    # Create SSL context
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        headers = {
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': api_secret
        }
        
        # Test fetching SPY bars
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        params = {
            'start': start_date.isoformat() + 'Z',
            'end': end_date.isoformat() + 'Z',
            'timeframe': '1Day',
            'limit': 100
        }
        
        try:
            async with session.get(
                'https://data.alpaca.markets/v2/stocks/SPY/bars',
                headers=headers,
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    bars = data.get('bars', [])
                    logger.info(f"✓ Successfully fetched {len(bars)} bars for SPY")
                    if bars:
                        logger.info(f"  Latest bar: {bars[-1]['t']} - Close: ${bars[-1]['c']}")
                    return True
                else:
                    logger.error(f"✗ Failed to fetch data: HTTP {response.status}")
                    text = await response.text()
                    logger.error(f"  Response: {text}")
                    return False
                    
        except Exception as e:
            logger.error(f"✗ Data fetch error: {e}")
            return False


def install_certificates():
    """Install certificates for macOS/Linux SSL issues"""
    import subprocess
    import sys
    
    if sys.platform == 'darwin':  # macOS
        logger.info("Attempting to install certificates for macOS...")
        try:
            # Try to run the Install Certificates command
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "certifi"])
            logger.info("✓ Certificates updated")
        except Exception as e:
            logger.error(f"Failed to update certificates: {e}")
    else:
        logger.info("Installing/updating certifi package...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "certifi"])


async def main():
    """Test Alpaca connection and data fetching"""
    logger.info("=== Testing Alpaca Connection with SSL Fix ===")
    
    # First try to install/update certificates
    install_certificates()
    
    # Test basic connection
    logger.info("\n1. Testing API connection...")
    connected = await test_alpaca_connection()
    
    if connected:
        # Test data fetching
        logger.info("\n2. Testing historical data fetch...")
        data_fetched = await test_historical_data_fetch()
        
        if data_fetched:
            logger.info("\n✓ All tests passed! You can now use historical data loading.")
            logger.info("\nTo train with historical data:")
            logger.info("python train_ppo_lstm.py --episodes 1000")
        else:
            logger.info("\n✗ Data fetching failed. Check your API permissions.")
    else:
        logger.info("\n✗ Connection failed. Please check:")
        logger.info("1. Your API keys in .env file")
        logger.info("2. Your internet connection")
        logger.info("3. Alpaca API status at https://status.alpaca.markets/")


if __name__ == "__main__":
    asyncio.run(main())