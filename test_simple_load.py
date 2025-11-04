#!/usr/bin/env python3
"""
Simple test to see what's being loaded
"""

import asyncio
import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Setup logging to see everything
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.historical_options_data import OptimizedHistoricalOptionsDataLoader

# Load environment variables
load_dotenv()

async def main():
    print("=" * 80)
    print("SIMPLE OPTIONS DATA LOAD TEST")
    print("=" * 80)
    
    # Initialize data loader
    data_loader = OptimizedHistoricalOptionsDataLoader(
        api_key=os.getenv('ALPACA_API_KEY'),
        api_secret=os.getenv('ALPACA_SECRET_KEY'),
        base_url='https://paper-api.alpaca.markets',
        data_url='https://data.alpaca.markets',
        cache_dir='data/options_cache'
    )
    
    # Load data for SPY only
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    print(f"\nLoading data for SPY from {start_date.date()} to {end_date.date()}")
    print("-" * 80)
    
    data = await data_loader.load_historical_data(['SPY'], start_date, end_date)
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    if 'SPY' in data:
        spy_data = data['SPY']
        print(f"\n✅ SPY data loaded: {len(spy_data)} records")
        
        if len(spy_data) > 0:
            # Check first record
            first_record = spy_data.iloc[0] if hasattr(spy_data, 'iloc') else spy_data[0]
            print(f"\nFirst record:")
            for key, value in first_record.items():
                print(f"  {key}: {value}")
    else:
        print("\n❌ No SPY data loaded")

if __name__ == "__main__":
    asyncio.run(main())

