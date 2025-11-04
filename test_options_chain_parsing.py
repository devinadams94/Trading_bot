#!/usr/bin/env python3
"""
Test Options Chain Parsing
Verify that the options chain data is correctly parsed from Alpaca API
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.historical_options_data import OptimizedHistoricalOptionsDataLoader

# Load environment variables
load_dotenv()

async def test_options_chain_parsing():
    """Test parsing of options chain data"""
    print("=" * 80)
    print("OPTIONS CHAIN PARSING TEST")
    print("=" * 80)
    
    # Initialize data loader
    data_loader = OptimizedHistoricalOptionsDataLoader(
        api_key=os.getenv('ALPACA_API_KEY'),
        api_secret=os.getenv('ALPACA_SECRET_KEY'),
        base_url='https://paper-api.alpaca.markets',
        data_url='https://data.alpaca.markets',
        cache_dir='data/options_cache'
    )
    
    # Test symbols
    symbols = ['SPY', 'AAPL']
    
    # Date range (last 7 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    print(f"\nLoading options data for {symbols}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print("-" * 80)
    
    try:
        # Load data
        data = await data_loader.load_historical_data(symbols, start_date, end_date)
        
        # Check results
        for symbol in symbols:
            print(f"\n{symbol}:")
            if symbol in data:
                symbol_data = data[symbol]
                print(f"  ✅ Data loaded: {len(symbol_data)} records")
                
                # Check if it's real or simulated
                if len(symbol_data) > 0:
                    sample = symbol_data.iloc[0] if hasattr(symbol_data, 'iloc') else symbol_data[0]
                    print(f"  Sample record:")
                    print(f"    Timestamp: {sample.get('timestamp', 'N/A')}")
                    print(f"    Underlying price: ${sample.get('underlying_price', 0):.2f}")
                    print(f"    Option symbol: {sample.get('option_symbol', 'N/A')}")
                    print(f"    Strike: ${sample.get('strike', 0):.2f}")
                    print(f"    Type: {sample.get('option_type', 'N/A')}")
                    print(f"    Close: ${sample.get('close', 0):.2f}")
                    print(f"    IV: {sample.get('implied_volatility', 0):.2%}")
                
                # Check quality metrics
                quality = data_loader.get_quality_metrics(f"{symbol}_options")
                if quality:
                    print(f"  Quality score: {quality.quality_score:.2%}")
                    print(f"  Total records: {quality.total_records}")
                    print(f"  Missing values: {quality.missing_values}")
                    print(f"  Outliers: {quality.outliers}")
            else:
                print(f"  ❌ No data loaded")
        
        print("\n" + "=" * 80)
        print("TEST COMPLETE")
        print("=" * 80)
        
        # Check if real data was loaded
        if any(len(data.get(symbol, [])) > 0 for symbol in symbols):
            print("\n✅ Options data loaded successfully!")
            print("\nCheck the logs above to see if real or simulated data was used.")
            print("Look for messages like:")
            print("  - '✅ Found X options in chain for SYMBOL' = Real data")
            print("  - 'Generating simulated options data for SYMBOL' = Simulated data")
        else:
            print("\n⚠️ No options data loaded")
        
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_options_chain_parsing())

