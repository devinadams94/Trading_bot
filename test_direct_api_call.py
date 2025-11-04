#!/usr/bin/env python3
"""
Direct API Call Test
Test the Alpaca options chain API directly
"""

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.data.historical import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest

# Load environment variables
load_dotenv()

print("=" * 80)
print("DIRECT ALPACA OPTIONS CHAIN API TEST")
print("=" * 80)

# Create client
api_key = os.getenv('ALPACA_API_KEY')
secret_key = os.getenv('ALPACA_SECRET_KEY')

print(f"\nAPI Key: {api_key[:8]}...{api_key[-4:]}")
print(f"Secret Key: {secret_key[:8]}...{secret_key[-4:]}")

options_client = OptionHistoricalDataClient(
    api_key=api_key,
    secret_key=secret_key
)

# Test request for SPY
symbol = 'SPY'
start_date = datetime.now()
end_date = start_date + timedelta(days=30)

print(f"\nFetching options chain for {symbol}")
print(f"Expiration range: {start_date.date()} to {end_date.date()}")
print("-" * 80)

chain_request = OptionChainRequest(
    underlying_symbol=symbol,
    expiration_date_gte=start_date.date(),
    expiration_date_lte=end_date.date()
)

try:
    options_chain = options_client.get_option_chain(chain_request)
    
    print(f"\n✅ API call successful!")
    print(f"Response type: {type(options_chain)}")
    print(f"Response is dict: {isinstance(options_chain, dict)}")
    
    if isinstance(options_chain, dict):
        print(f"Number of options: {len(options_chain)}")
        print(f"\nFirst option details:")
        opt_symbol, opt_data = list(options_chain.items())[0]
        print(f"Symbol: {opt_symbol}")
        print(f"Data type: {type(opt_data)}")
        print(f"Data: {opt_data}")

        print(f"\nFirst 5 option symbols:")
        for i, (opt_symbol, opt_data) in enumerate(list(options_chain.items())[:5]):
            print(f"  {i+1}. {opt_symbol}")
            if isinstance(opt_data, dict):
                print(f"     Keys: {list(opt_data.keys())}")
                if 'latest_quote' in opt_data and opt_data['latest_quote']:
                    quote = opt_data['latest_quote']
                    print(f"     Bid: ${quote.get('bid_price', 0):.2f}, Ask: ${quote.get('ask_price', 0):.2f}")
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("\n✅ Alpaca options chain API is working!")
    print(f"✅ Retrieved {len(options_chain) if isinstance(options_chain, dict) else 0} options")
    print("\nThe issue is in how the data loader processes this response.")
    print("The response is a dict with option symbols as keys, not a list.")
    
except Exception as e:
    print(f"\n❌ API call failed!")
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

