#!/usr/bin/env python3
"""Test script to verify MockAlpacaClient fix"""

import sys
sys.path.insert(0, '.')

print("Testing MockAlpacaClient fix...")
print("=" * 60)

# Import the module
from src.historical_options_data import HistoricalOptionsDataLoader
print("✅ Import successful")

# Check if we're using mock or real Alpaca
from src import historical_options_data
if historical_options_data.HAS_ALPACA:
    print("✅ Using real Alpaca API")
else:
    print("⚠️  Using MockAlpacaClient (Alpaca packages not installed)")
    
    # Verify mock has required methods
    mock_client = historical_options_data.MockAlpacaClient()
    assert hasattr(mock_client, 'get_stock_bars'), "MockAlpacaClient missing get_stock_bars method"
    print("✅ MockAlpacaClient has get_stock_bars method")
    
    assert hasattr(mock_client, 'get_option_bars'), "MockAlpacaClient missing get_option_bars method"
    print("✅ MockAlpacaClient has get_option_bars method")
    
    assert hasattr(mock_client, 'get_option_chain'), "MockAlpacaClient missing get_option_chain method"
    print("✅ MockAlpacaClient has get_option_chain method")
    
    # Test that methods return MockBarsResponse
    response = mock_client.get_stock_bars(None)
    assert hasattr(response, 'df'), "MockBarsResponse missing df attribute"
    print("✅ MockBarsResponse has df attribute")
    
    import pandas as pd
    assert isinstance(response.df, pd.DataFrame), "MockBarsResponse.df is not a DataFrame"
    print("✅ MockBarsResponse.df is a pandas DataFrame")

print("=" * 60)
print("✅ All tests passed!")

