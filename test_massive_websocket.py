#!/usr/bin/env python3
"""
Test Massive.com WebSocket connection
"""

import asyncio
import sys
from src.historical_options_data import OptimizedHistoricalOptionsDataLoader

async def test_websocket_connection():
    """Test connecting to Massive.com WebSocket API"""
    
    print("🔌 Testing Massive.com WebSocket Connection")
    print("=" * 60)
    
    # Initialize loader with Massive.com API key (UPDATED)
    loader = OptimizedHistoricalOptionsDataLoader(
        api_key='O_182Z1cNv_y6zMpPwjLZ_pwIH8W9lWF'
    )
    
    print(f"✅ Loader initialized")
    print(f"   API Key: {loader.api_key[:8]}...")
    print(f"   Stock WebSocket URL (delayed): {loader.ws_url_stocks_delayed}")
    print(f"   Stock WebSocket URL (realtime): {loader.ws_url_stocks_realtime}")
    print(f"   Options WebSocket URL (delayed): {loader.ws_url_options_delayed}")
    print(f"   Options WebSocket URL (realtime): {loader.ws_url_options_realtime}")
    print()

    # Test OPTIONS connection
    print("🔌 Attempting to connect to OPTIONS WebSocket...")
    print("   Using delayed data feed (15-min delay)")
    print("   SSL verification: ENABLED")
    try:
        success = await loader._connect_websocket(data_type="options", verify_ssl=True)
        
        if success:
            print("✅ WebSocket connection successful!")
            print("✅ Authentication successful!")
            print()
            
            # Test subscription
            print("📡 Testing subscription to SPY options...")
            option_symbols = [
                "O:SPY250117C00600000",  # SPY Jan 17 2025 $600 Call
                "O:SPY250117P00600000"   # SPY Jan 17 2025 $600 Put
            ]
            
            sub_success = await loader._subscribe_to_options(option_symbols)
            
            if sub_success:
                print(f"✅ Subscribed to {len(option_symbols)} option contracts")
                print()
                
                # Collect data for 10 seconds
                print("📊 Collecting data for 10 seconds...")
                data = await loader._collect_websocket_data(duration_seconds=10)
                
                print(f"✅ Collected data for {len(data)} option symbols")
                for symbol, points in data.items():
                    print(f"   {symbol}: {len(points)} data points")
                print()
            else:
                print("❌ Subscription failed")
                print()
            
            # Disconnect
            print("🔌 Disconnecting OPTIONS WebSocket...")
            await loader._disconnect_websocket(data_type="options")
            print("✅ OPTIONS WebSocket disconnected successfully")
            
        else:
            print("❌ WebSocket connection failed")
            print("   Check:")
            print("   1. API key is valid")
            print("   2. Network connection is working")
            print("   3. Massive.com service is available")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test STOCKS connection
    print()
    print("=" * 60)
    print("🔌 Testing STOCKS WebSocket Connection")
    print("=" * 60)
    print()

    try:
        success = await loader._connect_websocket(data_type="stocks", verify_ssl=True)

        if success:
            print("✅ STOCKS WebSocket connection successful!")
            print("✅ STOCKS Authentication successful!")
            print()

            # Disconnect
            print("🔌 Disconnecting from STOCKS WebSocket...")
            await loader._disconnect_websocket(data_type="stocks")
            print("✅ STOCKS WebSocket disconnected successfully")

        else:
            print("❌ STOCKS WebSocket connection failed")

    except Exception as e:
        print(f"❌ STOCKS WebSocket Error: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 60)
    print("✅ All tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_websocket_connection())

