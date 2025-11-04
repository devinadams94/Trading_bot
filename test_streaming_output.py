#!/usr/bin/env python3
"""
Test streaming output to verify real-time progress display
"""

import sys
import time
import asyncio

async def test_streaming():
    """Test that output streams in real-time"""
    
    print("=" * 80, flush=True)
    print("🧪 Testing Streaming Output", flush=True)
    print("=" * 80, flush=True)
    print("", flush=True)
    
    # Test 1: Simple print statements
    print("Test 1: Simple print statements with flush=True", flush=True)
    for i in range(5):
        print(f"  [{i+1}/5] Processing item {i+1}...", flush=True)
        await asyncio.sleep(0.5)
    print("✅ Test 1 complete", flush=True)
    print("", flush=True)
    
    # Test 2: Print + sys.stdout.flush()
    print("Test 2: Print with explicit sys.stdout.flush()", flush=True)
    for i in range(5):
        print(f"  [{i+1}/5] Processing item {i+1}...")
        sys.stdout.flush()
        sys.stderr.flush()
        await asyncio.sleep(0.5)
    print("✅ Test 2 complete", flush=True)
    print("", flush=True)
    
    # Test 3: Simulating API calls
    print("Test 3: Simulating API calls with delays", flush=True)
    symbols = ['SPY', 'QQQ', 'AAPL']
    for idx, symbol in enumerate(symbols, 1):
        print(f"  [{idx}/{len(symbols)}] 📥 Downloading {symbol}...", flush=True)
        await asyncio.sleep(0.3)
        
        print(f"  [{idx}/{len(symbols)}] 🌐 Calling API for {symbol}...", flush=True)
        await asyncio.sleep(0.5)
        
        print(f"  [{idx}/{len(symbols)}] ⏳ Waiting for response...", flush=True)
        await asyncio.sleep(1.0)
        
        print(f"  [{idx}/{len(symbols)}] 📦 Received response for {symbol}", flush=True)
        await asyncio.sleep(0.2)
        
        print(f"  [{idx}/{len(symbols)}] ✅ {symbol}: 1,234 bars (quality: 0.95)", flush=True)
        print("", flush=True)
    
    print("✅ Test 3 complete", flush=True)
    print("", flush=True)
    
    print("=" * 80, flush=True)
    print("✅ All streaming tests complete!", flush=True)
    print("=" * 80, flush=True)
    print("", flush=True)
    print("If you saw each line appear in real-time (not all at once),", flush=True)
    print("then streaming output is working correctly!", flush=True)

if __name__ == "__main__":
    print("\n🚀 Starting streaming output test...\n", flush=True)
    print("You should see lines appear one at a time, not all at once.\n", flush=True)
    
    asyncio.run(test_streaming())
    
    print("\n✅ Test complete!\n", flush=True)

