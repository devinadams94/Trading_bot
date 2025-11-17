#!/usr/bin/env python3
"""
Verify that Greeks are being loaded and used in training environment
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from flat_file_data_loader import FlatFileDataLoader
from working_options_env import WorkingOptionsEnvironment


async def verify_greeks_integration():
    """Verify Greeks are loaded and used in environment"""
    
    print("=" * 80)
    print("GREEKS INTEGRATION VERIFICATION")
    print("=" * 80)
    print()
    
    # Step 1: Load flat file data
    print("Step 1: Loading flat file data...")
    flat_loader = FlatFileDataLoader(
        data_dir='data/flat_files',
        file_format='parquet',
        cache_in_memory=True
    )
    
    symbols = ['SPY', 'QQQ', 'AAPL']
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=30)
    
    # Load stock data
    stock_data = await flat_loader.load_historical_stock_data(
        symbols, start_date, end_date
    )
    print(f"✅ Loaded stock data for {len(stock_data)} symbols")
    
    # Load options data
    options_data = await flat_loader.load_historical_options_data(
        symbols, start_date, end_date
    )
    print(f"✅ Loaded options data for {len(options_data)} symbols")
    
    # Check Greeks in options data
    total_contracts = sum(len(opts) for opts in options_data.values())
    print(f"   Total contracts: {total_contracts}")
    
    if total_contracts > 0:
        sample_symbol = list(options_data.keys())[0]
        sample_contract = options_data[sample_symbol][0]
        print(f"   Sample contract: {sample_symbol}")
        print(f"   - Delta: {sample_contract.get('delta', 'N/A')}")
        print(f"   - Gamma: {sample_contract.get('gamma', 'N/A')}")
        print(f"   - Theta: {sample_contract.get('theta', 'N/A')}")
        print(f"   - Vega: {sample_contract.get('vega', 'N/A')}")
    print()
    
    # Step 2: Initialize environment
    print("Step 2: Initializing environment with flat file loader...")
    env = WorkingOptionsEnvironment(
        data_loader=flat_loader,
        symbols=symbols,
        initial_capital=100000,
        max_positions=5,
        lookback_window=20,
        episode_length=100,
        use_realistic_costs=False
    )
    
    # Load data into environment
    await env.load_data(start_date, end_date)
    print(f"✅ Environment initialized")
    print(f"   Data loaded: {env.data_loaded}")
    print(f"   Has options_data: {hasattr(env, 'options_data')}")
    if hasattr(env, 'options_data'):
        print(f"   Options data symbols: {list(env.options_data.keys())}")
        total_env_contracts = sum(len(opts) for opts in env.options_data.values())
        print(f"   Total contracts in env: {total_env_contracts}")
    print()
    
    # Step 3: Reset environment and check observation
    print("Step 3: Resetting environment and checking observation...")
    obs = env.reset()
    print(f"✅ Environment reset")
    print(f"   Observation keys: {list(obs.keys())}")
    print(f"   Greeks summary shape: {obs['greeks_summary'].shape}")
    print(f"   Greeks summary (initial): {obs['greeks_summary']}")
    print()
    
    # Step 4: Execute buy call action and check Greeks
    print("Step 4: Executing BUY CALL action...")
    action = 1  # Buy call (0% OTM)
    obs, reward, done, info = env.step(action)
    
    print(f"✅ Action executed")
    print(f"   Positions: {len(env.positions)}")
    
    if len(env.positions) > 0:
        position = env.positions[0]
        print(f"   Position type: {position.get('type')}")
        print(f"   Position strike: {position.get('strike'):.2f}")
        print(f"   Position Greeks:")
        print(f"   - Delta: {position.get('delta', 'N/A')}")
        print(f"   - Gamma: {position.get('gamma', 'N/A')}")
        print(f"   - Theta: {position.get('theta', 'N/A')}")
        print(f"   - Vega: {position.get('vega', 'N/A')}")
        print()
        print(f"   Greeks in observation:")
        print(f"   - greeks_summary[0:4]: {obs['greeks_summary'][0:4]}")
        
        # Verify Greeks are non-zero
        greeks_nonzero = np.any(obs['greeks_summary'][0:4] != 0)
        if greeks_nonzero:
            print(f"   ✅ Greeks are NON-ZERO in observation!")
        else:
            print(f"   ⚠️  Greeks are still ZERO in observation")
    print()
    
    # Step 5: Execute buy put action and check Greeks
    print("Step 5: Executing BUY PUT action...")
    action = 11  # Buy put (0% OTM)
    obs, reward, done, info = env.step(action)
    
    print(f"✅ Action executed")
    print(f"   Positions: {len(env.positions)}")
    
    if len(env.positions) > 1:
        position = env.positions[1]
        print(f"   Position type: {position.get('type')}")
        print(f"   Position strike: {position.get('strike'):.2f}")
        print(f"   Position Greeks:")
        print(f"   - Delta: {position.get('delta', 'N/A')}")
        print(f"   - Gamma: {position.get('gamma', 'N/A')}")
        print(f"   - Theta: {position.get('theta', 'N/A')}")
        print(f"   - Vega: {position.get('vega', 'N/A')}")
        print()
        print(f"   Greeks in observation:")
        print(f"   - greeks_summary[4:8]: {obs['greeks_summary'][4:8]}")
        
        # Verify Greeks are non-zero
        greeks_nonzero = np.any(obs['greeks_summary'][4:8] != 0)
        if greeks_nonzero:
            print(f"   ✅ Greeks are NON-ZERO in observation!")
        else:
            print(f"   ⚠️  Greeks are still ZERO in observation")
    print()
    
    # Summary
    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"✅ Options data loaded: {total_contracts} contracts")
    print(f"✅ Environment has options_data: {hasattr(env, 'options_data')}")
    print(f"✅ Positions created: {len(env.positions)}")
    print(f"✅ Greeks stored in positions: {all('delta' in p for p in env.positions)}")
    print(f"✅ Greeks in observation space: {np.any(obs['greeks_summary'] != 0)}")
    print()
    
    if np.any(obs['greeks_summary'] != 0):
        print("🎉 SUCCESS: Greeks are being loaded and used in training!")
    else:
        print("⚠️  WARNING: Greeks are still zeros - check implementation")
    print()


if __name__ == '__main__':
    asyncio.run(verify_greeks_integration())

