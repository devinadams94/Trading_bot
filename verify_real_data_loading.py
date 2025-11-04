#!/usr/bin/env python3
"""
Verification Script: Confirm Real Options Data Loading
Tests that MultiLegOptionsEnvironment loads real Alpaca data
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.multi_leg_options_env import MultiLegOptionsEnvironment
from src.historical_options_data import OptimizedHistoricalOptionsDataLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def verify_data_loader():
    """Test 1: Verify data loader can fetch real data"""
    logger.info("=" * 60)
    logger.info("TEST 1: Verify Data Loader")
    logger.info("=" * 60)
    
    # Initialize data loader
    data_loader = OptimizedHistoricalOptionsDataLoader(
        api_key=os.getenv('ALPACA_API_KEY', 'demo_key'),
        api_secret=os.getenv('ALPACA_SECRET_KEY', 'demo_secret'),
        base_url='https://paper-api.alpaca.markets',
        data_url='https://data.alpaca.markets',
        cache_dir='data/options_cache'
    )
    
    # Load data for SPY (1 month)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    logger.info(f"Loading data for SPY from {start_date.date()} to {end_date.date()}")
    
    try:
        data = await data_loader.load_historical_data(['SPY'], start_date, end_date)
        
        if 'SPY' in data and len(data['SPY']) > 0:
            logger.info(f"✅ SUCCESS: Loaded {len(data['SPY'])} records for SPY")
            
            # Check data quality
            quality_metrics = data_loader.get_quality_metrics('SPY')
            if quality_metrics:
                logger.info(f"   Data quality score: {quality_metrics.quality_score:.2%}")
                logger.info(f"   Total records: {quality_metrics.total_records}")
                logger.info(f"   Missing values: {quality_metrics.missing_values}")
            
            # Show sample data
            sample = data['SPY'].iloc[0] if hasattr(data['SPY'], 'iloc') else data['SPY'][0]
            logger.info(f"   Sample record: {sample}")
            
            return True, data_loader
        else:
            logger.warning("⚠️ WARNING: No data loaded for SPY")
            logger.warning("   This may be normal if using demo API keys")
            logger.warning("   Environment will fall back to synthetic data")
            return False, data_loader
            
    except Exception as e:
        logger.error(f"❌ ERROR: Failed to load data: {e}")
        logger.warning("   Environment will fall back to synthetic data")
        return False, data_loader


async def verify_working_environment(data_loader):
    """Test 2: Verify WorkingOptionsEnvironment loads data"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Verify WorkingOptionsEnvironment")
    logger.info("=" * 60)
    
    from src.working_options_env import WorkingOptionsEnvironment
    
    # Initialize environment
    env = WorkingOptionsEnvironment(
        data_loader=data_loader,
        symbols=['SPY', 'AAPL'],
        initial_capital=100000,
        max_positions=5,
        use_realistic_costs=True,
        enable_slippage=True
    )
    
    # Load data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    logger.info(f"Loading data into environment...")
    await env.load_data(start_date, end_date)
    
    # Check if data loaded
    if env.data_loaded:
        logger.info(f"✅ SUCCESS: Environment data loaded")
        logger.info(f"   Symbols: {list(env.market_data.keys())}")
        for symbol in env.market_data:
            logger.info(f"   {symbol}: {len(env.market_data[symbol])} records")
        
        # Check if real or synthetic
        if data_loader and hasattr(data_loader, 'historical_data'):
            if len(data_loader.historical_data) > 0:
                logger.info(f"   Data source: REAL (from Alpaca API)")
            else:
                logger.info(f"   Data source: SYNTHETIC (fallback)")
        else:
            logger.info(f"   Data source: SYNTHETIC (no data loader)")
        
        return True
    else:
        logger.error("❌ ERROR: Environment data not loaded")
        return False


async def verify_multi_leg_environment(data_loader):
    """Test 3: Verify MultiLegOptionsEnvironment loads data"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Verify MultiLegOptionsEnvironment")
    logger.info("=" * 60)
    
    # Initialize environment with multi-leg enabled
    env = MultiLegOptionsEnvironment(
        data_loader=data_loader,
        symbols=['SPY', 'AAPL', 'TSLA'],
        initial_capital=100000,
        max_positions=5,
        enable_multi_leg=True,  # 91 actions
        use_realistic_costs=True,
        enable_slippage=True
    )
    
    # Load data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    logger.info(f"Loading data into multi-leg environment...")
    await env.load_data(start_date, end_date)
    
    # Check if data loaded
    if env.data_loaded:
        logger.info(f"✅ SUCCESS: Multi-leg environment data loaded")
        logger.info(f"   Symbols: {list(env.market_data.keys())}")
        for symbol in env.market_data:
            logger.info(f"   {symbol}: {len(env.market_data[symbol])} records")
        
        # Check action space
        logger.info(f"   Action space: {env.action_space.n} actions")
        logger.info(f"   Multi-leg enabled: {env.enable_multi_leg}")
        
        # Check if real or synthetic
        if data_loader and hasattr(data_loader, 'historical_data'):
            if len(data_loader.historical_data) > 0:
                logger.info(f"   Data source: REAL (from Alpaca API)")
            else:
                logger.info(f"   Data source: SYNTHETIC (fallback)")
        else:
            logger.info(f"   Data source: SYNTHETIC (no data loader)")
        
        # Test reset
        logger.info(f"\n   Testing environment reset...")
        obs = env.reset()
        logger.info(f"   ✅ Reset successful")
        logger.info(f"   Observation keys: {list(obs.keys())}")
        
        # Test step
        logger.info(f"\n   Testing environment step...")
        obs, reward, done, info = env.step(0)  # Hold action
        logger.info(f"   ✅ Step successful")
        logger.info(f"   Reward: {reward:.4f}")
        logger.info(f"   Done: {done}")
        
        return True
    else:
        logger.error("❌ ERROR: Multi-leg environment data not loaded")
        return False


async def verify_training_script_integration():
    """Test 4: Verify training script uses correct environment"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Verify Training Script Integration")
    logger.info("=" * 60)
    
    # Check train_enhanced_clstm_ppo.py
    script_path = Path(__file__).parent / 'train_enhanced_clstm_ppo.py'
    
    if not script_path.exists():
        logger.error("❌ ERROR: train_enhanced_clstm_ppo.py not found")
        return False
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check imports
    checks = {
        'MultiLegOptionsEnvironment import': 'from src.multi_leg_options_env import MultiLegOptionsEnvironment' in content,
        'OptimizedHistoricalOptionsDataLoader import': 'from src.historical_options_data import OptimizedHistoricalOptionsDataLoader' in content,
        'Data loader initialization': 'self.data_loader = OptimizedHistoricalOptionsDataLoader' in content,
        'Environment class selection': 'env_class = MultiLegOptionsEnvironment if self.enable_multi_leg else WorkingOptionsEnvironment' in content,
        'Real data loading': 'data_loader=self.data_loader' in content or 'data_loader=data_loader' in content,
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        if passed:
            logger.info(f"   ✅ {check_name}")
        else:
            logger.error(f"   ❌ {check_name}")
            all_passed = False
    
    if all_passed:
        logger.info(f"\n✅ SUCCESS: Training script correctly configured")
        logger.info(f"   - Uses OptimizedHistoricalOptionsDataLoader")
        logger.info(f"   - Uses MultiLegOptionsEnvironment (when enabled)")
        logger.info(f"   - Passes data_loader to environment")
        return True
    else:
        logger.error(f"\n❌ ERROR: Training script configuration issues")
        return False


async def main():
    """Run all verification tests"""
    logger.info("\n" + "=" * 80)
    logger.info("REAL OPTIONS DATA LOADING VERIFICATION")
    logger.info("=" * 80)
    
    results = {}
    
    # Test 1: Data loader
    results['data_loader'], data_loader = await verify_data_loader()
    
    # Test 2: WorkingOptionsEnvironment
    results['working_env'] = await verify_working_environment(data_loader)
    
    # Test 3: MultiLegOptionsEnvironment
    results['multi_leg_env'] = await verify_multi_leg_environment(data_loader)
    
    # Test 4: Training script integration
    results['training_script'] = await verify_training_script_integration()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "⚠️ WARN/FAIL"
        logger.info(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("\n" + "=" * 80)
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("=" * 80)
        logger.info("\nConclusion:")
        logger.info("  - MultiLegOptionsEnvironment is correctly configured")
        logger.info("  - Real data loading is working (or falls back to synthetic)")
        logger.info("  - Training script uses the correct environment")
        logger.info("  - Ready for production training!")
    else:
        logger.info("\n" + "=" * 80)
        logger.info("⚠️ SOME TESTS FAILED OR WARNED")
        logger.info("=" * 80)
        logger.info("\nNote:")
        logger.info("  - If using demo API keys, real data loading may fail")
        logger.info("  - Environment will automatically fall back to synthetic data")
        logger.info("  - This is expected behavior and training will still work")


if __name__ == "__main__":
    asyncio.run(main())

