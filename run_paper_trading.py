#!/usr/bin/env python3
"""
Simple script to run paper trading with the trained model
"""

import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_best_model():
    """Find the best available model checkpoint"""
    
    checkpoint_dir = Path("checkpoints/enhanced_clstm_ppo")
    
    # Priority order for model selection
    model_candidates = [
        "best_composite_model.pt",
        "best_win_rate_model.pt", 
        "best_profit_rate_model.pt",
        "latest_model.pt",
        "working_model_for_testing.pt"
    ]
    
    for model_name in model_candidates:
        model_path = checkpoint_dir / model_name
        if model_path.exists():
            logger.info(f"✅ Found model: {model_path}")
            return str(model_path)
    
    # Look for any .pt files
    pt_files = list(checkpoint_dir.glob("*.pt"))
    if pt_files:
        model_path = pt_files[0]
        logger.info(f"✅ Found model: {model_path}")
        return str(model_path)
    
    logger.error("❌ No model checkpoints found!")
    logger.error("   Please run training first or create a test model with:")
    logger.error("   python test_model_saving.py")
    return None

def run_paper_trading(model_path, symbols=None, capital=100000, duration=60):
    """Run paper trading with the specified model"""
    
    if symbols is None:
        symbols = ['SPY', 'QQQ', 'IWM']
    
    # Build command
    cmd_parts = [
        sys.executable,
        "paper_trading_bot.py",
        "--model", model_path,
        "--symbols"] + symbols + [
        "--capital", str(capital),
        "--duration", str(duration)
    ]
    
    cmd = " ".join(cmd_parts)
    logger.info(f"🚀 Running paper trading:")
    logger.info(f"   Command: {cmd}")
    logger.info(f"   Model: {model_path}")
    logger.info(f"   Symbols: {symbols}")
    logger.info(f"   Capital: ${capital:,}")
    logger.info(f"   Duration: {duration} minutes")
    
    # Run the command
    import subprocess
    try:
        result = subprocess.run(cmd_parts, check=True)
        logger.info("✅ Paper trading completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Paper trading failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        logger.error("❌ paper_trading_bot.py not found!")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run Paper Trading with Trained Model')
    parser.add_argument('--model', type=str, help='Path to model checkpoint (auto-detected if not specified)')
    parser.add_argument('--symbols', nargs='+', default=['SPY', 'QQQ', 'IWM'], 
                        help='Symbols to trade (default: SPY QQQ IWM)')
    parser.add_argument('--capital', type=float, default=100000,
                        help='Initial capital (default: 100000)')
    parser.add_argument('--duration', type=int, default=60,
                        help='Trading duration in minutes (default: 60)')
    parser.add_argument('--create-test-model', action='store_true',
                        help='Create a test model first if none exists')
    
    args = parser.parse_args()
    
    logger.info("📈 Paper Trading Runner")
    logger.info("=" * 40)
    
    # Find or create model
    model_path = args.model
    if not model_path:
        model_path = find_best_model()
        
        if not model_path and args.create_test_model:
            logger.info("🧪 Creating test model...")
            import subprocess
            try:
                subprocess.run([sys.executable, "test_model_saving.py"], check=True)
                model_path = find_best_model()
            except subprocess.CalledProcessError:
                logger.error("❌ Failed to create test model")
                return False
    
    if not model_path:
        logger.error("❌ No model available for trading!")
        logger.error("   Options:")
        logger.error("   1. Run training: python train_enhanced_clstm_ppo.py")
        logger.error("   2. Create test model: python test_model_saving.py")
        logger.error("   3. Use --create-test-model flag")
        return False
    
    # Verify model exists
    if not Path(model_path).exists():
        logger.error(f"❌ Model not found: {model_path}")
        return False
    
    # Run paper trading
    success = run_paper_trading(
        model_path=model_path,
        symbols=args.symbols,
        capital=args.capital,
        duration=args.duration
    )
    
    if success:
        logger.info("🎉 Paper trading session completed!")
    else:
        logger.error("💥 Paper trading session failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
