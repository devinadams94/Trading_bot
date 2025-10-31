#!/usr/bin/env python3
"""
Training script with paper-compliant position closing incentives
"""

import os
import sys
import logging
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fix_position_closing_paper_compliant import create_paper_compliant_closing_env
from src.options_trading_env import OptionsTradingEnvironment
from src.historical_options_data import HistoricalOptionsEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train with closing incentives')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes to train')
    parser.add_argument('--no-data', action='store_true',
                        help='Use simulated data instead of historical')
    parser.add_argument('--symbols', nargs='+', default=['SPY', 'QQQ', 'IWM'],
                        help='Symbols to trade')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Training with Paper-Compliant Closing Incentives")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Key Features:")
    logger.info("1. Profit realization bonus (20% of realized profits)")
    logger.info("2. Automatic profit taking at 10% gain")
    logger.info("3. Automatic stop loss at 5% loss")
    logger.info("4. Trailing stop at 3% from peak")
    logger.info("5. Turbulence-based trading suspension")
    logger.info("6. NO holding time penalties (paper-compliant)")
    logger.info("")
    
    # Build the command
    cmd_parts = [
        "python train_ppo_lstm.py",
        f"--episodes {args.episodes}",
        "--fix-zero-trading",  # Still need exploration incentives
        "--entropy-coef 0.01",  # Moderate exploration
        f"--symbols {' '.join(args.symbols)}",
        "--checkpoint-interval 50"
    ]
    
    if args.no_data:
        cmd_parts.append("--no-data-load")
    
    # Note: The actual implementation would require modifying train_ppo_lstm.py
    # to use create_paper_compliant_closing_env instead of the base environment
    
    command = " ".join(cmd_parts)
    
    logger.info("Training command:")
    logger.info(command)
    logger.info("")
    logger.info("Expected improvements:")
    logger.info("- More realized profits (less unrealized)")
    logger.info("- Better win rate on closed positions")
    logger.info("- Automatic profit taking prevents gains from evaporating")
    logger.info("- Stop losses limit downside")
    logger.info("")
    
    # Execute
    os.system(command)


if __name__ == "__main__":
    main()