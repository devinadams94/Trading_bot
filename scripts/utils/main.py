import asyncio
import argparse
from loguru import logger
import os
import sys
from datetime import datetime

from config.config import TradingConfig
from src.trading_bot import AlpacaCLSTMPPOTrader

# Configure logger
logger.add(
    "logs/trading_bot_{time}.log",
    rotation="1 day",
    retention="1 week",
    level="INFO"
)

def create_env_file():
    """Create a template .env file if it doesn't exist"""
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("""# Alpaca API credentials
ALPACA_API_KEY=
ALPACA_SECRET_KEY=

# LLM API keys (optional)
CLAUDE_API_KEY=
OPENAI_API_KEY=
""")
        logger.info(".env template file created. Please fill in your API keys.")
        logger.info("After adding your keys, restart the application.")
        sys.exit(0)
    else:
        logger.debug(".env file already exists")

async def run_trading_bot(config, mode="live", backtest_start=None, backtest_end=None, 
                    use_checkpoint=True, clear_checkpoints=False):
    """Run the trading bot in specified mode"""
    bot = None
    try:
        # Apply checkpoint settings from command line
        config.checkpoint_enabled = use_checkpoint
        
        # Clear checkpoints if requested
        if clear_checkpoints and use_checkpoint:
            logger.info("Clearing existing checkpoints...")
            # Create a temporary checkpoint manager just to clear checkpoints
            from src.checkpoint_manager import CheckpointManager
            temp_checkpoint_manager = CheckpointManager(config)
            temp_checkpoint_manager.clear_checkpoints()
            logger.info("Checkpoints cleared")
        
        # Initialize the trading bot with checkpoint settings
        bot = AlpacaCLSTMPPOTrader(config, restore_from_checkpoint=use_checkpoint)
        
        if mode == "live":
            # Run live trading
            logger.info("Starting live trading...")
            await bot.start()
        elif mode == "paper":
            # Set paper trading mode in config
            config.alpaca_paper_trading = True
            logger.info("Starting paper trading...")
            await bot.start()
        elif mode == "backtest":
            # Run backtest
            if not backtest_start or not backtest_end:
                from datetime import timedelta
                backtest_start = (datetime.now().replace(hour=0, minute=0, second=0) - 
                                timedelta(days=30)).strftime("%Y-%m-%d")
                backtest_end = datetime.now().strftime("%Y-%m-%d")
            
            logger.info(f"Starting backtest from {backtest_start} to {backtest_end}...")
            await bot.backtest(backtest_start, backtest_end)
        else:
            logger.error(f"Unknown mode: {mode}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
        if bot:
            await bot.stop()
    except Exception as e:
        logger.error(f"Error running trading bot: {e}", exc_info=True)
        if bot:
            await bot.stop()
        raise

def main():
    """Main entry point"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="CLSTM-PPO Trading Bot with Alpaca")
    parser.add_argument(
        "--mode", 
        choices=["live", "paper", "backtest"], 
        default="paper",
        help="Trading mode: live, paper, or backtest"
    )
    parser.add_argument(
        "--backtest-start", 
        type=str,
        help="Start date for backtest (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--backtest-end", 
        type=str,
        help="End date for backtest (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--no-checkpoint", 
        action="store_true",
        help="Disable checkpoint loading and saving"
    )
    parser.add_argument(
        "--clear-checkpoints", 
        action="store_true",
        help="Clear existing checkpoints and start fresh"
    )
    
    args = parser.parse_args()
    
    # Configure logger level
    if args.debug:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
        logger.add(
            "logs/trading_bot_{time}.log",
            rotation="1 day",
            retention="1 week",
            level="DEBUG"
        )
    
    # Create .env file if it doesn't exist
    create_env_file()
    
    # Create a directory for logs if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Load config
    config = TradingConfig()
    
    # Check for API keys
    if not config.alpaca_api_key or not config.alpaca_secret_key:
        logger.error("Alpaca API keys not found in .env file")
        logger.info("Please add your Alpaca API keys to the .env file and restart")
        sys.exit(1)
    
    # Run the trading bot
    asyncio.run(
        run_trading_bot(
            config, 
            mode=args.mode, 
            backtest_start=args.backtest_start, 
            backtest_end=args.backtest_end,
            use_checkpoint=not args.no_checkpoint,
            clear_checkpoints=args.clear_checkpoints
        )
    )

if __name__ == "__main__":
    main()