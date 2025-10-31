import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

class TradingConfig:
    # Alpaca API credentials
    alpaca_api_key = os.getenv('ALPACA_API_KEY')
    alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')
    alpaca_paper_trading = True  # Set to False for live trading
    
    # Symbols to trade (dynamically loaded from symbols_config.yaml)
    from config.symbols_loader import get_default_training_symbols
    symbols = get_default_training_symbols()  # ['SPY', 'QQQ', 'TSLA', 'AAPL', 'META', 'NVDA', 'AMD', 'PLTR']
    
    # Trading parameters
    max_position_size = 0.1  # Max position size as fraction of portfolio
    max_trades_per_day = 10  # Maximum number of trades per day
    use_options = True       # Set to True to enable options trading
    
    # Data parameters
    historical_days = 252  # ~1 year of trading days
    timeframe = '5Min'     # 5-minute bars for intraday trading
    
    # Model parameters
    chronos_model = None   # "amazon/chronos-bolt-small" - disabled by default
    finbert_model = "ProsusAI/finbert"
    yolo_model = None      # Disabled by default
    fingpt_model = None    # Optional: "AI4Finance-Foundation/FinGPT-v3.3"
    
    # CLSTM parameters
    feature_dim = 512
    clstm_hidden_units = 10
    clstm_layers = 5
    clstm_output_dim = 256
    
    # PPO parameters
    actor_layers = [64, 64]
    critic_layers = [64, 64]
    learning_rate = 3e-4
    gamma = 0.99
    
    # LLM integration
    use_llm_advisory = False  # Set to True to enable LLM integration
    
    # Testing parameters
    enforce_market_hours = False  # Set to False to allow trading anytime (useful for testing)
    
    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    models_dir = base_dir / "models"
    checkpoints_dir = base_dir / "checkpoints"
    
    # Ensure directories exist
    data_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)
    
    # Checkpoint settings
    checkpoint_enabled = True
    checkpoint_interval = 300  # Save checkpoint every 5 minutes
    auto_recover = True  # Automatically recover from checkpoint on startup