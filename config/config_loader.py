import os
import yaml
from typing import Dict, Any, List
from config.config import TradingConfig
from config.symbols_loader import SymbolsConfig


def load_config(config_path: str = None, mode: str = 'training') -> Dict[str, Any]:
    """Load configuration from YAML file or use default TradingConfig
    
    Args:
        config_path: Path to config YAML file
        mode: 'training', 'paper', or 'live' to determine default symbols
    """
    
    # Load symbols configuration
    symbols_config = SymbolsConfig()
    
    if config_path and os.path.exists(config_path) and config_path.endswith('.yaml'):
        # Load from YAML file
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Merge with environment variables
        config_dict['alpaca_api_key'] = os.getenv('ALPACA_API_KEY', config_dict.get('alpaca_api_key'))
        config_dict['alpaca_secret_key'] = os.getenv('ALPACA_SECRET_KEY', config_dict.get('alpaca_secret_key'))
        
        # Use symbols from config file or default symbols
        if 'symbols' not in config_dict or not config_dict['symbols']:
            config_dict['symbols'] = symbols_config.get_default_symbols(mode)
        
        return config_dict
    else:
        # Use default TradingConfig class
        config = TradingConfig()
        
        # Get default symbols based on mode
        default_symbols = symbols_config.get_default_symbols(mode)
        
        # Convert to dictionary
        config_dict = {
            'alpaca_api_key': config.alpaca_api_key,
            'alpaca_secret_key': config.alpaca_secret_key,
            'alpaca_base_url': 'https://paper-api.alpaca.markets',
            'alpaca_paper_trading': config.alpaca_paper_trading,
            'symbols': default_symbols,  # Use symbols from symbols_config
            'initial_capital': 100000,
            'max_positions': 10,
            'position_size_limit': config.max_position_size,
            'learning_rate': config.learning_rate,
            'gamma': config.gamma,
            'batch_size': 64,
            'n_epochs': 10,
            'num_episodes': 1000,
            'save_frequency': 50,
            'eval_frequency': 10,
            'commission': 0.65,
            'base_volatility': 0.2,
            'clstm_learning_rate': 1e-3,
            'pretrain_epochs': 50,
            'data_days': 60,
            'episode_length': 390,
        }
        
        return config_dict


def get_training_symbols() -> List[str]:
    """Get default symbols for training"""
    symbols_config = SymbolsConfig()
    return symbols_config.get_default_symbols('training')


def get_paper_trading_symbols() -> List[str]:
    """Get default symbols for paper trading"""
    symbols_config = SymbolsConfig()
    return symbols_config.get_default_symbols('paper')


def get_high_volatility_symbols() -> List[str]:
    """Get high volatility symbols (good for options)"""
    symbols_config = SymbolsConfig()
    return symbols_config.get_high_volatility_symbols()


def get_liquid_symbols(min_score: int = 8) -> List[str]:
    """Get highly liquid symbols"""
    symbols_config = SymbolsConfig()
    all_symbols = symbols_config.get_all_available_symbols()
    return symbols_config.filter_symbols_by_liquidity(all_symbols, min_score)