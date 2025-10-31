import os
import yaml
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SymbolsConfig:
    """Loads and manages trading symbols configuration"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'symbols_config.yaml')
        
        self.config_path = config_path
        self.symbols_data = self._load_symbols_config()
    
    def _load_symbols_config(self) -> Dict[str, Any]:
        """Load symbols configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading symbols config: {e}")
            # Return default minimal config
            return {
                'default_training_symbols': ['SPY', 'QQQ', 'TSLA', 'AAPL'],
                'default_paper_symbols': ['SPY', 'QQQ', 'TSLA', 'AAPL', 'META']
            }
    
    def get_default_symbols(self, mode: str = 'training') -> List[str]:
        """Get default symbols for a specific mode"""
        if mode == 'training':
            return self.symbols_data.get('default_training_symbols', ['SPY', 'QQQ'])
        elif mode == 'paper':
            return self.symbols_data.get('default_paper_symbols', ['SPY', 'QQQ'])
        elif mode == 'high_volume':
            return self.symbols_data.get('high_volume_options', ['SPY', 'QQQ'])
        else:
            return self.symbols_data.get('default_training_symbols', ['SPY', 'QQQ'])
    
    def get_symbols_by_category(self, category: str) -> List[str]:
        """Get symbols by category (e.g., 'mega_tech', 'volatile_tech')"""
        return self.symbols_data.get(category, [])
    
    def get_symbols_for_strategy(self, strategy: str) -> List[str]:
        """Get recommended symbols for a specific options strategy"""
        strategies = self.symbols_data.get('strategies', {})
        return strategies.get(strategy, [])
    
    def get_symbols_by_market_cap(self, cap_size: str) -> List[str]:
        """Get symbols by market cap category"""
        market_caps = self.symbols_data.get('by_market_cap', {})
        return market_caps.get(cap_size, [])
    
    def get_symbols_by_iv(self, iv_level: str) -> List[str]:
        """Get symbols by typical IV range"""
        iv_ranges = self.symbols_data.get('typical_iv_range', {})
        return iv_ranges.get(iv_level, [])
    
    def get_liquidity_score(self, symbol: str) -> int:
        """Get liquidity score for a symbol (1-10)"""
        scores = self.symbols_data.get('liquidity_scores', {})
        return scores.get(symbol, 5)  # Default score of 5
    
    def get_all_available_symbols(self) -> List[str]:
        """Get all unique symbols from all categories"""
        all_symbols = set()
        
        # Add from main categories
        for category in ['index_etfs', 'mega_tech', 'volatile_tech', 'ev_energy', 
                        'meme_stocks', 'financials', 'commodities', 'biotech', 'retail']:
            all_symbols.update(self.symbols_data.get(category, []))
        
        # Add from market cap categories
        market_caps = self.symbols_data.get('by_market_cap', {})
        for cap_list in market_caps.values():
            all_symbols.update(cap_list)
        
        return sorted(list(all_symbols))
    
    def filter_symbols_by_liquidity(self, symbols: List[str], min_score: int = 7) -> List[str]:
        """Filter symbols by minimum liquidity score"""
        scores = self.symbols_data.get('liquidity_scores', {})
        return [s for s in symbols if scores.get(s, 0) >= min_score]
    
    def get_high_volatility_symbols(self) -> List[str]:
        """Get symbols with high volatility (good for options premium)"""
        high_vol = []
        high_vol.extend(self.symbols_data.get('volatile_tech', []))
        high_vol.extend(self.symbols_data.get('meme_stocks', []))
        
        # Add high IV symbols
        iv_ranges = self.symbols_data.get('typical_iv_range', {})
        high_vol.extend(iv_ranges.get('high_iv', []))
        high_vol.extend(iv_ranges.get('extreme_iv', []))
        
        return list(set(high_vol))  # Remove duplicates
    
    def get_training_recommendations(self, 
                                   include_indices: bool = True,
                                   include_memes: bool = False,
                                   min_liquidity: int = 7) -> List[str]:
        """Get recommended symbols for training based on criteria"""
        recommendations = []
        
        # Always include top indices if requested
        if include_indices:
            recommendations.extend(['SPY', 'QQQ', 'IWM'])
        
        # Add high liquidity tech stocks
        tech_stocks = self.symbols_data.get('mega_tech', []) + self.symbols_data.get('volatile_tech', [])
        liquid_tech = self.filter_symbols_by_liquidity(tech_stocks, min_liquidity)
        recommendations.extend(liquid_tech[:10])  # Top 10
        
        # Optionally include meme stocks
        if include_memes:
            memes = self.symbols_data.get('meme_stocks', [])
            recommendations.extend(memes[:3])  # Top 3 memes
        
        # Remove duplicates and return
        return list(dict.fromkeys(recommendations))  # Preserves order while removing duplicates


# Convenience functions
def get_default_training_symbols() -> List[str]:
    """Quick function to get default training symbols"""
    config = SymbolsConfig()
    return config.get_default_symbols('training')


def get_default_paper_symbols() -> List[str]:
    """Quick function to get default paper trading symbols"""
    config = SymbolsConfig()
    return config.get_default_symbols('paper')


def get_high_volume_symbols() -> List[str]:
    """Quick function to get high volume options symbols"""
    config = SymbolsConfig()
    return config.get_default_symbols('high_volume')


def get_symbols_for_strategy(strategy: str) -> List[str]:
    """Quick function to get symbols for a specific strategy"""
    config = SymbolsConfig()
    return config.get_symbols_for_strategy(strategy)