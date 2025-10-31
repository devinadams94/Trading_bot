#!/usr/bin/env python3
"""Calculate historical volatility from real options data"""

import asyncio
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging

from src.historical_options_data import HistoricalOptionsDataLoader

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def calculate_historical_volatility():
    """Calculate actual volatility from historical options data"""
    
    loader = HistoricalOptionsDataLoader(
        api_key=os.getenv('ALPACA_API_KEY'),
        api_secret=os.getenv('ALPACA_SECRET_KEY')
    )
    
    # Load data for multiple symbols
    symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA']
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=30)
    
    logger.info(f"Loading historical data from {start_date} to {end_date}")
    
    volatility_stats = {}
    
    for symbol in symbols:
        try:
            data = await loader.load_historical_options_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                use_cache=True
            )
            
            if symbol in data and not data[symbol].empty:
                df = data[symbol]
                
                # Calculate price changes for options
                df = df.sort_values('timestamp')
                
                # Group by option symbol to track individual contracts
                option_volatilities = []
                
                for opt_symbol in df['option_symbol'].unique():
                    opt_data = df[df['option_symbol'] == opt_symbol].copy()
                    
                    if len(opt_data) > 1:
                        # Calculate returns
                        opt_data['price'] = opt_data['last']
                        opt_data['returns'] = opt_data['price'].pct_change()
                        
                        # Filter out extreme outliers (>100% moves)
                        returns = opt_data['returns'].dropna()
                        returns = returns[abs(returns) < 1.0]
                        
                        if len(returns) > 0:
                            # Calculate volatility (standard deviation of returns)
                            vol = returns.std()
                            if not np.isnan(vol) and vol > 0:
                                option_volatilities.append(vol)
                
                if option_volatilities:
                    # Calculate statistics
                    volatility_stats[symbol] = {
                        'mean_volatility': np.mean(option_volatilities),
                        'median_volatility': np.median(option_volatilities),
                        'std_volatility': np.std(option_volatilities),
                        'min_volatility': np.min(option_volatilities),
                        'max_volatility': np.max(option_volatilities),
                        'sample_size': len(option_volatilities)
                    }
                    
                    logger.info(f"\n{symbol} Option Volatility Statistics:")
                    logger.info(f"  Mean volatility per step: {volatility_stats[symbol]['mean_volatility']:.4f}")
                    logger.info(f"  Median volatility per step: {volatility_stats[symbol]['median_volatility']:.4f}")
                    logger.info(f"  Volatility range: {volatility_stats[symbol]['min_volatility']:.4f} - {volatility_stats[symbol]['max_volatility']:.4f}")
                    logger.info(f"  Sample size: {volatility_stats[symbol]['sample_size']} contracts")
                
                # Also calculate underlying stock volatility for comparison
                if 'underlying_price' in df.columns:
                    stock_prices = df.groupby('timestamp')['underlying_price'].first()
                    stock_returns = stock_prices.pct_change().dropna()
                    stock_vol = stock_returns.std()
                    logger.info(f"  Underlying stock volatility: {stock_vol:.4f}")
                    
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    # Calculate overall statistics
    if volatility_stats:
        all_means = [stats['mean_volatility'] for stats in volatility_stats.values()]
        overall_mean = np.mean(all_means)
        
        logger.info(f"\nOVERALL STATISTICS:")
        logger.info(f"Average option volatility across all symbols: {overall_mean:.4f}")
        logger.info(f"Recommended volatility for simulation: {overall_mean:.4f}")
        logger.info(f"Recommended mean return: {overall_mean * 0.1:.4f} (10% of volatility)")
        
        # Save to file for use in trading environment
        with open('config/historical_volatility.json', 'w') as f:
            import json
            json.dump({
                'overall_mean_volatility': overall_mean,
                'symbol_volatilities': volatility_stats,
                'recommended_settings': {
                    'option_price_mean_change': overall_mean * 0.1,
                    'option_price_volatility': overall_mean,
                    'atm_multiplier': 1.5
                }
            }, f, indent=2)
        logger.info("\nVolatility statistics saved to config/historical_volatility.json")

if __name__ == "__main__":
    asyncio.run(calculate_historical_volatility())