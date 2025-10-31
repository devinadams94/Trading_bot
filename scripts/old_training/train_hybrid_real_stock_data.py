#!/usr/bin/env python3
"""
Hybrid training using real stock data from Alpaca with simulated options
This provides more realistic training than pure simulation while avoiding paid options data
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
from loguru import logger
import argparse
from typing import List, Dict, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.options_trading_env import OptionContract, OptionsGreeksCalculator
from src.options_clstm_ppo import OptionsCLSTMPPOAgent
from src.feature_extraction import FeatureExtractor
from dotenv import load_dotenv

load_dotenv()


class HybridRealDataTrainer:
    """Trains using real stock data with realistically simulated options"""
    
    def __init__(self, symbols: List[str], lookback_days: int = 60):
        self.symbols = symbols
        self.lookback_days = lookback_days
        
        # Initialize Alpaca API
        self.api = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            'https://paper-api.alpaca.markets',
            api_version='v2'
        )
        
        # Greeks calculator
        self.greeks_calc = OptionsGreeksCalculator()
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor()
        
        # Cache for stock data
        self.stock_data_cache = {}
        
    def load_stock_data(self, symbol: str) -> pd.DataFrame:
        """Load real historical stock data from Alpaca"""
        if symbol in self.stock_data_cache:
            return self.stock_data_cache[symbol]
            
        logger.info(f"Loading {self.lookback_days} days of {symbol} stock data...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days * 1.5)  # Extra for weekends
        
        try:
            # Get daily bars for technical indicators
            daily_bars = self.api.get_bars(
                symbol,
                '1Day',
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d')
            ).df
            
            # Get minute bars for recent price action
            minute_bars = self.api.get_bars(
                symbol,
                '1Min',
                start=(end_date - timedelta(days=5)).strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                limit=10000
            ).df
            
            # Calculate technical indicators
            daily_bars['returns'] = daily_bars['close'].pct_change()
            daily_bars['volatility'] = daily_bars['returns'].rolling(20).std() * np.sqrt(252)
            daily_bars['sma_20'] = daily_bars['close'].rolling(20).mean()
            daily_bars['sma_50'] = daily_bars['close'].rolling(50).mean()
            daily_bars['rsi'] = self.feature_extractor.calculate_rsi(daily_bars['close'])
            
            self.stock_data_cache[symbol] = {
                'daily': daily_bars,
                'minute': minute_bars
            }
            
            logger.success(f"✓ Loaded {len(daily_bars)} days of data for {symbol}")
            return self.stock_data_cache[symbol]
            
        except Exception as e:
            logger.error(f"Failed to load stock data for {symbol}: {e}")
            return None
    
    def simulate_realistic_options_chain(
        self,
        symbol: str,
        current_price: float,
        volatility: float,
        date: datetime
    ) -> List[OptionContract]:
        """Generate realistic options chain based on real stock data"""
        
        options_chain = []
        
        # Generate strikes around current price
        strikes = []
        strike_interval = 1 if current_price < 50 else 5 if current_price < 500 else 10
        
        for i in range(-10, 11):
            strike = round(current_price + i * strike_interval)
            if strike > 0:
                strikes.append(strike)
        
        # Generate expirations (weekly for next 4 weeks, then monthly)
        expirations = []
        
        # Weekly expirations (Fridays)
        for weeks in [1, 2, 3, 4]:
            exp_date = date + timedelta(weeks=weeks)
            while exp_date.weekday() != 4:  # Find Friday
                exp_date += timedelta(days=1)
            expirations.append(exp_date)
        
        # Monthly expirations (3rd Friday)
        for months in [2, 3]:
            exp_date = date + timedelta(days=30*months)
            # Find 3rd Friday
            exp_date = exp_date.replace(day=1)
            fridays = 0
            while fridays < 3:
                if exp_date.weekday() == 4:
                    fridays += 1
                if fridays < 3:
                    exp_date += timedelta(days=1)
            expirations.append(exp_date)
        
        # Generate options for each strike/expiration
        for strike in strikes:
            for expiration in expirations:
                days_to_expiry = (expiration - date).days
                
                if days_to_expiry <= 0:
                    continue
                
                # Use real volatility from stock data
                time_to_expiry = days_to_expiry / 365.0
                
                for option_type in ['call', 'put']:
                    # Calculate Black-Scholes price
                    bs_price = self._black_scholes_price(
                        current_price, strike, time_to_expiry, 0.05, volatility, option_type
                    )
                    
                    # Add realistic bid-ask spread
                    spread_pct = 0.02 + 0.03 * abs(strike - current_price) / current_price
                    spread = bs_price * spread_pct
                    
                    bid = max(0.01, bs_price - spread/2)
                    ask = bs_price + spread/2
                    
                    # Calculate Greeks
                    greeks = self.greeks_calc.calculate_greeks(
                        current_price, strike, time_to_expiry, 0.05, volatility, option_type
                    )
                    
                    # Simulate realistic volume/OI based on moneyness
                    moneyness = abs(strike - current_price) / current_price
                    volume_base = 1000 * np.exp(-moneyness * 10)
                    volume = int(volume_base * np.random.uniform(0.5, 1.5))
                    open_interest = int(volume * np.random.uniform(5, 20))
                    
                    option = OptionContract(
                        symbol=f"{symbol}{expiration.strftime('%y%m%d')}{'C' if option_type == 'call' else 'P'}{int(strike*1000):08d}",
                        strike=strike,
                        expiration=expiration,
                        option_type=option_type,
                        bid=bid,
                        ask=ask,
                        last_price=(bid + ask) / 2,
                        volume=volume,
                        open_interest=open_interest,
                        implied_volatility=volatility + np.random.normal(0, 0.02),
                        delta=greeks['delta'],
                        gamma=greeks['gamma'],
                        theta=greeks['theta'],
                        vega=greeks['vega'],
                        rho=greeks['rho']
                    )
                    
                    options_chain.append(option)
        
        return options_chain
    
    def _black_scholes_price(
        self, S: float, K: float, T: float, r: float, sigma: float, option_type: str
    ) -> float:
        """Calculate Black-Scholes option price"""
        from scipy.stats import norm
        
        if T <= 0:
            return max(0, S - K) if option_type == 'call' else max(0, K - S)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return max(0.01, price)
    
    def train(self, episodes: int = 1000, save_path: str = "checkpoints/hybrid_real_data.pt"):
        """Train the agent using real stock data with simulated options"""
        
        logger.info("=== Starting Hybrid Real Data Training ===")
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Episodes: {episodes}")
        
        # Load all stock data first
        stock_data = {}
        for symbol in self.symbols:
            data = self.load_stock_data(symbol)
            if data:
                stock_data[symbol] = data
        
        if not stock_data:
            logger.error("Failed to load any stock data!")
            return
        
        # Initialize environment and agent
        from src.options_trading_env import OptionsTradingEnvironment
        
        env = OptionsTradingEnvironment(initial_capital=100000)
        
        agent = OptionsCLSTMPPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space.n
        )
        
        # Training loop
        episode_rewards = []
        
        for episode in range(episodes):
            # Random symbol for this episode
            symbol = np.random.choice(list(stock_data.keys()))
            data = stock_data[symbol]
            
            # Random starting point in historical data
            daily_df = data['daily']
            start_idx = np.random.randint(20, len(daily_df) - 20)
            
            obs = env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < 100:
                # Current market data
                current_data = daily_df.iloc[start_idx + step // 10]  # Move forward slowly
                current_price = current_data['close']
                volatility = current_data.get('volatility', 0.2)
                
                # Generate realistic options chain
                options_chain = self.simulate_realistic_options_chain(
                    symbol, current_price, volatility, datetime.now()
                )
                
                # Update observation with real market features
                obs['price_history'] = daily_df.iloc[start_idx-19:start_idx+1][
                    ['open', 'high', 'low', 'close', 'volume']
                ].values.astype(np.float32)
                
                obs['technical_indicators'] = np.array([
                    current_data.get('rsi', 50),
                    current_data.get('sma_20', current_price),
                    current_data.get('sma_50', current_price),
                    volatility * 100,
                    current_data['volume'],
                    # Add more indicators as needed
                ] + [0] * 15, dtype=np.float32)[:20]  # Ensure 20 features
                
                # Convert options chain to observation format
                if options_chain:
                    sorted_options = sorted(options_chain, 
                                          key=lambda x: x.volume * (1 - abs(x.strike - current_price)/current_price),
                                          reverse=True)[:20]
                    
                    options_features = []
                    for opt in sorted_options:
                        features = [
                            opt.strike, opt.bid, opt.ask, opt.last_price, opt.volume,
                            opt.open_interest, opt.implied_volatility, opt.delta,
                            opt.gamma, opt.theta, opt.vega, opt.rho,
                            1.0 if opt.option_type == 'call' else 0.0,
                            (opt.expiration - datetime.now()).days,
                            current_price
                        ]
                        options_features.append(features)
                    
                    while len(options_features) < 20:
                        options_features.append([0] * 15)
                    
                    obs['options_chain'] = np.array(options_features[:20], dtype=np.float32)
                
                # Agent action
                action, act_info = agent.act(obs)
                
                # Step environment
                next_obs, reward, done, info = env.step(action)
                
                # Store transition
                agent.store_transition(obs, action, reward, next_obs, done, act_info)
                
                obs = next_obs
                episode_reward += reward
                step += 1
            
            episode_rewards.append(episode_reward)
            
            # Train agent
            if len(agent.buffer) >= agent.batch_size:
                agent.train()
            
            # Log progress
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                logger.info(f"Episode {episode}/{episodes}: Avg Reward: {avg_reward:.2f}")
                logger.info(f"  Using real {symbol} data from {daily_df.index[start_idx].date()}")
        
        # Save model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        agent.save(save_path)
        logger.success(f"✓ Model saved to {save_path}")
        
        return agent


def main():
    parser = argparse.ArgumentParser(description='Hybrid training with real stock data')
    parser.add_argument('--symbols', nargs='+', default=['SPY', 'QQQ', 'IWM', 'DIA'],
                       help='Symbols to train on')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--lookback-days', type=int, default=60,
                       help='Days of historical data to load')
    
    args = parser.parse_args()
    
    trainer = HybridRealDataTrainer(
        symbols=args.symbols,
        lookback_days=args.lookback_days
    )
    
    trainer.train(episodes=args.episodes)


if __name__ == "__main__":
    main()