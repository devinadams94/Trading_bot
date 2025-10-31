import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import asyncio
import json
from tqdm import tqdm
import torch
import wandb

from src.options_trading_env import OptionsTradingEnvironment
from src.options_data_collector import AlpacaOptionsDataCollector, OptionsDataSimulator
from src.options_ppo_agent import OptionsPPOAgent
from src.feature_extraction import FeatureExtractor
from config.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptionsTrainingPipeline:
    def __init__(
        self,
        config_path: str = 'config/config.yaml',
        use_simulated_data: bool = True,
        checkpoint_dir: str = 'checkpoints/options',
        log_wandb: bool = False
    ):
        self.config = load_config(config_path)
        self.use_simulated_data = use_simulated_data
        self.checkpoint_dir = checkpoint_dir
        self.log_wandb = log_wandb
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize components
        self._initialize_components()
        
        # Initialize wandb if requested
        if self.log_wandb:
            wandb.init(
                project="options-trading-bot",
                config=self.config,
                name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def _initialize_components(self):
        # Initialize environment
        self.env = OptionsTradingEnvironment(
            initial_capital=self.config.get('initial_capital', 100000),
            max_positions=self.config.get('max_positions', 10),
            commission=self.config.get('commission', 0.65),
            min_capital_per_trade=self.config.get('min_capital_per_trade', 1000),
            max_capital_per_trade=self.config.get('max_capital_per_trade', 10000)
        )
        
        # Initialize data collector
        if self.use_simulated_data:
            self.data_collector = OptionsDataSimulator(
                base_volatility=self.config.get('base_volatility', 0.2)
            )
        else:
            api_key = os.getenv('ALPACA_API_KEY')
            api_secret = os.getenv('ALPACA_SECRET_KEY')
            self.data_collector = AlpacaOptionsDataCollector(
                api_key=api_key,
                api_secret=api_secret,
                base_url=self.config.get('alpaca_base_url', 'https://paper-api.alpaca.markets')
            )
        
        # Initialize PPO agent
        self.agent = OptionsPPOAgent(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space.n,
            learning_rate=self.config.get('learning_rate', 3e-4),
            gamma=self.config.get('gamma', 0.99),
            batch_size=self.config.get('batch_size', 64),
            n_epochs=self.config.get('n_epochs', 10)
        )
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor()
        
        # Training parameters
        self.num_episodes = self.config.get('num_episodes', 1000)
        self.max_steps_per_episode = self.config.get('max_steps_per_episode', 1000)
        self.save_frequency = self.config.get('save_frequency', 50)
        self.eval_frequency = self.config.get('eval_frequency', 10)
    
    async def collect_training_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        training_data = {}
        
        if self.use_simulated_data:
            logger.info("Generating simulated training data...")
            for symbol in symbols:
                # Simulate stock price data
                dates = pd.date_range(start=start_date, end=end_date, freq='5T')
                prices = self._simulate_stock_prices(symbol, len(dates))
                
                df = pd.DataFrame({
                    'timestamp': dates,
                    'open': prices,
                    'high': prices * np.random.uniform(1.0, 1.01, len(prices)),
                    'low': prices * np.random.uniform(0.99, 1.0, len(prices)),
                    'close': prices * np.random.uniform(0.995, 1.005, len(prices)),
                    'volume': np.random.randint(1000, 100000, len(prices))
                })
                
                training_data[symbol] = df
        else:
            logger.info("Collecting real training data from Alpaca...")
            async with self.data_collector as collector:
                for symbol in symbols:
                    # Collect historical data
                    bars = await collector.get_historical_bars(
                        symbol=symbol,
                        start=start_date,
                        end=end_date,
                        timeframe='5Min'
                    )
                    training_data[symbol] = bars
        
        return training_data
    
    def _simulate_stock_prices(self, symbol: str, length: int) -> np.ndarray:
        # Simple geometric Brownian motion
        initial_price = {
            'SPY': 450,
            'AAPL': 180,
            'MSFT': 380,
            'NVDA': 500,
            'TSLA': 250
        }.get(symbol, 100)
        
        returns = np.random.normal(0.0002, 0.02, length)
        prices = initial_price * np.exp(np.cumsum(returns))
        
        return prices
    
    def train(self, symbols: List[str] = None):
        if symbols is None:
            symbols = self.config.get('symbols', ['SPY', 'AAPL', 'MSFT', 'NVDA', 'TSLA'])
        
        logger.info(f"Starting training with symbols: {symbols}")
        
        # Training metrics
        episode_rewards = []
        episode_lengths = []
        training_losses = []
        
        for episode in range(self.num_episodes):
            # Reset environment
            obs = self.env.reset()
            
            # Episode variables
            total_reward = 0
            step = 0
            
            # Generate options chain for this episode
            if self.use_simulated_data:
                current_symbol = np.random.choice(symbols)
                stock_price = np.random.uniform(100, 500)
                options_chain = self.data_collector.simulate_options_chain(
                    symbol=current_symbol,
                    stock_price=stock_price,
                    num_strikes=20,
                    num_expirations=4
                )
            
            for step in range(self.max_steps_per_episode):
                # Update observation with current market data
                obs = self._update_observation(obs, options_chain, step)
                
                # Get action from agent
                action, info = self.agent.act(obs, deterministic=False)
                
                # Execute action in environment
                next_obs, reward, done, env_info = self.env.step(action)
                
                # Store transition
                self.agent.store_transition(obs, action, reward, next_obs, done, info)
                
                # Update variables
                obs = next_obs
                total_reward += reward
                
                if done:
                    break
                
                # Update options data periodically
                if step % 10 == 0 and self.use_simulated_data:
                    # Simulate price movement
                    stock_price *= np.random.uniform(0.99, 1.01)
                    options_chain = self.data_collector.simulate_options_chain(
                        symbol=current_symbol,
                        stock_price=stock_price,
                        num_strikes=20,
                        num_expirations=4
                    )
            
            # Store episode metrics
            episode_rewards.append(total_reward)
            episode_lengths.append(step + 1)
            
            # Train agent
            if len(self.agent.buffer) >= self.agent.batch_size:
                train_metrics = self.agent.train()
                if train_metrics:
                    training_losses.append(train_metrics['total_loss'])
                    
                    if self.log_wandb:
                        wandb.log({
                            'episode': episode,
                            'episode_reward': total_reward,
                            'episode_length': step + 1,
                            'total_loss': train_metrics['total_loss'],
                            'policy_loss': train_metrics['policy_loss'],
                            'value_loss': train_metrics['value_loss'],
                            'entropy': train_metrics['entropy']
                        })
            
            # Logging
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                avg_length = np.mean(episode_lengths[-100:]) if episode_lengths else 0
                avg_loss = np.mean(training_losses[-100:]) if training_losses else 0
                
                logger.info(
                    f"Episode {episode}/{self.num_episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Avg Length: {avg_length:.0f} | "
                    f"Avg Loss: {avg_loss:.4f}"
                )
            
            # Save checkpoint
            if episode % self.save_frequency == 0 and episode > 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"options_ppo_episode_{episode}.pt"
                )
                self.agent.save(checkpoint_path)
                logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Evaluation
            if episode % self.eval_frequency == 0 and episode > 0:
                eval_reward = self.evaluate(num_episodes=5)
                logger.info(f"Evaluation reward: {eval_reward:.2f}")
                
                if self.log_wandb:
                    wandb.log({
                        'episode': episode,
                        'eval_reward': eval_reward
                    })
        
        # Save final model
        final_path = os.path.join(self.checkpoint_dir, "options_ppo_final.pt")
        self.agent.save(final_path)
        logger.info(f"Training complete. Final model saved: {final_path}")
        
        if self.log_wandb:
            wandb.finish()
    
    def _update_observation(
        self,
        obs: Dict[str, np.ndarray],
        options_chain: List,
        step: int
    ) -> Dict[str, np.ndarray]:
        # Update options chain data in observation
        if options_chain:
            # Sort by volume and select top 20
            sorted_options = sorted(options_chain, key=lambda x: x.volume, reverse=True)[:20]
            
            # Create options features matrix
            options_features = []
            for opt in sorted_options:
                features = [
                    opt.strike,
                    opt.bid,
                    opt.ask,
                    opt.last_price,
                    opt.volume,
                    opt.open_interest,
                    opt.implied_volatility,
                    opt.delta,
                    opt.gamma,
                    opt.theta,
                    opt.vega,
                    opt.rho,
                    1.0 if opt.option_type == 'call' else 0.0,
                    (opt.expiration - datetime.now()).days,
                    (opt.bid + opt.ask) / 2  # Mid price
                ]
                options_features.append(features)
            
            # Pad if necessary
            while len(options_features) < 20:
                options_features.append([0] * 15)
            
            obs['options_chain'] = np.array(options_features, dtype=np.float32)
        
        return obs
    
    def evaluate(self, num_episodes: int = 10) -> float:
        total_rewards = []
        
        for _ in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Generate test options data
                if self.use_simulated_data:
                    stock_price = np.random.uniform(100, 500)
                    options_chain = self.data_collector.simulate_options_chain(
                        symbol='SPY',
                        stock_price=stock_price,
                        num_strikes=20,
                        num_expirations=4
                    )
                    obs = self._update_observation(obs, options_chain, 0)
                
                # Get action (deterministic for evaluation)
                action, _ = self.agent.act(obs, deterministic=True)
                
                # Step environment
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward
            
            total_rewards.append(episode_reward)
        
        return np.mean(total_rewards)


def main():
    parser = argparse.ArgumentParser(description='Train Options Trading Bot')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--simulated', action='store_true', help='Use simulated data for training')
    parser.add_argument('--wandb', action='store_true', help='Log to Weights & Biases')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/options', help='Checkpoint directory')
    parser.add_argument('--symbols', nargs='+', default=['SPY', 'AAPL', 'MSFT'], help='Symbols to trade')
    
    args = parser.parse_args()
    
    # Create training pipeline
    pipeline = OptionsTrainingPipeline(
        config_path=args.config,
        use_simulated_data=args.simulated,
        checkpoint_dir=args.checkpoint_dir,
        log_wandb=args.wandb
    )
    
    # Override config with command line args
    pipeline.num_episodes = args.episodes
    
    # Start training
    pipeline.train(symbols=args.symbols)


if __name__ == '__main__':
    main()