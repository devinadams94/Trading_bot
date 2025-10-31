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

from src.historical_options_data import HistoricalOptionsDataLoader, HistoricalOptionsEnvironment
from src.options_clstm_ppo import OptionsCLSTMPPOAgent
from src.feature_extraction import MarketFeatureExtractor as FeatureExtractor
from config.config_loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealDataOptionsTrainingPipeline:
    """Training pipeline using real historical options data from Alpaca"""
    
    def __init__(
        self,
        config_path: str = 'config/config.yaml',
        checkpoint_dir: str = 'checkpoints/options_real_data',
        log_wandb: bool = False
    ):
        self.config = load_config(config_path)
        self.checkpoint_dir = checkpoint_dir
        self.log_wandb = log_wandb
        
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs('data/options_cache', exist_ok=True)
        
        # Initialize components
        self._initialize_components()
        
        # Initialize wandb if requested
        if self.log_wandb:
            wandb.init(
                project="options-real-data-training",
                config=self.config,
                name=f"real_data_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def _initialize_components(self):
        # Initialize data loader
        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_SECRET_KEY')
        
        if not api_key or not api_secret:
            raise ValueError("Alpaca API credentials not found in environment variables")
        
        self.data_loader = HistoricalOptionsDataLoader(
            api_key=api_key,
            api_secret=api_secret,
            base_url=self.config.get('alpaca_base_url', 'https://paper-api.alpaca.markets')
        )
        
        # Agent will be initialized after we have data dimensions
        self.agent = None
        
        # Training parameters
        self.num_episodes = self.config.get('num_episodes', 1000)
        self.episode_length = self.config.get('episode_length', 390)  # Full trading day
        self.save_frequency = self.config.get('save_frequency', 50)
        self.eval_frequency = self.config.get('eval_frequency', 10)
        self.data_days = self.config.get('data_days', 60)  # Days of historical data to load
    
    async def load_training_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Load real historical options data for training"""
        logger.info(f"Loading {self.data_days} days of historical options data for {symbols}")
        
        end_date = datetime.now() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=self.data_days)
        
        # Load data with caching
        historical_data = await self.data_loader.load_historical_options_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )
        
        # Validate data
        total_points = 0
        for symbol, df in historical_data.items():
            logger.info(f"{symbol}: {len(df)} data points from {df.index.min()} to {df.index.max()}")
            total_points += len(df)
            
            # Check data quality
            if 'underlying_price' in df.columns:
                price_range = (df['underlying_price'].min(), df['underlying_price'].max())
                logger.info(f"  Price range: ${price_range[0]:.2f} - ${price_range[1]:.2f}")
            
            if 'implied_volatility' in df.columns:
                avg_iv = df['implied_volatility'].mean()
                logger.info(f"  Average IV: {avg_iv:.2%}")
        
        if total_points == 0:
            raise ValueError("No historical options data loaded!")
        
        logger.info(f"Total data points loaded: {total_points:,}")
        
        return historical_data
    
    def create_training_environment(self, historical_data: Dict[str, pd.DataFrame]):
        """Create training environment with real historical data"""
        return HistoricalOptionsEnvironment(
            historical_data=historical_data,
            initial_capital=self.config.get('initial_capital', 100000),
            lookback_window=20,
            episode_length=self.episode_length,
            commission=0.65
        )
    
    def validate_real_data_usage(self, env: HistoricalOptionsEnvironment, num_checks: int = 5):
        """Validate that real price movements are being used"""
        logger.info("Validating real data usage...")
        
        for i in range(num_checks):
            obs = env.reset()
            
            # Check that we have real price data
            price_history = obs['price_history']
            if np.all(price_history == 0):
                logger.warning(f"Check {i}: Price history is all zeros!")
            else:
                prices = price_history[:, 3]  # Close prices
                price_changes = np.diff(prices)
                logger.info(f"Check {i}: Price range ${prices.min():.2f}-${prices.max():.2f}, "
                          f"volatility: {np.std(price_changes):.4f}")
            
            # Check options chain
            options_chain = obs['options_chain']
            non_zero_options = np.sum(options_chain[:, 0] > 0)  # Count non-zero strikes
            logger.info(f"Check {i}: {non_zero_options} valid options in chain")
            
            # Step through a few actions and check rewards
            for j in range(10):
                action = np.random.randint(0, 11)
                next_obs, reward, done, info = env.step(action)
                
                if j == 0:
                    logger.info(f"  Symbol: {info['symbol']}, Date: {info['date']}")
                
                if reward != 0:
                    logger.info(f"  Step {j}: Action {action} -> Reward {reward:.2f}")
    
    async def train(self, symbols: List[str] = None):
        if symbols is None:
            # Use symbols from config, which now includes our default popular options symbols
            symbols = self.config.get('symbols', ['SPY', 'QQQ', 'TSLA', 'AAPL', 'META', 'NVDA', 'AMD', 'PLTR'])
        
        logger.info(f"Starting training with real historical data for: {symbols}")
        
        # Load historical data
        historical_data = await self.load_training_data(symbols)
        
        # Create training environment
        train_env = self.create_training_environment(historical_data)
        
        # Validate data
        self.validate_real_data_usage(train_env)
        
        # Initialize agent with environment specs
        if self.agent is None:
            self.agent = OptionsCLSTMPPOAgent(
                observation_space=train_env.observation_space,
                action_space=11,
                learning_rate_actor_critic=self.config.get('learning_rate', 3e-4),
                learning_rate_clstm=self.config.get('clstm_learning_rate', 1e-3),
                gamma=self.config.get('gamma', 0.99),
                batch_size=self.config.get('batch_size', 64),
                n_epochs=self.config.get('n_epochs', 10)
            )
            logger.info("Initialized CLSTM-PPO agent")
        
        # Pre-train CLSTM on historical patterns
        await self._pretrain_on_historical_data(historical_data)
        
        # Training metrics
        episode_rewards = []
        episode_lengths = []
        training_losses = []
        real_pnl = []
        
        # Early stopping variables
        best_avg_pnl = -float('inf')
        patience_counter = 0
        max_patience = 20  # Stop if no improvement for 20 episodes
        min_episodes = 50  # Train at least 50 episodes before early stopping
        
        logger.info("Starting reinforcement learning on historical data...")
        
        for episode in tqdm(range(self.num_episodes), desc="Training"):
            # Reset environment (selects random historical day)
            obs = train_env.reset()
            
            # Episode variables
            total_reward = 0
            total_real_pnl = 0
            step = 0
            
            for step in range(self.episode_length):
                # Get action from agent
                action, info = self.agent.act(obs, deterministic=False)
                
                # Step through real historical data
                next_obs, reward, done, env_info = train_env.step(action)
                
                # Store transition
                self.agent.store_transition(obs, action, reward, next_obs, done, info)
                
                # Track real P&L (reward is based on real price movements)
                total_real_pnl += reward
                
                # Update variables
                obs = next_obs
                total_reward += reward
                
                if done:
                    break
            
            # Store episode metrics
            episode_rewards.append(total_reward)
            episode_lengths.append(step + 1)
            real_pnl.append(total_real_pnl)
            
            # Train agent on real data experiences
            if len(self.agent.buffer) >= self.agent.batch_size:
                train_metrics = self.agent.train()
                if train_metrics:
                    training_losses.append(train_metrics['total_loss'])
                    
                    if self.log_wandb:
                        wandb.log({
                            'episode': episode,
                            'episode_reward': total_reward,
                            'real_pnl': total_real_pnl,
                            'episode_length': step + 1,
                            'total_loss': train_metrics['total_loss'],
                            'policy_loss': train_metrics['policy_loss'],
                            'value_loss': train_metrics['value_loss'],
                            'entropy': train_metrics['entropy'],
                            'clstm_loss': train_metrics.get('clstm_loss', 0),
                            'symbol': env_info['symbol'],
                            'date': str(env_info['date'])
                        })
            
            # Logging
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                avg_pnl = np.mean(real_pnl[-100:]) if real_pnl else 0
                avg_loss = np.mean(training_losses[-100:]) if training_losses else 0
                win_rate = sum(1 for r in real_pnl[-100:] if r > 0) / min(100, len(real_pnl)) if real_pnl else 0
                
                logger.info(
                    f"Episode {episode}/{self.num_episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Avg Real P&L: {avg_pnl:.2f} | "
                    f"Win Rate: {win_rate:.2%} | "
                    f"Avg Loss: {avg_loss:.4f}"
                )
                
                # Early stopping check
                if episode >= min_episodes:
                    if avg_pnl > best_avg_pnl:
                        best_avg_pnl = avg_pnl
                        patience_counter = 0
                        # Save best model
                        best_model_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                        self.agent.save(best_model_path)
                        logger.info(f"New best model saved with avg P&L: {avg_pnl:.2f}")
                    else:
                        patience_counter += 1
                        if patience_counter >= max_patience:
                            logger.warning(f"Early stopping triggered. No improvement for {max_patience} episodes.")
                            logger.info(f"Best average P&L: {best_avg_pnl:.2f}")
                            break
            
            # Save checkpoint
            if episode % self.save_frequency == 0 and episode > 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"real_data_episode_{episode}.pt"
                )
                self.agent.save(checkpoint_path)
                logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Evaluation on separate test data
            if episode % self.eval_frequency == 0 and episode > 0:
                eval_reward = await self.evaluate_on_test_data(historical_data)
                logger.info(f"Test evaluation reward: {eval_reward:.2f}")
                
                if self.log_wandb:
                    wandb.log({
                        'episode': episode,
                        'eval_reward': eval_reward
                    })
        
        # Save final model
        final_path = os.path.join(self.checkpoint_dir, "real_data_final.pt")
        self.agent.save(final_path)
        logger.info(f"Training complete. Final model saved: {final_path}")
        
        # Summary statistics
        logger.info("\nTraining Summary:")
        logger.info(f"Total episodes: {self.num_episodes}")
        logger.info(f"Average reward: {np.mean(episode_rewards):.2f}")
        logger.info(f"Average real P&L: {np.mean(real_pnl):.2f}")
        logger.info(f"Final 100-episode average: {np.mean(episode_rewards[-100:]):.2f}")
        
        if self.log_wandb:
            wandb.finish()
    
    async def _pretrain_on_historical_data(self, historical_data: Dict[str, pd.DataFrame]):
        """Pre-train CLSTM on historical price patterns"""
        logger.info("Pre-training CLSTM on historical patterns...")
        
        training_samples = []
        
        # Create environment to get proper observations
        env = self.create_training_environment(historical_data)
        
        for symbol, df in historical_data.items():
            # Create supervised learning samples from historical data
            # We need sequences of observations for CLSTM
            sequence_length = 20
            
            for i in range(40 + sequence_length, len(df) - 1):  # Need more history for sequences
                # Collect a sequence of observations
                sequence_features = []
                
                for j in range(sequence_length):
                    # Set environment to this timestep
                    env.current_step = i - sequence_length + j
                    env.current_symbol = symbol
                    env.training_data = df
                    
                    # Get observation at this timestep
                    obs = env._get_observation()
                    if obs is None:
                        break
                    
                    # Convert observation to tensor dict
                    obs_tensor = self.agent._observation_to_tensor(obs)
                    
                    # Flatten all observation components into a single tensor
                    features = []
                    features.append(obs_tensor['price_history'].flatten())
                    features.append(obs_tensor['technical_indicators'].flatten())
                    features.append(obs_tensor['options_chain'].flatten())
                    features.append(obs_tensor['portfolio_state'].flatten())
                    features.append(obs_tensor['greeks_summary'].flatten())
                    
                    combined_features = torch.cat(features, dim=0)
                    
                    # Debug: print shapes on first iteration
                    if i == 40 + sequence_length and j == 0:
                        logger.info(f"DEBUG - price_history shape: {obs_tensor['price_history'].shape} -> flattened: {obs_tensor['price_history'].flatten().shape}")
                        logger.info(f"DEBUG - technical_indicators shape: {obs_tensor['technical_indicators'].shape} -> flattened: {obs_tensor['technical_indicators'].flatten().shape}")
                        logger.info(f"DEBUG - options_chain shape: {obs_tensor['options_chain'].shape} -> flattened: {obs_tensor['options_chain'].flatten().shape}")
                        logger.info(f"DEBUG - portfolio_state shape: {obs_tensor['portfolio_state'].shape} -> flattened: {obs_tensor['portfolio_state'].flatten().shape}")
                        logger.info(f"DEBUG - greeks_summary shape: {obs_tensor['greeks_summary'].shape} -> flattened: {obs_tensor['greeks_summary'].flatten().shape}")
                        logger.info(f"DEBUG - Combined features shape: {combined_features.shape}")
                    
                    sequence_features.append(combined_features)
                
                if len(sequence_features) == sequence_length:
                    # Stack into a sequence tensor (seq_len, features)
                    sequence_tensor = torch.stack(sequence_features)
                    
                    # Targets (next timestep)
                    next_row = df.iloc[i]
                    
                    training_samples.append({
                        'features': sequence_tensor,
                        'price_target': next_row.get('underlying_price', 0),
                        'volatility_target': next_row.get('implied_volatility', 0.25),
                        'volume_target': next_row.get('underlying_volume', 1000000)
                    })
                
                # Limit samples per symbol
                if len(training_samples) >= 200:
                    break
        
        logger.info(f"Created {len(training_samples)} pre-training samples from historical data")
        
        if training_samples:
            # Pre-train CLSTM
            pretrain_metrics = self.agent.pretrain_clstm(
                training_samples[:100],  # Limit samples for faster training
                epochs=10,
                batch_size=16
            )
            
            logger.info(f"CLSTM pre-training complete. Final loss: {pretrain_metrics['final_loss']:.4f}")
    
    async def evaluate_on_test_data(self, historical_data: Dict[str, pd.DataFrame]) -> float:
        """Evaluate on test portion of historical data"""
        # For small datasets, use the same data for evaluation
        # In production, you'd want separate test data
        test_data = historical_data
        
        if not test_data or all(len(df) < 10 for df in test_data.values()):
            logger.warning("Not enough data for evaluation")
            return 0.0
        
        test_env = self.create_training_environment(test_data)
        
        total_rewards = []
        
        for _ in range(min(10, len(test_data))):  # Evaluate on up to 10 episodes
            obs = test_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Get action (deterministic for evaluation)
                action, _ = self.agent.act(obs, deterministic=True)
                
                # Step environment
                obs, reward, done, _ = test_env.step(action)
                episode_reward += reward
            
            total_rewards.append(episode_reward)
        
        return np.mean(total_rewards) if total_rewards else 0.0


def main():
    parser = argparse.ArgumentParser(description='Train Options Bot with Real Historical Data')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--data-days', type=int, default=60, help='Days of historical data to load')
    parser.add_argument('--wandb', action='store_true', help='Log to Weights & Biases')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/options_real_data', 
                       help='Checkpoint directory')
    parser.add_argument('--symbols', nargs='+', default=None, 
                       help='Symbols to train on (uses config defaults if not specified)')
    parser.add_argument('--validate-only', action='store_true', 
                       help='Only validate data loading without training')
    
    args = parser.parse_args()
    
    # Check API credentials
    if not os.getenv('ALPACA_API_KEY') or not os.getenv('ALPACA_SECRET_KEY'):
        logger.error("Alpaca API credentials not found in environment variables")
        logger.error("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY")
        sys.exit(1)
    
    # Create training pipeline
    pipeline = RealDataOptionsTrainingPipeline(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        log_wandb=args.wandb
    )
    
    # Override config with command line args
    pipeline.num_episodes = args.episodes
    pipeline.data_days = args.data_days
    
    if args.validate_only:
        # Just validate data loading
        async def validate():
            historical_data = await pipeline.load_training_data(args.symbols)
            env = pipeline.create_training_environment(historical_data)
            pipeline.validate_real_data_usage(env, num_checks=10)
        
        asyncio.run(validate())
    else:
        # Start training with real data
        asyncio.run(pipeline.train(symbols=args.symbols))


if __name__ == '__main__':
    main()