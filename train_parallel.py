#!/usr/bin/env python3
"""
High-Performance Parallel Training Script

Optimized for:
- 24 vCPUs: 16-20 parallel environments
- 141GB VRAM: Large batch sizes, batched inference
- 240GB RAM: In-memory data caching

Expected performance: 5,000-10,000+ steps/second vs ~100 steps/sec sequential
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.parallel_trainer import ParallelPPOTrainer
from src.models.ppo_agent import OptionsCLSTMPPOAgent
from src.envs.options_env import WorkingOptionsEnvironment
from src.data.massive_flat_file_loader import MassiveFlatFileLoader


def create_env_fn(data_loader, symbols, preloaded_data=None):
    """Factory function for creating environments with pre-loaded data"""
    def _create():
        env = WorkingOptionsEnvironment(
            data_loader=data_loader,
            symbols=symbols,
            initial_capital=100000,
            max_positions=5,
            episode_length=200,
            use_realistic_costs=True,
            enable_slippage=True
        )
        # Inject pre-loaded data directly to avoid async loading
        if preloaded_data is not None:
            env.market_data = preloaded_data
            env.data_loaded = True
            env.options_data = {}  # Options data loaded separately if needed
        return env
    return _create


def main():
    parser = argparse.ArgumentParser(description='High-Performance Parallel Training')
    parser.add_argument('--timesteps', type=int, default=1_000_000, help='Total timesteps to train')
    parser.add_argument('--n-envs', type=int, default=16, help='Number of parallel environments')
    parser.add_argument('--n-steps', type=int, default=200, help='Steps per rollout per env')
    parser.add_argument('--batch-size', type=int, default=2048, help='Training batch size')
    parser.add_argument('--n-epochs', type=int, default=10, help='PPO epochs per iteration')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (lowered for stability)')
    parser.add_argument('--data-dir', type=str, default='data/flat_files_processed', help='Data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/parallel_ppo', help='Checkpoint dir')
    parser.add_argument('--log-interval', type=int, default=1, help='Log every N iterations')
    parser.add_argument('--save-interval', type=int, default=10, help='Save every N iterations')
    args = parser.parse_args()
    
    # Print hardware info
    logger.info("=" * 80)
    logger.info("🖥️  Hardware Configuration")
    logger.info("=" * 80)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"   GPU: {gpu_name}")
        logger.info(f"   VRAM: {gpu_mem:.1f} GB")
    else:
        logger.info("   GPU: None (CPU only)")
    
    import os
    cpu_count = os.cpu_count()
    logger.info(f"   CPUs: {cpu_count}")
    
    import psutil
    ram_gb = psutil.virtual_memory().total / 1e9
    logger.info(f"   RAM: {ram_gb:.1f} GB")
    logger.info("=" * 80)
    
    # Load data
    logger.info("📂 Loading training data...")
    data_loader = MassiveFlatFileLoader(data_dir=args.data_dir)
    
    # Get available symbols
    symbols = ['SPY', 'QQQ', 'IWM']
    
    # Load all data into memory for fast access
    logger.info("📥 Pre-loading data into memory...")
    from datetime import datetime, timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    import asyncio
    data = asyncio.run(data_loader.load_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date
    ))
    
    logger.info(f"✅ Loaded data for {len(data)} symbols")
    for sym, df in data.items():
        logger.info(f"   {sym}: {len(df)} days of data")

    # Create environment factory with pre-loaded data
    env_fn = create_env_fn(data_loader, symbols, preloaded_data=data)
    
    # Create a sample env to get observation space
    sample_env = env_fn()
    observation_space = sample_env.observation_space.spaces
    action_space = sample_env.action_space.n
    sample_env.close() if hasattr(sample_env, 'close') else None
    
    logger.info(f"📊 Environment: {action_space} actions")
    
    # Create agent
    logger.info("🤖 Creating CLSTM-PPO agent...")
    agent = OptionsCLSTMPPOAgent(
        observation_space=observation_space,
        action_space=action_space,
        learning_rate_actor_critic=args.lr,
        learning_rate_clstm=args.lr,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    logger.info("✅ Agent created")
    
    # Create parallel trainer with reward normalization for stable training
    logger.info(f"🚀 Creating parallel trainer with {args.n_envs} environments...")
    trainer = ParallelPPOTrainer(
        env_fn=env_fn,
        agent=agent,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        max_grad_norm=0.5,  # Gradient clipping for stability
        normalize_rewards=True,  # Running reward normalization
        reward_clip=10.0,  # Clip normalized rewards
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    logger.info("=" * 80)
    logger.info(f"🎯 Starting training for {args.timesteps:,} timesteps")
    logger.info(f"   Batch size per iteration: {args.n_envs * args.n_steps:,}")
    logger.info(f"   Estimated iterations: {args.timesteps // (args.n_envs * args.n_steps)}")
    logger.info("=" * 80)
    
    try:
        history = trainer.train(
            total_timesteps=args.timesteps,
            log_interval=args.log_interval
        )
        
        # Save final model
        checkpoint_path = checkpoint_dir / f"model_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        torch.save({
            'agent_state_dict': agent.network.state_dict(),
            'ppo_optimizer': agent.ppo_optimizer.state_dict(),
            'clstm_optimizer': agent.clstm_optimizer.state_dict(),
            'total_steps': trainer.total_steps,
            'total_episodes': trainer.total_episodes
        }, checkpoint_path)
        logger.info(f"💾 Saved final model to {checkpoint_path}")
        
    except KeyboardInterrupt:
        logger.info("⚠️ Training interrupted by user")
    finally:
        trainer.close()
    
    logger.info("✅ Training complete!")


if __name__ == '__main__':
    main()

