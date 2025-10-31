#!/usr/bin/env python3
"""GPU-optimized training script for options trading bot"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from loguru import logger
import argparse
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.options_trading_env import OptionsTradingEnvironment
from src.options_data_collector import OptionsDataSimulator
from src.options_clstm_ppo import OptionsCLSTMPPOAgent

def setup_multi_gpu_training():
    """Setup for multi-GPU training with proper error handling"""
    
    device_info = {
        'available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'device': None,
        'multi_gpu': False
    }
    
    if device_info['available']:
        logger.info(f"🚀 GPU Training Enabled!")
        logger.info(f"Found {device_info['device_count']} GPU(s)")
        
        for i in range(device_info['device_count']):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        # Use first GPU by default
        device_info['device'] = torch.device('cuda:0')
        
        # Enable multi-GPU if available
        if device_info['device_count'] > 1:
            device_info['multi_gpu'] = True
            logger.info("Multi-GPU training enabled with DataParallel")
            
            # Set CUDA environment for better multi-GPU performance
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
    else:
        logger.warning("⚠️  No GPU detected - using CPU (training will be slower)")
        logger.info("For GPU support, ensure:")
        logger.info("1. NVIDIA drivers are installed")
        logger.info("2. CUDA toolkit is installed")
        logger.info("3. PyTorch is installed with CUDA support")
        device_info['device'] = torch.device('cpu')
    
    return device_info

def train_with_gpu_optimization(episodes=100, batch_size=64, num_workers=4):
    """Training with GPU optimization and multi-GPU support"""
    
    # Setup GPU configuration
    device_info = setup_multi_gpu_training()
    device = device_info['device']
    
    # Log training configuration
    logger.info("=" * 60)
    logger.info("Starting GPU-Optimized Training")
    logger.info("=" * 60)
    logger.info(f"Episodes: {episodes}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Device: {device}")
    logger.info(f"Multi-GPU: {device_info['multi_gpu']}")
    
    # Initialize environment
    env = OptionsTradingEnvironment(
        initial_capital=100000,
        commission=0.65
    )
    
    # Initialize agent with explicit device
    agent = OptionsCLSTMPPOAgent(
        observation_space=env.observation_space,
        action_space=env.action_space.n,
        batch_size=batch_size,
        device=str(device)
    )
    
    # Agent already handles DataParallel internally based on device parameter
    if device_info['multi_gpu']:
        logger.info(f"Agent configured to use DataParallel across {device_info['device_count']} GPUs")
    
    # Data simulator
    simulator = OptionsDataSimulator()
    
    # Symbols to train on
    symbols = ['SPY', 'QQQ', 'TSLA', 'NVDA', 'AMD', 'PLTR', 'META', 'COIN']
    
    # Training metrics
    episode_rewards = []
    training_times = []
    
    # Main training loop
    start_time = datetime.now()
    
    for episode in range(episodes):
        episode_start = datetime.now()
        
        obs = env.reset()
        episode_reward = 0
        step_rewards = []  # Track rewards per step
        done = False
        steps = 0
        
        # Random symbol for this episode
        symbol = np.random.choice(symbols)
        stock_price = np.random.uniform(100, 500)
        
        # Progress bar for steps within episode
        pbar = tqdm(total=200, desc=f"Episode {episode}/{episodes}", leave=False)
        
        while not done and steps < 200:
            # Generate options chain
            options_chain = simulator.simulate_options_chain(
                symbol=symbol,
                stock_price=stock_price,
                num_strikes=20,
                num_expirations=4
            )
            
            # Update observation
            if 'options_chain' in obs:
                sorted_options = sorted(options_chain, key=lambda x: x.volume, reverse=True)[:20]
                options_features = []
                for opt in sorted_options:
                    features = [
                        opt.strike, opt.bid, opt.ask, opt.last_price, opt.volume,
                        opt.open_interest, opt.implied_volatility, opt.delta,
                        opt.gamma, opt.theta, opt.vega, opt.rho,
                        1.0 if opt.option_type == 'call' else 0.0,
                        30, (opt.bid + opt.ask) / 2
                    ]
                    options_features.append(features)
                while len(options_features) < 20:
                    options_features.append([0] * 15)
                obs['options_chain'] = np.array(options_features[:20], dtype=np.float32)
            
            # Get action (computation happens on GPU)
            action, act_info = agent.act(obs)
            
            # Step environment
            next_obs, reward, done, env_info = env.step(action)
            
            # Store transition
            agent.store_transition(obs, action, reward, next_obs, done, act_info)
            
            obs = next_obs
            episode_reward += reward
            step_rewards.append(reward)
            steps += 1
            
            # Update progress bar with current reward
            pbar.update(1)
            pbar.set_postfix({'reward': f'{reward:.4f}', 'total': f'{episode_reward:.4f}'})
            
            # Simulate price movement
            stock_price *= np.random.uniform(0.99, 1.01)
        
        pbar.close()
        episode_rewards.append(episode_reward)
        
        # Log detailed episode info
        logger.info(f"Episode {episode}: Total Reward: {episode_reward:.4f}, Steps: {steps}")
        if len(step_rewards) > 0:
            logger.info(f"  Step rewards - Min: {min(step_rewards):.4f}, Max: {max(step_rewards):.4f}, Avg: {np.mean(step_rewards):.4f}")
            logger.info(f"  Action taken: {env_info.get('action', 'N/A')}, Portfolio value: ${env_info.get('portfolio_value', 0):.2f}")
        
        # Train after each episode if enough data
        if len(agent.buffer) >= batch_size:
            train_metrics = agent.train()
            
            # Log detailed progress every 10 episodes
            if episode > 0 and episode % 10 == 0:
                episode_time = (datetime.now() - episode_start).total_seconds()
                training_times.append(episode_time)
                avg_time = np.mean(training_times[-10:])
                
                logger.info(f"Episode {episode}/{episodes}")
                logger.info(f"  Reward: {episode_reward:.2f}")
                logger.info(f"  Avg Reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
                logger.info(f"  Episode time: {episode_time:.2f}s")
                logger.info(f"  Avg time/episode: {avg_time:.2f}s")
                logger.info(f"  Est. time remaining: {(episodes - episode) * avg_time / 60:.1f} min")
                
                if train_metrics:
                    logger.info(f"  Training loss: {train_metrics.get('total_loss', 0):.4f}")
                
                # GPU memory stats
                if device.type == 'cuda':
                    logger.info(f"  GPU memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        
        # Save checkpoints
        if (episode + 1) % 50 == 0:
            checkpoint_dir = "checkpoints/options_clstm_ppo"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"clstm_ppo_gpu_episode_{episode+1}.pt")
            
            # Handle DataParallel when saving
            if device_info['multi_gpu']:
                torch.save(agent.network.module.state_dict(), checkpoint_path)
            else:
                torch.save(agent.network.state_dict(), checkpoint_path)
            
            logger.info(f"💾 Saved checkpoint: {checkpoint_path}")
    
    # Training complete
    total_time = (datetime.now() - start_time).total_seconds()
    
    # Save final model
    final_path = os.path.join("checkpoints/options_clstm_ppo", "clstm_ppo_gpu_final.pt")
    if device_info['multi_gpu']:
        torch.save(agent.network.module.state_dict(), final_path)
    else:
        torch.save(agent.network.state_dict(), final_path)
    
    logger.info("=" * 60)
    logger.info("✅ Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Total training time: {total_time / 60:.1f} minutes")
    logger.info(f"Average time per episode: {total_time / episodes:.2f} seconds")
    logger.info(f"Final average reward: {np.mean(episode_rewards[-10:]):.2f}")
    logger.info(f"Model saved: {final_path}")
    
    # Clear GPU cache
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return final_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPU-optimized training for options trading bot')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data workers')
    
    args = parser.parse_args()
    
    # Run training
    train_with_gpu_optimization(
        episodes=args.episodes,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )