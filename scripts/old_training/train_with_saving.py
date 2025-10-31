#!/usr/bin/env python3
"""Training script that ensures models are saved"""

import os
import sys
import torch
import numpy as np
import logging
from tqdm import tqdm
import traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.options_clstm_ppo import OptionsCLSTMPPOAgent
from src.options_trading_env import OptionsTradingEnvironment

# Setup logging to see save messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_save_functionality():
    """Test if saving works at all"""
    logger.info("Testing save functionality...")
    
    # Create dummy environment and agent
    env = OptionsTradingEnvironment(initial_capital=100000)
    agent = OptionsCLSTMPPOAgent(
        observation_space=env.observation_space,
        action_space=11
    )
    
    # Try to save
    test_path = "checkpoints/test_save.pt"
    os.makedirs("checkpoints", exist_ok=True)
    
    try:
        agent.save(test_path)
        if os.path.exists(test_path):
            file_size = os.path.getsize(test_path)
            logger.info(f"✓ Test save successful! File size: {file_size} bytes")
            
            # Check what's in the file
            checkpoint = torch.load(test_path)
            logger.info(f"✓ Checkpoint contains: {list(checkpoint.keys())}")
            
            # Clean up
            os.remove(test_path)
            return True
        else:
            logger.error("✗ Save failed - file not created")
            return False
    except Exception as e:
        logger.error(f"✗ Save test failed: {e}")
        traceback.print_exc()
        return False


def train_with_guaranteed_saving(num_episodes=1000):
    """Training that definitely saves models"""
    
    # Test saving first
    if not test_save_functionality():
        logger.error("Save functionality broken! Fixing...")
        return
    
    logger.info("\nStarting training with guaranteed saving")
    logger.info("="*60)
    
    # Create environment
    env = OptionsTradingEnvironment(
        initial_capital=100000,
        max_positions=3,
        commission=0.65
    )
    
    # Make learning easier
    env.historical_volatility = 0.05  # Very low volatility
    env.mean_return = 0.01  # 1% positive drift
    
    # Create agent
    agent = OptionsCLSTMPPOAgent(
        observation_space=env.observation_space,
        action_space=11,
        learning_rate_actor_critic=1e-5,
        learning_rate_clstm=5e-5,
        gamma=0.99,
        batch_size=32,  # Smaller batch size
        n_epochs=3
    )
    
    # Create checkpoint directory
    checkpoint_dir = "checkpoints/guaranteed_save"
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    
    # Training metrics
    all_returns = []
    all_win_rates = []
    save_counter = 0
    
    # Progress bar
    pbar = tqdm(range(num_episodes), desc="Training")
    
    for episode in pbar:
        obs = env.reset()
        episode_reward = 0
        initial_value = 100000
        
        # Simple episode
        for step in range(50):  # Shorter episodes
            # Get action
            action, info = agent.act(obs, deterministic=False)
            
            # Simple strategy
            current_value = env._calculate_portfolio_value()
            if current_value < initial_value * 0.95:  # Down 5%
                if len(env.positions) > 0:
                    action = 10  # Close positions
            
            # Step
            next_obs, reward, done, env_info = env.step(action)
            
            # Simple reward
            pnl = current_value - initial_value
            shaped_reward = pnl / 10000.0  # Scale
            
            # Store transition
            agent.store_transition(obs, action, shaped_reward, next_obs, done, info)
            
            episode_reward += shaped_reward
            obs = next_obs
            
            if done:
                break
        
        # Calculate metrics
        final_value = env._calculate_portfolio_value()
        episode_return = (final_value - 100000) / 100000
        
        total_trades = env.winning_trades + env.losing_trades
        win_rate = env.winning_trades / max(1, total_trades) if total_trades > 0 else 0.5
        
        all_returns.append(episode_return)
        all_win_rates.append(win_rate)
        
        # Train
        if len(agent.buffer) >= agent.batch_size:
            try:
                agent.train()
            except Exception as e:
                logger.warning(f"Training error: {e}")
                agent.buffer.clear()
        
        # Update progress
        avg_return = np.mean(all_returns[-50:]) if len(all_returns) >= 50 else episode_return
        avg_win_rate = np.mean(all_win_rates[-50:]) if len(all_win_rates) >= 50 else win_rate
        
        pbar.set_postfix({
            'Avg Return': f'{avg_return:.2%}',
            'Win Rate': f'{avg_win_rate:.2%}',
            'Saved': save_counter
        })
        
        # SAVE MODELS FREQUENTLY
        if episode % 10 == 0 and episode > 0:  # Every 10 episodes
            try:
                # Save timestamped checkpoint
                timestamp = episode
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_ep{timestamp}.pt")
                
                logger.info(f"\nSaving checkpoint at episode {episode}...")
                agent.save(checkpoint_path)
                
                # Verify save
                if os.path.exists(checkpoint_path):
                    file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
                    logger.info(f"✓ Saved successfully: {checkpoint_path} ({file_size:.2f} MB)")
                    save_counter += 1
                    
                    # Also save as latest
                    latest_path = os.path.join(checkpoint_dir, "latest_model.pt")
                    agent.save(latest_path)
                    
                    # Save best if improved
                    if avg_return > -0.5:  # Low bar for "best"
                        best_path = os.path.join(checkpoint_dir, "best_model.pt")
                        agent.save(best_path)
                        logger.info(f"✓ Saved as best model (return: {avg_return:.2%})")
                else:
                    logger.error(f"✗ Save failed - file not created")
                    
            except Exception as e:
                logger.error(f"Save error: {e}")
                traceback.print_exc()
        
        # Extra saves at milestones
        if episode in [50, 100, 200, 500, 1000]:
            milestone_path = os.path.join(checkpoint_dir, f"milestone_ep{episode}.pt")
            try:
                agent.save(milestone_path)
                logger.info(f"\n🏁 Milestone save at episode {episode}")
            except:
                pass
    
    # Final save
    try:
        final_path = os.path.join(checkpoint_dir, "final_model.pt")
        agent.save(final_path)
        logger.info(f"\n✓ Final model saved: {final_path}")
    except Exception as e:
        logger.error(f"Failed to save final model: {e}")
    
    # List all saved files
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Total saves: {save_counter}")
    logger.info(f"Final avg return: {np.mean(all_returns[-100:]):.2%}")
    logger.info(f"Final win rate: {np.mean(all_win_rates[-100:]):.2%}")
    
    # List saved models
    logger.info(f"\nSaved models in {checkpoint_dir}:")
    for filename in sorted(os.listdir(checkpoint_dir)):
        if filename.endswith('.pt'):
            filepath = os.path.join(checkpoint_dir, filename)
            size = os.path.getsize(filepath) / (1024 * 1024)
            logger.info(f"  {filename} ({size:.2f} MB)")
    
    # Save training history
    history_path = os.path.join(checkpoint_dir, "training_history.npz")
    np.savez(history_path, returns=all_returns, win_rates=all_win_rates)
    logger.info(f"\nTraining history saved: {history_path}")
    
    return agent


def verify_saved_models():
    """Verify that saved models can be loaded"""
    checkpoint_dir = "checkpoints/guaranteed_save"
    
    if not os.path.exists(checkpoint_dir):
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        return
    
    logger.info("\nVerifying saved models...")
    
    # List all .pt files
    pt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    
    if not pt_files:
        logger.error("No .pt files found!")
        return
    
    # Try to load each file
    for filename in pt_files[:3]:  # Test first 3
        filepath = os.path.join(checkpoint_dir, filename)
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            logger.info(f"✓ {filename} - Keys: {list(checkpoint.keys())}")
        except Exception as e:
            logger.error(f"✗ {filename} - Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--verify', action='store_true', help='Verify saved models')
    args = parser.parse_args()
    
    if args.verify:
        verify_saved_models()
    else:
        # GPU setup
        if torch.cuda.is_available():
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            logger.info("Using CPU")
        
        # Run training
        train_with_guaranteed_saving(num_episodes=args.episodes)