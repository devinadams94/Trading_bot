#!/usr/bin/env python3
"""
Stable Training Script with Improved Hyperparameters
Purpose: Train the CLSTM-PPO model with stability fixes to prevent overfitting
"""

import asyncio
import argparse
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from train_enhanced_clstm_ppo import main as train_main


async def main():
    """Main training function with stable configuration"""
    
    # Load stable configuration
    config_path = Path("configs/stable_training.yaml")
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        print("   Please create configs/stable_training.yaml first")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print("🚀 STABLE TRAINING MODE")
    print("=" * 80)
    print(f"Configuration loaded from: {config_path}")
    print(f"\n📊 Key Parameters:")
    print(f"   Learning Rate (Actor/Critic): {config['learning_rate_actor_critic']}")
    print(f"   Learning Rate (CLSTM): {config['learning_rate_clstm']}")
    print(f"   Entropy Coefficient: {config['entropy_coef']} (decays to {config['min_entropy_coef']})")
    print(f"   Max Gradient Norm: {config['max_grad_norm']}")
    print(f"   L2 Regularization: {config['l2_reg']}")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Episodes: {config['num_episodes']}")
    print(f"   Symbols: {len(config['symbols'])}")
    print(f"\n🎯 Stability Features:")
    print(f"   ✅ Gradient clipping: {config['max_grad_norm']}")
    print(f"   ✅ L2 regularization: {config['l2_reg']}")
    print(f"   ✅ Entropy decay: {config['entropy_decay']}")
    print(f"   ✅ Reward normalization: {config['normalize_rewards']}")
    print(f"   ✅ Action diversity tracking: {config['track_action_diversity']}")
    print(f"   ✅ Train/val split: {config['train_val_split']}")
    print("=" * 80)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stable CLSTM-PPO Training')
    parser.add_argument('--episodes', type=int, default=config['num_episodes'],
                        help=f"Number of episodes (default: {config['num_episodes']})")
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/stable_training',
                        help='Checkpoint directory')
    parser.add_argument('--fresh-start', action='store_true',
                        help='Start fresh training (ignore checkpoints)')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume from checkpoint if available')
    parser.add_argument('--wandb', action='store_true',
                        help='Use wandb logging')
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test mode (10 episodes)')
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    if args.quick_test:
        config['num_episodes'] = 10
        config['save_frequency'] = 5
        config['log_frequency'] = 1
        print("\n⚡ QUICK TEST MODE: 10 episodes")
    
    # Build command line arguments for train_enhanced_clstm_ppo.py
    train_args = [
        '--episodes', str(args.episodes if not args.quick_test else 10),
        '--checkpoint-dir', args.checkpoint_dir,
        '--use-flat-files',
        '--flat-files-dir', config['flat_files_dir'],
        '--data-days', str(config['data_days']),
    ]
    
    if args.fresh_start:
        train_args.append('--fresh-start')
    
    if args.resume:
        train_args.append('--resume')
    
    if args.wandb:
        train_args.append('--wandb')
    
    if config.get('use_realistic_costs', True):
        train_args.append('--realistic-costs')
    
    if args.quick_test:
        train_args.append('--quick-test')
    
    # Set environment variables for stable training
    import os
    os.environ['STABLE_TRAINING'] = '1'
    os.environ['ENTROPY_COEF'] = str(config['entropy_coef'])
    os.environ['ENTROPY_DECAY'] = str(config['entropy_decay'])
    os.environ['MIN_ENTROPY_COEF'] = str(config['min_entropy_coef'])
    os.environ['MAX_GRAD_NORM'] = str(config['max_grad_norm'])
    os.environ['L2_REG'] = str(config['l2_reg'])
    os.environ['NORMALIZE_REWARDS'] = str(config['normalize_rewards'])
    os.environ['REWARD_CLIP'] = str(config['reward_clip'])
    os.environ['LEARNING_RATE_ACTOR_CRITIC'] = str(config['learning_rate_actor_critic'])
    os.environ['LEARNING_RATE_CLSTM'] = str(config['learning_rate_clstm'])
    os.environ['BATCH_SIZE'] = str(config['batch_size'])
    
    print(f"\n🎬 Starting training with stable configuration...")
    print(f"   Checkpoint directory: {args.checkpoint_dir}")
    print(f"   Fresh start: {args.fresh_start}")
    print(f"   Resume: {args.resume}")
    print("=" * 80)
    print()
    
    # Modify sys.argv to pass arguments to train_enhanced_clstm_ppo.py
    sys.argv = ['train_enhanced_clstm_ppo.py'] + train_args
    
    # Run training
    await train_main()


if __name__ == "__main__":
    asyncio.run(main())

