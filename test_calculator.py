#!/usr/bin/env python3
"""Test the training cost calculator"""

import sys
sys.path.insert(0, '.')

from training_cost_calculator import TrainingConfig, TrainingCostCalculator

def main():
    print("Testing training cost calculator...")
    
    # Create config
    config = TrainingConfig(
        num_episodes=5000,
        episode_length=252,
        batch_size=128,
        ppo_epochs=10,
        lookback_window=30,
        num_symbols=18,
        observation_dim=788,
        hidden_dim=256,
        num_lstm_layers=3,
        action_space=31
    )
    
    print(f"✅ Config created: {config.num_episodes} episodes")
    
    # Create calculator
    calculator = TrainingCostCalculator(config)
    
    print(f"✅ Calculator created")
    print(f"   Total FLOPs: {calculator.total_flops / 1e15:.2f} PetaFLOPs")
    print(f"   Memory required: {calculator.memory_required_gb:.2f} GB")
    
    # Get results
    results = calculator.compare_all_configs()
    
    print(f"✅ Results generated: {len(results)} configurations")
    
    # Find best options
    valid_results = [r for r in results if r['memory_sufficient']]
    
    best_value = min(valid_results, key=lambda x: x['total_cost'])
    fastest = min(valid_results, key=lambda x: x['training_time_hours'])
    
    print(f"\n🏆 BEST VALUE: {best_value['gpu_name']} ({best_value['num_gpus']}x)")
    print(f"   Time: {best_value['training_time_str']}")
    print(f"   Cost: ${best_value['total_cost']:.2f}")
    
    print(f"\n⚡ FASTEST: {fastest['gpu_name']} ({fastest['num_gpus']}x)")
    print(f"   Time: {fastest['training_time_str']}")
    print(f"   Cost: ${fastest['total_cost']:.2f}")
    
    print("\n✅ Test passed!")

if __name__ == '__main__':
    main()

