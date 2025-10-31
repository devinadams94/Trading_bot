#!/usr/bin/env python3
"""Benchmark training speed improvements"""

import time
import asyncio
import numpy as np
import pandas as pd
import logging
import os
from dotenv import load_dotenv
from train_profitable_fixed import ProfitableHistoricalEnvironment, load_historical_data
from train_profitable_optimized import OptimizedProfitableEnvironment

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def benchmark_environments():
    """Compare performance of original vs optimized environment"""
    
    # Load data
    logger.info("Loading historical data...")
    historical_data, data_loader = await load_historical_data()
    
    if not historical_data:
        logger.error("Failed to load data")
        return
    
    # Test parameters
    num_episodes = 5
    steps_per_episode = 100
    
    # Benchmark original environment
    logger.info("\n" + "="*60)
    logger.info("Benchmarking ORIGINAL environment")
    logger.info("="*60)
    
    original_env = ProfitableHistoricalEnvironment(
        historical_data=historical_data,
        data_loader=data_loader,
        symbols=['SPY'],
        initial_capital=100000,
        max_positions=2,
        commission=0.65,
        episode_length=steps_per_episode
    )
    
    original_times = []
    for episode in range(num_episodes):
        obs = original_env.reset()
        episode_start = time.time()
        
        for step in range(steps_per_episode):
            # Simulate trading actions
            if step % 10 == 0:
                action = 1  # buy call
            elif step % 10 == 5:
                action = 2  # buy put
            elif step % 20 == 15:
                action = 10  # close all
            else:
                action = 0  # hold
            
            obs, reward, done, info = original_env.step(action)
            if done:
                break
        
        episode_time = time.time() - episode_start
        original_times.append(episode_time)
        logger.info(f"Episode {episode + 1}: {episode_time:.2f}s ({steps_per_episode/episode_time:.2f} steps/s)")
    
    # Benchmark optimized environment
    logger.info("\n" + "="*60)
    logger.info("Benchmarking OPTIMIZED environment")
    logger.info("="*60)
    
    optimized_env = OptimizedProfitableEnvironment(
        historical_data=historical_data,
        data_loader=data_loader,
        symbols=['SPY'],
        initial_capital=100000,
        max_positions=2,
        commission=0.65,
        episode_length=steps_per_episode
    )
    
    optimized_times = []
    for episode in range(num_episodes):
        obs = optimized_env.reset()
        episode_start = time.time()
        
        for step in range(steps_per_episode):
            # Same trading actions
            if step % 10 == 0:
                action = 1  # buy call
            elif step % 10 == 5:
                action = 2  # buy put
            elif step % 20 == 15:
                action = 10  # close all
            else:
                action = 0  # hold
            
            obs, reward, done, info = optimized_env.step(action)
            if done:
                break
        
        episode_time = time.time() - episode_start
        optimized_times.append(episode_time)
        logger.info(f"Episode {episode + 1}: {episode_time:.2f}s ({steps_per_episode/episode_time:.2f} steps/s)")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("="*60)
    
    avg_original = np.mean(original_times)
    avg_optimized = np.mean(optimized_times)
    speedup = avg_original / avg_optimized
    
    logger.info(f"Original environment:")
    logger.info(f"  Average time per episode: {avg_original:.3f}s")
    logger.info(f"  Average steps per second: {steps_per_episode/avg_original:.1f}")
    logger.info(f"  Estimated time per iteration: {avg_original/steps_per_episode:.3f}s")
    
    logger.info(f"\nOptimized environment:")
    logger.info(f"  Average time per episode: {avg_optimized:.3f}s")
    logger.info(f"  Average steps per second: {steps_per_episode/avg_optimized:.1f}")
    logger.info(f"  Estimated time per iteration: {avg_optimized/steps_per_episode:.3f}s")
    
    logger.info(f"\n🚀 SPEEDUP: {speedup:.2f}x faster!")
    logger.info(f"Time saved per episode: {avg_original - avg_optimized:.3f}s")
    
    # Test specific bottleneck: option finding
    logger.info("\n" + "="*60)
    logger.info("OPTION FINDING BENCHMARK")
    logger.info("="*60)
    
    # Get sample data (use middle of dataset)
    spy_data = historical_data['SPY']
    sample_idx = min(100, len(spy_data) - 1)
    sample_data = spy_data.iloc[sample_idx]
    
    # Original method timing
    start = time.time()
    for _ in range(100):
        original_env._find_suitable_options(sample_data, 'buy_call')
    original_option_time = (time.time() - start) / 100
    
    # Optimized method timing
    start = time.time()
    for _ in range(100):
        optimized_env._find_suitable_options_vectorized(sample_data, 'buy_call')
    optimized_option_time = (time.time() - start) / 100
    
    logger.info(f"Original _find_suitable_options: {original_option_time*1000:.2f}ms per call")
    logger.info(f"Optimized _find_suitable_options: {optimized_option_time*1000:.2f}ms per call")
    logger.info(f"Option finding speedup: {original_option_time/optimized_option_time:.2f}x")


if __name__ == "__main__":
    asyncio.run(benchmark_environments())