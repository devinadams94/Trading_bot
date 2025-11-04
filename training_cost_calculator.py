#!/usr/bin/env python3
"""
CLSTM-PPO Training Cost Calculator
Calculates training costs and performance across different GPU configurations
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class GPUSpec:
    """GPU specifications and pricing"""
    name: str
    memory_gb: int
    tflops_fp32: float
    tflops_fp16: float
    tflops_tensor: float  # Tensor cores
    memory_bandwidth_gbps: float
    price_per_hour: float
    price_source: str
    relative_speed: float = 1.0  # Relative to RTX 4090


# GPU pricing and specifications
# Sources: GetDeploying.com, Lambda Labs, RunPod, Vast.ai (as of Oct 2025)
GPU_SPECS = {
    # === NVIDIA Blackwell Generation (2024-2025) ===
    'B200': GPUSpec(
        name='NVIDIA B200 192GB',
        memory_gb=192,
        tflops_fp32=80.0,
        tflops_fp16=160.0,
        tflops_tensor=4500.0,  # FP16 Tensor (estimated)
        memory_bandwidth_gbps=8000,
        price_per_hour=3.99,  # GetDeploying.com lowest
        price_source='GetDeploying.com (https://getdeploying.com/reference/cloud-gpu)',
        relative_speed=5.5
    ),
    'GB200_NVL72': GPUSpec(
        name='NVIDIA GB200 NVL72 (per GPU)',
        memory_gb=192,  # Per GPU in 72-GPU system
        tflops_fp32=80.0,
        tflops_fp16=160.0,
        tflops_tensor=4500.0,  # FP16 Tensor
        memory_bandwidth_gbps=8000,
        price_per_hour=42.00,  # GetDeploying.com (full system / 72)
        price_source='GetDeploying.com (https://getdeploying.com/reference/cloud-gpu)',
        relative_speed=6.0
    ),

    # === NVIDIA Hopper Generation (2023-2024) ===
    'H200': GPUSpec(
        name='NVIDIA H200 141GB SXM',
        memory_gb=141,
        tflops_fp32=67.0,
        tflops_fp16=268.0,
        tflops_tensor=3958.0,  # FP16 Tensor
        memory_bandwidth_gbps=4800,
        price_per_hour=2.30,  # GetDeploying.com lowest
        price_source='GetDeploying.com (https://getdeploying.com/reference/cloud-gpu)',
        relative_speed=5.0
    ),
    'H100_SXM5': GPUSpec(
        name='NVIDIA H100 80GB SXM5',
        memory_gb=80,
        tflops_fp32=67.0,
        tflops_fp16=268.0,
        tflops_tensor=3958.0,  # FP16 Tensor
        memory_bandwidth_gbps=3350,
        price_per_hour=0.99,  # GetDeploying.com lowest (spot)
        price_source='GetDeploying.com (https://getdeploying.com/reference/cloud-gpu)',
        relative_speed=4.5
    ),
    'H100_80GB': GPUSpec(
        name='NVIDIA H100 80GB PCIe',
        memory_gb=80,
        tflops_fp32=51.0,
        tflops_fp16=204.0,
        tflops_tensor=1979.0,  # FP16 Tensor
        memory_bandwidth_gbps=2000,
        price_per_hour=0.99,  # GetDeploying.com lowest
        price_source='GetDeploying.com (https://getdeploying.com/reference/cloud-gpu)',
        relative_speed=3.2
    ),
    'GH200': GPUSpec(
        name='NVIDIA GH200 Grace Hopper',
        memory_gb=96,  # 96GB HBM3
        tflops_fp32=67.0,
        tflops_fp16=268.0,
        tflops_tensor=3958.0,  # FP16 Tensor
        memory_bandwidth_gbps=4000,
        price_per_hour=1.49,  # GetDeploying.com lowest
        price_source='GetDeploying.com (https://getdeploying.com/reference/cloud-gpu)',
        relative_speed=4.2
    ),

    # === AMD Instinct MI300 Series (2024-2025) ===
    'MI325X': GPUSpec(
        name='AMD Instinct MI325X 256GB',
        memory_gb=256,
        tflops_fp32=163.0,
        tflops_fp16=653.0,
        tflops_tensor=5200.0,  # FP16 Matrix (estimated)
        memory_bandwidth_gbps=6000,
        price_per_hour=4.62,  # GetDeploying.com (36.92/8 GPUs)
        price_source='GetDeploying.com (https://getdeploying.com/reference/cloud-gpu)',
        relative_speed=5.8
    ),
    'MI300X': GPUSpec(
        name='AMD Instinct MI300X 192GB',
        memory_gb=192,
        tflops_fp32=163.0,
        tflops_fp16=653.0,
        tflops_tensor=5200.0,  # FP16 Matrix
        memory_bandwidth_gbps=5300,
        price_per_hour=1.99,  # GetDeploying.com lowest
        price_source='GetDeploying.com (https://getdeploying.com/reference/cloud-gpu)',
        relative_speed=5.2
    ),

    # === NVIDIA Ampere Generation (2020-2022) ===
    'A100_80GB': GPUSpec(
        name='NVIDIA A100 80GB PCIe',
        memory_gb=80,
        tflops_fp32=19.5,
        tflops_fp16=78.0,
        tflops_tensor=624.0,  # FP16 Tensor
        memory_bandwidth_gbps=1935,
        price_per_hour=0.40,  # GetDeploying.com lowest
        price_source='GetDeploying.com (https://getdeploying.com/reference/cloud-gpu)',
        relative_speed=2.1
    ),
    'A100_40GB': GPUSpec(
        name='NVIDIA A100 40GB PCIe',
        memory_gb=40,
        tflops_fp32=19.5,
        tflops_fp16=78.0,
        tflops_tensor=624.0,  # FP16 Tensor
        memory_bandwidth_gbps=1555,
        price_per_hour=0.40,  # GetDeploying.com lowest
        price_source='GetDeploying.com (https://getdeploying.com/reference/cloud-gpu)',
        relative_speed=2.0
    ),

    # === NVIDIA Ada Lovelace (2022-2023) ===
    'L40S': GPUSpec(
        name='NVIDIA L40S 48GB',
        memory_gb=48,
        tflops_fp32=91.6,
        tflops_fp16=183.0,
        tflops_tensor=1466.0,  # FP16 Tensor
        memory_bandwidth_gbps=864,
        price_per_hour=0.32,  # GetDeploying.com lowest
        price_source='GetDeploying.com (https://getdeploying.com/reference/cloud-gpu)',
        relative_speed=1.5
    ),
    'RTX_6000_Ada': GPUSpec(
        name='NVIDIA RTX 6000 Ada 48GB',
        memory_gb=48,
        tflops_fp32=91.1,
        tflops_fp16=182.5,
        tflops_tensor=1457.0,  # FP16 Tensor
        memory_bandwidth_gbps=960,
        price_per_hour=0.74,  # GetDeploying.com lowest
        price_source='GetDeploying.com (https://getdeploying.com/reference/cloud-gpu)',
        relative_speed=1.4
    ),
    'RTX_5090': GPUSpec(
        name='NVIDIA RTX 5090 32GB',
        memory_gb=32,
        tflops_fp32=92.0,
        tflops_fp16=184.0,
        tflops_tensor=1472.0,  # FP16 Tensor (estimated)
        memory_bandwidth_gbps=1792,
        price_per_hour=0.27,  # GetDeploying.com lowest
        price_source='GetDeploying.com (https://getdeploying.com/reference/cloud-gpu)',
        relative_speed=1.6
    ),
    'RTX_4090': GPUSpec(
        name='NVIDIA RTX 4090 24GB',
        memory_gb=24,
        tflops_fp32=82.6,
        tflops_fp16=165.2,
        tflops_tensor=1321.0,  # FP16 Tensor
        memory_bandwidth_gbps=1008,
        price_per_hour=0.18,  # GetDeploying.com lowest
        price_source='GetDeploying.com (https://getdeploying.com/reference/cloud-gpu)',
        relative_speed=1.0  # Baseline
    ),
    'RTX_3090': GPUSpec(
        name='NVIDIA RTX 3090 24GB',
        memory_gb=24,
        tflops_fp32=35.6,
        tflops_fp16=71.0,
        tflops_tensor=568.0,  # FP16 Tensor
        memory_bandwidth_gbps=936,
        price_per_hour=0.11,  # GetDeploying.com lowest
        price_source='GetDeploying.com (https://getdeploying.com/reference/cloud-gpu)',
        relative_speed=0.65
    ),
}


@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    num_episodes: int = 5000
    episode_length: int = 252  # Trading days per episode
    batch_size: int = 128
    ppo_epochs: int = 10
    lookback_window: int = 30
    num_symbols: int = 18
    observation_dim: int = 788
    hidden_dim: int = 256
    num_lstm_layers: int = 3
    action_space: int = 31
    
    # Estimated operations per training step
    def estimate_flops_per_step(self) -> float:
        """Estimate FLOPs per training step"""
        # CLSTM forward pass
        lstm_flops = (
            4 * self.num_lstm_layers * self.lookback_window * 
            self.hidden_dim * self.hidden_dim * 2  # 4 gates, 2 for matmul
        )
        
        # Attention mechanism (8 heads)
        attention_flops = (
            8 * self.lookback_window * self.lookback_window * 
            self.hidden_dim * 2
        )
        
        # Actor-Critic networks
        actor_critic_flops = (
            self.hidden_dim * 256 * 2 +  # Hidden layers
            256 * self.action_space * 2   # Output layers
        )
        
        # PPO update (multiple epochs)
        ppo_flops = (lstm_flops + attention_flops + actor_critic_flops) * self.ppo_epochs
        
        # Total per step (forward + backward ≈ 3x forward)
        total_flops = (lstm_flops + attention_flops + actor_critic_flops + ppo_flops) * 3
        
        return total_flops
    
    def estimate_memory_usage_gb(self) -> float:
        """Estimate GPU memory usage in GB"""
        # Model parameters
        lstm_params = self.num_lstm_layers * 4 * self.hidden_dim * self.hidden_dim
        attention_params = 8 * self.hidden_dim * self.hidden_dim * 3  # Q, K, V
        actor_critic_params = self.hidden_dim * 256 * 2 + 256 * self.action_space * 2
        total_params = lstm_params + attention_params + actor_critic_params
        
        # FP16 = 2 bytes per parameter
        # Model weights + gradients + optimizer states (Adam = 2x params)
        model_memory = total_params * 2 * (1 + 1 + 2) / 1e9  # GB
        
        # Batch memory (activations)
        batch_memory = (
            self.batch_size * self.lookback_window * self.hidden_dim * 
            self.num_lstm_layers * 2 / 1e9  # FP16
        )
        
        # Replay buffer (approximate)
        buffer_memory = 2.0  # GB
        
        return model_memory + batch_memory + buffer_memory


class TrainingCostCalculator:
    """Calculate training costs across different GPU configurations"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.total_steps = config.num_episodes * config.episode_length
        self.flops_per_step = config.estimate_flops_per_step()
        self.total_flops = self.total_steps * self.flops_per_step
        self.memory_required_gb = config.estimate_memory_usage_gb()
    
    def calculate_training_time(self, gpu: GPUSpec, num_gpus: int = 1) -> Tuple[float, str]:
        """Calculate training time in hours"""
        # Use FP16 tensor core performance (what PyTorch actually uses)
        effective_tflops = gpu.tflops_tensor * num_gpus
        
        # Account for GPU utilization (typically 60-80% for RL training)
        utilization = 0.70
        effective_tflops *= utilization
        
        # Account for memory bandwidth bottleneck
        # LSTM is memory-bound, not compute-bound
        memory_factor = min(1.0, gpu.memory_bandwidth_gbps / 1000.0)
        effective_tflops *= memory_factor
        
        # Calculate time
        total_tflops = self.total_flops / 1e12
        hours = total_tflops / effective_tflops
        
        # Apply relative speed adjustment (empirical)
        hours /= gpu.relative_speed
        
        # Format time
        if hours < 1:
            time_str = f"{hours * 60:.1f} minutes"
        elif hours < 24:
            time_str = f"{hours:.2f} hours"
        else:
            days = hours / 24
            time_str = f"{days:.2f} days ({hours:.1f} hours)"
        
        return hours, time_str
    
    def calculate_cost(self, gpu: GPUSpec, num_gpus: int = 1) -> Dict:
        """Calculate total training cost"""
        hours, time_str = self.calculate_training_time(gpu, num_gpus)
        total_cost = hours * gpu.price_per_hour * num_gpus
        
        # Check if GPU has enough memory
        memory_per_gpu = self.memory_required_gb / num_gpus
        memory_sufficient = memory_per_gpu <= gpu.memory_gb
        
        return {
            'gpu_name': gpu.name,
            'num_gpus': num_gpus,
            'training_time_hours': hours,
            'training_time_str': time_str,
            'cost_per_hour': gpu.price_per_hour * num_gpus,
            'total_cost': total_cost,
            'memory_required_gb': self.memory_required_gb,
            'memory_per_gpu_gb': memory_per_gpu,
            'gpu_memory_gb': gpu.memory_gb,
            'memory_sufficient': memory_sufficient,
            'price_source': gpu.price_source,
            'effective_tflops': gpu.tflops_tensor * 0.70 * min(1.0, gpu.memory_bandwidth_gbps / 1000.0),
            'relative_speed': gpu.relative_speed
        }
    
    def compare_all_configs(self) -> List[Dict]:
        """Compare all GPU configurations"""
        results = []

        # Single GPU configs
        for gpu_key, gpu in GPU_SPECS.items():
            result = self.calculate_cost(gpu, num_gpus=1)
            results.append(result)

        # Multi-GPU configs (for high-end GPUs)
        multi_gpu_configs = [
            'B200', 'H200', 'H100_SXM5', 'H100_80GB', 'GH200',
            'MI325X', 'MI300X', 'A100_80GB',
            'L40S', 'RTX_6000_Ada', 'RTX_5090', 'RTX_4090'
        ]

        for gpu_key in multi_gpu_configs:
            if gpu_key in GPU_SPECS:
                gpu = GPU_SPECS[gpu_key]
                for num_gpus in [2, 4, 8]:
                    result = self.calculate_cost(gpu, num_gpus=num_gpus)
                    results.append(result)

        # Sort by total cost
        results.sort(key=lambda x: x['total_cost'])

        return results
    
    def print_comparison_table(self):
        """Print formatted comparison table"""
        results = self.compare_all_configs()
        
        print("=" * 140)
        print("CLSTM-PPO TRAINING COST CALCULATOR")
        print("=" * 140)
        print(f"\nTraining Configuration:")
        print(f"  Episodes: {self.config.num_episodes:,}")
        print(f"  Episode Length: {self.config.episode_length} steps")
        print(f"  Total Steps: {self.total_steps:,}")
        print(f"  Batch Size: {self.config.batch_size}")
        print(f"  PPO Epochs: {self.config.ppo_epochs}")
        print(f"  Observation Dim: {self.config.observation_dim}")
        print(f"  Hidden Dim: {self.config.hidden_dim}")
        print(f"  LSTM Layers: {self.config.num_lstm_layers}")
        print(f"  Total FLOPs: {self.total_flops / 1e15:.2f} PetaFLOPs")
        print(f"  Memory Required: {self.memory_required_gb:.2f} GB")
        
        print("\n" + "=" * 140)
        print(f"{'GPU Configuration':<35} {'GPUs':<6} {'Time':<20} {'$/Hour':<10} {'Total Cost':<12} {'Memory':<15} {'Speed':<8} {'Status':<10}")
        print("=" * 140)
        
        for result in results:
            gpu_name = result['gpu_name']
            num_gpus = result['num_gpus']
            time_str = result['training_time_str']
            cost_per_hour = f"${result['cost_per_hour']:.2f}"
            total_cost = f"${result['total_cost']:.2f}"
            memory = f"{result['memory_per_gpu_gb']:.1f}/{result['gpu_memory_gb']:.0f} GB"
            speed = f"{result['relative_speed']:.1f}x"
            status = "✅ OK" if result['memory_sufficient'] else "❌ OOM"
            
            config_name = f"{gpu_name}"
            if num_gpus > 1:
                config_name = f"{num_gpus}x {gpu_name}"
            
            print(f"{config_name:<35} {num_gpus:<6} {time_str:<20} {cost_per_hour:<10} {total_cost:<12} {memory:<15} {speed:<8} {status:<10}")
        
        print("=" * 140)
        
        # Highlight current setup
        print("\n🎯 YOUR CURRENT SETUP:")
        print("=" * 140)

        current_setup = [r for r in results if 'RTX 6000 Ada' in r['gpu_name'] or 'RTX 4090' in r['gpu_name']]

        # Find dual GPU setup (RTX 6000 Ada + RTX 4090)
        rtx_6000 = next((r for r in results if 'RTX 6000 Ada' in r['gpu_name'] and r['num_gpus'] == 1), None)
        rtx_4090 = next((r for r in results if 'RTX 4090' in r['gpu_name'] and r['num_gpus'] == 1), None)

        if rtx_6000 and rtx_4090:
            # Calculate combined performance (heterogeneous setup)
            avg_speed = (rtx_6000['relative_speed'] + rtx_4090['relative_speed']) / 2
            avg_time_hours = (rtx_6000['training_time_hours'] + rtx_4090['training_time_hours']) / 2
            # For owned hardware, cost is $0 (only electricity)
            power_cost = 0.28  # From power analysis

            if avg_time_hours < 1:
                time_str = f"{avg_time_hours * 60:.1f} minutes"
            elif avg_time_hours < 24:
                time_str = f"{avg_time_hours:.2f} hours"
            else:
                days = avg_time_hours / 24
                time_str = f"{days:.2f} days ({avg_time_hours:.1f} hours)"

            print(f"Configuration: RTX 6000 Ada 48GB + RTX 4090 24GB (Heterogeneous)")
            print(f"  Training Time: {time_str}")
            print(f"  Cloud Cost: $0.00 (you own the hardware)")
            print(f"  Power Cost: ${power_cost:.2f} (electricity)")
            print(f"  Total Cost: ${power_cost:.2f}")
            print(f"  Average Speed: {avg_speed:.1f}x vs RTX 4090 baseline")
            print(f"  Memory: 48GB + 24GB = 72GB total")
            print(f"  Status: ✅ Sufficient memory")
            print(f"\n  Note: Heterogeneous setup may have ~10-15% overhead due to load balancing")
            print(f"        Actual training time may be closer to slower GPU (RTX 4090)")

        print("\n" + "=" * 140)
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        print("=" * 140)

        # Filter only memory-sufficient options
        valid_results = [r for r in results if r['memory_sufficient']]

        # Best value (lowest total cost)
        best_value = min(valid_results, key=lambda x: x['total_cost'])

        # Fastest (lowest training time)
        fastest = min(valid_results, key=lambda x: x['training_time_hours'])

        # Best value per hour (best cost efficiency)
        best_hourly = min(valid_results, key=lambda x: x['cost_per_hour'])

        # Best performance per dollar (speed / cost ratio)
        for r in valid_results:
            r['perf_per_dollar'] = r['relative_speed'] / max(r['total_cost'], 0.01)
        best_perf_per_dollar = max(valid_results, key=lambda x: x['perf_per_dollar'])

        # Best for single run (minimize total cost)
        single_run_best = min(valid_results, key=lambda x: x['total_cost'])

        # Best for 50 runs (minimize total cost × 50)
        for r in valid_results:
            r['cost_50_runs'] = r['total_cost'] * 50
        best_50_runs = min(valid_results, key=lambda x: x['cost_50_runs'])

        print(f"\n🏆 BEST VALUE (Lowest Total Cost):")
        print(f"   GPU: {best_value['gpu_name']} ({best_value['num_gpus']}x)")
        print(f"   Training Time: {best_value['training_time_str']}")
        print(f"   Total Cost: ${best_value['total_cost']:.2f}")
        print(f"   Cost per Hour: ${best_value['cost_per_hour']:.2f}")
        print(f"   Speed: {best_value['relative_speed']:.1f}x vs RTX 4090")
        print(f"   Memory: {best_value['memory_per_gpu_gb']:.1f}/{best_value['gpu_memory_gb']:.0f} GB per GPU")

        print(f"\n⚡ FASTEST (Lowest Training Time):")
        print(f"   GPU: {fastest['gpu_name']} ({fastest['num_gpus']}x)")
        print(f"   Training Time: {fastest['training_time_str']}")
        print(f"   Total Cost: ${fastest['total_cost']:.2f}")
        print(f"   Cost per Hour: ${fastest['cost_per_hour']:.2f}")
        print(f"   Speed: {fastest['relative_speed']:.1f}x vs RTX 4090")
        print(f"   Memory: {fastest['memory_per_gpu_gb']:.1f}/{fastest['gpu_memory_gb']:.0f} GB per GPU")

        print(f"\n💰 BEST HOURLY RATE (Cheapest $/Hour):")
        print(f"   GPU: {best_hourly['gpu_name']} ({best_hourly['num_gpus']}x)")
        print(f"   Training Time: {best_hourly['training_time_str']}")
        print(f"   Total Cost: ${best_hourly['total_cost']:.2f}")
        print(f"   Cost per Hour: ${best_hourly['cost_per_hour']:.2f}")
        print(f"   Speed: {best_hourly['relative_speed']:.1f}x vs RTX 4090")

        print(f"\n🎯 BEST PERFORMANCE PER DOLLAR (Speed/Cost Ratio):")
        print(f"   GPU: {best_perf_per_dollar['gpu_name']} ({best_perf_per_dollar['num_gpus']}x)")
        print(f"   Training Time: {best_perf_per_dollar['training_time_str']}")
        print(f"   Total Cost: ${best_perf_per_dollar['total_cost']:.2f}")
        print(f"   Cost per Hour: ${best_perf_per_dollar['cost_per_hour']:.2f}")
        print(f"   Speed: {best_perf_per_dollar['relative_speed']:.1f}x vs RTX 4090")
        print(f"   Perf/$ Ratio: {best_perf_per_dollar['perf_per_dollar']:.2f}")

        print(f"\n📊 BEST FOR SINGLE RUN:")
        print(f"   GPU: {single_run_best['gpu_name']} ({single_run_best['num_gpus']}x)")
        print(f"   Training Time: {single_run_best['training_time_str']}")
        print(f"   Total Cost: ${single_run_best['total_cost']:.2f}")

        print(f"\n🔄 BEST FOR 50 RUNS (Development):")
        print(f"   GPU: {best_50_runs['gpu_name']} ({best_50_runs['num_gpus']}x)")
        print(f"   Training Time per Run: {best_50_runs['training_time_str']}")
        print(f"   Cost per Run: ${best_50_runs['total_cost']:.2f}")
        print(f"   Total Cost (50 runs): ${best_50_runs['cost_50_runs']:.2f}")
        print(f"   Total Time (50 runs): {best_50_runs['training_time_hours'] * 50:.1f} hours")

        print("\n" + "=" * 140)

        # Cost-effectiveness analysis
        print("\n📈 COST-EFFECTIVENESS ANALYSIS:")
        print("=" * 140)

        # Compare top options
        print("\n🔍 Comparing Top Options:")
        print(f"\n{'Option':<40} {'Time':<15} {'Cost':<12} {'Speed':<8} {'Value Score':<12}")
        print("-" * 140)

        # Get top 5 by different metrics
        top_options = []

        # Add best value
        if best_value not in top_options:
            top_options.append(best_value)

        # Add fastest
        if fastest not in top_options:
            top_options.append(fastest)

        # Add best perf per dollar
        if best_perf_per_dollar not in top_options:
            top_options.append(best_perf_per_dollar)

        # Add best 50 runs
        if best_50_runs not in top_options:
            top_options.append(best_50_runs)

        # Add a few more top performers
        for r in sorted(valid_results, key=lambda x: x['total_cost'])[:5]:
            if r not in top_options and len(top_options) < 8:
                top_options.append(r)

        for opt in top_options:
            name = f"{opt['gpu_name']} ({opt['num_gpus']}x)"
            time_str = opt['training_time_str']
            cost_str = f"${opt['total_cost']:.2f}"
            speed_str = f"{opt['relative_speed']:.1f}x"
            value_score = opt['perf_per_dollar']
            value_str = f"{value_score:.2f}"

            # Add indicator
            indicators = []
            if opt == best_value:
                indicators.append("💰 CHEAPEST")
            if opt == fastest:
                indicators.append("⚡ FASTEST")
            if opt == best_perf_per_dollar:
                indicators.append("🎯 BEST VALUE")

            indicator_str = " | ".join(indicators) if indicators else ""

            print(f"{name:<40} {time_str:<15} {cost_str:<12} {speed_str:<8} {value_str:<12} {indicator_str}")

        print("\n" + "=" * 140)

        # Decision matrix
        print("\n🎯 DECISION MATRIX:")
        print("=" * 140)

        print("\n┌─────────────────────────────────────────────────────────────────────────┐")
        print("│ CHOOSE BASED ON YOUR PRIORITY:                                         │")
        print("├─────────────────────────────────────────────────────────────────────────┤")
        print(f"│ 💰 Minimize Cost:        {best_value['gpu_name']:<30} ({best_value['num_gpus']}x) │")
        print(f"│    → ${best_value['total_cost']:.2f} per run, {best_value['training_time_str']:<30}        │")
        print("│                                                                         │")
        print(f"│ ⚡ Minimize Time:         {fastest['gpu_name']:<30} ({fastest['num_gpus']}x) │")
        print(f"│    → {fastest['training_time_str']:<15}, ${fastest['total_cost']:.2f} per run                      │")
        print("│                                                                         │")
        print(f"│ 🎯 Best Value (Speed/$): {best_perf_per_dollar['gpu_name']:<30} ({best_perf_per_dollar['num_gpus']}x) │")
        print(f"│    → {best_perf_per_dollar['relative_speed']:.1f}x speed, ${best_perf_per_dollar['total_cost']:.2f} cost, {best_perf_per_dollar['perf_per_dollar']:.2f} perf/$          │")
        print("│                                                                         │")
        print(f"│ 🔄 Best for 50 Runs:     {best_50_runs['gpu_name']:<30} ({best_50_runs['num_gpus']}x) │")
        print(f"│    → ${best_50_runs['cost_50_runs']:.2f} total, {best_50_runs['training_time_hours'] * 50:.1f} hours total                  │")
        print("└─────────────────────────────────────────────────────────────────────────┘")

        # Time vs Cost trade-off
        print("\n⚖️  TIME vs COST TRADE-OFF:")
        print("=" * 140)

        print(f"\n{'GPU':<40} {'Time Saved':<20} {'Extra Cost':<15} {'Worth It?':<15}")
        print("-" * 140)

        # Compare fastest vs best value
        time_saved_hours = best_value['training_time_hours'] - fastest['training_time_hours']
        time_saved_pct = (time_saved_hours / best_value['training_time_hours']) * 100
        extra_cost = fastest['total_cost'] - best_value['total_cost']
        cost_per_hour_saved = extra_cost / time_saved_hours if time_saved_hours > 0 else 0

        if time_saved_hours < 0.017:  # Less than 1 minute
            time_saved_str = f"{time_saved_hours * 60:.1f} seconds ({time_saved_pct:.1f}%)"
        elif time_saved_hours < 1:
            time_saved_str = f"{time_saved_hours * 60:.1f} minutes ({time_saved_pct:.1f}%)"
        else:
            time_saved_str = f"{time_saved_hours:.2f} hours ({time_saved_pct:.1f}%)"

        worth_it = "✅ YES" if cost_per_hour_saved < 10 else "⚠️ MAYBE" if cost_per_hour_saved < 50 else "❌ NO"

        print(f"Fastest vs Best Value:")
        print(f"  {fastest['gpu_name']:<38} {time_saved_str:<20} ${extra_cost:>+.2f} ({cost_per_hour_saved:.2f}$/hr saved) {worth_it:<15}")

        # Compare a few other options
        mid_options = sorted(valid_results, key=lambda x: x['training_time_hours'])[1:4]
        for opt in mid_options:
            if opt != fastest and opt != best_value:
                time_saved_hours = best_value['training_time_hours'] - opt['training_time_hours']
                time_saved_pct = (time_saved_hours / best_value['training_time_hours']) * 100
                extra_cost = opt['total_cost'] - best_value['total_cost']
                cost_per_hour_saved = extra_cost / time_saved_hours if time_saved_hours > 0 else 0

                if time_saved_hours < 0.017:
                    time_saved_str = f"{time_saved_hours * 60:.1f} seconds ({time_saved_pct:.1f}%)"
                elif time_saved_hours < 1:
                    time_saved_str = f"{time_saved_hours * 60:.1f} minutes ({time_saved_pct:.1f}%)"
                else:
                    time_saved_str = f"{time_saved_hours:.2f} hours ({time_saved_pct:.1f}%)"

                worth_it = "✅ YES" if cost_per_hour_saved < 10 else "⚠️ MAYBE" if cost_per_hour_saved < 50 else "❌ NO"

                name = f"{opt['gpu_name']} ({opt['num_gpus']}x)"
                print(f"  {name:<38} {time_saved_str:<20} ${extra_cost:>+.2f} ({cost_per_hour_saved:.2f}$/hr saved) {worth_it:<15}")

        print("\n" + "=" * 140)

        # Pricing sources
        print("\n📊 PRICING SOURCES:")
        print("=" * 140)
        sources = set(gpu.price_source for gpu in GPU_SPECS.values())
        for source in sorted(sources):
            print(f"  • {source}")
        print("\n  Note: Prices are on-demand rates as of October 2025")
        print("        Spot/preemptible instances can be 50-80% cheaper")
        print("        Reserved instances offer 30-50% discounts")
        
        print("\n" + "=" * 140)


def main():
    """Main function"""
    # Default training configuration (5000 episodes)
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
    
    calculator = TrainingCostCalculator(config)
    calculator.print_comparison_table()
    
    # Save results to JSON
    results = calculator.compare_all_configs()
    output_file = 'training_cost_analysis.json'
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'num_episodes': config.num_episodes,
                'episode_length': config.episode_length,
                'total_steps': calculator.total_steps,
                'memory_required_gb': calculator.memory_required_gb,
                'total_petaflops': calculator.total_flops / 1e15
            },
            'results': results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("\n" + "=" * 140)


if __name__ == '__main__':
    main()

