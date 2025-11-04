"""
GPU Optimization utilities for multi-GPU training
Implements efficient multi-GPU training with DistributedDataParallel
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


class GPUOptimizer:
    """
    GPU optimization utilities for faster training
    Supports:
    - Multi-GPU training with DistributedDataParallel
    - Mixed precision training (FP16)
    - Gradient accumulation
    - Optimized data loading
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.world_size = torch.cuda.device_count()
        self.use_mixed_precision = self.config.get('mixed_precision', True)
        self.gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        
    def setup_device(self, rank: Optional[int] = None) -> torch.device:
        """
        Setup device for training
        Args:
            rank: GPU rank for distributed training (None for single GPU/CPU)
        Returns:
            torch.device
        """
        if not torch.cuda.is_available():
            logger.warning("⚠️ No GPU available, using CPU")
            return torch.device("cpu")
        
        gpu_count = torch.cuda.device_count()
        
        if gpu_count == 0:
            logger.warning("⚠️ No GPU available, using CPU")
            return torch.device("cpu")
        elif gpu_count == 1:
            logger.info("🔥 Single GPU setup")
            device = torch.device("cuda:0")
            self._optimize_single_gpu()
            return device
        else:
            logger.info(f"🔥 Multi-GPU setup: {gpu_count} GPUs available")
            if rank is not None:
                device = torch.device(f"cuda:{rank}")
            else:
                device = torch.device("cuda:0")
            return device
    
    def _optimize_single_gpu(self):
        """Optimize settings for single GPU"""
        if torch.cuda.is_available():
            # Enable TF32 for faster training on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable cuDNN autotuner
            torch.backends.cudnn.benchmark = True
            
            # Set memory allocation strategy
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            
            logger.info("✅ Single GPU optimizations enabled (TF32, cuDNN autotuner)")
    
    def setup_distributed(self, rank: int, world_size: int):
        """
        Setup distributed training
        Args:
            rank: Current process rank
            world_size: Total number of processes
        """
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl',  # Use NCCL for GPU training
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        # Set device for this process
        torch.cuda.set_device(rank)
        
        logger.info(f"✅ Distributed training initialized: rank {rank}/{world_size}")
    
    def cleanup_distributed(self):
        """Cleanup distributed training"""
        if dist.is_initialized():
            dist.destroy_process_group()
    
    def wrap_model_for_distributed(self, model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
        """
        Wrap model for distributed training
        Args:
            model: PyTorch model
            device: Device to use
        Returns:
            Wrapped model (DDP if multi-GPU, otherwise original)
        """
        model = model.to(device)
        
        if dist.is_initialized():
            # Use DistributedDataParallel for multi-GPU
            model = DDP(
                model,
                device_ids=[device.index] if device.type == 'cuda' else None,
                find_unused_parameters=True  # For complex models with optional paths
            )
            logger.info(f"✅ Model wrapped with DistributedDataParallel on {device}")
        
        return model
    
    def create_gradient_scaler(self) -> Optional[torch.cuda.amp.GradScaler]:
        """
        Create gradient scaler for mixed precision training
        Returns:
            GradScaler if mixed precision enabled, None otherwise
        """
        if self.use_mixed_precision and torch.cuda.is_available():
            scaler = torch.cuda.amp.GradScaler()
            logger.info("✅ Mixed precision training enabled (FP16)")
            return scaler
        return None
    
    def optimize_dataloader_settings(self) -> dict:
        """
        Get optimized DataLoader settings
        Returns:
            Dict with optimized settings
        """
        settings = {
            'num_workers': min(8, os.cpu_count() or 1),
            'pin_memory': torch.cuda.is_available(),
            'persistent_workers': True if (os.cpu_count() or 1) > 1 else False,
            'prefetch_factor': 2 if (os.cpu_count() or 1) > 1 else None,
        }
        
        logger.info(f"✅ Optimized DataLoader settings: {settings}")
        return settings
    
    def get_memory_stats(self, device: torch.device) -> dict:
        """
        Get GPU memory statistics
        Args:
            device: Device to check
        Returns:
            Dict with memory stats
        """
        if device.type != 'cuda':
            return {}
        
        stats = {
            'allocated_gb': torch.cuda.memory_allocated(device) / 1e9,
            'reserved_gb': torch.cuda.memory_reserved(device) / 1e9,
            'max_allocated_gb': torch.cuda.max_memory_allocated(device) / 1e9,
        }
        
        return stats
    
    def clear_cache(self):
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")
    
    @staticmethod
    def get_optimal_batch_size(model: torch.nn.Module, device: torch.device, 
                               input_shape: tuple, max_batch_size: int = 512) -> int:
        """
        Find optimal batch size through binary search
        Args:
            model: PyTorch model
            device: Device to test on
            input_shape: Shape of single input (without batch dimension)
            max_batch_size: Maximum batch size to try
        Returns:
            Optimal batch size
        """
        if device.type != 'cuda':
            return 32  # Default for CPU
        
        model = model.to(device)
        model.eval()
        
        def try_batch_size(batch_size: int) -> bool:
            """Test if batch size fits in memory"""
            try:
                # Create dummy input
                dummy_input = torch.randn(batch_size, *input_shape, device=device)
                
                # Forward pass
                with torch.no_grad():
                    _ = model(dummy_input)
                
                # Clear cache
                del dummy_input
                torch.cuda.empty_cache()
                return True
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    torch.cuda.empty_cache()
                    return False
                raise e
        
        # Binary search for optimal batch size
        low, high = 1, max_batch_size
        optimal = 32
        
        while low <= high:
            mid = (low + high) // 2
            if try_batch_size(mid):
                optimal = mid
                low = mid + 1
            else:
                high = mid - 1
        
        logger.info(f"✅ Optimal batch size found: {optimal}")
        return optimal


def launch_distributed_training(train_fn: Callable, world_size: int = -1, *args, **kwargs):
    """
    Launch distributed training across multiple GPUs
    Args:
        train_fn: Training function (must accept rank and world_size as first two args)
        world_size: Number of GPUs to use (-1 for all available, 0 for CPU)
        *args, **kwargs: Additional arguments for training function

    Example:
        def my_train_fn(rank, world_size, config):
            # Setup distributed
            setup_distributed(rank, world_size)

            # Create model and wrap with DDP
            model = MyModel().to(rank)
            model = DDP(model, device_ids=[rank])

            # Training loop
            ...

            # Cleanup
            dist.destroy_process_group()

        # Launch on all GPUs
        launch_distributed_training(my_train_fn, world_size=-1, config={})
    """
    # Determine number of GPUs to use
    available_gpus = torch.cuda.device_count()

    if world_size == -1:
        # Use all available GPUs
        world_size = available_gpus
    elif world_size == 0:
        # CPU training
        logger.info("🔧 Running on CPU (no GPU acceleration)")
        train_fn(0, 1, *args, **kwargs)
        return
    else:
        # Use specified number of GPUs
        world_size = min(world_size, available_gpus)

    if world_size == 0:
        logger.warning("⚠️ No GPUs available, falling back to CPU")
        train_fn(0, 1, *args, **kwargs)
        return

    logger.info(f"🚀 Launching distributed training on {world_size} GPU(s)")
    logger.info(f"   Available GPUs: {available_gpus}")

    if world_size == 1:
        # Single GPU training (no distributed setup needed)
        logger.info("🔥 Single GPU training mode")
        train_fn(0, 1, *args, **kwargs)
    else:
        # Multi-GPU distributed training
        logger.info(f"🔥 Multi-GPU distributed training mode ({world_size} GPUs)")
        mp.spawn(
            train_fn,
            args=(world_size, *args),
            nprocs=world_size,
            join=True
        )

    logger.info("✅ Distributed training completed")


def get_gpu_info() -> dict:
    """
    Get detailed information about available GPUs
    Returns:
        Dict with GPU information
    """
    if not torch.cuda.is_available():
        return {
            'available': False,
            'count': 0,
            'devices': []
        }

    gpu_count = torch.cuda.device_count()
    devices = []

    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        devices.append({
            'id': i,
            'name': props.name,
            'total_memory_gb': props.total_memory / 1e9,
            'compute_capability': f"{props.major}.{props.minor}",
            'multi_processor_count': props.multi_processor_count
        })

    return {
        'available': True,
        'count': gpu_count,
        'devices': devices
    }


def print_gpu_info():
    """Print detailed GPU information"""
    info = get_gpu_info()

    if not info['available']:
        print("❌ No GPUs available")
        return

    print(f"\n{'='*80}")
    print(f"🔥 GPU Information")
    print(f"{'='*80}")
    print(f"Total GPUs: {info['count']}")
    print()

    for device in info['devices']:
        print(f"GPU {device['id']}: {device['name']}")
        print(f"  Memory: {device['total_memory_gb']:.2f} GB")
        print(f"  Compute Capability: {device['compute_capability']}")
        print(f"  Multiprocessors: {device['multi_processor_count']}")
        print()

    print(f"{'='*80}\n")


def estimate_training_speedup(world_size: int, efficiency: float = 0.85) -> dict:
    """
    Estimate training speedup with multi-GPU
    Args:
        world_size: Number of GPUs
        efficiency: Scaling efficiency (0.0-1.0, default 0.85)
    Returns:
        Dict with speedup estimates
    """
    if world_size <= 1:
        return {
            'world_size': world_size,
            'theoretical_speedup': 1.0,
            'actual_speedup': 1.0,
            'efficiency': 1.0
        }

    # Theoretical speedup (linear)
    theoretical_speedup = world_size

    # Actual speedup (accounting for communication overhead)
    # Efficiency typically decreases with more GPUs
    # Common scaling: 1 GPU = 100%, 2 GPU = 95%, 4 GPU = 90%, 8 GPU = 85%
    actual_efficiency = efficiency ** (np.log2(world_size))
    actual_speedup = world_size * actual_efficiency

    return {
        'world_size': world_size,
        'theoretical_speedup': theoretical_speedup,
        'actual_speedup': actual_speedup,
        'efficiency': actual_efficiency,
        'overhead_percent': (1 - actual_efficiency) * 100
    }


def print_training_speedup_estimates():
    """Print estimated training speedup for different GPU configurations"""
    print(f"\n{'='*80}")
    print(f"⚡ Estimated Training Speedup (Multi-GPU)")
    print(f"{'='*80}")
    print(f"{'GPUs':<8} {'Theoretical':<15} {'Actual':<15} {'Efficiency':<12} {'Overhead':<10}")
    print(f"{'-'*80}")

    for gpus in [1, 2, 4, 8]:
        est = estimate_training_speedup(gpus)
        print(f"{est['world_size']:<8} "
              f"{est['theoretical_speedup']:.2f}x{'':<11} "
              f"{est['actual_speedup']:.2f}x{'':<11} "
              f"{est['efficiency']*100:.1f}%{'':<8} "
              f"{est['overhead_percent']:.1f}%")

    print(f"{'='*80}")
    print(f"Note: Actual speedup depends on model size, batch size, and communication overhead")
    print(f"{'='*80}\n")


259

