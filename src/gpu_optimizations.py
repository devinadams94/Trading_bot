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


def distributed_training_wrapper(rank: int, world_size: int, train_fn: Callable, *args, **kwargs):
    """
    Wrapper for distributed training
    Args:
        rank: Process rank
        world_size: Total number of processes
        train_fn: Training function to run
        *args, **kwargs: Arguments for training function
    """
    try:
        # Setup distributed training
        optimizer = GPUOptimizer()
        optimizer.setup_distributed(rank, world_size)
        
        # Run training function
        train_fn(rank, world_size, *args, **kwargs)
        
    finally:
        # Cleanup
        optimizer.cleanup_distributed()


def launch_distributed_training(train_fn: Callable, world_size: int, *args, **kwargs):
    """
    Launch distributed training across multiple GPUs
    Args:
        train_fn: Training function (must accept rank and world_size as first two args)
        world_size: Number of GPUs to use
        *args, **kwargs: Additional arguments for training function
    """
    if world_size <= 1:
        # Single GPU or CPU training
        train_fn(0, 1, *args, **kwargs)
    else:
        # Multi-GPU distributed training
        mp.spawn(
            distributed_training_wrapper,
            args=(world_size, train_fn, *args),
            nprocs=world_size,
            join=True
        )

