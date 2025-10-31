#!/usr/bin/env python3
"""Check GPU availability and PyTorch configuration"""

import torch
import sys

print("=" * 60)
print("GPU/CUDA Configuration Check")
print("=" * 60)

print(f"\nPyTorch version: {torch.__version__}")
print(f"Python version: {sys.version}")

print(f"\nCUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
else:
    print("\nNo CUDA GPUs detected!")
    print("\nPossible reasons:")
    print("1. No NVIDIA GPU installed")
    print("2. NVIDIA drivers not installed")
    print("3. CUDA toolkit not installed")
    print("4. PyTorch CPU-only version installed")
    
    print("\nTo install PyTorch with CUDA support:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

# Test tensor operations
print("\n" + "=" * 60)
print("Testing Tensor Operations")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Create a test tensor
x = torch.randn(1000, 1000).to(device)
y = torch.randn(1000, 1000).to(device)

# Time a matrix multiplication
import time
start = time.time()
z = torch.matmul(x, y)
torch.cuda.synchronize() if torch.cuda.is_available() else None
end = time.time()

print(f"Matrix multiplication (1000x1000) took: {(end - start)*1000:.2f} ms")

# Memory usage
if torch.cuda.is_available():
    print(f"\nGPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")