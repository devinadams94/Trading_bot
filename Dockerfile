# ============================================================================
# CLSTM-PPO Options Trading Bot - Optimized for DigitalOcean GPU Droplets
# ============================================================================
# Base: NVIDIA CUDA 12.1 with cuDNN 8 (compatible with PyTorch 2.0+)
# Target: DigitalOcean GPU Droplets (H100, A100, or RTX 4000 Ada)
# ============================================================================

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# ============================================================================
# Environment Configuration
# ============================================================================
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # CUDA/PyTorch optimizations
    CUDA_LAUNCH_BLOCKING=0 \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0" \
    # Memory optimizations
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    # Disable unnecessary features
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1

# ============================================================================
# System Dependencies
# ============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python
    python3.10 \
    python3-pip \
    python3-dev \
    # Build tools
    build-essential \
    gcc \
    g++ \
    make \
    # Utilities
    wget \
    curl \
    git \
    ca-certificates \
    # TA-Lib dependencies
    libgomp1 \
    # Minimal graphics libs (for matplotlib/seaborn)
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ============================================================================
# Install TA-Lib (Technical Analysis Library)
# ============================================================================
RUN wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz && \
    ldconfig

# ============================================================================
# Python Environment Setup
# ============================================================================
WORKDIR /app

# Upgrade pip and install build tools
RUN pip3 install --no-cache-dir --upgrade \
    pip==24.0 \
    setuptools==69.0.3 \
    wheel==0.42.0

# ============================================================================
# Install PyTorch with CUDA 12.1 Support (FIRST - largest dependency)
# ============================================================================
RUN pip3 install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cu121

# ============================================================================
# Install Core Dependencies (in order of importance)
# ============================================================================
# Copy requirements for layer caching
COPY requirements.txt .

# Install in batches to optimize layer caching
# Batch 1: Core ML/Data Science
RUN pip3 install --no-cache-dir \
    numpy==1.26.3 \
    pandas==2.1.4 \
    scipy==1.11.4 \
    scikit-learn==1.4.0

# Batch 2: RL Framework
RUN pip3 install --no-cache-dir \
    gymnasium==0.29.1 \
    stable-baselines3==2.2.1

# Batch 3: Technical Indicators
# Note: TA-Lib Python wrapper needs to be installed without build isolation
# to use the system TA-Lib library we compiled earlier
RUN pip3 install --no-cache-dir --no-build-isolation \
    TA-Lib \
    ta==0.11.0

# Batch 4: Visualization & Monitoring
RUN pip3 install --no-cache-dir \
    matplotlib==3.8.2 \
    seaborn==0.13.1 \
    tensorboard==2.15.1 \
    tqdm==4.66.1

# Batch 5: Trading APIs
# Note: alpaca-py and alpaca-trade-api have conflicting websockets dependencies
# Installing alpaca-py (newer SDK) and letting pip resolve websockets version
RUN pip3 install --no-cache-dir \
    alpaca-py==0.20.2

# Install alpaca-trade-api separately without version pinning for websockets
RUN pip3 install --no-cache-dir \
    alpaca-trade-api

# Batch 6: Async & Utilities
RUN pip3 install --no-cache-dir \
    aiohttp==3.9.1 \
    python-dotenv==1.0.0 \
    pytz==2024.1 \
    pyyaml==6.0.1 \
    boto3==1.34.34

# Batch 7: Optional - Data formats (for flat file support)
RUN pip3 install --no-cache-dir \
    pyarrow==14.0.2 \
    fastparquet==2024.2.0

# Batch 8: Optional - Websockets (for real-time data)
RUN pip3 install --no-cache-dir \
    websockets==12.0

# ============================================================================
# SKIP UNNECESSARY PACKAGES (commented out from original requirements.txt)
# ============================================================================
# These are NOT needed for training and bloat the image:
# - transformers (NLP - not used)
# - sentencepiece (NLP - not used)
# - ultralytics (YOLO/CV - not used)
# - pillow (image processing - minimal use)
# - wandb (optional - use TensorBoard instead)
# - loguru (using standard logging)
# - pytest (dev only)

# ============================================================================
# Application Code
# ============================================================================
COPY . .

# ============================================================================
# Create Runtime Directories
# ============================================================================
RUN mkdir -p \
    data/flat_files \
    data/options_cache \
    logs \
    runs \
    checkpoints \
    && chmod -R 755 /app

# ============================================================================
# Health Check
# ============================================================================
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || exit 1

# ============================================================================
# Expose Ports
# ============================================================================
EXPOSE 6006

# ============================================================================
# Runtime Configuration
# ============================================================================
# Set Python to use UTF-8 encoding
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# ============================================================================
# Entry Point
# ============================================================================
# Default: Show help (override with docker run command)
CMD ["python3", "train_enhanced_clstm_ppo.py", "--help"]

