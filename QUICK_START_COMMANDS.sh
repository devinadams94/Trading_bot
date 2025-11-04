#!/bin/bash
# Quick Start Commands for GPU Cloud Training
# Trading Bot - Enhanced CLSTM-PPO
# Date: 2025-11-04

echo "=========================================="
echo "GPU Cloud Training - Quick Start"
echo "=========================================="
echo ""

# ============================================
# OPTION 1: PRODUCTION TRAINING (RECOMMENDED)
# ============================================
# - 5000 episodes
# - All available GPUs
# - Multi-leg strategies (91 actions)
# - Ensemble methods (3 models)
# - Early stopping enabled
# - Time: 12-24 hours on 4x A100
# - Cost: $60-300

production_training() {
    echo "🚀 Starting PRODUCTION training..."
    tmux new-session -d -s training "python train_enhanced_clstm_ppo.py \
        --num_episodes 5000 \
        --num_gpus -1 \
        --enable-multi-leg \
        --use-ensemble \
        --num-ensemble-models 3 \
        --early-stopping-patience 500 \
        --checkpoint-dir checkpoints/production_run \
        --resume"
    
    echo "✅ Training started in tmux session 'training'"
    echo "📊 Monitor: tmux attach -t training"
    echo "📝 Logs: tail -f logs/training_*.log"
    echo "🖥️  GPU: watch -n 1 nvidia-smi"
}

# ============================================
# OPTION 2: VALIDATION TEST (QUICK)
# ============================================
# - 100 episodes
# - Single GPU
# - Multi-leg strategies
# - Time: 30-60 minutes
# - Cost: $0.50-3

validation_test() {
    echo "🧪 Starting VALIDATION test..."
    python train_enhanced_clstm_ppo.py \
        --num_episodes 100 \
        --num_gpus 1 \
        --enable-multi-leg \
        --checkpoint-dir checkpoints/validation_test \
        --fresh-start
}

# ============================================
# OPTION 3: MAXIMUM PERFORMANCE (8 GPUs)
# ============================================
# - 10000 episodes
# - 8 GPUs
# - Multi-leg strategies
# - Ensemble (5 models)
# - Time: 24-48 hours
# - Cost: $150-600

max_performance() {
    echo "⚡ Starting MAXIMUM PERFORMANCE training..."
    tmux new-session -d -s training "python train_enhanced_clstm_ppo.py \
        --num_episodes 10000 \
        --num_gpus 8 \
        --enable-multi-leg \
        --use-ensemble \
        --num-ensemble-models 5 \
        --early-stopping-patience 1000 \
        --checkpoint-dir checkpoints/max_performance \
        --resume"
    
    echo "✅ Training started in tmux session 'training'"
    echo "📊 Monitor: tmux attach -t training"
}

# ============================================
# OPTION 4: BUDGET TRAINING (1 GPU)
# ============================================
# - 3000 episodes
# - Single GPU
# - Multi-leg strategies
# - Time: 8-12 hours
# - Cost: $4-36

budget_training() {
    echo "💰 Starting BUDGET training..."
    tmux new-session -d -s training "python train_enhanced_clstm_ppo.py \
        --num_episodes 3000 \
        --num_gpus 1 \
        --enable-multi-leg \
        --early-stopping-patience 300 \
        --checkpoint-dir checkpoints/budget_run \
        --resume"
    
    echo "✅ Training started in tmux session 'training'"
    echo "📊 Monitor: tmux attach -t training"
}

# ============================================
# SETUP FUNCTIONS
# ============================================

setup_environment() {
    echo "🔧 Setting up training environment..."
    
    # Update system
    sudo apt-get update
    sudo apt-get install -y git python3-pip tmux htop nvtop
    
    # Install dependencies
    pip install -r requirements.txt
    
    # Verify GPU
    python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"
    
    # Create directories
    mkdir -p logs checkpoints/production_run
    
    echo "✅ Setup complete!"
}

verify_setup() {
    echo "🔍 Verifying setup..."
    
    echo "1. Checking Python..."
    python --version
    
    echo "2. Checking PyTorch..."
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    
    echo "3. Checking CUDA..."
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    
    echo "4. Checking GPUs..."
    python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
    
    echo "5. Checking dependencies..."
    python -c "import numpy, pandas, torch, alpaca; print('✅ All dependencies installed')"
    
    echo "6. Checking .env file..."
    if [ -f .env ]; then
        echo "✅ .env file found"
    else
        echo "⚠️  .env file not found - create from .env.example"
    fi
    
    echo "7. Checking GPU memory..."
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    
    echo ""
    echo "✅ Verification complete!"
}

monitor_training() {
    echo "📊 Monitoring options:"
    echo ""
    echo "1. Attach to training session:"
    echo "   tmux attach -t training"
    echo ""
    echo "2. View logs:"
    echo "   tail -f logs/training_*.log"
    echo ""
    echo "3. Monitor GPU usage:"
    echo "   watch -n 1 nvidia-smi"
    echo ""
    echo "4. Monitor GPU (better):"
    echo "   nvtop"
    echo ""
    echo "5. Check training progress:"
    echo "   grep 'Episode' logs/training_*.log | tail -20"
}

# ============================================
# MENU
# ============================================

show_menu() {
    echo ""
    echo "=========================================="
    echo "GPU Cloud Training - Quick Start Menu"
    echo "=========================================="
    echo ""
    echo "SETUP:"
    echo "  1) Setup environment (first time only)"
    echo "  2) Verify setup"
    echo ""
    echo "TRAINING:"
    echo "  3) Production training (RECOMMENDED) - 5000 episodes, all GPUs"
    echo "  4) Validation test - 100 episodes, 1 GPU"
    echo "  5) Maximum performance - 10000 episodes, 8 GPUs"
    echo "  6) Budget training - 3000 episodes, 1 GPU"
    echo ""
    echo "MONITORING:"
    echo "  7) Show monitoring commands"
    echo ""
    echo "  0) Exit"
    echo ""
    read -p "Select option: " choice
    
    case $choice in
        1) setup_environment ;;
        2) verify_setup ;;
        3) production_training ;;
        4) validation_test ;;
        5) max_performance ;;
        6) budget_training ;;
        7) monitor_training ;;
        0) exit 0 ;;
        *) echo "Invalid option" ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
    show_menu
}

# ============================================
# MAIN
# ============================================

# If script is run with argument, execute that function
if [ $# -eq 0 ]; then
    # No arguments - show menu
    show_menu
else
    # Execute function based on argument
    case $1 in
        setup) setup_environment ;;
        verify) verify_setup ;;
        production) production_training ;;
        validation) validation_test ;;
        max) max_performance ;;
        budget) budget_training ;;
        monitor) monitor_training ;;
        *)
            echo "Usage: $0 [setup|verify|production|validation|max|budget|monitor]"
            echo ""
            echo "Or run without arguments for interactive menu"
            exit 1
            ;;
    esac
fi

