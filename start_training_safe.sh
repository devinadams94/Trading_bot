#!/bin/bash
# Safe training starter - automatically uses tmux

echo "🚀 Safe Training Starter"
echo "========================"
echo ""

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "⚠️ tmux not found. Installing..."
    sudo apt-get update && sudo apt-get install -y tmux
fi

# Check if training is already running
if pgrep -f train_enhanced_clstm_ppo.py > /dev/null; then
    echo "⚠️ Training process already running!"
    echo ""
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t training"
    echo "  2. View logs: tail -f logs/training_*.log"
    echo "  3. Kill existing: pkill -f train_enhanced_clstm_ppo.py"
    exit 1
fi

# Check if tmux session already exists
if tmux has-session -t training 2>/dev/null; then
    echo "⚠️ tmux session 'training' already exists"
    echo ""
    echo "Options:"
    echo "  1. Attach: tmux attach -t training"
    echo "  2. Kill old session: tmux kill-session -t training"
    read -p "Kill old session and start new? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t training
    else
        exit 0
    fi
fi

# Parse arguments or use defaults
EPISODES=${1:-5000}
GPUS=${2:-1}
MODE=${3:-production}

echo "Configuration:"
echo "  Episodes: $EPISODES"
echo "  GPUs: $GPUS"
echo "  Mode: $MODE"
echo ""

# Build command based on mode
if [ "$MODE" == "quick" ]; then
    CMD="python train_enhanced_clstm_ppo.py --quick-test --num_gpus $GPUS --checkpoint-dir checkpoints/test --fresh-start"
    echo "🏃 Quick test mode (3 symbols, 90 days, 100 episodes)"
elif [ "$MODE" == "production" ]; then
    CMD="python train_enhanced_clstm_ppo.py \
        --num_episodes $EPISODES \
        --num_gpus $GPUS \
        --enable-multi-leg \
        --use-ensemble \
        --num-ensemble-models 3 \
        --early-stopping-patience 500 \
        --checkpoint-dir checkpoints/production_run \
        --resume"
    echo "🚀 Production mode (23 symbols, 730 days, $EPISODES episodes)"
else
    CMD="python train_enhanced_clstm_ppo.py --num_episodes $EPISODES --num_gpus $GPUS --checkpoint-dir checkpoints/$MODE --resume"
    echo "🔧 Custom mode: $MODE"
fi

echo ""
echo "Command: $CMD"
echo ""
read -p "Start training in tmux? (Y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    # Start tmux session with training
    tmux new-session -d -s training "$CMD"
    
    echo ""
    echo "✅ Training started in tmux session 'training'"
    echo ""
    echo "📋 Next steps:"
    echo "  1. Attach to session:  tmux attach -t training"
    echo "  2. View logs:          tail -f logs/training_*.log"
    echo "  3. Check status:       bash check_training_status.sh"
    echo "  4. Detach from tmux:   Ctrl+B, then D"
    echo ""
    echo "💡 You can now safely disconnect SSH - training will continue!"
    echo ""
    
    # Ask if user wants to attach now
    read -p "Attach to session now? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        tmux attach -t training
    fi
else
    echo "Cancelled."
fi

