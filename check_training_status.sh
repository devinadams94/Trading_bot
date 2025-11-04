#!/bin/bash
# Check training status after SSH reconnection

echo "🔍 TRAINING STATUS CHECK"
echo "========================================"
echo ""

# Check if training process is running
echo "1️⃣ Process Status:"
PROCESS=$(ps aux | grep train_enhanced_clstm_ppo.py | grep -v grep)
if [ -z "$PROCESS" ]; then
    echo "   ❌ Training process NOT running"
else
    echo "   ✅ Training process IS running"
    echo "   Process details:"
    echo "$PROCESS" | awk '{printf "      PID: %s | CPU: %s%% | MEM: %s%% | Time: %s\n", $2, $3, $4, $10}'
fi
echo ""

# Check GPU usage
echo "2️⃣ GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)
    if [ -n "$GPU_INFO" ]; then
        echo "   GPU Utilization: $(echo $GPU_INFO | awk -F',' '{print $1}')%"
        echo "   GPU Memory: $(echo $GPU_INFO | awk -F',' '{print $2}') MB / $(echo $GPU_INFO | awk -F',' '{print $3}') MB"
    else
        echo "   ⚠️ No GPU info available"
    fi
else
    echo "   ⚠️ nvidia-smi not found"
fi
echo ""

# Check latest log file
echo "3️⃣ Latest Log File:"
LATEST_LOG=$(ls -t logs/training_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "   📄 $LATEST_LOG"
    echo "   Last modified: $(stat -c %y "$LATEST_LOG" 2>/dev/null || stat -f "%Sm" "$LATEST_LOG")"
    echo ""
    echo "   Last 10 lines:"
    tail -10 "$LATEST_LOG" | sed 's/^/      /'
else
    echo "   ❌ No log files found"
fi
echo ""

# Check checkpoints
echo "4️⃣ Checkpoints:"
if [ -d "checkpoints/production_run" ]; then
    CHECKPOINT_COUNT=$(ls checkpoints/production_run/*.pt 2>/dev/null | wc -l)
    echo "   Total checkpoints: $CHECKPOINT_COUNT"
    echo "   Latest checkpoints:"
    ls -lth checkpoints/production_run/*.pt 2>/dev/null | head -3 | awk '{print "      " $9 " (" $6 " " $7 " " $8 ")"}'
else
    echo "   ⚠️ No checkpoint directory found"
fi
echo ""

# Check for episode progress
echo "5️⃣ Training Progress:"
if [ -n "$LATEST_LOG" ]; then
    LAST_EPISODE=$(grep -o "Episode [0-9]*" "$LATEST_LOG" | tail -1)
    LAST_REWARD=$(grep "Episode.*Reward" "$LATEST_LOG" | tail -1)
    
    if [ -n "$LAST_EPISODE" ]; then
        echo "   $LAST_EPISODE"
        if [ -n "$LAST_REWARD" ]; then
            echo "   $(echo $LAST_REWARD | sed 's/.*INFO - //')"
        fi
    else
        echo "   ⚠️ No episode information found yet"
    fi
else
    echo "   ❌ No log file to check"
fi
echo ""

# Provide helpful commands
echo "========================================"
echo "📋 Helpful Commands:"
echo ""
echo "   View live log:"
echo "   $ tail -f $LATEST_LOG"
echo ""
echo "   Monitor GPU:"
echo "   $ watch -n 1 nvidia-smi"
echo ""
echo "   Kill training:"
echo "   $ pkill -f train_enhanced_clstm_ppo.py"
echo ""
echo "   Attach to tmux (if running in tmux):"
echo "   $ tmux attach -t training"
echo ""
echo "========================================"

