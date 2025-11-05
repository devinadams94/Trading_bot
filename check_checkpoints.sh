#!/bin/bash

echo "=========================================="
echo "CHECKPOINT DIAGNOSTIC SCRIPT"
echo "=========================================="
echo ""

echo "1. Current directory:"
pwd
echo ""

echo "2. Checking if checkpoints directory exists:"
if [ -d "checkpoints" ]; then
    echo "✅ checkpoints/ directory exists"
    ls -lh checkpoints/
else
    echo "❌ checkpoints/ directory does NOT exist"
fi
echo ""

echo "3. Checking for checkpoint subdirectories:"
find checkpoints -maxdepth 1 -type d 2>/dev/null | head -10
echo ""

echo "4. Checking for .pt model files:"
find checkpoints -name "*.pt" 2>/dev/null | wc -l
echo "   Total .pt files found"
echo ""

echo "5. Checking for training_state.json files:"
find checkpoints -name "training_state.json" 2>/dev/null
echo ""

echo "6. Checking most recent checkpoint activity:"
find checkpoints -type f -name "*.pt" -o -name "*.json" 2>/dev/null | xargs ls -lht 2>/dev/null | head -5
echo ""

echo "7. Checking disk space:"
df -h . | tail -1
echo ""

echo "8. Checking permissions on checkpoints directory:"
ls -ld checkpoints 2>/dev/null || echo "Directory doesn't exist"
echo ""

echo "9. Checking if training is currently running:"
if pgrep -f train_enhanced_clstm_ppo.py > /dev/null; then
    echo "✅ Training IS currently running"
    ps aux | grep train_enhanced_clstm_ppo.py | grep -v grep
else
    echo "❌ Training is NOT currently running"
fi
echo ""

echo "10. Checking recent log files:"
ls -lht logs/*.log 2>/dev/null | head -3
echo ""

echo "=========================================="
echo "DIAGNOSTIC COMPLETE"
echo "=========================================="

