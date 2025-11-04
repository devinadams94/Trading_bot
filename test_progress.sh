#!/bin/bash
# Test script to verify progress indicators are working

echo "🧪 Testing progress indicators with quick-test mode..."
echo ""

# Kill any existing training processes
pkill -9 -f train_enhanced_clstm_ppo.py 2>/dev/null

# Run quick test
python train_enhanced_clstm_ppo.py \
    --quick-test \
    --num_gpus 1 \
    --checkpoint-dir checkpoints/test \
    --fresh-start

echo ""
echo "✅ Test complete!"

