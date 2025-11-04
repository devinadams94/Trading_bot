#!/bin/bash

# Clear Python cache
echo "🧹 Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
echo "✅ Cache cleared"

# Run training with verbose logging and only 5 episodes for quick test
echo "🚀 Starting training with verbose logging (5 episodes)..."
python -u train_enhanced_clstm_ppo.py \
    --quick-test \
    --num_gpus 1 \
    --checkpoint-dir checkpoints/test_verbose \
    --fresh-start \
    --episodes 5 \
    2>&1 | tee training_verbose.log

echo ""
echo "📊 Checking results..."
grep -E "(Episode [0-9]+:|trades=|unique actions)" training_verbose.log | tail -20

