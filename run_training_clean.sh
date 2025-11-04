#!/bin/bash

# Clear Python cache to ensure latest code is used
echo "🧹 Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
echo "✅ Cache cleared"

# Run training with unbuffered output
echo "🚀 Starting training..."
python -u train_enhanced_clstm_ppo.py \
    --quick-test \
    --num_gpus 1 \
    --checkpoint-dir checkpoints/test \
    --fresh-start

