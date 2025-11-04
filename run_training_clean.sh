#!/bin/bash

# Clear Python cache to ensure latest code is used
echo "🧹 Clearing Python cache..."
find . -type d -name "__pycache__" -print0 | xargs -0 rm -rf 2>/dev/null
find . -name "*.pyc" -print0 | xargs -0 rm -f 2>/dev/null
find . -name "*.pyo" -print0 | xargs -0 rm -f 2>/dev/null
echo "✅ Cache cleared"

# Verify the fix is in place
echo "🔍 Verifying code fix..."
if grep -q "total_trades = sum(trainer.episode_trades)" train_enhanced_clstm_ppo.py; then
    echo "✅ Code fix verified"
else
    echo "❌ Code fix NOT found - something is wrong!"
    exit 1
fi

# Run training with unbuffered output and PYTHONDONTWRITEBYTECODE to prevent cache
echo "🚀 Starting training..."
PYTHONDONTWRITEBYTECODE=1 python -u train_enhanced_clstm_ppo.py \
    --quick-test \
    --num_gpus 1 \
    --checkpoint-dir checkpoints/test \
    --fresh-start

