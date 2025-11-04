#!/bin/bash
#
# Run training with unbuffered output for real-time progress display
#

set -e

echo "🚀 Starting training with unbuffered output..."
echo ""

# Run Python with unbuffered output (-u flag)
# This ensures all print() and logging output appears immediately
python -u train_enhanced_clstm_ppo.py "$@"

