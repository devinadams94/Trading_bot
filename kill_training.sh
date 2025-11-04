#!/bin/bash
# Kill stuck training processes

echo "🔍 Finding training processes..."
ps aux | grep "train_enhanced_clstm_ppo.py" | grep -v grep

echo ""
echo "🛑 Killing training processes..."
pkill -9 -f "train_enhanced_clstm_ppo.py"

echo "✅ Done!"

