#!/bin/bash

# Script to disable real options data fetching and use simulated data instead
# This makes training 100x faster!

echo "=========================================="
echo "🔧 Disabling Real Options Data Fetching"
echo "=========================================="
echo ""
echo "Why? Real options data takes 12+ hours to fetch."
echo "Simulated data takes 1-2 minutes and works great!"
echo ""

# Backup the original file
echo "📋 Creating backup..."
cp src/historical_options_data.py src/historical_options_data.py.backup
echo "✅ Backup created: src/historical_options_data.py.backup"
echo ""

# Modify the file to skip real options data
echo "🔧 Modifying src/historical_options_data.py..."

# Use sed to change the condition
sed -i 's/if self\.has_options_data and sym in stock_data:/if False:  # SKIP REAL OPTIONS DATA - TOO SLOW! Original: if self.has_options_data and sym in stock_data:/' src/historical_options_data.py

if [ $? -eq 0 ]; then
    echo "✅ Successfully modified the file!"
    echo ""
    echo "=========================================="
    echo "✅ DONE!"
    echo "=========================================="
    echo ""
    echo "Real options data fetching is now DISABLED."
    echo "Training will use fast simulated data instead."
    echo ""
    echo "Expected time:"
    echo "  - Before: 12+ hours"
    echo "  - After:  1-2 minutes"
    echo ""
    echo "To restore original behavior:"
    echo "  cp src/historical_options_data.py.backup src/historical_options_data.py"
    echo ""
    echo "Now run training:"
    echo "  python -u train_enhanced_clstm_ppo.py --quick-test --num_gpus 1 --checkpoint-dir checkpoints/test --fresh-start"
    echo ""
else
    echo "❌ Error modifying the file!"
    echo "Please manually edit src/historical_options_data.py"
    echo "Find line ~520 and change:"
    echo "  if self.has_options_data and sym in stock_data:"
    echo "To:"
    echo "  if False:  # SKIP REAL OPTIONS DATA"
    exit 1
fi

