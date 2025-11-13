#!/bin/bash

echo "================================================================================"
echo "🔄 RESET TRAINING WITH NEW HYPERPARAMETERS"
echo "================================================================================"
echo ""

echo "⚠️  WARNING: This will delete all existing checkpoints and start fresh"
echo "   with the new hyperparameters (reduced entropy, increased reward scaling)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "📁 Backing up old checkpoints..."
BACKUP_DIR="checkpoints_backup_$(date +%Y%m%d_%H%M%S)"
if [ -d "checkpoints" ]; then
    mv checkpoints "$BACKUP_DIR"
    echo "✅ Old checkpoints backed up to: $BACKUP_DIR"
else
    echo "ℹ️  No existing checkpoints found"
fi

echo ""
echo "📁 Creating fresh checkpoint directory..."
mkdir -p checkpoints
echo "✅ Fresh checkpoint directory created"

echo ""
echo "================================================================================"
echo "📋 NEW HYPERPARAMETERS:"
echo "================================================================================"
echo "  Entropy coefficient: 0.05 (was 0.2) - Reduces overtrading"
echo "  Learning rate (AC):  3e-4 (was 1e-3) - More stable learning"
echo "  Learning rate (CLSTM): 1e-3 (was 3e-3) - More stable learning"
echo "  Reward scaling:      1e-3 (was 1e-4) - 10x stronger signal"
echo "  Trading penalty:     -0.02 per trade - Discourages overtrading"
echo ""

echo "================================================================================"
echo "🚀 STARTING TRAINING (Phase 1: No Transaction Costs)"
echo "================================================================================"
echo ""
echo "This will train for 500 episodes WITHOUT transaction costs to validate"
echo "that the model can learn profitably in an easier environment."
echo ""
echo "Expected results:"
echo "  - Trades per episode: 20-50 (down from 193)"
echo "  - Average return: Positive"
echo "  - Profitability rate: >40%"
echo ""
echo "Command: python3 train_enhanced_clstm_ppo.py --no-realistic-costs --episodes 500"
echo ""

python3 train_enhanced_clstm_ppo.py --no-realistic-costs --episodes 500

echo ""
echo "================================================================================"
echo "📊 PHASE 1 COMPLETE"
echo "================================================================================"
echo ""
echo "Check the results above. If the model is profitable, proceed to Phase 2:"
echo ""
echo "Phase 2 (with transaction costs):"
echo "  python3 train_enhanced_clstm_ppo.py --realistic-costs --episodes 2000"
echo ""
echo "Phase 3 (full training):"
echo "  python3 train_enhanced_clstm_ppo.py --realistic-costs --episodes 10000"
echo ""

