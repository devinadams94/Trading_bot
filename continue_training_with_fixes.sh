#!/bin/bash

echo "================================================================================"
echo "🔄 CONTINUE TRAINING WITH UPDATED HYPERPARAMETERS"
echo "================================================================================"
echo ""

echo "📋 This script will:"
echo "  1. Resume from your existing checkpoint (1738 episodes)"
echo "  2. Apply NEW hyperparameters to the loaded model:"
echo "     - Entropy: 0.2 → 0.05 (reduce overtrading)"
echo "     - Learning rates: 1e-3 → 3e-4, 3e-3 → 1e-3 (more stable)"
echo "     - Reward scaling: 1e-4 → 1e-3 (stronger signal)"
echo "     - Trading penalty: -0.02 per trade (new)"
echo "  3. Continue training for additional episodes"
echo ""

echo "================================================================================"
echo "📊 CURRENT STATUS (from your last run):"
echo "================================================================================"
echo "  Total episodes: 1738"
echo "  Avg return: -8.11%"
echo "  Sharpe ratio: -4.27"
echo "  Trades/episode: 193 (TOO HIGH - overtrading)"
echo "  Profitability: 0.0%"
echo ""

echo "================================================================================"
echo "🎯 EXPECTED IMPROVEMENTS:"
echo "================================================================================"
echo "  Trades/episode: 193 → 20-50 (reduced overtrading)"
echo "  Avg return: -8.11% → positive"
echo "  Sharpe ratio: -4.27 → positive"
echo "  Profitability: 0% → 40-60%"
echo ""

echo "================================================================================"
echo "🚀 OPTION 1: Continue WITHOUT transaction costs (recommended first)"
echo "================================================================================"
echo ""
echo "This will help the model learn profitability faster without the burden"
echo "of transaction costs. Once profitable, we can add costs back."
echo ""
echo "Command:"
echo "  python3 train_enhanced_clstm_ppo.py --no-realistic-costs --episodes 3000"
echo ""
read -p "Run Option 1? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting training WITHOUT transaction costs..."
    echo ""
    python3 train_enhanced_clstm_ppo.py --no-realistic-costs --episodes 3000
    exit 0
fi

echo ""
echo "================================================================================"
echo "🚀 OPTION 2: Continue WITH transaction costs"
echo "================================================================================"
echo ""
echo "This is harder but more realistic. The model will need to overcome"
echo "transaction costs to become profitable."
echo ""
echo "Command:"
echo "  python3 train_enhanced_clstm_ppo.py --realistic-costs --episodes 3000"
echo ""
read -p "Run Option 2? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting training WITH transaction costs..."
    echo ""
    python3 train_enhanced_clstm_ppo.py --realistic-costs --episodes 3000
    exit 0
fi

echo ""
echo "No option selected. Exiting."
echo ""
echo "To run manually:"
echo "  python3 train_enhanced_clstm_ppo.py --no-realistic-costs --episodes 3000"
echo "  python3 train_enhanced_clstm_ppo.py --realistic-costs --episodes 3000"
echo ""

