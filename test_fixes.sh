#!/bin/bash

echo "================================================================================"
echo "🧪 TESTING TRAINING FIXES"
echo "================================================================================"
echo ""

echo "📋 Changes Applied:"
echo "  1. Reduced entropy coefficient: 0.2 → 0.05 (prevent overtrading)"
echo "  2. Reduced learning rates: 1e-3 → 3e-4 (more stable)"
echo "  3. Increased reward scaling: 1e-4 → 1e-3 (stronger signal)"
echo "  4. Added trading penalty: -0.02 per trade (discourage overtrading)"
echo ""

echo "================================================================================"
echo "🚀 TEST 1: Quick test WITHOUT realistic costs (should learn quickly)"
echo "================================================================================"
echo ""
echo "Running: python3 train_enhanced_clstm_ppo.py --no-realistic-costs --quick-test --episodes 100"
echo ""

python3 train_enhanced_clstm_ppo.py --no-realistic-costs --quick-test --episodes 100

echo ""
echo "================================================================================"
echo "📊 TEST 1 COMPLETE"
echo "================================================================================"
echo ""
echo "Expected results:"
echo "  - Trades per episode: 20-50 (down from 191)"
echo "  - Average return: Positive (up from -8.48%)"
echo "  - Profitability rate: >0% (up from 0%)"
echo ""
echo "If Test 1 is successful, run Test 2:"
echo "  python3 train_enhanced_clstm_ppo.py --realistic-costs --quick-test --episodes 500"
echo ""

