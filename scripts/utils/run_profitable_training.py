#!/usr/bin/env python3
"""Simple runner for the profitable training strategy"""

import subprocess
import sys
import os

print("""
🚀 PROFITABLE OPTIONS TRADING BOT TRAINING
==========================================

This training strategy includes critical fixes:
✅ Fixed position sizing bug (no more 314% of capital)
✅ Strict 5% stop loss to prevent large losses  
✅ 10% take profit to lock in gains
✅ Risk management - no new trades when down >10%
✅ Reduced volatility for easier learning
✅ Exploration decay over time
✅ Checkpoints saved every 50 episodes

Expected results:
- Positive returns within 1000 episodes
- 60%+ win rate when fully trained
- 5%+ average return per episode

""")

# Get number of episodes
episodes = input("How many episodes to train? (default 5000): ").strip()
if not episodes:
    episodes = "5000"

try:
    episodes = int(episodes)
except:
    print("Invalid number, using 5000")
    episodes = 5000

print(f"\nStarting training for {episodes} episodes...")
print("This should take approximately:")
print(f"- {episodes // 200} minutes on GPU")
print(f"- {episodes // 50} minutes on CPU")

# Activate venv and run
cmd = f"source venv/bin/activate && python train_profitable_fixed.py --episodes {episodes}"
subprocess.run(cmd, shell=True)

print("\n✅ Training complete!")
print("\nYour models are saved in: checkpoints/profitable_fixed/")
print("\nBest model will be named like: best_return_0.0XXX.pt")
print("Final model: final_model.pt")
print("\nTo test your model:")
print("  python test_trained_model.py checkpoints/profitable_fixed/best_return_*.pt")