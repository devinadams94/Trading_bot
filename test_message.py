#!/usr/bin/env python3
"""
Test the final message logic
"""

class MockTrainer:
    def __init__(self, episode_trades):
        self.episode_trades = episode_trades

# Test 1: No trades
print("Test 1: No trades")
trainer = MockTrainer([0, 0, 0])
success = False
total_trades = sum(trainer.episode_trades) if trainer.episode_trades else 0
if total_trades > 0:
    print(f"⚠️ Training completed with {total_trades} total trades but not yet profitable")
else:
    print("❌ Training completed but no trades produced")
print()

# Test 2: Has trades but not profitable
print("Test 2: Has trades (19424) but not profitable")
trainer = MockTrainer([194] * 100)  # 194 trades per episode, 100 episodes
success = False
total_trades = sum(trainer.episode_trades) if trainer.episode_trades else 0
if total_trades > 0:
    print(f"⚠️ Training completed with {total_trades} total trades but not yet profitable")
    print("   Continue training or adjust hyperparameters for better performance")
else:
    print("❌ Training completed but no trades produced")
    print("   Check exploration settings and reward function")
print()

# Test 3: Success
print("Test 3: Success (profitable)")
trainer = MockTrainer([194] * 100)
success = True
if success:
    print("🎉 ENHANCED CLSTM-PPO TRAINING SUCCESS!")
else:
    total_trades = sum(trainer.episode_trades) if trainer.episode_trades else 0
    if total_trades > 0:
        print(f"⚠️ Training completed with {total_trades} total trades but not yet profitable")
    else:
        print("❌ Training completed but no trades produced")

