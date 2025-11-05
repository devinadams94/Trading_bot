#!/usr/bin/env python3
"""
Minimal test to see if checkpoints are being saved
"""
import sys
import os
from pathlib import Path

print("=" * 80)
print("MINIMAL TRAINING TEST - CHECKPOINT VERIFICATION")
print("=" * 80)
print()

# Test 1: Check if CLSTM-PPO can be imported
print("1. Testing CLSTM-PPO import:")
try:
    from src.options_clstm_ppo import OptionsCLSTMPPOAgent
    print("   ✅ CLSTM-PPO agent imported successfully")
    HAS_CLSTM_PPO = True
except ImportError as e:
    print(f"   ❌ CLSTM-PPO agent import failed: {e}")
    HAS_CLSTM_PPO = False
print()

# Test 2: Check checkpoint directory
checkpoint_dir = Path("checkpoints/minimal_test")
print(f"2. Creating checkpoint directory: {checkpoint_dir}")
try:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"   ✅ Directory created: {checkpoint_dir.absolute()}")
except Exception as e:
    print(f"   ❌ Failed to create directory: {e}")
    sys.exit(1)
print()

# Test 3: Try to save a training_state.json
print("3. Testing training_state.json save:")
try:
    import json
    checkpoint_path = checkpoint_dir / "training_state.json"
    test_data = {
        "episode": 0,
        "test": "minimal training test",
        "has_clstm_ppo": HAS_CLSTM_PPO
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"   ✅ training_state.json saved to: {checkpoint_path.absolute()}")
    
    # Verify it exists
    if checkpoint_path.exists():
        size = checkpoint_path.stat().st_size
        print(f"   ✅ File exists, size: {size} bytes")
    else:
        print(f"   ❌ File does not exist after saving!")
except Exception as e:
    print(f"   ❌ Failed to save training_state.json: {e}")
    import traceback
    traceback.print_exc()
print()

# Test 4: List files in checkpoint directory
print("4. Listing checkpoint directory contents:")
try:
    files = list(checkpoint_dir.iterdir())
    if files:
        print(f"   ✅ Found {len(files)} file(s):")
        for f in files:
            print(f"      - {f.name} ({f.stat().st_size} bytes)")
    else:
        print(f"   ❌ Directory is empty!")
except Exception as e:
    print(f"   ❌ Failed to list directory: {e}")
print()

# Test 5: Run actual training for 2 episodes
print("5. Running actual training for 2 episodes:")
print("   (This will test if the real training script saves checkpoints)")
print()

try:
    # Import the trainer
    sys.path.insert(0, str(Path(__file__).parent))
    
    # Set minimal config
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Run training
    cmd = [
        sys.executable, '-u', 'train_enhanced_clstm_ppo.py',
        '--quick-test',
        '--num_gpus', '1',
        '--checkpoint-dir', 'checkpoints/minimal_test',
        '--episodes', '2',  # Just 2 episodes
        '--fresh-start'
    ]
    
    print(f"   Running: {' '.join(cmd)}")
    print()
    
    import subprocess
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    
    print("   STDOUT:")
    print("   " + "\n   ".join(result.stdout.split('\n')[-30:]))  # Last 30 lines
    
    if result.returncode != 0:
        print()
        print("   STDERR:")
        print("   " + "\n   ".join(result.stderr.split('\n')[-20:]))  # Last 20 lines
    
    print()
    print(f"   Return code: {result.returncode}")
    
except subprocess.TimeoutExpired:
    print("   ⚠️ Training timed out after 5 minutes")
except Exception as e:
    print(f"   ❌ Failed to run training: {e}")
    import traceback
    traceback.print_exc()
print()

# Test 6: Check if checkpoints were created
print("6. Checking if checkpoints were created:")
try:
    checkpoint_dir = Path("checkpoints/minimal_test")
    if checkpoint_dir.exists():
        files = list(checkpoint_dir.iterdir())
        if files:
            print(f"   ✅ Found {len(files)} file(s) in checkpoint directory:")
            for f in sorted(files):
                size = f.stat().st_size
                print(f"      - {f.name} ({size:,} bytes)")
        else:
            print(f"   ❌ Checkpoint directory exists but is EMPTY")
    else:
        print(f"   ❌ Checkpoint directory does not exist")
except Exception as e:
    print(f"   ❌ Failed to check checkpoints: {e}")
print()

print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)

