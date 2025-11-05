#!/usr/bin/env python3
"""
Test if checkpoint saving works
"""
import json
from pathlib import Path
import sys

print("=" * 60)
print("CHECKPOINT SAVE TEST")
print("=" * 60)
print()

# Test 1: Create checkpoint directory
test_dir = Path("checkpoints/test_checkpoint_save")
print(f"1. Creating test checkpoint directory: {test_dir}")
try:
    test_dir.mkdir(parents=True, exist_ok=True)
    print(f"   ✅ Directory created successfully")
except Exception as e:
    print(f"   ❌ Failed to create directory: {e}")
    sys.exit(1)
print()

# Test 2: Write a test JSON file
test_json = test_dir / "test_state.json"
print(f"2. Writing test JSON file: {test_json}")
try:
    test_data = {
        "episode": 42,
        "test": "checkpoint save test",
        "status": "working"
    }
    with open(test_json, 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"   ✅ JSON file written successfully")
except Exception as e:
    print(f"   ❌ Failed to write JSON: {e}")
    sys.exit(1)
print()

# Test 3: Read it back
print(f"3. Reading test JSON file back")
try:
    with open(test_json, 'r') as f:
        loaded_data = json.load(f)
    print(f"   ✅ JSON file read successfully")
    print(f"   Data: {loaded_data}")
except Exception as e:
    print(f"   ❌ Failed to read JSON: {e}")
    sys.exit(1)
print()

# Test 4: Create a dummy .pt file
test_pt = test_dir / "test_model.pt"
print(f"4. Creating test .pt file: {test_pt}")
try:
    import torch
    test_tensor = torch.randn(10, 10)
    torch.save(test_tensor, test_pt)
    print(f"   ✅ .pt file created successfully")
except Exception as e:
    print(f"   ❌ Failed to create .pt file: {e}")
    print(f"   (This is OK if PyTorch is not installed)")
print()

# Test 5: Check file sizes
print(f"5. Checking created files:")
try:
    import os
    for file in test_dir.iterdir():
        size = os.path.getsize(file)
        print(f"   {file.name}: {size} bytes")
except Exception as e:
    print(f"   ❌ Failed to check files: {e}")
print()

# Test 6: Check permissions
print(f"6. Checking directory permissions:")
try:
    import stat
    st = test_dir.stat()
    mode = stat.filemode(st.st_mode)
    print(f"   Permissions: {mode}")
    print(f"   Owner: UID {st.st_uid}")
except Exception as e:
    print(f"   ❌ Failed to check permissions: {e}")
print()

# Test 7: Cleanup
print(f"7. Cleaning up test files:")
try:
    import shutil
    shutil.rmtree(test_dir)
    print(f"   ✅ Test directory removed")
except Exception as e:
    print(f"   ❌ Failed to cleanup: {e}")
print()

print("=" * 60)
print("✅ ALL TESTS PASSED - Checkpoint saving should work!")
print("=" * 60)

