# PyTorch 2.6 Checkpoint Loading Fix

## Issue
PyTorch 2.6 changed the default behavior of `torch.load()` to use `weights_only=True` for security reasons. This causes loading errors with checkpoints that contain NumPy scalars or other non-standard Python types.

## Quick Fix Applied
The training scripts have been updated to handle this automatically:
1. First tries to load with `weights_only=True` (secure mode)
2. Falls back to `weights_only=False` if needed (with a warning)

## Warning Message
You may see this warning when loading old checkpoints:
```
WARNING: Secure loading failed, using legacy mode: Weights only load failed...
```

This is **normal and safe** for checkpoints you created yourself.

## To Remove Warnings

### Option 1: Update Existing Checkpoints
```bash
# Update all checkpoints in a directory
python update_checkpoints.py checkpoints/ppo_lstm

# Update a specific checkpoint
python update_checkpoints.py checkpoints/ppo_lstm/best_model.pt

# Update without creating backups
python update_checkpoints.py checkpoints/ppo_lstm --no-backup
```

### Option 2: Start Fresh
Simply delete old checkpoints and train from scratch. New checkpoints will be saved in a compatible format.

### Option 3: Ignore Warnings
The warnings are harmless for your own checkpoints. Training will continue normally.

## Technical Details
The issue occurs because:
- Old checkpoints contain NumPy scalar types
- PyTorch 2.6 considers these "unsafe" by default
- Our fix automatically handles both old and new formats

## No Action Required
The scripts will continue to work with both old and new checkpoints. The warning is just informational.