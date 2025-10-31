#!/usr/bin/env python3
"""
Update old checkpoints to be compatible with PyTorch 2.6+ security features
"""

import torch
import os
import glob
import argparse
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_checkpoint(checkpoint_path: str, backup: bool = True):
    """
    Update a checkpoint file to be compatible with PyTorch 2.6+
    
    Args:
        checkpoint_path: Path to checkpoint file
        backup: Whether to create a backup
    """
    try:
        logger.info(f"Updating checkpoint: {checkpoint_path}")
        
        # Create backup if requested
        if backup:
            backup_path = checkpoint_path + ".backup"
            if not os.path.exists(backup_path):
                import shutil
                shutil.copy2(checkpoint_path, backup_path)
                logger.info(f"Created backup: {backup_path}")
        
        # Load checkpoint with legacy mode
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Convert numpy scalars to Python types
        def convert_numpy_scalars(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_scalars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_scalars(v) for v in obj]
            elif isinstance(obj, np.generic):
                return obj.item()
            else:
                return obj
        
        # Convert checkpoint data
        checkpoint = convert_numpy_scalars(checkpoint)
        
        # Save with new format
        torch.save(checkpoint, checkpoint_path)
        
        # Verify it can be loaded with weights_only=True
        try:
            test_load = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            logger.info(f"✅ Successfully updated {checkpoint_path}")
            return True
        except Exception as e:
            logger.warning(f"Updated checkpoint still requires weights_only=False: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to update {checkpoint_path}: {e}")
        return False


def update_all_checkpoints(directory: str, pattern: str = "*.pt", backup: bool = True):
    """
    Update all checkpoints in a directory
    
    Args:
        directory: Directory containing checkpoints
        pattern: File pattern to match
        backup: Whether to create backups
    """
    checkpoint_files = glob.glob(os.path.join(directory, pattern))
    
    if not checkpoint_files:
        logger.warning(f"No checkpoint files found matching {pattern} in {directory}")
        return
    
    logger.info(f"Found {len(checkpoint_files)} checkpoint files to update")
    
    success_count = 0
    for checkpoint_path in checkpoint_files:
        if checkpoint_path.endswith('.backup'):
            continue
        if update_checkpoint(checkpoint_path, backup):
            success_count += 1
    
    logger.info(f"Successfully updated {success_count}/{len(checkpoint_files)} checkpoints")


def main():
    parser = argparse.ArgumentParser(description='Update checkpoints for PyTorch 2.6+ compatibility')
    parser.add_argument('path', nargs='?', default='checkpoints',
                        help='Path to checkpoint file or directory (default: checkpoints)')
    parser.add_argument('--no-backup', action='store_true',
                        help='Do not create backup files')
    parser.add_argument('--pattern', type=str, default='*.pt',
                        help='File pattern for batch update (default: *.pt)')
    
    args = parser.parse_args()
    
    path = args.path
    
    if os.path.isfile(path):
        # Update single file
        update_checkpoint(path, backup=not args.no_backup)
    elif os.path.isdir(path):
        # Update all files in directory
        update_all_checkpoints(path, args.pattern, backup=not args.no_backup)
    else:
        logger.error(f"Path not found: {path}")
        return
    
    print("\n" + "="*60)
    print("Checkpoint update complete!")
    print("="*60)
    print("\nYour checkpoints should now load without warnings in PyTorch 2.6+")
    print("Backup files were created with .backup extension") if not args.no_backup else None


if __name__ == "__main__":
    main()