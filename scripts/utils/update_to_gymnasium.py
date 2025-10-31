#!/usr/bin/env python3
"""
Script to update the codebase from using OpenAI Gym to Gymnasium.
This script will update all imports and ensure compatibility.
"""

import os
import re
import sys
from pathlib import Path


def update_file_imports(file_path):
    """Update imports in a single file from gym to gymnasium"""
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Track if changes were made
    original_content = content
    
    # Update imports
    replacements = [
        # Basic imports
        (r'import gym\b', 'import gymnasium as gym'),
        (r'from gym import spaces', 'from gymnasium import spaces'),
        (r'from gym import Env', 'from gymnasium import Env'),
        (r'from gym\.spaces import', 'from gymnasium.spaces import'),
        (r'import gym\.spaces', 'import gymnasium.spaces'),
        
        # Update any direct gym references that might remain
        (r'gym\.Env\b', 'gym.Env'),
        (r'gym\.spaces\b', 'gym.spaces'),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    # Special case: if we have "import gymnasium as gym" but also "import gym", remove duplicate
    if 'import gymnasium as gym' in content and re.search(r'^import gym$', content, re.MULTILINE):
        content = re.sub(r'^import gym$', '', content, flags=re.MULTILINE)
    
    # Write back if changes were made
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False


def update_observation_space_compatibility(file_path):
    """Update observation space definitions for gymnasium compatibility"""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Gymnasium prefers np.float32 over np.float
    content = re.sub(r'dtype=np\.float\b', 'dtype=np.float32', content)
    
    # Update Dict space if needed (gymnasium uses different syntax)
    # This is more complex and might need manual review
    
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False


def main():
    """Main function to update all Python files"""
    
    # Get the project root
    project_root = Path(__file__).parent
    
    # Files to update
    files_to_update = [
        'train.py',
        'src/historical_options_data.py',
        'src/options_trading_env.py',
        'train_live_options.py'  # This already uses gymnasium
    ]
    
    print("🔄 Updating codebase from OpenAI Gym to Gymnasium...")
    
    updated_files = []
    
    for file_path in files_to_update:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  Checking {file_path}...")
            
            # Update imports
            if update_file_imports(full_path):
                updated_files.append(file_path)
                print(f"    ✅ Updated imports in {file_path}")
            
            # Update observation spaces
            if update_observation_space_compatibility(full_path):
                print(f"    ✅ Updated observation spaces in {file_path}")
        else:
            print(f"    ⚠️  File not found: {file_path}")
    
    print(f"\n✅ Update complete! Modified {len(updated_files)} files:")
    for file in updated_files:
        print(f"  - {file}")
    
    print("\n📝 Next steps:")
    print("1. Review the changes to ensure correctness")
    print("2. Run tests to verify compatibility")
    print("3. Test loading existing checkpoints")
    
    # Check if gymnasium is installed
    try:
        import gymnasium
        print("\n✅ Gymnasium is already installed")
    except ImportError:
        print("\n⚠️  Gymnasium is not installed. Run: pip install gymnasium")


if __name__ == "__main__":
    main()