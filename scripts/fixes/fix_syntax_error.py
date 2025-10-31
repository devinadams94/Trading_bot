#!/usr/bin/env python3
"""
Fix the syntax error in the f-string
"""

import re

# Read the training script
train_file = "/home/devin/Desktop/Trading_bot/train_profitable_optimized.py"

with open(train_file, 'r') as f:
    content = f.read()

print("Fixing syntax error in f-string...")

# Fix the broken f-string
broken_pattern = r'logger\.info\(f"\n⚠️  PERFORMANCE DECLINE DETECTED!"\)'
fixed_pattern = 'logger.info(f"\\n⚠️  PERFORMANCE DECLINE DETECTED!")'

content = content.replace(broken_pattern, fixed_pattern)

# Alternative approach if the above doesn't work - look for the specific line
if 'logger.info(f"' in content and '\n⚠️  PERFORMANCE DECLINE DETECTED!")' in content:
    # Find and fix the specific problematic section
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'logger.info(f"' in line and len(line.strip()) < 20 and i < len(lines) - 1:
            # Check if this is the broken line
            if '⚠️  PERFORMANCE DECLINE DETECTED!' in lines[i+1]:
                # Fix it by combining the lines
                lines[i] = '                        logger.info(f"\\n⚠️  PERFORMANCE DECLINE DETECTED!")'
                lines[i+1] = ''  # Remove the continuation
                break
    
    content = '\n'.join(line for line in lines if line is not None)

# Save the fixed file
with open(train_file, 'w') as f:
    f.write(content)

print("✅ Syntax error fixed!")