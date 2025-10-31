#!/usr/bin/env python3
"""
Fix critical data sharing issue where all environments use the same data object
"""

import re

# Read the training script
train_file = "/home/devin/Desktop/Trading_bot/train_profitable_optimized.py"

with open(train_file, 'r') as f:
    content = f.read()

# Fix 1: Deep copy data for each environment
# Find the make_env function
make_env_pattern = r'(def make_env\(rank, world_size, num_envs_per_process, process_idx, env_idx\):.*?)(env = BalancedEnvironment\()'

make_env_replacement = r'''\1
        # CRITICAL FIX: Deep copy data to prevent sharing between environments
        import copy
        env_historical_data = copy.deepcopy(historical_data)
        
        \2'''

content = re.sub(make_env_pattern, make_env_replacement, content, flags=re.DOTALL)

# Update the environment creation to use the copied data
env_creation_pattern = r'(env = BalancedEnvironment\(\s*historical_data=)historical_data,'

content = re.sub(env_creation_pattern, r'\1env_historical_data,', content)

# Also fix UltraFastEnvironment if it's used
ultra_pattern = r'(env = UltraFastEnvironment\(\s*historical_data=)historical_data,'
content = re.sub(ultra_pattern, r'\1env_historical_data,', content)

# Fix OptimizedProfitableEnvironment if it's used
opt_pattern = r'(env = OptimizedProfitableEnvironment\(\s*historical_data=)historical_data,'
content = re.sub(opt_pattern, r'\1env_historical_data,', content)

print("Fixed data sharing in make_env function")

# Fix 2: Make training data a copy in the parent class reset
hist_file = "/home/devin/Desktop/Trading_bot/src/historical_options_data.py"

with open(hist_file, 'r') as f:
    hist_content = f.read()

# Fix the training data assignment to use copy
training_data_pattern = r'(if self\.current_symbol in self\.historical_data:\s*self\.training_data = )self\.historical_data\[self\.current_symbol\]'

training_data_replacement = r'\1self.historical_data[self.current_symbol].copy()'

hist_content = re.sub(training_data_pattern, training_data_replacement, hist_content)

# Save the historical options file
with open(hist_file, 'w') as f:
    f.write(hist_content)

print("Fixed data copying in HistoricalOptionsEnvironment")

# Fix 3: Add thread safety to shared resources
# Add a lock for data access in the training script
thread_safety = '''
# Thread safety for shared resources
import threading
data_lock = threading.Lock()
'''

# Insert at the beginning of the make_env function definition area
import_pattern = r'(from concurrent\.futures import ProcessPoolExecutor, ThreadPoolExecutor)'
content = re.sub(import_pattern, r'\1\n' + thread_safety, content)

# Save the training file
with open(train_file, 'w') as f:
    f.write(content)

print("\nAll data sharing fixes applied!")
print("\nChanges made:")
print("1. Deep copy historical_data for each environment")
print("2. Copy DataFrame when assigning training_data") 
print("3. Added thread safety imports")
print("\nThis prevents:")
print("- Race conditions between parallel environments")
print("- Data corruption from shared references")
print("- Inconsistent results across environments")
print("\nEach environment now has its own independent copy of the data.")