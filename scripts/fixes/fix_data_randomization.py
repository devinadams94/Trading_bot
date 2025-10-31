#!/usr/bin/env python3
"""
Fix the issue where environments always start from the same data point,
causing repeated identical trades across episodes.
"""

import re

# Fix 1: Update the parent class reset method in historical_options_data.py
hist_file = "/home/devin/Desktop/Trading_bot/src/historical_options_data.py"

with open(hist_file, 'r') as f:
    hist_content = f.read()

# Find and replace the reset method to randomize starting position
old_reset = """    def reset(self):
        \"\"\"Reset the environment for a new episode\"\"\"
        self.capital = self.initial_capital
        self.positions = []
        self.current_step = 0
        self.current_symbol = np.random.choice(self.symbols)
        self.done = False"""

new_reset = """    def reset(self):
        \"\"\"Reset the environment for a new episode\"\"\"
        self.capital = self.initial_capital
        self.positions = []
        self.current_symbol = np.random.choice(self.symbols)
        self.done = False
        
        # Get training data for current symbol
        if self.current_symbol in self.historical_data:
            self.training_data = self.historical_data[self.current_symbol]
        elif self.data_loader:
            self.training_data = self.data_loader.get_training_data(self.current_symbol)
        else:
            self.training_data = pd.DataFrame()
        
        # Randomize starting position within the data
        if not self.training_data.empty:
            max_start = max(0, len(self.training_data) - self.episode_length - self.lookback_window)
            if max_start > 0:
                self.current_step = np.random.randint(0, max_start)
            else:
                self.current_step = 0
        else:
            self.current_step = 0
            logger.warning(f"No training data available for {self.current_symbol}")
            self.done = True"""

# Apply the fix
hist_content = hist_content.replace(old_reset, new_reset)

# Also need to move the training data loading before using it
# Find the section after setting done = False and before return
pattern = r"(self\.done = False)\s*\n\s*\n\s*# Get training data.*?self\.done = True"
replacement = r"\1"

hist_content = re.sub(pattern, replacement, hist_content, flags=re.DOTALL)

# Save the fixed file
with open(hist_file, 'w') as f:
    f.write(hist_content)

print("Fixed historical_options_data.py - added data randomization")

# Fix 2: Update the child classes in train_profitable_optimized.py to respect random starting position
train_file = "/home/devin/Desktop/Trading_bot/train_profitable_optimized.py"

with open(train_file, 'r') as f:
    train_content = f.read()

# Fix UltraFastEnvironment reset
# Find the reset method and add randomization
ultra_pattern = r'(class UltraFastEnvironment.*?def reset\(self\):.*?"""Fast reset without precomputation""".*?obs = super\(\)\.reset\(\))'

def add_randomization_after_super(match):
    return match.group(0) + """
        
        # Randomize starting position if not already set by parent
        if hasattr(self, 'training_data') and self.training_data is not None:
            if self.current_step == 0:  # Only randomize if parent didn't
                max_start = max(0, len(self.training_data) - 100)  # Leave room for episode
                if max_start > 0:
                    self.current_step = np.random.randint(0, max_start)"""

train_content = re.sub(ultra_pattern, add_randomization_after_super, train_content, flags=re.DOTALL)

# Fix BalancedEnvironment reset
balanced_pattern = r'(class BalancedEnvironment.*?def reset\(self\):.*?"""Fast reset with minimal precomputation""".*?obs = super\(\)\.reset\(\))'

train_content = re.sub(balanced_pattern, add_randomization_after_super, train_content, flags=re.DOTALL)

# Fix FastProfitableEnvironment reset if it exists
fast_pattern = r'(class FastProfitableEnvironment.*?def reset\(self\):.*?obs = super\(\)\.reset\(\))'
if re.search(fast_pattern, train_content, flags=re.DOTALL):
    train_content = re.sub(fast_pattern, add_randomization_after_super, train_content, flags=re.DOTALL)

# Fix OptimizedProfitableEnvironment reset if it exists  
opt_pattern = r'(class OptimizedProfitableEnvironment.*?def reset\(self\):.*?obs = super\(\)\.reset\(\))'
if re.search(opt_pattern, train_content, flags=re.DOTALL):
    train_content = re.sub(opt_pattern, add_randomization_after_super, train_content, flags=re.DOTALL)

# Save the fixed training file
with open(train_file, 'w') as f:
    f.write(train_content)

print("Fixed train_profitable_optimized.py - added data randomization to child classes")

print("\nChanges made:")
print("1. Parent class now randomizes starting position within available data")
print("2. Child classes also randomize if parent didn't")
print("3. Each episode will start from a different point in the historical data")
print("4. This prevents repeating the same trades over and over")
print("\nThe model should now see diverse data across episodes and learn properly.")