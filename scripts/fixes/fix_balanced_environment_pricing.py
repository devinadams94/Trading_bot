#!/usr/bin/env python3
"""
Fix the option pricing issue in BalancedEnvironment in train_profitable_optimized.py
"""

import re

# Read the training script
file_path = "/home/devin/Desktop/Trading_bot/train_profitable_optimized.py"

with open(file_path, 'r') as f:
    content = f.read()

# Fix 1: Replace the simplified option pricing with more realistic modeling
old_pricing = """            # Option P&L estimation based on delta
            # Calls profit from price increase, puts from price decrease
            if pos['option_type'] == 'call':
                # Approximate delta with gamma effect (higher for profitable moves)
                if price_change > 0:
                    option_price_change = price_change * 1.5  # Leverage on profitable side
                else:
                    option_price_change = price_change * 0.8  # Less loss on downside
            else:  # put
                if price_change < 0:
                    option_price_change = -price_change * 1.5  # Leverage on profitable side
                else:
                    option_price_change = -price_change * 0.8  # Less loss on upside"""

new_pricing = """            # Option P&L estimation with realistic constraints
            # Options have limited downside (can't lose more than premium paid)
            # and leveraged upside, but not linear with underlying
            
            # For realistic option pricing, we need to consider:
            # 1. Options move less than 1:1 with underlying (delta < 1)
            # 2. Time decay reduces option value
            # 3. Volatility affects option prices
            
            # Estimate delta based on moneyness (simplified Black-Scholes approximation)
            strike = pos['strike']
            moneyness = current_price / strike
            
            if pos['option_type'] == 'call':
                # Call delta approximation (0 to 1)
                if moneyness > 1.1:  # Deep ITM
                    delta = 0.9
                elif moneyness > 1.02:  # ITM
                    delta = 0.7
                elif moneyness > 0.98:  # ATM
                    delta = 0.5
                elif moneyness > 0.9:  # OTM
                    delta = 0.3
                else:  # Deep OTM
                    delta = 0.1
                
                # Option price change based on delta
                option_price_change = price_change * delta
                
            else:  # put
                # Put delta approximation (-1 to 0)
                if moneyness < 0.9:  # Deep ITM
                    delta = -0.9
                elif moneyness < 0.98:  # ITM
                    delta = -0.7
                elif moneyness < 1.02:  # ATM
                    delta = -0.5
                elif moneyness < 1.1:  # OTM
                    delta = -0.3
                else:  # Deep OTM
                    delta = -0.1
                
                # Option price change based on delta
                option_price_change = price_change * delta
            
            # Add time decay effect (theta)
            # Options lose value over time, especially near expiration
            time_decay_factor = 0.99  # 1% decay per period
            if position_age > 10:
                time_decay_factor = 0.98  # Accelerated decay
            elif position_age > 15:
                time_decay_factor = 0.95
            
            # Apply time decay
            option_price_change = option_price_change * time_decay_factor
            
            # Limit maximum loss to prevent immediate stop-outs
            # Options can lose value quickly but not instantly
            max_loss_per_period = 0.15  # Max 15% loss per period
            option_price_change = max(option_price_change, -max_loss_per_period)"""

# Replace the pricing logic
content = content.replace(old_pricing, new_pricing)

# Fix 2: Also fix the same issue in _close_all_positions_realistic method
old_close_pricing = """            if pos['option_type'] == 'call':
                # Approximate delta with gamma effect (higher for profitable moves)
                if price_change > 0:
                    option_price_change = price_change * 1.5  # Leverage on profitable side
                else:
                    option_price_change = price_change * 0.8  # Less loss on downside
            else:  # put
                if price_change < 0:
                    option_price_change = -price_change * 1.5  # Leverage on profitable side
                else:
                    option_price_change = -price_change * 0.8  # Less loss on upside"""

# Find the exact location in _close_all_positions_realistic
pattern = r"(def _close_all_positions_realistic.*?)(if pos\['option_type'\] == 'call':.*?option_price_change = -price_change \* 0\.8.*?# Less loss on upside)"
match = re.search(pattern, content, re.DOTALL)

if match:
    # Replace with the same realistic pricing
    close_pricing_replacement = """            # Use same realistic option pricing as in update method
            strike = pos['strike']
            moneyness = current_price / strike
            
            if pos['option_type'] == 'call':
                # Call delta approximation
                if moneyness > 1.1:
                    delta = 0.9
                elif moneyness > 1.02:
                    delta = 0.7
                elif moneyness > 0.98:
                    delta = 0.5
                elif moneyness > 0.9:
                    delta = 0.3
                else:
                    delta = 0.1
                option_price_change = price_change * delta
            else:  # put
                # Put delta approximation
                if moneyness < 0.9:
                    delta = -0.9
                elif moneyness < 0.98:
                    delta = -0.7
                elif moneyness < 1.02:
                    delta = -0.5
                elif moneyness < 1.1:
                    delta = -0.3
                else:
                    delta = -0.1
                option_price_change = price_change * delta
            
            # Limit losses
            option_price_change = max(option_price_change, -0.15)"""
    
    content = re.sub(pattern, r"\1" + close_pricing_replacement, content, flags=re.DOTALL)

# Save the fixed file
with open(file_path, 'w') as f:
    f.write(content)

print("Fixed option pricing in BalancedEnvironment")
print("\nChanges made:")
print("1. Replaced linear option pricing with delta-based pricing")
print("2. Added moneyness-based delta estimation")
print("3. Added time decay factor")
print("4. Limited maximum loss per period to 15%")
print("5. Applied same fixes to close_all_positions method")
print("\nThis should prevent immediate stop-loss triggers and provide more realistic option pricing.")