#!/usr/bin/env python3
"""
Dataset Enhancement Script
Implements recommended improvements to training data collection

Based on STRIKE_FILTER_ANALYSIS.md recommendations
"""

import sys
import os

def show_current_config():
    """Display current data collection configuration"""
    print("=" * 70)
    print("📊 CURRENT DATA COLLECTION CONFIGURATION")
    print("=" * 70)
    print()
    print("Strike Range:")
    print("  ├─ Current: ±7% (93%-107% of underlying)")
    print("  └─ Captures: ATM and slightly ITM/OTM options")
    print()
    print("Expiration Range:")
    print("  ├─ Current: 7-45 days")
    print("  └─ Captures: Short-term options")
    print()
    print("Symbols:")
    print("  ├─ Current: 22 symbols")
    print("  └─ Sectors: ETFs (3), Tech (15), Financials (4)")
    print()
    print("Historical Period:")
    print("  ├─ Current: 2 years (730 days)")
    print("  └─ Captures: Recent market cycles")
    print()
    print("Data Granularity:")
    print("  ├─ Stocks: 1 Hour bars")
    print("  └─ Options: Daily snapshots")
    print()
    print("Estimated Dataset Size: ~500 MB")
    print()

def show_recommended_config():
    """Display recommended configuration"""
    print("=" * 70)
    print("✨ RECOMMENDED CONFIGURATION (Phase 1)")
    print("=" * 70)
    print()
    print("Strike Range:")
    print("  ├─ Recommended: ±10% (90%-110% of underlying)")
    print("  ├─ Benefit: +40% more strikes, better strategy diversity")
    print("  └─ Impact: Still liquid, reasonable spreads")
    print()
    print("Expiration Range:")
    print("  ├─ Recommended: 7-60 days")
    print("  ├─ Benefit: +33% more expirations")
    print("  └─ Impact: Captures monthly options cycles")
    print()
    print("Symbols:")
    print("  ├─ Keep: 22 symbols (for now)")
    print("  └─ Future: Expand to 30 symbols (Phase 2)")
    print()
    print("Historical Period:")
    print("  ├─ Keep: 2 years")
    print("  └─ Future: Expand to 3 years (Phase 3)")
    print()
    print("Estimated Dataset Size: ~875 MB (+75%)")
    print("Expected Performance Gain: +15-30%")
    print()

def show_implementation_changes():
    """Show what code changes are needed"""
    print("=" * 70)
    print("🔧 REQUIRED CODE CHANGES")
    print("=" * 70)
    print()
    print("File: src/historical_options_data.py")
    print()
    print("Change 1: Expand strike range (Line 788)")
    print("  OLD: if abs(float(c['strike_price']) - stock_price) / stock_price < 0.07")
    print("  NEW: if abs(float(c['strike_price']) - stock_price) / stock_price < 0.10")
    print()
    print("Change 2: Increase contract limit (Line 789)")
    print("  OLD: ][:100]  # Limit to 100 most relevant contracts")
    print("  NEW: ][:150]  # Limit to 150 most relevant contracts")
    print()
    print("Change 3: Expand strike price range (Lines 893-894)")
    print("  OLD: 'strike_price_gte': stock_price * 0.93")
    print("       'strike_price_lte': stock_price * 1.07")
    print("  NEW: 'strike_price_gte': stock_price * 0.90")
    print("       'strike_price_lte': stock_price * 1.10")
    print()
    print("Change 4: Expand expiration range (Line 608)")
    print("  OLD: max_expiry = current_date + timedelta(days=45)")
    print("  NEW: max_expiry = current_date + timedelta(days=60)")
    print()
    print("Change 5: Update filter comment (Line 603)")
    print("  OLD: # Filter by strike price (within ±20% of current stock price)")
    print("  NEW: # Filter by strike price (within ±10% of current stock price)")
    print()

def show_next_steps():
    """Show what to do next"""
    print("=" * 70)
    print("📋 NEXT STEPS")
    print("=" * 70)
    print()
    print("Option 1: Manual Implementation (Recommended)")
    print("  1. Open src/historical_options_data.py")
    print("  2. Make the 5 changes listed above")
    print("  3. Save the file")
    print("  4. Clear cache: rm -rf data/cache/*")
    print("  5. Re-run training: python train_enhanced_clstm_ppo.py")
    print()
    print("Option 2: Automatic Implementation (Use with caution)")
    print("  1. Run: python enhance_dataset.py --apply")
    print("  2. Review changes: git diff src/historical_options_data.py")
    print("  3. Clear cache: rm -rf data/cache/*")
    print("  4. Re-run training: python train_enhanced_clstm_ppo.py")
    print()
    print("Option 3: Test First")
    print("  1. Create backup: cp src/historical_options_data.py src/historical_options_data.py.bak")
    print("  2. Apply changes (Option 1 or 2)")
    print("  3. Test data loading: python -c 'from src.historical_options_data import *; print(\"OK\")'")
    print("  4. If OK, proceed with training")
    print("  5. If error, restore: mv src/historical_options_data.py.bak src/historical_options_data.py")
    print()

def apply_changes():
    """Apply the recommended changes to the code"""
    import fileinput
    
    print("=" * 70)
    print("🔧 APPLYING CHANGES TO src/historical_options_data.py")
    print("=" * 70)
    print()
    
    file_path = "src/historical_options_data.py"
    
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found!")
        return False
    
    # Create backup
    backup_path = f"{file_path}.bak"
    import shutil
    shutil.copy2(file_path, backup_path)
    print(f"✅ Created backup: {backup_path}")
    
    # Read file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    changes_made = 0
    
    # Apply changes
    for i, line in enumerate(lines):
        # Change 1: Line 603 - Update comment
        if i == 602 and "within ±20%" in line:
            lines[i] = line.replace("±20%", "±10%")
            print(f"✅ Change 1: Updated comment on line {i+1}")
            changes_made += 1
        
        # Change 2: Line 604-605 - Update strike range
        if i == 603 and "stock_price * 0.8" in line:
            lines[i] = line.replace("0.8", "0.9")
            print(f"✅ Change 2a: Updated min strike on line {i+1}")
            changes_made += 1
        
        if i == 604 and "stock_price * 1.2" in line:
            lines[i] = line.replace("1.2", "1.1")
            print(f"✅ Change 2b: Updated max strike on line {i+1}")
            changes_made += 1
        
        # Change 3: Line 608 - Update expiration
        if i == 607 and "timedelta(days=45)" in line:
            lines[i] = line.replace("days=45", "days=60")
            print(f"✅ Change 3: Updated expiration range on line {i+1}")
            changes_made += 1
        
        # Change 4: Line 788 - Update moneyness filter
        if i == 787 and "< 0.07" in line and "Within 7%" in line:
            lines[i] = line.replace("< 0.07", "< 0.10").replace("7%", "10%")
            print(f"✅ Change 4: Updated moneyness filter on line {i+1}")
            changes_made += 1
        
        # Change 5: Line 789 - Update contract limit
        if i == 788 and "][:100]" in line:
            lines[i] = line.replace("[:100]", "[:150]").replace("100 most", "150 most")
            print(f"✅ Change 5: Updated contract limit on line {i+1}")
            changes_made += 1
        
        # Change 6: Lines 893-894 - Update API strike range
        if i == 892 and "stock_price * 0.93" in line:
            lines[i] = line.replace("0.93", "0.90").replace("7%", "10%")
            print(f"✅ Change 6a: Updated API min strike on line {i+1}")
            changes_made += 1
        
        if i == 893 and "stock_price * 1.07" in line:
            lines[i] = line.replace("1.07", "1.10").replace("7%", "10%")
            print(f"✅ Change 6b: Updated API max strike on line {i+1}")
            changes_made += 1
    
    # Write changes
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    print()
    print(f"✅ Applied {changes_made} changes to {file_path}")
    print(f"✅ Backup saved to {backup_path}")
    print()
    print("⚠️  IMPORTANT: Clear cache before training!")
    print("   Run: rm -rf data/cache/*")
    print()
    
    return True

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--apply":
        # Apply changes
        show_current_config()
        show_recommended_config()
        print()
        response = input("Apply these changes? (yes/no): ")
        if response.lower() in ['yes', 'y']:
            if apply_changes():
                print("✅ Changes applied successfully!")
                print()
                show_next_steps()
            else:
                print("❌ Failed to apply changes")
        else:
            print("❌ Changes not applied")
    else:
        # Show information only
        show_current_config()
        show_recommended_config()
        show_implementation_changes()
        show_next_steps()

if __name__ == "__main__":
    main()

