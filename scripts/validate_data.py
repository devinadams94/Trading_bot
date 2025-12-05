#!/usr/bin/env python3
"""
Data Validation Script for Options Training Data
Validates the processed parquet file before training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def load_data():
    """Load the processed parquet file"""
    data_path = Path('data/flat_files_processed/options_with_greeks_2020-01-02_to_2024-12-31.parquet')
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        sys.exit(1)
    
    print(f"📂 Loading data from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"✅ Loaded {len(df):,} records")
    return df

def check_structure(df):
    """Check data structure and columns"""
    print("\n" + "="*80)
    print("📊 DATA STRUCTURE CHECK")
    print("="*80)
    
    print(f"\n📋 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\n📋 Columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        dtype = df[col].dtype
        print(f"   {i+1:2}. {col:<30} ({dtype})")
    
    print(f"\n📋 Index type: {type(df.index).__name__}")
    if hasattr(df.index, 'min'):
        print(f"   Range: {df.index.min()} to {df.index.max()}")
    
    return True

def check_completeness(df):
    """Check for missing values and data coverage"""
    print("\n" + "="*80)
    print("📊 DATA COMPLETENESS CHECK")
    print("="*80)
    
    # Missing values
    print("\n📋 Missing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    critical_cols = ['underlying_ticker', 'strike', 'expiry', 'option_type', 
                     'bid', 'ask', 'underlying_price', 'delta', 'gamma', 'theta', 'vega', 'iv']
    
    issues = []
    for col in df.columns:
        if missing[col] > 0:
            status = "⚠️ " if col in critical_cols else "   "
            print(f"   {status}{col}: {missing[col]:,} ({missing_pct[col]:.2f}%)")
            if col in critical_cols and missing_pct[col] > 5:
                issues.append(f"{col} has {missing_pct[col]:.1f}% missing")
    
    if not any(missing > 0):
        print("   ✅ No missing values!")
    
    # Symbol coverage
    if 'underlying_ticker' in df.columns:
        print("\n📋 Symbol Coverage:")
        symbol_counts = df['underlying_ticker'].value_counts()
        for sym, count in symbol_counts.items():
            print(f"   {sym}: {count:,} records ({count/len(df)*100:.1f}%)")
    
    # Date coverage
    date_col = None
    for col in ['date', 'timestamp', 'trade_date']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None and isinstance(df.index, pd.DatetimeIndex):
        dates = df.index
        print(f"\n📋 Date Coverage (from index):")
    elif date_col:
        dates = pd.to_datetime(df[date_col])
        print(f"\n📋 Date Coverage ({date_col}):")
    else:
        dates = None
        print("\n⚠️  No date column found")
    
    if dates is not None:
        print(f"   Start: {dates.min()}")
        print(f"   End:   {dates.max()}")
        unique_dates = dates.nunique() if hasattr(dates, 'nunique') else len(set(dates))
        print(f"   Unique dates: {unique_dates}")
    
    return len(issues) == 0, issues

def check_greeks_validity(df):
    """Validate Greeks are within expected bounds"""
    print("\n" + "="*80)
    print("📊 GREEKS VALIDATION")
    print("="*80)
    
    issues = []
    
    # Expected bounds for Greeks
    greek_bounds = {
        'delta': (-1.0, 1.0),      # Calls: 0 to 1, Puts: -1 to 0
        'gamma': (0.0, 10.0),       # Always positive, usually < 1
        'theta': (-100.0, 0.0),     # Usually negative (time decay)
        'vega': (0.0, 100.0),       # Always positive
        'iv': (0.0, 5.0),           # 0% to 500% (5.0 = 500%)
    }
    
    for greek, (min_val, max_val) in greek_bounds.items():
        if greek not in df.columns:
            print(f"\n⚠️  {greek.upper()} not found in data!")
            issues.append(f"{greek} column missing")
            continue
        
        col = df[greek].dropna()
        if len(col) == 0:
            print(f"\n⚠️  {greek.upper()} is all NaN!")
            issues.append(f"{greek} is all NaN")
            continue
        
        print(f"\n📋 {greek.upper()}:")
        print(f"   Min: {col.min():.6f}")
        print(f"   Max: {col.max():.6f}")
        print(f"   Mean: {col.mean():.6f}")
        print(f"   Std: {col.std():.6f}")
        
        # Check bounds
        below_min = (col < min_val).sum()
        above_max = (col > max_val).sum()
        
        if below_min > 0:
            pct = below_min / len(col) * 100
            print(f"   ⚠️  {below_min:,} values below {min_val} ({pct:.2f}%)")
            if pct > 1:
                issues.append(f"{greek} has {pct:.1f}% below minimum")
        
        if above_max > 0:
            pct = above_max / len(col) * 100
            print(f"   ⚠️  {above_max:,} values above {max_val} ({pct:.2f}%)")
            if pct > 1:
                issues.append(f"{greek} has {pct:.1f}% above maximum")
        
        if below_min == 0 and above_max == 0:
            print(f"   ✅ All values within expected bounds [{min_val}, {max_val}]")
    
    return len(issues) == 0, issues

if __name__ == "__main__":
    print("🔍 OPTIONS DATA VALIDATION")
    print("="*80)
    
    df = load_data()
    check_structure(df)
    complete_ok, complete_issues = check_completeness(df)
    greeks_ok, greeks_issues = check_greeks_validity(df)
    
    print("\n" + "="*80)
    print("📊 VALIDATION SUMMARY")
    print("="*80)
    
    all_issues = complete_issues + greeks_issues
    if all_issues:
        print("\n⚠️  Issues Found:")
        for issue in all_issues:
            print(f"   - {issue}")
    else:
        print("\n✅ All checks passed!")

