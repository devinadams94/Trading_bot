#!/usr/bin/env python3
"""
Deep Greeks Validation - Check mathematical consistency
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from pathlib import Path

def load_data():
    data_path = Path('data/flat_files_processed/options_with_greeks_2020-01-02_to_2024-12-31.parquet')
    return pd.read_parquet(data_path)

def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculate Black-Scholes Greeks for comparison"""
    if T <= 0 or sigma <= 0:
        return {'delta': np.nan, 'gamma': np.nan, 'theta': np.nan, 'vega': np.nan}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:  # put
        delta = norm.cdf(d1) - 1
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # per 1% change in IV
    
    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}

def validate_greeks_relationships(df):
    """Check mathematical relationships between Greeks"""
    print("\n" + "="*80)
    print("📊 GREEKS RELATIONSHIP VALIDATION")
    print("="*80)
    
    # 1. Check Call vs Put Delta relationship
    print("\n📋 Call/Put Delta Analysis:")
    calls = df[df['option_type'] == 'call']
    puts = df[df['option_type'] == 'put']
    print(f"   Calls: {len(calls):,} ({len(calls)/len(df)*100:.1f}%)")
    print(f"   Puts:  {len(puts):,} ({len(puts)/len(df)*100:.1f}%)")
    
    # Call delta should be 0 to 1, Put delta should be -1 to 0
    call_delta_valid = ((calls['delta'] >= 0) & (calls['delta'] <= 1)).mean() * 100
    put_delta_valid = ((puts['delta'] >= -1) & (puts['delta'] <= 0)).mean() * 100
    print(f"   Call delta in [0, 1]: {call_delta_valid:.2f}%")
    print(f"   Put delta in [-1, 0]: {put_delta_valid:.2f}%")
    
    # 2. Check Gamma (always positive for both calls and puts)
    print("\n📋 Gamma Analysis:")
    gamma_positive = (df['gamma'] >= 0).mean() * 100
    print(f"   Gamma >= 0: {gamma_positive:.2f}%")
    
    # 3. Check Theta (usually negative - time decay)
    print("\n📋 Theta Analysis:")
    theta_negative = (df['theta'] <= 0).mean() * 100
    theta_positive = (df['theta'] > 0).sum()
    print(f"   Theta <= 0: {theta_negative:.2f}%")
    print(f"   Theta > 0 (unusual): {theta_positive:,} records")
    
    # Look at positive theta cases
    if theta_positive > 0:
        pos_theta = df[df['theta'] > 0]
        print(f"\n   Investigating positive theta cases:")
        print(f"   - Deep ITM puts: {((pos_theta['option_type'] == 'put') & (pos_theta['moneyness'] < 0.9)).sum():,}")
        print(f"   - Near expiry: {(pos_theta['days_to_expiry'] <= 3).sum():,}")
        print(f"   - Max positive theta: {pos_theta['theta'].max():.4f}")
    
    # 4. Check Vega (always positive)
    print("\n📋 Vega Analysis:")
    vega_positive = (df['vega'] >= 0).mean() * 100
    print(f"   Vega >= 0: {vega_positive:.2f}%")
    
    # 5. ATM options should have highest gamma
    print("\n📋 ATM Gamma Check (moneyness 0.97-1.03):")
    atm = df[(df['moneyness'] >= 0.97) & (df['moneyness'] <= 1.03)]
    otm_itm = df[(df['moneyness'] < 0.97) | (df['moneyness'] > 1.03)]
    print(f"   ATM mean gamma: {atm['gamma'].mean():.6f}")
    print(f"   OTM/ITM mean gamma: {otm_itm['gamma'].mean():.6f}")
    print(f"   Ratio (ATM should be higher): {atm['gamma'].mean() / otm_itm['gamma'].mean():.2f}x")
    
    return True

def spot_check_options(df, n_samples=5):
    """Spot check a few options against Black-Scholes"""
    print("\n" + "="*80)
    print("📊 SPOT CHECK: Comparing to Black-Scholes")
    print("="*80)
    
    r = 0.05  # risk-free rate assumption
    
    # Sample some ATM options for comparison
    atm = df[(df['moneyness'] >= 0.98) & (df['moneyness'] <= 1.02) & 
             (df['days_to_expiry'] >= 7) & (df['days_to_expiry'] <= 30)]
    
    if len(atm) == 0:
        print("No suitable ATM options found for spot check")
        return
    
    samples = atm.sample(min(n_samples, len(atm)))
    
    for idx, row in samples.iterrows():
        print(f"\n{'='*60}")
        print(f"Option: {row['ticker']}")
        print(f"  Type: {row['option_type']}, Strike: ${row['strike']:.2f}")
        print(f"  Underlying: ${row['underlying_price']:.2f}, DTE: {row['days_to_expiry']}")
        print(f"  IV: {row['iv']*100:.1f}%")
        
        # Calculate BS Greeks
        bs = black_scholes_greeks(
            S=row['underlying_price'],
            K=row['strike'],
            T=row['time_to_expiry'],
            r=r,
            sigma=row['iv'],
            option_type=row['option_type']
        )
        
        print(f"\n  {'Greek':<8} {'Data':<12} {'B-S Calc':<12} {'Diff %':<10}")
        print(f"  {'-'*42}")
        for g in ['delta', 'gamma', 'theta', 'vega']:
            data_val = row[g]
            bs_val = bs[g]
            if not np.isnan(bs_val) and abs(bs_val) > 0.0001:
                diff_pct = abs(data_val - bs_val) / abs(bs_val) * 100
                status = "✅" if diff_pct < 20 else "⚠️"
            else:
                diff_pct = 0
                status = "➖"
            print(f"  {g:<8} {data_val:<12.6f} {bs_val:<12.6f} {diff_pct:>6.1f}% {status}")

def check_iv_sanity(df):
    """Check implied volatility makes sense"""
    print("\n" + "="*80)
    print("📊 IMPLIED VOLATILITY SANITY CHECK")
    print("="*80)
    
    print(f"\n📋 IV Distribution:")
    print(f"   Min: {df['iv'].min()*100:.1f}%")
    print(f"   25th percentile: {df['iv'].quantile(0.25)*100:.1f}%")
    print(f"   Median: {df['iv'].median()*100:.1f}%")
    print(f"   75th percentile: {df['iv'].quantile(0.75)*100:.1f}%")
    print(f"   Max: {df['iv'].max()*100:.1f}%")
    
    # Check IV by symbol
    print(f"\n📋 IV by Symbol:")
    for sym in df['underlying'].unique():
        sym_df = df[df['underlying'] == sym]
        print(f"   {sym}: median IV = {sym_df['iv'].median()*100:.1f}%, "
              f"range [{sym_df['iv'].min()*100:.1f}% - {sym_df['iv'].max()*100:.1f}%]")

if __name__ == "__main__":
    print("🔍 DEEP GREEKS VALIDATION")
    print("="*80)
    
    df = load_data()
    print(f"✅ Loaded {len(df):,} records")
    
    validate_greeks_relationships(df)
    check_iv_sanity(df)
    spot_check_options(df, n_samples=5)
    
    print("\n" + "="*80)
    print("✅ DEEP VALIDATION COMPLETE")
    print("="*80)

