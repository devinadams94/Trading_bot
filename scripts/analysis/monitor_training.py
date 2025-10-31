#!/usr/bin/env python3
"""Real-time monitoring of training performance"""

import os
import pandas as pd
import time
import argparse
from datetime import datetime
import numpy as np


def load_latest_performance(checkpoint_dir):
    """Load the latest performance data"""
    csv_path = os.path.join(checkpoint_dir, 'performance_history.csv')
    
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    
    # Try to load from latest checkpoint
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])
    if checkpoint_files:
        import torch
        for checkpoint_file in reversed(checkpoint_files):
            try:
                checkpoint = torch.load(os.path.join(checkpoint_dir, checkpoint_file), map_location='cpu')
                if 'performance_history' in checkpoint:
                    return pd.DataFrame(checkpoint['performance_history'])
            except:
                continue
    
    return None


def calculate_improvement_metrics(df):
    """Calculate various improvement metrics"""
    metrics = {}
    
    if len(df) < 2:
        return metrics
    
    # Win rate improvement over different windows
    for window in [50, 100, 200]:
        if len(df) >= window:
            old_wr = df['win_rate'].iloc[-window]
            new_wr = df['win_rate'].iloc[-1]
            improvement = (new_wr - old_wr) * 100
            metrics[f'wr_improvement_{window}ep'] = improvement
    
    # Return improvement
    if len(df) >= 100:
        old_return = df['avg_return'].iloc[-100]
        new_return = df['avg_return'].iloc[-1]
        metrics['return_improvement_100ep'] = (new_return - old_return) * 100
    
    # Trend analysis (linear regression slope)
    if len(df) >= 50:
        episodes = np.array(df['episode'].iloc[-50:])
        win_rates = np.array(df['win_rate'].iloc[-50:])
        
        # Calculate slope
        n = len(episodes)
        slope = (n * np.sum(episodes * win_rates) - np.sum(episodes) * np.sum(win_rates)) / \
                (n * np.sum(episodes**2) - np.sum(episodes)**2)
        
        metrics['wr_trend_slope'] = slope * 1000  # Scale for readability
    
    return metrics


def display_performance_dashboard(df):
    """Display a text-based performance dashboard"""
    
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print("="*80)
    print("🤖 TRADING BOT PERFORMANCE MONITOR 📊")
    print("="*80)
    print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Episodes: {len(df)}")
    print("="*80)
    
    # Current Performance
    latest = df.iloc[-1]
    print("\n📈 CURRENT PERFORMANCE:")
    print(f"  Episode:        {latest['episode']}")
    print(f"  Win Rate:       {latest['win_rate']*100:.2f}%")
    print(f"  Avg Return:     {latest['avg_return']*100:.2f}%")
    print(f"  Total Trades:   {latest['total_trades']}")
    
    if 'sharpe_ratio' in df.columns:
        print(f"  Sharpe Ratio:   {latest['sharpe_ratio']:.3f}")
    if 'consistency_score' in df.columns:
        print(f"  Consistency:    {latest['consistency_score']:.3f}")
    
    # Moving Averages
    print("\n📊 MOVING AVERAGES:")
    if 'win_rate_ma_50' in df.columns:
        print(f"  Win Rate (50-EP):   {latest['win_rate_ma_50']*100:.2f}%")
    if 'win_rate_ma_200' in df.columns:
        print(f"  Win Rate (200-EP):  {latest['win_rate_ma_200']*100:.2f}%")
    if 'return_ma_50' in df.columns:
        print(f"  Return (50-EP):     {latest['return_ma_50']*100:.2f}%")
    
    # Best Performance
    print("\n🏆 BEST PERFORMANCE:")
    print(f"  Best Win Rate:      {df['win_rate'].max()*100:.2f}% (Episode {df['win_rate'].idxmax()})")
    print(f"  Best Avg Return:    {df['avg_return'].max()*100:.2f}% (Episode {df['avg_return'].idxmax()})")
    
    if len(df) >= 50:
        best_50ep_wr = df['win_rate'].rolling(50).mean().max()
        print(f"  Best 50-EP Win Rate: {best_50ep_wr*100:.2f}%")
    
    # Improvement Metrics
    improvements = calculate_improvement_metrics(df)
    if improvements:
        print("\n📈 IMPROVEMENT TRACKING:")
        
        for key, value in improvements.items():
            if 'wr_improvement' in key:
                window = key.split('_')[2]
                symbol = "↑" if value > 0 else "↓" if value < 0 else "→"
                print(f"  Win Rate Change ({window}): {value:+.2f}% {symbol}")
            elif key == 'return_improvement_100ep':
                symbol = "↑" if value > 0 else "↓" if value < 0 else "→"
                print(f"  Return Change (100ep):    {value:+.2f}% {symbol}")
            elif key == 'wr_trend_slope':
                direction = "Improving" if value > 0 else "Declining" if value < 0 else "Stable"
                print(f"  Win Rate Trend:           {direction} ({value:+.3f}/episode)")
    
    # Recent Performance (last 10 episodes)
    print("\n📉 RECENT EPISODES:")
    print("  Episode  Win Rate  Avg Return  Trades")
    print("  " + "-"*40)
    
    for i in range(max(0, len(df)-10), len(df)):
        row = df.iloc[i]
        print(f"  {row['episode']:7d}  {row['win_rate']*100:7.2f}%  {row['avg_return']*100:9.2f}%  {row['total_trades']:6.0f}")
    
    # Status Assessment
    print("\n🎯 STATUS ASSESSMENT:")
    
    # Check if profitable
    if latest['avg_return'] > 0 and latest['win_rate'] > 0.5:
        print("  ✅ PROFITABLE - Both win rate and returns are positive!")
    elif latest['avg_return'] > 0:
        print("  ⚠️  POSITIVE RETURNS - But win rate needs improvement")
    elif latest['win_rate'] > 0.5:
        print("  ⚠️  GOOD WIN RATE - But returns are negative")
    else:
        print("  ❌ NOT PROFITABLE - Both metrics need improvement")
    
    # Check improvement
    if 'improvement_rate' in df.columns and latest['improvement_rate'] > 0:
        print("  ✅ IMPROVING - Performance is trending upward")
    elif len(df) >= 100 and improvements.get('wr_improvement_100ep', 0) > 5:
        print("  ✅ SIGNIFICANT IMPROVEMENT - +5% win rate over 100 episodes")
    
    # Consistency check
    if 'consistency_score' in df.columns and latest['consistency_score'] > 0.7:
        print("  ✅ CONSISTENT - Low variance in performance")
    
    print("\n" + "="*80)


def monitor_loop(checkpoint_dir, refresh_interval=30):
    """Main monitoring loop"""
    
    print(f"Monitoring performance in: {checkpoint_dir}")
    print(f"Refresh interval: {refresh_interval} seconds")
    print("Press Ctrl+C to stop monitoring")
    
    last_episode = 0
    
    while True:
        try:
            df = load_latest_performance(checkpoint_dir)
            
            if df is not None and not df.empty:
                current_episode = df['episode'].iloc[-1]
                
                # Only update if new data
                if current_episode != last_episode:
                    display_performance_dashboard(df)
                    last_episode = current_episode
                else:
                    # Just update timestamp
                    print(f"\rNo new data. Last check: {datetime.now().strftime('%H:%M:%S')}", end='', flush=True)
            else:
                print(f"\rWaiting for performance data... {datetime.now().strftime('%H:%M:%S')}", end='', flush=True)
            
            time.sleep(refresh_interval)
            
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(refresh_interval)


def main():
    parser = argparse.ArgumentParser(description='Monitor trading bot training performance')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/profitable_fixed',
                       help='Directory containing checkpoints')
    parser.add_argument('--refresh', type=int, default=30,
                       help='Refresh interval in seconds')
    parser.add_argument('--once', action='store_true',
                       help='Display once and exit (no loop)')
    
    args = parser.parse_args()
    
    if args.once:
        df = load_latest_performance(args.checkpoint_dir)
        if df is not None and not df.empty:
            display_performance_dashboard(df)
        else:
            print(f"No performance data found in {args.checkpoint_dir}")
    else:
        monitor_loop(args.checkpoint_dir, args.refresh)


if __name__ == "__main__":
    main()