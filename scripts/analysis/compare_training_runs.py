#!/usr/bin/env python3
"""Compare performance across different training runs to identify best strategies"""

import os
import torch
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import glob


def load_all_training_runs(base_dir='checkpoints'):
    """Load performance data from all training runs"""
    
    training_runs = []
    
    # Find all checkpoint directories
    for run_dir in os.listdir(base_dir):
        run_path = os.path.join(base_dir, run_dir)
        if not os.path.isdir(run_path):
            continue
            
        # Look for performance history
        csv_path = os.path.join(run_path, 'performance_history.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['run_name'] = run_dir
            training_runs.append(df)
        else:
            # Try to load from checkpoints
            checkpoint_files = glob.glob(os.path.join(run_path, '*.pt'))
            for cp_file in checkpoint_files:
                try:
                    checkpoint = torch.load(cp_file, map_location='cpu')
                    if 'performance_history' in checkpoint:
                        df = pd.DataFrame(checkpoint['performance_history'])
                        df['run_name'] = run_dir
                        training_runs.append(df)
                        break
                except:
                    continue
    
    return training_runs


def analyze_training_run(df, run_name):
    """Analyze a single training run and return key metrics"""
    
    metrics = {
        'run_name': run_name,
        'total_episodes': len(df),
        'final_win_rate': df['win_rate'].iloc[-1] * 100 if len(df) > 0 else 0,
        'final_avg_return': df['avg_return'].iloc[-1] * 100 if len(df) > 0 else 0,
        'best_win_rate': df['win_rate'].max() * 100,
        'best_avg_return': df['avg_return'].max() * 100,
        'episodes_to_profitable': None,
        'episodes_to_50_wr': None,
        'final_sharpe': 0,
        'final_consistency': 0,
        'improvement_rate': 0,
        'stability_score': 0
    }
    
    # Find when it became profitable
    profitable_mask = (df['avg_return'] > 0) & (df['win_rate'] > 0.5)
    if profitable_mask.any():
        metrics['episodes_to_profitable'] = df[profitable_mask]['episode'].iloc[0]
    
    # Find when it reached 50% win rate
    wr_50_mask = df['win_rate'] >= 0.5
    if wr_50_mask.any():
        metrics['episodes_to_50_wr'] = df[wr_50_mask]['episode'].iloc[0]
    
    # Latest metrics
    if 'sharpe_ratio' in df.columns and len(df) > 0:
        metrics['final_sharpe'] = df['sharpe_ratio'].iloc[-1]
    
    if 'consistency_score' in df.columns and len(df) > 0:
        metrics['final_consistency'] = df['consistency_score'].iloc[-1]
    
    # Calculate improvement rate over last 25% of training
    if len(df) >= 100:
        cutoff = int(len(df) * 0.75)
        early_wr = df['win_rate'].iloc[cutoff:cutoff+50].mean()
        late_wr = df['win_rate'].iloc[-50:].mean()
        metrics['improvement_rate'] = (late_wr - early_wr) * 100
    
    # Calculate stability (lower std in last 100 episodes)
    if len(df) >= 100:
        recent_returns = df['avg_return'].iloc[-100:]
        metrics['stability_score'] = 1 / (1 + recent_returns.std())
    
    # Moving average performance
    if 'win_rate_ma_50' in df.columns and len(df) > 0:
        metrics['final_wr_ma50'] = df['win_rate_ma_50'].iloc[-1] * 100
    
    if 'win_rate_ma_200' in df.columns and len(df) > 0:
        metrics['final_wr_ma200'] = df['win_rate_ma_200'].iloc[-1] * 100
    
    return metrics


def create_comparison_report(training_runs):
    """Create a comprehensive comparison report"""
    
    all_metrics = []
    
    for df in training_runs:
        run_name = df['run_name'].iloc[0]
        metrics = analyze_training_run(df, run_name)
        all_metrics.append(metrics)
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(all_metrics)
    
    # Sort by final win rate
    comparison_df = comparison_df.sort_values('final_win_rate', ascending=False)
    
    print("\n" + "="*100)
    print("TRAINING RUNS COMPARISON REPORT")
    print("="*100)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runs analyzed: {len(comparison_df)}")
    print("="*100)
    
    # Summary table
    print("\n📊 PERFORMANCE SUMMARY:")
    print("-"*100)
    
    summary_cols = ['run_name', 'total_episodes', 'final_win_rate', 'final_avg_return', 
                   'best_win_rate', 'best_avg_return', 'final_sharpe']
    
    print(comparison_df[summary_cols].to_string(index=False, float_format='%.2f'))
    
    # Best performers
    print("\n🏆 TOP PERFORMERS:")
    print("-"*100)
    
    # Best final win rate
    best_wr = comparison_df.iloc[0]
    print(f"\nHighest Final Win Rate: {best_wr['run_name']}")
    print(f"  Win Rate: {best_wr['final_win_rate']:.2f}%")
    print(f"  Return: {best_wr['final_avg_return']:.2f}%")
    
    # Best final return
    best_return_idx = comparison_df['final_avg_return'].idxmax()
    best_return = comparison_df.loc[best_return_idx]
    print(f"\nHighest Final Return: {best_return['run_name']}")
    print(f"  Win Rate: {best_return['final_win_rate']:.2f}%")
    print(f"  Return: {best_return['final_avg_return']:.2f}%")
    
    # Most improved
    if 'improvement_rate' in comparison_df.columns:
        best_improved_idx = comparison_df['improvement_rate'].idxmax()
        best_improved = comparison_df.loc[best_improved_idx]
        print(f"\nMost Improved: {best_improved['run_name']}")
        print(f"  Improvement: {best_improved['improvement_rate']:+.2f}% win rate")
    
    # Fastest to profitable
    profitable_runs = comparison_df[comparison_df['episodes_to_profitable'].notna()]
    if not profitable_runs.empty:
        fastest_idx = profitable_runs['episodes_to_profitable'].idxmin()
        fastest = profitable_runs.loc[fastest_idx]
        print(f"\nFastest to Profitable: {fastest['run_name']}")
        print(f"  Episodes: {int(fastest['episodes_to_profitable'])}")
    
    # Training efficiency
    print("\n⚡ TRAINING EFFICIENCY:")
    print("-"*100)
    
    efficiency_cols = ['run_name', 'episodes_to_50_wr', 'episodes_to_profitable', 'improvement_rate']
    efficiency_df = comparison_df[efficiency_cols].copy()
    efficiency_df = efficiency_df.sort_values('episodes_to_50_wr')
    
    print(efficiency_df.to_string(index=False, float_format='%.2f', na_rep='Never'))
    
    # Stability analysis
    print("\n🎯 STABILITY & CONSISTENCY:")
    print("-"*100)
    
    stability_cols = ['run_name', 'final_consistency', 'stability_score', 'final_sharpe']
    stability_df = comparison_df[stability_cols].copy()
    stability_df = stability_df.sort_values('stability_score', ascending=False)
    
    print(stability_df.head(10).to_string(index=False, float_format='%.3f'))
    
    # Statistical summary
    print("\n📈 STATISTICAL SUMMARY:")
    print("-"*100)
    
    stats = {
        'Average Final Win Rate': comparison_df['final_win_rate'].mean(),
        'Average Final Return': comparison_df['final_avg_return'].mean(),
        'Std Dev Win Rate': comparison_df['final_win_rate'].std(),
        'Std Dev Return': comparison_df['final_avg_return'].std(),
        'Profitable Runs': (comparison_df['final_avg_return'] > 0).sum(),
        'Success Rate': (comparison_df['final_avg_return'] > 0).mean() * 100
    }
    
    for key, value in stats.items():
        print(f"{key:25s}: {value:8.2f}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("-"*100)
    
    # Find runs that are both high win rate and positive return
    good_runs = comparison_df[
        (comparison_df['final_win_rate'] > 50) & 
        (comparison_df['final_avg_return'] > 0)
    ]
    
    if not good_runs.empty:
        print(f"\n✅ {len(good_runs)} runs achieved both >50% win rate and positive returns:")
        for _, run in good_runs.iterrows():
            print(f"   - {run['run_name']}: {run['final_win_rate']:.1f}% WR, {run['final_avg_return']:.1f}% Return")
    
    # Identify patterns in successful runs
    if len(good_runs) >= 3:
        print("\n📊 Common patterns in successful runs:")
        print(f"   - Average episodes to 50% WR: {good_runs['episodes_to_50_wr'].mean():.0f}")
        print(f"   - Average final Sharpe ratio: {good_runs['final_sharpe'].mean():.2f}")
        print(f"   - Average consistency score: {good_runs['final_consistency'].mean():.2f}")
    
    return comparison_df


def export_comparison_data(comparison_df, output_dir='analysis'):
    """Export comparison data for further analysis"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full comparison
    csv_path = os.path.join(output_dir, f'training_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    comparison_df.to_csv(csv_path, index=False)
    print(f"\n📁 Comparison data saved to: {csv_path}")
    
    # Save summary report
    report_path = os.path.join(output_dir, f'training_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    with open(report_path, 'w') as f:
        f.write("TRAINING RUNS COMPARISON REPORT\n")
        f.write("="*100 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total runs analyzed: {len(comparison_df)}\n")
        f.write("="*100 + "\n\n")
        
        f.write(comparison_df.to_string(index=False))
    
    print(f"📄 Full report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare different training runs')
    parser.add_argument('--checkpoint-base', type=str, default='checkpoints',
                       help='Base directory containing all checkpoint folders')
    parser.add_argument('--export', action='store_true',
                       help='Export comparison data to files')
    parser.add_argument('--min-episodes', type=int, default=100,
                       help='Minimum episodes for a run to be included')
    
    args = parser.parse_args()
    
    # Load all training runs
    print(f"Loading training runs from: {args.checkpoint_base}")
    training_runs = load_all_training_runs(args.checkpoint_base)
    
    if not training_runs:
        print("No training runs found!")
        return
    
    print(f"Found {len(training_runs)} training runs")
    
    # Filter by minimum episodes
    training_runs = [df for df in training_runs if len(df) >= args.min_episodes]
    print(f"Analyzing {len(training_runs)} runs with >= {args.min_episodes} episodes")
    
    # Create comparison report
    comparison_df = create_comparison_report(training_runs)
    
    # Export if requested
    if args.export:
        export_comparison_data(comparison_df)


if __name__ == "__main__":
    main()