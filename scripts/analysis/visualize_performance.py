#!/usr/bin/env python3
"""Visualize training performance metrics to track improvement over time"""

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import argparse
import seaborn as sns

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10


def load_performance_data(checkpoint_path):
    """Load performance data from checkpoint or CSV"""
    if checkpoint_path.endswith('.csv'):
        # Load from CSV
        return pd.read_csv(checkpoint_path)
    elif checkpoint_path.endswith('.pt'):
        # Load from checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'performance_history' in checkpoint:
            return pd.DataFrame(checkpoint['performance_history'])
        else:
            print(f"No performance history found in {checkpoint_path}")
            return None
    else:
        # Try to find performance history CSV in directory
        if os.path.isdir(checkpoint_path):
            csv_path = os.path.join(checkpoint_path, 'performance_history.csv')
            if os.path.exists(csv_path):
                return pd.read_csv(csv_path)
        return None


def create_performance_plots(df, save_dir=None):
    """Create comprehensive performance visualization plots"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Win Rate Over Time with Moving Averages
    ax1 = plt.subplot(6, 2, 1)
    ax1.plot(df['episode'], df['win_rate'] * 100, alpha=0.3, label='Raw Win Rate', color='blue')
    if 'win_rate_ma_50' in df.columns:
        ax1.plot(df['episode'], df['win_rate_ma_50'] * 100, label='50-Episode MA', color='red', linewidth=2)
    if 'win_rate_ma_200' in df.columns:
        ax1.plot(df['episode'], df['win_rate_ma_200'] * 100, label='200-Episode MA', color='green', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Win Rate (%)')
    ax1.set_title('Win Rate Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Returns Over Time with Moving Averages
    ax2 = plt.subplot(6, 2, 2)
    ax2.plot(df['episode'], df['avg_return'] * 100, alpha=0.3, label='Raw Return', color='blue')
    if 'return_ma_50' in df.columns:
        ax2.plot(df['episode'], df['return_ma_50'] * 100, label='50-Episode MA', color='red', linewidth=2)
    if 'return_ma_200' in df.columns:
        ax2.plot(df['episode'], df['return_ma_200'] * 100, label='200-Episode MA', color='green', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Return (%)')
    ax2.set_title('Return Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Improvement Rate (Derivative of Win Rate)
    ax3 = plt.subplot(6, 2, 3)
    if 'improvement_rate' in df.columns:
        colors = ['green' if x > 0 else 'red' for x in df['improvement_rate']]
        ax3.bar(df['episode'], df['improvement_rate'] * 100, color=colors, alpha=0.7, width=1)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        # Add trend line
        z = np.polyfit(df['episode'], df['improvement_rate'], 1)
        p = np.poly1d(z)
        ax3.plot(df['episode'], p(df['episode']) * 100, "b--", alpha=0.8, linewidth=2, label=f'Trend: {z[0]:.6f}')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Improvement Rate (%)')
    ax3.set_title('Rate of Performance Improvement')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Sharpe Ratio Evolution
    ax4 = plt.subplot(6, 2, 4)
    if 'sharpe_ratio' in df.columns:
        ax4.plot(df['episode'], df['sharpe_ratio'], color='purple', linewidth=2)
        # Add 50-episode rolling average
        rolling_sharpe = df['sharpe_ratio'].rolling(window=50, min_periods=1).mean()
        ax4.plot(df['episode'], rolling_sharpe, color='orange', linewidth=2, label='50-Episode MA')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Good Sharpe (>1)')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Sharpe Ratio')
    ax4.set_title('Risk-Adjusted Performance (Sharpe Ratio)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Consistency Score
    ax5 = plt.subplot(6, 2, 5)
    if 'consistency_score' in df.columns:
        ax5.plot(df['episode'], df['consistency_score'], color='brown', linewidth=2)
        ax5.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        ax5.fill_between(df['episode'], 0, df['consistency_score'], alpha=0.3, color='brown')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Consistency Score')
    ax5.set_title('Performance Consistency (Higher is Better)')
    ax5.set_ylim(0, 1)
    ax5.grid(True, alpha=0.3)
    
    # 6. Trading Activity
    ax6 = plt.subplot(6, 2, 6)
    if 'total_trades' in df.columns:
        ax6.plot(df['episode'], df['total_trades'], color='teal', linewidth=1)
        # Add 50-episode rolling average
        rolling_trades = df['total_trades'].rolling(window=50, min_periods=1).mean()
        ax6.plot(df['episode'], rolling_trades, color='darkblue', linewidth=2, label='50-Episode MA')
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Total Trades')
    ax6.set_title('Trading Activity')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Win Rate vs Return Scatter
    ax7 = plt.subplot(6, 2, 7)
    if 'win_rate' in df.columns and 'avg_return' in df.columns:
        # Color by episode for time progression
        scatter = ax7.scatter(df['win_rate'] * 100, df['avg_return'] * 100, 
                            c=df['episode'], cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax7, label='Episode')
        ax7.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax7.axvline(x=50, color='black', linestyle='--', alpha=0.5)
        # Add target zone
        ax7.axhspan(5, 20, alpha=0.1, color='green', label='Target Return Zone')
        ax7.axvspan(60, 80, alpha=0.1, color='green', label='Target Win Rate Zone')
    ax7.set_xlabel('Win Rate (%)')
    ax7.set_ylabel('Average Return (%)')
    ax7.set_title('Win Rate vs Return Relationship')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Maximum Drawdown
    ax8 = plt.subplot(6, 2, 8)
    if 'max_drawdown' in df.columns:
        ax8.fill_between(df['episode'], 0, df['max_drawdown'] * 100, 
                        color='red', alpha=0.5, label='Drawdown')
        ax8.plot(df['episode'], df['max_drawdown'] * 100, color='darkred', linewidth=2)
        # Add acceptable drawdown threshold
        ax8.axhline(y=-10, color='orange', linestyle='--', alpha=0.7, label='10% Drawdown')
        ax8.axhline(y=-20, color='red', linestyle='--', alpha=0.7, label='20% Drawdown')
    ax8.set_xlabel('Episode')
    ax8.set_ylabel('Maximum Drawdown (%)')
    ax8.set_title('Maximum Drawdown Over Time')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Performance Heatmap (Episodes in blocks)
    ax9 = plt.subplot(6, 2, 9)
    if len(df) > 100:
        # Create blocks of 50 episodes
        block_size = 50
        n_blocks = len(df) // block_size
        heatmap_data = []
        
        for i in range(n_blocks):
            start_idx = i * block_size
            end_idx = (i + 1) * block_size
            block_data = df.iloc[start_idx:end_idx]
            
            heatmap_data.append([
                block_data['win_rate'].mean() * 100,
                block_data['avg_return'].mean() * 100,
                block_data['sharpe_ratio'].mean() if 'sharpe_ratio' in df.columns else 0,
                block_data['consistency_score'].mean() if 'consistency_score' in df.columns else 0
            ])
        
        heatmap_df = pd.DataFrame(heatmap_data, 
                                columns=['Win Rate', 'Avg Return', 'Sharpe', 'Consistency'])
        
        sns.heatmap(heatmap_df.T, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   xticklabels=[f'{i*block_size}-{(i+1)*block_size}' for i in range(n_blocks)],
                   ax=ax9, cbar_kws={'label': 'Value'})
        ax9.set_xlabel('Episode Blocks')
        ax9.set_title('Performance Heatmap by Episode Blocks')
    
    # 10. Cumulative Performance
    ax10 = plt.subplot(6, 2, 10)
    if 'avg_return' in df.columns:
        # Calculate cumulative returns
        cumulative_returns = (1 + df['avg_return']).cumprod() - 1
        ax10.plot(df['episode'], cumulative_returns * 100, color='darkgreen', linewidth=2)
        ax10.fill_between(df['episode'], 0, cumulative_returns * 100, alpha=0.3, color='green')
        ax10.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax10.set_xlabel('Episode')
    ax10.set_ylabel('Cumulative Return (%)')
    ax10.set_title('Cumulative Performance')
    ax10.grid(True, alpha=0.3)
    
    # 11. Best Performance Windows
    ax11 = plt.subplot(6, 2, 11)
    window_sizes = [10, 50, 100, 200]
    best_performances = []
    
    for window in window_sizes:
        if len(df) >= window:
            rolling_win_rate = df['win_rate'].rolling(window=window).mean()
            best_wr = rolling_win_rate.max() * 100
            best_performances.append(best_wr)
        else:
            best_performances.append(0)
    
    bars = ax11.bar(range(len(window_sizes)), best_performances, 
                    tick_label=[f'{w}-Episode' for w in window_sizes])
    
    # Color bars based on performance
    for i, (bar, perf) in enumerate(zip(bars, best_performances)):
        if perf >= 70:
            bar.set_color('green')
        elif perf >= 50:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    ax11.axhline(y=60, color='black', linestyle='--', alpha=0.5, label='Target: 60%')
    ax11.set_ylabel('Best Average Win Rate (%)')
    ax11.set_title('Best Performance by Window Size')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    
    # 12. Training Progress Summary
    ax12 = plt.subplot(6, 2, 12)
    ax12.axis('off')
    
    # Calculate summary statistics
    latest_episode = df['episode'].iloc[-1]
    latest_win_rate = df['win_rate'].iloc[-1] * 100
    latest_return = df['avg_return'].iloc[-1] * 100
    best_win_rate = df['win_rate'].max() * 100
    best_return = df['avg_return'].max() * 100
    
    # Average improvement over last 100 episodes
    if len(df) >= 100:
        recent_improvement = (df['win_rate'].iloc[-1] - df['win_rate'].iloc[-100]) * 100
        improvement_per_100 = recent_improvement
    else:
        improvement_per_100 = 0
    
    summary_text = f"""Training Progress Summary
    
Episodes Trained: {latest_episode}
Current Win Rate: {latest_win_rate:.2f}%
Current Avg Return: {latest_return:.2f}%

Best Win Rate Achieved: {best_win_rate:.2f}%
Best Avg Return Achieved: {best_return:.2f}%

Improvement (last 100 ep): {improvement_per_100:+.2f}%

Status: {'✅ PROFITABLE' if latest_return > 0 and latest_win_rate > 50 else '❌ NOT YET PROFITABLE'}
"""
    
    if 'win_rate_ma_50' in df.columns and len(df) > 0:
        ma50_wr = df['win_rate_ma_50'].iloc[-1] * 100
        summary_text += f"\n50-Episode MA Win Rate: {ma50_wr:.2f}%"
    
    if 'sharpe_ratio' in df.columns and len(df) > 0:
        latest_sharpe = df['sharpe_ratio'].iloc[-1]
        summary_text += f"\nLatest Sharpe Ratio: {latest_sharpe:.2f}"
    
    ax12.text(0.1, 0.9, summary_text, transform=ax12.transAxes, 
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"performance_analysis_{timestamp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance plots saved to: {save_path}")
        
        # Also save as PDF for better quality
        pdf_path = os.path.join(save_dir, f"performance_analysis_{timestamp}.pdf")
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"PDF version saved to: {pdf_path}")
    else:
        plt.show()
    
    return fig


def analyze_model_comparison(checkpoint_dir):
    """Compare performance across different model checkpoints"""
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    
    model_stats = []
    
    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            stats = {
                'model': checkpoint_file,
                'episode': checkpoint.get('episode', 0),
                'win_rate': checkpoint.get('win_rate', 0) * 100,
                'avg_return': checkpoint.get('avg_return', 0) * 100,
                'win_rate_ma_50': checkpoint.get('win_rate_ma_50', 0) * 100,
                'sharpe_ratio': checkpoint.get('sharpe_ratio', 0),
                'consistency_score': checkpoint.get('consistency_score', 0)
            }
            
            model_stats.append(stats)
        except:
            continue
    
    if model_stats:
        df = pd.DataFrame(model_stats)
        df = df.sort_values('win_rate', ascending=False)
        
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        # Find best models
        best_wr_model = df.loc[df['win_rate'].idxmax()]
        best_return_model = df.loc[df['avg_return'].idxmax()]
        
        print(f"\n🏆 Best Win Rate Model: {best_wr_model['model']}")
        print(f"   Win Rate: {best_wr_model['win_rate']:.2f}%")
        print(f"   Return: {best_wr_model['avg_return']:.2f}%")
        
        print(f"\n💰 Best Return Model: {best_return_model['model']}")
        print(f"   Win Rate: {best_return_model['win_rate']:.2f}%")
        print(f"   Return: {best_return_model['avg_return']:.2f}%")
        
        return df
    else:
        print("No model statistics found in checkpoints")
        return None


def main():
    parser = argparse.ArgumentParser(description='Visualize trading bot performance')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/profitable_fixed',
                       help='Directory containing checkpoints or path to specific checkpoint/CSV')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save plots instead of displaying them')
    parser.add_argument('--compare-models', action='store_true',
                       help='Compare performance across different model checkpoints')
    
    args = parser.parse_args()
    
    # Load performance data
    if os.path.exists(args.checkpoint_dir):
        if args.checkpoint_dir.endswith('.csv') or args.checkpoint_dir.endswith('.pt'):
            # Single file
            df = load_performance_data(args.checkpoint_dir)
            save_dir = os.path.dirname(args.checkpoint_dir) if args.save_plots else None
        else:
            # Directory - look for performance history
            df = load_performance_data(args.checkpoint_dir)
            save_dir = args.checkpoint_dir if args.save_plots else None
        
        if df is not None and not df.empty:
            print(f"Loaded {len(df)} episodes of performance data")
            
            # Create visualizations
            create_performance_plots(df, save_dir)
            
            # Compare models if requested
            if args.compare_models and os.path.isdir(args.checkpoint_dir):
                analyze_model_comparison(args.checkpoint_dir)
        else:
            print(f"No performance data found in {args.checkpoint_dir}")
    else:
        print(f"Path not found: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()