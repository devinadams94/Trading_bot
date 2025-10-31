#!/usr/bin/env python3
"""Display available symbols for options trading"""

import argparse
from config.symbols_loader import SymbolsConfig
from config.config_loader import get_training_symbols, get_paper_trading_symbols, get_high_volatility_symbols, get_liquid_symbols
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Display available symbols for options trading')
    parser.add_argument('--category', type=str, help='Show symbols from specific category')
    parser.add_argument('--strategy', type=str, help='Show symbols for specific strategy')
    parser.add_argument('--all', action='store_true', help='Show all available symbols')
    parser.add_argument('--liquidity', type=int, help='Show symbols with minimum liquidity score')
    parser.add_argument('--defaults', action='store_true', help='Show default symbols')
    
    args = parser.parse_args()
    
    config = SymbolsConfig()
    
    print("=" * 80)
    print("OPTIONS TRADING SYMBOLS CONFIGURATION")
    print("=" * 80)
    
    if args.defaults or (not any([args.category, args.strategy, args.all, args.liquidity])):
        # Show default symbols
        print("\nDEFAULT TRAINING SYMBOLS:")
        training_symbols = get_training_symbols()
        print(f"  {', '.join(training_symbols)}")
        
        print("\nDEFAULT PAPER TRADING SYMBOLS:")
        paper_symbols = get_paper_trading_symbols()
        print(f"  {', '.join(paper_symbols)}")
        
        print("\nHIGH VOLATILITY SYMBOLS (Good for Options Premium):")
        high_vol = get_high_volatility_symbols()[:15]  # Top 15
        print(f"  {', '.join(high_vol)}")
        
        print("\nMOST LIQUID SYMBOLS (Score >= 9):")
        liquid = get_liquid_symbols(min_score=9)
        print(f"  {', '.join(liquid)}")
    
    if args.category:
        symbols = config.get_symbols_by_category(args.category)
        if symbols:
            print(f"\nCATEGORY: {args.category.upper()}")
            print(f"  {', '.join(symbols)}")
        else:
            print(f"\nCategory '{args.category}' not found.")
            print("Available categories:")
            categories = ['index_etfs', 'mega_tech', 'volatile_tech', 'ev_energy',
                         'meme_stocks', 'financials', 'commodities', 'biotech', 'retail']
            for cat in categories:
                print(f"  - {cat}")
    
    if args.strategy:
        symbols = config.get_symbols_for_strategy(args.strategy)
        if symbols:
            print(f"\nSTRATEGY: {args.strategy.upper()}")
            print(f"  Recommended symbols: {', '.join(symbols)}")
        else:
            print(f"\nStrategy '{args.strategy}' not found.")
            print("Available strategies:")
            strategies = ['covered_calls', 'protective_puts', 'iron_condors', 'volatility_plays']
            for strat in strategies:
                print(f"  - {strat}")
    
    if args.liquidity:
        all_symbols = config.get_all_available_symbols()
        filtered = config.filter_symbols_by_liquidity(all_symbols, args.liquidity)
        print(f"\nSYMBOLS WITH LIQUIDITY SCORE >= {args.liquidity}:")
        print(f"  {', '.join(filtered)}")
    
    if args.all:
        all_symbols = config.get_all_available_symbols()
        print(f"\nALL AVAILABLE SYMBOLS ({len(all_symbols)} total):")
        
        # Group by categories for better display
        print("\nINDEX ETFs:")
        print(f"  {', '.join(config.get_symbols_by_category('index_etfs'))}")
        
        print("\nMEGA-CAP TECH:")
        print(f"  {', '.join(config.get_symbols_by_category('mega_tech'))}")
        
        print("\nVOLATILE TECH:")
        print(f"  {', '.join(config.get_symbols_by_category('volatile_tech'))}")
        
        print("\nMEME STOCKS:")
        print(f"  {', '.join(config.get_symbols_by_category('meme_stocks'))}")
        
        print("\nFINANCIALS:")
        print(f"  {', '.join(config.get_symbols_by_category('financials'))}")
        
        print("\nCOMMODITIES:")
        print(f"  {', '.join(config.get_symbols_by_category('commodities'))}")
    
    # Show liquidity scores
    print("\n" + "=" * 80)
    print("LIQUIDITY SCORES (1-10, higher is better):")
    print("=" * 80)
    
    scores_data = config.symbols_data.get('liquidity_scores', {})
    if scores_data:
        # Convert to DataFrame for nice display
        df = pd.DataFrame(list(scores_data.items()), columns=['Symbol', 'Score'])
        df = df.sort_values('Score', ascending=False)
        
        print("\nTop 20 Most Liquid:")
        print(df.head(20).to_string(index=False))
    
    # Show recommended training set
    print("\n" + "=" * 80)
    print("RECOMMENDED TRAINING SET:")
    print("=" * 80)
    recommendations = config.get_training_recommendations(
        include_indices=True,
        include_memes=False,
        min_liquidity=7
    )
    print(f"\nFor best results, train with these {len(recommendations)} symbols:")
    print(f"  {', '.join(recommendations)}")
    
    print("\n" + "=" * 80)
    print("USAGE EXAMPLES:")
    print("=" * 80)
    print("\nTraining with default symbols:")
    print("  python train_options_real_data.py")
    print("\nTraining with high volatility symbols:")
    print("  python train_options_real_data.py --symbols", ' '.join(get_high_volatility_symbols()[:8]))
    print("\nPaper trading with liquid symbols:")
    print("  python main_options_clstm_ppo.py --mode paper")
    print("\nShow symbols for iron condor strategy:")
    print("  python list_symbols.py --strategy iron_condors")


if __name__ == '__main__':
    main()