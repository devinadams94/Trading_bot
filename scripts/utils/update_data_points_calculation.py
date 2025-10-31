#!/usr/bin/env python3
"""
Updated calculation for 60 days of data
"""

# Trading hours per day (9:30 AM - 4:00 PM ET)
trading_hours_per_day = 6.5

# Trading days in 60 calendar days (approximately 42 trading days)
trading_days_in_60_days = 42

# Number of symbols
num_symbols = 20

# Calculate data points for stock price bars
hourly_bars_per_symbol = trading_hours_per_day * trading_days_in_60_days
total_stock_bars = hourly_bars_per_symbol * num_symbols

print("📊 Data Points Calculation for train.py (60 days)")
print("=" * 50)
print(f"Timeframe: 1 Hour bars")
print(f"Period: 60 days")
print(f"Trading days in 60 days: {trading_days_in_60_days}")
print(f"Trading hours per day: {trading_hours_per_day}")
print(f"Number of symbols: {num_symbols}")
print()

print("📈 Stock Price Data Points:")
print(f"Hourly bars per symbol (60 days): {int(hourly_bars_per_symbol):,}")
print(f"Total stock price data points: {int(total_stock_bars):,}")
print()

# Options data points (multiple options per underlying at each time)
filtered_options_per_day = 100
total_option_contracts_per_symbol = filtered_options_per_day * trading_days_in_60_days
total_option_data_points = total_option_contracts_per_symbol * num_symbols

print("📊 Options Data Points:")
print(f"Filtered option contracts per symbol per day: {filtered_options_per_day}")
print(f"Option contracts per symbol (60 days): {total_option_contracts_per_symbol:,}")
print(f"Total option data points (all symbols): {total_option_data_points:,}")
print()

# Total data volume
total_data_points = total_option_data_points
print("🎯 TOTAL DATA POINTS:")
print(f"Total option records for training: {total_data_points:,}")
print(f"Plus {int(total_stock_bars):,} hourly stock price bars")
print()

# Loading time estimation
print("⏱️  Loading Time Comparison:")
print(f"1 year (504,000 options): ~10-20 minutes")
print(f"60 days ({total_data_points:,} options): ~1-2 minutes")
print(f"Speed improvement: ~10x faster!")