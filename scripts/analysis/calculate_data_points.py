#!/usr/bin/env python3
"""
Calculate the number of data points for 1 year of hourly options data
"""

# Trading hours per day (9:30 AM - 4:00 PM ET)
trading_hours_per_day = 6.5

# Trading days per year (approximately 252)
trading_days_per_year = 252

# Number of symbols
num_symbols = 20

# Calculate data points for stock price bars
hourly_bars_per_symbol = trading_hours_per_day * trading_days_per_year
total_stock_bars = hourly_bars_per_symbol * num_symbols

print("📊 Data Points Calculation for train.py")
print("=" * 50)
print(f"Timeframe: 1 Hour bars")
print(f"Period: 1 year (365 days)")
print(f"Trading days per year: {trading_days_per_year}")
print(f"Trading hours per day: {trading_hours_per_day}")
print(f"Number of symbols: {num_symbols}")
print()

print("📈 Stock Price Data Points:")
print(f"Hourly bars per symbol per year: {int(hourly_bars_per_symbol):,}")
print(f"Total stock price data points: {int(total_stock_bars):,}")
print()

# Options data points (multiple options per underlying at each time)
# Assuming average of 50 relevant options per symbol per day (strikes near money)
avg_options_per_symbol_per_day = 50

# But the code filters to options within 7% of stock price (line 149 in historical_options_data.py)
# and limits to 100 contracts per day (line 150)
filtered_options_per_day = 100

total_option_contracts_per_symbol = filtered_options_per_day * trading_days_per_year
total_option_data_points = total_option_contracts_per_symbol * num_symbols

print("📊 Options Data Points:")
print(f"Filtered option contracts per symbol per day: {filtered_options_per_day}")
print(f"Option contracts per symbol per year: {total_option_contracts_per_symbol:,}")
print(f"Total option data points (all symbols): {total_option_data_points:,}")
print()

# Each option data point contains:
option_features = [
    "timestamp", "symbol", "option_symbol", "strike", "expiration",
    "option_type", "underlying_price", "bid", "ask", "last",
    "volume", "open_interest", "implied_volatility", 
    "delta", "gamma", "theta", "vega", "rho"
]

print("📝 Features per option data point:")
print(f"Number of features: {len(option_features)}")
print(f"Features: {', '.join(option_features)}")
print()

# Total data volume
total_data_points = total_option_data_points
print("🎯 TOTAL DATA POINTS:")
print(f"Total option records for training: {total_data_points:,}")
print(f"Plus {int(total_stock_bars):,} hourly stock price bars")
print()

# Memory estimation (rough)
bytes_per_float = 4
features_per_option = len(option_features)
memory_mb = (total_data_points * features_per_option * bytes_per_float) / (1024 * 1024)
print(f"💾 Estimated memory usage: ~{memory_mb:,.0f} MB")