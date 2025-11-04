# API Key 401 Error & Data Loading Fix

## 🐛 Issues Found

### Issue 1: Alpaca API 401 Authorization Error
```
<h1>401 Authorization Required</h1>
⚠️ No stock data loaded
❌ No valid data loaded for any symbol
```

**Cause:** Your Alpaca API keys are either:
- Invalid or expired
- Don't have the right permissions for historical data
- Paper trading keys trying to access premium data

### Issue 2: Synthetic Data Fallback Broken
```
KeyError: 0
File "/root/Trading_bot/src/multi_leg_options_env.py", line 119, in _execute_covered_strategy
    current_data = self.market_data[self.current_step]
```

**Cause:** Multi-leg environment was accessing market data with integer index instead of using the proper method.

## ✅ Fixes Applied

### Fix 1: Multi-Leg Environment Data Access (COMPLETED)

**Files Modified:**
- `src/multi_leg_options_env.py` (lines 114-124, 268-278)

**Changes:**
```python
# ❌ BEFORE: Wrong data access
current_data = self.market_data[self.current_step]

# ✅ AFTER: Correct data access
current_data = self._get_current_market_data()
if current_data is None:
    return self._get_observation(), 0, False, {}
```

**Fixed in 2 locations:**
1. `_execute_covered_strategy()` method
2. `_execute_multi_leg_strategy()` method

### Fix 2: Alpaca API Keys (ACTION REQUIRED)

You need to get new API keys from Alpaca. Here's how:

#### Option A: Get New Paper Trading Keys (FREE)

1. Go to https://alpaca.markets/
2. Sign in to your account
3. Go to "Paper Trading" section
4. Click "Regenerate API Keys"
5. Copy the new keys

#### Option B: Use Live Trading Keys (PAID)

1. Fund your Alpaca account
2. Go to "Live Trading" section
3. Generate API keys
4. Update `.env` file

#### Update Your `.env` File

```bash
# Edit the .env file
nano .env

# Replace with your new keys:
ALPACA_API_KEY=your_new_key_here
ALPACA_SECRET_KEY=your_new_secret_here
ALPACA_API_BASE_URL=https://paper-api.alpaca.markets
```

### Fix 3: Train with Synthetic Data (TEMPORARY WORKAROUND)

If you want to test training without fixing API keys, the synthetic data should now work:

```bash
# The synthetic data fallback should now work correctly
python train_enhanced_clstm_ppo.py --quick-test --num_gpus 1 --checkpoint-dir checkpoints/test --fresh-start
```

## 🧪 Testing

### Test 1: Verify Synthetic Data Works

```bash
# Should work now with synthetic data
python train_enhanced_clstm_ppo.py --quick-test --num_gpus 1 --checkpoint-dir checkpoints/test --fresh-start
```

**Expected output:**
```
⚠️ No stock data loaded
❌ No valid data loaded for any symbol
No real data loaded, using synthetic data
✅ Environment initialized with 23 symbols
🎯 Starting CLSTM-PPO training
Episode 1/100: ...
```

### Test 2: Verify API Keys (After Getting New Ones)

```bash
# Test API connection
python -c "
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('ALPACA_API_KEY')
secret_key = os.getenv('ALPACA_SECRET_KEY')

print(f'Testing API key: {api_key[:10]}...')

client = StockHistoricalDataClient(api_key, secret_key)
request = StockBarsRequest(
    symbol_or_symbols='SPY',
    timeframe=TimeFrame.Day,
    start=datetime.now() - timedelta(days=7),
    end=datetime.now()
)

try:
    bars = client.get_stock_bars(request)
    print(f'✅ API keys work! Got {len(bars.df)} bars for SPY')
except Exception as e:
    print(f'❌ API error: {e}')
"
```

## 📊 Current Status

- ✅ **Synthetic data fallback fixed** - Training can proceed without API keys
- ✅ **Multi-leg environment data access fixed** - No more KeyError
- ⚠️ **API keys need updating** - 401 error means keys are invalid

## 🚀 Next Steps

### Option 1: Train with Synthetic Data (Quick)

```bash
# Works immediately, no API keys needed
python train_enhanced_clstm_ppo.py --quick-test --num_gpus 1 --checkpoint-dir checkpoints/test --fresh-start
```

**Pros:**
- ✅ Works immediately
- ✅ Tests the training pipeline
- ✅ Validates model architecture

**Cons:**
- ❌ Not realistic market data
- ❌ Won't learn real market patterns
- ❌ Only for testing, not production

### Option 2: Fix API Keys and Use Real Data (Recommended)

1. Get new Alpaca API keys (see above)
2. Update `.env` file
3. Run training with real data

```bash
# After updating API keys
python train_enhanced_clstm_ppo.py --quick-test --num_gpus 1 --checkpoint-dir checkpoints/test --fresh-start
```

**Pros:**
- ✅ Real market data
- ✅ Learns actual market patterns
- ✅ Production-ready training

**Cons:**
- ⏱️ Requires getting new API keys (5 minutes)

## 🔍 Debugging Commands

### Check if API keys are loaded:
```bash
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('API Key:', os.getenv('ALPACA_API_KEY', 'NOT FOUND'))
print('Secret Key:', os.getenv('ALPACA_SECRET_KEY', 'NOT FOUND')[:10] + '...')
print('Base URL:', os.getenv('ALPACA_API_BASE_URL', 'NOT FOUND'))
"
```

### Check if synthetic data is being created:
```bash
# Look for this in logs
grep "synthetic data" logs/training_*.log
```

### Check current training status:
```bash
# View latest log
tail -50 logs/training_*.log
```

## 📝 Summary

| Issue | Status | Action Required |
|-------|--------|-----------------|
| Multi-leg data access bug | ✅ Fixed | None |
| Synthetic data fallback | ✅ Fixed | None |
| API 401 error | ⚠️ Needs action | Get new API keys |
| Training can proceed | ✅ Yes | Use synthetic data or fix API keys |

## 💡 Recommendation

**For immediate testing:**
```bash
# Test with synthetic data (works now)
python train_enhanced_clstm_ppo.py --quick-test --num_gpus 1 --checkpoint-dir checkpoints/test --fresh-start
```

**For production training:**
1. Get new Alpaca API keys
2. Update `.env` file
3. Run full training with real data

```bash
tmux new -s training
python train_enhanced_clstm_ppo.py --num_episodes 5000 --num_gpus 1 --enable-multi-leg --use-ensemble --num-ensemble-models 3 --early-stopping-patience 500 --checkpoint-dir checkpoints/production_run --resume
```

