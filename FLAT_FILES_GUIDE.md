# Flat Files Data Loading Guide

## 🚀 Overview

Flat file data loading provides **much faster** training by loading pre-downloaded data from disk instead of making REST API calls. This is ideal for:

- **Offline training** (no internet required)
- **Faster iteration** (10-100x faster data loading)
- **Reproducible experiments** (same data every time)
- **Cost savings** (no API rate limits or costs)

---

## 📊 Performance Comparison

| Method | Data Loading Time (3 years, 23 symbols) | Internet Required | Cost |
|--------|------------------------------------------|-------------------|------|
| **REST API** | 15-30 minutes | ✅ Yes | API calls |
| **Flat Files (Parquet)** | 5-15 seconds | ❌ No | None |
| **Flat Files (CSV)** | 10-30 seconds | ❌ No | None |

**Speed improvement: 60-360x faster!**

---

## 📁 File Structure

```
data/flat_files/
├── stocks/
│   ├── SPY.parquet
│   ├── QQQ.parquet
│   ├── AAPL.parquet
│   └── ...
└── options/
    ├── SPY_options.parquet
    ├── QQQ_options.parquet
    ├── AAPL_options.parquet
    └── ...
```

---

## 🔧 Setup Instructions

### Step 1: Install Dependencies

```bash
# For Parquet support (recommended - much faster)
pip install pyarrow

# Parquet is 2-5x faster than CSV and uses less disk space
```

### Step 2: Download Data to Flat Files

```bash
# Download 3 years of data for all symbols (default)
python3 download_data_to_flat_files.py

# Download specific date range
python3 download_data_to_flat_files.py --days 730  # 2 years

# Download specific symbols
python3 download_data_to_flat_files.py --symbols SPY QQQ AAPL

# Use CSV format instead of Parquet
python3 download_data_to_flat_files.py --format csv

# Custom output directory
python3 download_data_to_flat_files.py --output-dir /path/to/data
```

**Expected output:**
```
================================================================================
📥 DOWNLOADING DATA TO FLAT FILES
================================================================================

✅ Using Massive.com API key: O_182Z1c...
📅 Date range: 2022-11-13 to 2025-11-12 (1095 days)
📊 Symbols: 23
📁 Output directory: data/flat_files
📄 File format: parquet

================================================================================
📈 DOWNLOADING STOCK DATA
================================================================================

  [1/23] 🌐 Fetching historical stock data from REST API...
  [1/23] ✅ Fetched 756 real bars for SPY
  ...

💾 Saving stock data to flat files...

  ✅ SPY: 756 bars → data/flat_files/stocks/SPY.parquet
  ✅ QQQ: 756 bars → data/flat_files/stocks/QQQ.parquet
  ...

================================================================================
📊 DOWNLOADING OPTIONS DATA
================================================================================

  [1/23] 🌐 Fetching options data from REST API...
  [1/23] ✅ SPY: 187,500 contracts
  ...

💾 Saving options data to flat files...

  ✅ SPY: 187,500 contracts → data/flat_files/options/SPY_options.parquet
  ✅ QQQ: 156,000 contracts → data/flat_files/options/QQQ_options.parquet
  ...

================================================================================
✅ DOWNLOAD COMPLETE
================================================================================

📁 Data saved to: data/flat_files
📈 Stock files: 23
📊 Options files: 23
```

**Time:** 15-30 minutes (one-time download)

---

### Step 3: Train with Flat Files

```bash
# Use flat files instead of REST API
python3 train_enhanced_clstm_ppo.py --use-flat-files --episodes 2000

# Specify custom directory
python3 train_enhanced_clstm_ppo.py --use-flat-files --flat-files-dir /path/to/data

# Use CSV format
python3 train_enhanced_clstm_ppo.py --use-flat-files --flat-files-format csv

# Quick test with flat files
python3 train_enhanced_clstm_ppo.py --use-flat-files --quick-test
```

**Expected output:**
```
🔧 Initializing Enhanced CLSTM-PPO Trainer
📁 Using flat file data loader
   Data directory: data/flat_files
   File format: parquet

📊 Loading stock data for 23 symbols from flat files...
  [1/23] ✅ SPY: 756 bars
  [2/23] ✅ QQQ: 756 bars
  ...
✅ Loaded stock data for 23/23 symbols

📊 Loading options data for 23 symbols from flat files...
  [1/23] ✅ SPY: 187,500 contracts
  [2/23] ✅ QQQ: 156,000 contracts
  ...
✅ Loaded options data for 23/23 symbols

✅ Data loaded in 8.3 seconds (vs 18 minutes with REST API)
```

---

## 📋 Command Reference

### Download Data

```bash
# Basic usage
python3 download_data_to_flat_files.py

# All options
python3 download_data_to_flat_files.py \
    --symbols SPY QQQ AAPL MSFT \
    --days 1095 \
    --output-dir data/flat_files \
    --format parquet
```

### Train with Flat Files

```bash
# Basic usage
python3 train_enhanced_clstm_ppo.py --use-flat-files --episodes 2000

# All options
python3 train_enhanced_clstm_ppo.py \
    --use-flat-files \
    --flat-files-dir data/flat_files \
    --flat-files-format parquet \
    --episodes 5000 \
    --no-realistic-costs
```

---

## 🔄 Updating Data

To update your flat files with new data:

```bash
# Re-download all data (overwrites existing files)
python3 download_data_to_flat_files.py

# Download only recent data (append mode - not yet implemented)
# Coming soon: incremental updates
```

**Recommendation:** Update data weekly or monthly depending on your needs.

---

## 💾 Disk Space Requirements

| Symbols | Days | Format | Stock Data | Options Data | Total |
|---------|------|--------|------------|--------------|-------|
| 23 | 1095 (3 years) | Parquet | ~5 MB | ~500 MB | ~505 MB |
| 23 | 1095 (3 years) | CSV | ~15 MB | ~1.5 GB | ~1.5 GB |
| 100 | 1095 (3 years) | Parquet | ~20 MB | ~2 GB | ~2 GB |
| 100 | 1095 (3 years) | CSV | ~60 MB | ~6 GB | ~6 GB |

**Recommendation:** Use Parquet format for 3x smaller files and faster loading.

---

## 🎯 Best Practices

### 1. Use Parquet Format
```bash
pip install pyarrow
python3 download_data_to_flat_files.py --format parquet
```
- 3x smaller files
- 2-5x faster loading
- Better compression

### 2. Download Once, Train Many Times
```bash
# Download data once
python3 download_data_to_flat_files.py

# Train multiple times (fast!)
python3 train_enhanced_clstm_ppo.py --use-flat-files --episodes 1000
python3 train_enhanced_clstm_ppo.py --use-flat-files --episodes 2000 --no-realistic-costs
python3 train_enhanced_clstm_ppo.py --use-flat-files --episodes 5000
```

### 3. Keep Data Fresh
```bash
# Update data monthly
0 0 1 * * cd /path/to/Trading_bot && python3 download_data_to_flat_files.py
```

---

## 🐛 Troubleshooting

### Issue: "File not found" error

**Solution:** Download data first
```bash
python3 download_data_to_flat_files.py
```

### Issue: "pyarrow not installed"

**Solution:** Install pyarrow or use CSV
```bash
pip install pyarrow
# OR
python3 train_enhanced_clstm_ppo.py --use-flat-files --flat-files-format csv
```

### Issue: Data is outdated

**Solution:** Re-download data
```bash
python3 download_data_to_flat_files.py
```

---

## ✅ Summary

**Flat files provide:**
- ✅ 60-360x faster data loading
- ✅ Offline training (no internet required)
- ✅ Reproducible experiments
- ✅ No API rate limits
- ✅ Lower costs

**Setup:**
1. Install pyarrow: `pip install pyarrow`
2. Download data: `python3 download_data_to_flat_files.py`
3. Train: `python3 train_enhanced_clstm_ppo.py --use-flat-files --episodes 2000`

**That's it!** 🚀

