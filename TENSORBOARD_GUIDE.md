# TensorBoard Training Visualization Guide

## 📊 Overview

TensorBoard is now integrated into the CLSTM-PPO training script to provide **real-time visual tracking** of training progress, profitability, and model performance.

---

## 🚀 Quick Start

### **1. Start Training**
```bash
python train_enhanced_clstm_ppo.py --use-flat-files --episodes 5000
```

Training will automatically create TensorBoard logs in the `runs/` directory.

### **2. Launch TensorBoard**

**In a separate terminal:**
```bash
# Activate virtual environment
source venv/bin/activate

# Launch TensorBoard
tensorboard --logdir=runs
```

**Or specify a specific run:**
```bash
tensorboard --logdir=runs/clstm_ppo_20251118_192000
```

### **3. View Dashboard**

Open your browser and navigate to:
```
http://localhost:6006
```

TensorBoard will automatically refresh as new data is logged during training!

---

## 📈 Metrics Tracked

### **Episode Metrics** (Per Episode)

| Metric | Description | Tab |
|--------|-------------|-----|
| `Episode/Return` | Portfolio return for the episode | SCALARS |
| `Episode/Portfolio_Value` | Final portfolio value | SCALARS |
| `Episode/Win_Rate` | Percentage of profitable trades | SCALARS |
| `Episode/Total_Reward` | Sum of all rewards in episode | SCALARS |
| `Episode/Trades` | Number of trades executed | SCALARS |
| `Episode/Profitable_Trades` | Number of profitable trades | SCALARS |
| `Episode/Episode_Length` | Number of steps in episode | SCALARS |

### **Rolling Averages** (Last 100 Episodes)

| Metric | Description | Tab |
|--------|-------------|-----|
| `Rolling/Avg_Return_100` | Average return over last 100 episodes | SCALARS |
| `Rolling/Avg_Win_Rate_100` | Average win rate over last 100 episodes | SCALARS |
| `Rolling/Std_Return_100` | Standard deviation of returns | SCALARS |
| `Rolling/Sharpe_Ratio_100` | Sharpe ratio (return/volatility) | SCALARS |

### **Best Metrics** (All-Time Bests)

| Metric | Description | Tab |
|--------|-------------|-----|
| `Best/Performance` | Best episode return achieved | SCALARS |
| `Best/Win_Rate` | Best win rate achieved | SCALARS |
| `Best/Profit_Rate` | Best profit rate achieved | SCALARS |
| `Best/Sharpe_Ratio` | Best Sharpe ratio achieved | SCALARS |
| `Best/Composite_Score` | Best composite score (WR+PR+Return) | SCALARS |

### **Profitability Tracking**

| Metric | Description | Tab |
|--------|-------------|-----|
| `Profitability/Episode_Profitable` | 1.0 if episode was profitable, 0.0 otherwise | SCALARS |
| `Cumulative/Total_Return` | Sum of all episode returns | SCALARS |
| `Cumulative/Total_Trades` | Total trades across all episodes | SCALARS |
| `Cumulative/Profitability_Rate` | % of episodes that were profitable | SCALARS |

### **Risk Metrics**

| Metric | Description | Tab |
|--------|-------------|-----|
| `Risk/Max_Drawdown` | Maximum drawdown from peak | SCALARS |
| `Risk/Current_Drawdown` | Current drawdown from peak | SCALARS |

### **Model Training Metrics** (Per Training Step)

| Metric | Description | Tab |
|--------|-------------|-----|
| `Loss/PPO_Total` | Total PPO loss | SCALARS |
| `Loss/CLSTM` | CLSTM encoder loss | SCALARS |
| `Loss/Actor` | Actor network loss | SCALARS |
| `Loss/Critic` | Critic network loss | SCALARS |
| `Model/Entropy` | Policy entropy (exploration) | SCALARS |
| `Model/KL_Divergence` | KL divergence from old policy | SCALARS |

---

## 🎯 Key Visualizations to Monitor

### **1. Profitability Progress**
- **Chart:** `Cumulative/Profitability_Rate`
- **Goal:** Should trend upward toward 50%+ as training progresses
- **What to look for:** Steady increase indicates learning

### **2. Returns Over Time**
- **Chart:** `Episode/Return` and `Rolling/Avg_Return_100`
- **Goal:** Positive and increasing returns
- **What to look for:** Upward trend in rolling average

### **3. Win Rate**
- **Chart:** `Episode/Win_Rate` and `Rolling/Avg_Win_Rate_100`
- **Goal:** 50%+ win rate
- **What to look for:** Stabilization above 50%

### **4. Sharpe Ratio**
- **Chart:** `Rolling/Sharpe_Ratio_100`
- **Goal:** Positive and increasing (>1.0 is good, >2.0 is excellent)
- **What to look for:** Consistent positive values

### **5. Risk Management**
- **Chart:** `Risk/Max_Drawdown` and `Risk/Current_Drawdown`
- **Goal:** Low drawdown (<10% is good)
- **What to look for:** Controlled drawdowns, quick recovery

### **6. Model Training**
- **Charts:** `Loss/PPO_Total`, `Loss/CLSTM`, `Loss/Actor`, `Loss/Critic`
- **Goal:** Decreasing losses over time
- **What to look for:** Convergence (losses stabilize)

### **7. Exploration vs Exploitation**
- **Chart:** `Model/Entropy`
- **Goal:** High early (exploration), decreasing later (exploitation)
- **What to look for:** Gradual decrease as training progresses

---

## 🔍 TensorBoard Tips

### **Compare Multiple Runs**
```bash
tensorboard --logdir=runs
```
TensorBoard will automatically detect all runs in the `runs/` directory and allow you to compare them!

### **Smooth Noisy Curves**
- Use the **smoothing slider** in TensorBoard (top left) to smooth noisy metrics
- Recommended: 0.6-0.8 for episode metrics

### **Filter Metrics**
- Use the search box to filter specific metrics
- Example: Type "Win_Rate" to see only win rate metrics

### **Download Data**
- Click the download icon (⬇️) to export data as CSV or JSON

---

## 📂 File Structure

```
runs/
└── clstm_ppo_20251118_192000/     # Timestamp-based directory
    └── events.out.tfevents.*       # TensorBoard event files
```

---

## ✅ Summary

**TensorBoard provides:**
- ✅ Real-time visualization of training progress
- ✅ Profitability tracking over time
- ✅ Risk metrics (drawdown, volatility)
- ✅ Model performance metrics (losses, entropy, KL divergence)
- ✅ Comparison of multiple training runs
- ✅ Export capabilities for further analysis

**Start training and watch your model learn in real-time!** 🚀

