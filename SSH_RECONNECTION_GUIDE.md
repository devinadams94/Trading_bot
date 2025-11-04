# SSH Reconnection & Training Monitoring Guide

## 🚨 Problem: SSH Disconnected During Training

If your SSH connection drops while training is running (without tmux/screen), the training process will **continue running** but you won't see the output.

## ✅ Quick Status Check

Run this script to check everything at once:

```bash
bash check_training_status.sh
```

This will show:
- ✅ Process status (running or not)
- 🖥️ GPU usage
- 📄 Latest log file
- 💾 Checkpoint progress
- 📊 Last episode number and reward

## 📊 Manual Monitoring Methods

### **1. View Log Files (RECOMMENDED)**

The training script logs everything to `logs/training_YYYYMMDD_HHMMSS.log`:

```bash
# Find latest log
ls -lt logs/

# View in real-time (like tail -f)
tail -f logs/training_20251104_023307.log

# View last 50 lines
tail -50 logs/training_20251104_023307.log

# Search for specific episodes
grep "Episode 100" logs/training_*.log

# See all episode summaries
grep "Episode.*Reward" logs/training_*.log | tail -20

# Check for errors
grep "ERROR\|❌" logs/training_*.log

# Check data loading progress
grep "Loading\|✅" logs/training_*.log | tail -30
```

### **2. Check Process Status**

```bash
# Is training still running?
ps aux | grep train_enhanced_clstm_ppo.py | grep -v grep

# Detailed process info
ps aux | grep python | grep train

# Get PID
pgrep -f train_enhanced_clstm_ppo.py

# Monitor resource usage
top -p $(pgrep -f train_enhanced_clstm_ppo.py)
```

### **3. Monitor GPU Usage**

```bash
# Current GPU status
nvidia-smi

# Watch in real-time (updates every 1 second)
watch -n 1 nvidia-smi

# Compact view
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv

# Check if your process is using GPU
nvidia-smi | grep python
```

### **4. Check Checkpoints**

```bash
# List all checkpoints
ls -lth checkpoints/production_run/

# Count checkpoints (saved every 100 episodes)
ls checkpoints/production_run/*.pt | wc -l

# See latest checkpoint
ls -lt checkpoints/production_run/*.pt | head -1

# Estimate progress (if checkpoints saved every 100 episodes)
CHECKPOINT_COUNT=$(ls checkpoints/production_run/*.pt 2>/dev/null | wc -l)
echo "Approximately $((CHECKPOINT_COUNT * 100)) episodes completed"
```

### **5. Check Disk Usage**

```bash
# Make sure disk isn't full
df -h

# Check log file size
du -h logs/training_*.log

# Check checkpoint directory size
du -sh checkpoints/production_run/
```

## 🔄 For Next Time: Use tmux or screen

### **Using tmux (RECOMMENDED)**

**Before starting training:**
```bash
# Install tmux if needed
sudo apt-get install tmux

# Start a new tmux session
tmux new -s training

# Run your training
python train_enhanced_clstm_ppo.py --num_episodes 5000 ...

# Detach from session (keeps it running)
# Press: Ctrl+B, then D
```

**After SSH reconnection:**
```bash
# List all tmux sessions
tmux ls

# Reattach to your session
tmux attach -t training

# Or just
tmux a
```

**Tmux cheat sheet:**
- `Ctrl+B, D` - Detach (leave running)
- `Ctrl+B, [` - Scroll mode (use arrow keys, Q to exit)
- `Ctrl+B, C` - Create new window
- `Ctrl+B, N` - Next window
- `tmux kill-session -t training` - Kill session

### **Using screen (Alternative)**

```bash
# Start screen session
screen -S training

# Run training
python train_enhanced_clstm_ppo.py ...

# Detach: Ctrl+A, then D

# Reattach after reconnection
screen -r training

# List sessions
screen -ls
```

### **Using nohup (Simplest)**

```bash
# Start training with nohup
nohup python train_enhanced_clstm_ppo.py \
    --num_episodes 5000 \
    --num_gpus 1 \
    --enable-multi-leg \
    --checkpoint-dir checkpoints/production_run \
    --resume \
    > training_nohup.log 2>&1 &

# Get the process ID
echo $!

# View output
tail -f training_nohup.log

# Check if still running
ps aux | grep train_enhanced
```

## 📈 Monitoring Dashboard (Advanced)

Create a simple monitoring script:

```bash
# Create monitor.sh
cat > monitor.sh << 'EOF'
#!/bin/bash
while true; do
    clear
    echo "=== TRAINING MONITOR ==="
    echo "Time: $(date)"
    echo ""
    
    # Process status
    if pgrep -f train_enhanced_clstm_ppo.py > /dev/null; then
        echo "✅ Training: RUNNING"
    else
        echo "❌ Training: STOPPED"
    fi
    
    # GPU
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    
    # Latest log
    echo ""
    echo "Latest progress:"
    tail -5 logs/training_*.log 2>/dev/null
    
    sleep 5
done
EOF

chmod +x monitor.sh
./monitor.sh
```

## 🛑 Stopping Training

```bash
# Graceful stop (saves checkpoint)
pkill -SIGINT -f train_enhanced_clstm_ppo.py

# Force kill (if graceful doesn't work)
pkill -9 -f train_enhanced_clstm_ppo.py

# Or using PID
kill -SIGINT $(pgrep -f train_enhanced_clstm_ppo.py)
```

## 📋 Common Scenarios

### **Scenario 1: "Did my training finish?"**

```bash
# Check if process is still running
ps aux | grep train_enhanced_clstm_ppo.py | grep -v grep

# Check last log entry
tail -20 logs/training_*.log

# Look for completion message
grep "Training complete\|✅.*complete" logs/training_*.log
```

### **Scenario 2: "What episode is it on?"**

```bash
# Find last episode
grep "Episode [0-9]*/[0-9]*" logs/training_*.log | tail -1

# Or
grep -o "Episode [0-9]*" logs/training_*.log | tail -1
```

### **Scenario 3: "Is it making progress or stuck?"**

```bash
# Check if log file is being updated
stat logs/training_*.log

# Watch for new lines
tail -f logs/training_*.log

# Check GPU usage (should be high if training)
nvidia-smi
```

### **Scenario 4: "How long has it been running?"**

```bash
# Check process start time
ps -p $(pgrep -f train_enhanced_clstm_ppo.py) -o etime,start

# Check log file creation time
ls -l logs/training_*.log
```

## 🎯 Best Practices

1. **Always use tmux/screen** for long-running training
2. **Check logs regularly** to ensure progress
3. **Monitor GPU usage** to verify training is active
4. **Save checkpoints frequently** (already configured every 100 episodes)
5. **Use `--resume`** flag to continue from checkpoints
6. **Keep SSH alive** with keep-alive settings:

```bash
# Add to ~/.ssh/config on your LOCAL machine
Host *
    ServerAliveInterval 60
    ServerAliveCountMax 10
```

## 🔧 Troubleshooting

### **Log file not updating?**
- Process might be stuck or crashed
- Check with `ps aux | grep train`
- Check GPU with `nvidia-smi`

### **Can't find log file?**
```bash
find . -name "training_*.log" -type f
```

### **Process running but no GPU usage?**
- Might be stuck in data loading
- Check log file for progress
- Might be using CPU instead

### **Out of memory?**
```bash
# Check system memory
free -h

# Check GPU memory
nvidia-smi

# Check disk space
df -h
```

## 📞 Quick Reference Card

```
┌─────────────────────────────────────────────────┐
│ QUICK REFERENCE                                 │
├─────────────────────────────────────────────────┤
│ Check status:                                   │
│   bash check_training_status.sh                 │
│                                                 │
│ View logs:                                      │
│   tail -f logs/training_*.log                   │
│                                                 │
│ Check GPU:                                      │
│   nvidia-smi                                    │
│                                                 │
│ Reattach tmux:                                  │
│   tmux attach -t training                       │
│                                                 │
│ Stop training:                                  │
│   pkill -SIGINT -f train_enhanced_clstm_ppo.py  │
└─────────────────────────────────────────────────┘
```

## 🚀 Recommended Workflow

**Starting training:**
```bash
# 1. Start tmux
tmux new -s training

# 2. Run training
python train_enhanced_clstm_ppo.py --num_episodes 5000 --resume ...

# 3. Detach (Ctrl+B, D)
```

**After SSH reconnection:**
```bash
# 1. Quick status check
bash check_training_status.sh

# 2. Reattach to tmux
tmux attach -t training

# 3. Or just view logs
tail -f logs/training_*.log
```

Now you'll never lose track of your training progress! 🎉

