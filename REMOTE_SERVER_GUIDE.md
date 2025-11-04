# Remote Server Training Guide

## 🎯 Quick Start

### Option 1: Automated Script (Recommended)

```bash
# Make scripts executable
chmod +x ssh_and_run_training.sh deploy_to_server.sh

# Deploy code and run training
bash deploy_to_server.sh
```

This will:
1. Find your SSH key automatically
2. Sync code to the server
3. Give you options to run training

### Option 2: Manual SSH

```bash
# Find your SSH key
ls ~/Downloads/*.pem ~/Downloads/*.key ~/.ssh/id_*

# SSH into server (replace KEY_PATH with your key)
ssh -i ~/Downloads/your_key.pem root@162.243.13.8

# Once connected, run training
cd /root/Trading_bot
python -u train_enhanced_clstm_ppo.py --quick-test --num_gpus 1 --checkpoint-dir checkpoints/test --fresh-start
```

## 📋 Server Details

- **IP:** 162.243.13.8
- **User:** root
- **Directory:** /root/Trading_bot
- **SSH Key:** Should be in ~/Downloads/

## 🚀 Training Commands

### Quick Test (Recommended First)

```bash
# SSH into server
ssh -i ~/Downloads/your_key.pem root@162.243.13.8

# Run quick test
cd /root/Trading_bot
python -u train_enhanced_clstm_ppo.py \
    --quick-test \
    --num_gpus 1 \
    --checkpoint-dir checkpoints/test \
    --fresh-start
```

**Expected time:** 5-10 minutes
**What it does:** Tests with 3 symbols, 90 days, 100 episodes

### Production Training (In tmux)

```bash
# SSH into server
ssh -i ~/Downloads/your_key.pem root@162.243.13.8

# Start tmux session
tmux new -s training

# Run production training
cd /root/Trading_bot
python -u train_enhanced_clstm_ppo.py \
    --num_episodes 5000 \
    --num_gpus 1 \
    --enable-multi-leg \
    --use-ensemble \
    --num-ensemble-models 3 \
    --checkpoint-dir checkpoints/production_run \
    --resume

# Detach from tmux: Ctrl+B, then D
# Reattach later: tmux attach -t training
```

**Expected time:** Several hours to days
**What it does:** Full production training with all features

## 🔧 Troubleshooting

### Can't find SSH key?

```bash
# Search for SSH keys
find ~ -name "*.pem" -o -name "*.key" 2>/dev/null

# Common locations:
ls ~/Downloads/
ls ~/.ssh/
```

### SSH connection refused?

```bash
# Test connection
ssh -v -i ~/Downloads/your_key.pem root@162.243.13.8

# Check if key has correct permissions
chmod 600 ~/Downloads/your_key.pem
```

### Training appears frozen?

**Remember:** Use `python -u` flag for unbuffered output!

```bash
# Correct (shows real-time progress):
python -u train_enhanced_clstm_ppo.py --quick-test --num_gpus 1 --checkpoint-dir checkpoints/test --fresh-start

# Wrong (appears frozen):
python train_enhanced_clstm_ppo.py --quick-test --num_gpus 1 --checkpoint-dir checkpoints/test --fresh-start
```

### Check if training is running

```bash
# SSH into server
ssh -i ~/Downloads/your_key.pem root@162.243.13.8

# Check for Python processes
ps aux | grep train_enhanced_clstm_ppo.py

# Check GPU usage
nvidia-smi

# Check tmux sessions
tmux ls

# Attach to tmux session
tmux attach -t training
```

### View logs

```bash
# SSH into server
ssh -i ~/Downloads/your_key.pem root@162.243.13.8

# View latest log
cd /root/Trading_bot
tail -f logs/training_*.log

# Or view last 100 lines
tail -100 logs/training_*.log
```

## 📊 Monitoring Training

### From Another Terminal

```bash
# Terminal 1: SSH and run training
ssh -i ~/Downloads/your_key.pem root@162.243.13.8
cd /root/Trading_bot
python -u train_enhanced_clstm_ppo.py --quick-test --num_gpus 1 --checkpoint-dir checkpoints/test --fresh-start

# Terminal 2: Monitor GPU
ssh -i ~/Downloads/your_key.pem root@162.243.13.8
watch -n 1 nvidia-smi

# Terminal 3: Monitor logs
ssh -i ~/Downloads/your_key.pem root@162.243.13.8
tail -f /root/Trading_bot/logs/training_*.log
```

### Using tmux (Recommended)

```bash
# SSH into server
ssh -i ~/Downloads/your_key.pem root@162.243.13.8

# Create tmux session
tmux new -s training

# Run training
cd /root/Trading_bot
python -u train_enhanced_clstm_ppo.py --quick-test --num_gpus 1 --checkpoint-dir checkpoints/test --fresh-start

# Detach: Ctrl+B, then D
# You can now close SSH connection

# Later, reconnect and reattach:
ssh -i ~/Downloads/your_key.pem root@162.243.13.8
tmux attach -t training
```

## 🔄 Syncing Code to Server

### Using rsync (Fast)

```bash
# From local machine
rsync -avz --progress \
    -e "ssh -i ~/Downloads/your_key.pem" \
    --exclude 'cache/' \
    --exclude 'logs/' \
    --exclude 'checkpoints/' \
    --exclude '__pycache__/' \
    ~/Desktop/Trading_bot/ \
    root@162.243.13.8:/root/Trading_bot/
```

### Using scp (Simple)

```bash
# Copy single file
scp -i ~/Downloads/your_key.pem \
    train_enhanced_clstm_ppo.py \
    root@162.243.13.8:/root/Trading_bot/

# Copy directory
scp -i ~/Downloads/your_key.pem -r \
    src/ \
    root@162.243.13.8:/root/Trading_bot/
```

### Using git (Best Practice)

```bash
# On local machine: commit and push
git add .
git commit -m "Update training script"
git push

# On server: pull changes
ssh -i ~/Downloads/your_key.pem root@162.243.13.8
cd /root/Trading_bot
git pull
```

## 📁 Useful Server Commands

```bash
# Check disk space
df -h

# Check GPU
nvidia-smi

# Check running processes
ps aux | grep python

# Check memory usage
free -h

# View system resources
htop

# Kill training process
pkill -f train_enhanced_clstm_ppo.py

# View directory size
du -sh /root/Trading_bot/*

# Clean cache
rm -rf /root/Trading_bot/cache/*
```

## 🎯 Complete Workflow

### 1. First Time Setup

```bash
# Find SSH key
ls ~/Downloads/*.pem

# Test connection
ssh -i ~/Downloads/your_key.pem root@162.243.13.8

# Sync code
bash deploy_to_server.sh
```

### 2. Run Quick Test

```bash
# SSH into server
ssh -i ~/Downloads/your_key.pem root@162.243.13.8

# Run test
cd /root/Trading_bot
python -u train_enhanced_clstm_ppo.py --quick-test --num_gpus 1 --checkpoint-dir checkpoints/test --fresh-start
```

### 3. Run Production Training

```bash
# SSH into server
ssh -i ~/Downloads/your_key.pem root@162.243.13.8

# Start tmux
tmux new -s training

# Run training
cd /root/Trading_bot
python -u train_enhanced_clstm_ppo.py \
    --num_episodes 5000 \
    --num_gpus 1 \
    --enable-multi-leg \
    --use-ensemble \
    --num-ensemble-models 3 \
    --checkpoint-dir checkpoints/production_run \
    --resume

# Detach: Ctrl+B, then D
# Close SSH: exit
```

### 4. Monitor Progress

```bash
# Reconnect
ssh -i ~/Downloads/your_key.pem root@162.243.13.8

# Reattach to tmux
tmux attach -t training

# Or view logs
tail -f /root/Trading_bot/logs/training_*.log
```

## 💡 Pro Tips

1. **Always use `python -u`** for unbuffered output
2. **Use tmux** for long-running training
3. **Monitor GPU** with `nvidia-smi` to ensure it's being used
4. **Check logs** if training seems stuck
5. **Use rsync** to sync code efficiently
6. **Commit to git** before deploying to server

## 🆘 Emergency Commands

```bash
# Kill all Python processes
pkill -9 python

# Kill specific training
pkill -9 -f train_enhanced_clstm_ppo.py

# Kill tmux session
tmux kill-session -t training

# Reboot server (last resort)
sudo reboot
```

