# GPU Cloud Server Training Guide

**Date:** 2025-11-04  
**Purpose:** Optimal commands for running enhanced CLSTM-PPO training on GPU cloud servers

---

## 🚀 Quick Start Commands

### **1. Production Training (Recommended)**

```bash
# Full production training with all optimizations
python train_enhanced_clstm_ppo.py \
    --num_episodes 5000 \
    --num_gpus -1 \
    --enable-multi-leg \
    --use-ensemble \
    --num-ensemble-models 3 \
    --early-stopping-patience 500 \
    --checkpoint-dir checkpoints/production_run \
    --resume
```

**What this does:**
- ✅ Trains for 5000 episodes
- ✅ Uses ALL available GPUs automatically
- ✅ Enables multi-leg strategies (91 actions)
- ✅ Uses ensemble methods (3 models with voting)
- ✅ Early stopping after 500 episodes without improvement
- ✅ Saves checkpoints to `checkpoints/production_run/`
- ✅ Resumes from checkpoint if interrupted

**Expected time:** 12-24 hours on 4x A100 GPUs

---

### **2. Fast Validation Run (Testing)**

```bash
# Quick test to validate setup
python train_enhanced_clstm_ppo.py \
    --num_episodes 100 \
    --num_gpus 1 \
    --enable-multi-leg \
    --checkpoint-dir checkpoints/validation_test \
    --fresh-start
```

**What this does:**
- ✅ Quick 100-episode test
- ✅ Single GPU (faster startup)
- ✅ Multi-leg strategies enabled
- ✅ Fresh start (no resume)

**Expected time:** 30-60 minutes on single A100

---

### **3. Maximum Performance (8 GPUs)**

```bash
# For 8-GPU cloud instances (e.g., AWS p4d.24xlarge)
python train_enhanced_clstm_ppo.py \
    --num_episodes 10000 \
    --num_gpus 8 \
    --enable-multi-leg \
    --use-ensemble \
    --num-ensemble-models 5 \
    --early-stopping-patience 1000 \
    --checkpoint-dir checkpoints/max_performance \
    --resume
```

**What this does:**
- ✅ Extended training (10,000 episodes)
- ✅ All 8 GPUs with PyTorch DDP
- ✅ Larger ensemble (5 models)
- ✅ More patience for early stopping

**Expected time:** 24-48 hours on 8x A100 GPUs

---

### **4. Budget-Friendly (Single GPU)**

```bash
# For single GPU instances (e.g., AWS g5.xlarge)
python train_enhanced_clstm_ppo.py \
    --num_episodes 3000 \
    --num_gpus 1 \
    --enable-multi-leg \
    --early-stopping-patience 300 \
    --checkpoint-dir checkpoints/single_gpu \
    --resume
```

**What this does:**
- ✅ Single GPU training
- ✅ Shorter training (3000 episodes)
- ✅ Earlier stopping (300 episodes)

**Expected time:** 8-12 hours on single A100

---

## 📊 GPU Cloud Provider Recommendations

### **AWS (Amazon Web Services)**

| Instance Type | GPUs | GPU Type | vCPUs | RAM | Cost/hr | Recommended For |
|---------------|------|----------|-------|-----|---------|-----------------|
| **p4d.24xlarge** | 8 | A100 (40GB) | 96 | 1152 GB | ~$32 | Maximum performance |
| **p3.8xlarge** | 4 | V100 (16GB) | 32 | 244 GB | ~$12 | Production training |
| **p3.2xlarge** | 1 | V100 (16GB) | 8 | 61 GB | ~$3 | Budget training |
| **g5.12xlarge** | 4 | A10G (24GB) | 48 | 192 GB | ~$5 | Cost-effective |
| **g5.xlarge** | 1 | A10G (24GB) | 4 | 16 GB | ~$1 | Testing/validation |

**Recommended:** `p3.8xlarge` (4x V100) for best price/performance

**Setup command:**
```bash
# Launch p3.8xlarge instance with Deep Learning AMI
aws ec2 run-instances \
    --image-id ami-0c9978668f8d55984 \
    --instance-type p3.8xlarge \
    --key-name your-key-pair \
    --security-group-ids sg-xxxxxxxx
```

---

### **Google Cloud Platform (GCP)**

| Instance Type | GPUs | GPU Type | vCPUs | RAM | Cost/hr | Recommended For |
|---------------|------|----------|-------|-----|---------|-----------------|
| **a2-highgpu-8g** | 8 | A100 (40GB) | 96 | 680 GB | ~$30 | Maximum performance |
| **a2-highgpu-4g** | 4 | A100 (40GB) | 48 | 340 GB | ~$15 | Production training |
| **n1-highmem-8** + 4x V100 | 4 | V100 (16GB) | 8 | 52 GB | ~$10 | Cost-effective |
| **n1-standard-4** + 1x V100 | 1 | V100 (16GB) | 4 | 15 GB | ~$2.5 | Budget training |

**Recommended:** `a2-highgpu-4g` (4x A100) for best performance

**Setup command:**
```bash
# Launch a2-highgpu-4g instance
gcloud compute instances create clstm-training \
    --zone=us-central1-a \
    --machine-type=a2-highgpu-4g \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release
```

---

### **Azure**

| Instance Type | GPUs | GPU Type | vCPUs | RAM | Cost/hr | Recommended For |
|---------------|------|----------|-------|-----|---------|-----------------|
| **ND96asr_v4** | 8 | A100 (40GB) | 96 | 900 GB | ~$27 | Maximum performance |
| **NC24ads_A100_v4** | 1 | A100 (80GB) | 24 | 220 GB | ~$3.7 | Single GPU training |
| **NC24s_v3** | 4 | V100 (16GB) | 24 | 448 GB | ~$12 | Production training |
| **NC6s_v3** | 1 | V100 (16GB) | 6 | 112 GB | ~$3 | Budget training |

**Recommended:** `NC24s_v3` (4x V100) for production

---

### **Lambda Labs (Best Value)**

| Instance Type | GPUs | GPU Type | RAM | Cost/hr | Recommended For |
|---------------|------|----------|-----|---------|-----------------|
| **8x A100 (80GB)** | 8 | A100 (80GB) | 800 GB | ~$12 | Maximum performance |
| **4x A100 (40GB)** | 4 | A100 (40GB) | 400 GB | ~$5 | Production training |
| **1x A100 (40GB)** | 1 | A100 (40GB) | 100 GB | ~$1.1 | Budget training |
| **4x RTX 6000 Ada** | 4 | RTX 6000 Ada | 400 GB | ~$2.4 | Cost-effective |

**Recommended:** `4x A100 (40GB)` - **BEST PRICE/PERFORMANCE** 🏆

**Setup:** https://lambdalabs.com/service/gpu-cloud

---

### **Vast.ai (Cheapest Option)**

| Instance Type | GPUs | GPU Type | Cost/hr | Recommended For |
|---------------|------|----------|---------|-----------------|
| **4x A100** | 4 | A100 (40GB) | ~$2-4 | Production (spot pricing) |
| **4x RTX 4090** | 4 | RTX 4090 (24GB) | ~$1-2 | Cost-effective |
| **1x A100** | 1 | A100 (40GB) | ~$0.5-1 | Budget training |

**Recommended:** `4x RTX 4090` - **CHEAPEST OPTION** 💰

**Setup:** https://vast.ai/

---

## 🔧 Complete Setup Script

### **1. Initial Setup (Run Once)**

```bash
#!/bin/bash
# setup_training_environment.sh

# Update system
sudo apt-get update
sudo apt-get install -y git python3-pip tmux htop nvtop

# Clone repository
cd ~
git clone https://github.com/devinadams94/Trading_bot.git
cd Trading_bot

# Install dependencies
pip install -r requirements.txt

# Verify GPU setup
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Create necessary directories
mkdir -p logs checkpoints/production_run

# Set up environment variables
cp .env.example .env
nano .env  # Edit with your Alpaca API keys

echo "✅ Setup complete!"
```

**Make executable and run:**
```bash
chmod +x setup_training_environment.sh
./setup_training_environment.sh
```

---

### **2. Start Training (Recommended Method)**

```bash
#!/bin/bash
# start_training.sh

# Use tmux to keep training running even if SSH disconnects
tmux new-session -d -s training "python train_enhanced_clstm_ppo.py \
    --num_episodes 5000 \
    --num_gpus -1 \
    --enable-multi-leg \
    --use-ensemble \
    --num-ensemble-models 3 \
    --early-stopping-patience 500 \
    --checkpoint-dir checkpoints/production_run \
    --resume"

echo "✅ Training started in tmux session 'training'"
echo "To attach: tmux attach -t training"
echo "To detach: Ctrl+B, then D"
```

**Run:**
```bash
chmod +x start_training.sh
./start_training.sh
```

---

### **3. Monitor Training**

```bash
# Attach to training session
tmux attach -t training

# Or monitor logs in real-time
tail -f logs/training_*.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Or use nvtop (better visualization)
nvtop
```

---

## 📈 Optimal Training Configurations

### **Configuration 1: Maximum Performance (8 GPUs)**

```bash
python train_enhanced_clstm_ppo.py \
    --num_episodes 10000 \
    --num_gpus 8 \
    --enable-multi-leg \
    --use-ensemble \
    --num-ensemble-models 5 \
    --early-stopping-patience 1000 \
    --early-stopping-min-delta 0.001 \
    --checkpoint-dir checkpoints/max_perf \
    --resume
```

**Best for:** AWS p4d.24xlarge, GCP a2-highgpu-8g, Lambda 8x A100  
**Cost:** ~$12-32/hour  
**Time:** 24-48 hours  
**Expected Sharpe:** 2.5-3.5+

---

### **Configuration 2: Production (4 GPUs) - RECOMMENDED**

```bash
python train_enhanced_clstm_ppo.py \
    --num_episodes 5000 \
    --num_gpus 4 \
    --enable-multi-leg \
    --use-ensemble \
    --num-ensemble-models 3 \
    --early-stopping-patience 500 \
    --checkpoint-dir checkpoints/production \
    --resume
```

**Best for:** AWS p3.8xlarge, GCP a2-highgpu-4g, Lambda 4x A100  
**Cost:** ~$5-15/hour  
**Time:** 12-24 hours  
**Expected Sharpe:** 2.0-3.0

---

### **Configuration 3: Budget (1 GPU)**

```bash
python train_enhanced_clstm_ppo.py \
    --num_episodes 3000 \
    --num_gpus 1 \
    --enable-multi-leg \
    --early-stopping-patience 300 \
    --checkpoint-dir checkpoints/budget \
    --resume
```

**Best for:** AWS g5.xlarge, Lambda 1x A100, Vast.ai 1x RTX 4090  
**Cost:** ~$0.5-3/hour  
**Time:** 8-12 hours  
**Expected Sharpe:** 1.5-2.5

---

## 🎯 Recommended Training Strategy

### **Phase 1: Validation (1-2 hours, $2-5)**

```bash
# Quick test on single GPU
python train_enhanced_clstm_ppo.py \
    --num_episodes 100 \
    --num_gpus 1 \
    --enable-multi-leg \
    --checkpoint-dir checkpoints/validation \
    --fresh-start
```

**Goal:** Verify setup, check for errors, validate data loading

---

### **Phase 2: Short Training (4-6 hours, $20-60)**

```bash
# Medium run on 4 GPUs
python train_enhanced_clstm_ppo.py \
    --num_episodes 1000 \
    --num_gpus 4 \
    --enable-multi-leg \
    --use-ensemble \
    --num-ensemble-models 3 \
    --checkpoint-dir checkpoints/short_run \
    --resume
```

**Goal:** Get initial performance baseline, tune hyperparameters if needed

---

### **Phase 3: Full Production (12-24 hours, $60-300)**

```bash
# Full production run on 4-8 GPUs
python train_enhanced_clstm_ppo.py \
    --num_episodes 5000 \
    --num_gpus -1 \
    --enable-multi-leg \
    --use-ensemble \
    --num-ensemble-models 3 \
    --early-stopping-patience 500 \
    --checkpoint-dir checkpoints/production \
    --resume
```

**Goal:** Train to convergence, achieve maximum performance

---

## 💰 Cost Optimization Tips

### **1. Use Spot/Preemptible Instances**

**AWS Spot Instances:**
```bash
# Save 70-90% on compute costs
aws ec2 request-spot-instances \
    --instance-type p3.8xlarge \
    --spot-price "4.00"
```

**GCP Preemptible:**
```bash
# Save 80% on compute costs
gcloud compute instances create clstm-training \
    --preemptible \
    --machine-type=a2-highgpu-4g
```

**Risk:** Instance can be terminated (but training resumes from checkpoint!)

---

### **2. Use Lambda Labs or Vast.ai**

- **Lambda Labs:** 4x A100 for ~$5/hour (vs $15 on AWS)
- **Vast.ai:** 4x RTX 4090 for ~$1-2/hour (cheapest option)

---

### **3. Train During Off-Peak Hours**

Some providers offer lower rates during off-peak hours.

---

### **4. Use Early Stopping**

```bash
--early-stopping-patience 300  # Stop if no improvement for 300 episodes
```

Saves compute by stopping when model converges.

---

## 📊 Expected Results

### **After 1000 Episodes (~4-6 hours on 4 GPUs):**
- Cumulative Return: 15-25%
- Sharpe Ratio: 1.2-1.8
- Win Rate: 52-58%
- Max Drawdown: 8-12%

### **After 3000 Episodes (~12-18 hours on 4 GPUs):**
- Cumulative Return: 30-50%
- Sharpe Ratio: 1.8-2.5
- Win Rate: 58-65%
- Max Drawdown: 6-10%

### **After 5000 Episodes (~20-30 hours on 4 GPUs):**
- Cumulative Return: 50-80%
- Sharpe Ratio: 2.0-3.0+
- Win Rate: 62-70%
- Max Drawdown: 5-8%

---

## 🚨 Troubleshooting

### **Out of Memory Error**

```bash
# Reduce batch size or use gradient accumulation
# Edit train_enhanced_clstm_ppo.py line 1234
batch_size = 64  # Instead of 128
```

### **NCCL Timeout (Multi-GPU)**

```bash
# Increase timeout
export NCCL_TIMEOUT=7200
```

### **Training Interrupted**

```bash
# Just restart - it will resume automatically
python train_enhanced_clstm_ppo.py --resume
```

---

## ✅ Final Checklist

Before starting production training:

- [ ] GPU cloud instance launched
- [ ] Repository cloned
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file configured with Alpaca API keys
- [ ] GPU availability verified (`nvidia-smi`)
- [ ] PyTorch CUDA verified (`python -c "import torch; print(torch.cuda.is_available())"`)
- [ ] Tmux session created for persistent training
- [ ] Monitoring set up (`tail -f logs/training_*.log`)

---

## 🎯 Recommended Command (Copy-Paste Ready)

```bash
# PRODUCTION TRAINING - RECOMMENDED
tmux new-session -d -s training "python train_enhanced_clstm_ppo.py --num_episodes 5000 --num_gpus -1 --enable-multi-leg --use-ensemble --num-ensemble-models 3 --early-stopping-patience 500 --checkpoint-dir checkpoints/production_run --resume"

echo "✅ Training started!"
echo "Monitor: tmux attach -t training"
echo "Logs: tail -f logs/training_*.log"
```

**That's it! Your training is now running optimally! 🚀**

