#!/bin/bash
#
# Deploy code to remote server and run training
#

set -e

# Server details
SERVER_IP="162.243.13.8"
SERVER_USER="root"
SERVER_DIR="/root/Trading_bot"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Deploy and Run Training${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Find SSH key
echo -e "${YELLOW}🔍 Looking for SSH key...${NC}"

SSH_KEY=""
POSSIBLE_KEYS=(
    "$HOME/Downloads/*.pem"
    "$HOME/Downloads/*.key"
    "$HOME/.ssh/id_rsa"
    "$HOME/.ssh/id_ed25519"
)

for key_pattern in "${POSSIBLE_KEYS[@]}"; do
    for key in $key_pattern; do
        if [ -f "$key" ]; then
            echo -e "${GREEN}✅ Found: $key${NC}"
            SSH_KEY="$key"
            break 2
        fi
    done
done

if [ -z "$SSH_KEY" ]; then
    echo -e "${YELLOW}⚠️  No SSH key found${NC}"
    read -p "Enter SSH key path: " SSH_KEY
fi

chmod 600 "$SSH_KEY" 2>/dev/null || true

echo ""
echo -e "${BLUE}📦 Syncing code to server...${NC}"

# Sync code to server (excluding cache, logs, checkpoints)
rsync -avz --progress \
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
    --exclude 'cache/' \
    --exclude 'logs/' \
    --exclude 'checkpoints/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.git/' \
    ./ "$SERVER_USER@$SERVER_IP:$SERVER_DIR/"

echo ""
echo -e "${GREEN}✅ Code synced!${NC}"
echo ""

# Run training
echo -e "${BLUE}🚀 Starting training on server...${NC}"
echo ""

ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -t "$SERVER_USER@$SERVER_IP" << 'ENDSSH'
cd /root/Trading_bot

echo "========================================="
echo "Server Environment"
echo "========================================="
echo "Hostname: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU')"
echo "Python: $(python --version)"
echo "Working dir: $(pwd)"
echo ""

echo "========================================="
echo "Training Options"
echo "========================================="
echo "1. Quick test (3 symbols, 90 days, 100 episodes)"
echo "2. Production in tmux (23 symbols, 730 days, 5000 episodes)"
echo "3. View existing training (if running)"
echo ""
read -p "Select [1-3]: " OPT

case $OPT in
    1)
        echo ""
        echo "🚀 Starting quick test..."
        echo ""
        python -u train_enhanced_clstm_ppo.py \
            --quick-test \
            --num_gpus 1 \
            --checkpoint-dir checkpoints/test \
            --fresh-start
        ;;
    2)
        echo ""
        echo "🚀 Starting production training in tmux..."
        echo ""
        
        # Kill existing tmux session if exists
        tmux kill-session -t training 2>/dev/null || true
        
        # Start new tmux session
        tmux new -s training -d "python -u train_enhanced_clstm_ppo.py \
            --num_episodes 5000 \
            --num_gpus 1 \
            --enable-multi-leg \
            --use-ensemble \
            --num-ensemble-models 3 \
            --checkpoint-dir checkpoints/production_run \
            --resume"
        
        echo "✅ Training started in tmux session 'training'"
        echo ""
        echo "Attaching to session (Ctrl+B then D to detach)..."
        sleep 2
        tmux attach -t training
        ;;
    3)
        echo ""
        echo "📊 Checking for running training..."
        echo ""
        
        if tmux has-session -t training 2>/dev/null; then
            echo "✅ Found tmux session 'training'"
            echo "Attaching... (Ctrl+B then D to detach)"
            sleep 1
            tmux attach -t training
        else
            echo "⚠️  No tmux session found"
            echo ""
            echo "Checking for running Python processes..."
            ps aux | grep train_enhanced_clstm_ppo.py | grep -v grep || echo "No training process found"
        fi
        ;;
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac
ENDSSH

echo ""
echo -e "${GREEN}✅ Done!${NC}"

