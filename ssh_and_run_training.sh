#!/bin/bash
#
# SSH into remote server and run training with real-time progress
#

set -e

# Server details
SERVER_IP="162.243.13.8"
SERVER_USER="root"
SSH_KEY=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SSH Training Script Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Find SSH key
echo -e "${YELLOW}🔍 Looking for SSH key...${NC}"

# Check common locations
POSSIBLE_KEYS=(
    "$HOME/Downloads/*.pem"
    "$HOME/Downloads/*.key"
    "$HOME/Downloads/id_rsa"
    "$HOME/Downloads/id_ed25519"
    "$HOME/.ssh/id_rsa"
    "$HOME/.ssh/id_ed25519"
    "$HOME/.ssh/digitalocean"
    "$HOME/.ssh/do_key"
)

for key_pattern in "${POSSIBLE_KEYS[@]}"; do
    for key in $key_pattern; do
        if [ -f "$key" ]; then
            echo -e "${GREEN}✅ Found SSH key: $key${NC}"
            SSH_KEY="$key"
            break 2
        fi
    done
done

# If no key found, ask user
if [ -z "$SSH_KEY" ]; then
    echo -e "${YELLOW}⚠️  No SSH key found automatically${NC}"
    echo ""
    echo "Please provide the path to your SSH key:"
    read -p "SSH key path: " SSH_KEY
    
    if [ ! -f "$SSH_KEY" ]; then
        echo -e "${RED}❌ SSH key not found: $SSH_KEY${NC}"
        exit 1
    fi
fi

# Set correct permissions
chmod 600 "$SSH_KEY" 2>/dev/null || true

echo ""
echo -e "${BLUE}📋 Connection Details:${NC}"
echo -e "  Server: ${GREEN}$SERVER_USER@$SERVER_IP${NC}"
echo -e "  SSH Key: ${GREEN}$SSH_KEY${NC}"
echo ""

# Test connection
echo -e "${YELLOW}🧪 Testing SSH connection...${NC}"
if ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$SERVER_USER@$SERVER_IP" "echo 'Connection successful'" 2>/dev/null; then
    echo -e "${GREEN}✅ SSH connection successful!${NC}"
else
    echo -e "${RED}❌ SSH connection failed!${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "1. Check if the server IP is correct: $SERVER_IP"
    echo "2. Check if the SSH key has correct permissions: chmod 600 $SSH_KEY"
    echo "3. Try manual connection: ssh -i $SSH_KEY $SERVER_USER@$SERVER_IP"
    exit 1
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Deployment Options${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "1. Quick test (3 symbols, 90 days, 100 episodes)"
echo "2. Production training (23 symbols, 730 days, 5000 episodes)"
echo "3. Custom command"
echo "4. Just SSH into server (manual)"
echo ""
read -p "Select option [1-4]: " OPTION

case $OPTION in
    1)
        echo -e "${GREEN}🚀 Running quick test...${NC}"
        COMMAND="cd ~/Trading_bot && python -u train_enhanced_clstm_ppo.py --quick-test --num_gpus 1 --checkpoint-dir checkpoints/test --fresh-start"
        ;;
    2)
        echo -e "${GREEN}🚀 Running production training...${NC}"
        COMMAND="cd ~/Trading_bot && tmux new -s training -d 'python -u train_enhanced_clstm_ppo.py --num_episodes 5000 --num_gpus 1 --enable-multi-leg --use-ensemble --num-ensemble-models 3 --checkpoint-dir checkpoints/production_run --resume' && tmux attach -t training"
        ;;
    3)
        echo ""
        read -p "Enter custom command: " CUSTOM_CMD
        COMMAND="cd ~/Trading_bot && $CUSTOM_CMD"
        ;;
    4)
        echo -e "${GREEN}🔌 Connecting to server...${NC}"
        echo ""
        ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$SERVER_USER@$SERVER_IP"
        exit 0
        ;;
    *)
        echo -e "${RED}❌ Invalid option${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Executing Command${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Command:${NC} $COMMAND"
echo ""
echo -e "${YELLOW}⏳ Connecting and running...${NC}"
echo ""

# Execute command on remote server
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -t "$SERVER_USER@$SERVER_IP" "$COMMAND"

echo ""
echo -e "${GREEN}✅ Done!${NC}"

