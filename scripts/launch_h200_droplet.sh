#!/bin/bash

# DigitalOcean H200 GPU Droplet Launch Script
# This script creates an H200 GPU droplet and sets it up with Docker + the trading bot image

set -e

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Configuration
DROPLET_NAME="${DROPLET_NAME:-trading-bot-h200}"
REGION="${REGION:-nyc2}"  # H200 available in NYC2
SIZE="${SIZE:-gpu-h200x1-141gb}"  # H200 141GB VRAM, 24 vCPU, 240GB RAM - $3.44/hr
IMAGE="ubuntu-22-04-x64"
SSH_KEY_NAME="${DIGITALOCEAN_SSH_KEY_NAME}"  # From .env
DO_TOKEN="${DIGITALOCEAN_TOKEN}"  # From .env

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}DigitalOcean H200 GPU Droplet Launcher${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if DO_TOKEN is set
if [ -z "$DO_TOKEN" ]; then
    echo -e "${RED}Error: DIGITALOCEAN_TOKEN not set${NC}"
    echo "Add it to your .env file:"
    echo "  DIGITALOCEAN_TOKEN=your_token_here"
    echo ""
    echo "Get your token at: https://cloud.digitalocean.com/account/api/tokens"
    exit 1
fi

# Check if SSH_KEY_NAME is set
if [ -z "$SSH_KEY_NAME" ]; then
    echo -e "${RED}Error: DIGITALOCEAN_SSH_KEY_NAME not set${NC}"
    echo "Add it to your .env file:"
    echo "  DIGITALOCEAN_SSH_KEY_NAME=your-ssh-key-name"
    exit 1
fi

# Check if doctl is installed
if ! command -v doctl &> /dev/null; then
    echo -e "${RED}Error: doctl is not installed${NC}"
    echo "Install it with: brew install doctl (macOS) or snap install doctl (Linux)"
    exit 1
fi

# Authenticate with token
export DIGITALOCEAN_ACCESS_TOKEN="$DO_TOKEN"
doctl auth init --access-token "$DO_TOKEN" &> /dev/null

echo -e "${GREEN}✓ doctl installed and authenticated${NC}"
echo ""

# Generate dedicated SSH key for trading bot if it doesn't exist
TRADING_BOT_KEY=~/.ssh/trading_bot_ed25519
if [ ! -f "$TRADING_BOT_KEY" ]; then
    echo -e "${BLUE}Generating dedicated SSH key for trading bot...${NC}"
    ssh-keygen -t ed25519 -f "$TRADING_BOT_KEY" -N "" -C "trading-bot-auto"
    echo -e "${GREEN}✓ SSH key generated${NC}"
else
    echo -e "${GREEN}✓ Using existing SSH key: $TRADING_BOT_KEY${NC}"
fi

# Read the public key
TRADING_BOT_PUB_KEY=$(cat "${TRADING_BOT_KEY}.pub")
echo ""

# Create droplet
echo -e "${BLUE}Creating H200 GPU droplet...${NC}"
echo "  Name: $DROPLET_NAME"
echo "  Region: $REGION"
echo "  Size: $SIZE"
echo "  Image: $IMAGE"
echo ""

DROPLET_ID=$(doctl compute droplet create "$DROPLET_NAME" \
    --region "$REGION" \
    --size "$SIZE" \
    --image "$IMAGE" \
    --user-data "#cloud-config
ssh_authorized_keys:
  - $TRADING_BOT_PUB_KEY
chpasswd:
  expire: false
" \
    --wait \
    --format ID \
    --no-header)

if [ -z "$DROPLET_ID" ]; then
    echo -e "${RED}Error: Failed to create droplet${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Droplet created with ID: $DROPLET_ID${NC}"
echo ""

# Get droplet IP
echo "Fetching droplet IP address..."
DROPLET_IP=$(doctl compute droplet get "$DROPLET_ID" --format PublicIPv4 --no-header)

echo -e "${GREEN}✓ Droplet IP: $DROPLET_IP${NC}"
echo ""

# Wait for SSH to be ready
echo "Waiting for SSH to be ready..."
for i in {1..30}; do
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@"$DROPLET_IP" "echo 'SSH ready'" &> /dev/null; then
        echo -e "${GREEN}✓ SSH is ready${NC}"
        break
    fi
    echo "  Attempt $i/30..."
    sleep 10
done

echo ""

# Setup script to run on droplet
echo -e "${BLUE}Setting up droplet...${NC}"

ssh -o StrictHostKeyChecking=no root@"$DROPLET_IP" << 'ENDSSH'
set -e

echo "Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
rm get-docker.sh

echo "Installing NVIDIA Container Toolkit..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update
apt-get install -y nvidia-container-toolkit
systemctl restart docker

echo "Pulling trading bot Docker image..."
docker pull sethblakley/clstm-ppo-trading-bot:latest

echo "Creating directories..."
mkdir -p /root/trading-bot/{data,checkpoints,runs}

echo "Testing GPU access..."
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
ENDSSH

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Droplet is ready!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Droplet ID: $DROPLET_ID"
echo "IP Address: $DROPLET_IP"
echo ""
echo "SSH into droplet:"
echo "  ssh root@$DROPLET_IP"
echo ""
echo "Start training:"
echo "  cd /root/trading-bot"
echo "  docker run --gpus all --rm --name training \\"
echo "    -v \$(pwd)/data:/app/data \\"
echo "    -v \$(pwd)/checkpoints:/app/checkpoints \\"
echo "    -v \$(pwd)/runs:/app/runs \\"
echo "    sethblakley/clstm-ppo-trading-bot:latest \\"
echo "    python3 train_enhanced_clstm_ppo.py --episodes 5000"
echo ""

