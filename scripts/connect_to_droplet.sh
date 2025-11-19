#!/bin/bash

# Auto-connect to the trading bot droplet
# This script finds the droplet IP and updates SSH config automatically

set -e

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Authenticate with DigitalOcean
export DIGITALOCEAN_ACCESS_TOKEN="$DIGITALOCEAN_TOKEN"
doctl auth init --access-token "$DIGITALOCEAN_TOKEN" &> /dev/null

echo -e "${GREEN}Finding trading bot droplet...${NC}"

# Get the droplet IP
DROPLET_IP=$(doctl compute droplet list --format Name,PublicIPv4 --no-header | grep "trading-bot" | awk '{print $2}' | head -n 1)

if [ -z "$DROPLET_IP" ]; then
    echo -e "${RED}Error: No trading bot droplet found${NC}"
    echo "Run ./scripts/launch_h200_droplet.sh to create one"
    exit 1
fi

echo -e "${GREEN}✓ Found droplet at: $DROPLET_IP${NC}"

# Find SSH private key (prioritize trading bot key)
SSH_KEY_FILE=""
if [ -f ~/.ssh/trading_bot_ed25519 ]; then
    SSH_KEY_FILE=~/.ssh/trading_bot_ed25519
elif [ -f ~/.ssh/do_ed25519 ]; then
    SSH_KEY_FILE=~/.ssh/do_ed25519
elif [ -f ~/.ssh/id_ed25519 ]; then
    SSH_KEY_FILE=~/.ssh/id_ed25519
elif [ -f ~/.ssh/id_rsa ]; then
    SSH_KEY_FILE=~/.ssh/id_rsa
else
    echo -e "${RED}Error: No SSH private key found${NC}"
    echo "Expected ~/.ssh/trading_bot_ed25519, ~/.ssh/do_ed25519, ~/.ssh/id_ed25519 or ~/.ssh/id_rsa"
    exit 1
fi

echo -e "${GREEN}✓ Using SSH key: $SSH_KEY_FILE${NC}"

# Update SSH config
SSH_CONFIG=~/.ssh/config
mkdir -p ~/.ssh
touch "$SSH_CONFIG"

# Remove old entry if exists
sed -i '/Host trading-bot-h200/,/^$/d' "$SSH_CONFIG"

# Add new entry
cat >> "$SSH_CONFIG" << EOF

Host trading-bot-h200
    HostName $DROPLET_IP
    User root
    IdentityFile $SSH_KEY_FILE
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF

echo -e "${GREEN}✓ Updated SSH config${NC}"
echo ""
echo -e "${YELLOW}========================================${NC}"
echo -e "${GREEN}Ready to connect!${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""
echo "Option 1 - VSCode Remote-SSH:"
echo "  1. Press Ctrl+Shift+P"
echo "  2. Type 'Remote-SSH: Connect to Host'"
echo "  3. Select 'trading-bot-h200'"
echo ""
echo "Option 2 - Terminal SSH:"
echo "  ssh trading-bot-h200"
echo ""
echo -e "${YELLOW}Droplet IP: $DROPLET_IP${NC}"

