#!/bin/bash
#
# Quick SSH into server - finds key automatically
#

# Find SSH key
KEY=$(find ~/Downloads ~/.ssh -type f \( -name "*.pem" -o -name "*.key" -o -name "id_rsa" -o -name "id_ed25519" \) 2>/dev/null | head -1)

if [ -z "$KEY" ]; then
    echo "❌ No SSH key found in ~/Downloads or ~/.ssh"
    echo ""
    echo "Please specify the path to your SSH key:"
    read -p "SSH key path: " KEY
fi

if [ ! -f "$KEY" ]; then
    echo "❌ SSH key not found: $KEY"
    exit 1
fi

chmod 600 "$KEY" 2>/dev/null || true

echo "🔑 Using SSH key: $KEY"
echo "🌐 Connecting to root@162.243.13.8..."
echo ""

ssh -i "$KEY" -o StrictHostKeyChecking=no root@162.243.13.8

