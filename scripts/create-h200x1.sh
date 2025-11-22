#!/usr/bin/env bash
set -euo pipefail

# Droplet name (or pass as first arg)
DROPLET_NAME="${1:-h200x1-dev-1}"

# Region and size
REGION="nyc2"
SIZE_SLUG="gpu-h200x1-141gb"
IMAGE_SLUG_OR_ID="ubuntu-24-04-x64"
SSH_KEY_NAME="phuck"
DOCKER_IMAGE="sethblakley/clstm-ppo-trading-bot:latest"
if ! command -v doctl >/dev/null 2>&1; then
  echo "ERROR: doctl not found. Install it and run: doctl auth init"
  exit 1
fi

########################################
# Use existing SSH key in DigitalOcean
########################################

echo "Looking for SSH key '$SSH_KEY_NAME' in DigitalOcean..."
KEY_ID="$(doctl compute ssh-key list --no-header --format Name,ID \
  | awk -v name="$SSH_KEY_NAME" '$1 == name {print $2}')"

if [[ -z "${KEY_ID}" ]]; then
  echo "ERROR: Could not find an SSH key named '$SSH_KEY_NAME' in your DO account."
  echo "Run:  doctl compute ssh-key list"
  echo "Then set SSH_KEY_NAME in this script to one of the existing key names."
  exit 1
fi

echo "Using SSH key '$SSH_KEY_NAME' with ID: $KEY_ID"

USER_DATA=$(cat <<EOF
#!/bin/bash
set -euxo pipefail

# Install Docker
if ! command -v docker >/dev/null 2>&1; then
  apt-get update
  apt-get install -y docker.io
  systemctl enable --now docker
fi

# Pull your Docker Hub image
docker pull "$DOCKER_IMAGE"
EOF
)

echo "Creating droplet '$DROPLET_NAME' in $REGION with size $SIZE_SLUG..."

doctl compute droplet create "$DROPLET_NAME" \
  --region "$REGION" \
  --size "$SIZE_SLUG" \
  --image "$IMAGE_SLUG_OR_ID" \
  --ssh-keys "$KEY_ID" \
  --user-data "$USER_DATA" \
  --tag-name "gpu" \
  --wait \
  --format ID,Name,PublicIPv4

########################################
# Print connection info
########################################

IP="$(doctl compute droplet get "$DROPLET_NAME" --format PublicIPv4 --no-header)"

echo "==================================================="
echo "Droplet ready:"
echo "  Name:  $DROPLET_NAME"
echo "  IP:    $IP"
echo
echo "SSH into it as root with:"
echo "  ssh root@$IP"
echo
echo "On the droplet, your image should be pulled already:"
echo "  docker images | grep clstm-ppo-trading-bot"
echo "==================================================="
