#!/bin/bash
set -e

echo "🚀 Deploying Paper Trading Bot to Kubernetes"
echo "=============================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}❌ kubectl not found. Please install kubectl first.${NC}"
    exit 1
fi

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ docker not found. Please install docker first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"

# Step 1: Build Docker image
echo ""
echo "📦 Step 1: Building Docker image..."
docker build -f Dockerfile.papertrading -t trading-bot:latest .
echo -e "${GREEN}✅ Docker image built${NC}"

# Step 2: Create namespace
echo ""
echo "🏗️  Step 2: Creating Kubernetes namespace..."
kubectl apply -f infra/k8s/configs/namespace.yaml
echo -e "${GREEN}✅ Namespace created${NC}"

# Step 3: Create secrets from .env file
echo ""
echo "🔐 Step 3: Creating secrets..."
if [ -f .env ]; then
    echo -e "${YELLOW}⚠️  Creating secrets from .env file${NC}"
    kubectl create secret generic trading-bot-secrets \
        --from-env-file=.env \
        --namespace=trading-bot \
        --dry-run=client -o yaml | kubectl apply -f -
    echo -e "${GREEN}✅ Secrets created from .env${NC}"
else
    echo -e "${YELLOW}⚠️  .env file not found, using default secrets.yaml${NC}"
    echo -e "${RED}⚠️  IMPORTANT: Update infra/k8s/configs/secrets.yaml with your actual API keys!${NC}"
    kubectl apply -f infra/k8s/configs/secrets.yaml
fi

# Step 4: Create ConfigMap
echo ""
echo "⚙️  Step 4: Creating ConfigMap..."
kubectl apply -f infra/k8s/configs/configmap.yaml
echo -e "${GREEN}✅ ConfigMap created${NC}"

# Step 5: Create Persistent Volumes
echo ""
echo "💾 Step 5: Creating Persistent Volumes..."
kubectl apply -f infra/k8s/volumes/persistent-volumes.yaml
echo -e "${GREEN}✅ Persistent Volumes created${NC}"

# Step 6: Upload pretrained model (if exists)
echo ""
echo "📤 Step 6: Uploading pretrained model..."
MODEL_PATH="checkpoints/enhanced_clstm_ppo/best_composite.pt"
if [ -f "$MODEL_PATH" ]; then
    echo -e "${YELLOW}⚠️  Found model at $MODEL_PATH${NC}"
    
    # Wait for PVC to be bound
    echo "Waiting for models-pvc to be bound..."
    kubectl wait --for=condition=Bound pvc/models-pvc -n trading-bot --timeout=60s
    
    # Create a temporary pod to upload the model
    kubectl run model-uploader --image=busybox --restart=Never -n trading-bot \
        --overrides='
        {
          "spec": {
            "containers": [{
              "name": "model-uploader",
              "image": "busybox",
              "command": ["sleep", "3600"],
              "volumeMounts": [{
                "name": "models",
                "mountPath": "/models"
              }]
            }],
            "volumes": [{
              "name": "models",
              "persistentVolumeClaim": {
                "claimName": "models-pvc"
              }
            }]
          }
        }'
    
    # Wait for pod to be ready
    kubectl wait --for=condition=Ready pod/model-uploader -n trading-bot --timeout=60s
    
    # Copy model to pod
    kubectl cp "$MODEL_PATH" trading-bot/model-uploader:/models/best_composite.pt
    
    # Delete temporary pod
    kubectl delete pod model-uploader -n trading-bot
    
    echo -e "${GREEN}✅ Model uploaded${NC}"
else
    echo -e "${YELLOW}⚠️  No pretrained model found at $MODEL_PATH${NC}"
    echo -e "${YELLOW}   Bot will start with random weights${NC}"
fi

# Step 7: Deploy the application
echo ""
echo "🚢 Step 7: Deploying application..."
kubectl apply -f infra/k8s/deployments/paper-trading-bot.yaml
kubectl apply -f infra/k8s/services/paper-trading-bot.yaml
echo -e "${GREEN}✅ Application deployed${NC}"

# Step 8: Deploy monitoring (optional)
echo ""
read -p "Deploy monitoring stack (Prometheus + Grafana)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📊 Deploying monitoring stack..."
    kubectl apply -f infra/k8s/monitoring/prometheus-config.yaml
    kubectl apply -f infra/k8s/monitoring/prometheus-deployment.yaml
    kubectl apply -f infra/k8s/monitoring/grafana-config.yaml
    kubectl apply -f infra/k8s/monitoring/grafana-deployment.yaml
    echo -e "${GREEN}✅ Monitoring stack deployed${NC}"
fi

# Step 9: Wait for deployment
echo ""
echo "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s \
    deployment/paper-trading-bot -n trading-bot

echo ""
echo -e "${GREEN}🎉 Deployment complete!${NC}"
echo ""
echo "📊 Check status:"
echo "   kubectl get pods -n trading-bot"
echo "   kubectl logs -f deployment/paper-trading-bot -n trading-bot"
echo ""
echo "📈 Access Grafana (if deployed):"
echo "   kubectl port-forward svc/grafana 3000:3000 -n trading-bot"
echo "   Then open: http://localhost:3000 (admin/admin)"
echo ""
echo "🔍 View metrics:"
echo "   kubectl port-forward svc/paper-trading-bot 8000:8000 -n trading-bot"
echo "   Then open: http://localhost:8000/status"

