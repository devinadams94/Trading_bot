# Infrastructure

This directory contains all infrastructure-as-code for deploying the Trading Bot to Kubernetes.

## 📁 Directory Structure

```
infra/
├── k8s/                          # Kubernetes manifests
│   ├── configs/                  # Configuration resources
│   │   ├── namespace.yaml        # Namespace definition
│   │   ├── configmap.yaml        # Application configuration
│   │   └── secrets.yaml          # Secrets (API keys, credentials)
│   │
│   ├── volumes/                  # Persistent storage
│   │   └── persistent-volumes.yaml  # PVCs for models, checkpoints, logs
│   │
│   ├── deployments/              # Application deployments
│   │   └── paper-trading-bot.yaml   # Main trading bot deployment
│   │
│   ├── statefulsets/             # StatefulSets (if needed)
│   │   └── (future: database, message queue)
│   │
│   ├── services/                 # Service definitions
│   │   └── paper-trading-bot.yaml   # Service for trading bot
│   │
│   ├── monitoring/               # Monitoring stack
│   │   ├── prometheus-config.yaml
│   │   ├── prometheus-deployment.yaml
│   │   ├── grafana-config.yaml
│   │   ├── grafana-deployment.yaml
│   │   └── kustomization.yaml
│   │
│   ├── kustomization.yaml        # Kustomize base configuration
│   └── README.md                 # Detailed K8s documentation
│
└── README.md                     # This file

```

## 🚀 Quick Deploy

### Option 1: Using the deployment script (Recommended)

```bash
# From project root
chmod +x scripts/deploy_k8s.sh
./scripts/deploy_k8s.sh
```

### Option 2: Using kubectl directly

```bash
# Deploy in order
kubectl apply -f infra/k8s/configs/namespace.yaml
kubectl apply -f infra/k8s/configs/configmap.yaml
kubectl apply -f infra/k8s/configs/secrets.yaml  # Update with your API keys first!
kubectl apply -f infra/k8s/volumes/persistent-volumes.yaml
kubectl apply -f infra/k8s/deployments/paper-trading-bot.yaml
kubectl apply -f infra/k8s/services/paper-trading-bot.yaml
```

### Option 3: Using Kustomize

```bash
# Deploy everything
kubectl apply -k infra/k8s/

# Deploy with monitoring
kubectl apply -k infra/k8s/
kubectl apply -k infra/k8s/monitoring/
```

## 📊 Deploy Monitoring Stack

```bash
kubectl apply -f infra/k8s/monitoring/prometheus-config.yaml
kubectl apply -f infra/k8s/monitoring/prometheus-deployment.yaml
kubectl apply -f infra/k8s/monitoring/grafana-config.yaml
kubectl apply -f infra/k8s/monitoring/grafana-deployment.yaml
```

Or with Kustomize:

```bash
kubectl apply -k infra/k8s/monitoring/
```

## 🔧 Management

Use the management script:

```bash
chmod +x scripts/manage_k8s.sh

# Show status
./scripts/manage_k8s.sh status

# View logs
./scripts/manage_k8s.sh logs

# Scale deployment
./scripts/manage_k8s.sh scale 3

# Restart
./scripts/manage_k8s.sh restart
```

## 📝 Configuration

### Update Trading Parameters

Edit `infra/k8s/configs/configmap.yaml`:

```yaml
data:
  SYMBOLS: "SPY,QQQ,AAPL,MSFT,NVDA"
  INITIAL_CAPITAL: "100000"
  UPDATE_FREQUENCY: "100"
  SAVE_FREQUENCY: "1000"
```

Apply changes:

```bash
kubectl apply -f infra/k8s/configs/configmap.yaml
kubectl rollout restart deployment/paper-trading-bot -n trading-bot
```

### Update Secrets

**Never commit secrets to git!**

Create from `.env` file:

```bash
kubectl create secret generic trading-bot-secrets \
  --from-env-file=.env \
  --namespace=trading-bot \
  --dry-run=client -o yaml | kubectl apply -f -
```

Or update manually:

```bash
kubectl edit secret trading-bot-secrets -n trading-bot
```

## 🗂️ Resource Organization

### Configs (`configs/`)
- **namespace.yaml**: Defines the `trading-bot` namespace
- **configmap.yaml**: Non-sensitive configuration (symbols, parameters)
- **secrets.yaml**: Sensitive data (API keys, credentials)

### Volumes (`volumes/`)
- **persistent-volumes.yaml**: PVCs for:
  - Models (10Gi) - Pretrained model weights
  - Checkpoints (50Gi) - Online learning checkpoints
  - Logs (20Gi) - Trading logs and metrics

### Deployments (`deployments/`)
- **paper-trading-bot.yaml**: Main trading bot deployment
  - 1 replica (scale up for multiple strategies)
  - GPU support (optional)
  - Health checks and probes
  - Resource limits

### Services (`services/`)
- **paper-trading-bot.yaml**: Exposes trading bot
  - ClusterIP service
  - Port 8000 for metrics and health checks

### Monitoring (`monitoring/`)
- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization dashboards

### StatefulSets (`statefulsets/`)
Reserved for future stateful components:
- PostgreSQL database for trade history
- Redis for caching
- Message queue (RabbitMQ/Kafka)

## 🔍 Troubleshooting

### Check pod status
```bash
kubectl get pods -n trading-bot
kubectl describe pod <pod-name> -n trading-bot
```

### View logs
```bash
kubectl logs -f deployment/paper-trading-bot -n trading-bot
```

### Check events
```bash
kubectl get events -n trading-bot --sort-by='.lastTimestamp'
```

### Exec into pod
```bash
kubectl exec -it deployment/paper-trading-bot -n trading-bot -- /bin/bash
```

## 🗑️ Cleanup

```bash
# Delete everything
kubectl delete namespace trading-bot

# Or use the script
./scripts/manage_k8s.sh delete
```

## 📚 Next Steps

1. Review and customize `configs/configmap.yaml`
2. Add your API keys to `configs/secrets.yaml` or create from `.env`
3. Deploy using `scripts/deploy_k8s.sh`
4. Monitor with Grafana at http://localhost:3000
5. Check metrics at http://localhost:8000/status

For detailed Kubernetes documentation, see [k8s/README.md](k8s/README.md)

