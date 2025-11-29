# Kubernetes Paper Trading Environment

Production-grade Kubernetes deployment for the CLSTM-PPO Options Trading Bot with online learning capabilities.

## 📁 Directory Structure

```
infra/k8s/
├── configs/                      # Configuration resources
│   ├── namespace.yaml            # Namespace definition
│   ├── configmap.yaml            # Application configuration
│   └── secrets.yaml              # Secrets (API keys, credentials)
│
├── volumes/                      # Persistent storage
│   └── persistent-volumes.yaml   # PVCs for models, checkpoints, logs
│
├── deployments/                  # Application deployments
│   └── paper-trading-bot.yaml    # Main trading bot deployment
│
├── statefulsets/                 # StatefulSets (future use)
│   └── (database, message queue)
│
├── services/                     # Service definitions
│   └── paper-trading-bot.yaml    # Service for trading bot
│
├── monitoring/                   # Monitoring stack
│   ├── prometheus-config.yaml
│   ├── prometheus-deployment.yaml
│   ├── grafana-config.yaml
│   ├── grafana-deployment.yaml
│   └── kustomization.yaml
│
├── kustomization.yaml            # Kustomize base configuration
└── README.md                     # This file
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                        │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Paper Trading Bot Pod                              │    │
│  │  ┌──────────────────────────────────────────────┐  │    │
│  │  │  - Real-time data stream (WebSocket)         │  │    │
│  │  │  - RL Agent (CLSTM-PPO)                      │  │    │
│  │  │  - Online learning loop                      │  │    │
│  │  │  - Portfolio management                      │  │    │
│  │  │  - Health checks & metrics                   │  │    │
│  │  └──────────────────────────────────────────────┘  │    │
│  │                                                      │    │
│  │  Volumes:                                           │    │
│  │  - /models (pretrained weights)                     │    │
│  │  - /checkpoints (online learning saves)             │    │
│  │  - /logs (trading logs)                             │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────┐  ┌────────────────┐                    │
│  │  Prometheus    │  │    Grafana     │                    │
│  │  (Metrics)     │  │  (Dashboard)   │                    │
│  └────────────────┘  └────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Prerequisites

1. **Kubernetes Cluster** (one of):
   - Local: Minikube, Kind, Docker Desktop
   - Cloud: GKE, EKS, AKS
   - On-prem: kubeadm, k3s

2. **Tools**:
   - `kubectl` (v1.20+)
   - `docker` (v20.10+)
   - `bash` (for deployment scripts)

3. **Resources** (minimum per pod):
   - CPU: 2 cores
   - Memory: 4GB
   - Storage: 80GB (models + checkpoints + logs)
   - GPU: Optional (1x NVIDIA GPU for faster training)

4. **API Access**:
   - Massive.com API key (Advanced Options plan)

## 🚀 Quick Start

### 1. Configure Secrets

Create `.env` file in project root with your API credentials:

```bash
# Massive.com API
MASSIVE_API_KEY=your_api_key_here

# Optional: S3 credentials
S3_ACCESS_KEY=your_s3_access_key
S3_SECRET_KEY=your_s3_secret_key
```

### 2. Deploy to Kubernetes

**Option A: Using deployment script (Recommended)**

```bash
# From project root
chmod +x scripts/deploy_k8s.sh scripts/manage_k8s.sh
./scripts/deploy_k8s.sh
```

**Option B: Using kubectl directly**

```bash
# From project root
kubectl apply -f infra/k8s/configs/namespace.yaml
kubectl apply -f infra/k8s/configs/configmap.yaml

# Create secrets from .env
kubectl create secret generic trading-bot-secrets \
  --from-env-file=.env \
  --namespace=trading-bot

kubectl apply -f infra/k8s/volumes/persistent-volumes.yaml
kubectl apply -f infra/k8s/deployments/paper-trading-bot.yaml
kubectl apply -f infra/k8s/services/paper-trading-bot.yaml
```

**Option C: Using Kustomize**

```bash
# From project root
kubectl apply -k infra/k8s/
```

This will:
- ✅ Build Docker image
- ✅ Create namespace
- ✅ Create secrets from `.env`
- ✅ Deploy persistent volumes
- ✅ Upload pretrained model (if exists)
- ✅ Deploy trading bot
- ✅ Deploy monitoring stack (optional)

### 3. Verify Deployment

```bash
# Check status
./scripts/manage_k8s.sh status

# View logs
./scripts/manage_k8s.sh logs

# Check metrics
./scripts/manage_k8s.sh metrics
```

### 4. Access Dashboards

```bash
# Forward ports for local access
./scripts/manage_k8s.sh port-forward
```

Then open:
- **Trading Bot API**: http://localhost:8000/status
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

## 📊 Monitoring

### Prometheus Metrics

The bot exposes the following metrics at `/metrics`:

- `trading_bot_trades_total` - Total trades executed
- `trading_bot_portfolio_value` - Current portfolio value
- `trading_bot_win_rate` - Current win rate (%)
- `trading_bot_steps_total` - Total steps taken
- `trading_bot_reward_total` - Cumulative reward
- `trading_bot_action_latency_seconds` - Action execution time

### Health Checks

- **Liveness**: `/health` - Is the bot running?
- **Readiness**: `/ready` - Is the bot ready to trade?
- **Status**: `/status` - Detailed status with portfolio metrics

## 🔧 Management

### Scale Deployment

```bash
# Scale to 3 replicas (run 3 strategies in parallel)
./scripts/manage_k8s.sh scale 3

# Stop trading (scale to 0)
./scripts/manage_k8s.sh stop

# Resume trading
./scripts/manage_k8s.sh start
```

### Update Model

```bash
# Update with new checkpoint
./scripts/manage_k8s.sh update-model checkpoints/enhanced_clstm_ppo/best_composite.pt

# Restart to use new model
./scripts/manage_k8s.sh restart
```

### View Logs

```bash
# Tail logs
./scripts/manage_k8s.sh logs

# Or use kubectl directly
kubectl logs -f deployment/paper-trading-bot -n trading-bot
```

### Execute Shell in Pod

```bash
./scripts/manage_k8s.sh exec
```

## 🎯 Configuration

Edit `infra/k8s/configs/configmap.yaml` to change:

- **Symbols**: Which stocks/ETFs to trade
- **Capital**: Initial portfolio value
- **Update Frequency**: How often to retrain model
- **Trading Hours**: When to trade (UTC)

Then apply changes:

```bash
kubectl apply -f infra/k8s/configs/configmap.yaml
./scripts/manage_k8s.sh restart
```

## 🔐 Security Best Practices

1. **Never commit secrets** to git
2. **Use Kubernetes secrets** for API keys
3. **Enable RBAC** for pod access control
4. **Use network policies** to restrict traffic
5. **Rotate API keys** regularly
6. **Monitor for anomalies** in Grafana

## 📈 Scaling Strategies

### Horizontal Scaling (Multiple Strategies)

Run different strategies in parallel:

```bash
# Strategy 1: SPY only
kubectl set env deployment/paper-trading-bot SYMBOLS=SPY -n trading-bot

# Create second deployment for QQQ
kubectl create deployment paper-trading-bot-qqq \
  --image=trading-bot:latest \
  --namespace=trading-bot
kubectl set env deployment/paper-trading-bot-qqq SYMBOLS=QQQ -n trading-bot
```

### Vertical Scaling (More Resources)

```yaml
# Edit infra/k8s/deployments/paper-trading-bot.yaml
resources:
  requests:
    memory: "8Gi"
    cpu: "4000m"
  limits:
    memory: "16Gi"
    cpu: "8000m"
    nvidia.com/gpu: "2"  # Use 2 GPUs
```

## 🐛 Troubleshooting

### Pod Not Starting

```bash
# Check pod status
kubectl describe pod -l app=paper-trading-bot -n trading-bot

# Check events
kubectl get events -n trading-bot --sort-by='.lastTimestamp'
```

### Out of Memory

Increase memory limits in `infra/k8s/deployments/paper-trading-bot.yaml`:

```yaml
resources:
  limits:
    memory: "16Gi"  # Increase from 8Gi
```

### API Connection Issues

Check secrets:

```bash
kubectl get secret trading-bot-secrets -n trading-bot -o yaml
```

Verify API key is correct in `.env` and redeploy.

## 🗑️ Cleanup

```bash
# Delete everything
./scripts/manage_k8s.sh delete

# Or manually
kubectl delete namespace trading-bot
```

## 📚 Next Steps

1. **Monitor Performance**: Watch Grafana dashboards for win rate, portfolio value
2. **Tune Hyperparameters**: Adjust update frequency, learning rate in code
3. **Add Strategies**: Deploy multiple bots with different symbols
4. **Set Alerts**: Configure Prometheus alerts for low win rate, high drawdown
5. **Backup Checkpoints**: Regularly backup `/checkpoints` volume

## 🆘 Support

- Check logs: `./scripts/manage_k8s.sh logs`
- View metrics: `./scripts/manage_k8s.sh metrics`
- Exec into pod: `./scripts/manage_k8s.sh exec`

