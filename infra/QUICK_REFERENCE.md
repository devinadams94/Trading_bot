# Kubernetes Paper Trading - Quick Reference

## 🚀 Common Commands

### Deployment

```bash
# Deploy everything
./scripts/deploy_k8s.sh

# Deploy with kubectl
kubectl apply -k infra/k8s/

# Deploy monitoring
kubectl apply -k infra/k8s/monitoring/
```

### Management

```bash
# Status
./scripts/manage_k8s.sh status

# Logs
./scripts/manage_k8s.sh logs

# Metrics
./scripts/manage_k8s.sh metrics

# Scale
./scripts/manage_k8s.sh scale 3

# Restart
./scripts/manage_k8s.sh restart

# Stop
./scripts/manage_k8s.sh stop

# Start
./scripts/manage_k8s.sh start

# Update model
./scripts/manage_k8s.sh update-model path/to/model.pt

# Port forward
./scripts/manage_k8s.sh port-forward

# Shell access
./scripts/manage_k8s.sh exec

# Delete all
./scripts/manage_k8s.sh delete
```

### Manual kubectl Commands

```bash
# Get all resources
kubectl get all -n trading-bot

# Get pods
kubectl get pods -n trading-bot

# Describe pod
kubectl describe pod <pod-name> -n trading-bot

# Logs
kubectl logs -f deployment/paper-trading-bot -n trading-bot

# Exec into pod
kubectl exec -it deployment/paper-trading-bot -n trading-bot -- /bin/bash

# Port forward
kubectl port-forward svc/paper-trading-bot 8000:8000 -n trading-bot

# Scale
kubectl scale deployment/paper-trading-bot --replicas=3 -n trading-bot

# Restart
kubectl rollout restart deployment/paper-trading-bot -n trading-bot

# Check rollout status
kubectl rollout status deployment/paper-trading-bot -n trading-bot

# Edit ConfigMap
kubectl edit configmap trading-bot-config -n trading-bot

# Edit Secret
kubectl edit secret trading-bot-secrets -n trading-bot

# Get events
kubectl get events -n trading-bot --sort-by='.lastTimestamp'

# Delete namespace (cleanup)
kubectl delete namespace trading-bot
```

## 📊 Endpoints

| Endpoint | URL | Description |
|----------|-----|-------------|
| Health | http://localhost:8000/health | Liveness probe |
| Ready | http://localhost:8000/ready | Readiness probe |
| Status | http://localhost:8000/status | Detailed status |
| Metrics | http://localhost:8000/metrics | Prometheus metrics |
| Grafana | http://localhost:3000 | Dashboards (admin/admin) |
| Prometheus | http://localhost:9090 | Metrics UI |

## 📁 File Locations

| Resource | Path |
|----------|------|
| Namespace | `infra/k8s/configs/namespace.yaml` |
| ConfigMap | `infra/k8s/configs/configmap.yaml` |
| Secrets | `infra/k8s/configs/secrets.yaml` |
| Deployment | `infra/k8s/deployments/paper-trading-bot.yaml` |
| Service | `infra/k8s/services/paper-trading-bot.yaml` |
| Volumes | `infra/k8s/volumes/persistent-volumes.yaml` |
| Prometheus | `infra/k8s/monitoring/prometheus-*.yaml` |
| Grafana | `infra/k8s/monitoring/grafana-*.yaml` |

## ⚙️ Configuration

### ConfigMap (infra/k8s/configs/configmap.yaml)

```yaml
SYMBOLS: "SPY,QQQ,AAPL,MSFT,NVDA"
INITIAL_CAPITAL: "100000"
UPDATE_FREQUENCY: "100"
SAVE_FREQUENCY: "1000"
LOG_LEVEL: "INFO"
```

### Secrets (.env file)

```bash
MASSIVE_API_KEY=your_key_here
S3_ACCESS_KEY=your_s3_key
S3_SECRET_KEY=your_s3_secret
```

## 📊 Prometheus Metrics

| Metric | Description |
|--------|-------------|
| `trading_bot_trades_total` | Total trades executed |
| `trading_bot_portfolio_value` | Current portfolio value |
| `trading_bot_win_rate` | Win rate percentage |
| `trading_bot_steps_total` | Total steps taken |
| `trading_bot_reward_total` | Cumulative reward |
| `trading_bot_action_latency_seconds` | Action execution time |

## 🔍 Troubleshooting

### Pod not starting

```bash
kubectl describe pod <pod-name> -n trading-bot
kubectl get events -n trading-bot --sort-by='.lastTimestamp'
```

### Check logs

```bash
kubectl logs -f deployment/paper-trading-bot -n trading-bot
```

### Out of memory

Edit `infra/k8s/deployments/paper-trading-bot.yaml`:
```yaml
resources:
  limits:
    memory: "16Gi"  # Increase
```

### API connection issues

```bash
# Check secret
kubectl get secret trading-bot-secrets -n trading-bot -o yaml

# Update secret
kubectl create secret generic trading-bot-secrets \
  --from-env-file=.env \
  --namespace=trading-bot \
  --dry-run=client -o yaml | kubectl apply -f -
```

## 📚 Documentation

- [infra/README.md](README.md) - Infrastructure overview
- [infra/k8s/README.md](k8s/README.md) - Detailed K8s docs
- [infra/DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Step-by-step guide
- [infra/ARCHITECTURE.md](ARCHITECTURE.md) - Architecture diagrams

## 🎯 Quick Start Checklist

- [ ] Create `.env` file with API key
- [ ] Make scripts executable: `chmod +x scripts/*.sh`
- [ ] Deploy: `./scripts/deploy_k8s.sh`
- [ ] Check status: `./scripts/manage_k8s.sh status`
- [ ] View logs: `./scripts/manage_k8s.sh logs`
- [ ] Port forward: `./scripts/manage_k8s.sh port-forward`
- [ ] Access Grafana: http://localhost:3000
- [ ] Monitor trading: http://localhost:8000/status

