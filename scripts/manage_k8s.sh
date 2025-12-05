#!/bin/bash
# Kubernetes Management Script for Paper Trading Bot

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

NAMESPACE="trading-bot"

function show_help() {
    echo "Paper Trading Bot - Kubernetes Management"
    echo "=========================================="
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  status       - Show deployment status"
    echo "  logs         - Tail logs from trading bot"
    echo "  metrics      - Show current metrics"
    echo "  scale [n]    - Scale to n replicas"
    echo "  restart      - Restart the deployment"
    echo "  stop         - Stop the deployment (scale to 0)"
    echo "  start        - Start the deployment (scale to 1)"
    echo "  delete       - Delete all resources"
    echo "  port-forward - Forward ports for local access"
    echo "  exec         - Execute shell in pod"
    echo "  update-model - Update the model checkpoint"
    echo ""
}

function check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}❌ kubectl not found${NC}"
        exit 1
    fi
}

function show_status() {
    echo -e "${GREEN}📊 Deployment Status${NC}"
    echo "===================="
    kubectl get all -n $NAMESPACE
    echo ""
    echo -e "${GREEN}💾 Persistent Volumes${NC}"
    kubectl get pvc -n $NAMESPACE
    echo ""
    echo -e "${GREEN}🔐 Secrets & ConfigMaps${NC}"
    kubectl get secrets,configmaps -n $NAMESPACE
}

function show_logs() {
    echo -e "${GREEN}📜 Tailing logs...${NC}"
    kubectl logs -f deployment/paper-trading-bot -n $NAMESPACE
}

function show_metrics() {
    echo -e "${GREEN}📈 Fetching metrics...${NC}"
    
    # Port forward in background
    kubectl port-forward svc/paper-trading-bot 8000:8000 -n $NAMESPACE &
    PF_PID=$!
    
    # Wait for port forward
    sleep 2
    
    # Fetch metrics
    echo ""
    echo "Status:"
    curl -s http://localhost:8000/status | python3 -m json.tool
    
    echo ""
    echo "Prometheus Metrics:"
    curl -s http://localhost:8000/metrics
    
    # Kill port forward
    kill $PF_PID 2>/dev/null || true
}

function scale_deployment() {
    REPLICAS=$1
    if [ -z "$REPLICAS" ]; then
        echo -e "${RED}❌ Please specify number of replicas${NC}"
        echo "Usage: $0 scale [n]"
        exit 1
    fi
    
    echo -e "${GREEN}⚖️  Scaling to $REPLICAS replicas...${NC}"
    kubectl scale deployment/paper-trading-bot --replicas=$REPLICAS -n $NAMESPACE
    kubectl wait --for=condition=available --timeout=60s \
        deployment/paper-trading-bot -n $NAMESPACE || true
    echo -e "${GREEN}✅ Scaled to $REPLICAS replicas${NC}"
}

function restart_deployment() {
    echo -e "${GREEN}🔄 Restarting deployment...${NC}"
    kubectl rollout restart deployment/paper-trading-bot -n $NAMESPACE
    kubectl rollout status deployment/paper-trading-bot -n $NAMESPACE
    echo -e "${GREEN}✅ Deployment restarted${NC}"
}

function stop_deployment() {
    echo -e "${YELLOW}⏸️  Stopping deployment...${NC}"
    scale_deployment 0
}

function start_deployment() {
    echo -e "${GREEN}▶️  Starting deployment...${NC}"
    scale_deployment 1
}

function delete_all() {
    echo -e "${RED}⚠️  WARNING: This will delete ALL resources in namespace $NAMESPACE${NC}"
    read -p "Are you sure? (yes/no) " -r
    if [[ $REPLY == "yes" ]]; then
        echo -e "${RED}🗑️  Deleting all resources...${NC}"
        kubectl delete namespace $NAMESPACE
        echo -e "${GREEN}✅ All resources deleted${NC}"
    else
        echo "Cancelled"
    fi
}

function port_forward() {
    echo -e "${GREEN}🔌 Port forwarding...${NC}"
    echo "   Trading Bot API: http://localhost:8000"
    echo "   Grafana: http://localhost:3000"
    echo "   Prometheus: http://localhost:9090"
    echo ""
    echo "Press Ctrl+C to stop"
    echo ""
    
    kubectl port-forward svc/paper-trading-bot 8000:8000 -n $NAMESPACE &
    kubectl port-forward svc/grafana 3000:3000 -n $NAMESPACE &
    kubectl port-forward svc/prometheus 9090:9090 -n $NAMESPACE &
    
    wait
}

function exec_shell() {
    POD=$(kubectl get pods -n $NAMESPACE -l app=paper-trading-bot -o jsonpath='{.items[0].metadata.name}')
    echo -e "${GREEN}🐚 Executing shell in pod $POD${NC}"
    kubectl exec -it $POD -n $NAMESPACE -- /bin/bash
}

function update_model() {
    MODEL_PATH=$1
    if [ -z "$MODEL_PATH" ]; then
        MODEL_PATH="checkpoints/enhanced_clstm_ppo/best_composite.pt"
    fi
    
    if [ ! -f "$MODEL_PATH" ]; then
        echo -e "${RED}❌ Model not found at $MODEL_PATH${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}📤 Updating model from $MODEL_PATH${NC}"
    
    POD=$(kubectl get pods -n $NAMESPACE -l app=paper-trading-bot -o jsonpath='{.items[0].metadata.name}')
    kubectl cp "$MODEL_PATH" "$NAMESPACE/$POD:/models/best_composite.pt"
    
    echo -e "${GREEN}✅ Model updated${NC}"
    echo -e "${YELLOW}⚠️  Restart deployment to use new model${NC}"
}

# Main
check_kubectl

case "$1" in
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    metrics)
        show_metrics
        ;;
    scale)
        scale_deployment $2
        ;;
    restart)
        restart_deployment
        ;;
    stop)
        stop_deployment
        ;;
    start)
        start_deployment
        ;;
    delete)
        delete_all
        ;;
    port-forward)
        port_forward
        ;;
    exec)
        exec_shell
        ;;
    update-model)
        update_model $2
        ;;
    *)
        show_help
        ;;
esac

