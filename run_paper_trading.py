#!/usr/bin/env python3
"""
Paper Trading Runner with Kubernetes Support
Runs online learning loop with health checks and metrics
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

# FastAPI for health checks and metrics
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from prometheus_client.core import CollectorRegistry

from src.online_learning import OnlineLearningLoop

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/logs/paper_trading.log')
    ]
)
logger = logging.getLogger(__name__)

# Prometheus metrics
registry = CollectorRegistry()
trades_total = Counter('trading_bot_trades_total', 'Total number of trades executed', registry=registry)
portfolio_value = Gauge('trading_bot_portfolio_value', 'Current portfolio value', registry=registry)
win_rate = Gauge('trading_bot_win_rate', 'Current win rate', registry=registry)
step_count = Counter('trading_bot_steps_total', 'Total number of steps', registry=registry)
reward_total = Counter('trading_bot_reward_total', 'Total reward accumulated', registry=registry)
action_latency = Histogram('trading_bot_action_latency_seconds', 'Time to execute action', registry=registry)

# FastAPI app for health checks
app = FastAPI(title="Paper Trading Bot")

# Global state
learning_loop = None
health_status = {"status": "starting", "last_update": None}


@app.get("/health")
async def health_check():
    """Health check endpoint for Kubernetes liveness probe"""
    if health_status["status"] == "running":
        return JSONResponse({"status": "healthy", "last_update": health_status["last_update"]})
    else:
        return JSONResponse({"status": "unhealthy", "reason": health_status["status"]}, status_code=503)


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint for Kubernetes readiness probe"""
    if learning_loop and learning_loop.running:
        return JSONResponse({"status": "ready"})
    else:
        return JSONResponse({"status": "not_ready"}, status_code=503)


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(registry)


@app.get("/status")
async def status():
    """Detailed status endpoint"""
    if not learning_loop:
        return JSONResponse({"status": "not_initialized"})
    
    current_prices = learning_loop.env._extract_current_prices(
        learning_loop.env.data_stream.get_current_state()
    )
    portfolio_metrics = learning_loop.env.portfolio.get_metrics(current_prices)
    
    return JSONResponse({
        "status": "running" if learning_loop.running else "stopped",
        "step_count": learning_loop.step_count,
        "total_reward": learning_loop.total_reward,
        "portfolio": portfolio_metrics,
        "symbols": learning_loop.symbols,
        "last_update": health_status["last_update"]
    })


async def update_metrics():
    """Update Prometheus metrics periodically"""
    while learning_loop and learning_loop.running:
        try:
            current_prices = learning_loop.env._extract_current_prices(
                learning_loop.env.data_stream.get_current_state()
            )
            metrics = learning_loop.env.portfolio.get_metrics(current_prices)
            
            # Update Prometheus metrics
            portfolio_value.set(metrics['portfolio_value'])
            win_rate.set(metrics['win_rate'])
            trades_total._value.set(metrics['total_trades'])
            step_count._value.set(learning_loop.step_count)
            reward_total._value.set(learning_loop.total_reward)
            
            # Update health status
            health_status["status"] = "running"
            health_status["last_update"] = datetime.utcnow().isoformat()
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
        
        await asyncio.sleep(10)  # Update every 10 seconds


async def run_paper_trading():
    """Main paper trading loop"""
    global learning_loop
    
    # Get configuration from environment
    api_key = os.getenv('MASSIVE_API_KEY')
    if not api_key:
        logger.error("❌ MASSIVE_API_KEY not set in environment")
        sys.exit(1)
    
    symbols = os.getenv('SYMBOLS', 'SPY,QQQ,AAPL').split(',')
    initial_capital = float(os.getenv('INITIAL_CAPITAL', '100000'))
    model_path = os.getenv('MODEL_PATH', '/models/best_composite.pt')
    checkpoint_dir = os.getenv('CHECKPOINT_DIR', '/checkpoints/paper_trading')
    update_frequency = int(os.getenv('UPDATE_FREQUENCY', '100'))
    save_frequency = int(os.getenv('SAVE_FREQUENCY', '1000'))
    
    logger.info("🚀 Starting Paper Trading Bot")
    logger.info(f"   Symbols: {symbols}")
    logger.info(f"   Initial Capital: ${initial_capital:,.2f}")
    logger.info(f"   Model: {model_path}")
    logger.info(f"   Checkpoint Dir: {checkpoint_dir}")
    
    # Check if model exists
    if not Path(model_path).exists():
        logger.warning(f"⚠️ Model not found at {model_path}, starting with random weights")
        model_path = None
    
    # Create online learning loop
    learning_loop = OnlineLearningLoop(
        api_key=api_key,
        symbols=symbols,
        model_path=model_path,
        initial_capital=initial_capital,
        update_frequency=update_frequency,
        save_frequency=save_frequency,
        checkpoint_dir=checkpoint_dir
    )
    
    # Start metrics updater
    asyncio.create_task(update_metrics())
    
    # Run paper trading (indefinitely)
    try:
        await learning_loop.run()
    except Exception as e:
        logger.error(f"❌ Paper trading failed: {e}", exc_info=True)
        health_status["status"] = "failed"
        sys.exit(1)


async def main():
    """Main entry point"""
    # Start FastAPI server in background
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    
    # Run both FastAPI and paper trading
    await asyncio.gather(
        server.serve(),
        run_paper_trading()
    )


if __name__ == "__main__":
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("⚠️ Received shutdown signal")
        if learning_loop:
            learning_loop.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run
    asyncio.run(main())

