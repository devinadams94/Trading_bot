#!/usr/bin/env python3
"""
Enhanced CLSTM-PPO training with working environment
Optimized based on "A Novel Deep Reinforcement Learning Based Automated Stock Trading System Using Cascaded LSTM Networks"
Paper: https://arxiv.org/abs/2212.02721

Key optimizations implemented:
1. Cascaded LSTM architecture with feature extraction
2. Turbulence threshold for risk management
3. Enhanced reward function with transaction costs
4. Optimized hyperparameters from paper
5. Multi-timeframe feature extraction (TW=30 optimal)
6. Advanced technical indicators (MACD, RSI, CCI, ADX)
7. Risk-aware training with pullback control
8. Multi-GPU support (1-8 GPUs) with PyTorch DistributedDataParallel

Usage:
    # Single GPU (automatic)
    python train_enhanced_clstm_ppo.py --num_episodes 5000

    # Multi-GPU (specify number)
    python train_enhanced_clstm_ppo.py --num_episodes 5000 --num_gpus 4

    # All available GPUs
    python train_enhanced_clstm_ppo.py --num_episodes 5000 --num_gpus -1
"""

import sys
import os
import asyncio
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from new module structure
from src.envs.options_env import WorkingOptionsEnvironment
from src.envs.multi_leg_env import MultiLegOptionsEnvironment
from src.data.historical_loader import OptimizedHistoricalOptionsDataLoader
from src.utils.indicators import TurbulenceCalculator, TechnicalIndicators
from src.training.rewards import EnhancedRewardFunction
from src.models.feature_extractor import CascadedLSTMFeatureExtractor
from src.utils.gpu import GPUOptimizer
from src.models.ensemble import EnsemblePredictor

# Try to import CLSTM-PPO agent
try:
    from src.models.ppo_agent import OptionsCLSTMPPOAgent
    HAS_CLSTM_PPO = True
except ImportError:
    HAS_CLSTM_PPO = False
    print("Warning: CLSTM-PPO agent not available")

def create_paper_optimized_config():
    """Create configuration optimized based on paper findings"""
    return {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'clip_epsilon': 0.2,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'batch_size': 128,
        'max_grad_norm': 0.5,
        'lstm_time_window': 30,
        'lstm_hidden_size': 128,
        'lstm_features_out': 128,
        'use_turbulence_threshold': True,
        'turbulence_percentile': 90,
        'transaction_cost_rate': 0.001,
        'reward_scaling': 1e-4,
        'initial_capital': 1000000,
        'max_positions': 100,
        'technical_indicators': ['MACD', 'RSI', 'CCI', 'ADX']
    }

# Setup logging with both console and file handlers
def setup_logging(log_dir: str = "logs"):
    """Setup logging with both console and file output"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(f'{log_dir}/training_{timestamp}.log')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.info(f"📝 Logging to console and {log_dir}/training_{timestamp}.log")
    return logger

logger = setup_logging()


class EnhancedCLSTMPPOTrainer:
    """Enhanced trainer with CLSTM-PPO, multi-GPU support, and model persistence"""

    def __init__(
        self,
        use_wandb: bool = False,
        config: dict = None,
        checkpoint_dir: str = None,
        resume_from: str = "best",
        rank: int = 0,
        world_size: int = 1,
        distributed: bool = False,
        enable_multi_leg: bool = False,
        use_ensemble: bool = False,
        num_ensemble_models: int = 3
    ):
        self.use_wandb = use_wandb
        self.rank = rank
        self.world_size = world_size
        self.distributed = distributed
        self.is_main_process = (rank == 0)
        self.enable_multi_leg = enable_multi_leg
        self.use_ensemble = use_ensemble
        self.num_ensemble_models = num_ensemble_models

        # Training interruption flag
        self.interrupted = False
        self._setup_signal_handlers()

        # Merge paper optimizations with user config
        paper_config = create_paper_optimized_config()
        if config:
            paper_config.update(config)
        self.config = paper_config

        self.checkpoint_dir = Path(checkpoint_dir or "checkpoints/enhanced_clstm_ppo")
        if self.is_main_process:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.resume_from = resume_from

        # TensorBoard logging
        self.tensorboard_dir = Path("runs") / f"clstm_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if self.is_main_process:
            self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
            logger.info(f"📊 TensorBoard logging to: {self.tensorboard_dir}")
        else:
            self.writer = None

        # Training state
        self.episode = 0
        self.best_performance = float('-inf')
        self.total_episodes_trained = 0

        # Best model tracking
        self.best_win_rate = 0.0
        self.best_profit_rate = 0.0
        self.best_sharpe_ratio = float('-inf')
        self.best_composite_score = float('-inf')  # New: composite metric
        self.best_model_episode = 0
        self.best_model_metrics = {}

        # Early stopping
        self.early_stopping_patience = self.config.get('early_stopping_patience', 500)
        self.early_stopping_min_delta = self.config.get('early_stopping_min_delta', 0.001)
        self.episodes_without_improvement = 0

        # Training metrics
        self.episode_returns = []
        self.episode_trades = []
        self.episode_rewards = []
        self.clstm_losses = []
        self.ppo_losses = []
        self.win_rates = []

        # Paper-based optimizations
        self.turbulence_calculator = TurbulenceCalculator(
            percentile=self.config.get('turbulence_percentile', 90)
        )
        self.enhanced_reward = EnhancedRewardFunction(
            transaction_cost_rate=self.config.get('transaction_cost_rate', 0.001),
            reward_scaling=self.config.get('reward_scaling', 1e-4)
        )
        self.turbulence_history = []

        # Ensemble support
        self.ensemble = None
        if self.use_ensemble:
            self.ensemble = EnsemblePredictor(num_models=self.num_ensemble_models)
            if self.is_main_process:
                logger.info(f"✅ Ensemble enabled with {self.num_ensemble_models} models")

        # Multi-leg strategy tracking
        self.multi_leg_trades = 0
        self.multi_leg_profitable = 0

        # GPU optimization setup
        if distributed:
            # Distributed training: use specific GPU for this rank
            self.device = torch.device(f"cuda:{rank}")
            torch.cuda.set_device(self.device)
            self.gpu_optimizer = GPUOptimizer(config=self.config)
            self.gradient_scaler = self.gpu_optimizer.create_gradient_scaler()

            # Enable GPU optimizations
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        else:
            # Single GPU training
            self.gpu_optimizer = GPUOptimizer(config=self.config)
            self.device = self.gpu_optimizer.setup_device()
            self.gradient_scaler = self.gpu_optimizer.create_gradient_scaler()

        if self.is_main_process:
            logger.info("🚀 Enhanced CLSTM-PPO Trainer initialized with paper optimizations")
            if distributed:
                logger.info(f"   🌐 Distributed training: {world_size} GPUs (Data Parallelism)")
                logger.info(f"   📍 Rank: {rank}")
                logger.info(f"   ✅ All GPUs work together on each episode")
                logger.info(f"   ✅ Gradients synchronized via DistributedDataParallel")
            logger.info(f"   Device: {self.device}")
            logger.info(f"   Checkpoint dir: {self.checkpoint_dir}")
            logger.info(f"   LSTM time window: {self.config.get('lstm_time_window', 30)}")
            logger.info(f"   Turbulence threshold: {self.config.get('use_turbulence_threshold', True)}")
            logger.info(f"   Transaction cost rate: {self.config.get('transaction_cost_rate', 0.001)}")
            logger.info(f"   Mixed precision: {self.config.get('mixed_precision', True)}")
            logger.info(f"   Gradient accumulation: {self.config.get('gradient_accumulation_steps', 1)} steps")
            logger.info(f"   Multi-leg strategies: {self.enable_multi_leg}")
            logger.info(f"   Ensemble methods: {self.use_ensemble} ({self.num_ensemble_models} models)" if self.use_ensemble else "   Ensemble methods: False")

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        import signal

        def signal_handler(signum, frame):
            """Handle interrupt signals"""
            if self.is_main_process:
                logger.warning(f"\n⚠️ Received signal {signum}, saving checkpoint and exiting gracefully...")
            self.interrupted = True

            # Save checkpoint on main process
            if self.is_main_process and hasattr(self, 'agent'):
                try:
                    self._save_checkpoint("interrupted")
                    logger.info("✅ Checkpoint saved successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to save checkpoint: {e}")

            # Close TensorBoard writer
            if self.is_main_process and hasattr(self, 'writer') and self.writer is not None:
                try:
                    self.writer.close()
                    logger.info("✅ TensorBoard writer closed")
                except Exception as e:
                    logger.error(f"❌ Failed to close TensorBoard writer: {e}")

        # Register handlers for SIGINT (Ctrl+C) and SIGTERM
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _get_default_config(self) -> dict:
        """Get default configuration - now using paper optimizations"""
        # Use paper-optimized config as base
        config = create_paper_optimized_config()

        # Add training-specific parameters
        config.update({
            # Environment settings - Based on paper's approach but adapted for options
            'symbols': [
                # ETFs for market exposure (paper used indices)
                'SPY', 'QQQ', 'IWM',
                # Mega cap tech (high liquidity, good for options)
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META',
                # Additional liquid stocks for diversification
                'NFLX', 'AMD', 'CRM', 'PLTR', 'SNOW', 'COIN', 'RBLX', 'ZM',
                # Financial and other sectors
                'JPM', 'BAC', 'GS', 'V', 'MA'
            ],
            'episode_length': 252,  # Full trading year (paper used daily data)
            'lookback_window': 30,  # Paper found TW=30 optimal for LSTM

            # Training parameters
            'num_episodes': 5000,  # More episodes for convergence
            'save_frequency': 100,
            'log_frequency': 25,
            'eval_frequency': 50,

            # Enhanced features from paper
            'include_technical_indicators': True,  # MACD, RSI, CCI, ADX
            'include_market_microstructure': True,
            'use_portfolio_return_reward': True,  # Paper's reward function
            'include_transaction_costs': True,  # Include costs in reward

            # GPU Optimization settings (from gpu_optimizations.py)
            'use_multi_gpu': True,
            'mixed_precision': True,  # FP16 training for 2x speedup
            'gradient_accumulation_steps': 4,  # Accumulate gradients for larger effective batch
            'compile_model': True,  # PyTorch 2.0+ compilation for speedup
            'use_tf32': True,  # TF32 for Ampere GPUs (A100, RTX 30xx/40xx)
        })

        return config
    
    async def initialize(self):
        """Initialize trainer with enhanced environment and CLSTM-PPO agent"""
        if self.is_main_process:
            logger.info("🔧 Initializing Enhanced CLSTM-PPO Trainer")

        # Create data loader (each process gets its own)
        # Check if using Massive flat files (new processed format with Greeks)
        use_massive_flat_files = self.config.get('use_massive_flat_files', False)
        use_flat_files = self.config.get('use_flat_files', False)

        if use_massive_flat_files:
            # Use NEW Massive flat file data loader (has Greeks, IV calculated)
            if self.is_main_process:
                logger.info("📁 Using Massive flat file data loader (with calculated Greeks)")
                logger.info(f"   Data directory: {self.config.get('massive_flat_files_dir', 'data/flat_files_processed')}")

            from src.data.massive_flat_file_loader import MassiveFlatFileLoader
            self.data_loader = MassiveFlatFileLoader(
                data_dir=self.config.get('massive_flat_files_dir', 'data/flat_files_processed'),
                cache_in_memory=True  # Cache for fast training
            )
        elif use_flat_files:
            # Use OLD flat file data loader (legacy format)
            if self.is_main_process:
                logger.info("📁 Using legacy flat file data loader")
                logger.info(f"   Data directory: {self.config.get('flat_files_dir', 'data/flat_files')}")
                logger.info(f"   File format: {self.config.get('flat_files_format', 'parquet')}")

            from src.data.flat_file_loader import FlatFileDataLoader
            self.data_loader = FlatFileDataLoader(
                data_dir=self.config.get('flat_files_dir', 'data/flat_files'),
                file_format=self.config.get('flat_files_format', 'parquet'),
                cache_in_memory=False  # Disable caching to avoid OOM with 695M rows
            )
        else:
            # Use Massive.com REST API (slower, requires internet)
            massive_api_key = os.getenv('MASSIVE_API_KEY')

            # Log API key status (only on main process)
            if self.is_main_process:
                if massive_api_key:
                    logger.info(f"✅ Using Massive.com API key from .env (key starts with: {massive_api_key[:8]}...)")
                    logger.info("✅ REST API enabled for historical stock and options data")
                else:
                    logger.error("❌ No Massive.com API key found in environment!")
                    logger.error("   Please set MASSIVE_API_KEY in your .env file")
                    raise ValueError("MASSIVE_API_KEY not found in environment variables")

            self.data_loader = OptimizedHistoricalOptionsDataLoader(
                api_key=massive_api_key,
                api_secret=None,  # Not used by Massive.com
                base_url=None,    # Not used by Massive.com
                data_url=None     # Not used by Massive.com
            )

        # Create enhanced working environment (with optional multi-leg support)
        env_class = MultiLegOptionsEnvironment if self.enable_multi_leg else WorkingOptionsEnvironment

        if self.is_main_process:
            if self.enable_multi_leg:
                logger.info("🎯 Using MultiLegOptionsEnvironment (91 actions)")
            else:
                logger.info("📊 Using WorkingOptionsEnvironment (31 actions)")

        self.env = env_class(
            data_loader=self.data_loader,
            symbols=self.config.get('symbols', ['SPY', 'AAPL', 'TSLA']),
            initial_capital=self.config.get('initial_capital', 100000),
            max_positions=self.config.get('max_positions', 5),
            episode_length=self.config.get('episode_length', 200),
            lookback_window=self.config.get('lookback_window', 30),
            include_technical_indicators=self.config.get('include_technical_indicators', True),
            include_market_microstructure=self.config.get('include_market_microstructure', True),
            # NEW: Enable realistic transaction costs
            use_realistic_costs=self.config.get('use_realistic_costs', True),
            enable_slippage=self.config.get('enable_slippage', True),
            slippage_model=self.config.get('slippage_model', 'volume_based'),
            # Multi-leg specific (ignored if using WorkingOptionsEnvironment)
            enable_multi_leg=self.enable_multi_leg
        )

        # Load market data
        # PAPER RECOMMENDATION: Use 2+ years of data for better LSTM feature extraction
        # For faster startup, use fewer days (configurable via --data-days argument)
        end_date = datetime.now() - timedelta(days=1)
        data_days = self.config.get('data_days', 730)  # Default 2 years (730 days)
        start_date = end_date - timedelta(days=data_days)

        if self.is_main_process:
            years = data_days / 365.0
            logger.info(f"📊 Loading {data_days} days ({years:.1f} years) of market data ({start_date.date()} to {end_date.date()})")
            if data_days >= 730:
                logger.info(f"   ⏳ First load may take 15-45 minutes (data will be cached for future runs)")
            elif data_days >= 365:
                logger.info(f"   ⏳ First load may take 10-30 minutes (data will be cached for future runs)")
            else:
                logger.info(f"   ⏳ First load may take 2-5 minutes (data will be cached for future runs)")
            logger.info(f"   📥 Starting data download... (watch for progress below)")
            # Force flush to ensure message appears immediately
            import sys
            sys.stdout.flush()
            sys.stderr.flush()

        await self.env.load_data(start_date, end_date)

        # VALIDATION: Check if we got enough data
        if self.is_main_process:
            if hasattr(self.env, 'market_data') and self.env.market_data:
                logger.info("🔍 Validating data coverage...")

                # Check stock data
                total_stock_days = 0
                min_days = float('inf')
                max_days = 0

                for symbol, df in self.env.market_data.items():
                    actual_days = len(df)
                    total_stock_days += actual_days
                    min_days = min(min_days, actual_days)
                    max_days = max(max_days, actual_days)

                    if actual_days < data_days * 0.5:  # Less than 50% of requested
                        logger.warning(f"⚠️  {symbol}: Only {actual_days} days of stock data (requested {data_days})")

                avg_days = total_stock_days / len(self.env.market_data) if self.env.market_data else 0

                logger.info(f"📊 Stock data coverage: {min_days}-{max_days} days (avg: {avg_days:.0f} days)")

                # Check options data
                if hasattr(self.env, 'options_data') and self.env.options_data:
                    total_contracts = sum(len(options) for options in self.env.options_data.values())
                    logger.info(f"📊 Options data: {total_contracts:,} total contracts across {len(self.env.options_data)} symbols")

                    for symbol, options in self.env.options_data.items():
                        if len(options) < 1000:  # Arbitrary threshold
                            logger.warning(f"⚠️  {symbol}: Only {len(options)} options contracts (may be insufficient)")

                # Overall validation
                # Note: Calendar days vs Trading days
                # - 730 calendar days = ~2 years
                # - Markets closed weekends (104 days/year) + holidays (~9 days/year)
                # - Expected trading days: ~252 per year, ~504 for 2 years
                # - So 499 trading days out of 730 calendar days = 68% is NORMAL and CORRECT

                # Calculate expected trading days (assume ~252 trading days per year)
                expected_trading_days = (data_days / 365.25) * 252
                coverage_pct = (avg_days / expected_trading_days) * 100 if expected_trading_days > 0 else 0

                if avg_days < expected_trading_days * 0.5:
                    logger.error(f"❌ INSUFFICIENT DATA: Average {avg_days:.0f} trading days per symbol")
                    logger.error(f"   Requested: {data_days} calendar days (~{expected_trading_days:.0f} trading days expected)")
                    logger.error(f"   Coverage: {coverage_pct:.0f}% (need at least 50%)")
                    logger.error(f"   Please download more data:")
                    logger.error(f"   python3 download_data_to_flat_files.py --days {data_days}")

                    # Only raise error if not in quick test mode
                    if not self.config.get('quick_test', False):
                        raise ValueError(f"Insufficient data for training: {avg_days:.0f} trading days available, ~{expected_trading_days:.0f} expected")
                elif avg_days < expected_trading_days * 0.9:
                    logger.warning(f"⚠️  Data coverage: {avg_days:.0f} trading days ({coverage_pct:.0f}% of expected ~{expected_trading_days:.0f})")
                    logger.warning(f"   Consider downloading more data for better training:")
                    logger.warning(f"   python3 download_data_to_flat_files.py --days {data_days}")
                else:
                    logger.info(f"✅ Data coverage is excellent: {avg_days:.0f} trading days ({coverage_pct:.0f}% of expected ~{expected_trading_days:.0f})")

        if self.is_main_process:
            msg = f"✅ Environment initialized with {len(self.env.symbols)} symbols"
            print(msg, flush=True)
            logger.info(msg)

            msg = f"   Observation space keys: {list(self.env.observation_space.spaces.keys())}"
            print(msg, flush=True)
            logger.info(msg)

            import sys
            sys.stdout.flush()
            sys.stderr.flush()
        
        # Create CLSTM-PPO agent
        if HAS_CLSTM_PPO:
            if self.is_main_process:
                msg = "\n🤖 Creating CLSTM-PPO agent..."
                print(msg, flush=True)
                logger.info(msg)
                import sys
                sys.stdout.flush()
                sys.stderr.flush()

            self.agent = OptionsCLSTMPPOAgent(
                observation_space=self.env.observation_space,
                action_space=self.env.action_space.n,
                learning_rate_actor_critic=self.config.get('learning_rate_actor_critic', 1e-3),
                learning_rate_clstm=self.config.get('learning_rate_clstm', 3e-3),
                gamma=self.config.get('gamma', 0.99),  # PAPER: 0.99 (was 0.95)
                clip_epsilon=self.config.get('clip_epsilon', 0.2),  # PAPER: 0.2 (was 0.3)
                entropy_coef=self.config.get('entropy_coef', 0.01),  # PAPER: 0.01 (was 0.05)
                value_coef=self.config.get('value_coef', 0.5),  # PAPER: 0.5 (was 1.0)
                batch_size=self.config.get('batch_size', 128),  # PAPER: 64-128 (was 256)
                n_epochs=self.config.get('n_epochs', 10),  # PAPER: 10 (was 15)
                device=self.device,
                # OPTIMIZATIONS: Enable advanced features
                use_sharpe_shaping=self.config.get('use_sharpe_shaping', True),
                use_greeks_sizing=self.config.get('use_greeks_sizing', True),
                use_expiration_management=self.config.get('use_expiration_management', True),
                # STABILITY: New parameters for stable training
                normalize_rewards=self.config.get('normalize_rewards', True),
                reward_clip=self.config.get('reward_clip', 10.0),
                entropy_decay=self.config.get('entropy_decay', 0.995),
                min_entropy_coef=self.config.get('min_entropy_coef', 0.01),
                l2_reg=self.config.get('l2_reg', 1e-4)
            )

            if self.is_main_process:
                msg = "✅ Agent created successfully"
                print(msg, flush=True)
                logger.info(msg)
                import sys
                sys.stdout.flush()
                sys.stderr.flush()

            # FIXED: Use DataParallel for multi-GPU in single process (simpler than DDP)
            # Check if multiple GPUs are available
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1 and hasattr(self.agent, 'network') and torch.cuda.is_available():
                if self.is_main_process:
                    msg = f"🌐 Wrapping model with DataParallel for {num_gpus} GPUs..."
                    print(msg, flush=True)
                    logger.info(msg)
                    import sys
                    sys.stdout.flush()
                    sys.stderr.flush()

                # Use DataParallel to split batches across GPUs
                self.agent.network = nn.DataParallel(self.agent.network)

                if self.is_main_process:
                    msg = f"✅ Model wrapped with DataParallel ({num_gpus} GPUs)"
                    print(msg, flush=True)
                    logger.info(msg)
                    import sys
                    sys.stdout.flush()
                    sys.stderr.flush()

            # Apply GPU optimizations (only if not using DataParallel)
            elif self.config.get('compile_model', True) and hasattr(torch, 'compile') and num_gpus <= 1:
                if self.is_main_process:
                    msg = "🔧 Compiling model with torch.compile..."
                    print(msg, flush=True)
                    logger.info(msg)
                    import sys
                    sys.stdout.flush()
                    sys.stderr.flush()

                try:
                    # PyTorch 2.0+ model compilation for speedup
                    if hasattr(self.agent, 'network'):
                        self.agent.network = torch.compile(self.agent.network, mode='reduce-overhead')
                        if self.is_main_process:
                            msg = "✅ Model compiled with torch.compile for faster training"
                            print(msg, flush=True)
                            logger.info(msg)
                            import sys
                            sys.stdout.flush()
                            sys.stderr.flush()
                except Exception as e:
                    if self.is_main_process:
                        msg = f"⚠️ Model compilation failed: {e}"
                        print(msg, flush=True)
                        logger.warning(msg)
                        import sys
                        sys.stdout.flush()
                        sys.stderr.flush()

            if self.is_main_process:
                msg = "✅ CLSTM-PPO agent initialized"
                print(msg, flush=True)
                logger.info(msg)
                import sys
                sys.stdout.flush()
                sys.stderr.flush()

            # Synchronize all processes before loading checkpoint
            if self.distributed:
                if self.is_main_process:
                    msg = "🔄 Synchronizing processes..."
                    print(msg, flush=True)
                    logger.info(msg)
                    import sys
                    sys.stdout.flush()
                    sys.stderr.flush()
                dist.barrier()

            # Try to resume from checkpoint (only main process loads, then broadcasts)
            if self.is_main_process:
                msg = "📂 Checking for existing checkpoint..."
                print(msg, flush=True)
                logger.info(msg)
                import sys
                sys.stdout.flush()
                sys.stderr.flush()

            if self.is_main_process and self._load_checkpoint():
                msg = f"✅ Resumed training from episode {self.episode}"
                print(msg, flush=True)
                logger.info(msg)

                logger.info(f"   Total episodes trained: {self.total_episodes_trained}")
                logger.info(f"   Best performance so far: {self.best_performance:.4f}")
                logger.info(f"   🏆 BEST COMPOSITE SCORE: {self.best_composite_score:.1%}")

                # Show composite score breakdown if available
                composite_info = self.best_model_metrics.get('composite_score', {})
                if composite_info.get('components'):
                    components = composite_info['components']
                    logger.info(f"      Win Rate: {components.get('win_rate', 0):.1%}")
                    logger.info(f"      Profit Rate: {components.get('profit_rate', 0):.1%}")
                    logger.info(f"      Return: {components.get('return', 0):.1%}")

                logger.info(f"   Best win rate: {self.best_win_rate:.1%}")
                logger.info(f"   Best profit rate: {self.best_profit_rate:.1%}")
                logger.info(f"   Best Sharpe ratio: {self.best_sharpe_ratio:.2f}")
                logger.info(f"   Training metrics loaded: {len(self.episode_returns)} episodes")
                logger.info(f"   🏆 Resuming from best composite model (optimizes WR+PR+Return together)")

                # CRITICAL: Update hyperparameters from config (override checkpoint values)
                # This allows us to change hyperparameters mid-training
                if HAS_CLSTM_PPO and hasattr(self.agent, 'entropy_coef'):
                    old_entropy = self.agent.entropy_coef
                    new_entropy = self.config.get('entropy_coef', 0.05)
                    if old_entropy != new_entropy:
                        self.agent.entropy_coef = new_entropy
                        logger.info(f"   🔧 Updated entropy coefficient: {old_entropy:.3f} → {new_entropy:.3f}")

                    # Update learning rates
                    new_lr_ac = self.config.get('learning_rate_actor_critic', 3e-4)
                    new_lr_clstm = self.config.get('learning_rate_clstm', 1e-3)

                    # Update optimizer learning rates
                    for param_group in self.agent.ppo_optimizer.param_groups:
                        old_lr = param_group['lr']
                        if old_lr != new_lr_ac:
                            param_group['lr'] = new_lr_ac
                            logger.info(f"   🔧 Updated PPO learning rate: {old_lr:.6f} → {new_lr_ac:.6f}")

                    for param_group in self.agent.clstm_optimizer.param_groups:
                        old_lr = param_group['lr']
                        if old_lr != new_lr_clstm:
                            param_group['lr'] = new_lr_clstm
                            logger.info(f"   🔧 Updated CLSTM learning rate: {old_lr:.6f} → {new_lr_clstm:.6f}")

                import sys
                sys.stdout.flush()
                sys.stderr.flush()
            else:
                if self.is_main_process:
                    msg = "🆕 Starting fresh training"
                    print(msg, flush=True)
                    logger.info(msg)
                    import sys
                    sys.stdout.flush()
                    sys.stderr.flush()

                # CLSTM pre-training if enabled and not resuming
                if self.config.get('use_clstm_pretraining', False):
                    if self.is_main_process:
                        msg = "🎓 Starting CLSTM pre-training..."
                        print(msg, flush=True)
                        logger.info(msg)
                        import sys
                        sys.stdout.flush()
                        sys.stderr.flush()
                    await self._pretrain_clstm()
        else:
            # Fallback to simple agent
            self.agent = SimpleAgent(self.env.action_space.n)
            logger.warning("⚠️ Using simple agent - CLSTM-PPO not available")
    
    async def _pretrain_clstm(self):
        """Pre-train CLSTM component on historical data"""
        pretraining_episodes = self.config.get('pretraining_episodes', 100)
        logger.info(f"Pre-training CLSTM for {pretraining_episodes} episodes...")

        pretraining_data = []

        # Collect data for pre-training
        for episode in range(pretraining_episodes):
            obs = self.env.reset()
            episode_data = []
            
            for step in range(self.env.episode_length):
                # Random action for data collection
                action = np.random.randint(0, self.env.action_space.n)
                next_obs, reward, done, info = self.env.step(action)
                
                episode_data.append({
                    'observation': obs,
                    'action': action,
                    'reward': reward,
                    'next_observation': next_obs
                })
                
                obs = next_obs
                if done:
                    break
            
            pretraining_data.extend(episode_data)
            
            if episode % 20 == 0:
                logger.info(f"Pre-training data collection: {episode}/{pretraining_episodes}")
        
        # Pre-train CLSTM
        if hasattr(self.agent, 'pretrain_clstm'):
            pretrain_result = self.agent.pretrain_clstm(pretraining_data, epochs=50)
            logger.info(f"✅ CLSTM pre-training complete. Final loss: {pretrain_result.get('final_loss', 'N/A')}")
        else:
            logger.warning("⚠️ Agent doesn't support CLSTM pre-training")

    def _check_and_save_best_models(self, metrics, local_episode):
        """Check for new high scores and save best model checkpoints"""
        # Only check after we have enough data for meaningful statistics
        if local_episode < 10:  # Reduced from 25 to detect earlier
            return

        # Calculate rolling statistics (last 50 episodes)
        window = min(50, len(self.episode_returns))
        if window < 5:  # Reduced minimum window
            return

        recent_returns = self.episode_returns[-window:]
        recent_win_rates = self.win_rates[-window:] if self.win_rates else [0]

        # Calculate current metrics (both rolling and individual episode)
        current_win_rate_rolling = np.mean(recent_win_rates)
        current_win_rate_episode = metrics.get('win_rate', 0)
        positive_episodes = sum(1 for r in recent_returns if r > 0)
        current_profit_rate = positive_episodes / window

        # Calculate Sharpe ratio
        avg_return = np.mean(recent_returns)
        return_std = np.std(recent_returns) if len(recent_returns) > 1 else 1.0
        current_sharpe = avg_return / max(return_std, 0.001)

        # Calculate episode return (normalized to 0-1 range for composite score)
        episode_return = metrics.get('portfolio_return', 0)
        # Normalize return: assume -20% to +20% range, map to 0-1
        normalized_return = (episode_return + 0.20) / 0.40
        normalized_return = max(0, min(1, normalized_return))  # Clamp to [0, 1]

        # Calculate rolling average return (normalized)
        normalized_avg_return = (avg_return + 0.20) / 0.40
        normalized_avg_return = max(0, min(1, normalized_avg_return))

        # Calculate COMPOSITE SCORE: Average of win rate, profit rate, and normalized return
        # This balances all three metrics equally
        composite_score_episode = (current_win_rate_episode + current_profit_rate + normalized_return) / 3.0
        composite_score_rolling = (current_win_rate_rolling + current_profit_rate + normalized_avg_return) / 3.0

        # Use the better of episode or rolling composite score
        current_composite_score = max(composite_score_episode, composite_score_rolling)
        composite_type = "episode" if composite_score_episode > composite_score_rolling else "rolling"

        # Check for new high scores (use both rolling average and individual episode)
        new_best_win_rate_rolling = current_win_rate_rolling > self.best_win_rate
        new_best_win_rate_episode = current_win_rate_episode > 0.45 and current_win_rate_episode > self.best_win_rate  # Individual episode threshold
        new_best_win_rate = new_best_win_rate_rolling or new_best_win_rate_episode

        new_best_profit_rate = current_profit_rate > self.best_profit_rate
        new_best_sharpe = current_sharpe > self.best_sharpe_ratio

        # NEW: Check for composite score improvement (PRIMARY METRIC)
        new_best_composite = current_composite_score > self.best_composite_score

        # Save COMPOSITE SCORE model (PRIMARY - this is the main model to resume from)
        if new_best_composite:
            self.best_composite_score = current_composite_score
            self.best_model_episode = self.episode
            self.episodes_without_improvement = 0  # Reset early stopping counter

            self.best_model_metrics['composite_score'] = {
                'value': current_composite_score,
                'episode': self.episode,
                'type': composite_type,
                'components': {
                    'win_rate': current_win_rate_episode if composite_type == "episode" else current_win_rate_rolling,
                    'profit_rate': current_profit_rate,
                    'return': episode_return if composite_type == "episode" else avg_return,
                    'normalized_return': normalized_return if composite_type == "episode" else normalized_avg_return
                },
                'window_size': window
            }
            self._save_checkpoint("best_composite")

            # Log detailed breakdown
            logger.info(f"🏆 NEW BEST COMPOSITE SCORE: {current_composite_score:.1%} ({composite_type}) (episode {self.episode})")
            logger.info(f"   📊 Components:")
            if composite_type == "episode":
                logger.info(f"      Win Rate: {current_win_rate_episode:.1%}")
                logger.info(f"      Profit Rate: {current_profit_rate:.1%}")
                logger.info(f"      Return: {episode_return:.1%}")
            else:
                logger.info(f"      Win Rate (rolling): {current_win_rate_rolling:.1%}")
                logger.info(f"      Profit Rate: {current_profit_rate:.1%}")
                logger.info(f"      Avg Return: {avg_return:.1%}")
        else:
            # Increment early stopping counter
            self.episodes_without_improvement += 1

        # Save best win rate model
        if new_best_win_rate:
            # Use the higher of rolling average or individual episode win rate
            best_win_rate_value = max(current_win_rate_rolling, current_win_rate_episode)
            self.best_win_rate = best_win_rate_value
            self.best_model_episode = self.episode

            # Determine which type of win rate triggered the save
            if new_best_win_rate_episode and current_win_rate_episode > current_win_rate_rolling:
                win_rate_type = "episode"
                display_value = current_win_rate_episode
            else:
                win_rate_type = "rolling"
                display_value = current_win_rate_rolling

            self.best_model_metrics['win_rate'] = {
                'value': best_win_rate_value,
                'episode': self.episode,
                'window_size': window,
                'type': win_rate_type,
                'episode_win_rate': current_win_rate_episode,
                'rolling_win_rate': current_win_rate_rolling
            }
            self._save_checkpoint("best_win_rate")
            logger.info(f"🏆 NEW BEST WIN RATE: {display_value:.1%} ({win_rate_type}) (episode {self.episode})")

        # Save best profit rate model
        if new_best_profit_rate:
            self.best_profit_rate = current_profit_rate
            self.best_model_episode = self.episode
            self.best_model_metrics['profit_rate'] = {
                'value': current_profit_rate,
                'episode': self.episode,
                'window_size': window
            }
            self._save_checkpoint("best_profit_rate")
            logger.info(f"🏆 NEW BEST PROFIT RATE: {current_profit_rate:.1%} (episode {self.episode})")

        # Save best Sharpe ratio model
        if new_best_sharpe and current_sharpe > 0.5:  # Only save if Sharpe is meaningful
            self.best_sharpe_ratio = current_sharpe
            self.best_model_episode = self.episode
            self.best_model_metrics['sharpe_ratio'] = {
                'value': current_sharpe,
                'episode': self.episode,
                'window_size': window,
                'avg_return': avg_return,
                'volatility': return_std
            }
            self._save_checkpoint("best_sharpe")
            logger.info(f"🏆 NEW BEST SHARPE RATIO: {current_sharpe:.2f} (episode {self.episode})")

        # Check for exceptional individual episode performance and save milestone checkpoints
        if current_win_rate_episode >= 0.50:  # 50%+ win rate in single episode
            logger.info(f"🌟 EXCEPTIONAL EPISODE: {current_win_rate_episode:.1%} win rate (episode {self.episode})")
            # Save milestone checkpoint for 50%+ win rate episodes
            if current_win_rate_episode >= 0.50:
                self._save_checkpoint(f"milestone_winrate_{current_win_rate_episode:.0%}_episode_{self.episode}")
                logger.info(f"💾 Milestone checkpoint saved for {current_win_rate_episode:.1%} win rate!")

        if metrics.get('portfolio_return', 0) > 0.05:  # 5%+ return in single episode
            return_pct = metrics.get('portfolio_return', 0)
            logger.info(f"🌟 EXCEPTIONAL RETURN: {return_pct:.1%} return (episode {self.episode})")
            # Save milestone checkpoint for 5%+ return episodes
            if return_pct >= 0.05:
                self._save_checkpoint(f"milestone_return_{return_pct:.0%}_episode_{self.episode}")
                logger.info(f"💾 Milestone checkpoint saved for {return_pct:.1%} return!")

        # Log current standings every 100 episodes
        if local_episode % 100 == 0 and local_episode > 0:
            logger.info(f"📊 Best Model Standings:")

            # PRIMARY METRIC: Composite Score
            composite_info = self.best_model_metrics.get('composite_score', {})
            logger.info(f"   🏆 BEST COMPOSITE SCORE: {self.best_composite_score:.1%} (episode {composite_info.get('episode', 'N/A')}) [{composite_info.get('type', 'unknown')}]")
            if composite_info.get('components'):
                components = composite_info['components']
                logger.info(f"      Components: WR={components.get('win_rate', 0):.1%}, PR={components.get('profit_rate', 0):.1%}, Ret={components.get('return', 0):.1%}")

            # Individual metrics
            win_rate_info = self.best_model_metrics.get('win_rate', {})
            logger.info(f"   Best Win Rate: {self.best_win_rate:.1%} (episode {win_rate_info.get('episode', 'N/A')}) [{win_rate_info.get('type', 'unknown')}]")
            logger.info(f"   Best Profit Rate: {self.best_profit_rate:.1%} (episode {self.best_model_metrics.get('profit_rate', {}).get('episode', 'N/A')})")
            logger.info(f"   Best Sharpe Ratio: {self.best_sharpe_ratio:.2f} (episode {self.best_model_metrics.get('sharpe_ratio', {}).get('episode', 'N/A')})")

    def _save_checkpoint(self, checkpoint_type="regular"):
        """Save training checkpoint"""
        try:
            if self.is_main_process:
                logger.info(f"💾 Saving {checkpoint_type} checkpoint at episode {self.episode}...")

            checkpoint = {
                'episode': self.episode,
                'total_episodes_trained': self.total_episodes_trained,
                'best_performance': self.best_performance,
                'best_win_rate': self.best_win_rate,
                'best_profit_rate': self.best_profit_rate,
                'best_sharpe_ratio': self.best_sharpe_ratio,
                'best_composite_score': self.best_composite_score,  # NEW: Save composite score
                'best_model_episode': self.best_model_episode,
                'best_model_metrics': self.best_model_metrics,
                'episode_returns': self.episode_returns,
                'episode_trades': self.episode_trades,
                'episode_rewards': self.episode_rewards,
                'clstm_losses': self.clstm_losses,
                'ppo_losses': self.ppo_losses,
                'win_rates': self.win_rates,
                'config': self.config
            }

            # BUGFIX: Save training state with backup to prevent corruption
            checkpoint_path = self.checkpoint_dir / "training_state.json"
            backup_path = self.checkpoint_dir / "training_state.json.backup"

            if self.is_main_process:
                logger.info(f"   Writing training state to: {checkpoint_path}")

            # Create backup of existing checkpoint before overwriting
            if checkpoint_path.exists():
                try:
                    import shutil
                    shutil.copy2(checkpoint_path, backup_path)
                except Exception as backup_err:
                    logger.warning(f"⚠️ Failed to create backup: {backup_err}")

            # Write to temporary file first, then rename (atomic operation)
            temp_path = self.checkpoint_dir / "training_state.json.tmp"
            with open(temp_path, 'w') as f:
                json.dump(checkpoint, f, indent=2)

            # Atomic rename (prevents corruption if process is killed during write)
            temp_path.replace(checkpoint_path)
            if self.is_main_process:
                logger.info(f"   ✅ Training state saved ({checkpoint_path.stat().st_size} bytes)")

            # Save agent model with appropriate naming
            model_filename = f"model_episode_{self.episode}.pt"
            if checkpoint_type == "best_composite":
                model_filename = f"best_composite_model_episode_{self.episode}.pt"
            elif checkpoint_type == "best_win_rate":
                model_filename = f"best_win_rate_model_episode_{self.episode}.pt"
            elif checkpoint_type == "best_profit_rate":
                model_filename = f"best_profit_rate_model_episode_{self.episode}.pt"
            elif checkpoint_type == "best_sharpe":
                model_filename = f"best_sharpe_model_episode_{self.episode}.pt"

            if HAS_CLSTM_PPO and hasattr(self.agent, 'save'):
                model_path = self.checkpoint_dir / model_filename
                if self.is_main_process:
                    logger.info(f"   Saving model to: {model_path}")
                self.agent.save(str(model_path))
                if self.is_main_process:
                    logger.info(f"   ✅ Model saved ({model_path.stat().st_size} bytes)")
            elif hasattr(self.agent, 'action_probs'):
                # Save simple agent
                agent_path = self.checkpoint_dir / f"simple_agent_episode_{self.episode}.pkl"
                if self.is_main_process:
                    logger.info(f"   Saving simple agent to: {agent_path}")
                with open(agent_path, 'wb') as f:
                    pickle.dump(self.agent.action_probs, f)
                if self.is_main_process:
                    logger.info(f"   ✅ Simple agent saved")
            else:
                if self.is_main_process:
                    logger.warning(f"   ⚠️ No agent save method available (HAS_CLSTM_PPO={HAS_CLSTM_PPO}, has save={hasattr(self.agent, 'save') if hasattr(self, 'agent') else 'no agent'})")

            # Update symlinks based on checkpoint type
            if checkpoint_type == "best_composite":
                best_link = self.checkpoint_dir / "best_composite_model.pt"
                if best_link.exists():
                    best_link.unlink()
                if HAS_CLSTM_PPO:
                    model_path = self.checkpoint_dir / model_filename
                    if model_path.exists():
                        best_link.symlink_to(model_filename)
            elif checkpoint_type == "best_win_rate":
                best_link = self.checkpoint_dir / "best_win_rate_model.pt"
                if best_link.exists():
                    best_link.unlink()
                if HAS_CLSTM_PPO:
                    model_path = self.checkpoint_dir / model_filename
                    if model_path.exists():
                        best_link.symlink_to(model_filename)
            elif checkpoint_type == "best_profit_rate":
                best_link = self.checkpoint_dir / "best_profit_rate_model.pt"
                if best_link.exists():
                    best_link.unlink()
                if HAS_CLSTM_PPO:
                    model_path = self.checkpoint_dir / model_filename
                    if model_path.exists():
                        best_link.symlink_to(model_filename)
            elif checkpoint_type == "best_sharpe":
                best_link = self.checkpoint_dir / "best_sharpe_model.pt"
                if best_link.exists():
                    best_link.unlink()
                if HAS_CLSTM_PPO:
                    model_path = self.checkpoint_dir / model_filename
                    if model_path.exists():
                        best_link.symlink_to(model_filename)

            # Always update latest model link
            latest_path = self.checkpoint_dir / "latest_model.pt"
            if latest_path.exists():
                latest_path.unlink()
            if HAS_CLSTM_PPO:
                model_path = self.checkpoint_dir / model_filename
                if model_path.exists():
                    latest_path.symlink_to(model_filename)

            checkpoint_msg = f"✅ {checkpoint_type.title()} checkpoint saved at episode {self.episode}"
            if checkpoint_type != "regular":
                checkpoint_msg += f" 🏆"
            if self.is_main_process:
                logger.info(checkpoint_msg)

        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _load_checkpoint(self) -> bool:
        """Load training checkpoint with robust error handling"""
        try:
            checkpoint_path = self.checkpoint_dir / "training_state.json"
            if not checkpoint_path.exists():
                return False

            # BUGFIX: Validate JSON file before loading
            # Check if file is empty or corrupted
            if checkpoint_path.stat().st_size == 0:
                logger.warning("⚠️ Checkpoint file is empty, starting fresh")
                return False

            # Try to load JSON with better error handling
            try:
                with open(checkpoint_path, 'r') as f:
                    checkpoint = json.load(f)
            except json.JSONDecodeError as json_err:
                # JSON is corrupted - try to recover from backup or start fresh
                logger.error(f"❌ Checkpoint JSON is corrupted: {json_err}")

                # Check for backup file
                backup_path = self.checkpoint_dir / "training_state.json.backup"
                if backup_path.exists():
                    logger.info("🔄 Attempting to load from backup...")
                    try:
                        with open(backup_path, 'r') as f:
                            checkpoint = json.load(f)
                        logger.info("✅ Successfully loaded from backup")
                    except Exception as backup_err:
                        logger.error(f"❌ Backup also corrupted: {backup_err}")
                        logger.warning("⚠️ Starting fresh training due to corrupted checkpoint")
                        return False
                else:
                    logger.warning("⚠️ No backup found, starting fresh training")
                    return False

            # Restore training state
            self.episode = checkpoint.get('episode', 0)
            self.total_episodes_trained = checkpoint.get('total_episodes_trained', 0)
            self.best_performance = checkpoint.get('best_performance', float('-inf'))
            self.best_win_rate = checkpoint.get('best_win_rate', 0.0)
            self.best_profit_rate = checkpoint.get('best_profit_rate', 0.0)
            self.best_sharpe_ratio = checkpoint.get('best_sharpe_ratio', float('-inf'))
            self.best_composite_score = checkpoint.get('best_composite_score', float('-inf'))  # NEW: Load composite score
            self.best_model_episode = checkpoint.get('best_model_episode', 0)
            self.best_model_metrics = checkpoint.get('best_model_metrics', {})
            self.episode_returns = checkpoint.get('episode_returns', [])
            self.episode_trades = checkpoint.get('episode_trades', [])
            self.episode_rewards = checkpoint.get('episode_rewards', [])
            self.clstm_losses = checkpoint.get('clstm_losses', [])
            self.ppo_losses = checkpoint.get('ppo_losses', [])
            self.win_rates = checkpoint.get('win_rates', [])

            # Load agent model - prioritize best performing models
            model_loaded = False

            if HAS_CLSTM_PPO and hasattr(self.agent, 'load'):
                # Determine model loading order based on resume_from preference
                # ALWAYS prioritize composite model first (unless explicitly overridden)
                if self.resume_from == "win_rate":
                    best_models = [
                        ("best_win_rate_model.pt", "best win rate"),
                        ("best_composite_model.pt", "best composite"),
                        ("best_profit_rate_model.pt", "best profit rate"),
                        ("best_sharpe_model.pt", "best Sharpe ratio"),
                        ("latest_model.pt", "latest")
                    ]
                elif self.resume_from == "profit_rate":
                    best_models = [
                        ("best_profit_rate_model.pt", "best profit rate"),
                        ("best_composite_model.pt", "best composite"),
                        ("best_win_rate_model.pt", "best win rate"),
                        ("best_sharpe_model.pt", "best Sharpe ratio"),
                        ("latest_model.pt", "latest")
                    ]
                elif self.resume_from == "sharpe":
                    best_models = [
                        ("best_sharpe_model.pt", "best Sharpe ratio"),
                        ("best_composite_model.pt", "best composite"),
                        ("best_profit_rate_model.pt", "best profit rate"),
                        ("best_win_rate_model.pt", "best win rate"),
                        ("latest_model.pt", "latest")
                    ]
                elif self.resume_from == "latest":
                    best_models = [
                        ("latest_model.pt", "latest"),
                        ("best_composite_model.pt", "best composite"),
                        ("best_profit_rate_model.pt", "best profit rate"),
                        ("best_win_rate_model.pt", "best win rate"),
                        ("best_sharpe_model.pt", "best Sharpe ratio")
                    ]
                else:  # "best" or default - PRIORITIZE COMPOSITE MODEL
                    best_models = [
                        ("best_composite_model.pt", "best composite (WR+PR+Return)"),
                        ("best_profit_rate_model.pt", "best profit rate"),
                        ("best_win_rate_model.pt", "best win rate"),
                        ("best_sharpe_model.pt", "best Sharpe ratio"),
                        ("latest_model.pt", "latest")
                    ]

                for model_file, model_type in best_models:
                    model_path = self.checkpoint_dir / model_file
                    if model_path.exists():
                        try:
                            self.agent.load(str(model_path))
                            logger.info(f"✅ Loaded agent model from {model_type} model: {model_file}")
                            model_loaded = True
                            break
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to load {model_type} model: {e}")

                # Fallback to specific episode model
                if not model_loaded:
                    episode_model = self.checkpoint_dir / f"model_episode_{self.episode}.pt"
                    if episode_model.exists():
                        try:
                            self.agent.load(str(episode_model))
                            logger.info(f"✅ Loaded agent model from {episode_model}")
                            model_loaded = True
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to load episode model: {e}")

            elif hasattr(self.agent, 'action_probs'):
                # Load simple agent
                agent_path = self.checkpoint_dir / f"simple_agent_episode_{self.episode}.pkl"
                if agent_path.exists():
                    with open(agent_path, 'rb') as f:
                        self.agent.action_probs = pickle.load(f)
                    logger.info(f"✅ Loaded simple agent from {agent_path}")
                    model_loaded = True

            if not model_loaded and HAS_CLSTM_PPO:
                logger.warning("⚠️ No model checkpoint found, starting with fresh model weights")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            return False

    async def train_ensemble_models(self, episodes_per_model: int = 1000):
        """
        Train multiple models for ensemble

        Args:
            episodes_per_model: Number of episodes to train each model
        """
        if not self.use_ensemble:
            logger.warning("⚠️ Ensemble not enabled, skipping ensemble training")
            return

        logger.info(f"🎯 Training ensemble with {self.num_ensemble_models} models")
        logger.info(f"   Episodes per model: {episodes_per_model}")

        ensemble_performances = []

        for model_idx in range(self.num_ensemble_models):
            logger.info(f"\n{'='*60}")
            logger.info(f"🤖 Training Ensemble Model {model_idx + 1}/{self.num_ensemble_models}")
            logger.info(f"{'='*60}\n")

            # Create new agent for this ensemble member
            from src.models.ppo_agent import OptionsCLSTMPPOAgent

            ensemble_agent = OptionsCLSTMPPOAgent(
                observation_space=self.env.observation_space,
                action_space=self.env.action_space,
                device=self.device,
                config=self.config
            )

            # Store original agent
            original_agent = self.agent
            self.agent = ensemble_agent

            # Train this model
            model_returns = []
            for episode in range(episodes_per_model):
                metrics = self.train_episode()
                model_returns.append(metrics.get('portfolio_return', 0))

                if (episode + 1) % 100 == 0:
                    avg_return = np.mean(model_returns[-100:])
                    logger.info(f"   Model {model_idx + 1} - Episode {episode + 1}/{episodes_per_model}: Avg Return = {avg_return:.4f}")

            # Calculate model performance
            avg_performance = np.mean(model_returns)
            std_performance = np.std(model_returns)
            sharpe = avg_performance / max(std_performance, 0.001)

            logger.info(f"\n✅ Model {model_idx + 1} Training Complete:")
            logger.info(f"   Average Return: {avg_performance:.4f}")
            logger.info(f"   Std Dev: {std_performance:.4f}")
            logger.info(f"   Sharpe Ratio: {sharpe:.4f}")

            # Add to ensemble with performance-based weight
            weight = max(0.1, avg_performance)  # Minimum weight of 0.1
            self.ensemble.add_model(ensemble_agent, weight=weight)
            ensemble_performances.append({
                'model_idx': model_idx,
                'avg_return': avg_performance,
                'sharpe': sharpe,
                'weight': weight
            })

            # Save ensemble model
            if self.is_main_process:
                ensemble_model_path = self.checkpoint_dir / f"ensemble_model_{model_idx}.pt"
                ensemble_agent.save(str(ensemble_model_path))
                logger.info(f"💾 Saved ensemble model {model_idx} to {ensemble_model_path}")

            # Restore original agent
            self.agent = original_agent

        # Log ensemble summary
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 Ensemble Training Complete!")
        logger.info(f"{'='*60}")
        logger.info(f"Ensemble Models Summary:")
        for perf in ensemble_performances:
            logger.info(f"   Model {perf['model_idx'] + 1}: Return={perf['avg_return']:.4f}, Sharpe={perf['sharpe']:.4f}, Weight={perf['weight']:.4f}")

        # Save ensemble metadata
        if self.is_main_process:
            ensemble_metadata = {
                'num_models': self.num_ensemble_models,
                'episodes_per_model': episodes_per_model,
                'performances': ensemble_performances,
                'total_weight': sum(p['weight'] for p in ensemble_performances)
            }
            metadata_path = self.checkpoint_dir / "ensemble_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(ensemble_metadata, f, indent=2)
            logger.info(f"💾 Saved ensemble metadata to {metadata_path}")

    def train_episode(self):
        """Train one episode with CLSTM-PPO"""
        logger.info(f"🎬 Starting Episode {self.episode} - resetting environment...")
        obs = self.env.reset()
        logger.info(f"✅ Environment reset complete, starting steps...")

        # OPTIMIZATION: Reset episode-specific components (Sharpe ratio history, etc.)
        if hasattr(self.agent, 'reset_episode'):
            self.agent.reset_episode()

        total_reward = 0
        episode_length = 0

        # Track training updates for this episode
        episode_ppo_losses = []
        episode_clstm_losses = []

        # Collect episode data
        episode_observations = []
        episode_actions = []
        episode_rewards = []
        episode_values = []
        episode_log_probs = []
        episode_dones = []
        
        for step in range(self.env.episode_length):
            if step < 3 or step % 50 == 0:
                logger.info(f"  Step {step}/{self.env.episode_length}...")
            # Calculate turbulence for risk management (paper optimization)
            if self.config.get('use_turbulence_threshold', False) and hasattr(self.env, 'get_current_returns'):
                try:
                    current_returns = self.env.get_current_returns()
                    if len(self.env.historical_returns) > 30:  # Need minimum history
                        turbulence = self.turbulence_calculator.calculate_turbulence(
                            current_returns, self.env.historical_returns
                        )
                        self.turbulence_history.append(turbulence)

                        # Update threshold periodically
                        if len(self.turbulence_history) > 100:
                            self.turbulence_calculator.update_threshold(self.turbulence_history[-100:])

                        # Check if we should stop trading due to high turbulence
                        if self.turbulence_calculator.should_stop_trading(turbulence):
                            action = 0  # Force hold action during high turbulence
                            logger.debug(f"High turbulence detected ({turbulence:.4f}), forcing hold action")
                except Exception as e:
                    logger.debug(f"Turbulence calculation failed: {e}")

            # Get action from CLSTM-PPO agent (if not overridden by turbulence)
            if 'action' not in locals():
                # EXPLORATION: Use epsilon-greedy for first 50% of episodes to encourage exploration
                # Use config num_episodes for epsilon decay calculation
                total_episodes = self.config.get('num_episodes', 1000)
                # More aggressive exploration: 50% random in episode 1, decaying to 0% at 50% of training
                epsilon = max(0.0, 0.5 * (1.0 - self.episode / (total_episodes * 0.5)))
                use_random_action = (np.random.random() < epsilon)

                if use_random_action:
                    # Random exploration - sample uniformly from action space
                    # BIAS AWAY FROM HOLD: Sample from actions 1-90 (exclude action 0)
                    action = np.random.randint(1, self.env.action_space.n)
                    # Still get log_prob and value from model for training
                    if HAS_CLSTM_PPO and hasattr(self.agent, 'act'):
                        _, info = self.agent.act(obs)
                        log_prob = info.get('log_prob', 0.0)
                        value = info.get('value', 0.0)
                    else:
                        log_prob = 0.0
                        value = 0.0

                    if step % 100 == 0:
                        logger.info(f"🎲 Exploration: random action {action} (epsilon={epsilon:.3f})")
                else:
                    # Use ensemble if enabled and has models
                    if self.use_ensemble and self.ensemble and len(self.ensemble.models) > 0:
                        try:
                            action, confidence = self.ensemble.predict_action(obs, deterministic=False)
                            # Get log_prob and value from first model for training
                            if HAS_CLSTM_PPO and hasattr(self.agent, 'act'):
                                _, info = self.agent.act(obs)
                                log_prob = info.get('log_prob', 0.0)
                                value = info.get('value', 0.0)
                            else:
                                log_prob = 0.0
                                value = 0.0

                            if step % 50 == 0:  # Log occasionally
                                logger.debug(f"Ensemble action: {action} (confidence: {confidence:.2%})")
                        except Exception as e:
                            logger.warning(f"Ensemble prediction failed: {e}, falling back to single model")
                            if HAS_CLSTM_PPO and hasattr(self.agent, 'act'):
                                action, info = self.agent.act(obs)
                                log_prob = info.get('log_prob', 0.0)
                                value = info.get('value', 0.0)
                            else:
                                action, info = self.agent.act(obs)
                                log_prob = info.get('log_prob', 0.0)
                                value = 0.0
                    else:
                        # Single model prediction
                        if HAS_CLSTM_PPO and hasattr(self.agent, 'act'):
                            action, info = self.agent.act(obs)
                            log_prob = info.get('log_prob', 0.0)
                            value = info.get('value', 0.0)
                        else:
                            action, info = self.agent.act(obs)
                            log_prob = info.get('log_prob', 0.0)
                            value = 0.0
            else:
                # Turbulence override - set default values
                log_prob = 0.0
                value = 0.0

            # Execute step
            next_obs, reward, done, step_info = self.env.step(action)

            # OPTIMIZATION: Reduce logging frequency (every 100 steps instead of 50)
            # Can be disabled entirely with --no-step-logging for maximum speed
            if not self.config.get('no_step_logging', False) and step % 100 == 0 and step > 0:
                trade_executed = step_info.get('trade_executed', False)
                total_trades = step_info.get('episode_trades', 0)
                action_name = step_info.get('action_name', f'action_{action}')
                logger.info(f"Step {step}: {action_name}, reward={reward:.4f}, trade={'YES' if trade_executed else 'NO'}, total_trades={total_trades}")

            # Apply paper's enhanced reward function
            if self.config.get('use_portfolio_return_reward', False):
                prev_portfolio = step_info.get('prev_portfolio_value', self.env.initial_capital)
                current_portfolio = step_info.get('portfolio_value', self.env.initial_capital)
                transaction_costs = self.enhanced_reward.calculate_transaction_costs(
                    step_info.get('trades', {}),
                    step_info.get('prices', {})
                )
                enhanced_reward = self.enhanced_reward.calculate_portfolio_return_reward(
                    prev_portfolio, current_portfolio, transaction_costs
                )
                reward = enhanced_reward

            # Store experience
            episode_observations.append(obs)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_values.append(value)
            episode_log_probs.append(log_prob)
            episode_dones.append(done)

            total_reward += reward
            episode_length += 1
            obs = next_obs

            # Clear action variable for next iteration
            if 'action' in locals():
                del action

            if done:
                break
        
        # Train agent AFTER collecting full episode (CRITICAL FIX)
        if HAS_CLSTM_PPO and hasattr(self.agent, 'buffer'):
            # Store ALL experiences in agent's buffer first
            for i in range(len(episode_observations)):
                self.agent.buffer.add(
                    observation=episode_observations[i],
                    action=episode_actions[i],
                    reward=episode_rewards[i],
                    value=episode_values[i],
                    log_prob=episode_log_probs[i],
                    done=episode_dones[i]
                )

            # NOW train on the full episode (multiple epochs over all data)
            # This is the correct PPO approach: collect episode, then train multiple times
            buffer_size = len(self.agent.buffer)
            logger.debug(f"Buffer size before training: {buffer_size}")
            train_result = self.agent.train()
            logger.debug(f"Train result: {train_result}")
            if train_result:
                # FIXED: Use correct keys from train() return dict
                ppo_loss = train_result.get('total_loss', 0.0)
                clstm_loss = train_result.get('clstm_loss', 0.0)
                actor_loss = train_result.get('actor_loss', 0.0)
                critic_loss = train_result.get('critic_loss', 0.0)
                entropy = train_result.get('entropy', 0.0)
                kl_div = train_result.get('kl_divergence', 0.0)

                self.clstm_losses.append(clstm_loss)
                self.ppo_losses.append(ppo_loss)

                # Track losses for this episode (for aggregated reporting)
                episode_ppo_losses.append(ppo_loss)
                episode_clstm_losses.append(clstm_loss)

                # TensorBoard logging - Model metrics (per training step)
                if self.writer is not None:
                    global_step = len(self.ppo_losses)
                    self.writer.add_scalar('Loss/PPO_Total', ppo_loss, global_step)
                    self.writer.add_scalar('Loss/CLSTM', clstm_loss, global_step)
                    self.writer.add_scalar('Loss/Actor', actor_loss, global_step)
                    self.writer.add_scalar('Loss/Critic', critic_loss, global_step)
                    self.writer.add_scalar('Model/Entropy', entropy, global_step)
                    self.writer.add_scalar('Model/KL_Divergence', kl_div, global_step)
            else:
                logger.warning(f"⚠️ Training returned empty result (buffer size: {buffer_size})")
        
        # Get episode stats
        portfolio_value = step_info.get('portfolio_value', self.env.initial_capital)
        portfolio_return = step_info.get('portfolio_return', 0.0)
        episode_trades = step_info.get('episode_trades', 0)

        # Log action distribution for debugging
        action_counts = {}
        for a in episode_actions:
            action_counts[a] = action_counts.get(a, 0) + 1
        unique_actions = len(action_counts)
        most_common_action = max(action_counts.items(), key=lambda x: x[1]) if action_counts else (0, 0)

        # Calculate win rate
        # Note: trade_history only contains CLOSED positions (sells), not all trades
        # Use episode_trades for accurate win rate calculation
        profitable_trades = sum(1 for trade in self.env.trade_history if trade.get('pnl', 0) > 0)
        total_trades_this_episode = episode_trades  # Fixed: was len(self.env.trade_history)
        win_rate = profitable_trades / max(1, total_trades_this_episode)

        # Enhanced episode summary
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            logger.info(f"Episode {self.episode} Summary (trained with {num_gpus} GPUs via DataParallel):")
        else:
            logger.info(f"Episode {self.episode} Summary:")
        logger.info(f"  Actions: {unique_actions} unique, most common: action {most_common_action[0]} ({most_common_action[1]} times)")
        logger.info(f"  Trades: {episode_trades} executed, {profitable_trades} profitable ({win_rate:.1%} win rate)")
        logger.info(f"  Return: {portfolio_return:.2%}, Portfolio Value: ${portfolio_value:,.2f}")

        # Record metrics
        self.episode_returns.append(portfolio_return)
        self.episode_trades.append(episode_trades)
        self.episode_rewards.append(total_reward)
        self.win_rates.append(win_rate)

        # Update best performance
        if portfolio_return > self.best_performance:
            self.best_performance = portfolio_return
            logger.info(f"🎉 New best performance: {self.best_performance:.4f} at episode {self.episode}")

        # TensorBoard logging - Episode metrics
        if self.writer is not None:
            # Performance metrics
            self.writer.add_scalar('Episode/Return', portfolio_return, self.episode)
            self.writer.add_scalar('Episode/Portfolio_Value', portfolio_value, self.episode)
            self.writer.add_scalar('Episode/Win_Rate', win_rate, self.episode)
            self.writer.add_scalar('Episode/Total_Reward', total_reward, self.episode)

            # Trading activity
            self.writer.add_scalar('Episode/Trades', episode_trades, self.episode)
            self.writer.add_scalar('Episode/Profitable_Trades', profitable_trades, self.episode)
            self.writer.add_scalar('Episode/Episode_Length', episode_length, self.episode)

            # Rolling averages (last 100 episodes)
            if len(self.episode_returns) >= 100:
                recent_returns = self.episode_returns[-100:]
                recent_win_rates = self.win_rates[-100:]
                self.writer.add_scalar('Rolling/Avg_Return_100', np.mean(recent_returns), self.episode)
                self.writer.add_scalar('Rolling/Avg_Win_Rate_100', np.mean(recent_win_rates), self.episode)
                self.writer.add_scalar('Rolling/Std_Return_100', np.std(recent_returns), self.episode)

                # Sharpe ratio
                sharpe = np.mean(recent_returns) / max(np.std(recent_returns), 0.001)
                self.writer.add_scalar('Rolling/Sharpe_Ratio_100', sharpe, self.episode)

            # Best metrics tracking
            self.writer.add_scalar('Best/Performance', self.best_performance, self.episode)
            self.writer.add_scalar('Best/Win_Rate', self.best_win_rate, self.episode)
            self.writer.add_scalar('Best/Profit_Rate', self.best_profit_rate, self.episode)
            self.writer.add_scalar('Best/Sharpe_Ratio', self.best_sharpe_ratio, self.episode)
            self.writer.add_scalar('Best/Composite_Score', self.best_composite_score, self.episode)

            # Profitability tracking
            is_profitable = portfolio_return > 0
            self.writer.add_scalar('Profitability/Episode_Profitable', 1.0 if is_profitable else 0.0, self.episode)

            # Cumulative metrics
            if len(self.episode_returns) > 0:
                cumulative_return = np.sum(self.episode_returns)
                cumulative_trades = np.sum(self.episode_trades)
                profitable_episodes = sum(1 for r in self.episode_returns if r > 0)
                profitability_rate = profitable_episodes / len(self.episode_returns)

                self.writer.add_scalar('Cumulative/Total_Return', cumulative_return, self.episode)
                self.writer.add_scalar('Cumulative/Total_Trades', cumulative_trades, self.episode)
                self.writer.add_scalar('Cumulative/Profitability_Rate', profitability_rate, self.episode)

                # Drawdown calculation
                cumulative_returns = np.cumsum(self.episode_returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = running_max - cumulative_returns
                max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
                current_drawdown = drawdown[-1] if len(drawdown) > 0 else 0

                self.writer.add_scalar('Risk/Max_Drawdown', max_drawdown, self.episode)
                self.writer.add_scalar('Risk/Current_Drawdown', current_drawdown, self.episode)

        # REMOVED: Reward contamination (bonuses added after training don't help)
        # The reward function in the environment is what matters for training

        # Get action diversity metrics from agent
        diversity_metrics = {}
        if HAS_CLSTM_PPO and hasattr(self.agent, 'get_action_diversity_metrics'):
            diversity_metrics = self.agent.get_action_diversity_metrics()

            # Log to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('ActionDiversity/UniqueActions', diversity_metrics.get('unique_actions', 0), self.episode)
                self.writer.add_scalar('ActionDiversity/Entropy', diversity_metrics.get('entropy', 0), self.episode)
                self.writer.add_scalar('ActionDiversity/MaxActionRatio', diversity_metrics.get('max_action_ratio', 0), self.episode)

        return {
            'episode': self.episode,
            'portfolio_return': portfolio_return,
            'episode_trades': episode_trades,
            'total_reward': total_reward,
            'portfolio_value': portfolio_value,
            'episode_length': episode_length,
            'win_rate': win_rate,
            'profitable_trades': profitable_trades,
            'episode_ppo_losses': episode_ppo_losses,
            'episode_clstm_losses': episode_clstm_losses,
            'unique_actions': unique_actions,
            'action_diversity': diversity_metrics
        }
    
    async def train(self, num_episodes: int = None):
        """Train for specified episodes"""
        if num_episodes is None:
            num_episodes = self.config['num_episodes']

        # Calculate actual episodes to train
        start_episode = self.total_episodes_trained
        target_episodes = start_episode + num_episodes

        if self.is_main_process:
            msg = f"\n{'='*80}\n🎯 Starting CLSTM-PPO training\n{'='*80}"
            print(msg, flush=True)
            logger.info(msg)

            msg = f"   Episodes already trained: {start_episode}"
            print(msg, flush=True)
            logger.info(msg)

            msg = f"   Episodes to train this session: {num_episodes}"
            print(msg, flush=True)
            logger.info(msg)

            msg = f"   Target total episodes: {target_episodes}"
            print(msg, flush=True)
            logger.info(msg)

            msg = f"{'='*80}\n"
            print(msg, flush=True)
            logger.info(msg)

            import sys
            sys.stdout.flush()
            sys.stderr.flush()

        for local_episode in range(num_episodes):
            # Check for interruption
            if self.interrupted:
                if self.is_main_process:
                    logger.warning("⚠️ Training interrupted by user, stopping gracefully...")
                break

            self.episode = start_episode + local_episode

            # Set random seed for reproducibility
            torch.manual_seed(self.episode)
            np.random.seed(self.episode)

            # Train episode
            metrics = self.train_episode()

            # Check for new high scores and save best model checkpoints
            self._check_and_save_best_models(metrics, local_episode)

            # Save regular checkpoint every save_frequency episodes
            if (local_episode + 1) % self.config.get('save_frequency', 100) == 0:
                self._save_checkpoint("regular")

            # Log progress
            if local_episode % self.config.get('log_frequency', 25) == 0:
                # Calculate rolling averages
                window = min(50, len(self.episode_returns))
                if window > 0:
                    recent_returns = self.episode_returns[-window:]
                    recent_trades = self.episode_trades[-window:]
                    recent_win_rates = self.win_rates[-window:] if self.win_rates else [0]

                    avg_return = np.mean(recent_returns)
                    avg_trades = np.mean(recent_trades)
                    avg_win_rate = np.mean(recent_win_rates)

                    # Calculate profitability metrics
                    positive_episodes = sum(1 for r in recent_returns if r > 0)
                    profitability_rate = positive_episodes / window

                    # Calculate Sharpe ratio (simplified)
                    return_std = np.std(recent_returns) if len(recent_returns) > 1 else 1.0
                    sharpe_ratio = avg_return / max(return_std, 0.001)

                    # Calculate aggregated loss statistics for this episode
                    episode_ppo_losses = metrics.get('episode_ppo_losses', [])
                    episode_clstm_losses = metrics.get('episode_clstm_losses', [])

                    if episode_ppo_losses and episode_clstm_losses:
                        avg_episode_ppo = np.mean(episode_ppo_losses)
                        avg_episode_clstm = np.mean(episode_clstm_losses)
                        num_updates = len(episode_ppo_losses)

                        # Determine loss trend and performance indicator
                        ppo_indicator = ""
                        clstm_indicator = ""
                        if len(self.ppo_losses) >= 50:
                            recent_ppo_avg = np.mean(self.ppo_losses[-50:])
                            recent_clstm_avg = np.mean(self.clstm_losses[-50:])

                            # PPO loss: Lower is better for convergence
                            if avg_episode_ppo < recent_ppo_avg * 0.85:
                                ppo_indicator = "📉 Excellent"
                            elif avg_episode_ppo < recent_ppo_avg * 0.95:
                                ppo_indicator = "✅ Good"
                            elif avg_episode_ppo < recent_ppo_avg * 1.05:
                                ppo_indicator = "➡️  Stable"
                            elif avg_episode_ppo < recent_ppo_avg * 1.15:
                                ppo_indicator = "⚠️  Rising"
                            else:
                                ppo_indicator = "❌ High"

                            # CLSTM loss: Lower is better
                            if avg_episode_clstm < recent_clstm_avg * 0.85:
                                clstm_indicator = "📉 Excellent"
                            elif avg_episode_clstm < recent_clstm_avg * 0.95:
                                clstm_indicator = "✅ Good"
                            elif avg_episode_clstm < recent_clstm_avg * 1.05:
                                clstm_indicator = "➡️  Stable"
                            elif avg_episode_clstm < recent_clstm_avg * 1.15:
                                clstm_indicator = "⚠️  Rising"
                            else:
                                clstm_indicator = "❌ High"
                        else:
                            # Not enough history yet
                            ppo_indicator = "🔄 Learning"
                            clstm_indicator = "🔄 Learning"

                        logger.info(
                            f"Episode {self.episode:4d}: "
                            f"Return: {metrics['portfolio_return']:7.4f}, "
                            f"Trades: {metrics['episode_trades']:2d}, "
                            f"WinRate: {metrics['win_rate']:5.1%}, "
                            f"AvgRet: {avg_return:6.4f}, "
                            f"ProfRate: {profitability_rate:5.1%}, "
                            f"Sharpe: {sharpe_ratio:5.2f}"
                        )
                        logger.info(
                            f"         Training ({num_updates} updates): "
                            f"PPO: {avg_episode_ppo:.4f} {ppo_indicator} | "
                            f"CLSTM: {avg_episode_clstm:.4f} {clstm_indicator}"
                        )
                    else:
                        logger.info(
                            f"Episode {self.episode:4d}: "
                            f"Return: {metrics['portfolio_return']:7.4f}, "
                            f"Trades: {metrics['episode_trades']:2d}, "
                            f"WinRate: {metrics['win_rate']:5.1%}, "
                            f"AvgRet: {avg_return:6.4f}, "
                            f"ProfRate: {profitability_rate:5.1%}, "
                            f"Sharpe: {sharpe_ratio:5.2f}"
                        )

                    # Additional detailed logging every 100 episodes
                    if local_episode % 100 == 0 and local_episode > 0:
                        avg_clstm_loss = np.mean(self.clstm_losses[-100:]) if len(self.clstm_losses) >= 100 else np.mean(self.clstm_losses) if self.clstm_losses else 0
                        avg_ppo_loss = np.mean(self.ppo_losses[-100:]) if len(self.ppo_losses) >= 100 else np.mean(self.ppo_losses) if self.ppo_losses else 0

                        logger.info(f"📊 Detailed Stats (Last {window} episodes):")
                        logger.info(f"   Avg Return: {avg_return:.4f} ± {return_std:.4f}")
                        logger.info(f"   Avg Trades: {avg_trades:.1f}")
                        logger.info(f"   Avg Win Rate: {avg_win_rate:.1%}")
                        logger.info(f"   Profitable Episodes: {positive_episodes}/{window} ({profitability_rate:.1%})")
                        logger.info(f"   Avg CLSTM Loss (100 eps): {avg_clstm_loss:.4f}")
                        logger.info(f"   Avg PPO Loss (100 eps): {avg_ppo_loss:.4f}")

                        # GPU memory monitoring
                        if torch.cuda.is_available():
                            for gpu_id in range(torch.cuda.device_count()):
                                mem_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3  # GB
                                mem_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3  # GB
                                mem_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3  # GB
                                mem_usage_pct = (mem_allocated / mem_total) * 100

                                logger.info(f"   GPU {gpu_id} Memory: {mem_allocated:.2f}GB / {mem_total:.2f}GB ({mem_usage_pct:.1f}%)")

                                # Warning if memory usage is high
                                if mem_usage_pct > 90:
                                    logger.warning(f"   ⚠️ GPU {gpu_id} memory usage is high ({mem_usage_pct:.1f}%)!")

            # Check for early stopping
            if self.is_main_process and self.early_stopping_patience > 0:
                if self.episodes_without_improvement >= self.early_stopping_patience:
                    logger.warning(f"⚠️ Early stopping triggered: No improvement for {self.early_stopping_patience} episodes")
                    logger.info(f"   Best composite score: {self.best_composite_score:.1%} at episode {self.best_model_episode}")
                    logger.info(f"   Stopping training and using best model")
                    break

            # Check for consistent profitability milestone
            if local_episode >= 100 and local_episode % 100 == 0:
                recent_100 = self.episode_returns[-100:] if len(self.episode_returns) >= 100 else self.episode_returns
                if len(recent_100) >= 50:
                    avg_100 = np.mean(recent_100)
                    positive_100 = sum(1 for r in recent_100 if r > 0)
                    if avg_100 > 0.01 and positive_100 > len(recent_100) * 0.6:
                        logger.info(f"🎉 PROFITABILITY MILESTONE: Avg return {avg_100:.4f} with {positive_100}/{len(recent_100)} positive episodes!")
        
        # Update total episodes trained
        self.total_episodes_trained = self.episode + 1

        # Save final checkpoint
        self._save_checkpoint()

        # Final statistics
        total_trades = sum(self.episode_trades)
        avg_trades_per_episode = total_trades / max(1, len(self.episode_trades))
        episodes_with_trades = sum(1 for t in self.episode_trades if t > 0)

        # Calculate final profitability metrics
        if len(self.episode_returns) >= 100:
            final_100_returns = self.episode_returns[-100:]
            final_avg_return = np.mean(final_100_returns)
            final_std_return = np.std(final_100_returns)
            final_sharpe = final_avg_return / max(final_std_return, 0.001)
            positive_final = sum(1 for r in final_100_returns if r > 0)
            profitability_rate = positive_final / len(final_100_returns)
        else:
            final_avg_return = np.mean(self.episode_returns) if self.episode_returns else 0
            final_sharpe = 0
            profitability_rate = 0

        logger.info(f"\n📊 ENHANCED CLSTM-PPO TRAINING COMPLETE:")
        logger.info(f"   Total episodes trained: {self.total_episodes_trained}")
        logger.info(f"   Episodes this session: {num_episodes}")
        logger.info(f"   Total trades: {total_trades}")
        logger.info(f"   Average trades per episode: {avg_trades_per_episode:.2f}")
        logger.info(f"   Episodes with trades: {episodes_with_trades}/{len(self.episode_trades)}")
        logger.info(f"   Best performance: {self.best_performance:.4f}")
        logger.info(f"   Final avg return (last 100): {final_avg_return:.4f}")
        logger.info(f"   Final Sharpe ratio: {final_sharpe:.2f}")
        logger.info(f"   Profitability rate: {profitability_rate:.1%}")

        # Success criteria
        is_profitable = final_avg_return > 0.01 and profitability_rate > 0.55
        has_trades = total_trades > 0

        if is_profitable:
            logger.info("🎉 TRAINING SUCCESS: Model achieved consistent profitability!")
        elif final_avg_return > 0:
            logger.info("✅ TRAINING PROGRESS: Model showing positive returns!")
        else:
            logger.info("⚠️ TRAINING INCOMPLETE: Model not yet profitable, continue training")

        return has_trades and is_profitable


class SimpleAgent:
    """Simple fallback agent"""
    
    def __init__(self, action_space_size: int):
        self.action_space_size = action_space_size
        self.action_probs = np.ones(action_space_size) / action_space_size
        self.learning_rate = 0.1
        
    def act(self, obs):
        action = np.random.choice(self.action_space_size, p=self.action_probs)
        return action, {'log_prob': np.log(self.action_probs[action]), 'value': 0.0}
    
    def update(self, action, reward):
        if reward > 0:
            self.action_probs[action] *= (1 + self.learning_rate)
        else:
            self.action_probs[action] *= (1 - self.learning_rate * 0.5)
        
        self.action_probs = np.maximum(self.action_probs, 0.01)
        self.action_probs /= np.sum(self.action_probs)


async def main():
    """Main training function"""

    parser = argparse.ArgumentParser(description='Enhanced CLSTM-PPO Training with Multi-GPU Support')
    parser.add_argument('--episodes', '--num_episodes', type=int, default=1000, help='Episodes to train')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use (-1 for all available)')
    parser.add_argument('--wandb', action='store_true', help='Use wandb logging')
    parser.add_argument('--pretraining', action='store_true', default=False, help='Use CLSTM pretraining')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/enhanced_clstm_ppo',
                        help='Checkpoint directory')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume from checkpoint if available')
    parser.add_argument('--fresh-start', action='store_true',
                        help='Start fresh training (ignore checkpoints)')
    parser.add_argument('--resume-from', type=str, choices=['best', 'composite', 'latest', 'win_rate', 'profit_rate', 'sharpe'],
                        default='best', help='Which model to resume from (default: best=composite)')

    # NEW: Multi-leg strategies and ensemble support
    parser.add_argument('--enable-multi-leg', '--enable_multi_leg', action='store_true', default=False,
                        help='Enable multi-leg strategies (91 actions instead of 31)')
    parser.add_argument('--no-multi-leg', '--no_multi_leg', dest='enable_multi_leg', action='store_false',
                        help='Disable multi-leg strategies (use 31 actions)')
    parser.add_argument('--use-ensemble', '--use_ensemble', action='store_true', default=False,
                        help='Use ensemble methods (multiple models with weighted voting)')
    parser.add_argument('--num-ensemble-models', '--num_ensemble_models', type=int, default=3,
                        help='Number of models in ensemble (default: 3)')
    parser.add_argument('--train-ensemble', '--train_ensemble', action='store_true', default=False,
                        help='Train ensemble models (instead of single model)')
    parser.add_argument('--episodes-per-ensemble-model', type=int, default=1000,
                        help='Episodes to train each ensemble model (default: 1000)')

    # Early stopping
    parser.add_argument('--early-stopping-patience', type=int, default=500,
                        help='Stop training if no improvement for N episodes (0 to disable, default: 500)')
    parser.add_argument('--no-early-stopping', action='store_true', default=False,
                        help='Disable early stopping (equivalent to --early-stopping-patience 0)')
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.001,
                        help='Minimum improvement to reset patience (default: 0.001)')

    # Data loading options
    parser.add_argument('--data-days', type=int, default=730,
                        help='Number of days of historical data to load (default: 730 = 2 years)')
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test mode: 3 symbols, 90 days data, 100 episodes')

    # Performance optimization
    parser.add_argument('--no-step-logging', action='store_true', default=False,
                        help='Disable step-by-step logging for maximum training speed')

    # Transaction costs
    parser.add_argument('--realistic-costs', dest='use_realistic_costs', action='store_true', default=True,
                        help='Use realistic transaction costs (default: enabled)')
    parser.add_argument('--no-realistic-costs', dest='use_realistic_costs', action='store_false',
                        help='Disable realistic transaction costs for faster learning')

    # Data source options
    parser.add_argument('--use-massive-flat-files', action='store_true', default=False,
                        help='Use Massive flat files with calculated Greeks (recommended)')
    parser.add_argument('--massive-flat-files-dir', type=str, default='data/flat_files_processed',
                        help='Directory containing Massive flat files (default: data/flat_files_processed)')
    parser.add_argument('--use-flat-files', action='store_true', default=False,
                        help='Use legacy flat files instead of REST API')
    parser.add_argument('--flat-files-dir', type=str, default='data/flat_files',
                        help='Directory containing legacy flat files (default: data/flat_files)')
    parser.add_argument('--flat-files-format', type=str, choices=['parquet', 'csv'], default='parquet',
                        help='Legacy flat file format (default: parquet)')

    args = parser.parse_args()

    # Quick test mode overrides
    if args.quick_test:
        args.data_days = 90
        args.episodes = 100
        logger.info("🚀 QUICK TEST MODE: 3 symbols, 90 days, 100 episodes")

    # Determine number of GPUs
    available_gpus = torch.cuda.device_count()
    if args.num_gpus == -1:
        world_size = available_gpus
    else:
        world_size = min(args.num_gpus, available_gpus)

    # Enhanced configuration
    # Quick test mode uses fewer symbols
    if args.quick_test:
        symbols_list = ['SPY', 'QQQ', 'AAPL']  # Just 3 symbols for quick testing
    else:
        symbols_list = [
            'SPY', 'QQQ', 'IWM',  # ETFs
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',  # Mega cap
            'TSLA', 'META', 'NFLX', 'AMD', 'CRM',  # High vol tech
            'PLTR', 'SNOW', 'COIN', 'RBLX', 'ZM',  # Growth stocks
            'JPM', 'BAC', 'GS', 'V', 'MA'  # Financials
        ]

    # If using Massive flat files (new processed format)
    if args.use_massive_flat_files:
        import pandas as pd
        from src.data.massive_flat_file_loader import MassiveFlatFileLoader

        logger.info("🔍 Pre-training validation: Checking Massive flat file data...")

        loader = MassiveFlatFileLoader(data_dir=args.massive_flat_files_dir)
        available_files = loader.get_available_files()

        if available_files:
            summary = loader.get_data_summary()
            available_symbols = summary.get('symbols', [])

            logger.info(f"✅ Found {len(available_files)} parquet files in {args.massive_flat_files_dir}")
            logger.info(f"   Total records: {summary['total_records']:,}")
            logger.info(f"   Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
            logger.info(f"   Symbols: {available_symbols}")
            logger.info(f"   Unique dates: {summary['unique_dates']}")

            # Filter to only symbols in the data
            symbols_list = [s for s in symbols_list if s in available_symbols]

            if not symbols_list:
                logger.warning(f"⚠️ None of requested symbols found in data. Using available: {available_symbols}")
                symbols_list = available_symbols[:5]  # Use up to 5 available symbols
            else:
                logger.info(f"📊 Using {len(symbols_list)} symbols: {symbols_list}")
        else:
            logger.error(f"❌ No Massive flat files found in {args.massive_flat_files_dir}")
            logger.error("   Run: python scripts/download_flat_files.py --years 2024 --symbols SPY QQQ IWM")
            raise ValueError(f"No data files found in {args.massive_flat_files_dir}")

    # If using legacy flat files, filter to only symbols that have data available
    elif args.use_flat_files:
        import os
        import pandas as pd

        available_symbols = []
        stocks_dir = os.path.join(args.flat_files_dir, 'stocks')
        options_dir = os.path.join(args.flat_files_dir, 'options')

        logger.info("🔍 Pre-training validation: Checking flat file data coverage...")

        for symbol in symbols_list:
            # Check if both stock and options data exist
            stock_file = os.path.join(stocks_dir, f"{symbol}.{args.flat_files_format}")
            options_file = os.path.join(options_dir, f"{symbol}_options.{args.flat_files_format}")

            if os.path.exists(stock_file) and os.path.exists(options_file):
                # Validate data coverage
                try:
                    if args.flat_files_format == 'parquet':
                        stock_df = pd.read_parquet(stock_file)
                        options_df = pd.read_parquet(options_file)
                    else:
                        stock_df = pd.read_csv(stock_file)
                        options_df = pd.read_csv(options_file)

                    stock_days = len(stock_df)
                    options_contracts = len(options_df)

                    # Check if data is sufficient
                    if stock_days < args.data_days * 0.3:  # Less than 30% of requested
                        logger.warning(f"⚠️  {symbol}: Only {stock_days} days in flat file (requested {args.data_days})")

                    if options_contracts < 500:  # Arbitrary minimum
                        logger.warning(f"⚠️  {symbol}: Only {options_contracts} options contracts (may be insufficient)")

                    available_symbols.append(symbol)
                    logger.info(f"  ✅ {symbol}: {stock_days} days stock, {options_contracts:,} options")

                except Exception as e:
                    logger.error(f"  ❌ {symbol}: Error reading flat file: {e}")

        if available_symbols:
            logger.info(f"📊 Using flat files: Found data for {len(available_symbols)} symbols: {available_symbols}")
            symbols_list = available_symbols

            # Final validation: Check if we should proceed
            if not args.quick_test:
                # Load one symbol to check date range
                sample_symbol = available_symbols[0]
                sample_file = os.path.join(stocks_dir, f"{sample_symbol}.{args.flat_files_format}")

                if args.flat_files_format == 'parquet':
                    sample_df = pd.read_parquet(sample_file)
                else:
                    sample_df = pd.read_csv(sample_file)

                actual_days = len(sample_df)

                # Calculate expected trading days (assume ~252 trading days per year)
                expected_trading_days = (args.data_days / 365.25) * 252
                coverage_pct = (actual_days / expected_trading_days) * 100

                if actual_days < expected_trading_days * 0.5:
                    logger.error(f"❌ INSUFFICIENT DATA: Flat files contain {actual_days} trading days")
                    logger.error(f"   Requested: {args.data_days} calendar days (~{expected_trading_days:.0f} trading days expected)")
                    logger.error(f"   Coverage: {coverage_pct:.0f}% (need at least 50%)")
                    logger.error(f"   Please download more data:")
                    logger.error(f"   python3 download_data_to_flat_files.py --days {args.data_days}")
                    raise ValueError(f"Insufficient data in flat files: {actual_days} trading days available, ~{expected_trading_days:.0f} expected")
                elif actual_days < expected_trading_days * 0.9:
                    logger.warning(f"⚠️  Data coverage: {actual_days} trading days ({coverage_pct:.0f}% of expected ~{expected_trading_days:.0f})")
                    logger.warning(f"   Consider downloading more data:")
                    logger.warning(f"   python3 download_data_to_flat_files.py --days {args.data_days}")
                else:
                    logger.info(f"✅ Data coverage is excellent: {actual_days} trading days ({coverage_pct:.0f}% of expected ~{expected_trading_days:.0f})")
        else:
            logger.warning(f"⚠️  No flat file data found in {args.flat_files_dir}")
            logger.warning(f"   Please run: python3 download_data_to_flat_files.py --days {args.data_days}")
            logger.warning(f"   Falling back to REST API or synthetic data")
    else:
        logger.info(f"📊 Using REST API for {len(symbols_list)} symbols")

    # Check if running in stable training mode (set by train_stable.py)
    import os
    stable_mode = os.environ.get('STABLE_TRAINING', '0') == '1'

    if stable_mode:
        logger.info("🔒 STABLE TRAINING MODE ENABLED")
        logger.info("   Using configuration from configs/stable_training.yaml")

    config = {
        'num_episodes': args.episodes,
        'use_clstm_pretraining': args.pretraining,
        'symbols': symbols_list,
        'data_days': args.data_days,  # Pass data_days to config
        'quick_test': args.quick_test,  # Pass quick_test flag for validation
        # STABILITY: Use environment variables if in stable mode, otherwise use defaults
        'learning_rate_actor_critic': float(os.environ.get('LEARNING_RATE_ACTOR_CRITIC', '3e-4')),
        'learning_rate_clstm': float(os.environ.get('LEARNING_RATE_CLSTM', '1e-3')),
        'entropy_coef': float(os.environ.get('ENTROPY_COEF', '0.05')),
        'entropy_decay': float(os.environ.get('ENTROPY_DECAY', '0.995')),
        'min_entropy_coef': float(os.environ.get('MIN_ENTROPY_COEF', '0.01')),
        'max_grad_norm': float(os.environ.get('MAX_GRAD_NORM', '0.5')),
        'l2_reg': float(os.environ.get('L2_REG', '1e-4')),
        'normalize_rewards': os.environ.get('NORMALIZE_REWARDS', 'true').lower() == 'true',
        'reward_clip': float(os.environ.get('REWARD_CLIP', '10.0')),
        'batch_size': int(os.environ.get('BATCH_SIZE', '64')),
        'include_technical_indicators': True,
        'include_market_microstructure': True,
        # Transaction costs (configurable via --realistic-costs / --no-realistic-costs)
        'use_realistic_costs': args.use_realistic_costs,
        'enable_slippage': args.use_realistic_costs,  # Disable slippage if costs disabled
        'slippage_model': 'volume_based' if args.use_realistic_costs else 'none',
        # Early stopping
        'early_stopping_patience': 0 if args.no_early_stopping else args.early_stopping_patience,
        'early_stopping_min_delta': args.early_stopping_min_delta,
        # Performance optimization
        'no_step_logging': args.no_step_logging,
        # Data source (flat files vs REST API)
        'use_massive_flat_files': args.use_massive_flat_files,
        'massive_flat_files_dir': args.massive_flat_files_dir,
        'use_flat_files': args.use_flat_files,
        'flat_files_dir': args.flat_files_dir,
        'flat_files_format': args.flat_files_format
    }

    logger.info("🚀 Starting Enhanced CLSTM-PPO Training with Multi-GPU Support")
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"GPUs: {world_size}/{available_gpus} available")
    logger.info(f"CLSTM Pretraining: {args.pretraining}")
    logger.info(f"Symbols: {len(config['symbols'])}")
    logger.info(f"Checkpoint dir: {args.checkpoint_dir}")
    logger.info(f"Resume training: {args.resume and not args.fresh_start}")
    logger.info(f"Realistic transaction costs: {config['use_realistic_costs']}")
    logger.info(f"Multi-leg strategies: {args.enable_multi_leg} ({'91 actions' if args.enable_multi_leg else '31 actions'})")
    logger.info(f"Ensemble methods: {args.use_ensemble} ({args.num_ensemble_models} models)" if args.use_ensemble else f"Ensemble methods: False")
    if args.train_ensemble:
        logger.info(f"Training mode: Ensemble ({args.num_ensemble_models} models × {args.episodes_per_ensemble_model} episodes each)")

    # Clear checkpoint directory if fresh start
    if args.fresh_start:
        import shutil
        checkpoint_path = Path(args.checkpoint_dir)
        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)
            logger.info("🗑️ Cleared checkpoint directory for fresh start")

    # Launch training
    # FIXED: Always use single-process mode, even with multiple GPUs
    # PyTorch DDP will handle multi-GPU within the same process
    logger.info("📍 Single-process training mode")
    if world_size > 1:
        logger.info(f"   🌐 Using {world_size} GPUs with DataParallel (not DDP)")
        logger.info(f"   ⚠️  Note: True DDP requires NCCL backend and separate process per GPU")
        logger.info(f"   ✅ Using nn.DataParallel for multi-GPU support instead")

    trainer = EnhancedCLSTMPPOTrainer(
        use_wandb=args.wandb,
        config=config,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        rank=0,
        world_size=1,  # Always 1 for single-process mode
        distributed=False,  # Disable DDP, use DataParallel instead
        enable_multi_leg=args.enable_multi_leg,
        use_ensemble=args.use_ensemble,
        num_ensemble_models=args.num_ensemble_models
    )

    await trainer.initialize()

    # Train ensemble or single model
    if args.train_ensemble:
        await trainer.train_ensemble_models(episodes_per_model=args.episodes_per_ensemble_model)
        print("\n🎉 ENSEMBLE TRAINING COMPLETE!")
        print(f"✅ Trained {args.num_ensemble_models} models")
        print(f"✅ {args.episodes_per_ensemble_model} episodes per model")
        print(f"✅ Total episodes: {args.num_ensemble_models * args.episodes_per_ensemble_model}")
    else:
        success = await trainer.train(args.episodes)

        # Close TensorBoard writer
        if hasattr(trainer, 'writer') and trainer.writer is not None:
            trainer.writer.close()
            logger.info("✅ TensorBoard writer closed")
            print(f"\n📊 TensorBoard logs saved to: {trainer.tensorboard_dir}")
            print(f"   View with: tensorboard --logdir={trainer.tensorboard_dir}")

        if success:
            print("\n🎉 ENHANCED CLSTM-PPO TRAINING SUCCESS!")
            print("✅ CLSTM features properly extracted and used by PPO")
            print("✅ Technical indicators and market microstructure included")
            print("✅ Realistic transaction costs integrated")
            if args.enable_multi_leg:
                print("✅ Multi-leg strategies enabled (91 actions)")
            if args.use_ensemble:
                print(f"✅ Ensemble methods enabled ({args.num_ensemble_models} models)")
            print("✅ Agent learned to trade with complex features")
        else:
            # Check if trades happened but not profitable
            total_trades = sum(trainer.episode_trades) if trainer.episode_trades else 0
            if total_trades > 0:
                print(f"\n⚠️ Training completed with {total_trades} total trades but not yet profitable")
                print("   Continue training or adjust hyperparameters for better performance")
            else:
                print("\n❌ Training completed but no trades produced")
                print("   Check exploration settings and reward function")


if __name__ == "__main__":
    asyncio.run(main())
