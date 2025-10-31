import os
import json
import pickle
from typing import Dict, List, Any, Optional
from datetime import datetime
import time
from pathlib import Path
import threading
from loguru import logger

class CheckpointManager:
    """
    Manages checkpoints for the trading bot to allow recovery from crashes.
    Saves trading state, positions, and other relevant information.
    """
    
    def __init__(self, config):
        """Initialize the checkpoint manager with the config"""
        self.config = config
        
        # Create checkpoints directory if it doesn't exist
        self.checkpoint_dir = config.base_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Checkpoint file paths
        self.trading_state_path = self.checkpoint_dir / "trading_state.json"
        self.positions_path = self.checkpoint_dir / "positions.json"
        self.runtime_state_path = self.checkpoint_dir / "runtime_state.pkl"
        
        # Lock for thread-safe file operations - marked as non-picklable
        self._lock = threading.RLock()
        
        # Auto-save interval in seconds (default: 5 minutes)
        self.autosave_interval = 300
        self._autosave_thread = None
        self._stop_autosave = threading.Event()
        
        logger.info(f"CheckpointManager initialized with checkpoint directory: {self.checkpoint_dir}")
        
    def __getstate__(self):
        """Custom state handling for pickling"""
        state = self.__dict__.copy()
        # Don't pickle the lock
        state.pop('_lock', None)
        # Don't pickle the thread
        state.pop('_autosave_thread', None)
        return state
    
    def __setstate__(self, state):
        """Custom state handling for unpickling"""
        self.__dict__.update(state)
        # Recreate the lock when unpickling
        self._lock = threading.RLock()
        self._autosave_thread = None
    
    def start_autosave(self):
        """Start the autosave thread"""
        if self._autosave_thread is None or not self._autosave_thread.is_alive():
            self._stop_autosave.clear()
            self._autosave_thread = threading.Thread(target=self._autosave_loop, daemon=True)
            self._autosave_thread.start()
            logger.info(f"Autosave started with interval: {self.autosave_interval}s")
    
    def stop_autosave(self):
        """Stop the autosave thread"""
        if self._autosave_thread and self._autosave_thread.is_alive():
            self._stop_autosave.set()
            self._autosave_thread.join(timeout=10)
            logger.info("Autosave stopped")
    
    def _autosave_loop(self):
        """Background thread that periodically saves checkpoints"""
        while not self._stop_autosave.is_set():
            try:
                # Save current state
                self.save_checkpoint()
                logger.debug("Auto-saved checkpoint")
            except Exception as e:
                logger.error(f"Error during autosave: {e}")
            
            # Sleep for the interval or until stopped
            self._stop_autosave.wait(self.autosave_interval)
    
    def save_checkpoint(self, trading_bot=None):
        """
        Save a complete checkpoint of the trading bot state.
        If trading_bot is provided, extract state from it.
        Otherwise, save the latest state data that was set with set_*() methods.
        """
        with self._lock:
            try:
                # Get timestamp for the checkpoint
                timestamp = datetime.now().isoformat()
                
                if trading_bot:
                    # Extract state from the trading bot
                    trading_state = {
                        "timestamp": timestamp,
                        "is_trading": trading_bot.is_trading,
                        "trading_symbols": trading_bot.trading_symbols,
                        "last_trade_time": {
                            symbol: dt.isoformat() if dt else None 
                            for symbol, dt in trading_bot.last_trade_time.items()
                        },
                        "_screener_run_today": getattr(trading_bot, '_screener_run_today', False)
                    }
                    
                    # Get positions from the executor
                    positions = trading_bot.executor.get_positions()
                    positions_data = {
                        "timestamp": timestamp,
                        "positions": {
                            symbol: {
                                "qty": float(pos.qty),
                                "market_value": float(pos.market_value),
                                "avg_entry_price": float(pos.avg_entry_price),
                                "current_price": float(pos.current_price),
                                "unrealized_pl": float(pos.unrealized_pl),
                                "unrealized_plpc": float(pos.unrealized_plpc),
                                "side": "long" if float(pos.qty) > 0 else "short"
                            }
                            for symbol, pos in positions.items()
                        },
                        "account": {
                            "cash": float(trading_bot.executor.account.cash),
                            "equity": float(trading_bot.executor.account.equity),
                            "buying_power": float(trading_bot.executor.account.buying_power)
                        }
                    }
                    
                    # Save trading state
                    with open(self.trading_state_path, 'w') as f:
                        json.dump(trading_state, f, indent=2)
                    
                    # Save positions
                    with open(self.positions_path, 'w') as f:
                        json.dump(positions_data, f, indent=2)
                    
                    # Instead of trying to pickle the entire trading bot, save essential state
                    # This is more reliable than trying to pickle the entire bot
                    runtime_state = {
                        "timestamp": timestamp,
                        "ppo_agent_state": trading_bot.ppo_agent.get_state_dict() if hasattr(trading_bot.ppo_agent, 'get_state_dict') else None,
                        "is_trading": trading_bot.is_trading,
                        "trading_symbols": trading_bot.trading_symbols,
                        "config": trading_bot.config.__dict__.copy()
                    }
                    with open(self.runtime_state_path, 'wb') as f:
                        pickle.dump(runtime_state, f)
                    
                    # Save models separately
                    trading_bot.save_models()
                    
                    logger.info(f"Checkpoint saved at {timestamp}")
                    return True
                else:
                    logger.warning("No trading bot provided for checkpoint save")
                    return False
                
            except Exception as e:
                logger.error(f"Error saving checkpoint: {e}", exc_info=True)
                return False
    
    def load_checkpoint(self) -> Dict[str, Any]:
        """
        Load the latest checkpoint data.
        Returns a dictionary with:
        - trading_state: The saved trading state
        - positions: The saved positions
        - runtime_state: (Optional) The pickled runtime state if it exists
        """
        with self._lock:
            result = {
                "trading_state": None,
                "positions": None,
                "runtime_state": None,
                "exists": False
            }
            
            # Check if checkpoint files exist
            trading_state_exists = self.trading_state_path.exists()
            positions_exists = self.positions_path.exists()
            runtime_state_exists = self.runtime_state_path.exists()
            
            if not (trading_state_exists or positions_exists or runtime_state_exists):
                logger.info("No checkpoint found")
                return result
            
            # Set exists flag
            result["exists"] = True
            
            try:
                # Load trading state
                if trading_state_exists:
                    with open(self.trading_state_path, 'r') as f:
                        trading_state = json.load(f)
                    result["trading_state"] = trading_state
                    
                    # Convert ISO timestamp strings back to datetime objects
                    if "last_trade_time" in trading_state:
                        trading_state["last_trade_time"] = {
                            symbol: datetime.fromisoformat(dt) if dt else None
                            for symbol, dt in trading_state["last_trade_time"].items()
                        }
                
                # Load positions
                if positions_exists:
                    with open(self.positions_path, 'r') as f:
                        positions = json.load(f)
                    result["positions"] = positions
                
                # Load runtime state if needed (this is a large pickle file with the whole bot)
                if runtime_state_exists:
                    # We'll defer actual loading of the pickle until explicitly requested
                    result["runtime_state_available"] = True
                
                logger.info(f"Checkpoint loaded from {self.checkpoint_dir}")
                return result
                
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}", exc_info=True)
                return result
    
    def load_runtime_state(self) -> Optional[Dict[str, Any]]:
        """
        Load the pickled runtime state.
        This is separated from load_checkpoint() as it can be a large file.
        """
        with self._lock:
            if not self.runtime_state_path.exists():
                logger.warning("No runtime state file found")
                return None
            
            try:
                with open(self.runtime_state_path, 'rb') as f:
                    runtime_state = pickle.load(f)
                
                logger.info(f"Runtime state loaded from {self.runtime_state_path}")
                return runtime_state
                
            except Exception as e:
                logger.error(f"Error loading runtime state: {e}", exc_info=True)
                return None
    
    def clear_checkpoints(self):
        """Clear all checkpoint files"""
        with self._lock:
            try:
                # Remove individual files
                if self.trading_state_path.exists():
                    self.trading_state_path.unlink()
                
                if self.positions_path.exists():
                    self.positions_path.unlink()
                
                if self.runtime_state_path.exists():
                    self.runtime_state_path.unlink()
                
                logger.info("All checkpoint files cleared")
                return True
                
            except Exception as e:
                logger.error(f"Error clearing checkpoints: {e}", exc_info=True)
                return False
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about available checkpoints"""
        with self._lock:
            info = {
                "checkpoint_dir": str(self.checkpoint_dir),
                "files": {},
                "has_checkpoint": False
            }
            
            # Check individual files
            if self.trading_state_path.exists():
                info["files"]["trading_state"] = {
                    "path": str(self.trading_state_path),
                    "size": self.trading_state_path.stat().st_size,
                    "modified": datetime.fromtimestamp(self.trading_state_path.stat().st_mtime).isoformat()
                }
                info["has_checkpoint"] = True
            
            if self.positions_path.exists():
                info["files"]["positions"] = {
                    "path": str(self.positions_path),
                    "size": self.positions_path.stat().st_size,
                    "modified": datetime.fromtimestamp(self.positions_path.stat().st_mtime).isoformat()
                }
                info["has_checkpoint"] = True
            
            if self.runtime_state_path.exists():
                info["files"]["runtime_state"] = {
                    "path": str(self.runtime_state_path),
                    "size": self.runtime_state_path.stat().st_size,
                    "modified": datetime.fromtimestamp(self.runtime_state_path.stat().st_mtime).isoformat()
                }
                info["has_checkpoint"] = True
            
            return info