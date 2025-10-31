#!/usr/bin/env python3
"""
Live Paper Trading Training Script with PPO-CLSTM Architecture
This script trains the model using real-time paper trading with Alpaca API
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import argparse
import traceback
import asyncio
from dotenv import load_dotenv
import time
from collections import deque
import json
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
import yfinance as yf
import threading
import queue

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from options_clstm_ppo import OptionsCLSTMPPOAgent, OptionsCLSTMPPONetwork
from live_trader import TechnicalIndicators

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/live_paper_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LivePaperTradingEnvironment:
    """Real-time paper trading environment using Alpaca API"""
    
    def __init__(self, api_key, api_secret, base_url, symbols, initial_capital=100000):
        self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        self.symbols = symbols
        self.initial_capital = initial_capital
        
        # State tracking
        self.current_positions = {}
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.total_pnl = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.trade_history = []
        
        # Market data cache
        self.price_history = {symbol: deque(maxlen=100) for symbol in symbols}
        self.volume_history = {symbol: deque(maxlen=100) for symbol in symbols}
        self.technical_indicators = {symbol: {} for symbol in symbols}
        
        # Real-time data collection
        self.market_data_queue = queue.Queue()
        self.is_running = False
        self.data_thread = None
        
        # Options data approximation (since Alpaca doesn't support options)
        # We'll simulate options behavior based on stock movements
        self.implied_volatility = {symbol: 0.25 for symbol in symbols}  # Default 25% IV
        
        logger.info(f"Initialized live paper trading environment with symbols: {symbols}")
    
    def start_data_collection(self):
        """Start real-time data collection thread"""
        self.is_running = True
        self.data_thread = threading.Thread(target=self._collect_market_data)
        self.data_thread.daemon = True
        self.data_thread.start()
        logger.info("Started real-time data collection")
    
    def stop_data_collection(self):
        """Stop data collection"""
        self.is_running = False
        if self.data_thread:
            self.data_thread.join()
        logger.info("Stopped data collection")
    
    def _collect_market_data(self):
        """Background thread to collect real-time market data"""
        while self.is_running:
            try:
                # Get latest quotes for all symbols
                for symbol in self.symbols:
                    try:
                        # Get latest quote
                        quote = self.api.get_latest_quote(symbol)
                        trade = self.api.get_latest_trade(symbol)
                        
                        # Update price history
                        self.price_history[symbol].append(float(trade.price))
                        self.volume_history[symbol].append(float(trade.size))
                        
                        # Calculate technical indicators if we have enough data
                        if len(self.price_history[symbol]) >= 26:
                            self._update_technical_indicators(symbol)
                        
                        # Put data in queue for processing
                        self.market_data_queue.put({
                            'symbol': symbol,
                            'price': float(trade.price),
                            'bid': float(quote.bid_price),
                            'ask': float(quote.ask_price),
                            'volume': float(trade.size),
                            'timestamp': datetime.now()
                        })
                        
                    except Exception as e:
                        logger.error(f"Error collecting data for {symbol}: {e}")
                
                # Sleep to avoid rate limits (5 requests per second for free tier)
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error in data collection thread: {e}")
                time.sleep(1)
    
    def _update_technical_indicators(self, symbol):
        """Update technical indicators for a symbol"""
        prices = list(self.price_history[symbol])
        
        # Calculate indicators
        self.technical_indicators[symbol] = TechnicalIndicators.calculate_all_indicators(
            prices=prices,
            high_prices=prices,  # Approximation
            low_prices=prices    # Approximation
        )
        
        # Update implied volatility estimate
        if len(prices) >= 20:
            returns = np.diff(prices) / prices[:-1]
            self.implied_volatility[symbol] = np.std(returns) * np.sqrt(252)  # Annualized
    
    def get_observation(self):
        """Get current observation state for the model"""
        obs = {
            'price_history': np.zeros((len(self.symbols), 50, 5)),  # OHLCV
            'technical_indicators': np.zeros((20,)),
            'options_chain': np.zeros((10, 20, 8)),  # Simulated options data
            'portfolio_state': np.zeros((5,)),
            'greeks_summary': np.zeros((5,))
        }
        
        # Fill price history
        for i, symbol in enumerate(self.symbols):
            if len(self.price_history[symbol]) > 0:
                prices = list(self.price_history[symbol])[-50:]
                for j, price in enumerate(prices):
                    obs['price_history'][i, j, :] = [price, price, price, price, self.volume_history[symbol][j] if j < len(self.volume_history[symbol]) else 0]
        
        # Fill technical indicators (averaged across symbols)
        indicator_values = []
        for symbol in self.symbols:
            if symbol in self.technical_indicators and self.technical_indicators[symbol]:
                indicators = self.technical_indicators[symbol]
                indicator_values.append([
                    indicators.get('macd_histogram', 0) / 10.0,
                    (indicators.get('rsi', 50) - 50) / 50.0,
                    indicators.get('cci', 0) / 200.0,
                    indicators.get('adx', 0) / 50.0,
                    indicators.get('bollinger_width', 0.02) * 50,
                    indicators.get('stochastic_k', 50) / 100.0,
                    indicators.get('stochastic_d', 50) / 100.0,
                    self.implied_volatility.get(symbol, 0.25) * 4,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                ])
        
        if indicator_values:
            obs['technical_indicators'] = np.mean(indicator_values, axis=0)[:20]
        
        # Simulate options chain (since Alpaca doesn't support options)
        # This is a simplified representation
        for i in range(min(10, len(self.symbols))):
            symbol = self.symbols[i % len(self.symbols)]
            current_price = self.price_history[symbol][-1] if self.price_history[symbol] else 100
            
            for j in range(20):
                strike_offset = (j - 10) * 0.01  # -10% to +10% strikes
                strike = current_price * (1 + strike_offset)
                
                # Simple Black-Scholes approximation for options
                moneyness = current_price / strike
                time_to_expiry = 30 / 365  # 30 days
                iv = self.implied_volatility.get(symbol, 0.25)
                
                # Simplified option pricing
                if j < 10:  # Calls
                    intrinsic = max(0, current_price - strike)
                    time_value = current_price * 0.05 * iv * np.sqrt(time_to_expiry)
                    option_price = intrinsic + time_value
                else:  # Puts
                    intrinsic = max(0, strike - current_price)
                    time_value = current_price * 0.05 * iv * np.sqrt(time_to_expiry)
                    option_price = intrinsic + time_value
                
                obs['options_chain'][i, j, :] = [
                    strike, option_price * 0.95, option_price * 1.05, option_price,
                    1000, 0.1 if j < 10 else -0.1, 0.5, moneyness
                ]
        
        # Portfolio state
        obs['portfolio_state'] = np.array([
            self.cash / self.initial_capital,
            len(self.current_positions) / 10.0,
            self.portfolio_value / self.initial_capital,
            self.total_pnl / self.initial_capital,
            (self.winning_trades + self.losing_trades) / 100.0
        ])
        
        # Simulated Greeks summary
        total_delta = sum([pos.get('delta', 0) for pos in self.current_positions.values()])
        obs['greeks_summary'] = np.array([
            total_delta / 10.0, 0.0, 0.0, 0.0, 0.0
        ])
        
        return obs
    
    def execute_action(self, action, action_info):
        """Execute trading action in paper trading account"""
        action_mapping = {
            0: 'hold',
            1: 'buy_stock',  # Simplified from buy_call
            2: 'sell_stock', # Simplified from buy_put
            3: 'close_positions'
        }
        
        action_name = action_mapping.get(action, 'hold')
        reward = 0
        
        try:
            if action_name == 'hold':
                # No action, just update portfolio value
                self._update_portfolio_value()
                
            elif action_name == 'buy_stock' and len(self.current_positions) < 5:
                # Select best symbol to buy
                symbol = self._select_best_symbol('buy')
                if symbol:
                    # Calculate position size
                    current_price = self.price_history[symbol][-1]
                    position_size = min(self.cash * 0.1, 10000)  # Max 10% or $10k
                    shares = int(position_size / current_price)
                    
                    if shares > 0:
                        # Submit order
                        order = self.api.submit_order(
                            symbol=symbol,
                            qty=shares,
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )
                        
                        # Track position
                        self.current_positions[symbol] = {
                            'shares': shares,
                            'entry_price': current_price,
                            'entry_time': datetime.now(),
                            'order_id': order.id,
                            'unrealized_pnl': 0,
                            'delta': shares / 100  # Approximate delta
                        }
                        
                        self.cash -= shares * current_price
                        reward = 0.1  # Small positive reward for taking action
                        logger.info(f"Bought {shares} shares of {symbol} at ${current_price:.2f}")
            
            elif action_name == 'sell_stock' and len(self.current_positions) < 5:
                # For paper trading, we'll use inverse ETFs or just track inverse positions
                symbol = self._select_best_symbol('sell')
                if symbol and symbol not in self.current_positions:
                    current_price = self.price_history[symbol][-1]
                    position_size = min(self.cash * 0.1, 10000)
                    shares = int(position_size / current_price)
                    
                    if shares > 0:
                        # Track as negative position (short)
                        self.current_positions[symbol] = {
                            'shares': -shares,  # Negative for short
                            'entry_price': current_price,
                            'entry_time': datetime.now(),
                            'order_id': f"short_{symbol}_{datetime.now().timestamp()}",
                            'unrealized_pnl': 0,
                            'delta': -shares / 100
                        }
                        
                        self.cash -= shares * current_price * 0.1  # Margin requirement
                        reward = 0.1
                        logger.info(f"Shorted {shares} shares of {symbol} at ${current_price:.2f}")
            
            elif action_name == 'close_positions':
                # Close all positions
                total_pnl = 0
                for symbol, position in list(self.current_positions.items()):
                    pnl = self._close_position(symbol)
                    total_pnl += pnl
                    if pnl > 0:
                        self.winning_trades += 1
                        reward += 10
                    else:
                        self.losing_trades += 1
                        reward -= 2
                
                self.total_pnl += total_pnl
                logger.info(f"Closed all positions. Total PnL: ${total_pnl:.2f}")
            
            # Update portfolio value
            self._update_portfolio_value()
            
            # Calculate step reward based on portfolio change
            portfolio_change = (self.portfolio_value - self.initial_capital) / self.initial_capital
            reward += portfolio_change * 100
            
            # Win rate bonus
            if self.winning_trades + self.losing_trades >= 5:
                win_rate = self.winning_trades / (self.winning_trades + self.losing_trades)
                if win_rate >= 0.6:
                    reward += 5
                elif win_rate >= 0.5:
                    reward += 2
                elif win_rate < 0.3:
                    reward -= 5
            
        except Exception as e:
            logger.error(f"Error executing action {action_name}: {e}")
            reward = -1
        
        return reward
    
    def _select_best_symbol(self, direction='buy'):
        """Select best symbol based on technical indicators"""
        scores = {}
        
        for symbol in self.symbols:
            if symbol not in self.technical_indicators or not self.technical_indicators[symbol]:
                continue
            
            indicators = self.technical_indicators[symbol]
            score = 0
            
            if direction == 'buy':
                # Bullish signals
                if indicators.get('rsi', 50) < 30:  # Oversold
                    score += 2
                if indicators.get('macd_histogram', 0) > 0:  # Bullish MACD
                    score += 1
                if indicators.get('adx', 0) > 25:  # Strong trend
                    score += 1
            else:
                # Bearish signals
                if indicators.get('rsi', 50) > 70:  # Overbought
                    score += 2
                if indicators.get('macd_histogram', 0) < 0:  # Bearish MACD
                    score += 1
                if indicators.get('adx', 0) > 25:  # Strong trend
                    score += 1
            
            # Avoid if already have position
            if symbol not in self.current_positions:
                score += 1
            
            scores[symbol] = score
        
        if scores:
            best_symbol = max(scores, key=scores.get)
            return best_symbol if scores[best_symbol] > 0 else None
        return None
    
    def _close_position(self, symbol):
        """Close a specific position"""
        if symbol not in self.current_positions:
            return 0
        
        position = self.current_positions[symbol]
        current_price = self.price_history[symbol][-1] if self.price_history[symbol] else position['entry_price']
        
        shares = position['shares']
        entry_price = position['entry_price']
        
        if shares > 0:  # Long position
            pnl = (current_price - entry_price) * shares
            self.cash += current_price * shares
        else:  # Short position
            pnl = (entry_price - current_price) * abs(shares)
            self.cash += abs(shares) * entry_price * 0.1  # Return margin
            self.cash += pnl
        
        # Update trade history
        self.trade_history.append({
            'symbol': symbol,
            'shares': shares,
            'entry_price': entry_price,
            'exit_price': current_price,
            'pnl': pnl,
            'duration': (datetime.now() - position['entry_time']).total_seconds() / 3600,
            'exit_time': datetime.now()
        })
        
        del self.current_positions[symbol]
        return pnl
    
    def _update_portfolio_value(self):
        """Update current portfolio value"""
        # Get account info from Alpaca
        try:
            account = self.api.get_account()
            self.portfolio_value = float(account.equity)
            self.cash = float(account.cash)
        except:
            # Fallback calculation
            positions_value = 0
            for symbol, position in self.current_positions.items():
                current_price = self.price_history[symbol][-1] if self.price_history[symbol] else position['entry_price']
                
                if position['shares'] > 0:
                    positions_value += current_price * position['shares']
                else:
                    # Short position value
                    pnl = (position['entry_price'] - current_price) * abs(position['shares'])
                    positions_value += abs(position['shares']) * position['entry_price'] * 0.1 + pnl
                
                # Update unrealized PnL
                position['unrealized_pnl'] = (current_price - position['entry_price']) * position['shares']
            
            self.portfolio_value = self.cash + positions_value
    
    def reset(self):
        """Reset environment (close all positions)"""
        # Close all positions
        for symbol in list(self.current_positions.keys()):
            self._close_position(symbol)
        
        # Reset stats
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0
        
        return self.get_observation()
    
    def get_info(self):
        """Get current environment info"""
        return {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'positions': len(self.current_positions),
            'total_pnl': self.total_pnl,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.winning_trades / max(1, self.winning_trades + self.losing_trades)
        }


def train_live_paper_trading(args):
    """Main training function for live paper trading"""
    logger.info("Starting live paper trading training")
    
    # Initialize environment
    env = LivePaperTradingEnvironment(
        api_key=os.getenv('APCA_API_KEY_ID'),
        api_secret=os.getenv('APCA_API_SECRET_KEY'),
        base_url=os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets'),
        symbols=args.symbols.split(','),
        initial_capital=args.initial_capital
    )
    
    # Start data collection
    env.start_data_collection()
    
    # Wait for initial data
    logger.info("Waiting for initial market data...")
    time.sleep(10)
    
    # Initialize agent
    obs = env.get_observation()
    observation_space = {
        'price_history': type('', (), {'shape': obs['price_history'].shape})(),
        'technical_indicators': type('', (), {'shape': obs['technical_indicators'].shape})(),
        'options_chain': type('', (), {'shape': obs['options_chain'].shape})(),
        'portfolio_state': type('', (), {'shape': obs['portfolio_state'].shape})(),
        'greeks_summary': type('', (), {'shape': obs['greeks_summary'].shape})()
    }
    
    agent = OptionsCLSTMPPOAgent(
        observation_space=observation_space,
        action_space=4,  # Simplified: hold, buy, sell, close_all
        learning_rate_actor_critic=args.lr_actor_critic,
        learning_rate_clstm=args.lr_clstm,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        batch_size=args.batch_size,
        n_epochs=args.ppo_epochs
    )
    
    # Load checkpoint if resuming
    if args.resume and args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        agent.load(args.checkpoint)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    win_rates = []
    
    # Main training loop
    try:
        for episode in range(args.episodes):
            logger.info(f"\n{'='*60}")
            logger.info(f"Episode {episode + 1}/{args.episodes}")
            logger.info(f"{'='*60}")
            
            # Reset environment
            obs = env.reset()
            episode_reward = 0
            episode_steps = 0
            
            # Episode loop
            for step in range(args.max_steps_per_episode):
                # Get action from agent
                action, action_info = agent.act(obs, deterministic=False)
                
                # Execute action
                reward = env.execute_action(action, action_info)
                
                # Get next observation
                next_obs = env.get_observation()
                
                # Store transition
                agent.store_transition(
                    observation=obs,
                    action=action,
                    reward=reward,
                    next_observation=next_obs,
                    done=False,
                    info=action_info
                )
                
                # Update state
                obs = next_obs
                episode_reward += reward
                episode_steps += 1
                
                # Train agent when buffer is full
                if len(agent.buffer) >= agent.batch_size * 2:
                    train_metrics = agent.train()
                    if episode % 10 == 0 and step == 0:
                        logger.info(f"Training metrics: {train_metrics}")
                
                # Log progress
                if step % 10 == 0:
                    info = env.get_info()
                    logger.info(f"Step {step}: Portfolio: ${info['portfolio_value']:.2f}, "
                              f"PnL: ${info['total_pnl']:.2f}, "
                              f"Positions: {info['positions']}, "
                              f"Win Rate: {info['win_rate']:.1%}")
                
                # Check for market close
                now = datetime.now()
                market_close = now.replace(hour=16, minute=0, second=0)
                if now >= market_close:
                    logger.info("Market closed, ending episode")
                    break
                
                # Safety check - maximum daily loss
                if info['total_pnl'] < -args.initial_capital * 0.05:  # 5% daily loss limit
                    logger.warning("Daily loss limit reached, ending episode")
                    break
            
            # Episode complete
            info = env.get_info()
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)
            win_rates.append(info['win_rate'])
            
            logger.info(f"\nEpisode {episode + 1} Summary:")
            logger.info(f"Total Reward: {episode_reward:.2f}")
            logger.info(f"Episode Length: {episode_steps}")
            logger.info(f"Final Portfolio: ${info['portfolio_value']:.2f}")
            logger.info(f"Total PnL: ${info['total_pnl']:.2f}")
            logger.info(f"Win Rate: {info['win_rate']:.1%} ({info['winning_trades']}W/{info['losing_trades']}L)")
            
            # Save checkpoint periodically
            if (episode + 1) % args.save_interval == 0:
                checkpoint_path = f"checkpoints/live_paper/checkpoint_episode_{episode + 1}.pt"
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                agent.save(checkpoint_path)
                logger.info(f"Saved checkpoint: {checkpoint_path}")
                
                # Save metrics
                metrics = {
                    'episode': episode + 1,
                    'rewards': episode_rewards[-args.save_interval:],
                    'lengths': episode_lengths[-args.save_interval:],
                    'win_rates': win_rates[-args.save_interval:],
                    'avg_reward': np.mean(episode_rewards[-args.save_interval:]),
                    'avg_win_rate': np.mean(win_rates[-args.save_interval:])
                }
                
                with open(f"checkpoints/live_paper/metrics_episode_{episode + 1}.json", 'w') as f:
                    json.dump(metrics, f, indent=2)
            
            # Check for convergence
            if len(win_rates) >= 50:
                recent_win_rate = np.mean(win_rates[-50:])
                if recent_win_rate >= 0.6:
                    logger.info(f"🎉 Achieved 60%+ win rate: {recent_win_rate:.1%}")
                    
                    # Save best model
                    agent.save("checkpoints/live_paper/best_model.pt")
                    logger.info("Saved best model!")
            
            # Wait before next episode (to respect API rate limits)
            if episode < args.episodes - 1:
                wait_time = max(0, 60 - episode_steps)  # At least 1 minute between episodes
                if wait_time > 0:
                    logger.info(f"Waiting {wait_time} seconds before next episode...")
                    time.sleep(wait_time)
    
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {e}")
        traceback.print_exc()
    finally:
        # Cleanup
        env.stop_data_collection()
        
        # Close all positions
        for symbol in list(env.current_positions.keys()):
            env._close_position(symbol)
        
        # Save final model
        agent.save("checkpoints/live_paper/final_model.pt")
        logger.info("Training complete. Final model saved.")


def main():
    parser = argparse.ArgumentParser(description='Live Paper Trading Training with PPO-CLSTM')
    
    # Environment settings
    parser.add_argument('--symbols', type=str, default='AAPL,MSFT,GOOGL,AMZN,TSLA',
                       help='Comma-separated list of symbols to trade')
    parser.add_argument('--initial-capital', type=float, default=100000,
                       help='Initial capital for paper trading')
    
    # Training settings
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of training episodes')
    parser.add_argument('--max-steps-per-episode', type=int, default=390,
                       help='Maximum steps per episode (6.5 hours * 60 minutes)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for PPO training')
    parser.add_argument('--ppo-epochs', type=int, default=4,
                       help='Number of PPO epochs per update')
    
    # Learning rates
    parser.add_argument('--lr-actor-critic', type=float, default=3e-4,
                       help='Learning rate for actor-critic')
    parser.add_argument('--lr-clstm', type=float, default=1e-3,
                       help='Learning rate for CLSTM encoder')
    
    # Checkpointing
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Save checkpoint every N episodes')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Validate Alpaca credentials
    if not all([os.getenv('APCA_API_KEY_ID'), os.getenv('APCA_API_SECRET_KEY')]):
        logger.error("Missing Alpaca API credentials. Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY")
        return
    
    # Create directories
    os.makedirs('checkpoints/live_paper', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Start training
    train_live_paper_trading(args)


if __name__ == '__main__':
    main()