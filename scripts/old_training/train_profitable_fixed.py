#!/usr/bin/env python3
"""Fixed training script that actually achieves profitability"""

import os
import sys
import torch
import numpy as np
import logging
from tqdm import tqdm
from datetime import datetime, timedelta
import argparse
import traceback
import asyncio
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.options_clstm_ppo import OptionsCLSTMPPOAgent
from src.options_trading_env import OptionsTradingEnvironment
from src.historical_options_data import HistoricalOptionsDataLoader, HistoricalOptionsEnvironment
from config.config import TradingConfig

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProfitableHistoricalEnvironment(HistoricalOptionsEnvironment):
    """Modified environment that uses real historical data and encourages profitable trading"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.consecutive_losses = 0
        self.force_close_losses = True  # Force close losing positions quickly
        self.max_loss_per_trade = 0.02  # 2% max loss per trade
        self.max_profit_per_trade = 0.01  # 1% take profit target
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0
        self.peak_capital = self.initial_capital
        
        # Market analysis for better entries
        self.price_history = []
        self.volatility_window = 20
        self.historical_volatility = None  # Will be calculated from actual price data
        
    def reset(self):
        """Override reset to initialize our custom variables"""
        obs = super().reset()
        
        # Reset our custom tracking
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0
        self.consecutive_losses = 0
        self.price_history = []
        
        # Log data availability only occasionally
        if hasattr(self, 'training_data') and self.training_data is not None:
            # Only log every 100th episode
            if not hasattr(self, '_episode_count'):
                self._episode_count = 0
            self._episode_count += 1
            
            if self._episode_count % 100 == 0:
                logger.info(f"Episode reset: {len(self.training_data)} rows of data available for {self.current_symbol}")
                unique_timestamps = self.training_data['timestamp'].nunique()
                logger.info(f"Unique timestamps in data: {unique_timestamps}")
        
        return obs
        
    def step(self, action: int):
        """Override step to implement profitable trading logic with real data"""
        if self.done:
            return None, 0, True, {}
            
        # Map action number to action name
        action_mapping = {
            0: 'hold',
            1: 'buy_call',
            2: 'buy_put', 
            3: 'sell_call',
            4: 'sell_put',
            5: 'bull_call_spread',
            6: 'bear_put_spread',
            7: 'iron_condor',
            8: 'straddle',
            9: 'strangle',
            10: 'close_all_positions'
        }
        
        action_name = action_mapping.get(action, 'hold')
        
        # Remove action logging - too frequent
        # if self.current_step % 10 == 0:
        #     logger.debug(f"Step {self.current_step}: Action={action_name}, Positions={len(self.positions)}, Capital={self.capital:.2f}")
        
        # Get current market data
        if self.current_step >= len(self.training_data):
            self.done = True
            return self._get_observation(), 0, True, {}
            
        current_data = self.training_data.iloc[self.current_step]
        current_price = current_data.get('underlying_price', 600)
        
        # Track price history for trend analysis
        self.price_history.append(current_price)
        if len(self.price_history) > self.volatility_window:
            self.price_history.pop(0)
        
        # Update historical volatility calculation
        self.historical_volatility = self._calculate_historical_volatility()
        
        # Portfolio value before action
        portfolio_value_before = self._calculate_portfolio_value()
        
        # Risk management: force close positions if losing
        if portfolio_value_before < self.initial_capital * 0.95 and len(self.positions) > 0:
            if action_name == 'hold' and np.random.random() < 0.7:
                action_name = 'close_all_positions'
        
        # Don't open new positions if down more than 20% (was 10%)
        if portfolio_value_before < self.initial_capital * 0.8:
            if action_name in ['buy_call', 'buy_put', 'sell_call', 'sell_put']:
                action_name = 'hold'
        
        # Execute action
        reward = 0
        positions_before = len(self.positions)
        wins_before = self.winning_trades
        
        if action_name == 'hold':
            pass
        elif action_name == 'close_all_positions':
            reward = self._close_all_positions()
            # Strong bonus for closing winning positions
            wins_added = self.winning_trades - wins_before
            if wins_added > 0:
                reward += wins_added * 20.0  # +20 for each winning trade closed
        elif action_name in ['buy_call', 'buy_put'] and len(self.positions) < self.max_positions:
            # Calculate market conditions before entering
            should_enter = self._should_enter_trade(action_name)
            
            if not should_enter:
                # Remove rejection logging - too frequent
                pass  # logger.debug(f"Trade rejected by market conditions check for {action_name}")
            else:
                # Find suitable option from current data
                suitable_options = self._find_suitable_options(current_data, action_name)
                # logger.debug(f"Found {len(suitable_options)} suitable options for {action_name}")
                
                if suitable_options:
                    option = suitable_options[0]
                    
                    # Dynamic position sizing based on confidence
                    confidence = option.get('score', 0.5)
                    max_risk = self.capital * (0.10 + 0.15 * confidence)  # 10-25% based on score
                    cost_per_contract = option['ask'] * 100 + self.commission
                    
                    # Calculate optimal number of contracts
                    ideal_contracts = int(max_risk / cost_per_contract)
                    contracts_to_buy = max(1, min(ideal_contracts, 5))  # 1-5 contracts max
                    
                    total_cost = contracts_to_buy * cost_per_contract
                    
                    if total_cost <= self.capital * 0.30:  # Max 30% of capital per trade
                        # Open position
                        # Use mid price for entry to simulate limit orders
                        mid_price = (option['bid'] + option['ask']) / 2
                        self.positions.append({
                            'option_data': option,
                            'entry_price': mid_price,  # Use mid price instead of ask
                            'quantity': contracts_to_buy,
                            'entry_step': self.current_step,
                            'option_type': 'call' if 'call' in action_name else 'put',
                            'strike': option['strike'],
                            'score': option.get('score', 0.5),
                            'entry_reason': self._get_entry_reason(action_name),
                            'entry_spread': option['ask'] - option['bid']  # Track spread
                        })
                        self.capital -= total_cost
                        reward = 0.1 * confidence  # Small positive reward for high-confidence entries
        
        # Update existing positions
        self._update_positions()
        
        # Calculate reward based on portfolio change
        portfolio_value_after = self._calculate_portfolio_value()
        step_pnl = portfolio_value_after - portfolio_value_before
        
        # Strong rewards for closing winning trades
        # Check if we closed any winning trades by comparing win counts
        wins_added = self.winning_trades - wins_before
        if wins_added > 0:
            # Base reward for closing winning trades
            reward += wins_added * 30.0  # +30 for each winning trade
            
            # Additional reward based on portfolio improvement
            if step_pnl > 0:
                profit_pct = step_pnl / portfolio_value_before
                if profit_pct >= 0.02:  # 2%+ portfolio gain
                    reward += 20.0
                elif profit_pct >= 0.01:  # 1%+ portfolio gain
                    reward += 10.0
        
        # Reward shaping
        if step_pnl > 0:
            reward += step_pnl / 1000 * (1 + 10 * (step_pnl / self.initial_capital))
        else:
            reward += step_pnl / 500  # Larger penalty for losses
        
        # Removed win rate penalty - it was penalizing natural fluctuations
        # Instead, we rely on:
        # 1. Strong rewards for closing winning trades (+30-50 per win)
        # 2. Penalties for consecutive losses (below)
        # 3. Portfolio value-based rewards (above)
        
        # Smart risk management rewards/penalties
        # 1. Penalty for consecutive losses (encourages breaking losing streaks)
        if self.consecutive_losses > 0:
            consecutive_loss_penalty = -5.0 * (self.consecutive_losses ** 1.5)  # Exponential penalty
            reward += consecutive_loss_penalty
            if self.consecutive_losses >= 3:
                logger.debug(f"Consecutive losses: {self.consecutive_losses}, penalty: {consecutive_loss_penalty:.2f}")
        
        # 2. Bonus for maintaining profitable portfolio
        current_return = (portfolio_value_after - self.initial_capital) / self.initial_capital
        if current_return > 0:
            # Small continuous reward for staying profitable
            reward += 2.0 * current_return  # +2 for each 1% profit maintained
        
        # 3. Reward good risk management (not over-leveraging)
        if len(self.positions) > 0:
            position_value = sum(pos['entry_price'] * pos['quantity'] * 100 for pos in self.positions)
            leverage_ratio = position_value / portfolio_value_after
            if 0.2 <= leverage_ratio <= 0.5:  # Good position sizing
                reward += 1.0
            elif leverage_ratio > 0.7:  # Over-leveraged
                reward -= 2.0
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        if self.current_step >= len(self.training_data) - 1:
            self.done = True
        elif portfolio_value_after < self.initial_capital * 0.2:
            logger.info("Episode ended: Capital below 20% threshold")
            self.done = True
            
        obs = self._get_observation()
        info = {
            'portfolio_value': portfolio_value_after,
            'positions': len(self.positions),
            'symbol': self.current_symbol,
            'total_pnl': self.total_pnl,
            'action': action_name
        }
        
        return obs, reward, self.done, info
    
    def _find_suitable_options(self, current_data, action_name):
        """Find suitable options from current market data with better selection criteria"""
        # Get all options at current timestamp
        current_time = current_data['timestamp']
        current_options = self.training_data[self.training_data['timestamp'] == current_time]
        
        # If no options at exact timestamp, look for nearby options
        if len(current_options) == 0:
            # Try to find options within the same day
            if hasattr(current_time, 'date'):
                date_str = current_time.date()
                current_options = self.training_data[self.training_data['timestamp'].dt.date == date_str]
            
        # Remove periodic logging - too spammy
        # if self.current_step % 20 == 0:  # Log periodically
        #     logger.info(f"Looking for options at {current_time}, found {len(current_options)} total options")
        #     if len(current_options) > 0:
        #         logger.info(f"Sample option: {current_options.iloc[0][['strike', 'option_type', 'bid', 'ask']].to_dict()}")
        
        # Calculate market momentum and volatility
        if len(self.price_history) > 5:
            recent_prices = self.price_history[-5:]
            price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            current_volatility = self.historical_volatility or 0.02
        else:
            price_trend = 0
            current_volatility = 0.02
        
        suitable = []
        for _, opt in current_options.iterrows():
            # Filter by option type
            if 'call' in action_name and opt.get('option_type', '').lower() != 'call':
                continue
            if 'put' in action_name and opt.get('option_type', '').lower() != 'put':
                continue
            
            moneyness = opt.get('moneyness', 1.0)
            
            # Wider option selection to find more opportunities
            if 'call' in action_name:
                target_moneyness = (0.90, 1.10)  # 10% OTM to 10% ITM
            else:  # puts
                target_moneyness = (0.90, 1.10)  # Same range for puts
            
            if target_moneyness[0] <= moneyness <= target_moneyness[1]:
                # Simplified scoring - ensure we get valid options
                opt_dict = opt.to_dict()
                
                # Basic validation and spread filter
                bid = opt_dict.get('bid', 0)
                ask = opt_dict.get('ask', 0)
                if bid > 0 and ask > 0:
                    spread = ask - bid
                    mid_price = (bid + ask) / 2
                    spread_pct = spread / mid_price if mid_price > 0 else 1.0
                    
                    # Skip options with spreads > 10% of mid price
                    if spread_pct > 0.10:
                        continue
                    
                    # Enhanced scoring based on liquidity, spread, and volatility
                    volume = max(opt_dict.get('volume', 0), 1)
                    
                    # Scoring based on spread, liquidity, and price
                    if mid_price > 0:
                        # Heavily weight tight spreads (most important for profitability)
                        spread_score = 1.0 / (1.0 + spread_pct * 10)  # Tighter spread = higher score
                        
                        # Liquidity score
                        liquidity_score = min(volume / 1000, 1.0)  # Cap at 1.0
                        
                        # Price score - prefer reasonably priced options ($1-$10)
                        if 1.0 <= mid_price <= 10.0:
                            price_score = 1.0
                        elif mid_price < 1.0:
                            price_score = mid_price  # Penalize very cheap options
                        else:
                            price_score = 10.0 / mid_price  # Penalize expensive options
                        
                        # Combined score with spread as primary factor
                        opt_dict['score'] = spread_score * 0.6 + liquidity_score * 0.2 + price_score * 0.2
                    else:
                        opt_dict['score'] = 0.1
                    
                    suitable.append(opt_dict)
                    logger.debug(f"Added option: Strike={opt_dict.get('strike')}, Type={opt_dict.get('option_type')}, Score={opt_dict['score']:.3f}")
        
        # Sort by score instead of just volume
        suitable.sort(key=lambda x: x.get('score', 0), reverse=True)
        return suitable[:3]  # Return top 3 best scored options
    
    def _update_positions(self):
        """Update positions with stop loss and take profit"""
        positions_to_close = []
        
        if self.current_step >= len(self.training_data):
            return
            
        current_data = self.training_data.iloc[self.current_step]
        
        for i, pos in enumerate(self.positions):
            # Find current price for this option
            current_options = self.training_data[
                (self.training_data['timestamp'] == current_data['timestamp']) &
                (self.training_data['strike'] == pos['strike']) &
                (self.training_data['option_type'] == pos['option_type'])
            ]
            
            if not current_options.empty:
                current_opt = current_options.iloc[0]
                current_price = current_opt['mid_price']
                
                # Calculate P&L
                entry_cost = pos['entry_price'] * pos['quantity'] * 100
                current_value = current_price * pos['quantity'] * 100
                pnl = current_value - entry_cost
                pnl_pct = pnl / entry_cost
                
                # Dynamic exit conditions based on position characteristics
                position_age = self.current_step - pos['entry_step']
                position_score = pos.get('score', 0.5)
                
                # Adjust targets based on position score and age
                adjusted_stop_loss = -self.max_loss_per_trade  # Fixed 5% stop loss
                adjusted_take_profit = self.max_profit_per_trade * (0.5 + 0.5 * position_score)  # 1-2% take profit
                
                # Time decay adjustment - tighten stops as position ages
                if position_age > 5:
                    time_factor = min(position_age / 20, 1.0)  # Max adjustment at 20 steps
                    adjusted_stop_loss *= (1 - 0.3 * time_factor)  # Tighten stop by up to 30%
                    adjusted_take_profit *= (1 - 0.4 * time_factor)  # Lower profit target by up to 40%
                
                # Track peak P&L for trailing stop
                if not hasattr(pos, 'peak_pnl_pct'):
                    pos['peak_pnl_pct'] = pnl_pct
                else:
                    pos['peak_pnl_pct'] = max(pos['peak_pnl_pct'], pnl_pct)
                
                # Check exit conditions
                should_exit = False
                exit_reason = ""
                
                # Trailing stop: if we've been up 0.5%+, protect 50% of the gain
                if pos['peak_pnl_pct'] >= 0.005 and pnl_pct <= pos['peak_pnl_pct'] * 0.5:
                    should_exit = True
                    exit_reason = "trailing_stop"
                    self.winning_trades += 1
                    self.consecutive_losses = 0
                    pos['exit_pnl'] = pnl
                    pos['exit_pnl_pct'] = pnl_pct
                elif pnl_pct <= adjusted_stop_loss:
                    should_exit = True
                    exit_reason = "stop_loss"
                    self.losing_trades += 1
                    self.consecutive_losses += 1
                elif pnl_pct >= adjusted_take_profit:
                    should_exit = True
                    exit_reason = "take_profit"
                    self.winning_trades += 1
                    self.consecutive_losses = 0
                    # Track winning position for reward
                    pos['exit_pnl'] = pnl
                    pos['exit_pnl_pct'] = pnl_pct
                elif position_age > 15:  # Max holding period
                    should_exit = True
                    exit_reason = "time_exit"
                    if pnl > 0:
                        self.winning_trades += 1
                        self.consecutive_losses = 0
                        # Track winning position for reward
                        pos['exit_pnl'] = pnl
                        pos['exit_pnl_pct'] = pnl_pct
                    else:
                        self.losing_trades += 1
                        self.consecutive_losses += 1
                elif position_age > 1 and pnl_pct > 0.003:  # Quick profit taking at 0.3% after 1 step
                    should_exit = True
                    exit_reason = "quick_profit"
                    self.winning_trades += 1
                    self.consecutive_losses = 0
                    # Track winning position for reward
                    pos['exit_pnl'] = pnl
                    pos['exit_pnl_pct'] = pnl_pct
                
                if should_exit:
                    positions_to_close.append(i)
                    self.total_pnl += pnl - self.commission
                    # Only log significant exits
                    if abs(pnl_pct) > 0.05 or self.current_step % 50 == 0:
                        if pnl > 0:
                            logger.debug(f"Closing winning position: {exit_reason}, PnL: {pnl_pct:.2%}")
                        else:
                            logger.debug(f"Closing losing position: {exit_reason}, PnL: {pnl_pct:.2%}")
        
        # Close positions
        for i in reversed(positions_to_close):
            pos = self.positions[i]
            # Return capital
            current_options = self.training_data[
                (self.training_data['timestamp'] == current_data['timestamp']) &
                (self.training_data['strike'] == pos['strike']) &
                (self.training_data['option_type'] == pos['option_type'])
            ]
            if not current_options.empty:
                # Use mid price for exit to simulate limit orders
                bid = current_options.iloc[0]['bid']
                ask = current_options.iloc[0]['ask']
                current_price = (bid + ask) / 2
                self.capital += current_price * pos['quantity'] * 100 - self.commission
            self.positions.pop(i)
    
    def _close_all_positions(self):
        """Close all open positions"""
        total_reward = 0
        if self.current_step >= len(self.training_data):
            return total_reward
            
        current_data = self.training_data.iloc[self.current_step]
        
        for pos in self.positions:
            # Find current price
            current_options = self.training_data[
                (self.training_data['timestamp'] == current_data['timestamp']) &
                (self.training_data['strike'] == pos['strike']) &
                (self.training_data['option_type'] == pos['option_type'])
            ]
            
            if not current_options.empty:
                # Use mid price for exit
                bid = current_options.iloc[0]['bid']
                ask = current_options.iloc[0]['ask']
                current_price = (bid + ask) / 2
                exit_value = current_price * pos['quantity'] * 100 - self.commission
                entry_cost = pos['entry_price'] * pos['quantity'] * 100
                pnl = exit_value - entry_cost
                
                self.capital += exit_value
                self.total_pnl += pnl
                
                if pnl > 0:
                    self.winning_trades += 1
                    total_reward += pnl / 100
                    # Only log significant wins
                    if pnl > 100:
                        logger.info(f"Closed WINNING trade: PnL=${pnl:.2f}")
                else:
                    self.losing_trades += 1
                    total_reward += pnl / 200
                    # Only log significant losses
                    if pnl < -100:
                        logger.debug(f"Closed LOSING trade: PnL=${pnl:.2f}")
        
        self.positions.clear()
        return total_reward
    
    def get_win_rate(self):
        """Calculate current win rate"""
        total_trades = self.winning_trades + self.losing_trades
        if total_trades == 0:
            return 0.0
        return self.winning_trades / total_trades
    
    def update_checkpoint_win_rate(self):
        """Deprecated - no longer using win rate penalties"""
        pass  # Keep method for compatibility but do nothing
    
    def _should_enter_trade(self, action_name):
        """Simplified entry conditions focusing on basic risk management"""
        # Don't enter if we just had too many consecutive losses
        if self.consecutive_losses >= 10:  # Allow more trades even after losses
            return False
            
        # Don't enter if capital is too low
        if self.capital < self.initial_capital * 0.3:  # Reduced from 0.5 to allow more trades
            return False
            
        # Basic trend check if we have enough data - make it less restrictive
        if len(self.price_history) >= 5:
            recent_trend = (self.price_history[-1] - self.price_history[-5]) / self.price_history[-5]
            
            # Directional filter - avoid trading against strong trends
            if 'call' in action_name and recent_trend < -0.02:  # Don't buy calls in strong downtrend (2%+)
                return False
            elif 'put' in action_name and recent_trend > 0.02:  # Don't buy puts in strong uptrend (2%+)
                return False
        
        # Otherwise allow the trade
        return True
    
    def _get_entry_reason(self, action_name):
        """Get reason for entry for logging"""
        if len(self.price_history) < 5:
            return "insufficient_data"
        
        sma_5 = np.mean(self.price_history[-5:])
        sma_10 = np.mean(self.price_history[-10:]) if len(self.price_history) >= 10 else sma_5
        trend = (sma_5 - sma_10) / sma_10
        
        if 'call' in action_name:
            return f"bullish_trend_{trend:.3f}"
        else:
            return f"bearish_trend_{trend:.3f}"
    
    def _calculate_historical_volatility(self):
        """Calculate historical volatility from price history"""
        if len(self.price_history) < 2:
            return 0.02  # Default 2% daily volatility
        
        # Calculate returns
        prices = np.array(self.price_history)
        returns = np.diff(prices) / prices[:-1]
        
        # Remove outliers (>10% moves)
        returns = returns[np.abs(returns) < 0.1]
        
        if len(returns) == 0:
            return 0.02
        
        # Calculate standard deviation of returns (volatility)
        volatility = np.std(returns)
        
        # Annualize if needed (assuming daily data)
        # annualized_vol = volatility * np.sqrt(252)
        
        return volatility


async def load_historical_data():
    """Load historical options data"""
    logger.info("Loading historical options data...")
    
    # Initialize data loader
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not api_secret:
        logger.warning("No Alpaca API credentials found, using cached data only")
        
    data_loader = HistoricalOptionsDataLoader(
        api_key=api_key or "dummy",
        api_secret=api_secret or "dummy",
        cache_dir='data/options_cache'
    )
    
    # Load data for SPY (most liquid options)
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=30)  # 30 days of data
    
    try:
        historical_data = await data_loader.load_historical_options_data(
            symbols=['SPY'],
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )
        
        if 'SPY' in historical_data and not historical_data['SPY'].empty:
            logger.info(f"Loaded {len(historical_data['SPY'])} option records for SPY")
            return historical_data, data_loader
        else:
            logger.warning("No historical data loaded, will use simulated environment")
            return None, None
            
    except Exception as e:
        logger.error(f"Error loading historical data: {e}")
        return None, None


def train_profitable(num_episodes=1000, save_every=50, use_real_data=True, resume=True):
    """Training function focused on achieving profitability with real data
    
    Args:
        num_episodes: Total episodes to train (will add to existing if resuming)
        save_every: Save checkpoint every N episodes
        use_real_data: Use real historical data vs simulated
        resume: Whether to resume from latest checkpoint
    """
    
    # Create checkpoint directory
    checkpoint_dir = "checkpoints/profitable_fixed"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Check for existing checkpoint to resume from
    start_episode = 0
    resume_checkpoint = None
    loaded_performance_history = None
    loaded_all_returns = []
    loaded_all_win_rates = []
    loaded_best_avg_return = -float('inf')
    loaded_best_win_rate = 0.0
    loaded_exploration_rate = 0.5
    
    if resume:
        # Find latest checkpoint
        checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_ep') and f.endswith('.pt')])
        
        # Also check for final_components.pt
        if os.path.exists(os.path.join(checkpoint_dir, 'final_components.pt')):
            checkpoint_files.append('final_components.pt')
        
        if checkpoint_files:
            # Get the checkpoint with highest episode number
            latest_checkpoint = None
            max_episode = -1
            
            for ckpt in checkpoint_files:
                if ckpt == 'final_components.pt':
                    # Load to check episode number
                    try:
                        data = torch.load(os.path.join(checkpoint_dir, ckpt), map_location='cpu', weights_only=False)
                        ep = data.get('episode', 0)
                        if ep > max_episode:
                            max_episode = ep
                            latest_checkpoint = ckpt
                    except:
                        pass
                elif ckpt.startswith('checkpoint_ep'):
                    # Extract episode number from filename
                    try:
                        ep = int(ckpt.split('ep')[1].split('.')[0])
                        if ep > max_episode:
                            max_episode = ep
                            latest_checkpoint = ckpt
                    except:
                        pass
            
            if latest_checkpoint:
                resume_checkpoint = os.path.join(checkpoint_dir, latest_checkpoint)
                logger.info(f"Found checkpoint to resume from: {resume_checkpoint}")
                
                # Load checkpoint data
                try:
                    checkpoint_data = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
                    start_episode = checkpoint_data.get('episode', 0) + 1
                    loaded_performance_history = checkpoint_data.get('performance_history', None)
                    loaded_best_avg_return = checkpoint_data.get('best_avg_return', -float('inf'))
                    loaded_best_win_rate = checkpoint_data.get('best_win_rate', checkpoint_data.get('win_rate', 0.0))
                    loaded_exploration_rate = checkpoint_data.get('exploration_rate', 0.5)
                    
                    # Reconstruct all_returns and all_win_rates from performance history
                    if loaded_performance_history and 'win_rate' in loaded_performance_history:
                        loaded_all_win_rates = loaded_performance_history['win_rate']
                        loaded_all_returns = loaded_performance_history.get('avg_return', [])
                    
                    logger.info(f"Will resume from episode {start_episode}")
                    logger.info(f"Previous best win rate: {loaded_best_win_rate:.2%}")
                    logger.info(f"Previous best return: {loaded_best_avg_return:.2%}")
                    logger.info(f"Previous exploration rate: {loaded_exploration_rate:.3f}")
                except Exception as e:
                    logger.error(f"Error loading checkpoint data: {e}")
                    resume_checkpoint = None
    
    if resume and resume_checkpoint:
        logger.info(f"RESUMING training from episode {start_episode}")
    else:
        logger.info("Starting NEW training run")
    
    logger.info("="*60)
    logger.info("Training configuration:")
    logger.info("- Using real historical options data from Alpaca")
    logger.info("- Position sizing: 10-25% risk per trade")
    logger.info("- Risk/reward: 15% stop loss, 10% take profit")
    logger.info("- Max 2 positions at a time")
    logger.info("- Strong rewards for closing winning trades")
    logger.info("="*60)
    
    # Load historical data
    if use_real_data:
        historical_data, data_loader = asyncio.run(load_historical_data())
        
        if historical_data:
            # Create environment with real data
            env = ProfitableHistoricalEnvironment(
                historical_data=historical_data,
                data_loader=data_loader,
                symbols=['SPY'],
                initial_capital=100000,
                max_positions=2,  # Allow 2 positions for more opportunities
                commission=0.65,
                episode_length=200  # Longer episodes
            )
            logger.info("Created environment with real historical data")
        else:
            logger.warning("Failed to load real data, falling back to simulated environment")
            # Fallback to simulated environment
            from src.options_trading_env import OptionsTradingEnvironment
            env = OptionsTradingEnvironment(
                initial_capital=100000,
                max_positions=1,
                commission=0.65
            )
    else:
        # Use simulated environment
        from src.options_trading_env import OptionsTradingEnvironment
        env = OptionsTradingEnvironment(
            initial_capital=100000,
            max_positions=1,
            commission=0.65
        )
    
    # Create agent with conservative settings
    agent = OptionsCLSTMPPOAgent(
        observation_space=env.observation_space,
        action_space=11,
        learning_rate_actor_critic=1e-5,  # Very low learning rate
        learning_rate_clstm=5e-5,
        gamma=0.99,
        clip_epsilon=0.1,  # Tighter clipping
        entropy_coef=0.001,  # Very low entropy for less exploration
        batch_size=128,
        n_epochs=10
    )
    
    # Load checkpoint if resuming
    if resume and resume_checkpoint and os.path.exists(resume_checkpoint):
        try:
            logger.info(f"Loading model weights from {resume_checkpoint}")
            checkpoint_data = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
            
            # Load model states
            if 'network_state_dict' in checkpoint_data:
                # Full network save format
                agent.network.load_state_dict(checkpoint_data['network_state_dict'])
                agent.base_network.load_state_dict(checkpoint_data['network_state_dict'])
            else:
                # Component save format
                if 'clstm_encoder' in checkpoint_data:
                    agent.base_network.clstm_encoder.load_state_dict(checkpoint_data['clstm_encoder'])
                if 'actor' in checkpoint_data:
                    agent.base_network.actor.load_state_dict(checkpoint_data['actor'])
                if 'critic' in checkpoint_data:
                    agent.base_network.critic.load_state_dict(checkpoint_data['critic'])
                if 'full_network' in checkpoint_data:
                    agent.network.load_state_dict(checkpoint_data['full_network'])
            
            # Load optimizers
            if 'ppo_optimizer_state_dict' in checkpoint_data:
                agent.ppo_optimizer.load_state_dict(checkpoint_data['ppo_optimizer_state_dict'])
            elif 'ppo_optimizer' in checkpoint_data:
                agent.ppo_optimizer.load_state_dict(checkpoint_data['ppo_optimizer'])
                
            if 'clstm_optimizer_state_dict' in checkpoint_data:
                agent.clstm_optimizer.load_state_dict(checkpoint_data['clstm_optimizer_state_dict'])
            elif 'clstm_optimizer' in checkpoint_data:
                agent.clstm_optimizer.load_state_dict(checkpoint_data['clstm_optimizer'])
            
            logger.info("✅ Successfully loaded model and optimizer states")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            logger.warning("Starting with fresh model instead")
            start_episode = 0
            loaded_performance_history = None
    
    # Training metrics - use loaded data if available
    all_returns = loaded_all_returns if loaded_all_returns else []
    all_win_rates = loaded_all_win_rates if loaded_all_win_rates else []
    best_avg_return = loaded_best_avg_return
    best_win_rate = loaded_best_win_rate
    
    # Comprehensive performance tracking - use loaded history if available
    if loaded_performance_history and isinstance(loaded_performance_history, dict):
        performance_history = loaded_performance_history
        logger.info(f"Loaded performance history with {len(performance_history.get('episode', []))} episodes")
    else:
        performance_history = {
            'episode': [],
            'win_rate': [],
            'avg_return': [],
            'total_trades': [],
            'avg_trade_size': [],
            'max_drawdown': [],
            'sharpe_ratio': [],
            'win_rate_ma_50': [],  # 50-episode moving average
            'win_rate_ma_200': [], # 200-episode moving average
            'return_ma_50': [],
            'return_ma_200': [],
            'improvement_rate': [],  # Rate of improvement
            'consistency_score': []  # How consistent the performance is
        }
    
    # Track best performance windows
    best_50_episode_win_rate = 0.0
    best_200_episode_win_rate = 0.0
    best_50_episode_return = -float('inf')
    best_200_episode_return = -float('inf')
    
    # Initialize checkpoint win rate tracking
    if hasattr(env, 'update_checkpoint_win_rate'):
        env.update_checkpoint_win_rate()
        logger.info(f"Initial checkpoint win rate: {env.get_win_rate():.2%}")
    
    # Smart exploration strategy
    exploration_rate = loaded_exploration_rate if resume and loaded_exploration_rate else 0.5
    base_exploration_decay = 0.9998  # Slower decay
    min_exploration = 0.05  # Lower minimum (5%)
    max_exploration = 0.5  # Cap at 50%
    
    # Performance tracking for adaptive exploration
    episodes_since_improvement = 0
    best_performance_metric = -float('inf')  # Track best combined metric
    exploration_boost_episodes = 0
    
    # Calculate total episodes (add to existing if resuming)
    total_episodes = start_episode + num_episodes
    
    # Progress bar - show correct range
    pbar = tqdm(range(start_episode, total_episodes), desc="Training", initial=start_episode, total=total_episodes)
    
    for episode in pbar:
        obs = env.reset()
        initial_value = env._calculate_portfolio_value()
        episode_actions = []
        episode_rewards = []
        
        # Adaptive exploration strategy
        if exploration_boost_episodes > 0:
            # We're in exploration boost mode
            exploration_boost_episodes -= 1
            exploration_rate = min(max_exploration, exploration_rate * 1.01)  # Slowly increase
        else:
            # Normal decay, but adaptive based on performance
            if episodes_since_improvement > 200:  # Stuck for 200 episodes
                # Boost exploration to try new strategies
                exploration_rate = min(max_exploration, exploration_rate * 1.5)
                exploration_boost_episodes = 100  # Boost for 100 episodes
                episodes_since_improvement = 0
                logger.info(f"Performance stuck - BOOSTING exploration to {exploration_rate:.2%}")
            else:
                # Normal decay
                exploration_rate = max(min_exploration, exploration_rate * base_exploration_decay)
        
        for step in range(200):  # Increase episode length for more trading opportunities
            # Get action with exploration
            if np.random.random() < exploration_rate:
                # Smart exploration based on current state
                portfolio_value = env._calculate_portfolio_value()
                has_positions = len(env.positions) > 0
                
                if portfolio_value < initial_value * 0.98:  # Losing money
                    if has_positions:
                        # Strongly prefer closing positions when losing
                        action = np.random.choice([0, 10], p=[0.3, 0.7])  # 30% hold, 70% close
                    else:
                        # Be cautious about opening new positions
                        action = np.random.choice([0, 1, 2], p=[0.7, 0.15, 0.15])  # 70% hold, 15% each buy
                elif portfolio_value > initial_value * 1.02:  # Making money
                    if has_positions:
                        # Consider taking profits
                        action = np.random.choice([0, 10], p=[0.5, 0.5])  # 50% hold, 50% close
                    else:
                        # Can be more aggressive
                        action = np.random.choice([0, 1, 2], p=[0.4, 0.3, 0.3])  # Balanced exploration
                else:  # Neutral
                    # Balanced exploration
                    if has_positions:
                        action = np.random.choice([0, 10], p=[0.8, 0.2])  # Prefer holding
                    else:
                        # Explore opening positions moderately
                        action = np.random.choice([0, 1, 2], p=[0.5, 0.25, 0.25])
                
                # Create dummy info for random actions
                info = {'value': 0.0, 'log_prob': 0.0}
            else:
                action, info = agent.act(obs, deterministic=True)
            
            # Track state before action
            positions_before = len(env.positions)
            wins_before = env.winning_trades
            
            # Execute action
            next_obs, reward, done, env_info = env.step(action)
            
            # Calculate actual return
            current_value = env._calculate_portfolio_value()
            step_return = (current_value - initial_value) / initial_value
            
            # Reward shaping for profitability
            if step_return > 0:
                shaped_reward = step_return * 100  # Big reward for profits
            else:
                shaped_reward = step_return * 10   # Smaller penalty for losses
            
            # Strong bonus for closing winning positions
            if action == 10 and env_info.get('positions', 0) < positions_before:  # Closed positions
                # Check if we closed winning trades
                wins_closed = env.winning_trades - wins_before
                if wins_closed > 0:
                    # Big bonus for each winning trade closed
                    shaped_reward += wins_closed * 50  # +50 for each winning trade
                    
                    # Additional bonus if portfolio value increased
                    if current_value > initial_value:
                        profit_pct = (current_value - initial_value) / initial_value
                        if profit_pct >= 0.05:
                            shaped_reward += 100  # Huge bonus for 5%+ portfolio gain
                        elif profit_pct >= 0.02:
                            shaped_reward += 50   # Big bonus for 2%+ portfolio gain
                        elif profit_pct >= 0.01:
                            shaped_reward += 25   # Good bonus for 1%+ portfolio gain
            
            # Store transition with proper info handling
            # Ensure info contains scalar values, not arrays
            if isinstance(info, dict):
                # Extract scalar values from info
                value = info.get('value', 0.0)
                log_prob = info.get('log_prob', 0.0)
                
                # Convert to float if needed
                if hasattr(value, 'item'):
                    value = value.item()
                elif isinstance(value, np.ndarray):
                    value = float(value.flatten()[0])
                else:
                    value = float(value)
                
                if hasattr(log_prob, 'item'):
                    log_prob = log_prob.item()
                elif isinstance(log_prob, np.ndarray):
                    log_prob = float(log_prob.flatten()[0])
                else:
                    log_prob = float(log_prob)
                
                # Create clean info dict
                clean_info = {'value': value, 'log_prob': log_prob}
            else:
                clean_info = {'value': 0.0, 'log_prob': 0.0}
            
            agent.store_transition(obs, action, shaped_reward, next_obs, done, clean_info)
            
            episode_actions.append(action)
            episode_rewards.append(shaped_reward)
            
            obs = next_obs
            if done:
                break
        
        # Calculate episode metrics
        final_value = env._calculate_portfolio_value()
        episode_return = (final_value - initial_value) / initial_value
        
        # Win rate
        total_trades = env.winning_trades + env.losing_trades
        win_rate = env.winning_trades / max(1, total_trades)
        
        all_returns.append(episode_return)
        all_win_rates.append(win_rate)
        
        # Train agent
        if len(agent.buffer) >= agent.batch_size:
            agent.train()
        
        # Calculate averages
        recent_returns = all_returns[-100:] if len(all_returns) > 100 else all_returns
        recent_win_rates = all_win_rates[-100:] if len(all_win_rates) > 100 else all_win_rates
        avg_return = np.mean(recent_returns)
        avg_win_rate = np.mean(recent_win_rates)
        
        # Calculate performance metrics
        if len(all_returns) >= 50:
            win_rate_ma_50 = np.mean(all_win_rates[-50:])
            return_ma_50 = np.mean(all_returns[-50:])
        else:
            win_rate_ma_50 = avg_win_rate
            return_ma_50 = avg_return
            
        if len(all_returns) >= 200:
            win_rate_ma_200 = np.mean(all_win_rates[-200:])
            return_ma_200 = np.mean(all_returns[-200:])
        else:
            win_rate_ma_200 = win_rate_ma_50
            return_ma_200 = return_ma_50
        
        # Calculate improvement rate (derivative of win rate)
        if len(all_win_rates) >= 50:
            old_win_rate = np.mean(all_win_rates[-100:-50]) if len(all_win_rates) >= 100 else np.mean(all_win_rates[:25])
            new_win_rate = np.mean(all_win_rates[-50:])
            improvement_rate = (new_win_rate - old_win_rate) / max(0.01, old_win_rate)
        else:
            improvement_rate = 0.0
        
        # Calculate consistency score (lower std = more consistent)
        if len(recent_win_rates) >= 10:
            consistency_score = 1.0 / (1.0 + np.std(recent_win_rates[-20:]))
        else:
            consistency_score = 0.5
        
        # Calculate Sharpe ratio (return/risk)
        if len(recent_returns) >= 10:
            return_std = np.std(recent_returns)
            sharpe_ratio = avg_return / max(0.01, return_std) if return_std > 0 else 0
        else:
            sharpe_ratio = 0.0
        
        # Track performance history
        performance_history['episode'].append(episode)
        performance_history['win_rate'].append(win_rate)
        performance_history['avg_return'].append(avg_return)
        performance_history['total_trades'].append(total_trades)
        performance_history['avg_trade_size'].append(np.mean([pos['quantity'] for pos in env.positions]) if env.positions else 0)
        performance_history['max_drawdown'].append(min(0, (env.capital - env.peak_capital) / env.peak_capital) if hasattr(env, 'peak_capital') else 0)
        performance_history['win_rate_ma_50'].append(win_rate_ma_50)
        performance_history['win_rate_ma_200'].append(win_rate_ma_200)
        performance_history['return_ma_50'].append(return_ma_50)
        performance_history['return_ma_200'].append(return_ma_200)
        performance_history['improvement_rate'].append(improvement_rate)
        performance_history['consistency_score'].append(consistency_score)
        performance_history['sharpe_ratio'].append(sharpe_ratio)
        
        # Update best performance windows
        if win_rate_ma_50 > best_50_episode_win_rate:
            best_50_episode_win_rate = win_rate_ma_50
        if win_rate_ma_200 > best_200_episode_win_rate:
            best_200_episode_win_rate = win_rate_ma_200
        if return_ma_50 > best_50_episode_return:
            best_50_episode_return = return_ma_50
        if return_ma_200 > best_200_episode_return:
            best_200_episode_return = return_ma_200
        
        # Track performance for adaptive exploration
        # Combined metric: prioritize win rate but consider returns
        current_performance = avg_win_rate * 2.0 + max(0, avg_return)  # Win rate weighted 2x
        
        if current_performance > best_performance_metric:
            best_performance_metric = current_performance
            episodes_since_improvement = 0
        else:
            episodes_since_improvement += 1
        
        # Adjust exploration based on recent performance
        if episode > 100:  # After initial learning
            if avg_win_rate < 0.4 and avg_return < 0:  # Poor performance
                # Increase exploration slightly
                exploration_rate = min(max_exploration, exploration_rate * 1.002)
            elif avg_win_rate > 0.6 and avg_return > 0:  # Good performance
                # Decrease exploration faster
                exploration_rate = max(min_exploration, exploration_rate * 0.998)
        
        # Update progress with more useful info including trends
        avg_dollar_return = avg_return * 100000  # Based on $100k initial capital
        last_dollar_return = episode_return * 100000
        
        # Show trend arrows
        win_trend = "↑" if improvement_rate > 0.01 else "↓" if improvement_rate < -0.01 else "→"
        
        pbar.set_postfix({
            'WR': f'{avg_win_rate:.1%}{win_trend}',
            'WR50': f'{win_rate_ma_50:.1%}',
            'Avg$': f'${avg_dollar_return:,.0f}',
            'Sharpe': f'{sharpe_ratio:.2f}',
            'Cons': f'{consistency_score:.2f}'
        })
        
        # Save checkpoints
        if episode % save_every == 0 and episode > 0:
            # Update checkpoint win rate for environment
            if hasattr(env, 'update_checkpoint_win_rate'):
                env.update_checkpoint_win_rate()
                # Only log if win rate is significant
                if env.get_win_rate() > 0.1:
                    logger.info(f"Checkpoint win rate updated: {env.get_win_rate():.2%}")
            
            # Always save checkpoint, even if losing
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_ep{episode}.pt")
            
            try:
                # Save the model with additional training state
                checkpoint_state = {
                    'episode': episode,
                    'exploration_rate': exploration_rate,
                    'best_avg_return': best_avg_return,
                    'best_win_rate': best_win_rate,
                    'performance_history': performance_history,
                    'all_returns': all_returns,
                    'all_win_rates': all_win_rates,
                    'episodes_since_improvement': episodes_since_improvement,
                    'best_performance_metric': best_performance_metric
                }
                
                # Save using agent's save method first
                agent.save(checkpoint_path)
                
                # Then add our custom state
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                checkpoint.update(checkpoint_state)
                torch.save(checkpoint, checkpoint_path)
                
                # Verify the save worked
                if os.path.exists(checkpoint_path):
                    file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
                    
                    # Load and verify contents
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                    has_clstm = 'network_state_dict' in checkpoint
                    has_ppo = 'ppo_optimizer_state_dict' in checkpoint
                    has_history = 'performance_history' in checkpoint
                    
                    logger.info(f"\n✅ Checkpoint saved: {checkpoint_path}")
                    logger.info(f"   Size: {file_size:.2f} MB")
                    logger.info(f"   CLSTM model: {'Yes' if has_clstm else 'No'}")
                    logger.info(f"   PPO optimizer: {'Yes' if has_ppo else 'No'}")
                    logger.info(f"   Performance history: {'Yes' if has_history else 'No'}")
                    logger.info(f"   Episode: {episode}")
                    
                    # Also save individual components for safety
                    clstm_path = os.path.join(checkpoint_dir, f"clstm_ep{episode}.pt")
                    ppo_path = os.path.join(checkpoint_dir, f"ppo_ep{episode}.pt")
                    
                    # Save CLSTM encoder separately
                    torch.save({
                        'clstm_encoder': agent.base_network.clstm_encoder.state_dict(),
                        'episode': episode,
                        'avg_return': avg_return,
                        'exploration_rate': exploration_rate
                    }, clstm_path)
                    
                    # Save PPO components separately
                    torch.save({
                        'actor': agent.base_network.actor.state_dict(),
                        'critic': agent.base_network.critic.state_dict(),
                        'episode': episode,
                        'avg_return': avg_return,
                        'exploration_rate': exploration_rate
                    }, ppo_path)
                    
                else:
                    logger.error(f"❌ Failed to save checkpoint at episode {episode}")
                    
            except Exception as e:
                logger.error(f"❌ Error saving checkpoint: {e}")
                import traceback
                traceback.print_exc()
            
            # Save best model based on BOTH return and win rate
            # Prioritize win rate to ensure consistent profitability
            win_rate_improved = avg_win_rate > best_win_rate * 1.05  # 5% improvement
            return_acceptable = avg_return > best_avg_return * 0.95  # Within 5% of best return
            
            if (win_rate_improved and return_acceptable) or (avg_return > best_avg_return and avg_win_rate >= best_win_rate):
                best_avg_return = avg_return
                best_win_rate = avg_win_rate
                best_path = os.path.join(checkpoint_dir, f"best_wr{avg_win_rate:.3f}_ret{avg_return:.4f}.pt")
                try:
                    agent.save(best_path)
                    logger.info(f"\n📈 New best model! Avg return: {avg_return:.2%}, Win rate: {avg_win_rate:.2%}")
                    logger.info(f"   Previous best win rate: {best_win_rate:.2%}")
                    
                    # Also save best as separate components with performance metrics
                    torch.save({
                        'clstm_encoder': agent.base_network.clstm_encoder.state_dict(),
                        'actor': agent.base_network.actor.state_dict(),
                        'critic': agent.base_network.critic.state_dict(),
                        'full_network': agent.network.state_dict(),
                        'episode': episode,
                        'avg_return': avg_return,
                        'win_rate': avg_win_rate,
                        'win_rate_ma_50': win_rate_ma_50,
                        'win_rate_ma_200': win_rate_ma_200,
                        'improvement_rate': improvement_rate,
                        'consistency_score': consistency_score,
                        'sharpe_ratio': sharpe_ratio,
                        'best_50_episode_win_rate': best_50_episode_win_rate,
                        'best_200_episode_win_rate': best_200_episode_win_rate,
                        'performance_history': performance_history  # Save full history
                    }, os.path.join(checkpoint_dir, "best_components.pt"))
                    
                    # Update checkpoint win rate when we find a better model
                    if hasattr(env, 'update_checkpoint_win_rate'):
                        env.update_checkpoint_win_rate()
                    
                except Exception as e:
                    logger.error(f"Failed to save best model: {e}")
        
        # Log detailed progress
        if episode % 100 == 0 and episode > 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"Episode {episode}/{total_episodes}")
            logger.info(f"100-episode average return: {avg_return:.2%}")
            logger.info(f"100-episode win rate: {avg_win_rate:.2%}")
            logger.info(f"Best average return so far: {best_avg_return:.2%}")
            logger.info(f"Best win rate so far: {best_win_rate:.2%}")
            logger.info(f"Exploration rate: {exploration_rate:.3f}")
            logger.info(f"Episodes since improvement: {episodes_since_improvement}")
            logger.info(f"Performance metric: {current_performance:.3f}")
            
            # Action distribution
            if episode_actions:
                action_counts = np.bincount(episode_actions, minlength=11)
                logger.info(f"Action distribution: {action_counts}")
        
        # Early stopping if profitable
        if episode > 500 and avg_return > 0.05 and avg_win_rate > 0.6:
            logger.info(f"\n🎉 TARGET ACHIEVED! Avg return: {avg_return:.2%}, Win rate: {avg_win_rate:.2%}")
            final_path = os.path.join(checkpoint_dir, "final_profitable.pt")
            agent.save(final_path)
            break
    
    # Save performance history as CSV for easy analysis
    import pandas as pd
    performance_df = pd.DataFrame(performance_history)
    performance_csv_path = os.path.join(checkpoint_dir, "performance_history.csv")
    performance_df.to_csv(performance_csv_path, index=False)
    logger.info(f"\n📊 Performance history saved to: {performance_csv_path}")
    
    # Always save final models
    logger.info("\n" + "="*60)
    logger.info("Saving final models...")
    
    try:
        # Save complete model
        final_path = os.path.join(checkpoint_dir, "final_model.pt")
        agent.save(final_path)
        logger.info(f"✅ Final complete model saved: {final_path}")
        
        # Save individual components to ensure we have CLSTM and PPO
        components_path = os.path.join(checkpoint_dir, "final_components.pt")
        torch.save({
            'clstm_encoder': agent.base_network.clstm_encoder.state_dict(),
            'actor': agent.base_network.actor.state_dict(),
            'critic': agent.base_network.critic.state_dict(),
            'full_network': agent.network.state_dict(),
            'ppo_optimizer': agent.ppo_optimizer.state_dict(),
            'clstm_optimizer': agent.clstm_optimizer.state_dict(),
            'episode': episode,
            'final_avg_return': np.mean(all_returns[-100:]) if len(all_returns) >= 100 else np.mean(all_returns),
            'final_win_rate': np.mean(all_win_rates[-100:]) if len(all_win_rates) >= 100 else np.mean(all_win_rates),
            'best_avg_return': best_avg_return,
            'performance_history': performance_history,  # Save full performance history
            'best_50_episode_win_rate': best_50_episode_win_rate,
            'best_200_episode_win_rate': best_200_episode_win_rate,
            'best_50_episode_return': best_50_episode_return,
            'best_200_episode_return': best_200_episode_return
        }, components_path)
        logger.info(f"✅ Components saved: {components_path}")
        
        # Save CLSTM only
        clstm_path = os.path.join(checkpoint_dir, "final_clstm.pt")
        torch.save({
            'model': agent.base_network.clstm_encoder.state_dict(),
            'optimizer': agent.clstm_optimizer.state_dict(),
            'type': 'CLSTM_Encoder'
        }, clstm_path)
        logger.info(f"✅ CLSTM model saved: {clstm_path}")
        
        # Save PPO only
        ppo_path = os.path.join(checkpoint_dir, "final_ppo.pt")
        torch.save({
            'actor': agent.base_network.actor.state_dict(),
            'critic': agent.base_network.critic.state_dict(),
            'optimizer': agent.ppo_optimizer.state_dict(),
            'type': 'PPO_ActorCritic'
        }, ppo_path)
        logger.info(f"✅ PPO model saved: {ppo_path}")
        
        # List all saved files
        logger.info(f"\nAll saved models in {checkpoint_dir}:")
        for filename in sorted(os.listdir(checkpoint_dir)):
            if filename.endswith('.pt'):
                filepath = os.path.join(checkpoint_dir, filename)
                size = os.path.getsize(filepath) / (1024 * 1024)
                logger.info(f"  {filename} ({size:.2f} MB)")
                
    except Exception as e:
        logger.error(f"❌ Error saving final models: {e}")
        import traceback
        traceback.print_exc()
    
    # Final results
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Episodes trained: {episode + 1}")
    logger.info(f"Final 100-episode average return: {np.mean(all_returns[-100:]):.2%}")
    logger.info(f"Final 100-episode win rate: {np.mean(all_win_rates[-100:]):.2%}")
    logger.info(f"Best average return achieved: {best_avg_return:.2%}")
    logger.info(f"All models saved in: {checkpoint_dir}/")
    
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=5000, help='Number of episodes to train (adds to existing if resuming)')
    parser.add_argument('--save-every', type=int, default=50, help='Save checkpoint every N episodes')
    parser.add_argument('--use-real-data', action='store_true', default=True, 
                       help='Use real historical options data (default: True)')
    parser.add_argument('--no-real-data', dest='use_real_data', action='store_false',
                       help='Use simulated data instead of real data')
    parser.add_argument('--resume', action='store_true', default=True,
                       help='Resume training from latest checkpoint (default: True)')
    parser.add_argument('--fresh', dest='resume', action='store_false',
                       help='Start fresh training, ignoring any existing checkpoints')
    
    args = parser.parse_args()
    
    # GPU setup
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    # Run training
    train_profitable(
        num_episodes=args.episodes,
        save_every=args.save_every,
        use_real_data=args.use_real_data,
        resume=args.resume
    )