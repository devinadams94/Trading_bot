#!/usr/bin/env python3
"""
Simplified Paper Trading Bot for PPO-LSTM Options Model
This is a paper-trading only version that's easier to run and debug
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
import json

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.options_clstm_ppo import OptionsCLSTMPPONetwork
from src.options_trading_env import OptionsTradingEnvironment, OptionContract

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('paper_trading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PaperPosition:
    """Track a paper trading position"""
    symbol: str
    option_type: str  # 'call' or 'put'
    strike: float
    expiration: datetime
    quantity: int
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    
    @property
    def pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.quantity * 100
    
    @property
    def pnl_percent(self) -> float:
        if self.entry_price > 0:
            return (self.current_price - self.entry_price) / self.entry_price
        return 0.0


class SimplePaperTradingBot:
    """Simplified paper trading bot using trained model"""
    
    def __init__(self, model_path: str, initial_capital: float = 100000):
        self.model_path = model_path
        self.initial_capital = initial_capital
        self.capital = initial_capital

        # Initialize environment first (needed for model loading)
        self.env = OptionsTradingEnvironment(
            initial_capital=initial_capital,
            max_positions=5,
            commission=0.65
        )

        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        
        # Paper trading state
        self.positions: Dict[str, PaperPosition] = {}
        self.trade_history = []
        self.daily_pnl = 0
        self.total_trades = 0
        self.winning_trades = 0
        
        logger.info(f"Initialized Paper Trading Bot")
        logger.info(f"Model: {model_path}")
        logger.info(f"Initial Capital: ${initial_capital:,.2f}")
    
    def _load_model(self) -> OptionsCLSTMPPONetwork:
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        # Initialize model
        model = OptionsCLSTMPPONetwork(
            observation_space=self.env.observation_space,
            action_dim=11
        ).to(self.device)
        
        # Load checkpoint
        logger.info(f"Loading model from {self.model_path}")
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        except:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        model.load_state_dict(checkpoint['network_state_dict'])
        model.eval()
        
        # Log model info
        episode = checkpoint.get('episode', 'unknown')
        logger.info(f"Model trained for {episode} episodes")
        
        return model
    
    def generate_simulated_market_data(self, symbol: str) -> Dict[str, Any]:
        """Generate realistic market data for simulation"""
        # Simulate realistic stock prices
        base_prices = {'SPY': 550, 'QQQ': 480, 'IWM': 220, 'AAPL': 195, 'MSFT': 430}
        base_price = base_prices.get(symbol, 100)
        
        # Add some randomness
        price = base_price * (1 + np.random.normal(0, 0.01))
        
        return {
            'symbol': symbol,
            'price': price,
            'volume': np.random.randint(1000000, 10000000),
            'bid': price - 0.01,
            'ask': price + 0.01
        }
    
    def generate_options_chain(self, symbol: str, stock_price: float) -> List[OptionContract]:
        """Generate realistic options chain"""
        options = []
        
        # Generate strikes around current price
        strikes = np.arange(
            int(stock_price * 0.9), 
            int(stock_price * 1.1) + 1, 
            5 if stock_price > 100 else 1
        )
        
        # Generate for next 4 Fridays
        today = datetime.now()
        expiration_dates = []
        for i in range(1, 5):
            days_ahead = (4 - today.weekday()) % 7 + 7 * i  # Next Fridays
            exp_date = today + timedelta(days=days_ahead)
            expiration_dates.append(exp_date)
        
        for strike in strikes:
            for exp_date in expiration_dates[:2]:  # Just next 2 expirations
                days_to_expiry = (exp_date - today).days
                
                for option_type in ['call', 'put']:
                    # Calculate realistic option price
                    moneyness = stock_price / strike
                    time_value = np.sqrt(days_to_expiry / 365) * stock_price * 0.02
                    
                    if option_type == 'call':
                        intrinsic = max(0, stock_price - strike)
                    else:
                        intrinsic = max(0, strike - stock_price)
                    
                    option_price = intrinsic + time_value + np.random.uniform(0, 0.5)
                    
                    # Create option contract
                    option = OptionContract(
                        symbol=symbol,
                        strike=strike,
                        expiration=exp_date,
                        option_type=option_type,
                        bid=max(0.01, option_price - 0.05),
                        ask=max(0.01, option_price + 0.05),
                        last_price=option_price,
                        volume=np.random.randint(10, 1000),
                        open_interest=np.random.randint(100, 10000),
                        implied_volatility=0.15 + np.random.uniform(-0.05, 0.10),
                        delta=0.5 if option_type == 'call' else -0.5,
                        gamma=0.01,
                        theta=-0.05,
                        vega=0.10,
                        rho=0.01
                    )
                    
                    options.append(option)
        
        return options
    
    def prepare_state(self, symbol: str, market_data: Dict, options_chain: List[OptionContract]) -> Dict[str, np.ndarray]:
        """Prepare state for model input"""
        # Price history (simplified)
        price_history = np.zeros((self.env.lookback_window, 5), dtype=np.float32)
        price_history[-1] = [
            market_data['price'] * 0.99,  # open
            market_data['price'] * 1.01,  # high
            market_data['price'] * 0.98,  # low
            market_data['price'],         # close
            market_data['volume']         # volume
        ]
        
        # Technical indicators (simplified)
        technical_indicators = np.random.randn(20).astype(np.float32) * 0.1
        
        # Options chain features
        options_features = np.zeros((20, 15), dtype=np.float32)
        for i, option in enumerate(options_chain[:20]):
            options_features[i] = [
                option.strike,
                option.bid,
                option.ask,
                option.last_price,
                option.volume,
                option.open_interest,
                option.implied_volatility,
                option.delta,
                option.gamma,
                option.theta,
                option.vega,
                option.rho,
                1 if option.option_type == 'call' else 0,
                (option.expiration - datetime.now()).days,
                option.strike / market_data['price']
            ]
        
        # Portfolio state
        portfolio_value = self.capital + sum(pos.pnl for pos in self.positions.values())
        portfolio_state = np.array([
            self.capital,
            len(self.positions),
            portfolio_value,
            self.daily_pnl,
            self.total_trades
        ], dtype=np.float32)
        
        # Greeks summary
        greeks_summary = np.zeros(5, dtype=np.float32)
        
        return {
            'price_history': price_history,
            'technical_indicators': technical_indicators,
            'options_chain': options_features,
            'portfolio_state': portfolio_state,
            'greeks_summary': greeks_summary
        }
    
    def get_action(self, state: Dict[str, np.ndarray]) -> Tuple[int, float]:
        """Get action from model"""
        with torch.no_grad():
            # Process state
            features = []
            for key in ['price_history', 'technical_indicators', 'options_chain', 
                       'portfolio_state', 'greeks_summary']:
                if key in state:
                    tensor = torch.tensor(state[key], dtype=torch.float32).to(self.device)
                    features.append(tensor.flatten())
            
            combined = torch.cat(features, dim=0).unsqueeze(0).unsqueeze(0)
            
            # Get action
            lstm_features = self.model.clstm_encoder(combined).squeeze(0)
            action_logits = self.model.actor(lstm_features.unsqueeze(0))
            action_probs = torch.softmax(action_logits, dim=-1)
            
            # Take most likely action (deterministic for paper trading)
            action = torch.argmax(action_probs, dim=-1).item()
            confidence = action_probs[0, action].item()
        
        return action, confidence
    
    def execute_trade(self, action: int, symbol: str, options_chain: List[OptionContract]) -> Dict[str, Any]:
        """Execute paper trade"""
        action_map = {
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
        
        action_name = action_map.get(action, 'hold')
        result = {'action': action_name, 'success': False, 'message': ''}
        
        if action_name == 'hold':
            result['success'] = True
            result['message'] = 'Holding positions'
            
        elif action_name == 'close_all_positions':
            self.close_all_positions()
            result['success'] = True
            result['message'] = 'Closed all positions'
            
        elif action_name in ['buy_call', 'buy_put']:
            if len(self.positions) >= 5:
                result['message'] = 'Max positions reached'
            else:
                # Select ATM option
                option_type = 'call' if 'call' in action_name else 'put'
                selected = None
                
                for opt in options_chain:
                    if opt.option_type == option_type:
                        selected = opt
                        break
                
                if selected:
                    # Calculate position size
                    position_size = min(10, int(self.capital * 0.1 / (selected.ask * 100)))
                    if position_size > 0:
                        cost = position_size * selected.ask * 100 + 0.65 * position_size
                        
                        if cost <= self.capital:
                            # Open position
                            position_id = f"{symbol}_{option_type}_{len(self.positions)}"
                            self.positions[position_id] = PaperPosition(
                                symbol=symbol,
                                option_type=option_type,
                                strike=selected.strike,
                                expiration=selected.expiration,
                                quantity=position_size,
                                entry_price=selected.ask,
                                entry_time=datetime.now(),
                                current_price=selected.ask
                            )
                            
                            self.capital -= cost
                            result['success'] = True
                            result['message'] = f'Bought {position_size} {option_type} contracts'
                            
                            logger.info(f"Opened {action_name}: {position_size} contracts @ ${selected.ask:.2f}")
        
        return result
    
    def update_positions(self, options_chains: Dict[str, List[OptionContract]]):
        """Update position values and check exits"""
        positions_to_close = []
        
        for pos_id, position in self.positions.items():
            # Find current option price
            if position.symbol in options_chains:
                for opt in options_chains[position.symbol]:
                    if (opt.option_type == position.option_type and 
                        opt.strike == position.strike):
                        position.current_price = opt.bid  # Use bid for exit
                        break
            
            # Check exit conditions
            if position.pnl_percent >= 0.20:  # 20% profit
                logger.info(f"Taking profit on {pos_id}: {position.pnl_percent:.1%}")
                positions_to_close.append(pos_id)
            elif position.pnl_percent <= -0.10:  # 10% loss
                logger.info(f"Stop loss on {pos_id}: {position.pnl_percent:.1%}")
                positions_to_close.append(pos_id)
            elif position.expiration <= datetime.now() + timedelta(days=1):
                logger.info(f"Closing expiring position {pos_id}")
                positions_to_close.append(pos_id)
        
        # Close positions
        for pos_id in positions_to_close:
            self.close_position(pos_id)
    
    def close_position(self, position_id: str):
        """Close a position"""
        if position_id in self.positions:
            position = self.positions[position_id]
            proceeds = position.quantity * position.current_price * 100 - 0.65 * position.quantity
            self.capital += proceeds
            
            pnl = position.pnl
            self.daily_pnl += pnl
            self.total_trades += 1
            
            if pnl > 0:
                self.winning_trades += 1
            
            # Record trade
            self.trade_history.append({
                'symbol': position.symbol,
                'type': position.option_type,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'exit_price': position.current_price,
                'pnl': pnl,
                'pnl_percent': position.pnl_percent,
                'holding_time': (datetime.now() - position.entry_time).seconds / 3600
            })
            
            del self.positions[position_id]
            
            logger.info(f"Closed {position_id} - P&L: ${pnl:.2f} ({position.pnl_percent:.1%})")
    
    def close_all_positions(self):
        """Close all open positions"""
        position_ids = list(self.positions.keys())
        for pos_id in position_ids:
            self.close_position(pos_id)
    
    def run_trading_session(self, symbols: List[str], duration_minutes: int = 60):
        """Run a paper trading session"""
        logger.info(f"Starting paper trading session for {duration_minutes} minutes")
        logger.info(f"Symbols: {symbols}")
        
        start_time = time.time()
        iteration = 0
        
        while (time.time() - start_time) < duration_minutes * 60:
            iteration += 1
            logger.info(f"\n--- Iteration {iteration} ---")
            
            # Generate options chains for all symbols
            all_options = {}
            for symbol in symbols:
                market_data = self.generate_simulated_market_data(symbol)
                options_chain = self.generate_options_chain(symbol, market_data['price'])
                all_options[symbol] = options_chain
                
                # Prepare state
                state = self.prepare_state(symbol, market_data, options_chain)
                
                # Get action
                action, confidence = self.get_action(state)
                
                # Execute if confident
                if confidence > 0.3:
                    result = self.execute_trade(action, symbol, options_chain)
                    if result['success'] and result['action'] != 'hold':
                        logger.info(f"{symbol}: {result['message']} (confidence: {confidence:.2%})")
            
            # Update all positions
            self.update_positions(all_options)
            
            # Log status
            portfolio_value = self.capital + sum(pos.pnl for pos in self.positions.values())
            win_rate = self.winning_trades / max(1, self.total_trades)
            
            logger.info(f"Portfolio Value: ${portfolio_value:,.2f}")
            logger.info(f"Daily P&L: ${self.daily_pnl:,.2f}")
            logger.info(f"Open Positions: {len(self.positions)}")
            logger.info(f"Win Rate: {win_rate:.1%} ({self.winning_trades}/{self.total_trades})")
            
            # Wait before next iteration
            time.sleep(10)  # 10 seconds between iterations
        
        # Final summary
        self.print_summary()
    
    def print_summary(self):
        """Print trading session summary"""
        logger.info("\n" + "="*50)
        logger.info("TRADING SESSION SUMMARY")
        logger.info("="*50)
        
        # Close any remaining positions
        if self.positions:
            logger.info(f"Closing {len(self.positions)} remaining positions...")
            self.close_all_positions()
        
        final_value = self.capital
        total_return = (final_value - self.initial_capital) / self.initial_capital
        win_rate = self.winning_trades / max(1, self.total_trades)
        
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Final Capital: ${final_value:,.2f}")
        logger.info(f"Total Return: ${final_value - self.initial_capital:,.2f} ({total_return:.1%})")
        logger.info(f"Total Trades: {self.total_trades}")
        logger.info(f"Winning Trades: {self.winning_trades}")
        logger.info(f"Win Rate: {win_rate:.1%}")
        
        if self.trade_history:
            df = pd.DataFrame(self.trade_history)
            logger.info(f"Average P&L: ${df['pnl'].mean():.2f}")
            logger.info(f"Best Trade: ${df['pnl'].max():.2f}")
            logger.info(f"Worst Trade: ${df['pnl'].min():.2f}")
            logger.info(f"Average Hold Time: {df['holding_time'].mean():.1f} hours")
        
        # Save trade history
        if self.trade_history:
            filename = f"paper_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(self.trade_history, f, indent=2, default=str)
            logger.info(f"Trade history saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Paper Trading Bot')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--symbols', nargs='+', default=['SPY', 'QQQ', 'IWM'],
                        help='Symbols to trade')
    parser.add_argument('--capital', type=float, default=100000,
                        help='Initial capital')
    parser.add_argument('--duration', type=int, default=60,
                        help='Duration in minutes')
    
    args = parser.parse_args()
    
    # Create and run bot
    bot = SimplePaperTradingBot(
        model_path=args.model,
        initial_capital=args.capital
    )
    
    bot.run_trading_session(
        symbols=args.symbols,
        duration_minutes=args.duration
    )


if __name__ == "__main__":
    main()