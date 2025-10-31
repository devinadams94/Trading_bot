#!/usr/bin/env python3
"""
Script to add live trading capability to train.py
"""

import re
from pathlib import Path


def add_live_trading_imports():
    """Add necessary imports for live trading"""
    return """
# Live trading imports
import alpaca_trade_api as tradeapi
from src.options_data_collector import AlpacaOptionsDataCollector
from typing import Union
"""


def create_live_trading_environment():
    """Create a live trading environment class"""
    return '''
class LiveTradingEnvironment(HistoricalOptionsEnvironment):
    """Environment for live trading with real-time data from Alpaca"""
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = 'https://paper-api.alpaca.markets',
        symbols: List[str] = None,
        initial_capital: float = 100000,
        max_positions: int = 5,
        commission: float = 0.65,
        episode_length: int = 390,  # Trading minutes in a day
        position_size_pct: float = 0.05,  # 5% of capital per trade
        stop_loss_pct: float = 0.10,  # 10% stop loss
        take_profit_pct: float = 0.20,  # 20% take profit
        max_daily_loss_pct: float = 0.02,  # 2% daily loss limit
        live_mode: bool = True
    ):
        # Initialize parent class with empty data (we'll fetch live)
        super().__init__(
            historical_data={},
            symbols=symbols or ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA'],
            initial_capital=initial_capital,
            commission=commission,
            max_positions=max_positions,
            episode_length=episode_length
        )
        
        # Live trading specific attributes
        self.live_mode = live_mode
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.daily_starting_capital = initial_capital
        
        # Initialize Alpaca API
        self.api = tradeapi.REST(api_key, api_secret, base_url)
        self.data_collector = AlpacaOptionsDataCollector(
            api_key=api_key,
            api_secret=api_secret,
            base_url=base_url
        )
        
        # Track real positions
        self.real_positions = []
        self.daily_pnl = 0
        self.trade_log = []
        
        # Real-time data cache
        self.realtime_cache = {}
        self.last_update_time = {}
        
        logger.info(f"🔴 LIVE TRADING MODE {'ACTIVE' if live_mode else 'SIMULATED'}")
        logger.info(f"   Capital: ${initial_capital:,.2f}")
        logger.info(f"   Max positions: {max_positions}")
        logger.info(f"   Position size: {position_size_pct*100:.1f}% of capital")
        logger.info(f"   Stop loss: {stop_loss_pct*100:.1f}%")
        logger.info(f"   Daily loss limit: {max_daily_loss_pct*100:.1f}%")
    
    def reset(self):
        """Reset for a new trading day"""
        obs = super().reset()
        
        # Update account info
        if self.live_mode:
            try:
                account = self.api.get_account()
                self.capital = float(account.buying_power)
                self.daily_starting_capital = self.capital
                self.daily_pnl = 0
                logger.info(f"Account updated - Buying power: ${self.capital:,.2f}")
            except Exception as e:
                logger.error(f"Failed to get account info: {e}")
        
        return obs
    
    def _get_live_market_data(self, symbol: str):
        """Fetch real-time market data"""
        try:
            # Get latest bar data
            bars = self.api.get_latest_bar(symbol)
            
            # Get options chain
            current_price = bars.c  # Close price
            options = self.data_collector.get_options_chain_sync(
                symbol=symbol,
                min_strike=current_price * 0.95,
                max_strike=current_price * 1.05,
                min_expiry_days=7,
                max_expiry_days=45
            )
            
            return {
                'price': current_price,
                'volume': bars.v,
                'options': options,
                'timestamp': bars.t
            }
        except Exception as e:
            logger.error(f"Error fetching live data for {symbol}: {e}")
            return None
    
    def _execute_real_trade(self, action_name: str, symbol: str, quantity: int, contract_symbol: str):
        """Execute a real trade through Alpaca"""
        if not self.live_mode:
            logger.info(f"SIMULATED: Would execute {action_name} for {quantity} {contract_symbol}")
            return True
        
        try:
            # Check daily loss limit
            if self.daily_pnl < -self.daily_starting_capital * self.max_daily_loss_pct:
                logger.warning("Daily loss limit reached - no new trades allowed")
                return False
            
            # Submit order
            if action_name in ['buy_call', 'buy_put']:
                order = self.api.submit_order(
                    symbol=contract_symbol,
                    qty=quantity,
                    side='buy',
                    type='limit',
                    time_in_force='day',
                    limit_price=None  # Use market order for now
                )
                
                logger.info(f"✅ Order submitted: {action_name} {quantity} {contract_symbol}")
                logger.info(f"   Order ID: {order.id}")
                
                # Track position
                self.real_positions.append({
                    'order_id': order.id,
                    'symbol': symbol,
                    'contract_symbol': contract_symbol,
                    'quantity': quantity,
                    'entry_price': float(order.limit_price or order.filled_avg_price or 0),
                    'entry_time': datetime.now(),
                    'action': action_name
                })
                
                return True
            
            elif action_name == 'close_position':
                # Find matching position
                position = next((p for p in self.real_positions if p['contract_symbol'] == contract_symbol), None)
                if position:
                    order = self.api.submit_order(
                        symbol=contract_symbol,
                        qty=position['quantity'],
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
                    
                    logger.info(f"✅ Close order submitted for {contract_symbol}")
                    self.real_positions.remove(position)
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to execute trade: {e}")
            return False
        
        return False
    
    def step(self, action: int):
        """Execute action with live trading"""
        # Get base observation and reward
        obs, reward, done, info = super().step(action)
        
        # If we're in live mode and action involves trading
        actions = ['hold', 'buy_call', 'buy_put', 'sell_call', 'sell_put', 
                  'bull_call_spread', 'bear_put_spread', 'iron_condor', 
                  'straddle', 'strangle', 'close_all_positions']
        action_name = actions[action] if action < len(actions) else 'hold'
        
        # Handle live trades
        if self.live_mode and action_name in ['buy_call', 'buy_put', 'close_all_positions']:
            symbol = self.current_symbol
            
            # Get live data
            live_data = self._get_live_market_data(symbol)
            if live_data:
                if action_name == 'close_all_positions':
                    # Close all real positions
                    for position in self.real_positions[:]:
                        self._execute_real_trade('close_position', position['symbol'], 
                                               position['quantity'], position['contract_symbol'])
                else:
                    # Calculate position size
                    position_value = self.capital * self.position_size_pct
                    # Find suitable option contract
                    if live_data['options']:
                        # Pick ATM option
                        option = self._select_best_option(live_data['options'], action_name)
                        if option:
                            quantity = int(position_value / (option['ask'] * 100))
                            if quantity > 0:
                                self._execute_real_trade(action_name, symbol, quantity, option['symbol'])
        
        # Update P&L tracking
        if self.live_mode:
            self._update_real_pnl()
        
        return obs, reward, done, info
    
    def _select_best_option(self, options: List[Dict], action_name: str):
        """Select the best option contract based on criteria"""
        option_type = 'call' if 'call' in action_name else 'put'
        
        # Filter by type and liquidity
        filtered = [
            opt for opt in options 
            if opt['type'] == option_type and opt['volume'] > 100
        ]
        
        if not filtered:
            return None
        
        # Sort by distance to ATM
        filtered.sort(key=lambda x: abs(x['strike'] - x['underlying_price']))
        
        # Return the most liquid near-ATM option
        return filtered[0]
    
    def _update_real_pnl(self):
        """Update real P&L from actual positions"""
        if not self.real_positions:
            return
        
        try:
            # Get current positions from Alpaca
            positions = self.api.list_positions()
            
            total_pnl = 0
            for pos in positions:
                # Match with our tracked positions
                tracked = next((p for p in self.real_positions 
                              if p['contract_symbol'] == pos.symbol), None)
                if tracked:
                    current_value = float(pos.market_value)
                    entry_value = tracked['entry_price'] * tracked['quantity'] * 100
                    pnl = current_value - entry_value
                    total_pnl += pnl
                    
                    # Check stop loss / take profit
                    pnl_pct = pnl / entry_value
                    if pnl_pct <= -self.stop_loss_pct or pnl_pct >= self.take_profit_pct:
                        self._execute_real_trade('close_position', tracked['symbol'],
                                               tracked['quantity'], tracked['contract_symbol'])
            
            self.daily_pnl = total_pnl
            
        except Exception as e:
            logger.error(f"Error updating P&L: {e}")
'''


def add_argument_parser_modifications():
    """Add live mode argument to parser"""
    return """    parser.add_argument('--live-mode', action='store_true', help='Enable live trading through Alpaca')
    parser.add_argument('--paper-trading', action='store_true', help='Use paper trading account (default)')
    parser.add_argument('--live-capital', type=float, default=10000, help='Capital for live trading')
    parser.add_argument('--live-symbols', nargs='+', default=['SPY', 'QQQ'], help='Symbols for live trading')
    parser.add_argument('--position-size', type=float, default=0.05, help='Position size as fraction of capital')
    parser.add_argument('--daily-loss-limit', type=float, default=0.02, help='Daily loss limit as fraction of capital')
"""


def modify_train_function():
    """Modify train function to support live mode"""
    return '''def train(num_episodes=10000, save_interval=100, use_real_data=True, resume=False, checkpoint_path=None, no_distributed=False, live_mode=False, live_config=None):
    """Main training function that spawns distributed processes"""
    
    # Check if we have multiple GPUs
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        
        # Live mode only supports single GPU
        if live_mode and world_size > 1:
            logger.warning("Live trading mode only supports single GPU - forcing single GPU mode")
            world_size = 1
            no_distributed = True
        
        if world_size > 1 and not no_distributed and not args.force_single_gpu:
            logger.info(f"Starting distributed training on {world_size} GPUs")
            
            # Find a free port to avoid conflicts
            free_port = find_free_port()
            os.environ['MASTER_PORT'] = str(free_port)
            logger.info(f"Using port {free_port} for distributed training")
            
            # Spawn processes for distributed training
            mp.spawn(
                train_distributed,
                args=(world_size, num_episodes, save_interval, use_real_data, resume, checkpoint_path, live_mode, live_config),
                nprocs=world_size,
                join=True
            )
        else:
            if no_distributed or args.force_single_gpu:
                logger.info("Single GPU mode (distributed training disabled)")
            else:
                logger.info("Single GPU detected, running standard training")
            # Run single GPU training
            train_distributed(0, 1, num_episodes, save_interval, use_real_data, resume, checkpoint_path, live_mode, live_config)
    else:
        logger.error("No GPU available! This training script requires GPU.")
        sys.exit(1)'''


def modify_train_distributed_signature():
    """Modify train_distributed function signature"""
    return 'def train_distributed(rank, world_size, num_episodes=10000, save_interval=100, use_real_data=True, resume=False, checkpoint_path=None, live_mode=False, live_config=None):'


def add_live_environment_creation():
    """Add code to create live environment when in live mode"""
    return '''
    # Create environment based on mode
    if live_mode:
        # Live trading environment
        logger.info("🔴 LIVE TRADING MODE ENABLED")
        
        # Get API credentials
        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_SECRET_KEY')
        
        if not api_key or not api_secret:
            logger.error("Alpaca API credentials not found! Set ALPACA_API_KEY and ALPACA_SECRET_KEY")
            return
        
        # Determine base URL
        base_url = 'https://paper-api.alpaca.markets' if live_config.get('paper_trading', True) else 'https://api.alpaca.markets'
        
        # Create single live environment (no vectorization for live trading)
        env = LiveTradingEnvironment(
            api_key=api_key,
            api_secret=api_secret,
            base_url=base_url,
            symbols=live_config.get('symbols', ['SPY', 'QQQ']),
            initial_capital=live_config.get('capital', 10000),
            max_positions=3,  # Conservative for live trading
            position_size_pct=live_config.get('position_size', 0.05),
            stop_loss_pct=0.10,
            take_profit_pct=0.20,
            max_daily_loss_pct=live_config.get('daily_loss_limit', 0.02),
            live_mode=True  # Actually execute trades
        )
        
        # Wrap in a simple vectorized wrapper for compatibility
        class SingleEnvWrapper:
            def __init__(self, env):
                self.envs = [env]
                self.num_envs = 1
                self.observation_space = env.observation_space
                self.action_space = env.action_space
            
            def reset(self):
                return [self.envs[0].reset()]
            
            def step(self, actions):
                obs, reward, done, info = self.envs[0].step(actions[0])
                return [obs], [reward], [done], [info]
            
            def close(self):
                pass
        
        env = SingleEnvWrapper(env)
        n_envs = 1
        
        logger.info("✅ Live trading environment created")
        logger.info(f"   Symbols: {live_config.get('symbols')}")
        logger.info(f"   Capital: ${live_config.get('capital'):,.2f}")
        logger.info(f"   {'PAPER' if live_config.get('paper_trading') else 'REAL'} trading account")
    else:'''


def main():
    """Main function to modify train.py"""
    
    print("📝 Adding live trading capability to train.py...")
    
    # Read the current train.py
    train_path = Path('train.py')
    if not train_path.exists():
        print("❌ train.py not found!")
        return
    
    with open(train_path, 'r') as f:
        content = f.read()
    
    # Create backup
    backup_path = Path('train_backup_before_live.py')
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"✅ Created backup at {backup_path}")
    
    # 1. Add imports at the top
    import_match = re.search(r'(import argparse.*?\n)', content)
    if import_match:
        insert_pos = import_match.end()
        content = content[:insert_pos] + add_live_trading_imports() + content[insert_pos:]
        print("✅ Added live trading imports")
    
    # 2. Add LiveTradingEnvironment class after BalancedEnvironment
    balanced_env_match = re.search(r'(class BalancedEnvironment.*?(?=\n\nclass|\n\n\w|\Z))', content, re.DOTALL)
    if balanced_env_match:
        insert_pos = balanced_env_match.end()
        content = content[:insert_pos] + "\n\n" + create_live_trading_environment() + content[insert_pos:]
        print("✅ Added LiveTradingEnvironment class")
    
    # 3. Add arguments to parser
    parser_match = re.search(r'(parser\.add_argument.*?args = parser\.parse_args\(\))', content, re.DOTALL)
    if parser_match:
        # Insert before args = parser.parse_args()
        insert_pos = content.find('args = parser.parse_args()', parser_match.start())
        content = content[:insert_pos] + add_argument_parser_modifications() + "\n    " + content[insert_pos:]
        print("✅ Added live mode arguments")
    
    # 4. Modify train function signature
    content = re.sub(
        r'def train\(num_episodes=10000, save_interval=100, use_real_data=True, resume=False, checkpoint_path=None, no_distributed=False\):',
        modify_train_function().split('\n')[0],
        content
    )
    print("✅ Modified train function signature")
    
    # 5. Modify train_distributed signature
    content = re.sub(
        r'def train_distributed\(rank, world_size, num_episodes=10000, save_interval=100, use_real_data=True, resume=False, checkpoint_path=None\):',
        modify_train_distributed_signature(),
        content
    )
    print("✅ Modified train_distributed signature")
    
    # 6. Add live environment creation in train_distributed
    # Find where environments are created
    env_creation_match = re.search(r'# Create environment functions\s*\n\s*def make_env', content)
    if env_creation_match:
        insert_pos = env_creation_match.start()
        content = content[:insert_pos] + add_live_environment_creation() + "\n        " + content[insert_pos:]
        print("✅ Added live environment creation")
    
    # 7. Update the train function call at the bottom
    content = re.sub(
        r'train\(\s*num_episodes=args\.episodes,\s*save_interval=args\.save_interval,',
        '''train(
        num_episodes=args.episodes,
        save_interval=args.save_interval,''',
        content
    )
    
    # Add live mode parameters to train call
    train_call_match = re.search(r'(train\(.*?no_distributed=args\.no_distributed)', content, re.DOTALL)
    if train_call_match:
        content = content[:train_call_match.end()] + ''',
        live_mode=args.live_mode,
        live_config={
            'paper_trading': args.paper_trading,
            'capital': args.live_capital,
            'symbols': args.live_symbols,
            'position_size': args.position_size,
            'daily_loss_limit': args.daily_loss_limit
        } if args.live_mode else None''' + content[train_call_match.end():]
        print("✅ Updated train function call")
    
    # Write the modified content
    with open(train_path, 'w') as f:
        f.write(content)
    
    print("\n✅ Successfully added live trading capability to train.py!")
    print("\n📋 Usage:")
    print("1. Paper trading (recommended for testing):")
    print("   python train.py --live-mode --paper-trading --episodes 100")
    print("\n2. Real trading (use with caution):")
    print("   python train.py --live-mode --live-capital 5000 --symbols SPY QQQ")
    print("\n⚠️  WARNINGS:")
    print("- Always test with paper trading first!")
    print("- Start with small capital when going live")
    print("- Monitor the bot closely during live trading")
    print("- Set appropriate stop losses and position sizes")


if __name__ == "__main__":
    main()