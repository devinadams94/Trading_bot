import asyncio
import argparse
from loguru import logger
import os
import sys
from datetime import datetime
import warnings

from config.config import TradingConfig
from src.options_trading_env import OptionsTradingEnvironment
from src.options_data_collector import AlpacaOptionsDataCollector, OptionsDataSimulator
from src.options_ppo_agent import OptionsPPOAgent
from src.options_executor import AlpacaOptionsExecutor

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logger
logger.add(
    "logs/options_trading_bot_{time}.log",
    rotation="1 day",
    retention="1 week",
    level="INFO"
)


class OptionsTrader:
    def __init__(self, config: TradingConfig, mode: str = "paper"):
        self.config = config
        self.mode = mode
        
        # Initialize components
        self._initialize_components()
        
        # Trading state
        self.is_running = False
        self.current_positions = []
        self.daily_pnl = 0
        
    def _initialize_components(self):
        # Initialize environment
        self.env = OptionsTradingEnvironment(
            initial_capital=self.config.initial_capital,
            max_positions=self.config.max_positions,
            commission=0.65  # Options commission per contract
        )
        
        # Initialize data collector
        if self.mode == "simulation":
            self.data_collector = OptionsDataSimulator()
            logger.info("Using simulated options data")
        else:
            self.data_collector = AlpacaOptionsDataCollector(
                api_key=self.config.alpaca_api_key,
                api_secret=self.config.alpaca_secret_key,
                base_url=self.config.alpaca_base_url
            )
            logger.info(f"Using Alpaca API for {self.mode} trading")
        
        # Initialize PPO agent
        self.agent = OptionsPPOAgent(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space.n
        )
        
        # Load trained model if available
        self._load_model()
        
        # Initialize executor for live/paper trading
        if self.mode in ["live", "paper"]:
            self.executor = AlpacaOptionsExecutor(
                api_key=self.config.alpaca_api_key,
                api_secret=self.config.alpaca_secret_key,
                base_url=self.config.alpaca_base_url,
                max_position_size=self.config.position_size_limit * self.config.initial_capital,
                max_positions=self.config.max_positions,
                risk_check_enabled=True
            )
    
    def _load_model(self):
        checkpoint_path = "checkpoints/options/options_ppo_final.pt"
        if os.path.exists(checkpoint_path):
            self.agent.load(checkpoint_path)
            logger.info(f"Loaded model from {checkpoint_path}")
        else:
            logger.warning("No trained model found. Using untrained agent.")
    
    async def start(self):
        self.is_running = True
        logger.info(f"Starting options trading bot in {self.mode} mode")
        
        if self.mode == "simulation":
            await self._run_simulation()
        else:
            await self._run_trading()
    
    async def _run_simulation(self):
        """Run in simulation mode for testing"""
        logger.info("Running in simulation mode...")
        
        obs = self.env.reset()
        episode_reward = 0
        step = 0
        
        # Simulate for one trading day
        while self.is_running and step < 390:  # 6.5 hours of trading
            # Generate simulated options data
            stock_price = 450 + np.random.normal(0, 5)  # SPY around $450
            options_chain = self.data_collector.simulate_options_chain(
                symbol='SPY',
                stock_price=stock_price,
                num_strikes=20,
                num_expirations=4
            )
            
            # Update observation
            obs = self._update_observation(obs, options_chain, stock_price)
            
            # Get action from agent
            action, _ = self.agent.act(obs, deterministic=True)
            action_name = self.env.action_mapping[action]
            
            logger.info(f"Step {step}: Price=${stock_price:.2f}, Action={action_name}")
            
            # Execute action in environment
            obs, reward, done, info = self.env.step(action)
            episode_reward += reward
            
            # Display current state
            self.env.render()
            
            if done:
                break
            
            step += 1
            await asyncio.sleep(1)  # Simulate real-time
        
        logger.info(f"Simulation complete. Total reward: {episode_reward:.2f}")
    
    async def _run_trading(self):
        """Run live or paper trading"""
        logger.info(f"Starting {self.mode} trading...")
        
        obs = self.env.reset()
        
        while self.is_running:
            try:
                # Check if market is open
                market_hours = self.executor.get_option_market_hours()
                if not market_hours['is_open']:
                    logger.info("Market is closed. Waiting...")
                    await asyncio.sleep(60)
                    continue
                
                # Get current positions
                positions = self.executor.get_positions()
                account_info = self.executor.get_account_info()
                
                logger.info(f"Account: Equity=${account_info['equity']:,.2f}, "
                          f"Positions={len(positions)}, "
                          f"Daily P&L=${account_info['daily_pnl']:,.2f}")
                
                # Fetch options data for each symbol
                for symbol in self.config.symbols:
                    async with self.data_collector as collector:
                        # Get current stock price
                        stock_price = await collector._get_current_stock_price(symbol)
                        
                        # Get options chain
                        options_chain = await collector.get_options_chain(
                            symbol=symbol,
                            strike_price_range=(stock_price * 0.9, stock_price * 1.1)
                        )
                        
                        if not options_chain:
                            logger.warning(f"No options data for {symbol}")
                            continue
                        
                        # Update observation
                        obs = self._update_observation(obs, options_chain, stock_price)
                        
                        # Get action from agent
                        action, _ = self.agent.act(obs, deterministic=True)
                        action_name = self.env.action_mapping[action]
                        
                        logger.info(f"{symbol}: Price=${stock_price:.2f}, "
                                  f"Recommended action={action_name}")
                        
                        # Execute action through Alpaca
                        if action_name != 'hold':
                            result = self.executor.execute_options_strategy(
                                strategy=action_name,
                                symbol=symbol,
                                current_price=stock_price,
                                options_chain=options_chain
                            )
                            
                            if result['success']:
                                logger.info(f"Executed {action_name} for {symbol}: "
                                          f"{result['message']}")
                            else:
                                logger.error(f"Failed to execute {action_name} for {symbol}: "
                                           f"{result['message']}")
                
                # Wait before next iteration
                await asyncio.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                await asyncio.sleep(60)
    
    def _update_observation(self, obs, options_chain, stock_price):
        """Update observation with current market data"""
        import numpy as np
        
        # Update options chain features
        if options_chain:
            # Sort by volume and select top options
            sorted_options = sorted(options_chain, key=lambda x: x.volume, reverse=True)[:20]
            
            options_features = []
            for opt in sorted_options:
                features = [
                    opt.strike,
                    opt.bid,
                    opt.ask,
                    opt.last_price,
                    opt.volume,
                    opt.open_interest,
                    opt.implied_volatility,
                    opt.delta,
                    opt.gamma,
                    opt.theta,
                    opt.vega,
                    opt.rho,
                    1.0 if opt.option_type == 'call' else 0.0,
                    (opt.expiration - datetime.now()).days,
                    (opt.bid + opt.ask) / 2
                ]
                options_features.append(features)
            
            # Pad if necessary
            while len(options_features) < 20:
                options_features.append([0] * 15)
            
            obs['options_chain'] = np.array(options_features, dtype=np.float32)
        
        # Update price history (simplified)
        if 'price_history' not in obs or obs['price_history'] is None:
            obs['price_history'] = np.zeros((20, 5), dtype=np.float32)
        
        # Shift and add new price
        obs['price_history'] = np.roll(obs['price_history'], -1, axis=0)
        obs['price_history'][-1] = [stock_price, stock_price, stock_price, stock_price, 10000]
        
        return obs
    
    async def stop(self):
        self.is_running = False
        
        # Close all positions if in live/paper mode
        if hasattr(self, 'executor'):
            logger.info("Closing all positions...")
            result = self.executor._close_all_positions()
            logger.info(f"Closed positions: {result['message']}")
        
        logger.info("Options trading bot stopped")


def main():
    parser = argparse.ArgumentParser(description="Options Trading Bot")
    parser.add_argument(
        "--mode",
        choices=["live", "paper", "simulation"],
        default="simulation",
        help="Trading mode"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["SPY", "QQQ", "IWM"],
        help="Symbols to trade options on"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Configure logger
    if args.debug:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    # Create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints/options", exist_ok=True)
    
    # Load config
    config = TradingConfig()
    config.symbols = args.symbols
    
    # Check API keys for non-simulation modes
    if args.mode != "simulation":
        if not config.alpaca_api_key or not config.alpaca_secret_key:
            logger.error("Alpaca API keys not found in .env file")
            sys.exit(1)
    
    # Create and run trader
    trader = OptionsTrader(config, mode=args.mode)
    
    try:
        asyncio.run(trader.start())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        asyncio.run(trader.stop())


if __name__ == "__main__":
    # Add numpy import for simulation
    import numpy as np
    main()