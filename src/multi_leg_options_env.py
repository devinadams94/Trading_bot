"""
Multi-Leg Options Trading Environment
Extends WorkingOptionsEnvironment with Phase 2 strategy diversity

Action Space (91 actions):
- 0: Hold
- 1-15: Buy Calls (15 strikes)
- 16-30: Buy Puts (15 strikes)
- 31-45: Sell Calls / Covered Calls (15 strikes)
- 46-60: Sell Puts / Cash-Secured Puts (15 strikes)
- 61-65: Bull Call Spreads (5 variations)
- 66-70: Bear Put Spreads (5 variations)
- 71-75: Long Straddles (5 expirations)
- 76-80: Long Strangles (5 expirations)
- 81-85: Iron Condors (5 variations)
- 86-90: Butterfly Spreads (5 variations)
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from gymnasium import spaces
from datetime import datetime, timedelta

from .working_options_env import WorkingOptionsEnvironment
from .multi_leg_strategies import (
    MultiLegStrategyBuilder,
    StrategyType,
    MultiLegStrategy,
    OptionLeg
)

logger = logging.getLogger(__name__)


class MultiLegOptionsEnvironment(WorkingOptionsEnvironment):
    """Enhanced environment with multi-leg strategy support"""
    
    def __init__(self, *args, enable_multi_leg: bool = True, **kwargs):
        """
        Initialize multi-leg options environment
        
        Args:
            enable_multi_leg: Enable multi-leg strategies (91 actions) or use simple (31 actions)
            All other args passed to WorkingOptionsEnvironment
        """
        # Initialize parent environment
        super().__init__(*args, **kwargs)
        
        self.enable_multi_leg = enable_multi_leg
        self.strategy_builder = MultiLegStrategyBuilder()
        
        # Expand action space if multi-leg enabled
        if enable_multi_leg:
            self.action_space = spaces.Discrete(91)
            logger.info("✅ Multi-leg strategies enabled: 91 actions")
        else:
            logger.info("⚠️ Multi-leg strategies disabled: 31 actions (legacy mode)")
        
        # Track multi-leg positions separately
        self.multi_leg_positions = []
        
        # Statistics
        self.multi_leg_trades = 0
        self.multi_leg_profitable = 0
    
    def _get_action_description(self, action: int) -> str:
        """Get human-readable description of action"""
        if action == 0:
            return "Hold"
        elif 1 <= action <= 15:
            return f"Buy Call (strike offset {(action-8)*0.01:.1%})"
        elif 16 <= action <= 30:
            return f"Buy Put (strike offset {(action-23)*0.01:.1%})"
        elif 31 <= action <= 45:
            return f"Sell Call / Covered Call (strike offset {(action-38)*0.01:.1%})"
        elif 46 <= action <= 60:
            return f"Sell Put / Cash-Secured Put (strike offset {(action-53)*0.01:.1%})"
        elif 61 <= action <= 65:
            return f"Bull Call Spread (variation {action-60})"
        elif 66 <= action <= 70:
            return f"Bear Put Spread (variation {action-65})"
        elif 71 <= action <= 75:
            return f"Long Straddle ({7*(action-70)} days)"
        elif 76 <= action <= 80:
            return f"Long Strangle ({7*(action-75)} days)"
        elif 81 <= action <= 85:
            return f"Iron Condor (variation {action-80})"
        elif 86 <= action <= 90:
            return f"Butterfly Spread (variation {action-85})"
        else:
            return f"Unknown action {action}"
    
    def step(self, action: int):
        """Execute action with multi-leg strategy support"""
        # Handle legacy actions (0-30) with parent class
        if action <= 30:
            return super().step(action)
        
        # Handle new multi-leg actions (31-90)
        if not self.enable_multi_leg:
            logger.warning(f"Multi-leg action {action} attempted but multi-leg disabled")
            return super().step(0)  # Default to hold
        
        # Execute multi-leg strategy
        if 31 <= action <= 60:
            return self._execute_covered_strategy(action)
        elif 61 <= action <= 90:
            return self._execute_multi_leg_strategy(action)
        else:
            logger.warning(f"Invalid action {action}")
            return super().step(0)
    
    def _execute_covered_strategy(self, action: int):
        """Execute covered call or cash-secured put"""
        if len(self.market_data) == 0:
            return self._get_observation(), 0, False, {}

        # Get current market data using parent class method
        current_data = self._get_current_market_data()
        if current_data is None:
            return self._get_observation(), 0, False, {}

        current_price = current_data.get('close', 100.0)
        
        # Covered calls (31-45)
        if 31 <= action <= 45:
            return self._execute_covered_call(action, current_price, current_data)
        
        # Cash-secured puts (46-60)
        elif 46 <= action <= 60:
            return self._execute_cash_secured_put(action, current_price, current_data)
        
        return self._get_observation(), 0, False, {}
    
    def _execute_covered_call(self, action: int, current_price: float, current_data: Dict):
        """Execute covered call strategy"""
        # Check if we have enough capital or stock
        if self.capital < current_price * 100:
            return self._get_observation(), -0.01, False, {'error': 'insufficient_capital'}
        
        # Calculate strike (similar to buy call logic)
        strike_offset = (action - 38) * 0.01  # -7% to +7%
        strike_price = current_price * (1 + strike_offset)
        
        # Estimate option premium (simplified)
        option_premium = max(0.5, current_price * 0.03 * (1 - abs(strike_offset)))
        
        # Calculate transaction cost
        moneyness = strike_price / current_price
        spread_pct = 0.02 + 0.03 * abs(1 - moneyness)
        option_data = {
            'bid': option_premium * (1 - spread_pct / 2),
            'ask': option_premium * (1 + spread_pct / 2),
            'last': option_premium,
            'volume': current_data.get('volume', 1000) / 100,
            'open_interest': 1000,
            'moneyness': moneyness,
            'implied_volatility': current_data.get('volatility', 0.3)
        }
        
        transaction_cost, cost_breakdown = self._calculate_transaction_cost(
            option_data, quantity=1, side='sell'
        )
        
        # Sell call (receive premium)
        if self.use_realistic_costs:
            execution_price = option_data['bid']  # Sell at bid
        else:
            execution_price = option_premium
        
        premium_received = execution_price * 100 - transaction_cost
        self.capital += premium_received
        
        # Record position
        self.positions.append({
            'type': 'covered_call',
            'strike': strike_price,
            'entry_price': execution_price,
            'entry_step': self.current_step,
            'quantity': 1,
            'premium_received': premium_received,
            'transaction_cost': transaction_cost
        })
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.market_data) - 1
        
        # Calculate reward (premium received is profit)
        reward = premium_received * self.reward_scaling
        
        return self._get_observation(), reward, done, {
            'action': 'covered_call',
            'strike': strike_price,
            'premium_received': premium_received,
            'transaction_cost': transaction_cost
        }
    
    def _execute_cash_secured_put(self, action: int, current_price: float, current_data: Dict):
        """Execute cash-secured put strategy"""
        # Calculate strike
        strike_offset = (action - 53) * 0.01  # -7% to +7%
        strike_price = current_price * (1 + strike_offset)
        
        # Check if we have enough capital to secure the put
        required_capital = strike_price * 100
        if self.capital < required_capital:
            return self._get_observation(), -0.01, False, {'error': 'insufficient_capital'}
        
        # Estimate option premium
        option_premium = max(0.5, current_price * 0.03 * (1 + abs(strike_offset)))
        
        # Calculate transaction cost
        moneyness = strike_price / current_price
        spread_pct = 0.02 + 0.03 * abs(1 - moneyness)
        option_data = {
            'bid': option_premium * (1 - spread_pct / 2),
            'ask': option_premium * (1 + spread_pct / 2),
            'last': option_premium,
            'volume': current_data.get('volume', 1000) / 100,
            'open_interest': 1000,
            'moneyness': moneyness,
            'implied_volatility': current_data.get('volatility', 0.3)
        }
        
        transaction_cost, cost_breakdown = self._calculate_transaction_cost(
            option_data, quantity=1, side='sell'
        )
        
        # Sell put (receive premium, reserve capital)
        if self.use_realistic_costs:
            execution_price = option_data['bid']
        else:
            execution_price = option_premium
        
        premium_received = execution_price * 100 - transaction_cost
        self.capital += premium_received
        self.capital -= required_capital  # Reserve capital
        
        # Record position
        self.positions.append({
            'type': 'cash_secured_put',
            'strike': strike_price,
            'entry_price': execution_price,
            'entry_step': self.current_step,
            'quantity': 1,
            'premium_received': premium_received,
            'capital_reserved': required_capital,
            'transaction_cost': transaction_cost
        })
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.market_data) - 1
        
        # Calculate reward
        reward = premium_received * self.reward_scaling
        
        return self._get_observation(), reward, done, {
            'action': 'cash_secured_put',
            'strike': strike_price,
            'premium_received': premium_received,
            'capital_reserved': required_capital,
            'transaction_cost': transaction_cost
        }
    
    def _execute_multi_leg_strategy(self, action: int):
        """Execute multi-leg strategy (spreads, straddles, etc.)"""
        if len(self.market_data) == 0:
            return self._get_observation(), 0, False, {}

        # Get current market data using parent class method
        current_data = self._get_current_market_data()
        if current_data is None:
            return self._get_observation(), 0, False, {}

        current_price = current_data.get('close', 100.0)
        
        # Determine strategy type and build it
        strategy = None
        expiration_days = 30  # Default
        
        if 61 <= action <= 65:
            # Bull call spread
            strategy = self.strategy_builder.build_bull_call_spread(current_price, 1, expiration_days)
        elif 66 <= action <= 70:
            # Bear put spread
            strategy = self.strategy_builder.build_bear_put_spread(current_price, 1, expiration_days)
        elif 71 <= action <= 75:
            # Long straddle (vary expiration)
            expiration_days = 7 * (action - 70)
            strategy = self.strategy_builder.build_long_straddle(current_price, 1, expiration_days)
        elif 76 <= action <= 80:
            # Long strangle (vary expiration)
            expiration_days = 7 * (action - 75)
            strategy = self.strategy_builder.build_long_strangle(current_price, 1, expiration_days)
        elif 81 <= action <= 85:
            # Iron condor
            strategy = self.strategy_builder.build_iron_condor(current_price, 1, expiration_days)
        elif 86 <= action <= 90:
            # Butterfly spread
            strategy = self.strategy_builder.build_butterfly_spread(current_price, 1, expiration_days)
        
        if strategy is None:
            return self._get_observation(), 0, False, {'error': 'invalid_strategy'}
        
        # Check if we have enough capital
        if self.capital < strategy.capital_required:
            return self._get_observation(), -0.01, False, {'error': 'insufficient_capital'}
        
        # Execute strategy (deduct capital)
        self.capital -= strategy.capital_required
        
        # Record multi-leg position
        self.multi_leg_positions.append({
            'strategy_type': strategy.strategy_type.value,
            'entry_step': self.current_step,
            'entry_price': current_price,
            'capital_required': strategy.capital_required,
            'max_profit': strategy.max_profit,
            'max_loss': strategy.max_loss,
            'legs': strategy.legs
        })
        
        self.multi_leg_trades += 1
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.market_data) - 1
        
        # Calculate reward (negative for capital deployed, will be positive if profitable)
        reward = -strategy.capital_required * self.reward_scaling * 0.1  # Small penalty for capital usage
        
        return self._get_observation(), reward, done, {
            'action': 'multi_leg_strategy',
            'strategy_type': strategy.strategy_type.value,
            'capital_required': strategy.capital_required,
            'max_profit': strategy.max_profit,
            'max_loss': strategy.max_loss
        }
    
    def reset(self):
        """Reset environment including multi-leg positions"""
        self.multi_leg_positions = []
        self.multi_leg_trades = 0
        self.multi_leg_profitable = 0
        return super().reset()

