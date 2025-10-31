#!/usr/bin/env python3
"""
Fix for models that accumulate positions without closing them
Adds incentives for realizing profits and managing positions
"""

import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class PositionClosingIncentiveMixin:
    """
    Mixin to add position closing incentives to any trading environment
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Position closing incentives
        self.holding_time_penalty_rate = 0.001  # Penalty per step for holding
        self.realized_profit_bonus = 0.5  # 50% bonus for realizing profits
        self.position_limit_penalty = 0.1  # Penalty for approaching max positions
        self.profit_taking_threshold = 0.05  # 5% profit triggers bonus
        self.max_holding_steps = 100  # Maximum steps before forced closing
        
        # Track position holding times
        self.position_holding_times = {}
        self.realized_profits_episode = 0
        self.unrealized_profits_episode = 0
        
    def _calculate_reward_with_closing_incentives(self, base_reward: float, action: int, info: Dict[str, Any]) -> float:
        """
        Enhance reward to encourage position closing
        """
        reward = base_reward
        
        # 1. Holding Time Penalty - Increases with time
        holding_penalty = 0
        for pos_id, steps in self.position_holding_times.items():
            # Exponential penalty for long holding
            penalty = self.holding_time_penalty_rate * (steps ** 1.5) / 100
            holding_penalty += penalty
        
        reward -= holding_penalty
        
        # 2. Realized Profit Bonus - Reward closing profitable positions
        if 'trade_result' in info and info['trade_result'].get('success'):
            message = info['trade_result'].get('message', '')
            
            # Check if position was closed
            if 'closed' in message.lower() or 'close' in message.lower():
                if 'pnl' in info['trade_result']:
                    pnl = info['trade_result']['pnl']
                    if pnl > 0:
                        # Big bonus for realizing profits
                        profit_bonus = pnl * self.realized_profit_bonus * 1e-4
                        reward += profit_bonus
                        self.realized_profits_episode += pnl
                        
                        logger.debug(f"Realized profit bonus: {profit_bonus:.4f} for P&L: ${pnl:.2f}")
        
        # 3. Position Limit Penalty - Discourage maxing out positions
        if hasattr(self, 'positions'):
            position_ratio = len(self.positions) / self.max_positions
            if position_ratio > 0.7:  # Start penalizing at 70% capacity
                position_penalty = self.position_limit_penalty * (position_ratio - 0.7)
                reward -= position_penalty
        
        # 4. Action-specific bonuses
        action_name = self.action_mapping.get(action, 'hold')
        
        # Bonus for close_all_positions when profitable
        if action_name == 'close_all_positions' and hasattr(self, 'positions'):
            total_pnl = sum(pos.pnl for pos in self.positions if hasattr(pos, 'pnl'))
            if total_pnl > 0:
                reward += total_pnl * 0.2 * 1e-4  # 20% bonus for closing all profitable
        
        # 5. Unrealized profits tracking (for logging)
        if hasattr(self, 'positions'):
            self.unrealized_profits_episode = sum(
                pos.pnl for pos in self.positions 
                if hasattr(pos, 'pnl')
            )
        
        return reward
    
    def _update_position_holding_times(self):
        """Update holding time for each position"""
        if hasattr(self, 'positions'):
            # Update existing positions
            current_position_ids = set()
            for pos in self.positions:
                pos_id = id(pos)  # Use object id as unique identifier
                current_position_ids.add(pos_id)
                
                if pos_id not in self.position_holding_times:
                    self.position_holding_times[pos_id] = 0
                self.position_holding_times[pos_id] += 1
                
                # Force close very old positions
                if self.position_holding_times[pos_id] > self.max_holding_steps:
                    logger.info(f"Force closing position held for {self.max_holding_steps} steps")
                    self._force_close_position(pos)
            
            # Remove closed positions from tracking
            closed_positions = set(self.position_holding_times.keys()) - current_position_ids
            for pos_id in closed_positions:
                del self.position_holding_times[pos_id]
    
    def _force_close_position(self, position):
        """Force close a position that's been held too long"""
        if hasattr(position, 'pnl'):
            pnl = position.pnl
            self._close_position(position)
            logger.info(f"Force closed position with P&L: ${pnl:.2f}")
    
    def step(self, action: int):
        """Override step to add closing incentives"""
        # Get base step results
        if hasattr(super(), 'step'):
            result = super().step(action)
            if len(result) == 4:
                obs, reward, done, info = result
                truncated = False
            else:
                obs, reward, done, truncated, info = result
        else:
            raise NotImplementedError("Parent class must implement step()")
        
        # Update position holding times
        self._update_position_holding_times()
        
        # Apply closing incentives to reward
        enhanced_reward = self._calculate_reward_with_closing_incentives(reward, action, info)
        
        # Add episode stats to info
        info['realized_profits'] = self.realized_profits_episode
        info['unrealized_profits'] = self.unrealized_profits_episode
        info['position_holding_times'] = list(self.position_holding_times.values())
        
        if len(result) == 4:
            return obs, enhanced_reward, done, info
        else:
            return obs, enhanced_reward, done, truncated, info
    
    def reset(self, **kwargs):
        """Reset tracking variables"""
        if hasattr(super(), 'reset'):
            obs = super().reset(**kwargs)
        else:
            raise NotImplementedError("Parent class must implement reset()")
            
        self.position_holding_times = {}
        self.realized_profits_episode = 0
        self.unrealized_profits_episode = 0
        
        return obs


class ProfitRealizationEnvironment:
    """
    Enhanced environment that strongly encourages profit realization
    """
    
    def __init__(self, base_env, config: Optional[Dict[str, Any]] = None):
        self.base_env = base_env
        self.config = config or {}
        
        # Profit realization parameters
        self.min_profit_threshold = self.config.get('min_profit_threshold', 0.02)  # 2%
        self.profit_decay_rate = self.config.get('profit_decay_rate', 0.001)  # Decay per step
        self.close_bonus_multiplier = self.config.get('close_bonus_multiplier', 2.0)
        
        # Track profits
        self.step_count = 0
        self.positions_closed_this_episode = 0
        self.profitable_closes = 0
        
    def __getattr__(self, name):
        """Pass through to base environment"""
        return getattr(self.base_env, name)
    
    def step(self, action):
        """Enhanced step with profit realization incentives"""
        # Get base results
        result = self.base_env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            truncated = False
        else:
            obs, reward, done, truncated, info = result
        
        self.step_count += 1
        
        # Track position closes
        if 'trade_result' in info and info['trade_result'].get('success'):
            message = info['trade_result'].get('message', '')
            if 'closed' in message.lower():
                self.positions_closed_this_episode += 1
                if info['trade_result'].get('pnl', 0) > 0:
                    self.profitable_closes += 1
        
        # Apply profit realization incentives
        enhanced_reward = self._apply_profit_incentives(reward, action, info)
        
        # Add stats
        info['positions_closed'] = self.positions_closed_this_episode
        info['profitable_closes'] = self.profitable_closes
        info['close_ratio'] = self.profitable_closes / max(1, self.positions_closed_this_episode)
        
        if len(result) == 4:
            return obs, enhanced_reward, done, info
        else:
            return obs, enhanced_reward, done, truncated, info
    
    def _apply_profit_incentives(self, base_reward: float, action: int, info: Dict) -> float:
        """Apply strong incentives for profit realization"""
        reward = base_reward
        
        # Decay unrealized profits over time
        if hasattr(self.base_env, 'positions'):
            for pos in self.base_env.positions:
                if hasattr(pos, 'pnl') and hasattr(pos, 'entry_time'):
                    # Calculate holding time (simplified)
                    holding_steps = self.step_count
                    
                    # Decay the value of unrealized profits
                    if pos.pnl > 0:
                        decay_penalty = pos.pnl * self.profit_decay_rate * holding_steps * 1e-4
                        reward -= decay_penalty
        
        # Bonus for closing profitable positions
        if 'trade_result' in info and 'closed' in str(info['trade_result'].get('message', '')):
            pnl = info['trade_result'].get('pnl', 0)
            if pnl > 0:
                # Bigger bonus for larger profits
                close_bonus = pnl * self.close_bonus_multiplier * 1e-4
                reward += close_bonus
                
                # Extra bonus for quick profits
                if self.step_count < 50:  # Closed within 50 steps
                    reward += pnl * 0.5 * 1e-4
        
        return reward
    
    def reset(self, **kwargs):
        """Reset tracking"""
        obs = self.base_env.reset(**kwargs)
        self.step_count = 0
        self.positions_closed_this_episode = 0
        self.profitable_closes = 0
        return obs


def create_position_closing_environment(base_env_class, **env_kwargs):
    """
    Create an environment with position closing incentives
    """
    
    # Create enhanced environment class
    class EnhancedTradingEnvironment(PositionClosingIncentiveMixin, base_env_class):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            
            # Override position management
            self.original_max_holding_period = getattr(self, 'max_holding_period', 5)
            self.max_holding_period = 3  # Reduce max holding period
            
        def _calculate_reward(self, trade_result: Dict[str, Any]) -> float:
            """Override to use enhanced reward calculation"""
            # Get base reward from parent
            base_reward = super()._calculate_reward(trade_result)
            
            # Apply closing incentives
            action = getattr(self, 'last_action', 0)
            info = {'trade_result': trade_result}
            
            return self._calculate_reward_with_closing_incentives(base_reward, action, info)
        
        def step(self, action: int):
            """Store action for reward calculation"""
            self.last_action = action
            return super().step(action)
    
    # Create instance
    return EnhancedTradingEnvironment(**env_kwargs)


# Reward shaping specifically for closing positions
class ClosingRewardShaper:
    """
    Shapes rewards to encourage closing positions at the right time
    """
    
    @staticmethod
    def calculate_closing_reward(
        pnl: float,
        holding_time: int,
        position_size: int,
        total_positions: int
    ) -> float:
        """
        Calculate reward for closing a position
        
        Returns higher rewards for:
        - Higher P&L
        - Quicker closes (lower holding time)
        - Not maxing out positions
        """
        
        # Base reward from P&L
        base_reward = pnl * 1e-4
        
        # Time efficiency bonus (decay with holding time)
        time_bonus = max(0, (100 - holding_time) / 100) * abs(pnl) * 0.5 * 1e-4
        
        # Position management bonus (reward keeping slots open)
        position_bonus = (1 - total_positions / 10) * abs(pnl) * 0.2 * 1e-4
        
        # Size bonus (larger positions = more risk taken)
        size_bonus = np.log1p(position_size) * 0.1 * pnl * 1e-4
        
        total_reward = base_reward + time_bonus + position_bonus + size_bonus
        
        return total_reward
    
    @staticmethod
    def calculate_holding_penalty(
        unrealized_pnl: float,
        holding_time: int,
        volatility: float = 0.02
    ) -> float:
        """
        Calculate penalty for holding positions too long
        
        Increases with:
        - Holding time
        - Unrealized P&L (opportunity cost)
        - Market volatility (risk)
        """
        
        # Time-based penalty (exponential growth)
        time_penalty = -0.001 * (1.02 ** holding_time)
        
        # Opportunity cost of unrealized gains
        if unrealized_pnl > 0:
            opportunity_penalty = -unrealized_pnl * 0.01 * holding_time * 1e-4
        else:
            opportunity_penalty = 0
        
        # Volatility risk penalty
        risk_penalty = -volatility * holding_time * 0.1
        
        return time_penalty + opportunity_penalty + risk_penalty


# Configuration for different closing strategies
CLOSING_STRATEGIES = {
    'aggressive': {
        'holding_time_penalty_rate': 0.002,
        'realized_profit_bonus': 1.0,
        'profit_taking_threshold': 0.03,  # 3%
        'max_holding_steps': 50
    },
    'moderate': {
        'holding_time_penalty_rate': 0.001,
        'realized_profit_bonus': 0.5,
        'profit_taking_threshold': 0.05,  # 5%
        'max_holding_steps': 100
    },
    'patient': {
        'holding_time_penalty_rate': 0.0005,
        'realized_profit_bonus': 0.3,
        'profit_taking_threshold': 0.10,  # 10%
        'max_holding_steps': 200
    }
}


def apply_closing_strategy(env, strategy_name: str = 'moderate'):
    """
    Apply a predefined closing strategy to an environment
    """
    if strategy_name not in CLOSING_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    strategy = CLOSING_STRATEGIES[strategy_name]
    
    # Apply strategy parameters
    for key, value in strategy.items():
        setattr(env, key, value)
    
    logger.info(f"Applied {strategy_name} closing strategy")
    return env