#!/usr/bin/env python3
"""
Reward structure implementation based on the research paper:
"A Novel Deep Reinforcement Learning Based Automated Stock Trading System Using Cascaded LSTM Networks"
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ResearchPaperRewardMixin:
    """
    Implements the exact reward structure from the research paper
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Paper parameters
        self.transaction_cost_rate = 0.001  # 0.1% per transaction as in paper
        self.reward_scaling_factor = 1e-4   # As specified in section 3.1.5
        self.turbulence_threshold = None    # Will be set based on historical data
        
        # Track portfolio value for reward calculation
        self.last_portfolio_value = getattr(self, 'initial_capital', 100000)
        
    def _calculate_reward(self, trade_result: Dict[str, Any]) -> float:
        """
        Calculate reward as per Equation (1) in the paper:
        
        Return_t(s_t, a_t, s_{t+1}) = (b_{t+1} + p^T_{t+1} * h_{t+1}) - (b_t + p^T_t * h_t) - c_t
        
        Where:
        - b_t: available balance at time t
        - p_t: price vector at time t
        - h_t: holdings vector at time t
        - c_t: transaction cost
        """
        
        # Calculate current portfolio value
        current_portfolio_value = self._calculate_portfolio_value()
        
        # Calculate raw return (change in portfolio value)
        raw_return = current_portfolio_value - self.last_portfolio_value
        
        # Calculate transaction cost if a trade was executed
        transaction_cost = 0
        if trade_result.get('success', False) and 'cost' in trade_result.get('trade_details', {}):
            # Transaction cost as per Equation (2): c_t = 0.1% * p^T * k
            trade_value = trade_result['trade_details'].get('cost', 0)
            transaction_cost = self.transaction_cost_rate * abs(trade_value)
        
        # Apply reward as per Equation (1)
        reward = raw_return - transaction_cost
        
        # Apply scaling factor as mentioned in section 3.1.5
        scaled_reward = reward * self.reward_scaling_factor
        
        # Update last portfolio value for next calculation
        self.last_portfolio_value = current_portfolio_value
        
        # Log for debugging
        logger.debug(f"Reward calculation: Portfolio change: {raw_return:.2f}, "
                    f"Transaction cost: {transaction_cost:.2f}, "
                    f"Scaled reward: {scaled_reward:.6f}")
        
        return scaled_reward
    
    def _check_turbulence_threshold(self) -> bool:
        """
        Implement turbulence threshold check as per section 3.1.4
        
        The paper uses turbulence index to stop trading during extreme market conditions
        """
        # This would need market-wide data to calculate properly
        # For now, return False (no turbulence)
        return False
    
    def reset(self, **kwargs):
        """Reset tracking variables"""
        obs = super().reset(**kwargs)
        self.last_portfolio_value = getattr(self, 'initial_capital', 100000)
        return obs


class PaperBasedTradingEnvironment:
    """
    Environment wrapper that implements the exact specifications from the research paper
    """
    
    def __init__(self, base_env):
        self.base_env = base_env
        
        # Paper specifications from section 3.1.5
        self.initial_capital = 1000000  # $1 million
        self.max_shares_per_trade = 100  # h_max
        self.reward_scaling_factor = 1e-4
        self.transaction_cost_rate = 0.001  # 0.1%
        
        # Override base environment parameters
        if hasattr(base_env, 'initial_capital'):
            base_env.initial_capital = self.initial_capital
        if hasattr(base_env, 'capital'):
            base_env.capital = self.initial_capital
            
        # Action space normalization (section 3.1.2)
        # The paper normalizes actions to [-1, 1]
        self.action_space_size = 11  # For discrete approximation
        
    def normalize_action(self, action: int) -> float:
        """
        Normalize discrete action to [-1, 1] range as per paper
        """
        # Map action index to normalized range
        return (action / (self.action_space_size - 1)) * 2 - 1
    
    def step(self, action):
        """
        Execute action with paper's specifications
        """
        # Normalize action as per paper
        normalized_action = self.normalize_action(action)
        
        # Execute in base environment
        obs, reward, done, info = self.base_env.step(action)
        
        # Apply paper's reward calculation
        portfolio_value = self._calculate_portfolio_value()
        raw_return = portfolio_value - self.last_portfolio_value
        
        # Calculate transaction cost
        transaction_cost = 0
        if 'trade_result' in info and info['trade_result'].get('success'):
            if 'trade_details' in info['trade_result']:
                trade_value = info['trade_result']['trade_details'].get('cost', 0)
                transaction_cost = self.transaction_cost_rate * abs(trade_value)
        
        # Paper's reward formula
        paper_reward = (raw_return - transaction_cost) * self.reward_scaling_factor
        
        self.last_portfolio_value = portfolio_value
        
        # Add paper-specific info
        info['paper_reward'] = paper_reward
        info['normalized_action'] = normalized_action
        info['transaction_cost'] = transaction_cost
        
        return obs, paper_reward, done, info
    
    def reset(self, **kwargs):
        """Reset with paper's specifications"""
        obs = self.base_env.reset(**kwargs)
        self.last_portfolio_value = self.initial_capital
        return obs
    
    def _calculate_portfolio_value(self):
        """Calculate total portfolio value"""
        if hasattr(self.base_env, '_calculate_portfolio_value'):
            return self.base_env._calculate_portfolio_value()
        elif hasattr(self.base_env, 'capital'):
            # Simplified calculation
            positions_value = 0
            if hasattr(self.base_env, 'positions'):
                for pos in self.base_env.positions:
                    if hasattr(pos, 'current_value'):
                        positions_value += pos.current_value
            return self.base_env.capital + positions_value
        else:
            return self.initial_capital
    
    def __getattr__(self, name):
        """Pass through to base environment"""
        return getattr(self.base_env, name)


def apply_paper_reward_structure(env_class):
    """
    Decorator to apply paper's reward structure to any environment class
    """
    class PaperRewardEnvironment(ResearchPaperRewardMixin, env_class):
        def __init__(self, **kwargs):
            # Override initial capital as per paper
            kwargs['initial_capital'] = 1000000  # $1 million
            super().__init__(**kwargs)
            
            # Set paper parameters
            self.reward_scaling_factor = 1e-4
            self.transaction_cost_rate = 0.001
            
    return PaperRewardEnvironment


# Additional reward improvements mentioned in the paper's conclusion
class ImprovedRewardFunction:
    """
    Implements improved reward functions mentioned in section 5
    """
    
    @staticmethod
    def sharpe_ratio_reward(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio-based reward as mentioned in the paper
        
        The paper suggests using Sharpe ratio to weigh risk and return
        """
        if len(returns) < 2:
            return 0.0
            
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
            
        # Annualized Sharpe ratio
        sharpe = (mean_return - risk_free_rate) / std_return * np.sqrt(252)
        
        return sharpe
    
    @staticmethod
    def normalize_reward(reward: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
        """
        Normalize reward to [-1, 1] range as suggested in the paper
        
        This helps with numerical stability during training
        """
        # Clip extreme values
        clipped = np.clip(reward, min_val * 10, max_val * 10)
        
        # Normalize using tanh for smooth scaling
        normalized = np.tanh(clipped / 2)
        
        return normalized