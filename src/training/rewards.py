"""Reward shaping functions for RL training"""

import numpy as np
from typing import Dict
from collections import deque
import logging

logger = logging.getLogger(__name__)


class SharpeRatioRewardShaper:
    """
    Sharpe Ratio-based reward shaping for risk-adjusted returns
    Encourages consistent profits with controlled risk
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.04,
        window_size: int = 50,
        sharpe_weight: float = 0.1,
        annualization_factor: float = 252
    ):
        self.risk_free_rate = risk_free_rate
        self.window_size = window_size
        self.sharpe_weight = sharpe_weight
        self.annualization_factor = annualization_factor
        self.returns_history = deque(maxlen=window_size)
        
    def calculate_sharpe_ratio(self) -> float:
        """Calculate annualized Sharpe ratio from recent returns"""
        if len(self.returns_history) < 10:
            return 0.0
        
        returns = np.array(self.returns_history)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return < 1e-6:
            return 0.0
        
        daily_rf = self.risk_free_rate / self.annualization_factor
        sharpe = (mean_return - daily_rf) / std_return * np.sqrt(self.annualization_factor)
        
        return sharpe
    
    def shape_reward(self, portfolio_return: float) -> float:
        """Shape reward with Sharpe ratio component"""
        self.returns_history.append(portfolio_return)
        sharpe = self.calculate_sharpe_ratio()
        sharpe_bonus = self.sharpe_weight * np.tanh(sharpe / 2.0)
        shaped_reward = portfolio_return + sharpe_bonus
        return shaped_reward
    
    def reset(self):
        """Reset history for new episode"""
        self.returns_history.clear()


class EnhancedRewardFunction:
    """
    Enhanced reward function based on paper's portfolio return approach
    Includes transaction costs and risk management
    """
    
    def __init__(self, transaction_cost_rate: float = 0.001, reward_scaling: float = 1e-4):
        self.transaction_cost_rate = transaction_cost_rate
        self.reward_scaling = reward_scaling
        
    def calculate_portfolio_return_reward(
        self,
        prev_portfolio_value: float,
        current_portfolio_value: float,
        transaction_costs: float
    ) -> float:
        """Calculate reward as portfolio value change minus transaction costs"""
        portfolio_change = current_portfolio_value - prev_portfolio_value
        net_return = portfolio_change - transaction_costs
        scaled_reward = net_return * self.reward_scaling
        return scaled_reward
    
    def calculate_transaction_costs(self, trades: Dict[str, float], prices: Dict[str, float]) -> float:
        """Calculate transaction costs as 0.1% of trade value"""
        total_cost = 0.0
        for symbol, quantity in trades.items():
            if symbol in prices and quantity != 0:
                trade_value = abs(quantity * prices[symbol])
                cost = trade_value * self.transaction_cost_rate
                total_cost += cost
        return total_cost

