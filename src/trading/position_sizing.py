"""Greeks-based position sizing for options trading"""

import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class GreeksBasedPositionSizer:
    """
    Position sizing based on portfolio Greeks
    Reduces position size when Greeks exposure is too high
    """
    
    def __init__(
        self,
        max_portfolio_delta: float = 0.5,
        max_portfolio_gamma: float = 0.1,
        max_theta_decay: float = 0.02,
        delta_reduction_factor: float = 0.5,
        gamma_reduction_factor: float = 0.3
    ):
        self.max_portfolio_delta = max_portfolio_delta
        self.max_portfolio_gamma = max_portfolio_gamma
        self.max_theta_decay = max_theta_decay
        self.delta_reduction_factor = delta_reduction_factor
        self.gamma_reduction_factor = gamma_reduction_factor
    
    def adjust_position_size(
        self,
        base_position_size: float,
        portfolio_greeks: Dict[str, float],
        new_position_greeks: Dict[str, float]
    ) -> float:
        """
        Adjust position size based on portfolio Greeks
        
        Args:
            base_position_size: Original position size (0-1)
            portfolio_greeks: Current portfolio Greeks
            new_position_greeks: Greeks of new position to add
            
        Returns:
            Adjusted position size
        """
        adjusted_size = base_position_size
        
        # Calculate new portfolio Greeks if position is added
        new_delta = abs(portfolio_greeks.get('delta', 0) + new_position_greeks.get('delta', 0))
        new_gamma = abs(portfolio_greeks.get('gamma', 0) + new_position_greeks.get('gamma', 0))
        new_theta = abs(portfolio_greeks.get('theta', 0) + new_position_greeks.get('theta', 0))
        
        # Reduce size if delta exposure too high
        if new_delta > self.max_portfolio_delta:
            delta_ratio = self.max_portfolio_delta / max(new_delta, 1e-6)
            adjusted_size *= (delta_ratio * self.delta_reduction_factor)
            logger.debug(f"Reducing position size due to delta: {new_delta:.3f} > {self.max_portfolio_delta}")
        
        # Reduce size if gamma exposure too high
        if new_gamma > self.max_portfolio_gamma:
            gamma_ratio = self.max_portfolio_gamma / max(new_gamma, 1e-6)
            adjusted_size *= (gamma_ratio * self.gamma_reduction_factor)
            logger.debug(f"Reducing position size due to gamma: {new_gamma:.3f} > {self.max_portfolio_gamma}")
        
        # Reduce size if theta decay too high
        if new_theta > self.max_theta_decay:
            theta_ratio = self.max_theta_decay / max(new_theta, 1e-6)
            adjusted_size *= theta_ratio
            logger.debug(f"Reducing position size due to theta: {new_theta:.3f} > {self.max_theta_decay}")
        
        # Ensure size is in valid range
        adjusted_size = np.clip(adjusted_size, 0.0, 1.0)
        
        return adjusted_size

