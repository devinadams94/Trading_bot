"""Expiration management for options positions"""

from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class ExpirationManager:
    """
    Manages option positions near expiration
    Auto-closes positions to avoid assignment risk
    """
    
    def __init__(
        self,
        close_days_threshold: int = 7,
        warning_days_threshold: int = 14,
        itm_close_threshold: float = 0.05
    ):
        self.close_days_threshold = close_days_threshold
        self.warning_days_threshold = warning_days_threshold
        self.itm_close_threshold = itm_close_threshold
    
    def should_close_position(
        self,
        days_to_expiry: int,
        moneyness: float,
        option_type: str
    ) -> Tuple[bool, str]:
        """
        Determine if position should be closed
        
        Args:
            days_to_expiry: Days until option expiration
            moneyness: How far ITM/OTM the option is
            option_type: 'call' or 'put'
            
        Returns:
            (should_close, reason)
        """
        # Close if very near expiration
        if days_to_expiry <= self.close_days_threshold:
            return True, f"Near expiration ({days_to_expiry} days)"
        
        # Close if deep ITM near expiration (assignment risk)
        if days_to_expiry <= self.warning_days_threshold:
            if option_type == 'call' and moneyness < -self.itm_close_threshold:
                return True, f"Deep ITM call near expiration (moneyness: {moneyness:.2%})"
            elif option_type == 'put' and moneyness > self.itm_close_threshold:
                return True, f"Deep ITM put near expiration (moneyness: {moneyness:.2%})"
        
        return False, ""
    
    def get_expiration_penalty(self, days_to_expiry: int) -> float:
        """
        Get penalty for holding positions near expiration
        
        Returns:
            Penalty value (0 to 1, higher = worse)
        """
        if days_to_expiry <= 0:
            return 1.0
        elif days_to_expiry <= self.close_days_threshold:
            return 0.5
        elif days_to_expiry <= self.warning_days_threshold:
            return 0.2
        else:
            return 0.0

