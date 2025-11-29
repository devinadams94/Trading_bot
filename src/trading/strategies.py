"""
Multi-Leg Options Strategies for Enhanced Trading
Implements Phase 2 strategy diversity improvements

Strategies included:
1. Bull Call Spread - Defined risk bullish
2. Bear Put Spread - Defined risk bearish
3. Long Straddle - High volatility play
4. Long Strangle - Lower cost volatility play
5. Iron Condor - Range-bound income
6. Butterfly Spread - Neutral strategy
7. Covered Call - Income generation
8. Cash-Secured Put - Income generation
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Types of multi-leg strategies"""
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    LONG_STRADDLE = "long_straddle"
    LONG_STRANGLE = "long_strangle"
    IRON_CONDOR = "iron_condor"
    BUTTERFLY_SPREAD = "butterfly_spread"
    COVERED_CALL = "covered_call"
    CASH_SECURED_PUT = "cash_secured_put"


@dataclass
class OptionLeg:
    """Single leg of a multi-leg strategy"""
    option_type: str  # 'call' or 'put'
    action: str  # 'buy' or 'sell'
    strike: float
    quantity: int
    expiration_days: int


@dataclass
class MultiLegStrategy:
    """Complete multi-leg strategy definition"""
    strategy_type: StrategyType
    legs: List[OptionLeg]
    max_profit: float
    max_loss: float
    breakeven_points: List[float]
    capital_required: float
    risk_profile: str  # 'bullish', 'bearish', 'neutral', 'volatility'


class MultiLegStrategyBuilder:
    """Builder for creating multi-leg options strategies"""
    
    def __init__(self):
        self.strategies = {}
    
    def build_bull_call_spread(
        self,
        current_price: float,
        quantity: int = 1,
        expiration_days: int = 30
    ) -> MultiLegStrategy:
        """
        Bull Call Spread: Buy ATM call, sell OTM call
        - Max profit: Difference in strikes - net debit
        - Max loss: Net debit paid
        - Breakeven: Long strike + net debit
        """
        atm_strike = round(current_price, 0)
        otm_strike = round(current_price * 1.05, 0)  # 5% OTM
        
        legs = [
            OptionLeg('call', 'buy', atm_strike, quantity, expiration_days),
            OptionLeg('call', 'sell', otm_strike, quantity, expiration_days)
        ]
        
        # Estimate costs (simplified)
        long_call_cost = current_price * 0.05 * quantity * 100  # ~5% of stock price
        short_call_credit = current_price * 0.02 * quantity * 100  # ~2% of stock price
        net_debit = long_call_cost - short_call_credit
        
        max_profit = (otm_strike - atm_strike) * quantity * 100 - net_debit
        max_loss = net_debit
        breakeven = atm_strike + (net_debit / (quantity * 100))
        
        return MultiLegStrategy(
            strategy_type=StrategyType.BULL_CALL_SPREAD,
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=[breakeven],
            capital_required=net_debit,
            risk_profile='bullish'
        )
    
    def build_bear_put_spread(
        self,
        current_price: float,
        quantity: int = 1,
        expiration_days: int = 30
    ) -> MultiLegStrategy:
        """
        Bear Put Spread: Buy ATM put, sell OTM put
        - Max profit: Difference in strikes - net debit
        - Max loss: Net debit paid
        - Breakeven: Long strike - net debit
        """
        atm_strike = round(current_price, 0)
        otm_strike = round(current_price * 0.95, 0)  # 5% OTM
        
        legs = [
            OptionLeg('put', 'buy', atm_strike, quantity, expiration_days),
            OptionLeg('put', 'sell', otm_strike, quantity, expiration_days)
        ]
        
        # Estimate costs
        long_put_cost = current_price * 0.05 * quantity * 100
        short_put_credit = current_price * 0.02 * quantity * 100
        net_debit = long_put_cost - short_put_credit
        
        max_profit = (atm_strike - otm_strike) * quantity * 100 - net_debit
        max_loss = net_debit
        breakeven = atm_strike - (net_debit / (quantity * 100))
        
        return MultiLegStrategy(
            strategy_type=StrategyType.BEAR_PUT_SPREAD,
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=[breakeven],
            capital_required=net_debit,
            risk_profile='bearish'
        )
    
    def build_long_straddle(
        self,
        current_price: float,
        quantity: int = 1,
        expiration_days: int = 30
    ) -> MultiLegStrategy:
        """
        Long Straddle: Buy ATM call + ATM put
        - Max profit: Unlimited
        - Max loss: Total premium paid
        - Breakeven: Strike ± total premium
        """
        atm_strike = round(current_price, 0)
        
        legs = [
            OptionLeg('call', 'buy', atm_strike, quantity, expiration_days),
            OptionLeg('put', 'buy', atm_strike, quantity, expiration_days)
        ]
        
        # Estimate costs
        call_cost = current_price * 0.05 * quantity * 100
        put_cost = current_price * 0.05 * quantity * 100
        total_cost = call_cost + put_cost
        
        max_profit = float('inf')  # Unlimited
        max_loss = total_cost
        breakeven_upper = atm_strike + (total_cost / (quantity * 100))
        breakeven_lower = atm_strike - (total_cost / (quantity * 100))
        
        return MultiLegStrategy(
            strategy_type=StrategyType.LONG_STRADDLE,
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=[breakeven_lower, breakeven_upper],
            capital_required=total_cost,
            risk_profile='volatility'
        )
    
    def build_long_strangle(
        self,
        current_price: float,
        quantity: int = 1,
        expiration_days: int = 30
    ) -> MultiLegStrategy:
        """
        Long Strangle: Buy OTM call + OTM put
        - Max profit: Unlimited
        - Max loss: Total premium paid
        - Breakeven: Strikes ± total premium
        """
        otm_call_strike = round(current_price * 1.05, 0)  # 5% OTM
        otm_put_strike = round(current_price * 0.95, 0)  # 5% OTM
        
        legs = [
            OptionLeg('call', 'buy', otm_call_strike, quantity, expiration_days),
            OptionLeg('put', 'buy', otm_put_strike, quantity, expiration_days)
        ]
        
        # Estimate costs (cheaper than straddle)
        call_cost = current_price * 0.03 * quantity * 100
        put_cost = current_price * 0.03 * quantity * 100
        total_cost = call_cost + put_cost
        
        max_profit = float('inf')
        max_loss = total_cost
        breakeven_upper = otm_call_strike + (total_cost / (quantity * 100))
        breakeven_lower = otm_put_strike - (total_cost / (quantity * 100))
        
        return MultiLegStrategy(
            strategy_type=StrategyType.LONG_STRANGLE,
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=[breakeven_lower, breakeven_upper],
            capital_required=total_cost,
            risk_profile='volatility'
        )
    
    def build_iron_condor(
        self,
        current_price: float,
        quantity: int = 1,
        expiration_days: int = 30
    ) -> MultiLegStrategy:
        """
        Iron Condor: Sell call spread + sell put spread
        - Max profit: Net credit received
        - Max loss: Width of spread - net credit
        - Breakeven: Short strikes ± net credit
        """
        # Call spread: Sell ATM call, buy OTM call
        short_call_strike = round(current_price * 1.05, 0)
        long_call_strike = round(current_price * 1.10, 0)
        
        # Put spread: Sell ATM put, buy OTM put
        short_put_strike = round(current_price * 0.95, 0)
        long_put_strike = round(current_price * 0.90, 0)
        
        legs = [
            OptionLeg('call', 'sell', short_call_strike, quantity, expiration_days),
            OptionLeg('call', 'buy', long_call_strike, quantity, expiration_days),
            OptionLeg('put', 'sell', short_put_strike, quantity, expiration_days),
            OptionLeg('put', 'buy', long_put_strike, quantity, expiration_days)
        ]
        
        # Estimate net credit
        short_call_credit = current_price * 0.03 * quantity * 100
        long_call_cost = current_price * 0.01 * quantity * 100
        short_put_credit = current_price * 0.03 * quantity * 100
        long_put_cost = current_price * 0.01 * quantity * 100
        net_credit = (short_call_credit + short_put_credit) - (long_call_cost + long_put_cost)
        
        spread_width = (long_call_strike - short_call_strike) * quantity * 100
        max_profit = net_credit
        max_loss = spread_width - net_credit
        breakeven_upper = short_call_strike + (net_credit / (quantity * 100))
        breakeven_lower = short_put_strike - (net_credit / (quantity * 100))
        
        return MultiLegStrategy(
            strategy_type=StrategyType.IRON_CONDOR,
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=[breakeven_lower, breakeven_upper],
            capital_required=max_loss,  # Margin requirement
            risk_profile='neutral'
        )
    
    def build_butterfly_spread(
        self,
        current_price: float,
        quantity: int = 1,
        expiration_days: int = 30
    ) -> MultiLegStrategy:
        """
        Butterfly Spread: Buy 1 ITM call, sell 2 ATM calls, buy 1 OTM call
        - Max profit: Middle strike - lower strike - net debit
        - Max loss: Net debit paid
        - Breakeven: Lower strike + net debit, upper strike - net debit
        """
        lower_strike = round(current_price * 0.95, 0)
        middle_strike = round(current_price, 0)
        upper_strike = round(current_price * 1.05, 0)
        
        legs = [
            OptionLeg('call', 'buy', lower_strike, quantity, expiration_days),
            OptionLeg('call', 'sell', middle_strike, quantity * 2, expiration_days),
            OptionLeg('call', 'buy', upper_strike, quantity, expiration_days)
        ]
        
        # Estimate costs
        lower_call_cost = current_price * 0.06 * quantity * 100
        middle_call_credit = current_price * 0.05 * quantity * 2 * 100
        upper_call_cost = current_price * 0.03 * quantity * 100
        net_debit = lower_call_cost + upper_call_cost - middle_call_credit
        
        max_profit = (middle_strike - lower_strike) * quantity * 100 - net_debit
        max_loss = net_debit
        breakeven_lower = lower_strike + (net_debit / (quantity * 100))
        breakeven_upper = upper_strike - (net_debit / (quantity * 100))
        
        return MultiLegStrategy(
            strategy_type=StrategyType.BUTTERFLY_SPREAD,
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=[breakeven_lower, breakeven_upper],
            capital_required=net_debit,
            risk_profile='neutral'
        )
    
    def get_strategy_description(self, strategy_type: StrategyType) -> str:
        """Get human-readable description of strategy"""
        descriptions = {
            StrategyType.BULL_CALL_SPREAD: "Bullish defined-risk strategy using call spread",
            StrategyType.BEAR_PUT_SPREAD: "Bearish defined-risk strategy using put spread",
            StrategyType.LONG_STRADDLE: "High volatility play expecting large price movement",
            StrategyType.LONG_STRANGLE: "Lower cost volatility play with wider breakevens",
            StrategyType.IRON_CONDOR: "Range-bound income strategy with defined risk",
            StrategyType.BUTTERFLY_SPREAD: "Neutral strategy profiting from low volatility"
        }
        return descriptions.get(strategy_type, "Unknown strategy")

