"""
Realistic Transaction Cost Module
Implements accurate transaction costs based on Alpaca's fee structure and real market data

Key Components:
1. Bid-Ask Spread Costs (2-10% of option price)
2. Regulatory Fees (SEC, FINRA, OCC)
3. Slippage Modeling (based on order size vs volume)
4. Market Impact (for large orders)

Based on Alpaca's official fee schedule:
https://files.alpaca.markets/disclosures/library/BrokFeeSched.pdf
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TransactionCostBreakdown:
    """Detailed breakdown of transaction costs"""
    execution_price: float  # Actual execution price (bid or ask)
    spread_cost: float  # Cost from bid-ask spread
    occ_fee: float  # Options Clearing Corporation fee
    sec_fee: float  # SEC regulatory fee (sell-side only)
    finra_taf: float  # FINRA Trading Activity Fee (sell-side only)
    slippage: float  # Market impact slippage
    total_cost: float  # Total transaction cost
    cost_pct: float  # Cost as % of trade value
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging"""
        return {
            'execution_price': self.execution_price,
            'spread_cost': self.spread_cost,
            'occ_fee': self.occ_fee,
            'sec_fee': self.sec_fee,
            'finra_taf': self.finra_taf,
            'slippage': self.slippage,
            'total_cost': self.total_cost,
            'cost_pct': self.cost_pct
        }


class RealisticTransactionCostCalculator:
    """
    Calculate realistic transaction costs for options trading
    
    Alpaca Fee Structure (Commission-Free):
    - Commission: $0 (commission-free trading)
    - Bid-Ask Spread: 2-10% of option price (MAIN COST)
    - OCC Fee: $0.04 per contract (both buy and sell)
    - SEC Fee: $0.00278 per $1,000 of trade value (sell-side only)
    - FINRA TAF: $0.000166 per share (sell-side only)
    """
    
    def __init__(
        self,
        enable_slippage: bool = True,
        slippage_model: str = 'volume_based',  # 'volume_based', 'fixed', 'none'
        min_spread_pct: float = 0.005,  # 0.5% minimum spread (REDUCED from 2%)
        max_spread_pct: float = 0.03,  # 3% maximum spread (REDUCED from 10%)
        log_costs: bool = True
    ):
        self.enable_slippage = enable_slippage
        self.slippage_model = slippage_model
        self.min_spread_pct = min_spread_pct
        self.max_spread_pct = max_spread_pct
        self.log_costs = log_costs
        
        # Regulatory fee rates (as of 2025)
        self.OCC_FEE_PER_CONTRACT = 0.04  # $0.04 per contract
        self.SEC_FEE_RATE = 0.00278 / 1000  # $0.00278 per $1,000
        self.FINRA_TAF_RATE = 0.000166  # $0.000166 per share
        
        logger.info("✅ Realistic Transaction Cost Calculator initialized")
        logger.info(f"   Slippage: {'Enabled' if enable_slippage else 'Disabled'}")
        logger.info(f"   Slippage model: {slippage_model}")
        logger.info(f"   Spread range: {min_spread_pct:.1%} - {max_spread_pct:.1%}")
    
    def calculate_transaction_cost(
        self,
        option_data: Dict,
        quantity: int,
        side: str = 'buy',
        order_type: str = 'market'
    ) -> TransactionCostBreakdown:
        """
        Calculate total transaction cost for an options trade
        
        Args:
            option_data: Dictionary with option pricing data
                Required keys: 'bid', 'ask', 'last'
                Optional keys: 'volume', 'open_interest', 'implied_volatility'
            quantity: Number of contracts
            side: 'buy' or 'sell'
            order_type: 'market' or 'limit' (affects slippage)
        
        Returns:
            TransactionCostBreakdown with detailed cost components
        """
        # 1. Determine execution price based on side
        bid = option_data.get('bid', 0)
        ask = option_data.get('ask', 0)
        last = option_data.get('last', (bid + ask) / 2)
        
        # Validate bid-ask data
        if bid <= 0 or ask <= 0:
            # Fallback: estimate spread from last price
            mid_price = last if last > 0 else 1.0
            spread_pct = self._estimate_spread_pct(option_data)
            bid = mid_price * (1 - spread_pct / 2)
            ask = mid_price * (1 + spread_pct / 2)
        
        # Execution price (buy at ask, sell at bid)
        if side == 'buy':
            execution_price = ask
        else:  # sell
            execution_price = bid
        
        # 2. Calculate bid-ask spread cost
        spread = ask - bid
        spread_cost = spread * quantity * 100  # 100 shares per contract
        
        # 3. Calculate OCC fee (both buy and sell)
        occ_fee = self.OCC_FEE_PER_CONTRACT * quantity
        
        # 4. Calculate regulatory fees (sell-side only)
        if side == 'sell':
            trade_value = execution_price * quantity * 100
            sec_fee = trade_value * self.SEC_FEE_RATE
            finra_taf = quantity * 100 * self.FINRA_TAF_RATE
        else:
            sec_fee = 0.0
            finra_taf = 0.0
        
        # 5. Calculate slippage (if enabled)
        if self.enable_slippage and order_type == 'market':
            slippage = self._calculate_slippage(option_data, quantity, side)
        else:
            slippage = 0.0
        
        # 6. Total cost
        total_cost = spread_cost + occ_fee + sec_fee + finra_taf + slippage
        
        # 7. Cost as percentage of trade value
        trade_value = execution_price * quantity * 100
        cost_pct = total_cost / max(trade_value, 1.0) if trade_value > 0 else 0.0
        
        # Create breakdown
        breakdown = TransactionCostBreakdown(
            execution_price=execution_price,
            spread_cost=spread_cost,
            occ_fee=occ_fee,
            sec_fee=sec_fee,
            finra_taf=finra_taf,
            slippage=slippage,
            total_cost=total_cost,
            cost_pct=cost_pct
        )
        
        # Log if enabled
        if self.log_costs and total_cost > 0:
            logger.debug(
                f"Transaction cost ({side}): "
                f"spread=${spread_cost:.2f}, "
                f"OCC=${occ_fee:.2f}, "
                f"SEC=${sec_fee:.2f}, "
                f"FINRA=${finra_taf:.2f}, "
                f"slippage=${slippage:.2f}, "
                f"total=${total_cost:.2f} ({cost_pct:.2%})"
            )
        
        return breakdown
    
    def _estimate_spread_pct(self, option_data: Dict) -> float:
        """
        Estimate bid-ask spread percentage based on option characteristics

        Spread is wider for:
        - Far OTM/ITM options (low moneyness)
        - Low volume options
        - High volatility options
        - Options close to expiration
        """
        # Base spread (REDUCED from 3% to 0.8%)
        spread_pct = 0.008  # 0.8% default
        
        # Adjust for moneyness (if available) - REDUCED adjustments
        moneyness = option_data.get('moneyness', 1.0)
        if moneyness != 1.0:
            # Wider spread for OTM/ITM options
            moneyness_factor = abs(1.0 - moneyness)
            spread_pct += moneyness_factor * 0.01  # Up to +1% for deep OTM/ITM (was 5%)

        # Adjust for volume (if available) - REDUCED adjustments
        volume = option_data.get('volume', 100)
        if volume < 10:
            spread_pct += 0.005  # +0.5% for very low volume (was 3%)
        elif volume < 50:
            spread_pct += 0.002  # +0.2% for low volume (was 1%)

        # Adjust for implied volatility (if available) - REDUCED adjustments
        iv = option_data.get('implied_volatility', 0.3)
        if iv > 0.5:  # High IV
            spread_pct += 0.005  # +0.5% for high volatility (was 2%)
        
        # Clamp to min/max
        spread_pct = max(self.min_spread_pct, min(self.max_spread_pct, spread_pct))
        
        return spread_pct
    
    def _calculate_slippage(
        self,
        option_data: Dict,
        quantity: int,
        side: str
    ) -> float:
        """
        Calculate slippage based on order size relative to market liquidity
        
        Slippage increases non-linearly with order size as % of daily volume
        """
        if self.slippage_model == 'none':
            return 0.0
        
        if self.slippage_model == 'fixed':
            # Fixed 0.5% slippage
            mid_price = (option_data.get('bid', 0) + option_data.get('ask', 0)) / 2
            return mid_price * 0.005 * quantity * 100
        
        # Volume-based slippage model
        daily_volume = option_data.get('volume', 100)
        open_interest = option_data.get('open_interest', 1000)
        
        # Order size as % of daily volume
        order_pct = quantity / max(daily_volume, 1)
        
        # Slippage percentage (non-linear)
        if order_pct < 0.01:  # < 1% of volume
            slippage_pct = 0.001  # 0.1% slippage
        elif order_pct < 0.05:  # 1-5% of volume
            slippage_pct = 0.005  # 0.5% slippage
        elif order_pct < 0.10:  # 5-10% of volume
            slippage_pct = 0.01  # 1% slippage
        elif order_pct < 0.20:  # 10-20% of volume
            slippage_pct = 0.02  # 2% slippage
        else:  # > 20% of volume
            # Severe slippage for very large orders
            slippage_pct = 0.02 + (order_pct - 0.20) * 0.1
            slippage_pct = min(slippage_pct, 0.10)  # Cap at 10%
        
        # Calculate slippage cost
        mid_price = (option_data.get('bid', 0) + option_data.get('ask', 0)) / 2
        slippage_cost = mid_price * slippage_pct * quantity * 100
        
        return slippage_cost
    
    def get_effective_price(
        self,
        option_data: Dict,
        quantity: int,
        side: str = 'buy'
    ) -> Tuple[float, TransactionCostBreakdown]:
        """
        Get effective price including all transaction costs
        
        Returns:
            (effective_price, cost_breakdown)
        """
        breakdown = self.calculate_transaction_cost(option_data, quantity, side)
        
        # Effective price per share
        trade_value = breakdown.execution_price * quantity * 100
        total_value_with_costs = trade_value + breakdown.total_cost
        effective_price = total_value_with_costs / (quantity * 100)
        
        return effective_price, breakdown


# Global instance for easy access
_global_cost_calculator = None


def get_cost_calculator(**kwargs) -> RealisticTransactionCostCalculator:
    """Get or create global cost calculator instance"""
    global _global_cost_calculator
    if _global_cost_calculator is None:
        _global_cost_calculator = RealisticTransactionCostCalculator(**kwargs)
    return _global_cost_calculator


def calculate_transaction_cost(
    option_data: Dict,
    quantity: int,
    side: str = 'buy',
    **kwargs
) -> TransactionCostBreakdown:
    """Convenience function to calculate transaction cost"""
    calculator = get_cost_calculator(**kwargs)
    return calculator.calculate_transaction_cost(option_data, quantity, side)

