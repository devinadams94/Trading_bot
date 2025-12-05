"""
Unified Greeks Calculation Module

This module provides consistent Black-Scholes Greeks calculations for use in:
- Historical data preprocessing
- Live/paper trading environments

All environments should use THIS module to ensure state representation consistency.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Constants
TRADING_DAYS_PER_YEAR = 252
SECONDS_PER_TRADING_DAY = 6.5 * 3600  # 6.5 hours


@dataclass
class GreeksResult:
    """Container for Greeks calculation results"""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float = 0.0
    iv: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'rho': self.rho,
            'iv': self.iv
        }
    
    def to_array(self) -> np.ndarray:
        return np.array([self.delta, self.gamma, self.theta, self.vega])


class GreeksCalculator:
    """
    Black-Scholes Greeks Calculator
    
    Use the same instance for both historical and live data to ensure consistency.
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
        min_time_to_expiry: float = 1e-6,  # Minimum T to avoid division by zero
        default_iv: float = 0.30  # Default IV when calculation fails
    ):
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.min_time_to_expiry = min_time_to_expiry
        self.default_iv = default_iv
    
    def calculate_greeks(
        self,
        underlying_price: float,
        strike: float,
        time_to_expiry: float,  # In years
        iv: float,
        option_type: str = 'call',  # 'call' or 'put'
        risk_free_rate: Optional[float] = None
    ) -> GreeksResult:
        """
        Calculate all Greeks for an option.
        
        Args:
            underlying_price: Current price of underlying (S)
            strike: Strike price (K)
            time_to_expiry: Time to expiration in years (T)
            iv: Implied volatility (σ)
            option_type: 'call' or 'put'
            risk_free_rate: Override default risk-free rate
        
        Returns:
            GreeksResult with delta, gamma, theta, vega, rho
        """
        S = underlying_price
        K = strike
        T = max(time_to_expiry, self.min_time_to_expiry)
        sigma = max(iv, 0.001)  # Minimum IV to avoid issues
        r = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        q = self.dividend_yield
        is_call = option_type.lower() == 'call'
        
        # Validate inputs
        if S <= 0 or K <= 0:
            return self._zero_greeks(iv)
        
        try:
            # d1 and d2
            sqrt_T = np.sqrt(T)
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
            d2 = d1 - sigma * sqrt_T
            
            # Standard normal PDF and CDF
            N_d1 = norm.cdf(d1)
            N_d2 = norm.cdf(d2)
            N_neg_d1 = norm.cdf(-d1)
            N_neg_d2 = norm.cdf(-d2)
            n_d1 = norm.pdf(d1)
            
            # Discount factors
            exp_qT = np.exp(-q * T)
            exp_rT = np.exp(-r * T)
            
            # Delta
            if is_call:
                delta = exp_qT * N_d1
            else:
                delta = -exp_qT * N_neg_d1
            
            # Gamma (same for calls and puts)
            gamma = (exp_qT * n_d1) / (S * sigma * sqrt_T)
            
            # Theta (per day, negative means time decay)
            if is_call:
                theta = (-(S * sigma * exp_qT * n_d1) / (2 * sqrt_T)
                        - r * K * exp_rT * N_d2
                        + q * S * exp_qT * N_d1)
            else:
                theta = (-(S * sigma * exp_qT * n_d1) / (2 * sqrt_T)
                        + r * K * exp_rT * N_neg_d2
                        - q * S * exp_qT * N_neg_d1)
            
            # Convert theta to per-day (divide by trading days)
            theta = theta / TRADING_DAYS_PER_YEAR
            
            # Vega (per 1% move in IV)
            vega = S * exp_qT * sqrt_T * n_d1 / 100.0
            
            # Rho (per 1% move in interest rate)
            if is_call:
                rho = K * T * exp_rT * N_d2 / 100.0
            else:
                rho = -K * T * exp_rT * N_neg_d2 / 100.0
            
            return GreeksResult(
                delta=float(delta),
                gamma=float(gamma),
                theta=float(theta),
                vega=float(vega),
                rho=float(rho),
                iv=float(sigma)
            )
            
        except Exception as e:
            logger.warning(f"Greeks calculation failed: {e}")
            return self._zero_greeks(iv)
    
    def _zero_greeks(self, iv: float = 0.0) -> GreeksResult:
        """Return zero Greeks for invalid inputs"""
        return GreeksResult(delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0, iv=iv)

    def calculate_iv(
        self,
        option_price: float,
        underlying_price: float,
        strike: float,
        time_to_expiry: float,
        option_type: str = 'call',
        risk_free_rate: Optional[float] = None
    ) -> float:
        """
        Calculate implied volatility from option price using Brent's method.

        Args:
            option_price: Market price of the option
            underlying_price: Current underlying price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            option_type: 'call' or 'put'
            risk_free_rate: Override default risk-free rate

        Returns:
            Implied volatility (annualized)
        """
        S = underlying_price
        K = strike
        T = max(time_to_expiry, self.min_time_to_expiry)
        r = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        q = self.dividend_yield
        is_call = option_type.lower() == 'call'

        # Validate inputs
        if S <= 0 or K <= 0 or option_price <= 0:
            return self.default_iv

        # Check for intrinsic value violations
        if is_call:
            intrinsic = max(0, S * np.exp(-q * T) - K * np.exp(-r * T))
        else:
            intrinsic = max(0, K * np.exp(-r * T) - S * np.exp(-q * T))

        if option_price < intrinsic * 0.99:  # Allow small tolerance
            return self.default_iv

        def bs_price(sigma: float) -> float:
            """Black-Scholes price for given volatility"""
            if sigma <= 0:
                return 0.0

            sqrt_T = np.sqrt(T)
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
            d2 = d1 - sigma * sqrt_T

            if is_call:
                price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

            return price

        def objective(sigma: float) -> float:
            return bs_price(sigma) - option_price

        try:
            # Brent's method with reasonable bounds
            iv = brentq(objective, 0.001, 5.0, xtol=1e-6, maxiter=100)
            return float(iv)
        except (ValueError, RuntimeError):
            # Fallback: try wider bounds or return default
            try:
                iv = brentq(objective, 0.0001, 10.0, xtol=1e-4, maxiter=50)
                return float(iv)
            except:
                return self.default_iv

    def calculate_greeks_from_price(
        self,
        option_price: float,
        underlying_price: float,
        strike: float,
        time_to_expiry: float,
        option_type: str = 'call',
        risk_free_rate: Optional[float] = None
    ) -> GreeksResult:
        """
        Calculate Greeks when you have option price but not IV.
        First calculates IV, then calculates Greeks.

        This is the recommended method for historical data where IV may be missing.
        """
        iv = self.calculate_iv(
            option_price=option_price,
            underlying_price=underlying_price,
            strike=strike,
            time_to_expiry=time_to_expiry,
            option_type=option_type,
            risk_free_rate=risk_free_rate
        )

        return self.calculate_greeks(
            underlying_price=underlying_price,
            strike=strike,
            time_to_expiry=time_to_expiry,
            iv=iv,
            option_type=option_type,
            risk_free_rate=risk_free_rate
        )

    def calculate_iv_batch(
        self,
        option_prices: np.ndarray,
        underlying_prices: np.ndarray,
        strikes: np.ndarray,
        times_to_expiry: np.ndarray,
        option_types: np.ndarray,
        risk_free_rate: Optional[float] = None,
        n_jobs: int = -1
    ) -> np.ndarray:
        """
        Calculate implied volatility for a batch of options using vectorized Newton-Raphson.

        This is ~100x faster than the sequential Brent's method.

        Args:
            option_prices: Array of option prices
            underlying_prices: Array of underlying prices
            strikes: Array of strike prices
            times_to_expiry: Array of times to expiry (years)
            option_types: Array of option types ('call'/'put')
            risk_free_rate: Override default risk-free rate
            n_jobs: Number of parallel jobs (unused, kept for API compatibility)

        Returns:
            Array of implied volatilities
        """
        return self._vectorized_newton_raphson_iv(
            option_prices=np.asarray(option_prices, dtype=np.float64),
            underlying_prices=np.asarray(underlying_prices, dtype=np.float64),
            strikes=np.asarray(strikes, dtype=np.float64),
            times_to_expiry=np.asarray(times_to_expiry, dtype=np.float64),
            option_types=option_types,
            risk_free_rate=risk_free_rate
        )

    def _vectorized_newton_raphson_iv(
        self,
        option_prices: np.ndarray,
        underlying_prices: np.ndarray,
        strikes: np.ndarray,
        times_to_expiry: np.ndarray,
        option_types: np.ndarray,
        risk_free_rate: Optional[float] = None,
        max_iterations: int = 50,
        tolerance: float = 1e-6
    ) -> np.ndarray:
        """
        Vectorized Newton-Raphson IV solver - processes all options simultaneously.

        Uses vega (dPrice/dSigma) as the derivative for Newton-Raphson iteration.
        """
        n = len(option_prices)
        r = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        q = self.dividend_yield

        # Convert option types to boolean mask (True = call)
        if isinstance(option_types, np.ndarray):
            is_call = np.array([str(t).lower() == 'call' for t in option_types])
        else:
            is_call = np.array([str(option_types).lower() == 'call'] * n)

        S = underlying_prices
        K = strikes
        T = np.maximum(times_to_expiry, self.min_time_to_expiry)
        prices = option_prices

        # Initialize IV with a reasonable starting guess based on moneyness
        moneyness = S / K
        sigma = np.where(
            is_call,
            np.where(moneyness < 1.0, 0.3 + 0.2 * (1.0 - moneyness), 0.25),
            np.where(moneyness > 1.0, 0.3 + 0.2 * (moneyness - 1.0), 0.25)
        )

        # Validate inputs - mark invalid ones
        valid = (S > 0) & (K > 0) & (prices > 0) & (T > 0)

        # Calculate intrinsic values
        exp_qT = np.exp(-q * T)
        exp_rT = np.exp(-r * T)
        intrinsic_call = np.maximum(0, S * exp_qT - K * exp_rT)
        intrinsic_put = np.maximum(0, K * exp_rT - S * exp_qT)
        intrinsic = np.where(is_call, intrinsic_call, intrinsic_put)

        # Mark options with price below intrinsic as invalid
        valid = valid & (prices >= intrinsic * 0.99)

        # Track convergence
        converged = np.zeros(n, dtype=bool)
        sqrt_T = np.sqrt(T)

        for iteration in range(max_iterations):
            # Only process unconverged valid options
            mask = valid & ~converged
            if not np.any(mask):
                break

            # Vectorized Black-Scholes price calculation
            sig_sqrt_T = sigma[mask] * sqrt_T[mask]
            d1 = (np.log(S[mask] / K[mask]) + (r - q + 0.5 * sigma[mask]**2) * T[mask]) / sig_sqrt_T
            d2 = d1 - sig_sqrt_T

            # Calculate prices
            Nd1 = norm.cdf(d1)
            Nd2 = norm.cdf(d2)
            Nmd1 = norm.cdf(-d1)
            Nmd2 = norm.cdf(-d2)

            call_price = S[mask] * exp_qT[mask] * Nd1 - K[mask] * exp_rT[mask] * Nd2
            put_price = K[mask] * exp_rT[mask] * Nmd2 - S[mask] * exp_qT[mask] * Nmd1

            bs_price = np.where(is_call[mask], call_price, put_price)

            # Calculate vega (derivative of price w.r.t. sigma)
            pdf_d1 = norm.pdf(d1)
            vega = S[mask] * exp_qT[mask] * sqrt_T[mask] * pdf_d1

            # Newton-Raphson update: sigma_new = sigma - (price - target) / vega
            price_diff = bs_price - prices[mask]

            # Avoid division by zero
            vega_safe = np.maximum(vega, 1e-10)
            update = price_diff / vega_safe

            # Limit update size for stability
            update = np.clip(update, -0.5, 0.5)
            new_sigma = sigma[mask] - update
            new_sigma = np.clip(new_sigma, 0.01, 5.0)

            sigma[mask] = new_sigma
            converged[mask] = np.abs(price_diff) < tolerance

        # For non-converged options, use default IV
        sigma[~converged & valid] = self.default_iv
        sigma[~valid] = self.default_iv
        sigma = np.clip(sigma, 0.01, 5.0)

        return sigma.astype(np.float32)

    def calculate_greeks_from_price_batch(
        self,
        option_prices: np.ndarray,
        underlying_prices: np.ndarray,
        strikes: np.ndarray,
        times_to_expiry: np.ndarray,
        option_types: np.ndarray,
        risk_free_rate: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Calculate Greeks for a batch of options from prices.
        First calculates IV, then calculates Greeks.

        This is the recommended method for processing flat files.

        Args:
            option_prices: Array of option prices (use mid or close price)
            underlying_prices: Array of underlying prices
            strikes: Array of strike prices
            times_to_expiry: Array of times to expiry (years)
            option_types: Array of option types ('call'/'put')
            risk_free_rate: Override default risk-free rate

        Returns:
            Dictionary with arrays for delta, gamma, theta, vega, iv
        """
        # First calculate IVs
        ivs = self.calculate_iv_batch(
            option_prices=option_prices,
            underlying_prices=underlying_prices,
            strikes=strikes,
            times_to_expiry=times_to_expiry,
            option_types=option_types,
            risk_free_rate=risk_free_rate
        )

        # Then calculate Greeks from IVs
        return self.calculate_greeks_batch(
            underlying_prices=underlying_prices,
            strikes=strikes,
            times_to_expiry=times_to_expiry,
            ivs=ivs,
            option_types=option_types,
            risk_free_rate=risk_free_rate
        )

    def calculate_greeks_batch(
        self,
        underlying_prices: np.ndarray,
        strikes: np.ndarray,
        times_to_expiry: np.ndarray,
        ivs: np.ndarray,
        option_types: np.ndarray,  # Array of 'call'/'put' or 1/-1
        risk_free_rate: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Vectorized Greeks calculation for batch processing.

        Args:
            underlying_prices: Array of underlying prices
            strikes: Array of strike prices
            times_to_expiry: Array of times to expiry (years)
            ivs: Array of implied volatilities
            option_types: Array of option types ('call'/'put' or 1 for call, -1 for put)

        Returns:
            Dictionary with arrays for delta, gamma, theta, vega
        """
        n = len(underlying_prices)

        S = np.array(underlying_prices, dtype=np.float64)
        K = np.array(strikes, dtype=np.float64)
        T = np.maximum(np.array(times_to_expiry, dtype=np.float64), self.min_time_to_expiry)
        sigma = np.maximum(np.array(ivs, dtype=np.float64), 0.001)
        r = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        q = self.dividend_yield

        # Convert option types to boolean array
        if isinstance(option_types[0], str):
            is_call = np.array([t.lower() == 'call' for t in option_types])
        else:
            is_call = np.array(option_types) > 0

        # Vectorized calculations
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        n_d1 = norm.pdf(d1)

        exp_qT = np.exp(-q * T)
        exp_rT = np.exp(-r * T)

        # Delta
        delta = np.where(is_call, exp_qT * N_d1, -exp_qT * (1 - N_d1))

        # Gamma
        gamma = (exp_qT * n_d1) / (S * sigma * sqrt_T)

        # Theta (per day)
        base_theta = -(S * sigma * exp_qT * n_d1) / (2 * sqrt_T)
        call_theta = base_theta - r * K * exp_rT * N_d2 + q * S * exp_qT * N_d1
        put_theta = base_theta + r * K * exp_rT * (1 - N_d2) - q * S * exp_qT * (1 - N_d1)
        theta = np.where(is_call, call_theta, put_theta) / TRADING_DAYS_PER_YEAR

        # Vega (per 1% IV move)
        vega = S * exp_qT * sqrt_T * n_d1 / 100.0

        # Handle invalid inputs
        invalid = (S <= 0) | (K <= 0) | np.isnan(d1) | np.isinf(d1)
        delta = np.where(invalid, 0.0, delta)
        gamma = np.where(invalid, 0.0, gamma)
        theta = np.where(invalid, 0.0, theta)
        vega = np.where(invalid, 0.0, vega)

        return {
            'delta': delta.astype(np.float32),
            'gamma': gamma.astype(np.float32),
            'theta': theta.astype(np.float32),
            'vega': vega.astype(np.float32),
            'iv': sigma.astype(np.float32)
        }

    def time_to_expiry_years(
        self,
        expiry_timestamp: Union[int, float],
        current_timestamp: Union[int, float]
    ) -> float:
        """
        Convert timestamps to time-to-expiry in years.

        Args:
            expiry_timestamp: Unix timestamp of expiration
            current_timestamp: Unix timestamp of current time

        Returns:
            Time to expiry in years (trading time)
        """
        seconds_remaining = expiry_timestamp - current_timestamp
        if seconds_remaining <= 0:
            return self.min_time_to_expiry

        # Convert to trading years (252 days, 6.5 hours per day)
        trading_seconds_per_year = TRADING_DAYS_PER_YEAR * SECONDS_PER_TRADING_DAY
        return seconds_remaining / trading_seconds_per_year

    def time_to_expiry_from_days(self, days: float) -> float:
        """Convert calendar days to trading years"""
        trading_days = days * (5/7)  # Rough weekday adjustment
        return trading_days / TRADING_DAYS_PER_YEAR


# Global singleton instance for consistency
_default_calculator: Optional[GreeksCalculator] = None


def get_greeks_calculator(
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0
) -> GreeksCalculator:
    """
    Get or create the global Greeks calculator.

    Use this to ensure the same calculator is used everywhere.
    """
    global _default_calculator
    if _default_calculator is None:
        _default_calculator = GreeksCalculator(
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield
        )
    return _default_calculator


def calculate_greeks(
    underlying_price: float,
    strike: float,
    time_to_expiry: float,
    iv: float,
    option_type: str = 'call'
) -> GreeksResult:
    """
    Convenience function to calculate Greeks using the global calculator.
    """
    return get_greeks_calculator().calculate_greeks(
        underlying_price=underlying_price,
        strike=strike,
        time_to_expiry=time_to_expiry,
        iv=iv,
        option_type=option_type
    )


def calculate_greeks_from_price(
    option_price: float,
    underlying_price: float,
    strike: float,
    time_to_expiry: float,
    option_type: str = 'call'
) -> GreeksResult:
    """
    Convenience function to calculate Greeks from option price.
    """
    return get_greeks_calculator().calculate_greeks_from_price(
        option_price=option_price,
        underlying_price=underlying_price,
        strike=strike,
        time_to_expiry=time_to_expiry,
        option_type=option_type
    )

