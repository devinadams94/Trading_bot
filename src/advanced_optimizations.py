#!/usr/bin/env python3
"""
Advanced Optimizations for CLSTM-PPO Options Trading
Implements optimization suggestions from TRAINING_ENVIRONMENT_VERIFICATION.md
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
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
        risk_free_rate: float = 0.04,  # 4% annual risk-free rate
        window_size: int = 50,
        sharpe_weight: float = 0.1,
        annualization_factor: float = 252  # Trading days per year
    ):
        self.risk_free_rate = risk_free_rate
        self.window_size = window_size
        self.sharpe_weight = sharpe_weight
        self.annualization_factor = annualization_factor
        
        # Track returns for Sharpe calculation
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
        
        # Annualized Sharpe ratio
        daily_rf = self.risk_free_rate / self.annualization_factor
        sharpe = (mean_return - daily_rf) / std_return * np.sqrt(self.annualization_factor)
        
        return sharpe
    
    def shape_reward(self, portfolio_return: float) -> float:
        """
        Shape reward with Sharpe ratio component
        
        Args:
            portfolio_return: Raw portfolio return for this step
            
        Returns:
            Shaped reward combining return and Sharpe ratio
        """
        # Add return to history
        self.returns_history.append(portfolio_return)
        
        # Calculate Sharpe ratio
        sharpe = self.calculate_sharpe_ratio()
        
        # Combine portfolio return with Sharpe ratio bonus
        # Higher Sharpe = more consistent returns = bonus
        sharpe_bonus = self.sharpe_weight * np.tanh(sharpe / 2.0)  # Normalize to [-0.1, 0.1]
        
        shaped_reward = portfolio_return + sharpe_bonus
        
        return shaped_reward
    
    def reset(self):
        """Reset history for new episode"""
        self.returns_history.clear()


class ImpliedVolatilityPredictor(nn.Module):
    """
    Implied Volatility prediction head for CLSTM
    Helps the model learn options pricing dynamics
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.iv_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # IV is between 0 and 1 (0-100%)
        )
    
    def forward(self, encoded_features: torch.Tensor) -> torch.Tensor:
        """Predict implied volatility from encoded features"""
        iv = self.iv_predictor(encoded_features)
        return iv * 2.0  # Scale to [0, 2] for typical IV range


class GreeksBasedPositionSizer:
    """
    Adjust position sizes based on portfolio Greeks
    Manages directional risk (delta) and gamma exposure
    """
    
    def __init__(
        self,
        max_portfolio_delta: float = 0.5,
        max_portfolio_gamma: float = 0.3,
        max_theta_decay: float = 100.0,  # Max daily theta decay
        delta_reduction_factor: float = 0.5,
        gamma_reduction_factor: float = 0.7
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


class ExpirationManager:
    """
    Manages option positions near expiration
    Auto-closes positions to avoid assignment risk
    """
    
    def __init__(
        self,
        close_days_threshold: int = 7,
        warning_days_threshold: int = 14,
        itm_close_threshold: float = 0.05  # Close if 5% ITM
    ):
        self.close_days_threshold = close_days_threshold
        self.warning_days_threshold = warning_days_threshold
        self.itm_close_threshold = itm_close_threshold
    
    def should_close_position(
        self,
        days_to_expiry: int,
        moneyness: float,  # (strike - spot) / spot for calls, (spot - strike) / spot for puts
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
        Used to discourage the model from holding too long
        
        Returns:
            Penalty value (0 to 1, higher = worse)
        """
        if days_to_expiry <= 0:
            return 1.0  # Maximum penalty for expired options
        elif days_to_expiry <= self.close_days_threshold:
            return 0.5  # High penalty
        elif days_to_expiry <= self.warning_days_threshold:
            return 0.2  # Moderate penalty
        else:
            return 0.0  # No penalty


class EnsemblePredictor:
    """
    Ensemble of multiple CLSTM-PPO models for robust predictions
    Uses majority voting for discrete actions
    """
    
    def __init__(self, num_models: int = 3):
        self.num_models = num_models
        self.models = []
        self.model_weights = []  # Performance-based weights
    
    def add_model(self, model, weight: float = 1.0):
        """Add a model to the ensemble"""
        self.models.append(model)
        self.model_weights.append(weight)
    
    def predict_action(
        self,
        observation: Dict,
        deterministic: bool = False
    ) -> Tuple[int, float]:
        """
        Get ensemble prediction using weighted voting
        
        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            (action, confidence)
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Get predictions from all models
        actions = []
        action_probs = []
        
        for model in self.models:
            action, log_prob, _ = model.network.get_action(observation, deterministic)
            actions.append(action)
            action_probs.append(np.exp(log_prob.item()))
        
        # Weighted voting
        action_votes = {}
        for action, prob, weight in zip(actions, action_probs, self.model_weights):
            if action not in action_votes:
                action_votes[action] = 0.0
            action_votes[action] += prob * weight
        
        # Select action with highest weighted vote
        best_action = max(action_votes.items(), key=lambda x: x[1])
        action = best_action[0]
        confidence = best_action[1] / sum(action_votes.values())
        
        return action, confidence
    
    def update_weights(self, model_performances: List[float]):
        """
        Update model weights based on recent performance
        
        Args:
            model_performances: Performance metric for each model (higher = better)
        """
        if len(model_performances) != len(self.models):
            raise ValueError("Performance list must match number of models")
        
        # Softmax to convert performances to weights
        performances = np.array(model_performances)
        exp_perf = np.exp(performances - np.max(performances))  # Numerical stability
        self.model_weights = (exp_perf / exp_perf.sum()).tolist()
        
        logger.info(f"Updated ensemble weights: {[f'{w:.3f}' for w in self.model_weights]}")


class TransferLearningManager:
    """
    Manages transfer learning from stock data to options data
    Pre-trains CLSTM on large stock dataset, then fine-tunes on options
    """
    
    def __init__(self, clstm_encoder: nn.Module):
        self.clstm_encoder = clstm_encoder
        self.pretrain_optimizer = None
        self.pretrain_losses = []
    
    def pretrain_on_stock_data(
        self,
        stock_data: List[Dict],
        num_epochs: int = 10,
        learning_rate: float = 1e-3,
        batch_size: int = 64
    ):
        """
        Pre-train CLSTM encoder on stock price prediction
        
        Args:
            stock_data: List of stock data samples
            num_epochs: Number of pre-training epochs
            learning_rate: Learning rate for pre-training
            batch_size: Batch size
        """
        logger.info(f"Starting CLSTM pre-training on {len(stock_data)} stock samples")
        
        self.pretrain_optimizer = torch.optim.Adam(
            self.clstm_encoder.parameters(),
            lr=learning_rate
        )
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Mini-batch training
            for i in range(0, len(stock_data), batch_size):
                batch = stock_data[i:i+batch_size]
                
                # Prepare batch tensors
                sequences = torch.stack([s['sequence'] for s in batch])
                price_targets = torch.stack([s['next_price'] for s in batch])
                vol_targets = torch.stack([s['volatility'] for s in batch])
                volume_targets = torch.stack([s['volume'] for s in batch])
                
                # Forward pass
                loss = self.clstm_encoder.get_supervised_loss(
                    sequences, price_targets, vol_targets, volume_targets
                )
                
                # Backward pass
                self.pretrain_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.clstm_encoder.parameters(), 1.0)
                self.pretrain_optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / max(num_batches, 1)
            self.pretrain_losses.append(avg_loss)
            
            logger.info(f"Pre-training epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}")
        
        logger.info("✅ CLSTM pre-training complete")
    
    def freeze_encoder_layers(self, num_layers_to_freeze: int = 1):
        """
        Freeze bottom layers of CLSTM during fine-tuning
        Preserves learned low-level features
        """
        for i, lstm_layer in enumerate(self.clstm_encoder.lstm_layers):
            if i < num_layers_to_freeze:
                for param in lstm_layer.parameters():
                    param.requires_grad = False
                logger.info(f"Froze CLSTM layer {i}")
    
    def unfreeze_all_layers(self):
        """Unfreeze all layers for full fine-tuning"""
        for param in self.clstm_encoder.parameters():
            param.requires_grad = True
        logger.info("Unfroze all CLSTM layers")

