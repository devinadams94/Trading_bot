"""Transfer learning manager for CLSTM pre-training"""

import torch
import torch.nn as nn
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


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
            
            for i in range(0, len(stock_data), batch_size):
                batch = stock_data[i:i+batch_size]
                
                sequences = torch.stack([s['sequence'] for s in batch])
                price_targets = torch.stack([s['next_price'] for s in batch])
                vol_targets = torch.stack([s['volatility'] for s in batch])
                volume_targets = torch.stack([s['volume'] for s in batch])
                
                loss = self.clstm_encoder.get_supervised_loss(
                    sequences, price_targets, vol_targets, volume_targets
                )
                
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
        """Freeze bottom layers of CLSTM during fine-tuning"""
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

