import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from collections import deque
import random
from datetime import datetime

logger = logging.getLogger(__name__)


class CLSTMEncoder(nn.Module):
    """Cascaded LSTM Encoder specifically for options trading features"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Cascaded LSTM layers with attention
        self.lstm_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(num_layers):
            # LSTM layer
            self.lstm_layers.append(
                nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=False)
            )
            
            # Multi-head attention layer
            self.attention_layers.append(
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                )
            )
            
            # Layer normalization
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            
            # Dropout
            self.dropouts.append(nn.Dropout(dropout))
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, return_sequences: bool = False) -> torch.Tensor:
        # Input shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x)
        
        # Store intermediate representations for supervised learning
        layer_outputs = []
        
        # Pass through cascaded LSTM layers
        for i in range(self.num_layers):
            # LSTM forward pass
            lstm_out, (h_n, c_n) = self.lstm_layers[i](x)
            
            # Self-attention
            attn_out, _ = self.attention_layers[i](lstm_out, lstm_out, lstm_out)
            
            # Residual connection + layer norm
            x = self.layer_norms[i](lstm_out + attn_out)
            
            # Dropout
            x = self.dropouts[i](x)
            
            # Store layer output
            layer_outputs.append(x)
        
        # Output projection
        output = self.output_projection(x)
        
        if return_sequences:
            return output, layer_outputs
        else:
            # Return only the last timestep
            return output[:, -1, :]
    
    def get_supervised_loss(
        self,
        x: torch.Tensor,
        price_targets: torch.Tensor,
        volatility_targets: torch.Tensor,
        volume_targets: torch.Tensor
    ) -> torch.Tensor:
        """Supervised learning loss for pre-training"""
        output, layer_outputs = self.forward(x, return_sequences=True)
        
        # Use different layers for different predictions
        price_pred = self.price_head(layer_outputs[0][:, -1, :])
        vol_pred = self.volatility_head(layer_outputs[1][:, -1, :])
        volume_pred = self.volume_head(layer_outputs[2][:, -1, :])
        
        # Calculate losses (ensure predictions and targets have same shape)
        price_pred_flat = price_pred.squeeze(-1) if price_pred.dim() > 1 else price_pred
        vol_pred_flat = vol_pred.squeeze(-1) if vol_pred.dim() > 1 else vol_pred
        volume_pred_flat = volume_pred.squeeze(-1) if volume_pred.dim() > 1 else volume_pred

        price_targets_flat = price_targets.squeeze(-1) if price_targets.dim() > 1 else price_targets
        vol_targets_flat = volatility_targets.squeeze(-1) if volatility_targets.dim() > 1 else volatility_targets
        volume_targets_flat = volume_targets.squeeze(-1) if volume_targets.dim() > 1 else volume_targets

        price_loss = F.mse_loss(price_pred_flat, price_targets_flat)
        vol_loss = F.mse_loss(vol_pred_flat, vol_targets_flat)
        volume_loss = F.mse_loss(volume_pred_flat, volume_targets_flat)
        
        return price_loss + vol_loss + volume_loss


class OptionsCLSTMPPONetwork(nn.Module):
    """Combined CLSTM-PPO network for options trading"""
    
    def __init__(
        self,
        observation_space: Dict,
        action_dim: int = 11,
        hidden_dim: int = 256,
        clstm_layers: int = 3
    ):
        super().__init__()
        
        # Calculate input dimensions
        price_history_dim = np.prod(observation_space['price_history'].shape)
        technical_dim = observation_space['technical_indicators'].shape[0]
        options_chain_dim = np.prod(observation_space['options_chain'].shape)
        portfolio_dim = observation_space['portfolio_state'].shape[0]
        greeks_dim = observation_space['greeks_summary'].shape[0]
        
        # Check if symbol encoding is provided
        self.has_symbol_encoding = 'symbol_encoding' in observation_space
        symbol_dim = 0
        if self.has_symbol_encoding:
            symbol_dim = observation_space['symbol_encoding'].shape[0]
            # Create symbol-specific layers
            self.symbol_embedding = nn.Linear(symbol_dim, hidden_dim // 4)
        
        total_input_dim = (
            price_history_dim + technical_dim + options_chain_dim +
            portfolio_dim + greeks_dim + symbol_dim
        )
        
        # CLSTM encoder for temporal features
        self.clstm_encoder = CLSTMEncoder(
            input_dim=total_input_dim,
            hidden_dim=hidden_dim,
            num_layers=clstm_layers
        )
        
        # Additional heads for supervised pre-training
        self.clstm_encoder.price_head = nn.Linear(hidden_dim, 1)
        self.clstm_encoder.volatility_head = nn.Linear(hidden_dim, 1)
        self.clstm_encoder.volume_head = nn.Linear(hidden_dim, 1)
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Feature memory for temporal processing
        self.feature_memory = deque(maxlen=20)  # Store last 20 timesteps
        self.memory_initialized = False
    
    def forward(
        self,
        observation: Dict[str, torch.Tensor],
        return_features: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Flatten and concatenate all features
        batch_size = observation['price_history'].shape[0]
        
        # Normalize each feature type before concatenation
        features = []
        
        # Normalize price history (divide by 1000 for typical stock prices)
        price_history = observation['price_history'].flatten(1) / 1000.0
        features.append(price_history)
        
        # Technical indicators are already normalized (-1 to 1 range typically)
        features.append(observation['technical_indicators'])
        
        # Normalize options chain data
        options_chain = observation['options_chain'].flatten(1)
        # Normalize strikes, prices by 1000, keep Greeks as is (already small)
        options_chain = options_chain / 100.0  # Rough normalization
        features.append(options_chain)
        
        # Normalize portfolio state (capital by 100k, positions by 10)
        portfolio = observation['portfolio_state'].clone()
        portfolio[:, 0] = portfolio[:, 0] / 100000.0  # Capital
        portfolio[:, 1] = portfolio[:, 1] / 10.0      # Positions
        portfolio[:, 2] = portfolio[:, 2] / 100000.0  # Portfolio value
        portfolio[:, 4] = portfolio[:, 4] / 1000.0    # Step count
        features.append(portfolio)
        
        # Greeks are already small values, minimal normalization
        greeks = observation['greeks_summary'] / 10.0
        features.append(greeks)
        
        # Add symbol encoding if available
        if self.has_symbol_encoding and 'symbol_encoding' in observation:
            symbol_features = observation['symbol_encoding']
            features.append(symbol_features)
        
        combined_features = torch.cat(features, dim=1)
        
        # Handle feature memory per sample in batch
        # For training, we process the entire batch at once
        device = combined_features.device
        
        if not self.memory_initialized or len(self.feature_memory) == 0:
            # Initialize with zeros on the correct device
            for _ in range(20):
                self.feature_memory.append(torch.zeros_like(combined_features[0:1]).to(device))
            self.memory_initialized = True
        
        # For single inference (batch_size=1), update memory
        if batch_size == 1:
            # Ensure new feature is on the same device
            self.feature_memory.append(combined_features[0:1].to(device))
            # Move all memory tensors to the same device before stacking
            memory_tensors = [tensor.to(device) for tensor in self.feature_memory]
            sequence = torch.stack(memory_tensors, dim=1)
        else:
            # For batch processing during training, create sequence differently
            # Just repeat the current features for simplicity
            sequence = combined_features.unsqueeze(1).repeat(1, 20, 1)
        
        # Encode with CLSTM
        encoded_features = self.clstm_encoder(sequence)
        
        # Add symbol-specific modulation if available
        if self.has_symbol_encoding and 'symbol_encoding' in observation:
            symbol_emb = self.symbol_embedding(observation['symbol_encoding'])
            # Modulate the encoded features with symbol information
            encoded_features = encoded_features + symbol_emb
        
        # Get action logits and value
        action_logits = self.actor(encoded_features)
        value = self.critic(encoded_features)
        
        if return_features:
            return action_logits, value, encoded_features
        else:
            return action_logits, value
    
    def get_action(
        self,
        observation: Dict[str, torch.Tensor],
        deterministic: bool = False
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        action_logits, value = self.forward(observation)
        
        # Create action distribution
        dist = Categorical(logits=action_logits)
        
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            action = dist.sample()
        
        action_log_prob = dist.log_prob(action)
        
        return action.item(), action_log_prob, value


class OptionsCLSTMPPOAgent:
    """Combined CLSTM-PPO agent with both supervised and RL training"""
    
    def __init__(
        self,
        observation_space: Dict,
        action_space: int = 11,
        learning_rate_actor_critic: float = 3e-4,
        learning_rate_clstm: float = 3e-4,  # FIXED: Reduced from 1e-3 for stability
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        batch_size: int = 256,  # FIXED: Increased from 64 for better GPU utilization
        n_epochs: int = 10,
        device: str = None
    ):
        # Auto-detect best available device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device('cpu')
                logger.warning("No GPU detected, using CPU. Training will be slower.")
        else:
            self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        # Create combined CLSTM-PPO network
        self.network = OptionsCLSTMPPONetwork(
            observation_space=observation_space,
            action_dim=action_space
        ).to(self.device)
        
        # Don't use DataParallel here - let the training script handle distributed training
        self.base_network = self.network
        
        # Separate optimizers for different components
        self.ppo_optimizer = optim.Adam(
            list(self.base_network.actor.parameters()) + 
            list(self.base_network.critic.parameters()),
            lr=learning_rate_actor_critic,
            eps=1e-5
        )
        
        self.clstm_optimizer = optim.Adam(
            self.base_network.clstm_encoder.parameters(),
            lr=learning_rate_clstm,
            eps=1e-5
        )
        
        # Experience buffer
        self.buffer = RolloutBuffer()
        
        # Supervised learning buffer for CLSTM
        self.supervised_buffer = SupervisedBuffer(maxlen=10000)
        
        # Training metrics
        self.training_step = 0
        self.episode_rewards = deque(maxlen=100)
        
        logger.info(f"Initialized CLSTM-PPO agent on {self.device}")
    
    def pretrain_clstm(
        self,
        training_data: List[Dict],
        epochs: int = 50,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """Pretrain CLSTM with supervised learning"""
        logger.info(f"Starting CLSTM pretraining for {epochs} epochs")
        
        losses = []
        
        for epoch in range(epochs):
            epoch_losses = []
            
            # Shuffle data
            random.shuffle(training_data)
            
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i + batch_size]
                
                # Prepare batch tensors
                features = torch.stack([d['features'] for d in batch]).to(self.device)
                price_targets = torch.tensor([d['price_target'] for d in batch], dtype=torch.float32).to(self.device)
                vol_targets = torch.tensor([d['volatility_target'] for d in batch], dtype=torch.float32).to(self.device)
                volume_targets = torch.tensor([d['volume_target'] for d in batch], dtype=torch.float32).to(self.device)
                
                # Forward pass
                loss = self.base_network.clstm_encoder.get_supervised_loss(
                    features, price_targets, vol_targets, volume_targets
                )
                
                # Backward pass
                self.clstm_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.base_network.clstm_encoder.parameters(),
                    self.max_grad_norm
                )
                self.clstm_optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Pretraining epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
        
        return {'final_loss': losses[-1], 'all_losses': losses}
    
    def act(self, observation: Dict[str, np.ndarray], deterministic: bool = False) -> Tuple[int, Dict]:
        # Convert observation to tensor
        obs_tensor = self._observation_to_tensor(observation)
        
        with torch.no_grad():
            action, log_prob, value = self.base_network.get_action(obs_tensor, deterministic)
        
        return action, {
            'log_prob': log_prob.cpu().numpy(),
            'value': value.cpu().numpy()
        }
    
    def train(self, winners_only: bool = False) -> Dict[str, float]:
        """Train both CLSTM and PPO components following Algorithm 2

        Args:
            winners_only: If True, train only on winning episodes
        """
        # FIXED: Allow training with smaller batches (episodes are ~200 steps)
        # Require at least 32 samples to have meaningful gradients
        if len(self.buffer) < 32:
            return {}
        
        # Get experiences, optionally filtering for winners
        observations, actions, rewards, values, log_probs, dones = self.buffer.get(winners_only=winners_only)
        
        # If using winners only and no winners available, skip training
        if winners_only and not observations:
            logger.info("No winning episodes available for training")
            return {'skipped': True}
        
        # Compute returns and advantages (Step 11: At = rt + γV(st+1) - V(st))
        returns, advantages = self._compute_gae(rewards, values, dones)
        
        # Convert to tensors
        obs_tensor = self._observations_to_tensor(observations)
        actions_tensor = torch.tensor(np.array(actions), dtype=torch.long).to(self.device)
        old_log_probs_tensor = torch.tensor(np.array(log_probs), dtype=torch.float32).to(self.device)
        returns_tensor = torch.tensor(np.array(returns), dtype=torch.float32).to(self.device)
        advantages_tensor = torch.tensor(np.array(advantages), dtype=torch.float32).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Training metrics
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_clstm_loss = 0
        
        for epoch in range(self.n_epochs):
            # Random permutation for mini-batches
            indices = torch.randperm(len(observations))

            # FIXED: Use smaller batch size if buffer is smaller than configured batch_size
            effective_batch_size = min(self.batch_size, len(observations))

            for start in range(0, len(observations), effective_batch_size):
                end = start + effective_batch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_obs = {k: v[batch_indices] for k, v in obs_tensor.items()}
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                
                # Forward pass - use base_network to avoid DataParallel issues
                action_logits, values, encoded_features = self.base_network.forward(
                    batch_obs, return_features=True
                )
                dist = Categorical(logits=action_logits)
                
                # PPO losses
                new_log_probs = dist.log_prob(batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Ensure values and returns have the same shape
                values_flat = values.squeeze(-1) if values.dim() > 1 else values
                returns_flat = batch_returns.squeeze(-1) if batch_returns.dim() > 1 else batch_returns
                value_loss = F.mse_loss(values_flat, returns_flat)
                entropy = dist.entropy().mean()
                
                # Combined PPO loss
                ppo_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Train PPO (gradients flow through CLSTM encoder)
                self.ppo_optimizer.zero_grad()
                self.clstm_optimizer.zero_grad()  # Also zero CLSTM optimizer

                ppo_loss.backward(retain_graph=True)

                # Clip gradients for both PPO and CLSTM
                torch.nn.utils.clip_grad_norm_(
                    list(self.base_network.actor.parameters()) +
                    list(self.base_network.critic.parameters()),
                    self.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.base_network.clstm_encoder.parameters(),
                    self.max_grad_norm
                )

                self.ppo_optimizer.step()
                self.clstm_optimizer.step()  # Update CLSTM weights

                # Track CLSTM feature reconstruction loss for monitoring
                # This helps ensure CLSTM is learning meaningful representations
                with torch.no_grad():
                    # Simple reconstruction loss: L2 norm of encoded features
                    # Lower is better (more compact representations)
                    feature_loss = (encoded_features ** 2).mean()
                    total_clstm_loss += feature_loss.item()

                # Train CLSTM with auxiliary supervised loss if we have labeled data
                if len(self.supervised_buffer) > 0:
                    sup_batch = self.supervised_buffer.sample(min(32, len(self.supervised_buffer)))
                    
                    sup_features = torch.stack([d['features'] for d in sup_batch]).to(self.device)
                    sup_price_targets = torch.tensor([d['price_target'] for d in sup_batch]).to(self.device)
                    sup_vol_targets = torch.tensor([d['volatility_target'] for d in sup_batch]).to(self.device)
                    sup_volume_targets = torch.tensor([d['volume_target'] for d in sup_batch]).to(self.device)
                    
                    clstm_loss = self.base_network.clstm_encoder.get_supervised_loss(
                        sup_features, sup_price_targets, sup_vol_targets, sup_volume_targets
                    )
                    
                    self.clstm_optimizer.zero_grad()
                    clstm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.base_network.clstm_encoder.parameters(),
                        self.max_grad_norm
                    )
                    self.clstm_optimizer.step()
                    
                    total_clstm_loss += clstm_loss.item()
                
                # Track metrics
                total_loss += ppo_loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        # Clear buffer
        self.buffer.clear()
        
        # Update training step
        self.training_step += 1
        
        # Return training metrics
        # FIXED: Calculate n_updates based on effective batch size
        effective_batch_size = min(self.batch_size, len(observations))
        if effective_batch_size == 0:
            effective_batch_size = 1
        n_updates = self.n_epochs * max(1, (len(observations) // effective_batch_size))
        if n_updates == 0:
            n_updates = 1  # Avoid division by zero

        metrics = {
            'total_loss': total_loss / n_updates,
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'clstm_loss': total_clstm_loss / n_updates,  # FIXED: Always include CLSTM loss
            'training_step': self.training_step
        }

        return metrics
    
    def store_transition(
        self,
        observation: Dict[str, np.ndarray],
        action: int,
        reward: float,
        next_observation: Dict[str, np.ndarray],
        done: bool,
        info: Dict
    ):
        self.buffer.add(
            observation=observation,
            action=action,
            reward=reward,
            value=info['value'],
            log_prob=info['log_prob'],
            done=done
        )
    
    def add_supervised_sample(
        self,
        features: torch.Tensor,
        price_target: float,
        volatility_target: float,
        volume_target: float
    ):
        """Add supervised learning sample for CLSTM training"""
        self.supervised_buffer.add({
            'features': features,
            'price_target': price_target,
            'volatility_target': volatility_target,
            'volume_target': volume_target
        })
    
    def _compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool]
    ) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.array(rewards)
        values = np.array(values)
        dones = np.array(dones)
        
        gae = 0
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            
            returns[t] = gae + values[t]
            advantages[t] = gae
        
        return returns, advantages
    
    def _observation_to_tensor(self, observation: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {
            k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(self.device)
            for k, v in observation.items()
        }
    
    def _observations_to_tensor(self, observations: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        stacked = {}
        for key in observations[0].keys():
            stacked[key] = torch.tensor(
                np.stack([obs[key] for obs in observations]),
                dtype=torch.float32
            ).to(self.device)
        return stacked
    
    def save(self, path: str):
        """Save both CLSTM and PPO models"""
        checkpoint_data = {
            'network_state_dict': self.network.state_dict(),
            'ppo_optimizer_state_dict': self.ppo_optimizer.state_dict(),
            'clstm_optimizer_state_dict': self.clstm_optimizer.state_dict(),
            'training_step': self.training_step,
            'feature_memory': list(self.base_network.feature_memory),
            '_save_metadata': {
                'pytorch_version': torch.__version__,
                'save_timestamp': datetime.now().isoformat(),
                'safe_format': True
            }
        }
        
        try:
            torch.save(checkpoint_data, path)
            logger.info(f"CLSTM-PPO model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save CLSTM-PPO model to {path}: {e}")
    
    def load(self, path: str):
        """Load both CLSTM and PPO models"""
        try:
            # Add safe globals for PyTorch 2.6+ compatibility
            import torch.serialization
            safe_globals = [torch.torch_version.TorchVersion]

            # Try loading with weights_only=True and safe globals
            try:
                with torch.serialization.safe_globals(safe_globals):
                    checkpoint = torch.load(path, map_location=self.device, weights_only=True)
                logger.info(f"Checkpoint loaded securely from {path}")
            except Exception:
                # Fallback to weights_only=False for older checkpoints
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                logger.debug(f"Checkpoint loaded with compatibility mode from {path}")

            # Handle torch.compile wrapper - remove _orig_mod. prefix if present
            state_dict = checkpoint.get('network_state_dict', checkpoint)

            # Check if this is a compiled model checkpoint
            is_compiled_checkpoint = any(k.startswith('_orig_mod.') for k in state_dict.keys())
            is_compiled_model = hasattr(self.network, '_orig_mod')

            # Fix key mismatch between compiled and non-compiled models
            if is_compiled_checkpoint and not is_compiled_model:
                # Checkpoint is compiled, but model is not - remove _orig_mod. prefix
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_key = k.replace('_orig_mod.', '') if k.startswith('_orig_mod.') else k
                    new_state_dict[new_key] = v
                state_dict = new_state_dict
                logger.debug("Removed _orig_mod. prefix from compiled checkpoint")
            elif not is_compiled_checkpoint and is_compiled_model:
                # Checkpoint is not compiled, but model is - add _orig_mod. prefix
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_key = f'_orig_mod.{k}' if not k.startswith('_orig_mod.') else k
                    new_state_dict[new_key] = v
                state_dict = new_state_dict
                logger.debug("Added _orig_mod. prefix for compiled model")

            # Load the model components with fixed state dict
            if isinstance(checkpoint, dict) and 'network_state_dict' in checkpoint:
                # Full checkpoint with optimizer states
                self.network.load_state_dict(state_dict)
                self.ppo_optimizer.load_state_dict(checkpoint['ppo_optimizer_state_dict'])
                self.clstm_optimizer.load_state_dict(checkpoint['clstm_optimizer_state_dict'])
                self.training_step = checkpoint['training_step']
            else:
                # Just state dict (older format)
                self.network.load_state_dict(state_dict)
            
            # Restore feature memory
            if 'feature_memory' in checkpoint:
                self.base_network.feature_memory.clear()
                for feat in checkpoint['feature_memory']:
                    self.base_network.feature_memory.append(feat)
            
            logger.info(f"CLSTM-PPO model successfully loaded from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load CLSTM-PPO model from {path}: {e}")
            raise


class RolloutBuffer:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # Episode tracking for winning episode filtering
        self.current_episode = []
        self.winning_episodes = []
        self.episode_returns = []
        
    def add(self, observation, action, reward, value, log_prob, done):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
        # Track current episode
        self.current_episode.append({
            'observation': observation,
            'action': action,
            'reward': reward,
            'value': value,
            'log_prob': log_prob,
            'done': done
        })
        
        # If episode is done, check if it's a winner
        if done:
            episode_return = sum(item['reward'] for item in self.current_episode)
            self.episode_returns.append(episode_return)
            
            # Store winning episodes separately
            if episode_return > 0:
                self.winning_episodes.append(self.current_episode.copy())
                
            self.current_episode = []
    
    def get(self, winners_only=False):
        """Get experiences, optionally only from winning episodes"""
        if winners_only and self.winning_episodes:
            # Rebuild buffers from winning episodes only
            obs, acts, rews, vals, lps, dns = [], [], [], [], [], []
            for episode in self.winning_episodes:
                for exp in episode:
                    obs.append(exp['observation'])
                    acts.append(exp['action'])
                    rews.append(exp['reward'])
                    vals.append(exp['value'])
                    lps.append(exp['log_prob'])
                    dns.append(exp['done'])
            return obs, acts, rews, vals, lps, dns
        else:
            return (
                self.observations,
                self.actions,
                self.rewards,
                self.values,
                self.log_probs,
                self.dones
            )
    
    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        # Keep winning episodes for future training
        if len(self.winning_episodes) > 100:  # Limit memory usage
            self.winning_episodes = self.winning_episodes[-100:]
    
    def __len__(self):
        return len(self.observations)
    
    def get_win_rate(self):
        """Calculate win rate from episode returns"""
        if not self.episode_returns:
            return 0.0
        wins = sum(1 for r in self.episode_returns if r > 0)
        return wins / len(self.episode_returns)


class SupervisedBuffer:
    def __init__(self, maxlen: int = 10000):
        self.buffer = deque(maxlen=maxlen)
    
    def add(self, sample: Dict):
        self.buffer.append(sample)
    
    def sample(self, batch_size: int) -> List[Dict]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)