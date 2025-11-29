"""Experience buffers for RL training"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import random
import logging

logger = logging.getLogger(__name__)


class RolloutBuffer:
    """Buffer for storing PPO rollout data"""
    
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.episode_returns = []
        self._current_episode_return = 0.0
    
    def add(
        self,
        observation: Dict[str, np.ndarray],
        action: int,
        reward: float,
        value: np.ndarray,
        log_prob: np.ndarray,
        done: bool
    ):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value.item() if hasattr(value, 'item') else float(value))
        self.log_probs.append(log_prob.item() if hasattr(log_prob, 'item') else float(log_prob))
        self.dones.append(done)
        
        self._current_episode_return += reward
        if done:
            self.episode_returns.append(self._current_episode_return)
            self._current_episode_return = 0.0
    
    def get(self, winners_only: bool = False) -> Tuple[List, List, List, List, List, List]:
        if not winners_only:
            return (
                self.observations,
                self.actions,
                self.rewards,
                self.values,
                self.log_probs,
                self.dones
            )
        
        # Filter to only winning episodes
        if not self.episode_returns:
            return [], [], [], [], [], []
        
        winning_threshold = np.percentile(self.episode_returns, 50)
        
        filtered_obs = []
        filtered_actions = []
        filtered_rewards = []
        filtered_values = []
        filtered_log_probs = []
        filtered_dones = []
        
        episode_start = 0
        episode_idx = 0
        
        for i, done in enumerate(self.dones):
            if done:
                if episode_idx < len(self.episode_returns) and self.episode_returns[episode_idx] >= winning_threshold:
                    filtered_obs.extend(self.observations[episode_start:i+1])
                    filtered_actions.extend(self.actions[episode_start:i+1])
                    filtered_rewards.extend(self.rewards[episode_start:i+1])
                    filtered_values.extend(self.values[episode_start:i+1])
                    filtered_log_probs.extend(self.log_probs[episode_start:i+1])
                    filtered_dones.extend(self.dones[episode_start:i+1])
                episode_start = i + 1
                episode_idx += 1
        
        return (
            filtered_obs,
            filtered_actions,
            filtered_rewards,
            filtered_values,
            filtered_log_probs,
            filtered_dones
        )
    
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.episode_returns = []
        self._current_episode_return = 0.0
    
    def __len__(self):
        return len(self.observations)


class SupervisedBuffer:
    """Buffer for supervised learning samples"""
    
    def __init__(self, maxlen: int = 10000):
        self.buffer = deque(maxlen=maxlen)
    
    def add(self, sample: Dict):
        self.buffer.append(sample)
    
    def sample(self, batch_size: int) -> List[Dict]:
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class ExperienceBuffer:
    """Experience buffer for online learning"""
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience: Dict):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Dict]:
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
    
    def get_recent(self, n: int) -> List[Dict]:
        return list(self.buffer)[-n:]
    
    def clear(self):
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)

