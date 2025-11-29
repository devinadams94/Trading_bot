#!/usr/bin/env python3
"""
Online Learning Loop for Paper Trading
Continuously improves model from real market feedback
"""

import asyncio
import logging
import torch
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import deque
import json
from pathlib import Path

from src.envs.paper_trading_env import PaperTradingEnvironment
from src.models.ppo_agent import OptionsCLSTMPPOAgent

logger = logging.getLogger(__name__)


class ExperienceBuffer:
    """Buffer for storing paper trading experiences"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        
    def add(self, observation, action, reward, next_observation, done, info):
        """Add experience to buffer"""
        self.buffer.append({
            'observation': observation,
            'action': action,
            'reward': reward,
            'next_observation': next_observation,
            'done': done,
            'info': info,
            'timestamp': datetime.utcnow()
        })
    
    def sample(self, batch_size: int) -> List[Dict]:
        """Sample random batch from buffer"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def get_recent(self, n: int) -> List[Dict]:
        """Get n most recent experiences"""
        return list(self.buffer)[-n:]
    
    def __len__(self):
        return len(self.buffer)
    
    def save(self, filepath: str):
        """Save buffer to disk"""
        data = {
            'experiences': [
                {
                    'observation': exp['observation'].tolist() if isinstance(exp['observation'], np.ndarray) else exp['observation'],
                    'action': int(exp['action']),
                    'reward': float(exp['reward']),
                    'next_observation': exp['next_observation'].tolist() if isinstance(exp['next_observation'], np.ndarray) else exp['next_observation'],
                    'done': bool(exp['done']),
                    'info': exp['info'],
                    'timestamp': exp['timestamp'].isoformat()
                }
                for exp in self.buffer
            ]
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"💾 Saved {len(self.buffer)} experiences to {filepath}")


class OnlineLearningLoop:
    """
    Continuous learning from paper trading
    
    Features:
    - Real-time decision making
    - Experience collection from live markets
    - Periodic model updates
    - Performance tracking
    - Automatic checkpointing
    """
    
    def __init__(self, 
                 api_key: str,
                 symbols: List[str],
                 model_path: Optional[str] = None,
                 initial_capital: float = 100000,
                 update_frequency: int = 100,  # Update model every N steps
                 save_frequency: int = 1000,   # Save checkpoint every N steps
                 checkpoint_dir: str = 'checkpoints/paper_trading'):
        
        self.api_key = api_key
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.update_frequency = update_frequency
        self.save_frequency = save_frequency
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Paper trading environment
        self.env = PaperTradingEnvironment(
            api_key=api_key,
            symbols=symbols,
            initial_capital=initial_capital
        )
        
        # RL Agent
        if model_path:
            logger.info(f"📦 Loading pretrained model from {model_path}")
            self.agent = self._load_agent(model_path)
        else:
            logger.warning("⚠️ No pretrained model provided, starting from scratch")
            self.agent = self._create_new_agent()
        
        # Experience buffer
        self.experience_buffer = ExperienceBuffer(max_size=10000)
        
        # Metrics tracking
        self.step_count = 0
        self.episode_count = 0
        self.total_reward = 0
        self.episode_rewards = []
        self.portfolio_values = []
        self.win_rates = []
        
        # Running state
        self.running = False
        
    def _load_agent(self, model_path: str) -> OptionsCLSTMPPOAgent:
        """Load pretrained agent"""
        # TODO: Implement proper model loading
        # For now, create new agent and load weights
        agent = self._create_new_agent()
        
        try:
            checkpoint = torch.load(model_path)
            agent.base_network.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"✅ Loaded model weights from {model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            logger.warning("⚠️ Starting with random weights")
        
        return agent
    
    def _create_new_agent(self) -> OptionsCLSTMPPOAgent:
        """Create new agent from scratch"""
        # Get observation space size from environment
        obs_size = len(self.env.get_observation())
        action_size = self.env.action_space_n
        
        agent = OptionsCLSTMPPOAgent(
            observation_size=obs_size,
            action_size=action_size,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        return agent

    async def run(self, duration_hours: Optional[float] = None, max_steps: Optional[int] = None):
        """
        Run paper trading with online learning

        Args:
            duration_hours: Run for this many hours (None = run indefinitely)
            max_steps: Maximum number of steps (None = no limit)
        """
        logger.info("🚀 Starting online learning loop...")
        logger.info(f"   Symbols: {self.symbols}")
        logger.info(f"   Initial capital: ${self.initial_capital:,.2f}")
        logger.info(f"   Update frequency: every {self.update_frequency} steps")
        logger.info(f"   Save frequency: every {self.save_frequency} steps")

        # Initialize environment
        await self.env.initialize()

        # Get initial observation
        obs = self.env.get_observation()

        # Set running flag
        self.running = True
        start_time = datetime.utcnow()

        try:
            while self.running:
                # Check duration limit
                if duration_hours and (datetime.utcnow() - start_time).total_seconds() > duration_hours * 3600:
                    logger.info(f"⏰ Duration limit reached ({duration_hours} hours)")
                    break

                # Check step limit
                if max_steps and self.step_count >= max_steps:
                    logger.info(f"🎯 Step limit reached ({max_steps} steps)")
                    break

                # Agent selects action
                action, log_prob, value = self.agent.select_action(
                    torch.FloatTensor(obs).unsqueeze(0).to(self.agent.device)
                )
                action = action.item()

                # Execute action in environment
                next_obs, reward, done, info = await self.env.step(action)

                # Store experience
                self.experience_buffer.add(obs, action, reward, next_obs, done, info)

                # Update metrics
                self.step_count += 1
                self.total_reward += reward

                # Log progress
                if self.step_count % 10 == 0:
                    portfolio_value = info['portfolio_value']
                    self.portfolio_values.append(portfolio_value)

                    logger.info(
                        f"Step {self.step_count}: "
                        f"Action={action}, "
                        f"Reward={reward:.4f}, "
                        f"Portfolio=${portfolio_value:,.2f}, "
                        f"Positions={info['num_positions']}"
                    )

                # Train agent periodically
                if self.step_count % self.update_frequency == 0 and len(self.experience_buffer) >= 32:
                    await self._train_agent()

                # Save checkpoint periodically
                if self.step_count % self.save_frequency == 0:
                    self._save_checkpoint()

                # Update observation
                obs = next_obs

                # Small delay to avoid overwhelming the API
                await asyncio.sleep(1)  # 1 second between actions

        except KeyboardInterrupt:
            logger.info("⚠️ Interrupted by user")
        except Exception as e:
            logger.error(f"❌ Error in online learning loop: {e}", exc_info=True)
        finally:
            # Cleanup
            await self.env.close()
            self._save_checkpoint(final=True)
            self._save_metrics()
            logger.info("🛑 Online learning loop stopped")

    async def _train_agent(self):
        """Train agent on recent experiences"""
        logger.info(f"🎓 Training agent (buffer size: {len(self.experience_buffer)})")

        # Sample batch from experience buffer
        batch = self.experience_buffer.sample(min(128, len(self.experience_buffer)))

        # Prepare batch for training
        observations = torch.FloatTensor([exp['observation'] for exp in batch]).to(self.agent.device)
        actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.agent.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.agent.device)
        next_observations = torch.FloatTensor([exp['next_observation'] for exp in batch]).to(self.agent.device)
        dones = torch.FloatTensor([exp['done'] for exp in batch]).to(self.agent.device)

        # Add to agent's buffer and train
        for i in range(len(batch)):
            # Get value and log_prob for the observation
            with torch.no_grad():
                _, log_prob, value = self.agent.select_action(observations[i].unsqueeze(0))

            self.agent.buffer.add(
                observation=observations[i].cpu().numpy(),
                action=actions[i].item(),
                reward=rewards[i].item(),
                value=value.item() if isinstance(value, torch.Tensor) else value,
                log_prob=log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob,
                done=dones[i].item()
            )

        # Train
        train_result = self.agent.train()

        if train_result:
            logger.info(
                f"   Loss: {train_result.get('total_loss', 0):.4f}, "
                f"   Actor: {train_result.get('actor_loss', 0):.4f}, "
                f"   Critic: {train_result.get('critic_loss', 0):.4f}"
            )

    def _save_checkpoint(self, final: bool = False):
        """Save model checkpoint"""
        suffix = 'final' if final else f'step_{self.step_count}'
        checkpoint_path = self.checkpoint_dir / f'online_learning_{suffix}.pt'

        torch.save({
            'step': self.step_count,
            'model_state_dict': self.agent.base_network.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'total_reward': self.total_reward,
            'portfolio_values': self.portfolio_values[-100:],  # Last 100 values
        }, checkpoint_path)

        logger.info(f"💾 Saved checkpoint to {checkpoint_path}")

    def _save_metrics(self):
        """Save performance metrics"""
        metrics_path = self.checkpoint_dir / f'metrics_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.json'

        portfolio_metrics = self.env.portfolio.get_metrics(
            self.env._extract_current_prices(self.env.data_stream.get_current_state())
        )

        metrics = {
            'step_count': self.step_count,
            'total_reward': self.total_reward,
            'portfolio_metrics': portfolio_metrics,
            'portfolio_values': self.portfolio_values,
            'trade_history': [
                {k: str(v) if isinstance(v, datetime) else v for k, v in trade.items()}
                for trade in self.env.portfolio.trade_history
            ]
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"📊 Saved metrics to {metrics_path}")

        # Also save experience buffer
        buffer_path = self.checkpoint_dir / f'experiences_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.json'
        self.experience_buffer.save(str(buffer_path))

    def stop(self):
        """Stop the online learning loop"""
        self.running = False

