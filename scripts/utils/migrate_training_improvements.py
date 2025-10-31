#!/usr/bin/env python3
"""
Script to migrate improvements from v2 to the original training script
This allows for selective application of improvements
"""

import os
import shutil
import argparse
import re


def apply_improvement_1_reward_structure(content):
    """Apply simplified reward structure"""
    print("Applying Improvement 1: Simplified reward structure...")
    
    # Replace the exponential reward function with simplified version
    old_pattern = r'def _calculate_exponential_reward\(self, pnl, pnl_pct, base_reward=1\.0\):.*?return reward'
    new_reward = '''def _calculate_trade_reward(self, pnl, pnl_pct, base_reward=1.0):
        """
        Simplified reward structure focused on win rate and profitability
        """
        if pnl > 0:
            # Reward wins proportionally
            reward = 10 * pnl_pct  # 5% win = 0.5 reward
        else:
            # Smaller penalty for losses to encourage learning
            reward = 5 * pnl_pct   # 5% loss = -0.25 reward
        
        return reward * base_reward'''
    
    content = re.sub(old_pattern, new_reward, content, flags=re.DOTALL)
    
    # Add win rate bonus function
    win_rate_bonus = '''
    def _calculate_win_rate_bonus(self, episode_num):
        """Calculate bonus based on current win rate"""
        if self.winning_trades + self.losing_trades < 5:
            return 0  # Not enough trades
            
        win_rate = self.winning_trades / (self.winning_trades + self.losing_trades)
        
        # Progressive win rate bonuses
        if episode_num % 100 == 0:
            if win_rate > 0.6:
                return 50
            elif win_rate > 0.5:
                return 20
        
        # Regular bonuses
        if win_rate > 0.7:
            return 15
        elif win_rate > 0.6:
            return 8
        elif win_rate > 0.5:
            return 4
        elif win_rate < 0.3:
            return -5  # Penalty for poor win rate
        
        return 0'''
    
    # Add after the reward function
    content = content.replace('return reward * base_reward', 
                             f'return reward * base_reward\n{win_rate_bonus}')
    
    # Update all references to _calculate_exponential_reward
    content = content.replace('_calculate_exponential_reward', '_calculate_trade_reward')
    
    return content


def apply_improvement_2_technical_indicators(content):
    """Properly integrate technical indicators into observation space"""
    print("Applying Improvement 2: Technical indicators integration...")
    
    # Add enhanced observation method
    enhanced_obs_method = '''
    def _get_enhanced_observation(self, base_obs):
        """Enhanced observation with properly integrated technical indicators"""
        if base_obs is None:
            return None
        
        # Calculate technical indicators using price history
        if len(self.price_history) >= 26:  # Minimum for MACD
            prices = self.price_history
            
            # Calculate all indicators
            macd_line, signal_line, macd_hist = TechnicalIndicators.calculate_macd(prices)
            rsi = TechnicalIndicators.calculate_rsi(prices)
            cci = TechnicalIndicators.calculate_cci(prices)
            adx = TechnicalIndicators.calculate_adx(prices)
            
            # Additional calculations
            volatility = self._calculate_volatility()
            momentum = self._calculate_momentum()
            
            # Market regime
            regime = self._detect_market_regime()
            regime_encoding = {"volatile": 0, "trending": 1, "ranging": 2, "mixed": 3}
            regime_value = regime_encoding.get(regime, 3)
            
            # Win rate
            win_rate = self.winning_trades / max(1, self.winning_trades + self.losing_trades)
            
            # Fill technical indicators array
            technical_features = np.array([
                macd_hist / 10.0,  # Normalized MACD histogram
                (rsi - 50) / 50.0,  # Normalized RSI (-1 to 1)
                cci / 200.0,  # Normalized CCI
                adx / 50.0,  # Normalized ADX
                volatility * 100,  # Volatility percentage
                momentum * 100,  # Momentum percentage
                regime_value / 3.0,  # Normalized regime
                win_rate,  # Current win rate
                len(self.positions) / float(self.max_positions),  # Position utilization
                self.consecutive_losses / 10.0,  # Normalized consecutive losses
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Padding to 20
            ], dtype=np.float32)
            
            base_obs['technical_indicators'] = technical_features[:20]
        
        return base_obs'''
    
    # Find a good place to add this method (after _detect_market_regime)
    market_regime_end = content.find('return regime\n\n')
    if market_regime_end != -1:
        insert_pos = market_regime_end + len('return regime\n\n')
        content = content[:insert_pos] + enhanced_obs_method + '\n' + content[insert_pos:]
    
    return content


def apply_improvement_3_position_sizing(content):
    """Add dynamic position sizing"""
    print("Applying Improvement 3: Dynamic position sizing...")
    
    position_sizing_method = '''
    def _calculate_dynamic_position_size(self, confidence, volatility, win_rate):
        """Implement dynamic position sizing based on multiple factors"""
        base_size = 0.15  # 15% base position
        
        # Adjust for confidence (0.5x to 1.5x)
        size = base_size * (0.5 + confidence)
        
        # Adjust for volatility (inverse relationship)
        volatility_factor = 1.0 - min(0.5, volatility * 10)
        size *= volatility_factor
        
        # Bonus for high win rate
        if win_rate > 0.6:
            size *= 1.2
        elif win_rate < 0.4 and self.winning_trades + self.losing_trades > 10:
            size *= 0.8
        
        return min(0.3, max(0.05, size))  # 5-30% of capital'''
    
    # Add after win rate bonus function
    win_rate_pos = content.find('return 0\n\n\nclass')
    if win_rate_pos != -1:
        insert_pos = win_rate_pos + len('return 0')
        content = content[:insert_pos] + '\n' + position_sizing_method + '\n' + content[insert_pos:]
    
    return content


def apply_improvement_4_smart_exits(content):
    """Add smart exit logic"""
    print("Applying Improvement 4: Smart exit logic...")
    
    smart_exit_method = '''
    def _should_exit_position(self, position, current_price, market_regime):
        """Smart exit logic based on market regime and position metrics"""
        pnl = self._calculate_position_pnl(position, current_price)
        pnl_pct = pnl / (position['entry_price'] * position['quantity'] * 100)
        holding_time = self.current_step - position['entry_step']
        
        # Dynamic exit thresholds based on market regime
        if market_regime == "volatile":
            take_profit = 0.03  # 3% in volatile markets
            stop_loss = -0.02   # 2% stop loss
            max_holding = 20    # Shorter holding period
        elif market_regime == "trending":
            momentum = self._calculate_momentum()
            if (position['option_type'] == 'call' and momentum > 0) or \\
               (position['option_type'] == 'put' and momentum < 0):
                take_profit = 0.08  # 8% for trend-following positions
                stop_loss = -0.03   # 3% stop loss
                max_holding = 40    # Longer holding period
            else:
                take_profit = 0.02  # Quick exit for counter-trend
                stop_loss = -0.015
                max_holding = 10
        else:  # ranging
            take_profit = 0.04  # 4% in ranging markets
            stop_loss = -0.02   # 2% stop loss
            max_holding = 25
        
        # Exit conditions
        if pnl_pct >= take_profit:
            return True, "take_profit"
        elif pnl_pct <= stop_loss:
            return True, "stop_loss"
        elif holding_time > max_holding:
            return True, "time_exit"
        
        return False, None'''
    
    # Add after position sizing method
    sizing_pos = content.find('# 5-30% of capital')
    if sizing_pos != -1:
        insert_pos = content.find('\n\n', sizing_pos) + 2
        content = content[:insert_pos] + smart_exit_method + '\n' + content[insert_pos:]
    
    return content


def apply_improvement_5_hyperparameters(content):
    """Update training hyperparameters"""
    print("Applying Improvement 5: Updated hyperparameters...")
    
    # Update learning rates
    content = re.sub(r'learning_rate_actor_critic\s*=\s*[\d.e-]+', 
                     'learning_rate_actor_critic = 1e-4', content)
    content = re.sub(r'learning_rate_clstm\s*=\s*[\d.e-]+', 
                     'learning_rate_clstm = 3e-4', content)
    
    # Update PPO epochs
    content = re.sub(r'ppo_epochs\s*=\s*\d+', 'ppo_epochs = 4', content)
    
    # Update batch size
    content = re.sub(r'batch_size\s*=\s*\d+', 'batch_size = 4096', content)
    
    return content


def apply_improvement_6_trade_metrics(content):
    """Add trade quality metrics tracking"""
    print("Applying Improvement 6: Trade quality metrics...")
    
    # Add metrics initialization in __init__
    metrics_init = '''
        # Trade quality metrics
        self.trade_metrics = {
            'avg_win_size': deque(maxlen=100),
            'avg_loss_size': deque(maxlen=100),
            'profit_factors': deque(maxlen=100),
            'max_drawdowns': deque(maxlen=100),
            'sharpe_ratios': deque(maxlen=100),
            'win_rates': deque(maxlen=100)
        }'''
    
    # Find __init__ method and add metrics
    init_pos = content.find('self.peak_capital = self.initial_capital')
    if init_pos != -1:
        insert_pos = init_pos + len('self.peak_capital = self.initial_capital')
        content = content[:insert_pos] + '\n' + metrics_init + content[insert_pos:]
    
    return content


def apply_all_improvements(input_file, output_file):
    """Apply all improvements to the training script"""
    print(f"Reading {input_file}...")
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Apply improvements in order
    content = apply_improvement_1_reward_structure(content)
    content = apply_improvement_2_technical_indicators(content)
    content = apply_improvement_3_position_sizing(content)
    content = apply_improvement_4_smart_exits(content)
    content = apply_improvement_5_hyperparameters(content)
    content = apply_improvement_6_trade_metrics(content)
    
    # Write output
    print(f"Writing improved script to {output_file}...")
    with open(output_file, 'w') as f:
        f.write(content)
    
    print("Migration complete!")


def main():
    parser = argparse.ArgumentParser(description='Migrate training script improvements')
    parser.add_argument('--input', default='train_profitable_optimized.py',
                       help='Input training script')
    parser.add_argument('--output', default='train_profitable_optimized_improved.py',
                       help='Output training script')
    parser.add_argument('--backup', action='store_true',
                       help='Create backup of original file')
    
    args = parser.parse_args()
    
    # Create backup if requested
    if args.backup:
        backup_file = args.input + '.backup'
        shutil.copy2(args.input, backup_file)
        print(f"Created backup: {backup_file}")
    
    # Apply improvements
    apply_all_improvements(args.input, args.output)


if __name__ == '__main__':
    main()