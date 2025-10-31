#!/usr/bin/env python3
"""
Script to simplify the reward mechanism in train.py to focus on trade returns
"""

import re
from pathlib import Path


def create_simplified_step_method():
    """Create a simplified step method that focuses on trade returns"""
    
    simplified_code = '''
    def step(self, action: int):
        """Simplified step focusing on trade returns"""
        if self.done or self.current_step >= self._data_length:
            self.done = True
            return self._get_observation(), 0, True, {}
        
        # Action mapping
        actions = ['hold', 'buy_call', 'buy_put', 'sell_call', 'sell_put', 
                  'bull_call_spread', 'bear_put_spread', 'iron_condor', 
                  'straddle', 'strangle', 'close_all_positions']
        action_name = actions[action] if action < len(actions) else 'hold'
        
        # Track current price
        current_price = self._prices[self.current_step]
        self.price_history.append(current_price)
        if len(self.price_history) > self.volatility_window:
            self.price_history.pop(0)
        
        # Simple reward based on trade returns
        reward = 0
        
        # Execute action
        if action_name == 'hold':
            # Small penalty for holding when no positions (encourage trading)
            if len(self.positions) == 0:
                reward = -0.01
        elif action_name == 'close_all_positions' and self.positions:
            # Close all positions and get the total P&L as reward
            total_pnl = self._close_all_positions_realistic()
            # Scale P&L to reasonable reward range
            reward = total_pnl / 1000.0  # Divide by 1000 to normalize
        elif action_name in ['buy_call', 'buy_put'] and len(self.positions) < self.max_positions:
            # Execute trade (small penalty for transaction cost)
            trade_reward = self._execute_trade(action_name, current_price)
            reward = trade_reward if trade_reward < 0 else -0.01  # Transaction cost
        
        # Update positions and check for exits
        closed_pnl = self._update_positions_realistic()
        
        # Add P&L from any positions that were closed this step
        if closed_pnl != 0:
            reward += closed_pnl / 1000.0  # Normalize by dividing by 1000
        
        # Update step
        self.current_step += 1
        
        # Done conditions
        portfolio_value = self._calculate_portfolio_value_fast()
        if self.current_step >= self._data_length - 1:
            self.done = True
        elif portfolio_value < self.initial_capital * 0.2:  # 80% loss
            self.done = True
            reward -= 1.0  # Additional penalty for blowing up the account
        
        return self._get_observation(), reward, self.done, {
            'portfolio_value': portfolio_value,
            'positions': len(self.positions),
            'symbol': self.current_symbol,
            'win_rate': self.winning_trades / max(1, self.winning_trades + self.losing_trades) if hasattr(self, 'winning_trades') else 0
        }
'''
    return simplified_code


def create_simplified_update_positions():
    """Create a simplified position update method that returns P&L"""
    
    simplified_code = '''
    def _update_positions_realistic(self):
        """Update positions and return total P&L from closed positions"""
        if self.current_step >= self._data_length:
            return 0
        
        current_time = self._timestamps[self.current_step]
        current_price = self._prices[self.current_step]
        positions_to_close = []
        total_closed_pnl = 0
        
        for i, pos in enumerate(self.positions):
            # Calculate position age
            position_age = self.current_step - pos['entry_step']
            entry_underlying = pos['entry_underlying']
            
            # Calculate price movement
            price_change = (current_price - entry_underlying) / entry_underlying
            
            # Simple option P&L calculation
            if pos['option_type'] == 'call':
                option_price_change = price_change * 1.5 if price_change > 0 else price_change * 0.8
            else:  # put
                option_price_change = -price_change * 1.5 if price_change < 0 else -price_change * 0.8
            
            # Calculate P&L
            entry_cost = pos['entry_price'] * pos['quantity'] * 100
            current_value = pos['entry_price'] * (1 + option_price_change) * pos['quantity'] * 100
            current_value = max(0, current_value)
            
            pnl = current_value - entry_cost - self.commission
            pnl_pct = pnl / entry_cost
            
            # Simple exit rules
            should_exit = False
            
            # Stop loss at -20%
            if pnl_pct <= -0.20:
                should_exit = True
                if hasattr(self, 'losing_trades'):
                    self.losing_trades += 1
            
            # Take profit at +30%
            elif pnl_pct >= 0.30:
                should_exit = True
                if hasattr(self, 'winning_trades'):
                    self.winning_trades += 1
            
            # Time exit after 20 steps
            elif position_age > 20:
                should_exit = True
                if pnl > 0 and hasattr(self, 'winning_trades'):
                    self.winning_trades += 1
                elif hasattr(self, 'losing_trades'):
                    self.losing_trades += 1
            
            if should_exit:
                positions_to_close.append(i)
                self.capital += current_value - self.commission
                total_closed_pnl += pnl
                if hasattr(self, 'total_pnl'):
                    self.total_pnl += pnl
        
        # Remove closed positions
        for i in reversed(positions_to_close):
            self.positions.pop(i)
        
        return total_closed_pnl
'''
    return simplified_code


def create_simplified_close_all_positions():
    """Create a simplified close all positions method"""
    
    simplified_code = '''
    def _close_all_positions_realistic(self):
        """Close all positions and return total P&L"""
        total_pnl = 0
        current_price = self._prices[self.current_step]
        
        for pos in self.positions:
            # Calculate P&L
            entry_underlying = pos['entry_underlying']
            price_change = (current_price - entry_underlying) / entry_underlying
            
            if pos['option_type'] == 'call':
                option_price_change = price_change * 1.5 if price_change > 0 else price_change * 0.8
            else:  # put
                option_price_change = -price_change * 1.5 if price_change < 0 else -price_change * 0.8
            
            # Calculate final value
            entry_cost = pos['entry_price'] * pos['quantity'] * 100
            current_value = pos['entry_price'] * (1 + option_price_change) * pos['quantity'] * 100
            current_value = max(0, current_value)
            
            pnl = current_value - entry_cost - self.commission
            
            self.capital += current_value - self.commission
            total_pnl += pnl
            
            if hasattr(self, 'total_pnl'):
                self.total_pnl += pnl
            
            if pnl > 0 and hasattr(self, 'winning_trades'):
                self.winning_trades += 1
            elif hasattr(self, 'losing_trades'):
                self.losing_trades += 1
        
        self.positions = []
        return total_pnl
'''
    return simplified_code


def create_simplified_execute_trade():
    """Create a simplified execute trade method"""
    
    simplified_code = '''
    def _execute_trade(self, action_name, current_price):
        """Execute trade with minimal validation"""
        current_time = self._timestamps[self.current_step]
        
        # Simple momentum check for basic filtering
        if len(self.price_history) >= 5:
            momentum = (self.price_history[-1] - self.price_history[-5]) / self.price_history[-5]
            
            # Basic directional filter
            if action_name == 'buy_call' and momentum < -0.02:
                return -0.01  # Small penalty for buying calls in downtrend
            elif action_name == 'buy_put' and momentum > 0.02:
                return -0.01  # Small penalty for buying puts in uptrend
        
        # Find suitable option
        cache_key = (self.current_step, action_name)
        if cache_key in self._cache:
            option = self._cache[cache_key]
        else:
            # Find options at current timestamp
            time_mask = self._option_data['timestamps'] == current_time
            type_mask = self._option_data['types'] == ('call' if 'call' in action_name else 'put')
            money_mask = (self._option_data['moneyness'] >= 0.95) & (self._option_data['moneyness'] <= 1.05)
            bid_mask = self._option_data['bids'] > 0
            
            valid_mask = time_mask & type_mask & money_mask & bid_mask
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) > 0:
                # Pick the one with highest volume
                volumes = self._option_data['volumes'][valid_indices]
                best_idx = valid_indices[np.argmax(volumes)]
                
                option = {
                    'strike': self._option_data['strikes'][best_idx],
                    'bid': self._option_data['bids'][best_idx],
                    'ask': self._option_data['asks'][best_idx],
                    'volume': self._option_data['volumes'][best_idx]
                }
                self._cache[cache_key] = option
            else:
                self._cache[cache_key] = None
                return 0
        
        option = self._cache[cache_key]
        
        if option is not None:
            # Calculate position size (use 10% of capital per trade)
            mid_price = (option['bid'] + option['ask']) / 2
            position_size = min(10, int(self.capital * 0.1 / (mid_price * 100)))
            
            if position_size > 0:
                cost = position_size * option['ask'] * 100 + self.commission
                if cost <= self.capital * 0.5:  # Don't use more than 50% of capital
                    self.positions.append({
                        'entry_price': option['ask'],
                        'quantity': position_size,
                        'entry_step': self.current_step,
                        'entry_underlying': current_price,
                        'option_type': 'call' if 'call' in action_name else 'put',
                        'strike': option['strike']
                    })
                    self.capital -= cost
                    return 0  # No immediate reward for opening position
        
        return 0
'''
    return simplified_code


def main():
    """Main function to update train.py with simplified rewards"""
    
    print("📝 Simplifying reward mechanism in train.py...")
    
    # Read the current train.py
    train_path = Path('train.py')
    if not train_path.exists():
        print("❌ train.py not found!")
        return
    
    with open(train_path, 'r') as f:
        content = f.read()
    
    # Create backup
    backup_path = Path('train_backup_complex_rewards.py')
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"✅ Created backup at {backup_path}")
    
    # Find the BalancedEnvironment class
    class_match = re.search(r'class BalancedEnvironment\(HistoricalOptionsEnvironment\):', content)
    if not class_match:
        print("❌ Could not find BalancedEnvironment class")
        return
    
    # Generate replacement methods
    new_step = create_simplified_step_method()
    new_update = create_simplified_update_positions()
    new_close = create_simplified_close_all_positions()
    new_execute = create_simplified_execute_trade()
    
    # Find and replace each method
    replacements = [
        (r'def step\(self, action: int\):.*?(?=\n    def \w|\n\nclass|\Z)', new_step),
        (r'def _update_positions_realistic\(self\):.*?(?=\n    def \w|\n\nclass|\Z)', new_update),
        (r'def _close_all_positions_realistic\(self\):.*?(?=\n    def \w|\n\nclass|\Z)', new_close),
        (r'def _execute_trade\(self, action_name, current_price\):.*?(?=\n    def \w|\n\nclass|\Z)', new_execute)
    ]
    
    # Apply replacements
    modified_content = content
    for pattern, replacement in replacements:
        match = re.search(pattern, modified_content, re.DOTALL)
        if match:
            # Preserve indentation
            indent = '    '  # 4 spaces for class methods
            indented_replacement = '\n'.join(indent + line if line else line 
                                           for line in replacement.strip().split('\n'))
            modified_content = modified_content[:match.start()] + '\n' + indented_replacement + modified_content[match.end():]
            print(f"✅ Replaced method: {replacement.split('(')[0].strip().split()[-1]}")
        else:
            print(f"⚠️  Could not find method to replace: {pattern[:30]}...")
    
    # Write the modified content
    with open(train_path, 'w') as f:
        f.write(modified_content)
    
    print("\n✅ Successfully simplified reward mechanism!")
    print("\n📋 Summary of changes:")
    print("1. Removed complex reward components (win rate bonus, Sharpe ratio, etc.)")
    print("2. Reward is now primarily based on realized P&L from trades")
    print("3. Small penalties for holding with no positions and transaction costs")
    print("4. Simple stop loss (-20%) and take profit (+30%) rules")
    print("5. Time-based exit after 20 steps")
    print("\n💡 The model will now learn primarily from trade profitability!")


if __name__ == "__main__":
    main()