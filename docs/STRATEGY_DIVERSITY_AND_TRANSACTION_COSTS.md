# Strategy Diversity & Realistic Transaction Costs Implementation Plan

**Date:** November 2, 2025  
**Status:** Planning Phase  
**Priority:** HIGH - Critical for realistic training

---

## 🎯 **Executive Summary**

This document outlines improvements to:
1. **Strategy Diversity** - Expand from simple buy/sell to multi-leg strategies
2. **Transaction Costs** - Replace fixed commissions with realistic bid-ask spreads and regulatory fees

**Key Finding:** Alpaca Markets offers **commission-free** options trading, but real costs come from:
- Bid-ask spreads (2-10% of option price)
- Regulatory fees (SEC, FINRA, OCC)
- Slippage on market orders

---

## 📊 **Current State Analysis**

### **Current Action Space (31 actions)**

<augment_code_snippet path="src/working_options_env.py" mode="EXCERPT">
````python
# Action space: 0=hold, 1-10=buy calls, 11-20=buy puts, 21-30=sell positions
self.action_space = spaces.Discrete(31)
````
</augment_code_snippet>

**Limitations:**
- ❌ Only single-leg strategies (buy call, buy put)
- ❌ No spreads, straddles, or iron condors
- ❌ No short selling (covered calls, cash-secured puts)
- ❌ Limited strike price selection (10 strikes per type)

### **Current Transaction Costs**

<augment_code_snippet path="src/working_options_env.py" mode="EXCERPT">
````python
commission: float = 0.65,  # Fixed commission per contract
````
</augment_code_snippet>

**Limitations:**
- ❌ Fixed $0.65 commission (outdated - Alpaca is commission-free)
- ❌ No bid-ask spread modeling
- ❌ No regulatory fees (SEC, FINRA, OCC)
- ❌ No slippage modeling
- ❌ Doesn't reflect real market conditions

---

## 💰 **Alpaca Transaction Cost Structure**

### **Commission-Free Trading**

From Alpaca's fee schedule (https://files.alpaca.markets/disclosures/library/BrokFeeSched.pdf):

✅ **$0 commission** on options trades  
✅ **$0 commission** on stock trades  
✅ **$0 platform fees**

### **Real Costs**

| Cost Type | Amount | When Applied |
|-----------|--------|--------------|
| **Bid-Ask Spread** | 2-10% of option price | Every trade (buy at ask, sell at bid) |
| **SEC Fee** | $0.00278 per $1,000 | Sell-side only |
| **FINRA TAF** | $0.000166 per share | Sell-side only |
| **OCC Fee** | $0.04 per contract | Both sides |
| **Regulatory Transaction Fee** | Variable | Based on volume |

**Example Trade:**
- Buy 1 SPY call @ $5.00 (ask price)
- Bid-ask spread: $4.90 - $5.10 (4% spread)
- **Immediate cost:** $0.10 per share × 100 = $10 (2% of $500 position)
- **OCC fee:** $0.04
- **Total entry cost:** $10.04 (2.01% of position)

---

## 🎨 **Strategy Diversity Improvements**

### **Phase 1: Expand Single-Leg Strategies**

**New Actions (31 → 61 actions):**

| Action Range | Strategy | Description |
|--------------|----------|-------------|
| 0 | Hold | No action |
| 1-15 | Buy Calls | 15 strikes (ATM ± 10%) |
| 16-30 | Buy Puts | 15 strikes (ATM ± 10%) |
| 31-45 | Sell Calls | Covered calls (need stock) |
| 46-60 | Sell Puts | Cash-secured puts |

**Benefits:**
- ✅ Enables income strategies (covered calls, cash-secured puts)
- ✅ More granular strike selection
- ✅ Better risk management

### **Phase 2: Multi-Leg Strategies**

**New Actions (61 → 91 actions):**

| Action Range | Strategy | Description | Risk Profile |
|--------------|----------|-------------|--------------|
| 61-65 | Bull Call Spread | Buy ATM call, sell OTM call | Defined risk bullish |
| 66-70 | Bear Put Spread | Buy ATM put, sell OTM put | Defined risk bearish |
| 71-75 | Long Straddle | Buy ATM call + ATM put | High volatility play |
| 76-80 | Long Strangle | Buy OTM call + OTM put | Lower cost volatility play |
| 81-85 | Iron Condor | Sell call spread + put spread | Range-bound income |
| 86-90 | Butterfly Spread | 3-leg defined risk | Neutral strategy |

**Benefits:**
- ✅ Defined risk strategies (spreads limit max loss)
- ✅ Volatility strategies (straddles, strangles)
- ✅ Income strategies (iron condors)
- ✅ More realistic professional trading

### **Phase 3: Dynamic Strategy Selection**

**Intelligent Action Masking:**

```python
def get_valid_actions(self, state):
    """Mask invalid actions based on current state"""
    valid_actions = [0]  # Hold always valid
    
    # Can only sell covered calls if we own stock
    if self.stock_positions > 0:
        valid_actions.extend(range(31, 46))  # Sell calls
    
    # Can only sell cash-secured puts if we have capital
    required_capital = self._calculate_put_collateral()
    if self.capital >= required_capital:
        valid_actions.extend(range(46, 61))  # Sell puts
    
    # Can only do spreads if we have capital for both legs
    if self.capital >= self._calculate_spread_cost():
        valid_actions.extend(range(61, 91))  # Multi-leg
    
    return valid_actions
```

**Benefits:**
- ✅ Prevents invalid trades
- ✅ Faster training (smaller action space)
- ✅ More realistic constraints

---

## 💸 **Realistic Transaction Cost Implementation**

### **Phase 1: Bid-Ask Spread Modeling**

**Current Data (from Alpaca):**

<augment_code_snippet path="src/historical_options_data.py" mode="EXCERPT">
````python
# Simulate bid-ask spread (wider for less liquid options)
spread_pct = 0.02 + 0.03 * abs(1 - moneyness)
bid = option_price * (1 - spread_pct/2)
ask = option_price * (1 + spread_pct/2)
````
</augment_code_snippet>

**Improvement: Use Real Alpaca Bid-Ask Data**

```python
def get_realistic_transaction_cost(self, option_data, quantity, side='buy'):
    """Calculate realistic transaction cost using Alpaca data"""
    
    # 1. Bid-Ask Spread Cost
    if side == 'buy':
        execution_price = option_data['ask']  # Buy at ask
        spread_cost = (option_data['ask'] - option_data['bid']) * quantity * 100
    else:  # sell
        execution_price = option_data['bid']  # Sell at bid
        spread_cost = (option_data['ask'] - option_data['bid']) * quantity * 100
    
    # 2. OCC Fee (both sides)
    occ_fee = 0.04 * quantity
    
    # 3. Regulatory Fees (sell-side only)
    if side == 'sell':
        trade_value = execution_price * quantity * 100
        sec_fee = (trade_value / 1000) * 0.00278
        finra_taf = quantity * 100 * 0.000166
    else:
        sec_fee = 0
        finra_taf = 0
    
    # 4. Total Cost
    total_cost = spread_cost + occ_fee + sec_fee + finra_taf
    
    return {
        'execution_price': execution_price,
        'spread_cost': spread_cost,
        'occ_fee': occ_fee,
        'sec_fee': sec_fee,
        'finra_taf': finra_taf,
        'total_cost': total_cost,
        'cost_pct': total_cost / (execution_price * quantity * 100)
    }
```

### **Phase 2: Slippage Modeling**

**Market Impact Based on Volume:**

```python
def calculate_slippage(self, option_data, quantity):
    """Model slippage based on order size vs daily volume"""
    
    daily_volume = option_data.get('volume', 100)
    open_interest = option_data.get('open_interest', 1000)
    
    # Order size as % of daily volume
    order_pct = quantity / max(daily_volume, 1)
    
    # Slippage increases non-linearly with order size
    if order_pct < 0.01:  # < 1% of volume
        slippage_pct = 0.001  # 0.1% slippage
    elif order_pct < 0.05:  # 1-5% of volume
        slippage_pct = 0.005  # 0.5% slippage
    elif order_pct < 0.10:  # 5-10% of volume
        slippage_pct = 0.01   # 1% slippage
    else:  # > 10% of volume
        slippage_pct = 0.02 + (order_pct - 0.10) * 0.1  # 2%+ slippage
    
    mid_price = (option_data['bid'] + option_data['ask']) / 2
    slippage_cost = mid_price * slippage_pct * quantity * 100
    
    return slippage_cost
```

### **Phase 3: Updated Reward Function**

**Incorporate All Transaction Costs:**

```python
def _calculate_reward(self, trade_result):
    """Calculate reward with realistic transaction costs"""
    
    # Portfolio value change
    portfolio_value_after = self._calculate_portfolio_value()
    raw_return = portfolio_value_after - self.previous_portfolio_value
    
    # Transaction costs (if trade executed)
    if trade_result and trade_result.get('success'):
        costs = trade_result.get('transaction_costs', {})
        
        # Breakdown of costs
        spread_cost = costs.get('spread_cost', 0)
        occ_fee = costs.get('occ_fee', 0)
        sec_fee = costs.get('sec_fee', 0)
        finra_taf = costs.get('finra_taf', 0)
        slippage = costs.get('slippage', 0)
        
        total_cost = spread_cost + occ_fee + sec_fee + finra_taf + slippage
        
        # Log for analysis
        if total_cost > 0:
            logger.debug(f"Transaction costs: spread=${spread_cost:.2f}, "
                        f"OCC=${occ_fee:.2f}, SEC=${sec_fee:.2f}, "
                        f"FINRA=${finra_taf:.2f}, slippage=${slippage:.2f}, "
                        f"total=${total_cost:.2f}")
    else:
        total_cost = 0
    
    # Net return after costs
    net_return = raw_return - total_cost
    
    # Apply scaling factor (1e-4 from research paper)
    reward = net_return * 1e-4
    
    # Update for next step
    self.previous_portfolio_value = portfolio_value_after
    
    return reward
```

---

## 📈 **Expected Impact**

### **Strategy Diversity**

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Action Space** | 31 | 91 | +193% |
| **Strategy Types** | 2 (buy call/put) | 8 (spreads, straddles, etc.) | +300% |
| **Risk Profiles** | 1 (directional) | 4 (directional, volatility, income, neutral) | +300% |
| **Win Rate** | Baseline | +10-20% | Better risk management |
| **Sharpe Ratio** | Baseline | +15-25% | Defined risk strategies |

### **Transaction Costs**

| Metric | Before | After | Impact |
|--------|--------|-------|--------|
| **Commission** | $0.65 fixed | $0 (Alpaca) | -100% |
| **Bid-Ask Spread** | Not modeled | 2-10% of price | +2-10% cost |
| **Regulatory Fees** | Not modeled | ~$0.05-0.10 | +$0.05-0.10 |
| **Slippage** | Not modeled | 0.1-2% | +0.1-2% cost |
| **Total Cost** | $0.65 | $10-50 per trade | More realistic |
| **Training Realism** | Low | High | Better generalization |

---

## 🚀 **Implementation Roadmap**

### **Week 1: Transaction Costs**

- [ ] Extract real bid-ask spreads from Alpaca data
- [ ] Implement regulatory fee calculations
- [ ] Add slippage modeling
- [ ] Update reward function
- [ ] Test with current training

### **Week 2: Strategy Diversity (Phase 1)**

- [ ] Expand action space to 61 actions
- [ ] Add covered call logic
- [ ] Add cash-secured put logic
- [ ] Implement action masking
- [ ] Test with expanded actions

### **Week 3: Strategy Diversity (Phase 2)**

- [ ] Add multi-leg strategies (spreads, straddles)
- [ ] Implement iron condor logic
- [ ] Add butterfly spreads
- [ ] Expand action space to 91 actions
- [ ] Test all strategies

### **Week 4: Integration & Testing**

- [ ] Integrate all improvements
- [ ] Run full training (5,000 episodes)
- [ ] Compare performance metrics
- [ ] Document results
- [ ] Create final report

---

## 📋 **Next Steps**

1. **Review this document** and approve implementation plan
2. **Start with Week 1** (transaction costs) - highest impact, lowest risk
3. **Validate with backtesting** before moving to Week 2
4. **Iterate based on results** - adjust if needed

**Estimated Total Time:** 4 weeks  
**Expected Performance Improvement:** +20-40% (combined effect)

---

## 🎯 **Success Criteria**

### **Transaction Costs**

- ✅ Bid-ask spreads match Alpaca data (within 10%)
- ✅ Regulatory fees calculated correctly
- ✅ Slippage model validated against real trades
- ✅ Training converges with new costs

### **Strategy Diversity**

- ✅ All 91 actions work correctly
- ✅ Multi-leg strategies execute properly
- ✅ Action masking prevents invalid trades
- ✅ Win rate improves by 10-20%
- ✅ Sharpe ratio improves by 15-25%

---

**Ready to implement! Start with Week 1 (Transaction Costs) for immediate impact.** 🚀

