# 🐛 Bug Fix: GPUOptimizer Initialization Error

## ❌ Error

When running multi-GPU training with ensemble methods:

```bash
python train_enhanced_clstm_ppo.py \
    --num_episodes 5000 \
    --num_gpus 2 \
    --enable-multi-leg \
    --use-ensemble \
    --num-ensemble-models 3
```

**Error message:**
```
TypeError: GPUOptimizer.__init__() got an unexpected keyword argument 'device'
```

---

## 🔍 Root Cause

**File:** `train_enhanced_clstm_ppo.py` (line 161)

**Problem:** The code was passing `device=self.device` to `GPUOptimizer.__init__()`, but the class only accepts `config` parameter.

**Incorrect code:**
```python
# Line 161 (BEFORE FIX)
self.gpu_optimizer = GPUOptimizer(config=self.config, device=self.device)
```

**GPUOptimizer signature:**
```python
# src/gpu_optimizations.py (line 27)
def __init__(self, config: dict = None):
    self.config = config or {}
    # ... (no device parameter)
```

---

## ✅ Fix Applied

**File:** `train_enhanced_clstm_ppo.py` (line 161)

**Changed:**
```python
# BEFORE (line 161)
self.gpu_optimizer = GPUOptimizer(config=self.config, device=self.device)

# AFTER (line 161)
self.gpu_optimizer = GPUOptimizer(config=self.config)
```

**Explanation:**
- Removed the `device=self.device` parameter
- The device is already set via `torch.cuda.set_device(self.device)` on line 160
- `GPUOptimizer` doesn't need the device parameter since it uses the current CUDA device

---

## 🧪 Verification

**Syntax check:**
```bash
python -c "import ast; ast.parse(open('train_enhanced_clstm_ppo.py').read()); print('✅ Syntax check passed!')"
```

**Result:** ✅ No syntax errors

---

## 📝 Complete Fix

**Lines 156-174 (AFTER FIX):**

```python
# GPU optimization setup
if distributed:
    # Distributed training: use specific GPU for this rank
    self.device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(self.device)
    self.gpu_optimizer = GPUOptimizer(config=self.config)  # ← FIXED: Removed device parameter
    self.gradient_scaler = self.gpu_optimizer.create_gradient_scaler()

    # Enable GPU optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
else:
    # Single GPU training
    self.gpu_optimizer = GPUOptimizer(config=self.config)
    self.device = self.gpu_optimizer.setup_device()
    self.gradient_scaler = self.gpu_optimizer.create_gradient_scaler()
```

---

## 🚀 Next Steps

**Try running the training command again:**

```bash
python train_enhanced_clstm_ppo.py \
    --num_episodes 5000 \
    --num_gpus 2 \
    --enable-multi-leg \
    --use-ensemble \
    --num-ensemble-models 3
```

**Expected behavior:**
- ✅ No `TypeError` about unexpected keyword argument
- ✅ Multi-GPU training should start successfully
- ✅ Ensemble training with 3 models should work
- ✅ Multi-leg strategies (91 actions) should be enabled

---

## 📊 Summary

| Item | Before | After |
|------|--------|-------|
| **Error** | ❌ TypeError on GPUOptimizer init | ✅ Fixed |
| **Line 161** | `GPUOptimizer(config=..., device=...)` | `GPUOptimizer(config=...)` |
| **Status** | ❌ Training crashes | ✅ Ready to train |

**Fix applied successfully! 🎉**

