# NCCL Timeout Fix

If you encounter NCCL timeout errors during distributed training, here are the solutions:

## Quick Fix - Use Single GPU Mode

Run the training script with the `--no-distributed` flag to bypass distributed training:

```bash
python train.py --no-distributed
```

This will use only one GPU and avoid all NCCL communication issues.

## Root Causes

1. **Synchronization Issues**: Different GPUs taking different amounts of time to process
2. **Large Model/Batch Size**: Causing memory pressure and slowdowns
3. **Network Communication**: Issues with GPU-to-GPU communication

## Fixes Applied

1. **Increased NCCL Timeout**: Set to 60 minutes (was 30)
2. **Disabled P2P Communication**: Added `NCCL_P2P_DISABLE=1`
3. **Disabled InfiniBand**: Added `NCCL_IB_DISABLE=1`
4. **Added Timeout Protection**: `all_gather_object` now has a 5-second timeout
5. **Made Barriers Optional**: Disabled some synchronization barriers to prevent deadlocks

## Additional Recommendations

1. **Reduce Batch Size**: If using multiple GPUs, try reducing the batch size:
   - Current: 4096
   - Try: 2048 or 1024

2. **Monitor GPU Memory**: Use `nvidia-smi` to check if GPUs are running out of memory

3. **Check GPU Temperatures**: Thermal throttling can cause slowdowns

4. **Use Fewer GPUs**: Try using 2 GPUs instead of all available:
   ```bash
   CUDA_VISIBLE_DEVICES=0,1 python train.py
   ```

## Debug Mode

For more detailed NCCL debugging:
```bash
export NCCL_DEBUG=INFO
python train.py
```

## Environment Variables

You can also set these before running:
```bash
export NCCL_TIMEOUT=7200  # 2 hours
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
python train.py
```