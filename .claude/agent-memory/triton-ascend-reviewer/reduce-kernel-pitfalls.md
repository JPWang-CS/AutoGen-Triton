---
name: Common Reduce Kernel Pitfalls
description: UB overflow, core overlap, dtype promotion issues found in reduce_sum review
type: reference
---

## UB (片上内存) Overflow

- Atlas A2 片上内存 = 192KB (ub_size in tests = 98304*2 = 192KB)
- 开启 doublebuffer 后可用容量减半 (96KB)
- 1D full reduction: `next_power_of_2(n_elements)` as BLOCK_SIZE can easily exceed UB
- 2D reduction: `XBLOCK_SUB * RBLOCK * dtype_size * 2` must fit in UB
- **Must cap BLOCK_SIZE/RBLOCK**: layer-norm tutorial uses `MAX_FUSED_SIZE = 65536 // element_size()`
- Official test_sum.py has `reduce_check_ub_mem_overflow()` check: skip if `dtype_size * prod(shape) >= ub_size / 6`

## Core Overlap in XBLOCK/XBLOCK_SUB Pattern

- When `XBLOCK` is not divisible by `XBLOCK_SUB`, using `loops: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB` with `range(loops)` causes extra iterations
- Extra iterations process rows beyond `[xoffset, xoffset+XBLOCK)`, overlapping with next core's range
- **Fix**: Use `for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):` (official standard pattern)
- Official test_reduce_sum.py ensures `ncore * xblock * shape[-1] == numel` (strict divisibility)

## dtype Promotion for Reduction

- float16/bfloat16 MUST be promoted to float32 before `tl.sum()` to avoid precision loss and overflow
- Official pattern: `tl.sum(x.to(tl.float32), axis)` then `.to(original_dtype)` for output
- bool MUST be converted to float32 before sum (test_reduce_count_vector.py pattern)
- torch_sum reference in test_sum.py: `torch.sum(x1.to(torch.float32), dim=dim).to(x1.dtype)`

## Output Shape Recovery for Multi-dim Input

- When reshaping N-dim input to 2D `(n_rows, n_cols)` for kernel, must reshape output back to `original_shape[:-1]`
- Easy to forget since kernel naturally produces 1D output

## Two-loop Accumulation Pattern (for large reduce axis)

- When reduce axis exceeds UB capacity, must split into chunks and accumulate:
```python
_tmp = tl.full([BLOCK, RBLOCK], 0, tl.float32)
for roffset in range(0, rnumel, RBLOCK):
    # load chunk and accumulate
    _tmp = _tmp + loaded_chunk
result = tl.sum(_tmp, axis)  # final reduction
```
- See: test_sum.py sum_loop_high / sum_loop_low kernels
