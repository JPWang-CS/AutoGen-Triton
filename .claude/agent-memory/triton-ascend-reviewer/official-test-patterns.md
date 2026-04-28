---
name: Official Test Patterns
description: Key reference files and patterns from triton-ascend test suite for code review
type: reference
---

## Key Reference Files (in triton-ascend/third_party/ascend/unittest/pytest_ut/)

| File | Pattern | Notes |
|------|---------|-------|
| test_reduce_sum.py | Pointwise-Reduction (PR) | XBLOCK/XBLOCK_SUB/RBLOCK 3-level tiling, `for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB)` |
| test_sum.py | Dual-loop accumulation | sum_loop_high / sum_loop_low with chunked reduce axis |
| test_reduce_count_vector.py | bool->float32 conversion | `(x == val).to(tl.float32)` before `tl.sum()` |
| test_mean_dim0.py / test_mean_dim1.py | Mean reduction with dtype | `tl.sum(x.to(tl.float32), dim) / N` |

## Pointwise-Reduction Standard Pattern (from test_reduce_sum.py)

```python
@triton.jit
def kernel(out_ptr, in_ptr, xnumel, rnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr, RBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)
        xmask = xindex[:, None] < xnumel
        data = tl.load(in_ptr + (rindex + RBLOCK * xindex[:, None]), xmask & rmask)
        data = tl.reshape(data, [XBLOCK_SUB, RBLOCK])
        result = tl.sum(data, 1)
        tl.store(out_ptr + xindex, result, None)
```

Key: uses flat indexing `rindex + RBLOCK * xindex[:, None]` not stride-based indexing.

## loops: tl.constexpr Pattern (from API docs)

```python
loops1: tl.constexpr = XBLOCK // XBLOCK_SUB  # assumes exact divisibility
for loop1 in range(loops1):
    x_index = offset + (loop1 * XBLOCK_SUB) + base1
```

Note: This pattern assumes `XBLOCK % XBLOCK_SUB == 0`. The `range(0, XBLOCK, XBLOCK_SUB)` pattern handles non-divisible cases more safely.

## ub_size Reference

- `test_common.py` line 142: `ub_size = 98304 * 2` = 196608 bytes = 192KB
- This is the per-core UB capacity for Atlas A2

## Test Coverage Checklist for Reduce Operations

- [ ] float32, float16, bfloat16 dtypes
- [ ] int8, bool types (if applicable)
- [ ] Non-power-of-2 dimensions
- [ ] 1D, 2D, 3D+ input shapes
- [ ] Edge cases: empty tensor, single element, single row, single column
- [ ] Large tensors (UB overflow boundary)
- [ ] axis=None (full reduction) and axis=-1
- [ ] Both naive and optimized modes
