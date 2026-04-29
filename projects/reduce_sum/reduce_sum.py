"""
Triton-Ascend Reduce Sum

对输入张量沿指定 axis 求和:
  output = sum(input, axis=dim)

支持两种累加策略 (precision 参数):
1. "simple":   直接 acc += tl.sum(x, RBLOCK) — 最快，精度一般
2. "pairwise": RBLOCK=8 逐元素累加到 [xsub, 8]，最后 tl.sum(8) — 精度好

  pairwise 原理 (a+b+c+d+e+f+g+h):
    simple:   ((((a+b)+c)+d)+e)+f)+g)+h)     ← 大数吃小数
    pairwise: (a+b)+(c+d)+(e+f)+(g+h)          ← 等量级相加
              = col_0_sum + col_1_sum + ... + col_7_sum

NPU 特殊处理:
- 累加器始终使用 float32
- bfloat16 输入需 .to(tl.float32) 再 reduce
"""

import triton
import triton.language as tl
import triton.runtime.driver as driver
import torch

_UB_MAX_INPUT_ELEMENTS = 12 * 1024
_PAIRWISE_RBLOCK = 8


# ============================================================================
# 2D Optimized kernel — simple 累加 (大块 tl.sum)
# ============================================================================

@triton.jit
def _reduce_sum_opt_simple_kernel(
    in_ptr, out_ptr,
    n_rows, n_cols,
    stride_in_row, stride_out_row,
    XBLOCK: tl.constexpr,
    XBLOCK_SUB: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    rbase = tl.arange(0, RBLOCK)
    x_loops: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    r_loops: tl.constexpr = (n_cols + RBLOCK - 1) // RBLOCK

    for x_loop in range(x_loops):
        row_idx = xoffset + x_loop * XBLOCK_SUB + tl.arange(0, XBLOCK_SUB)
        row_mask = row_idx < n_rows
        acc = tl.zeros([XBLOCK_SUB], dtype=tl.float32)

        for r_loop in range(r_loops):
            col_start = r_loop * RBLOCK
            rindex = col_start + rbase
            rmask = rindex < n_cols
            ptrs = in_ptr + row_idx[:, None] * stride_in_row + rindex[None, :]
            block_mask = row_mask[:, None] & rmask[None, :]
            x = tl.load(ptrs, mask=block_mask, other=0.0)
            acc = acc + tl.sum(x, axis=1).to(tl.float32)

        tl.store(out_ptr + row_idx * stride_out_row, acc, mask=row_mask)


# ============================================================================
# 2D Optimized kernel — pairwise (小块逐元素累加, 最后 tl.sum(8))
# ============================================================================

@triton.jit
def _reduce_sum_opt_pairwise_kernel(
    in_ptr, out_ptr,
    n_rows, n_cols,
    stride_in_row, stride_out_row,
    XBLOCK: tl.constexpr,
    XBLOCK_SUB: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    """
    Pairwise: RBLOCK=8, 逐元素累加到 [xsub, 8], 最后 tl.sum(8).
    每 8 个元素为一组，各列独立累加 → 最后 8 个等量级列和相加 → 精度好.
    """
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    rbase = tl.arange(0, RBLOCK)
    x_loops: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    r_loops: tl.constexpr = (n_cols + RBLOCK - 1) // RBLOCK

    for x_loop in range(x_loops):
        row_idx = xoffset + x_loop * XBLOCK_SUB + tl.arange(0, XBLOCK_SUB)
        row_mask = row_idx < n_rows
        acc = tl.zeros([XBLOCK_SUB, RBLOCK], dtype=tl.float32)

        for r_loop in range(r_loops):
            col_start = r_loop * RBLOCK
            rindex = col_start + rbase
            rmask = rindex < n_cols
            ptrs = in_ptr + row_idx[:, None] * stride_in_row + rindex[None, :]
            block_mask = row_mask[:, None] & rmask[None, :]
            x = tl.load(ptrs, mask=block_mask, other=0.0).to(tl.float32)
            acc = acc + x

        result = tl.sum(acc, axis=1)
        tl.store(out_ptr + row_idx * stride_out_row, result, mask=row_mask)


# ============================================================================
# Naive kernels
# ============================================================================

@triton.jit
def _reduce_sum_naive_simple_kernel(
    in_ptr, out_ptr,
    n_rows, n_cols,
    stride_in_row, stride_out_row,
    RBLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_idx = pid
    rbase = tl.arange(0, RBLOCK)
    r_loops: tl.constexpr = (n_cols + RBLOCK - 1) // RBLOCK
    acc = tl.zeros([], dtype=tl.float32)
    for r_loop in range(r_loops):
        col_start = r_loop * RBLOCK
        rindex = col_start + rbase
        rmask = rindex < n_cols
        ptrs = in_ptr + row_idx * stride_in_row + rindex
        x = tl.load(ptrs, mask=rmask, other=0.0)
        acc = acc + tl.sum(x).to(tl.float32)
    tl.store(out_ptr + row_idx * stride_out_row, acc)


@triton.jit
def _reduce_sum_naive_pairwise_kernel(
    in_ptr, out_ptr,
    n_rows, n_cols,
    stride_in_row, stride_out_row,
    RBLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_idx = pid
    rbase = tl.arange(0, RBLOCK)
    r_loops: tl.constexpr = (n_cols + RBLOCK - 1) // RBLOCK
    acc = tl.zeros([RBLOCK], dtype=tl.float32)
    for r_loop in range(r_loops):
        col_start = r_loop * RBLOCK
        rindex = col_start + rbase
        rmask = rindex < n_cols
        ptrs = in_ptr + row_idx * stride_in_row + rindex
        x = tl.load(ptrs, mask=rmask, other=0.0).to(tl.float32)
        acc = acc + x
    tl.store(out_ptr + row_idx * stride_out_row, tl.sum(acc))


# ============================================================================
# 1D 全量 sum (多核部分和)
# ============================================================================

@triton.jit
def _reduce_sum_1d_kernel(
    in_ptr, out_ptr,
    n_elements,
    XBLOCK: tl.constexpr,
    XBLOCK_SUB: tl.constexpr,
):
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    base = tl.arange(0, XBLOCK_SUB)
    loops: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    acc = tl.zeros([XBLOCK_SUB], dtype=tl.float32)
    for loop_idx in range(loops):
        x_index = xoffset + loop_idx * XBLOCK_SUB + base
        mask = x_index < n_elements
        x = tl.load(in_ptr + x_index, mask=mask, other=0.0)
        acc = acc + x.to(tl.float32)
    tl.store(out_ptr + pid, tl.sum(acc))


# ============================================================================
# Host 函数
# ============================================================================

def _get_num_vector_cores() -> int:
    device = torch.npu.current_device()
    props = driver.active.utils.get_device_properties(device)
    return props["num_vectorcore"]


def _safe_rblock(n_cols: int, xblock_sub: int, element_bytes: int = 4) -> int:
    max_elements = _UB_MAX_INPUT_ELEMENTS // max(xblock_sub, 1)
    rblock = triton.next_power_of_2(min(n_cols, max_elements))
    return max(rblock, 1)


def reduce_sum(
    x: torch.Tensor,
    axis: int = -1,
    mode: str = "optimized",
    precision: str = "pairwise",
) -> torch.Tensor:
    """
    Reduce sum: output = sum(x, axis=dim)

    参数:
        x: 输入张量
        axis: 归约轴 (支持 -1/last axis 和 None/全量)
        mode: "naive" | "optimized"
        precision: "simple" | "pairwise"
          - simple:   acc += tl.sum(x, axis=1) 一次大块归约，最快
          - pairwise: RBLOCK=8 逐元素累加 [xsub, 8]，最后 tl.sum(8)
                      等量级相加: (a+b)+(c+d)+...

    返回:
        归约结果张量
    """
    if axis is None:
        return _reduce_sum_all(x)
    if axis == -1 or axis == x.ndim - 1:
        return _reduce_sum_last_axis(x, mode=mode, precision=precision)
    raise NotImplementedError(
        f"reduce_sum only supports axis=-1 and axis=None. Got axis={axis}"
    )


def _reduce_sum_all(x: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    x_flat = x.reshape(-1).contiguous()
    num_cores = _get_num_vector_cores()
    num_cores = min(num_cores, n_elements)
    xblock = triton.cdiv(n_elements, num_cores)
    xblock_sub = min(xblock, 1024)
    partial = torch.zeros(num_cores, device=x.device, dtype=x.dtype)
    _reduce_sum_1d_kernel[(num_cores, 1, 1)](
        x_flat, partial, n_elements,
        XBLOCK=xblock, XBLOCK_SUB=xblock_sub,
    )
    return partial.sum()


def _reduce_sum_last_axis(
    x: torch.Tensor,
    mode: str = "optimized",
    precision: str = "pairwise",
) -> torch.Tensor:
    assert x.is_contiguous(), "Input must be contiguous"

    original_shape = x.shape
    n_cols = x.shape[-1]
    n_rows = x.numel() // n_cols
    x_2d = x.reshape(n_rows, n_cols)
    out_shape = original_shape[:-1]
    output = torch.empty(n_rows, device=x.device, dtype=x.dtype)
    element_bytes = x.element_size()
    num_cores = _get_num_vector_cores()

    if mode == "naive":
        if precision == "pairwise":
            rblock = _PAIRWISE_RBLOCK
            _reduce_sum_naive_pairwise_kernel[(n_rows,)](
                x_2d, output, n_rows, n_cols,
                x_2d.stride(0), output.stride(0),
                RBLOCK=rblock,
            )
        else:
            rblock = _safe_rblock(n_cols, 1, element_bytes)
            _reduce_sum_naive_simple_kernel[(n_rows,)](
                x_2d, output, n_rows, n_cols,
                x_2d.stride(0), output.stride(0),
                RBLOCK=rblock,
            )
    else:
        xblock = triton.cdiv(n_rows, num_cores)
        xblock_sub = min(xblock, 8)

        if precision == "pairwise":
            rblock = _PAIRWISE_RBLOCK
            _reduce_sum_opt_pairwise_kernel[(num_cores, 1, 1)](
                x_2d, output, n_rows, n_cols,
                x_2d.stride(0), output.stride(0),
                XBLOCK=xblock, XBLOCK_SUB=xblock_sub, RBLOCK=rblock,
            )
        else:
            rblock = _safe_rblock(n_cols, xblock_sub, element_bytes)
            _reduce_sum_opt_simple_kernel[(num_cores, 1, 1)](
                x_2d, output, n_rows, n_cols,
                x_2d.stride(0), output.stride(0),
                XBLOCK=xblock, XBLOCK_SUB=xblock_sub, RBLOCK=rblock,
            )

    return output.reshape(out_shape)


def ref_program(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    return torch.sum(x, dim=axis)


def main():
    torch.npu.set_device(0)
    num_cores = _get_num_vector_cores()
    print(f"Vector Cores: {num_cores}")

    tests = [
        ("2D small",      (128, 512),   -1),
        ("2D large cols", (256, 8192),  -1),
    ]

    for name, shape, axis in tests:
        x = torch.randn(*shape, device="npu", dtype=torch.float32)
        ref = ref_program(x, axis=axis)
        for prec in ["simple", "pairwise"]:
            result = reduce_sum(x, axis=axis, mode="optimized", precision=prec)
            diff = torch.max(torch.abs(result.cpu() - ref.cpu())).item()
            rtol = 1e-3 if "large" in name else 1e-4
            passed = torch.allclose(result.cpu(), ref.cpu(), rtol=rtol, atol=rtol)
            print(f"  {name:<16} precision={prec:<10} diff={diff:.2e} {'PASS' if passed else 'FAIL'}")

    print("\nAll checks done!")


if __name__ == "__main__":
    main()
