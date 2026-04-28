"""
Triton-Ascend Reduce Sum

对输入张量沿指定 axis 求和:
  output = sum(input, axis=dim)

支持两种模式:
1. naive:      简单实现，每行一个 program
2. optimized:  XBLOCK/XBLOCK_SUB/RBLOCK 三层切分 + constexpr 循环 (NPU 推荐)

UB 安全:
- RBLOCK 受 UB 容量限制 (192KB / doublebuffer = 96KB 可用)
- 当 reduce axis 过长时，自动分块累加
- 1D 全量 sum 使用多核 + 子块循环模式

NPU 特殊处理:
- bfloat16 输入需要 .to(tl.float32) 后再 sum
- bool 必须转 float32 后再 sum
- float16 可以直接 sum
"""

import triton
import triton.language as tl
import triton.runtime.driver as driver
import torch

# UB 安全阈值: 192KB, doublebuffer 后可用 ~96KB, float32=4B
# 留一半给输入/输出缓冲区 → 单个输入块最大 ~24K float32 元素
_UB_MAX_ELEMENTS = 24 * 1024


# ============================================================================
# 1D 全量 sum (多核 + 子块循环)
# ============================================================================

@triton.jit
def _reduce_sum_1d_kernel(
    in_ptr, out_ptr,
    n_elements,
    XBLOCK: tl.constexpr,
    XBLOCK_SUB: tl.constexpr,
):
    """
    1D 全量 sum: 多核并行，每个 core 处理一段连续元素并累加到本地，
    最终写回一个部分和。host 端再做一次小规模 reduce 得到最终结果。
    """
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

    # 每个 core 输出一个部分和
    partial_sum = tl.sum(acc)
    tl.store(out_ptr + pid, partial_sum)


# ============================================================================
# 2D Pointwise-Reduction (沿 last axis sum，多核并行)
# ============================================================================

@triton.jit
def _reduce_sum_last_axis_kernel(
    in_ptr, out_ptr,
    n_rows, n_cols,
    stride_in_row, stride_out_row,
    XBLOCK: tl.constexpr,
    XBLOCK_SUB: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    """
    Pointwise-Reduction 模式:
    - 沿 last axis (axis=-1) 做 sum
    - 每个 core 处理若干行
    - 当 n_cols > RBLOCK 时，reduce axis 需要分块循环累加
    """
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    rbase = tl.arange(0, RBLOCK)

    loops: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    r_loops: tl.constexpr = (triton.cdiv(n_cols, RBLOCK))

    for loop_idx in range(loops):
        row_idx = xoffset + loop_idx * XBLOCK_SUB + tl.arange(0, XBLOCK_SUB)
        row_mask = row_idx < n_rows

        # 沿 reduce axis 分块累加
        acc = tl.zeros([XBLOCK_SUB], dtype=tl.float32)
        for r_loop in range(r_loops):
            col_start = r_loop * RBLOCK
            rindex = col_start + rbase
            rmask = rindex < n_cols

            ptrs = in_ptr + row_idx[:, None] * stride_in_row + rindex[None, :]
            block_mask = row_mask[:, None] & rmask[None, :]
            x = tl.load(ptrs, mask=block_mask, other=0.0)
            acc = acc + tl.sum(x, axis=1)

        # 写回每行的 sum 结果
        out_ptrs = out_ptr + row_idx * stride_out_row
        tl.store(out_ptrs, acc, mask=row_mask)


# ============================================================================
# Host 函数
# ============================================================================

def _get_num_vector_cores() -> int:
    device = torch.npu.current_device()
    props = driver.active.utils.get_device_properties(device)
    return props["num_vectorcore"]


def _safe_rblock(n_cols: int, xblock_sub: int, element_bytes: int = 4) -> int:
    """计算安全的 RBLOCK 大小，确保不超出 UB"""
    max_elements = _UB_MAX_ELEMENTS // max(xblock_sub, 1)
    rblock = triton.next_power_of_2(min(n_cols, max_elements))
    return max(rblock, 1)


def reduce_sum(
    x: torch.Tensor,
    axis: int = -1,
    mode: str = "optimized",
) -> torch.Tensor:
    """
    Reduce sum: output = sum(x, axis=dim)

    参数:
        x: 输入张量
        axis: 归约轴 (支持 -1/last axis 和 None/全量)
        mode: "naive" | "optimized"

    返回:
        归约结果张量
    """
    if axis is None:
        return _reduce_sum_all(x)
    if axis == -1 or axis == x.ndim - 1:
        return _reduce_sum_last_axis(x, mode=mode)
    raise NotImplementedError(
        f"reduce_sum only supports axis=-1 (last axis) and axis=None (full reduction). "
        f"Got axis={axis}"
    )


def _reduce_sum_all(x: torch.Tensor) -> torch.Tensor:
    """全量 sum: 多核部分和 + host 端最终 reduce"""
    n_elements = x.numel()
    x_flat = x.reshape(-1).contiguous()

    num_cores = _get_num_vector_cores()
    num_cores = min(num_cores, n_elements)

    xblock = triton.cdiv(n_elements, num_cores)
    xblock_sub = min(xblock, 1024)

    partial = torch.zeros(num_cores, device=x.device, dtype=x.dtype)

    grid = (num_cores, 1, 1)
    _reduce_sum_1d_kernel[grid](
        x_flat, partial,
        n_elements,
        XBLOCK=xblock, XBLOCK_SUB=xblock_sub,
    )

    # 最终 reduce: 对 num_cores 个部分和求总和
    return partial.sum()


def _reduce_sum_last_axis(x: torch.Tensor, mode: str = "optimized") -> torch.Tensor:
    """沿 last axis sum"""
    assert x.is_contiguous(), "Input must be contiguous"

    original_shape = x.shape
    n_cols = x.shape[-1]
    n_rows = x.numel() // n_cols
    x_2d = x.reshape(n_rows, n_cols)

    # 输出: 去掉 last axis 的形状
    out_shape = original_shape[:-1]
    output = torch.empty(n_rows, device=x.device, dtype=x.dtype)

    if mode == "naive":
        _reduce_sum_last_axis_naive(x_2d, output, n_rows, n_cols)
    else:
        _reduce_sum_last_axis_optimized(x_2d, output, n_rows, n_cols)

    return output.reshape(out_shape)


def _reduce_sum_last_axis_naive(
    x_2d: torch.Tensor, output: torch.Tensor,
    n_rows: int, n_cols: int,
):
    """Naive: 每个 program 处理一行"""
    element_bytes = x_2d.element_size()
    rblock = _safe_rblock(n_cols, xblock_sub=1, element_bytes=element_bytes)
    grid = (n_rows,)
    _reduce_sum_last_axis_kernel[grid](
        x_2d, output,
        n_rows, n_cols,
        x_2d.stride(0), output.stride(0),
        XBLOCK=1, XBLOCK_SUB=1, RBLOCK=rblock,
    )


def _reduce_sum_last_axis_optimized(
    x_2d: torch.Tensor, output: torch.Tensor,
    n_rows: int, n_cols: int,
):
    """Optimized: XBLOCK/XBLOCK_SUB/RBLOCK 三层切分"""
    element_bytes = x_2d.element_size()
    num_cores = _get_num_vector_cores()
    xblock = triton.cdiv(n_rows, num_cores)
    xblock_sub = min(xblock, 8)
    rblock = _safe_rblock(n_cols, xblock_sub, element_bytes)

    grid = (num_cores, 1, 1)
    _reduce_sum_last_axis_kernel[grid](
        x_2d, output,
        n_rows, n_cols,
        x_2d.stride(0), output.stride(0),
        XBLOCK=xblock, XBLOCK_SUB=xblock_sub, RBLOCK=rblock,
    )


def ref_program(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    return torch.sum(x, dim=axis)


def main():
    torch.npu.set_device(0)
    num_cores = _get_num_vector_cores()
    print(f"Vector Cores: {num_cores}")

    # 1D 全量 sum
    x = torch.randn(1024, device="npu", dtype=torch.float32)
    result = reduce_sum(x, axis=None)
    ref = ref_program(x, axis=None)
    torch.testing.assert_close(result.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)
    print(f"1D full sum: PASS (result={result.item():.4f}, ref={ref.item():.4f})")

    # 2D last axis sum (naive)
    x = torch.randn(128, 512, device="npu", dtype=torch.float32)
    result = reduce_sum(x, axis=-1, mode="naive")
    ref = ref_program(x, axis=-1)
    torch.testing.assert_close(result.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)
    print(f"2D last axis sum (naive): PASS, shape={result.shape}")

    # 2D last axis sum (optimized)
    result = reduce_sum(x, axis=-1, mode="optimized")
    torch.testing.assert_close(result.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)
    print(f"2D last axis sum (optimized): PASS, shape={result.shape}")

    # 3D
    x = torch.randn(8, 128, 64, device="npu", dtype=torch.float32)
    result = reduce_sum(x, axis=-1, mode="optimized")
    ref = ref_program(x, axis=-1)
    torch.testing.assert_close(result.cpu(), ref.cpu(), rtol=1e-3, atol=1e-3)
    print(f"3D last axis sum (optimized): PASS, shape={result.shape}")

    # 大 reduce axis
    x = torch.randn(256, 8192, device="npu", dtype=torch.float32)
    result = reduce_sum(x, axis=-1, mode="optimized")
    ref = ref_program(x, axis=-1)
    torch.testing.assert_close(result.cpu(), ref.cpu(), rtol=1e-3, atol=1e-3)
    print(f"Large reduce axis (8192): PASS, shape={result.shape}")

    print("\nAll correctness checks passed!")


if __name__ == "__main__":
    main()
