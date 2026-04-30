"""
Triton-Ascend 向量加法 (Vector Add)

两种实现模式:
1. naive:     GPU 风格，每个 program 一段连续元素
2. optimized: @triton.autotune 自动搜索最优 BLOCK_SIZE + multibuffer + constexpr 循环

NPU 内存 bound 操作分析:
  vector_add = 2 load + 1 store + 1 add，计算量极小
  瓶颈是 GM 带宽，不是核心调度/UB 容量
  kernel 结构优化对纯内存 bound 操作收益有限
  需要通过 autotune 搜索最优 block size + multibuffer 组合
"""

import triton
import triton.language as tl
import triton.runtime.driver as driver
import torch

_UB_PRACTICAL_LIMIT = 48 * 1024  # 48KB 实用上限（留余量）
_IO_TENSORS = 3  # a, b, c


# ============================================================================
# Naive kernel (GPU 风格 baseline)
# ============================================================================

@triton.jit
def _vector_add_kernel_naive(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)


# ============================================================================
# Optimized kernel — XBLOCK/XBLOCK_SUB constexpr 循环 (NPU 推荐模式)
# ============================================================================

@triton.jit
def _vector_add_kernel_blocking(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    XBLOCK: tl.constexpr,
    XBLOCK_SUB: tl.constexpr,
):
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    base = tl.arange(0, XBLOCK_SUB)
    loops: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB

    for loop_idx in range(loops):
        x_index = xoffset + loop_idx * XBLOCK_SUB + base
        mask = x_index < n_elements
        a = tl.load(a_ptr + x_index, mask=mask)
        b = tl.load(b_ptr + x_index, mask=mask)
        tl.store(c_ptr + x_index, a + b, mask=mask)


# ============================================================================
# Autotune kernel — 自动搜索最优 BLOCK_SIZE + multibuffer
# skill 推荐 element-wise block 范围: 128 ~ 4096
# 搜索 multibuffer True/False: True 流水重叠, False UB 翻倍可用更大 block
# ============================================================================

@triton.autotune(
    configs=[
        # multibuffer=True: 流水重叠（UB 减半为 96KB slot）
        triton.Config({'BLOCK_SIZE': 512},  multibuffer=True),
        triton.Config({'BLOCK_SIZE': 1024}, multibuffer=True),
        triton.Config({'BLOCK_SIZE': 2048}, multibuffer=True),
        triton.Config({'BLOCK_SIZE': 4096}, multibuffer=True),
        # multibuffer=False: UB 全量 192KB，适合更大 block
        triton.Config({'BLOCK_SIZE': 2048}, multibuffer=False),
        triton.Config({'BLOCK_SIZE': 4096}, multibuffer=False),
        triton.Config({'BLOCK_SIZE': 8192}, multibuffer=False),
    ],
    key=['n_elements'],
)
@triton.jit
def _vector_add_kernel_autotune(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)
    for block_id in tl.range(pid, tl.cdiv(n_elements, BLOCK_SIZE), num_cores):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        tl.store(c_ptr + offsets, a + b, mask=mask)


# ============================================================================
# Host 函数
# ============================================================================

def _get_num_vector_cores() -> int:
    device = torch.npu.current_device()
    props = driver.active.utils.get_device_properties(device)
    return props["num_vectorcore"]


def _safe_block_size(element_bytes: int) -> int:
    """UB 安全的 block size (2 的幂).

    使用实用上限 48KB 而非理论 96KB，留余量给编译器中间变量。
    3 个 I/O 张量 (a, b, c) 平分实用上限。
    """
    max_elements = _UB_PRACTICAL_LIMIT // (_IO_TENSORS * element_bytes)
    return max(1 << (max_elements.bit_length() - 1), 64)


def vector_add(
    a: torch.Tensor,
    b: torch.Tensor,
    mode: str = "optimized",
    xblock_sub: int = 0,
) -> torch.Tensor:
    """
    向量加法: C = A + B

    参数:
        a: 输入向量 A
        b: 输入向量 B
        mode: "naive" | "optimized" | "autotune"
        xblock_sub: 核内子块大小 (仅 optimized 模式)，0 表示 UB 安全自动计算

    返回:
        输出向量 C
    """
    assert a.shape == b.shape, "Input shapes must match"
    n_elements = a.numel()
    element_bytes = a.element_size()

    if xblock_sub <= 0:
        xblock_sub = _safe_block_size(element_bytes)

    if mode == "naive":
        c = torch.empty_like(a)
        grid = (triton.cdiv(n_elements, xblock_sub),)
        _vector_add_kernel_naive[grid](
            a, b, c, n_elements, BLOCK_SIZE=xblock_sub,
        )
        return c

    elif mode == "optimized":
        num_cores = _get_num_vector_cores()
        xblock = triton.cdiv(n_elements, num_cores)
        xblock = triton.cdiv(xblock, xblock_sub) * xblock_sub
        c = torch.empty_like(a)
        grid = (num_cores, 1, 1)
        _vector_add_kernel_blocking[grid](
            a, b, c, n_elements,
            XBLOCK=xblock, XBLOCK_SUB=xblock_sub,
        )
        return c

    elif mode == "autotune":
        num_cores = _get_num_vector_cores()
        c = torch.empty_like(a)
        grid = (num_cores,)
        _vector_add_kernel_autotune[grid](
            a, b, c, n_elements,
        )
        return c

    else:
        raise ValueError(f"Unknown mode: {mode}")


def ref_program(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b


def main():
    torch.npu.set_device(0)

    num_cores = _get_num_vector_cores()
    safe_block = _safe_block_size(4)
    print(f"Vector Cores: {num_cores}")
    print(f"Safe block size (fp32): {safe_block} elements = "
          f"{safe_block * _IO_TENSORS * 4 / 1024:.0f} KB/block")

    for size in [64 * 1024, 1024 * 1024]:
        a = torch.randn(size, device="npu", dtype=torch.float32)
        b = torch.randn(size, device="npu", dtype=torch.float32)
        ref_c = ref_program(a, b)

        for mode in ["naive", "optimized", "autotune"]:
            c = vector_add(a, b, mode=mode)
            torch.testing.assert_close(c.cpu(), ref_c.cpu(), rtol=1e-6, atol=1e-6)
            print(f"  size={size:>8}, mode={mode:>12}: PASS")

    print("\nAll correctness checks passed!")


if __name__ == "__main__":
    main()
