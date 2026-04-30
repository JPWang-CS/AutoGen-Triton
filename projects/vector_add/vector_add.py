"""
Triton-Ascend 向量加法 (Vector Add)

四种实现模式:
1. naive:     GPU 风格，每个 program 一段连续元素，grid 数量远超物理核数
2. persistent: 固定核数 + tl.range 跨步分配（循环非 constexpr）
3. optimized: XBLOCK/XBLOCK_SUB 双层切分 + constexpr 循环（编译器可展开流水线）
4. autotune:  @triton.autotune 自动搜索最优 BLOCK_SIZE + multibuffer 组合

UB 优化:
- 根据数据类型自动计算最优 block size，填满 UB
- doublebuffer 后 96KB 可用，3 个 I/O 张量 → float32 最优 8192 元素
- autotune 模式额外搜索 multibuffer=False（UB 翻倍，适合大 block）
"""

import triton
import triton.language as tl
import triton.runtime.driver as driver
import torch

_UB_TOTAL_BYTES = 192 * 1024  # 192KB

_IO_TENSORS = 3  # a, b, c


# ============================================================================
# Naive kernel (GPU 风格)
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
# Persistent kernel (固定核数 + tl.range，循环非 constexpr)
# ============================================================================

@triton.jit
def _vector_add_kernel_persistent(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    for block_id in tl.range(pid, tl.cdiv(n_elements, BLOCK_SIZE), NUM_CORES):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        tl.store(c_ptr + offsets, a + b, mask=mask)


# ============================================================================
# Optimized kernel (XBLOCK/XBLOCK_SUB constexpr 循环)
# ============================================================================

@triton.jit
def _vector_add_kernel_optimized(
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
# Autotune kernel — @triton.autotune 自动搜索最优配置
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, multibuffer=True),
        triton.Config({'BLOCK_SIZE': 4096}, multibuffer=True),
        triton.Config({'BLOCK_SIZE': 8192}, multibuffer=True),
        triton.Config({'BLOCK_SIZE': 4096}, multibuffer=False),
        triton.Config({'BLOCK_SIZE': 8192}, multibuffer=False),
        triton.Config({'BLOCK_SIZE': 16384}, multibuffer=False),
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


def _optimal_block_size(element_bytes: int) -> int:
    """计算填满 UB 的最优 block size (2 的幂).

    doublebuffer 将 192KB 分为两个 96KB 槽位.
    每个槽位存放 _IO_TENSORS 个张量 (a, b, c).
    """
    effective_ub = _UB_TOTAL_BYTES // 2
    max_elements = effective_ub // (_IO_TENSORS * element_bytes)
    return max(1 << (max_elements.bit_length() - 1), 64)


def vector_add(
    a: torch.Tensor,
    b: torch.Tensor,
    mode: str = "optimized",
    xblock_sub: int = 0,
) -> torch.Tensor:
    """
    向量加法主机函数: C = A + B

    参数:
        a: 输入向量 A
        b: 输入向量 B
        mode: "naive" | "persistent" | "optimized" | "autotune"
        xblock_sub: 核内子块大小 (仅 optimized 模式)，0 表示自动计算

    返回:
        输出向量 C，形状与输入相同
    """
    assert a.shape == b.shape, "Input shapes must match"
    n_elements = a.numel()
    element_bytes = a.element_size()

    if xblock_sub <= 0:
        xblock_sub = _optimal_block_size(element_bytes)

    if mode == "naive":
        c = torch.empty_like(a)
        grid = (triton.cdiv(n_elements, xblock_sub),)
        _vector_add_kernel_naive[grid](
            a, b, c, n_elements, BLOCK_SIZE=xblock_sub,
        )
        return c

    elif mode == "persistent":
        num_cores = _get_num_vector_cores()
        c = torch.empty_like(a)
        grid = (num_cores,)
        _vector_add_kernel_persistent[grid](
            a, b, c, n_elements,
            BLOCK_SIZE=xblock_sub, NUM_CORES=num_cores,
        )
        return c

    elif mode == "optimized":
        num_cores = _get_num_vector_cores()
        xblock = triton.cdiv(n_elements, num_cores)
        xblock = triton.cdiv(xblock, xblock_sub) * xblock_sub
        c = torch.empty_like(a)
        grid = (num_cores, 1, 1)
        _vector_add_kernel_optimized[grid](
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
    element_bytes = 4  # float32
    optimal = _optimal_block_size(element_bytes)
    print(f"Vector Cores: {num_cores}")
    print(f"Optimal block size (float32): {optimal} elements = {optimal * _IO_TENSORS * element_bytes / 1024:.0f} KB/block, "
          f"UB utilization: {optimal * _IO_TENSORS * element_bytes / (_UB_TOTAL_BYTES // 2) * 100:.0f}%")

    for size in [1024 * 64, 1024 * 1024]:
        a = torch.randn(size, device="npu", dtype=torch.float32)
        b = torch.randn(size, device="npu", dtype=torch.float32)
        ref_c = ref_program(a, b)

        for mode in ["naive", "persistent", "optimized", "autotune"]:
            c = vector_add(a, b, mode=mode)
            torch.testing.assert_close(c.cpu(), ref_c.cpu(), rtol=1e-6, atol=1e-6)
            print(f"  size={size:>8}, mode={mode:>12}: PASS")

    print("\nAll correctness checks passed!")


if __name__ == "__main__":
    main()
