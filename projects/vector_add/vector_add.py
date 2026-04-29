"""
Triton-Ascend 向量加法 (Vector Add)

四种实现模式:
1. naive:     GPU 风格，每个 program 一段连续元素，grid 数量远超物理核数
2. persistent: 固定核数 + tl.range 跨步分配（循环非 constexpr）
3. optimized: XBLOCK/XBLOCK_SUB 双层切分 + constexpr 循环（编译器可展开流水线）
4. pipelined: optimized + 显式 multibuffer 控制 + 无 mask（最大带宽模式）

流水线原理 (multibuffer):
  编译器看到 constexpr 循环时，自动将 UB 分为多个缓冲区，
  使"拷入 N+1"与"计算 N"与"拷出 N-1"并行执行:

  MTE2:   [load₀] [load₁] [load₂] [load₃] ...
  Vector:         [calc₀] [calc₁] [calc₂] ...
  MTE2:                  [store₀] [store₁] [store₂] ...
"""

import triton
import triton.language as tl
import triton.runtime.driver as driver
import torch


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
    """
    constexpr 循环 → 编译器展开 → multibuffer 流水线自动生效
    mask 在尾部块起保护作用
    """
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
# Pipelined kernel (无 mask + multibuffer 显式控制)
# ============================================================================

@triton.jit
def _vector_add_kernel_pipelined(
    a_ptr, b_ptr, c_ptr,
    XBLOCK: tl.constexpr,
    XBLOCK_SUB: tl.constexpr,
):
    """
    最大带宽模式:
    - 无 mask: 纯连续内存访问，MTE2 可以满带宽搬运
    - 无 n_elements 参数: 要求输入已被 pad 到对齐大小
    - constexpr 循环 + 无 mask → 编译器生成最优流水线

    无 mask 为什么快:
    1. 避免 Vector 计算 mask (offsets < n) 的开销
    2. 避免 MTE2 等待 Vector 填充 padding=0 的同步点
    3. 纯连续寻址，硬件预取效率最高
    """
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    base = tl.arange(0, XBLOCK_SUB)
    loops: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB

    for loop_idx in range(loops):
        x_index = xoffset + loop_idx * XBLOCK_SUB + base
        # 无 mask — 纯连续内存访问
        a = tl.load(a_ptr + x_index)
        b = tl.load(b_ptr + x_index)
        tl.store(c_ptr + x_index, a + b)


# ============================================================================
# Host 函数
# ============================================================================

def _get_num_vector_cores() -> int:
    device = torch.npu.current_device()
    props = driver.active.utils.get_device_properties(device)
    return props["num_vectorcore"]


def _align_size(n: int, num_cores: int, xblock_sub: int) -> int:
    """将 size 向上对齐到 num_cores * xblock_sub 的倍数"""
    alignment = num_cores * xblock_sub
    return ((n + alignment - 1) // alignment) * alignment


def vector_add(
    a: torch.Tensor,
    b: torch.Tensor,
    mode: str = "optimized",
    xblock_sub: int = 1024,
) -> torch.Tensor:
    """
    向量加法主机函数: C = A + B

    参数:
        a: 输入向量 A
        b: 输入向量 B
        mode: "naive" | "persistent" | "optimized" | "pipelined"
        xblock_sub: 核内子块大小

    返回:
        输出向量 C，形状与输入相同
    """
    assert a.shape == b.shape, "Input shapes must match"
    n_elements = a.numel()

    if mode == "naive":
        block_size = 1024
        c = torch.empty_like(a)
        grid = (triton.cdiv(n_elements, block_size),)
        _vector_add_kernel_naive[grid](
            a, b, c, n_elements, BLOCK_SIZE=block_size,
        )
        return c

    elif mode == "persistent":
        num_cores = _get_num_vector_cores()
        block_size = 1024
        c = torch.empty_like(a)
        grid = (num_cores,)
        _vector_add_kernel_persistent[grid](
            a, b, c, n_elements,
            BLOCK_SIZE=block_size, NUM_CORES=num_cores,
        )
        return c

    elif mode == "optimized":
        num_cores = _get_num_vector_cores()
        xblock = triton.cdiv(n_elements, num_cores)
        c = torch.empty_like(a)
        grid = (num_cores, 1, 1)
        _vector_add_kernel_optimized[grid](
            a, b, c, n_elements,
            XBLOCK=xblock, XBLOCK_SUB=xblock_sub,
        )
        return c

    elif mode == "pipelined":
        num_cores = _get_num_vector_cores()
        n_padded = _align_size(n_elements, num_cores, xblock_sub)
        xblock = n_padded // num_cores

        # Pad input to aligned size (extra bytes are uninitialized but won't be used)
        if n_padded > n_elements:
            a_pad = torch.empty(n_padded, device=a.device, dtype=a.dtype)
            a_pad[:n_elements] = a
            b_pad = torch.empty(n_padded, device=b.device, dtype=b.dtype)
            b_pad[:n_elements] = b
            c_pad = torch.empty(n_padded, device=a.device, dtype=a.dtype)
        else:
            a_pad, b_pad, c_pad = a, b, torch.empty_like(a)

        grid = (num_cores, 1, 1)
        _vector_add_kernel_pipelined[grid](
            a_pad, b_pad, c_pad,
            XBLOCK=xblock, XBLOCK_SUB=xblock_sub,
        )
        return c_pad[:n_elements].reshape(a.shape)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def ref_program(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b


def main():
    torch.npu.set_device(0)

    num_cores = _get_num_vector_cores()
    print(f"Vector Cores: {num_cores}")

    for size in [1024 * 64, 1024 * 1024]:
        a = torch.randn(size, device="npu", dtype=torch.float32)
        b = torch.randn(size, device="npu", dtype=torch.float32)
        ref_c = ref_program(a, b)

        for mode in ["naive", "persistent", "optimized", "pipelined"]:
            c = vector_add(a, b, mode=mode)
            torch.testing.assert_close(c.cpu(), ref_c.cpu(), rtol=1e-6, atol=1e-6)
            print(f"  size={size:>8}, mode={mode:>12}: PASS")

    print("\nAll correctness checks passed!")


if __name__ == "__main__":
    main()
