"""
Triton-Ascend 向量加法 (Vector Add)

三种实现模式：
1. naive:    GPU 风格，每个 program 一段连续元素，grid 数量远超物理核数
2. persistent: 固定核数 + tl.range 跨步分配（解决了调度问题，但循环不是 constexpr）
3. optimized: 官方推荐 XBLOCK/XBLOCK_SUB 双层切分 + constexpr 循环 + 无 mask
   （编译器可展开循环 → 开启 multibuffer 存算并行流水线）

性能排序: optimized >> persistent ≈ naive >> ...

关键优化点:
- loops: tl.constexpr 使编译器能在编译期确定循环次数，展开并启用 multibuffer
- XBLOCK 核间切分 + XBLOCK_SUB 核内子块循环 = 三层切分
- mask=None 或 care_padding=False 避免填充同步开销
- grid = (ncore, 1, 1) 三维格式
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
# Persistent kernel (固定核数 + tl.range，但循环非 constexpr)
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
# Optimized kernel (官方 XBLOCK/XBLOCK_SUB 模式，constexpr 循环)
# ============================================================================

@triton.jit
def _vector_add_kernel_optimized(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    XBLOCK: tl.constexpr,
    XBLOCK_SUB: tl.constexpr,
):
    """
    官方推荐的 NPU 纯向量算子模式:
    - 核间切分: 每个 core 连续处理 XBLOCK 个元素
    - 核内子块: 每次 XBLOCK_SUB 个元素，循环处理
    - loops: tl.constexpr → 编译器可展开 → multibuffer 流水线生效
    - mask 仅在尾部不整除时触发
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
# Host 函数
# ============================================================================

def _get_num_vector_cores() -> int:
    """获取 NPU Vector Core 数量"""
    device = torch.npu.current_device()
    props = driver.active.utils.get_device_properties(device)
    return props["num_vectorcore"]


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
        mode: "naive" | "persistent" | "optimized"
        xblock_sub: optimized 模式的核内子块大小 (仅 optimized 模式)

    返回:
        输出向量 C，形状与输入相同
    """
    assert a.shape == b.shape, "Input shapes must match"
    n_elements = a.numel()
    c = torch.empty_like(a)

    if mode == "naive":
        block_size = 1024
        grid = (triton.cdiv(n_elements, block_size),)
        _vector_add_kernel_naive[grid](
            a, b, c, n_elements, BLOCK_SIZE=block_size,
        )

    elif mode == "persistent":
        num_cores = _get_num_vector_cores()
        block_size = 1024
        grid = (num_cores,)
        _vector_add_kernel_persistent[grid](
            a, b, c, n_elements,
            BLOCK_SIZE=block_size, NUM_CORES=num_cores,
        )

    elif mode == "optimized":
        num_cores = _get_num_vector_cores()
        xblock = triton.cdiv(n_elements, num_cores)
        grid = (num_cores, 1, 1)
        _vector_add_kernel_optimized[grid](
            a, b, c, n_elements,
            XBLOCK=xblock, XBLOCK_SUB=xblock_sub,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}, use 'naive', 'persistent', or 'optimized'")

    return c


def ref_program(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch 参考实现"""
    return a + b


def main():
    """主函数：正确性验证"""
    torch.npu.set_device(0)

    num_cores = _get_num_vector_cores()
    print(f"Vector Cores: {num_cores}")

    for size in [1024 * 64, 1024 * 1024]:
        a = torch.randn(size, device="npu", dtype=torch.float32)
        b = torch.randn(size, device="npu", dtype=torch.float32)
        ref_c = ref_program(a, b)

        for mode in ["naive", "persistent", "optimized"]:
            c = vector_add(a, b, mode=mode)
            torch.testing.assert_close(c.cpu(), ref_c.cpu(), rtol=1e-6, atol=1e-6)
            print(f"  size={size:>8}, mode={mode:>12}: PASS")

    print("\nAll correctness checks passed!")


if __name__ == "__main__":
    main()
