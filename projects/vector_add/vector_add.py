"""
Triton-Ascend 向量加法 (Vector Add)

两种实现模式：
1. naive: GPU 风格，每个 program 处理一段连续元素（grid 数量可能远超物理核数）
2. persistent: NPU 推荐模式，固定核数启动，跨步分配任务（tl.range）

设计要点:
- naive 模式: grid = (cdiv(n, BLOCK_SIZE),) — 大量 program，NPU 调度开销大
- persistent 模式: grid = (num_cores,) — 固定核数，每个核循环处理多个块
"""

import triton
import triton.language as tl
import triton.runtime.driver as driver
import torch


# ============================================================================
# Naive kernel (GPU 风格，NPU 上性能差)
# ============================================================================

@triton.jit
def _vector_add_kernel_naive(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """每个 program 处理一段连续元素 — GPU 思维"""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)


# ============================================================================
# Persistent kernel (NPU 推荐模式)
# ============================================================================

@triton.jit
def _vector_add_kernel_persistent(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    """
    持久化 kernel: 固定 NUM_CORES 个 program，跨步分配任务。

    每个 program 处理 pid, pid+NUM_CORES, pid+2*NUM_CORES, ... 的块。
    使用 tl.range 替代 Python range，让编译器优化循环。
    """
    pid = tl.program_id(axis=0)

    # tl.range: 编译器优化的跨步循环
    for block_id in tl.range(pid, tl.cdiv(n_elements, BLOCK_SIZE), NUM_CORES):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        c = a + b
        tl.store(c_ptr + offsets, c, mask=mask)


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
    block_size: int = 1024,
    persistent: bool = True,
) -> torch.Tensor:
    """
    向量加法主机函数: C = A + B

    参数:
        a: 输入向量 A，形状 (N,)
        b: 输入向量 B，形状 (N,)
        block_size: 每个 task 处理的元素数
        persistent: True 使用持久化模式（推荐），False 使用 naive 模式

    返回:
        输出向量 C，形状与输入相同
    """
    assert a.shape == b.shape, "Input shapes must match"
    n_elements = a.numel()
    c = torch.empty_like(a)

    if persistent:
        num_cores = _get_num_vector_cores()
        grid = (num_cores,)
        _vector_add_kernel_persistent[grid](
            a, b, c,
            n_elements,
            BLOCK_SIZE=block_size,
            NUM_CORES=num_cores,
        )
    else:
        grid = (triton.cdiv(n_elements, block_size),)
        _vector_add_kernel_naive[grid](
            a, b, c,
            n_elements,
            BLOCK_SIZE=block_size,
        )

    return c


def ref_program(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch 参考实现"""
    return a + b


def main():
    """主函数：正确性验证"""
    torch.npu.set_device(0)

    size = 1024 * 64
    print(f"Vector size: {size}")
    print(f"Device: NPU")

    num_cores = _get_num_vector_cores()
    print(f"Vector Cores: {num_cores}")

    a = torch.randn(size, device="npu", dtype=torch.float32)
    b = torch.randn(size, device="npu", dtype=torch.float32)

    # 测试 naive 模式
    c_naive = vector_add(a, b, persistent=False)
    ref_c = ref_program(a, b)
    torch.testing.assert_close(c_naive.cpu(), ref_c.cpu(), rtol=1e-6, atol=1e-6)
    print("Naive mode: Correctness check passed!")

    # 测试 persistent 模式
    c_persistent = vector_add(a, b, persistent=True)
    torch.testing.assert_close(c_persistent.cpu(), ref_c.cpu(), rtol=1e-6, atol=1e-6)
    print("Persistent mode: Correctness check passed!")

    print(f"\nInput A: {a.shape}, dtype={a.dtype}")
    print(f"Input B: {b.shape}, dtype={b.dtype}")
    print(f"Output C: {c_persistent.shape}, dtype={c_persistent.dtype}")


if __name__ == "__main__":
    main()
