"""
Triton-Ascend 向量加法 (Vector Add)

最基础的 Triton 算子实现：C = A + B。
每个 program 处理一段连续元素，通过 tl.load 读取输入，
执行加法运算，再通过 tl.store 写入输出。

设计要点:
- 1D grid: (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE 个 program
- 每个 program 处理 BLOCK_SIZE 个元素
- 使用 mask 处理尾部不完整的块
"""

import triton
import triton.language as tl
import torch


@triton.jit
def _vector_add_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    向量加法 kernel: C[i] = A[i] + B[i]

    参数:
        a_ptr: 输入向量 A 的指针
        b_ptr: 输入向量 B 的指针
        c_ptr: 输出向量 C 的指针
        n_elements: 元素总数
        BLOCK_SIZE: 每个 program 处理的元素数
    """
    # 获取当前 program 的 ID
    pid = tl.program_id(axis=0)

    # 计算当前 program 负责的元素偏移量
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 创建 mask，处理尾部不完整的块
    mask = offsets < n_elements

    # 从全局内存加载数据
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    # 执行加法运算
    c = a + b

    # 将结果写回全局内存
    tl.store(c_ptr + offsets, c, mask=mask)


def vector_add(
    a: torch.Tensor,
    b: torch.Tensor,
    block_size: int = 1024,
) -> torch.Tensor:
    """
    向量加法主机函数: C = A + B

    参数:
        a: 输入向量 A，形状 (N,)
        b: 输入向量 B，形状 (N,)
        block_size: 每个 program 处理的元素数

    返回:
        输出向量 C，形状与输入相同
    """
    assert a.shape == b.shape, "Input shapes must match"
    n_elements = a.numel()

    # 分配输出张量
    c = torch.empty_like(a)

    # 计算 grid 大小
    grid = (triton.cdiv(n_elements, block_size),)

    # 启动 kernel
    _vector_add_kernel[grid](
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

    a = torch.randn(size, device="npu", dtype=torch.float32)
    b = torch.randn(size, device="npu", dtype=torch.float32)

    c = vector_add(a, b)
    ref_c = ref_program(a, b)
    torch.testing.assert_close(c.cpu(), ref_c.cpu(), rtol=1e-6, atol=1e-6)
    print("Correctness check passed!")

    print(f"\nInput A: {a.shape}, dtype={a.dtype}")
    print(f"Input B: {b.shape}, dtype={b.dtype}")
    print(f"Output C: {c.shape}, dtype={c.dtype}")


if __name__ == "__main__":
    main()
