"""
Triton-Ascend GEMM 参考实现

基于标准 Triton 的矩阵乘法实现，使用 tl.dot 进行分块矩阵乘法，
通过 @triton.autotune 自动搜索最优 BLOCK_SIZE 配置。

关键设计:
- 2D grid: (M // BLOCK_M, N // BLOCK_N) 个 program
- K 维分块: 沿 K 维循环累加，使用 tl.dot 进行块矩阵乘法
- @triton.autotune: 自动测试多种 BLOCK 配置，选择最优
- 累加器: 使用 tl.zeros 初始化，支持 float32 累加精度

NPU 上的数据流:
  GM --(DMA)--> SRAM(block) --(tl.dot)--> accumulator --(tl.store)--> GM
"""

import triton
import triton.language as tl
import torch


# ============================================================================
# 自动调优配置
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32}),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _gemm_kernel(
    # 指针
    a_ptr, b_ptr, c_ptr,
    # 维度
    M, N, K,
    # 步长
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # BLOCK 大小（编译时常量）
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    GEMM kernel: C = A @ B

    每个 program 负责输出矩阵 C 中的一个 (BLOCK_M, BLOCK_N) 块。

    参数:
        a_ptr, b_ptr, c_ptr: 输入/输出张量指针
        M, N, K: 矩阵维度
        stride_*: 各维度的步长
        BLOCK_M, BLOCK_N, BLOCK_K: 分块大小
    """
    # 获取当前 program 负责的输出块位置
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # 计算当前块在 M 和 N 维度上的起始位置
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 初始化累加器（使用 float32 精度）
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K 维分块循环
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # 计算 A 矩阵块的偏移量: A[offs_m, k_start:k_end]
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        # 计算 B 矩阵块的偏移量: B[k_start:k_end, offs_n]
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # 加载 A 和 B 的分块，处理边界
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # 矩阵乘法并累加
        accumulator += tl.dot(a, b)

    # 将累加结果写回全局内存
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def gemm(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    GEMM 主机函数: C = A @ B

    参数:
        a: 输入矩阵 A，形状 (M, K)
        b: 输入矩阵 B，形状 (K, N)

    返回:
        输出矩阵 C，形状 (M, N)
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"

    M, K = a.shape
    K, N = b.shape

    # 分配输出张量
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # 计算 grid 大小
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    # 启动 kernel
    _gemm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )

    return c


def ref_program(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch 参考实现"""
    return a @ b


def main():
    """主函数：正确性验证"""
    torch.npu.set_device(0)

    M, N, K = 1024, 1024, 1024
    print(f"Problem size: M={M}, N={N}, K={K}")
    print(f"Device: NPU")

    a = torch.randn(M, K, device="npu", dtype=torch.float16)
    b = torch.randn(K, N, device="npu", dtype=torch.float16)

    c = gemm(a, b)
    ref_c = ref_program(a, b)
    torch.testing.assert_close(c.cpu(), ref_c.cpu(), rtol=1e-2, atol=1e-2)
    print("Correctness check passed!")

    print(f"\nInput A: {a.shape}, dtype={a.dtype}")
    print(f"Input B: {b.shape}, dtype={b.dtype}")
    print(f"Output C: {c.shape}, dtype={c.dtype}")


if __name__ == "__main__":
    main()
