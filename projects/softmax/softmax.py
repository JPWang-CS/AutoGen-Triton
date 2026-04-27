"""
Triton-Ascend Fused Softmax

融合 softmax 实现，对每一行独立计算:
  softmax(x_i) = exp(x_i - max(x)) / sum(exp(x - max(x)))

采用数值稳定的 "online softmax" 三遍扫描模式:
  1. 计算每行的最大值（避免 exp 溢出）
  2. 计算每行 exp(x - max) 的和
  3. 归一化: output = exp(x - max) / sum

设计要点:
- 1D grid: 每行一个 program
- BLOCK_SIZE >= N，每个 program 处理完整的一行
- 使用 tl.max 和 tl.sum 进行行级归约
"""

import triton
import triton.language as tl
import torch


@triton.jit
def _softmax_kernel(
    input_ptr, output_ptr,
    num_rows, num_cols,
    stride_in_row, stride_out_row,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused softmax kernel

    每个 program 处理矩阵的一行，计算该行的 softmax。

    参数:
        input_ptr: 输入指针
        output_ptr: 输出指针
        num_rows: 行数
        num_cols: 列数
        stride_in_row: 输入的行步长
        stride_out_row: 输出的行步长
        BLOCK_SIZE: 每个 program 处理的列数（需 >= num_cols）
    """
    # 获取当前 program 负责的行号
    row_idx = tl.program_id(axis=0)

    # 计算当前行的起始偏移
    row_start = row_idx * stride_in_row
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # 创建列方向 mask
    mask = col_offsets < num_cols

    # 加载一行数据
    row_data = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=-float("inf"))

    # 第1步: 计算行的最大值（数值稳定性）
    row_max = tl.max(row_data, axis=0)

    # 第2步: 计算 exp(x - max)
    numerator = tl.exp(row_data - row_max)

    # 第3步: 计算分母 sum(exp(x - max))
    denominator = tl.sum(numerator, axis=0)

    # 第4步: 归一化
    softmax_output = numerator / denominator

    # 写回结果
    out_row_start = row_idx * stride_out_row
    tl.store(output_ptr + out_row_start + col_offsets, softmax_output, mask=mask)


def softmax(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Fused softmax 主机函数

    对输入张量的最后一维执行 softmax 操作。

    参数:
        x: 输入张量，形状 (..., N)

    返回:
        输出张量，形状与输入相同
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"

    # 将输入重塑为 2D (num_rows, num_cols)
    original_shape = x.shape
    num_cols = x.shape[-1]
    num_rows = x.numel() // num_cols
    x_2d = x.reshape(num_rows, num_cols)

    # 分配输出张量
    output_2d = torch.empty_like(x_2d)

    # 选择 BLOCK_SIZE（需要 >= num_cols 且为 2 的幂）
    BLOCK_SIZE = triton.next_power_of_2(num_cols)

    # 启动 kernel，每行一个 program
    grid = (num_rows,)
    _softmax_kernel[grid](
        x_2d, output_2d,
        num_rows, num_cols,
        x_2d.stride(0), output_2d.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # 恢复原始形状
    return output_2d.reshape(original_shape)


def ref_program(x: torch.Tensor) -> torch.Tensor:
    """PyTorch 参考实现"""
    return torch.nn.functional.softmax(x, dim=-1)


def main():
    """主函数：正确性验证"""
    torch.npu.set_device(0)

    shape = (128, 512)
    print(f"Input shape: {shape}")
    print(f"Device: NPU")

    x = torch.randn(*shape, device="npu", dtype=torch.float32)

    output = softmax(x)
    ref_output = ref_program(x)
    torch.testing.assert_close(output.cpu(), ref_output.cpu(), rtol=1e-5, atol=1e-5)
    print("Correctness check passed!")

    print(f"\nInput: {x.shape}, dtype={x.dtype}")
    print(f"Output: {output.shape}, dtype={output.dtype}")

    # 验证输出行和为 1
    row_sums = output.cpu().sum(dim=-1)
    print(f"Row sums (should be ~1.0): {row_sums[:5]}")


if __name__ == "__main__":
    main()
