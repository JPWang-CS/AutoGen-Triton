"""
Triton-Ascend Layer Normalization

对输入的最后一维执行 Layer Normalization:
  y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta

计算步骤:
  1. 计算每行的均值: mean = sum(x) / N
  2. 计算每行的方差: var = sum((x - mean)^2) / N
  3. 归一化: norm = (x - mean) / sqrt(var + eps)
  4. 仿射变换: output = norm * gamma + beta

设计要点:
- 1D grid: 每行一个 program
- BLOCK_SIZE >= N，每个 program 处理完整的一行
- 使用 tl.sum 进行行级归约计算均值和方差
- 支持 gamma/beta 仿射参数（可选）
"""

import triton
import triton.language as tl
import torch


@triton.jit
def _layer_norm_kernel(
    input_ptr, gamma_ptr, beta_ptr, output_ptr,
    num_rows, num_cols,
    stride_in_row, stride_out_row,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Layer Normalization kernel

    每个 program 处理一行，计算该行的 layer norm。

    参数:
        input_ptr: 输入指针
        gamma_ptr: 缩放参数指针（可为 None）
        beta_ptr: 偏移参数指针（可为 None）
        output_ptr: 输出指针
        num_rows: 行数
        num_cols: 列数（归一化维度）
        stride_in_row: 输入的行步长
        stride_out_row: 输出的行步长
        eps: 防止除零的小常数
        BLOCK_SIZE: 处理的列数（需 >= num_cols）
    """
    # 获取当前 program 负责的行号
    row_idx = tl.program_id(axis=0)

    # 计算当前行的起始偏移
    row_start = row_idx * stride_in_row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < num_cols

    # 加载一行数据并转为 float32 进行计算
    row_data = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # 第1步: 计算均值
    mean = tl.sum(row_data, axis=0) / num_cols

    # 第2步: 计算方差
    deviation = row_data - mean
    variance = tl.sum(deviation * deviation, axis=0) / num_cols

    # 第3步: 归一化
    rstd = 1.0 / tl.sqrt(variance + eps)
    normalized = deviation * rstd

    # 第4步: 仿射变换（如果有 gamma/beta）
    if gamma_ptr is not None:
        gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=1.0)
        normalized = normalized * gamma

    if beta_ptr is not None:
        beta = tl.load(beta_ptr + col_offsets, mask=mask, other=0.0)
        normalized = normalized + beta

    # 写回结果
    out_row_start = row_idx * stride_out_row
    tl.store(output_ptr + out_row_start + col_offsets, normalized, mask=mask)


def layer_norm(
    x: torch.Tensor,
    gamma: torch.Tensor = None,
    beta: torch.Tensor = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Layer Normalization 主机函数

    对输入张量的最后一维执行 layer normalization。

    参数:
        x: 输入张量，形状 (..., N)
        gamma: 缩放参数，形状 (N,)，可选
        beta: 偏移参数，形状 (N,)，可选
        eps: 防止除零的小常数

    返回:
        输出张量，形状与输入相同
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"

    # 将输入重塑为 2D
    original_shape = x.shape
    num_cols = x.shape[-1]
    num_rows = x.numel() // num_cols
    x_2d = x.reshape(num_rows, num_cols)

    # 分配输出张量
    output_2d = torch.empty_like(x_2d)

    # 选择 BLOCK_SIZE
    BLOCK_SIZE = triton.next_power_of_2(num_cols)

    # 处理 gamma/beta 指针（使用 int 0 表示 None）
    gamma_ptr = gamma if gamma is not None else None
    beta_ptr = beta if beta is not None else None

    # 启动 kernel
    grid = (num_rows,)
    _layer_norm_kernel[grid](
        x_2d, gamma_ptr, beta_ptr, output_2d,
        num_rows, num_cols,
        x_2d.stride(0), output_2d.stride(0),
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output_2d.reshape(original_shape)


def ref_program(
    x: torch.Tensor,
    gamma: torch.Tensor = None,
    beta: torch.Tensor = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """PyTorch 参考实现"""
    if gamma is not None and beta is not None:
        return torch.nn.functional.layer_norm(
            x, [x.shape[-1]], weight=gamma, bias=beta, eps=eps
        )
    else:
        return torch.nn.functional.layer_norm(x, [x.shape[-1]], eps=eps)


def main():
    """主函数：正确性验证"""
    torch.npu.set_device(0)

    batch_size, hidden_dim = 128, 512
    print(f"Input shape: ({batch_size}, {hidden_dim})")
    print(f"Device: NPU")

    x = torch.randn(batch_size, hidden_dim, device="npu", dtype=torch.float32)
    gamma = torch.randn(hidden_dim, device="npu", dtype=torch.float32)
    beta = torch.randn(hidden_dim, device="npu", dtype=torch.float32)

    # 测试不带 gamma/beta
    output = layer_norm(x)
    ref_output = ref_program(x.cpu())
    torch.testing.assert_close(output.cpu(), ref_output, rtol=1e-4, atol=1e-4)
    print("Correctness check passed (no gamma/beta)!")

    # 测试带 gamma/beta
    output_affine = layer_norm(x, gamma, beta)
    ref_output_affine = ref_program(x.cpu(), gamma.cpu(), beta.cpu())
    torch.testing.assert_close(output_affine.cpu(), ref_output_affine, rtol=1e-4, atol=1e-4)
    print("Correctness check passed (with gamma/beta)!")


if __name__ == "__main__":
    main()
