"""
Triton-Ascend 空算子模板

基于标准 Triton 编程模型，适配华为昇腾 NPU。
使用 @triton.jit 装饰器定义 kernel，通过 tl.load/tl.store 进行内存操作。

关键API:
- @triton.jit: kernel JIT 编译装饰器
- tl.program_id(axis): 获取并行 program ID
- tl.arange(0, BLOCK_SIZE): 生成偏移量向量
- tl.load / tl.store: 内存读写操作（支持 mask）
- triton.cdiv: 向上取整除法，用于计算 grid 大小
"""

import triton
import triton.language as tl
import torch


@triton.jit
def _your_op_kernel(
    x_ptr, y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    空算子 kernel 模板

    每个 program 处理一段连续的元素，通过 program_id 计算偏移量。
    这是一个 element-wise 操作模板，可根据需要修改计算逻辑。

    参数:
        x_ptr: 输入指针
        y_ptr: 输出指针
        n_elements: 元素总数
        BLOCK_SIZE: 每个 program 处理的元素数（编译时常量）
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # TODO: 在此添加计算逻辑
    output = x
    tl.store(y_ptr + offsets, output, mask=mask)


def your_op(x: torch.Tensor) -> torch.Tensor:
    """
    空算子的主机函数

    负责分配输出张量、计算 grid 大小并启动 kernel。

    参数:
        x: 输入张量，形状任意

    返回:
        输出张量，与输入形状相同
    """
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _your_op_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output


def ref_program(x: torch.Tensor) -> torch.Tensor:
    """PyTorch 参考实现，用于正确性验证"""
    # TODO: 根据实际计算逻辑修改参考实现
    return x


def main():
    """简单的主函数，用于快速验证"""
    torch.npu.set_device(0)

    x = torch.randn(1024, device="npu", dtype=torch.float32)
    output = your_op(x)
    ref = ref_program(x)
    torch.testing.assert_close(output.cpu(), ref.cpu(), rtol=1e-2, atol=1e-2)
    print("Correctness check passed!")

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Device: {output.device}")


if __name__ == "__main__":
    main()
