"""
Triton-Ascend 算子性能基准测试模板

该文件提供了 Triton 算子性能测试的基础模板。
使用 triton.testing.perf_report 框架进行性能测试和可视化。

请根据实际算子需求修改以下内容：
1. 算子名称（your_op_name -> 实际算子名）
2. FLOPs 计算逻辑
3. 需要测试的 shape 范围
"""

import argparse
import torch
import triton
import triton.testing

from your_op_name import your_op


def ref_program(x: torch.Tensor) -> torch.Tensor:
    """PyTorch 参考实现（CPU 上执行）"""
    # TODO: 替换为实际的 PyTorch 操作（如 torch.nn.functional.xxx）
    return x


def calculate_flops(n_elements: int) -> int:
    """
    计算 FLOPs（每秒浮点运算数）

    对于 element-wise 操作，每个元素 1 次运算。
    请根据实际算子修改此函数。
    """
    return n_elements


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(12, 22, 1)],  # 4K to 4M elements
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="ms",
        plot_name="your-op-performance",
        args={},
    )
)
def benchmark(size, provider):
    """
    性能基准测试函数

    参数:
        size: 输入元素数
        provider: 'triton' 或 'torch'
    """
    x = torch.randn(size, device="npu", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: ref_program(x), quantiles=quantiles
        )
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: your_op(x), quantiles=quantiles
        )

    return ms, max_ms, min_ms


def run_single_benchmark(size: int, warmup: int = 10, rep: int = 100):
    """运行单次性能测试"""
    torch.npu.set_device(0)

    x = torch.randn(size, device="npu", dtype=torch.float32)

    # 正确性验证
    output = your_op(x)
    ref_output = ref_program(x.cpu())
    torch.testing.assert_close(output.cpu(), ref_output, rtol=1e-2, atol=1e-2)
    print("Correctness check passed!")

    # 性能测试
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: your_op(x), quantiles=quantiles, warmup=warmup, rep=rep
    )

    # 计算性能指标
    flops = calculate_flops(size)
    gflops = flops / ms * 1e-6

    # 打印结果
    print("\n" + "=" * 60)
    print("Triton-Ascend Benchmark Results:")
    print("=" * 60)
    print(f"Input size:         {size}")
    print(f"Triton Latency:     {ms:.3f} ms")
    print(f"GFLOPS:             {gflops:.2f}")
    print(f"Min Latency:        {min_ms:.3f} ms")
    print(f"Max Latency:        {max_ms:.3f} ms")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Triton-Ascend YourOpName Benchmark"
    )
    parser.add_argument(
        "--size", type=int, default=1024 * 1024,
        help="Input size (number of elements)"
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Warmup iterations"
    )
    parser.add_argument(
        "--rep", type=int, default=100, help="Repeat iterations"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate performance plot"
    )
    args = parser.parse_args()

    if args.plot:
        benchmark.run(print_data=True)
    else:
        run_single_benchmark(args.size, args.warmup, args.rep)


if __name__ == "__main__":
    main()
