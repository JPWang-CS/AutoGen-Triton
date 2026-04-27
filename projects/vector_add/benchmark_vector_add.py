"""
Triton-Ascend 向量加法性能基准测试

使用 triton.testing.perf_report 框架进行性能测试和可视化。
对比 Triton kernel 与 PyTorch 原生实现的延迟。
"""

import argparse
import torch
import triton
import triton.testing

from vector_add import vector_add


def ref_program(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch 参考实现"""
    return a + b


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(12, 24, 1)],  # 4K to 16M elements
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="vector-add-performance",
        args={},
    )
)
def benchmark(size, provider):
    """
    性能基准测试函数

    使用带宽 (GB/s) 作为性能指标，因为向量加法是内存带宽受限操作。
    """
    a = torch.randn(size, device="npu", dtype=torch.float32)
    b = torch.randn(size, device="npu", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: ref_program(a, b), quantiles=quantiles
        )
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: vector_add(a, b), quantiles=quantiles
        )

    # 计算带宽: 读 2 个输入 + 写 1 个输出 = 3 * size * 4 bytes
    gbps = lambda ms: 3 * size * 4 / (ms * 1e-6) * 1e-9
    return gbps(ms), gbps(max_ms), gbps(min_ms)


def run_single_benchmark(size: int, warmup: int = 10, rep: int = 100):
    """运行单次性能测试"""
    torch.npu.set_device(0)

    a = torch.randn(size, device="npu", dtype=torch.float32)
    b = torch.randn(size, device="npu", dtype=torch.float32)

    # 正确性验证
    c = vector_add(a, b)
    ref_c = ref_program(a, b)
    torch.testing.assert_close(c.cpu(), ref_c.cpu(), rtol=1e-6, atol=1e-6)
    print("Correctness check passed!")

    # 性能测试
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: vector_add(a, b), quantiles=quantiles, warmup=warmup, rep=rep
    )

    # 计算带宽
    bytes_moved = 3 * size * 4  # 2 reads + 1 write, float32
    bandwidth = bytes_moved / (ms * 1e-6) * 1e-9  # GB/s

    print("\n" + "=" * 60)
    print("Vector Add Benchmark Results (NPU):")
    print("=" * 60)
    print(f"Vector size:        {size} ({size * 4 / 1024 / 1024:.1f} MB)")
    print(f"Triton Latency:     {ms:.3f} ms")
    print(f"Bandwidth:          {bandwidth:.2f} GB/s")
    print(f"Min Latency:        {min_ms:.3f} ms")
    print(f"Max Latency:        {max_ms:.3f} ms")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Triton-Ascend Vector Add Benchmark"
    )
    parser.add_argument(
        "--size", type=int, default=1024 * 1024,
        help="Vector size (number of elements)"
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
