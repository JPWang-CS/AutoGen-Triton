"""
Triton-Ascend 矩阵乘法性能基准测试

使用 triton.testing.perf_report 框架，对比 Triton kernel 与 PyTorch matmul。
性能指标: TFLOPS（每秒万亿次浮点运算）。
"""

import argparse
import torch
import triton
import triton.testing

from matmul import matmul


def ref_program(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch 参考实现"""
    return a @ b


def calculate_flops(M: int, N: int, K: int) -> int:
    """
    计算 GEMM 的 FLOPs

    C = A @ B, 其中 A 为 (M, K), B 为 (K, N)
    每个输出元素需要 2*K 次浮点运算（K 次乘法 + K-1 次加法）
    FLOPs = 2 * M * N * K
    """
    return 2 * M * N * K


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],
        x_vals=[128, 256, 512, 1024, 2048, 4096],
        x_log=False,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="TFLOPS",
        plot_name="matmul-performance",
        args={"N": 1024, "K": 1024},
    )
)
def benchmark(M, N, K, provider):
    """
    性能基准测试函数

    以 TFLOPS 作为性能指标。
    """
    a = torch.randn(M, K, device="npu", dtype=torch.float16)
    b = torch.randn(K, N, device="npu", dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: ref_program(a, b), quantiles=quantiles
        )
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul(a, b), quantiles=quantiles
        )

    # 计算 TFLOPS
    flops = calculate_flops(M, N, K)
    tflops = lambda ms: flops / (ms * 1e-3) * 1e-12
    return tflops(ms), tflops(max_ms), tflops(min_ms)


def run_single_benchmark(
    M: int, N: int, K: int,
    warmup: int = 10, rep: int = 100
):
    """运行单次性能测试"""
    torch.npu.set_device(0)

    a = torch.randn(M, K, device="npu", dtype=torch.float16)
    b = torch.randn(K, N, device="npu", dtype=torch.float16)

    # 正确性验证
    c = matmul(a, b)
    ref_c = ref_program(a, b)
    torch.testing.assert_close(c.cpu(), ref_c.cpu(), rtol=1e-2, atol=1e-2)
    print("Correctness check passed!")

    # Triton 性能测试
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: matmul(a, b), quantiles=quantiles, warmup=warmup, rep=rep
    )

    # PyTorch 性能测试
    ref_ms, _, _ = triton.testing.do_bench(
        lambda: ref_program(a, b), quantiles=quantiles, warmup=warmup, rep=rep
    )

    # 计算性能指标
    flops = calculate_flops(M, N, K)
    tflops = flops / (ms * 1e-3) * 1e-12
    ref_tflops = flops / (ref_ms * 1e-3) * 1e-12

    print("\n" + "=" * 60)
    print("Matmul Benchmark Results (NPU):")
    print("=" * 60)
    print(f"Problem size:       M={M}, N={N}, K={K}")
    print(f"Triton Latency:     {ms:.3f} ms")
    print(f"Triton TFLOPS:      {tflops:.2f}")
    print(f"PyTorch Latency:    {ref_ms:.3f} ms")
    print(f"PyTorch TFLOPS:     {ref_tflops:.2f}")
    print(f"Speedup:            {ref_ms / ms:.2f}x")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Triton-Ascend Matmul Benchmark"
    )
    parser.add_argument("--M", type=int, default=1024, help="Matrix M dimension")
    parser.add_argument("--N", type=int, default=1024, help="Matrix N dimension")
    parser.add_argument("--K", type=int, default=1024, help="Matrix K dimension")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--rep", type=int, default=100, help="Repeat iterations")
    parser.add_argument("--plot", action="store_true", help="Generate performance plot")
    args = parser.parse_args()

    if args.plot:
        benchmark.run(print_data=True)
    else:
        run_single_benchmark(args.M, args.N, args.K, args.warmup, args.rep)


if __name__ == "__main__":
    main()
