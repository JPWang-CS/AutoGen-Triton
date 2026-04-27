"""
Triton-Ascend Layer Normalization 性能基准测试

使用 triton.testing.perf_report 框架，对比 Triton kernel 与 PyTorch layer_norm。
性能指标: 延迟 (ms) 和带宽 (GB/s)。
"""

import argparse
import torch
import triton
import triton.testing

from layer_norm import layer_norm


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


def calculate_bytes(num_rows: int, num_cols: int, has_affine: bool = False) -> int:
    """
    计算数据搬运量（字节）

    不带 affine: 读 1 次 + 写 1 次 = 2 * total * 4
    带 affine: 读 1 次 + 读 gamma/beta + 写 1 次 = 4 * total * 4
    """
    dtype_size = 4  # float32
    total = num_rows * num_cols
    if has_affine:
        # input + gamma + beta + output
        return (total + num_cols + num_cols + total) * dtype_size
    else:
        return 2 * total * dtype_size


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_cols"],
        x_vals=[128, 256, 512, 1024, 2048, 4096],
        x_log=False,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="ms",
        plot_name="layer-norm-performance",
        args={"num_rows": 1024},
    )
)
def benchmark(num_rows, num_cols, provider):
    """
    性能基准测试函数

    以延迟 (ms) 作为性能指标。
    """
    x = torch.randn(num_rows, num_cols, device="npu", dtype=torch.float32)
    gamma = torch.randn(num_cols, device="npu", dtype=torch.float32)
    beta = torch.randn(num_cols, device="npu", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: ref_program(x, gamma, beta), quantiles=quantiles
        )
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: layer_norm(x, gamma, beta), quantiles=quantiles
        )

    return ms, max_ms, min_ms


def run_single_benchmark(
    num_rows: int, num_cols: int,
    warmup: int = 10, rep: int = 100
):
    """运行单次性能测试"""
    torch.npu.set_device(0)

    x = torch.randn(num_rows, num_cols, device="npu", dtype=torch.float32)
    gamma = torch.randn(num_cols, device="npu", dtype=torch.float32)
    beta = torch.randn(num_cols, device="npu", dtype=torch.float32)

    # 正确性验证
    output = layer_norm(x, gamma, beta)
    ref_output = ref_program(x, gamma, beta)
    torch.testing.assert_close(output.cpu(), ref_output.cpu(), rtol=1e-4, atol=1e-4)
    print("Correctness check passed!")

    # Triton 性能测试
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: layer_norm(x, gamma, beta), quantiles=quantiles,
        warmup=warmup, rep=rep
    )

    # PyTorch 性能测试
    ref_ms, _, _ = triton.testing.do_bench(
        lambda: ref_program(x, gamma, beta), quantiles=quantiles,
        warmup=warmup, rep=rep
    )

    # 计算带宽
    total_bytes = calculate_bytes(num_rows, num_cols, has_affine=True)
    bandwidth = total_bytes / (ms * 1e-6) * 1e-9  # GB/s
    ref_bandwidth = total_bytes / (ref_ms * 1e-6) * 1e-9  # GB/s

    print("\n" + "=" * 60)
    print("Layer Normalization Benchmark Results (NPU):")
    print("=" * 60)
    print(f"Input shape:         ({num_rows}, {num_cols})")
    print(f"Triton Latency:      {ms:.3f} ms")
    print(f"Triton Bandwidth:    {bandwidth:.2f} GB/s")
    print(f"PyTorch Latency:     {ref_ms:.3f} ms")
    print(f"PyTorch Bandwidth:   {ref_bandwidth:.2f} GB/s")
    print(f"Speedup:             {ref_ms / ms:.2f}x")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Triton-Ascend Layer Normalization Benchmark"
    )
    parser.add_argument("--rows", type=int, default=1024, help="Number of rows")
    parser.add_argument("--cols", type=int, default=1024, help="Number of columns")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--rep", type=int, default=100, help="Repeat iterations")
    parser.add_argument("--plot", action="store_true", help="Generate performance plot")
    args = parser.parse_args()

    if args.plot:
        benchmark.run(print_data=True)
    else:
        run_single_benchmark(args.rows, args.cols, args.warmup, args.rep)


if __name__ == "__main__":
    main()
