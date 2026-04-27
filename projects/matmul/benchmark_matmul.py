"""
Triton-Ascend 矩阵乘法性能基准测试

对比 Triton kernel 与 PyTorch matmul：
  - 执行时间 (ms)
  - TFLOPS (每秒万亿次浮点运算)
  - 加速比 (Triton / PyTorch)
  - 数值精度 (是否通过 rtol=1e-3, atol=1e-3)
"""

import argparse
import torch
import torch_npu
import triton
import triton.testing

from matmul import matmul


def ref_program(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch 参考实现"""
    return a @ b


def calculate_flops(M: int, N: int, K: int) -> int:
    """GEMM FLOPs = 2 * M * N * K"""
    return 2 * M * N * K


# ============================================================================
# perf_report 可视化 benchmark
# ============================================================================

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

    flops = calculate_flops(M, N, K)
    tflops = lambda ms: flops / (ms * 1e-3) * 1e-12
    return tflops(ms), tflops(max_ms), tflops(min_ms)


# ============================================================================
# 详细对比 benchmark
# ============================================================================

def run_comparison_benchmark(
    M: int, N: int, K: int,
    dtype: torch.dtype = torch.float16,
    warmup: int = 10, rep: int = 100,
):
    element_bytes = torch.tensor([], dtype=dtype).element_size()
    dtype_name = {
        torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16",
    }.get(dtype, str(dtype))

    a = torch.randn(M, K, device="npu", dtype=dtype)
    b = torch.randn(K, N, device="npu", dtype=dtype)

    # 数值精度
    c_triton = matmul(a, b)
    c_torch = ref_program(a, b)
    max_diff = torch.max(torch.abs(c_triton.cpu().float() - c_torch.cpu().float())).item()
    mean_diff = torch.mean(torch.abs(c_triton.cpu().float() - c_torch.cpu().float())).item()
    rtol, atol = 1e-3, 1e-3
    precision_pass = torch.allclose(
        c_triton.cpu().float(), c_torch.cpu().float(), rtol=rtol, atol=atol
    )

    # 性能
    quantiles = [0.5, 0.2, 0.8]
    ms_triton, min_triton, max_triton = triton.testing.do_bench(
        lambda: matmul(a, b), quantiles=quantiles, warmup=warmup, rep=rep
    )
    ms_torch, min_torch, max_torch = triton.testing.do_bench(
        lambda: ref_program(a, b), quantiles=quantiles, warmup=warmup, rep=rep
    )

    flops = calculate_flops(M, N, K)
    tflops_triton = flops / (ms_triton * 1e-3) * 1e-12
    tflops_torch = flops / (ms_torch * 1e-3) * 1e-12
    speedup = ms_torch / ms_triton

    data_mb = (M * K + K * N + M * N) * element_bytes / 1024 / 1024

    print("\n" + "=" * 70)
    print(f"  Matmul Benchmark  |  {M}x{K} @ {K}x{N}  |  dtype={dtype_name}")
    print(f"  数据量: {data_mb:.2f} MB total")
    print("=" * 70)
    print(f"\n  {'指标':<20} {'Triton':>12} {'PyTorch':>12} {'单位':>8}")
    print(f"  {'-'*52}")
    print(f"  {'中位数延迟':<20} {ms_triton:>12.4f} {ms_torch:>12.4f} {'ms':>8}")
    print(f"  {'最小延迟':<20} {min_triton:>12.4f} {min_torch:>12.4f} {'ms':>8}")
    print(f"  {'最大延迟':<20} {max_triton:>12.4f} {max_torch:>12.4f} {'ms':>8}")
    print(f"  {'TFLOPS':<20} {tflops_triton:>12.2f} {tflops_torch:>12.2f} {'TFLOPS':>8}")
    print(f"\n  {'加速比 (Triton/PyTorch)':<30} {speedup:.3f}x")
    if speedup > 1.0:
        print(f"  Triton 比 PyTorch 快 {(speedup - 1) * 100:.1f}%")
    else:
        print(f"  Triton 比 PyTorch 慢 {(1 - speedup) * 100:.1f}%")
    print(f"\n  {'数值精度':<20} {'值':>12}")
    print(f"  {'-'*32}")
    print(f"  {'最大绝对误差':<20} {max_diff:>12.2e}")
    print(f"  {'平均绝对误差':<20} {mean_diff:>12.2e}")
    print(f"  {'精度要求':<20} {'rtol=1e-3, atol=1e-3':>12}")
    print(f"  {'是否通过':<20} {'PASS':>12}" if precision_pass else f"  {'是否通过':<20} {'FAIL':>12}")
    print("=" * 70)

    return {
        "shape": (M, N, K), "ms_triton": ms_triton, "ms_torch": ms_torch,
        "tflops_triton": tflops_triton, "tflops_torch": tflops_torch,
        "speedup": speedup, "max_diff": max_diff, "precision_pass": precision_pass,
    }


def run_sweep_benchmark(dtype: torch.dtype = torch.float16, warmup: int = 10, rep: int = 100):
    sizes = [(128, 128, 128), (256, 256, 256), (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]
    element_bytes = torch.tensor([], dtype=dtype).element_size()
    dtype_name = {torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16"}.get(dtype, str(dtype))

    print("\n" + "=" * 110)
    print(f"  Matmul Sweep Benchmark  |  dtype={dtype_name}")
    print("=" * 110)
    print(f"  {'Shape':>16} | {'Triton(ms)':>10} {'Torch(ms)':>10} | "
          f"{'Triton(TF)':>10} {'Torch(TF)':>10} | {'Speedup':>8} | {'MaxDiff':>10} | {'Pass':>5}")
    print(f"  {'-'*106}")

    results = []
    for M, N, K in sizes:
        a = torch.randn(M, K, device="npu", dtype=dtype)
        b = torch.randn(K, N, device="npu", dtype=dtype)
        c_triton = matmul(a, b)
        c_torch = ref_program(a, b)
        max_diff = torch.max(torch.abs(c_triton.cpu().float() - c_torch.cpu().float())).item()
        precision_pass = torch.allclose(c_triton.cpu().float(), c_torch.cpu().float(), rtol=1e-3, atol=1e-3)

        ms_triton, _, _ = triton.testing.do_bench(lambda: matmul(a, b), quantiles=[0.5, 0.2, 0.8], warmup=warmup, rep=rep)
        ms_torch, _, _ = triton.testing.do_bench(lambda: ref_program(a, b), quantiles=[0.5, 0.2, 0.8], warmup=warmup, rep=rep)
        flops = calculate_flops(M, N, K)
        tf_triton = flops / (ms_triton * 1e-3) * 1e-12
        tf_torch = flops / (ms_torch * 1e-3) * 1e-12
        speedup = ms_torch / ms_triton
        pass_str = "PASS" if precision_pass else "FAIL"

        print(f"  {M:>5}x{K:>5}x{N:<5} | {ms_triton:>10.4f} {ms_torch:>10.4f} | "
              f"{tf_triton:>10.2f} {tf_torch:>10.2f} | {speedup:>7.3f}x | {max_diff:>10.2e} | {pass_str:>5}")
        results.append({"shape": (M,N,K), "speedup": speedup, "max_diff": max_diff, "pass": precision_pass})

    print("=" * 110)
    all_pass = all(r["pass"] for r in results)
    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    print(f"\n  精度检查 (rtol=1e-3, atol=1e-3): {'ALL PASS' if all_pass else 'HAS FAIL'}")
    print(f"  平均加速比: {avg_speedup:.3f}x")
    return results


def main():
    parser = argparse.ArgumentParser(description="Triton-Ascend Matmul Benchmark")
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--sweep", action="store_true", help="Run sweep benchmark")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    if args.plot:
        benchmark.run(print_data=True)
    elif args.sweep:
        run_sweep_benchmark(dtype=dtype, warmup=args.warmup, rep=args.rep)
    else:
        run_comparison_benchmark(args.M, args.N, args.K, dtype=dtype, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    main()
