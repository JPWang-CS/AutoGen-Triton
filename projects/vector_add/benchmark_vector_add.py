"""
Triton-Ascend 向量加法性能基准测试

使用 triton.testing.perf_report 框架进行性能测试。
对比 Triton kernel 与 PyTorch 原生实现：
  - 执行时间 (ms)
  - 带宽 (GB/s)
  - 加速比 (Triton / PyTorch)
  - 数值精度 (是否通过 rtol=1e-3, atol=1e-3)
"""

import argparse
import torch
import torch_npu
import triton
import triton.testing

from vector_add import vector_add


def ref_program(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch 参考实现"""
    return a + b


# ============================================================================
# perf_report 可视化 benchmark
# ============================================================================

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
    性能基准测试函数（带宽指标）
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

    # 带宽: 读 2 个输入 + 写 1 个输出 = 3 * size * 4 bytes (float32)
    gbps = lambda ms: 3 * size * 4 / (ms * 1e-6) * 1e-9
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# ============================================================================
# 详细对比 benchmark
# ============================================================================

def run_comparison_benchmark(
    size: int,
    dtype: torch.dtype = torch.float32,
    warmup: int = 10,
    rep: int = 100,
):
    """
    运行 Triton vs PyTorch 详细对比测试，输出：
      - 执行时间
      - 带宽
      - 加速比
      - 数值精度
    """
    element_bytes = torch.tensor([], dtype=dtype).element_size()
    dtype_name = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }.get(dtype, str(dtype))

    # 创建输入
    a = torch.randn(size, device="npu", dtype=dtype)
    b = torch.randn(size, device="npu", dtype=dtype)

    # ---------- 数值精度验证 ----------
    c_triton = vector_add(a, b)
    c_torch = ref_program(a, b)

    max_diff = torch.max(torch.abs(c_triton.cpu().float() - c_torch.cpu().float())).item()
    mean_diff = torch.mean(torch.abs(c_triton.cpu().float() - c_torch.cpu().float())).item()

    # 精度检查: rtol=1e-3, atol=1e-3
    rtol, atol = 1e-3, 1e-3
    precision_pass = torch.allclose(
        c_triton.cpu().float(), c_torch.cpu().float(), rtol=rtol, atol=atol
    )

    # ---------- 性能测试 ----------
    quantiles = [0.5, 0.2, 0.8]

    ms_triton, min_triton, max_triton = triton.testing.do_bench(
        lambda: vector_add(a, b), quantiles=quantiles, warmup=warmup, rep=rep
    )
    ms_torch, min_torch, max_torch = triton.testing.do_bench(
        lambda: ref_program(a, b), quantiles=quantiles, warmup=warmup, rep=rep
    )

    # 带宽: 2 读 + 1 写
    bytes_moved = 3 * size * element_bytes
    bw_triton = bytes_moved / (ms_triton * 1e-6) * 1e-9  # GB/s
    bw_torch = bytes_moved / (ms_torch * 1e-6) * 1e-9

    # 加速比
    speedup = ms_torch / ms_triton

    # ---------- 输出结果 ----------
    data_size_mb = size * element_bytes / 1024 / 1024

    print("\n" + "=" * 70)
    print(f"  Vector Add Benchmark  |  size={size:,}  |  dtype={dtype_name}")
    print(f"  数据量: {data_size_mb:.2f} MB per tensor, 总搬运 {3 * data_size_mb:.2f} MB")
    print("=" * 70)

    # 执行时间
    print(f"\n  {'指标':<20} {'Triton':>12} {'PyTorch':>12} {'单位':>8}")
    print(f"  {'-'*52}")
    print(f"  {'中位数延迟':<20} {ms_triton:>12.4f} {ms_torch:>12.4f} {'ms':>8}")
    print(f"  {'最小延迟':<20} {min_triton:>12.4f} {min_torch:>12.4f} {'ms':>8}")
    print(f"  {'最大延迟':<20} {max_triton:>12.4f} {max_torch:>12.4f} {'ms':>8}")
    print(f"  {'带宽':<20} {bw_triton:>12.2f} {bw_torch:>12.2f} {'GB/s':>8}")

    # 加速比
    print(f"\n  {'加速比 (Triton/PyTorch)':<30} {speedup:.3f}x")
    if speedup > 1.0:
        print(f"  Triton 比 PyTorch 快 {(speedup - 1) * 100:.1f}%")
    else:
        print(f"  Triton 比 PyTorch 慢 {(1 - speedup) * 100:.1f}%")

    # 数值精度
    print(f"\n  {'数值精度':<20} {'值':>12}")
    print(f"  {'-'*32}")
    print(f"  {'最大绝对误差':<20} {max_diff:>12.2e}")
    print(f"  {'平均绝对误差':<20} {mean_diff:>12.2e}")
    print(f"  {'精度要求':<20} {'rtol=1e-3, atol=1e-3':>12}")
    print(f"  {'是否通过':<20} {'PASS':>12}" if precision_pass else f"  {'是否通过':<20} {'FAIL':>12}")

    print("=" * 70)

    return {
        "size": size,
        "dtype": dtype_name,
        "ms_triton": ms_triton,
        "ms_torch": ms_torch,
        "bw_triton": bw_triton,
        "bw_torch": bw_torch,
        "speedup": speedup,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "precision_pass": precision_pass,
    }


def run_sweep_benchmark(
    sizes=None,
    dtype: torch.dtype = torch.float32,
    warmup: int = 10,
    rep: int = 100,
):
    """
    多规模扫描测试，汇总输出对比表格
    """
    if sizes is None:
        sizes = [4 * 1024, 16 * 1024, 64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024]

    element_bytes = torch.tensor([], dtype=dtype).element_size()
    dtype_name = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }.get(dtype, str(dtype))

    print("\n" + "=" * 100)
    print(f"  Vector Add Sweep Benchmark  |  dtype={dtype_name}")
    print("=" * 100)
    print(f"  {'Size':>10} | {'Triton(ms)':>10} {'Torch(ms)':>10} | "
          f"{'Triton(GB/s)':>12} {'Torch(GB/s)':>12} | "
          f"{'Speedup':>8} | {'MaxDiff':>10} | {'Pass':>5}")
    print(f"  {'-'*96}")

    results = []
    for size in sizes:
        a = torch.randn(size, device="npu", dtype=dtype)
        b = torch.randn(size, device="npu", dtype=dtype)

        # 精度
        c_triton = vector_add(a, b)
        c_torch = ref_program(a, b)
        max_diff = torch.max(torch.abs(c_triton.cpu().float() - c_torch.cpu().float())).item()
        precision_pass = torch.allclose(
            c_triton.cpu().float(), c_torch.cpu().float(), rtol=1e-3, atol=1e-3
        )

        # 性能
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: vector_add(a, b), quantiles=[0.5, 0.2, 0.8], warmup=warmup, rep=rep
        )
        ms_torch, _, _ = triton.testing.do_bench(
            lambda: ref_program(a, b), quantiles=[0.5, 0.2, 0.8], warmup=warmup, rep=rep
        )

        bytes_moved = 3 * size * element_bytes
        bw_triton = bytes_moved / (ms_triton * 1e-6) * 1e-9
        bw_torch = bytes_moved / (ms_torch * 1e-6) * 1e-9
        speedup = ms_torch / ms_triton

        size_str = f"{size//1024}K" if size < 1024*1024 else f"{size//1024//1024}M"
        pass_str = "PASS" if precision_pass else "FAIL"

        print(f"  {size_str:>10} | {ms_triton:>10.4f} {ms_torch:>10.4f} | "
              f"{bw_triton:>12.2f} {bw_torch:>12.2f} | "
              f"{speedup:>7.3f}x | {max_diff:>10.2e} | {pass_str:>5}")

        results.append({
            "size": size, "ms_triton": ms_triton, "ms_torch": ms_torch,
            "bw_triton": bw_triton, "bw_torch": bw_torch,
            "speedup": speedup, "max_diff": max_diff, "pass": precision_pass,
        })

    print("=" * 100)

    # 汇总
    all_pass = all(r["pass"] for r in results)
    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    print(f"\n  精度检查 (rtol=1e-3, atol=1e-3): {'ALL PASS' if all_pass else 'HAS FAIL'}")
    print(f"  平均加速比: {avg_speedup:.3f}x")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Triton-Ascend Vector Add Benchmark"
    )
    parser.add_argument(
        "--size", type=int, default=1024 * 1024,
        help="Vector size (number of elements), 用于单次对比测试"
    )
    parser.add_argument(
        "--dtype", type=str, default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="数据类型"
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="运行多规模扫描测试"
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

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    if args.plot:
        benchmark.run(print_data=True)
    elif args.sweep:
        run_sweep_benchmark(dtype=dtype, warmup=args.warmup, rep=args.rep)
    else:
        run_comparison_benchmark(args.size, dtype=dtype, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    main()
