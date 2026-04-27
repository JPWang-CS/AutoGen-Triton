"""
Triton-Ascend 向量加法性能基准测试

三种模式对比：
  - Triton Naive: GPU 风格 grid，可能远超物理核数
  - Triton Persistent: NPU 推荐的固定核数 + 跨步分配
  - PyTorch: 原生 a + b

性能指标:
  - 执行时间 (ms)
  - 带宽 (GB/s)
  - 加速比
  - 数值精度 (rtol=1e-3, atol=1e-3)
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
        line_vals=["naive", "persistent", "torch"],
        line_names=["Triton-Naive", "Triton-Persistent", "PyTorch"],
        styles=[("red", "-"), ("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="vector-add-performance",
        args={},
    )
)
def benchmark(size, provider):
    a = torch.randn(size, device="npu", dtype=torch.float32)
    b = torch.randn(size, device="npu", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: ref_program(a, b), quantiles=quantiles
        )
    elif provider == "naive":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: vector_add(a, b, persistent=False), quantiles=quantiles
        )
    elif provider == "persistent":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: vector_add(a, b, persistent=True), quantiles=quantiles
        )

    gbps = lambda ms: 3 * size * 4 / (ms * 1e-6) * 1e-9
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# ============================================================================
# 详细对比 benchmark (三路对比)
# ============================================================================

def run_comparison_benchmark(
    size: int,
    dtype: torch.dtype = torch.float32,
    warmup: int = 10,
    rep: int = 100,
):
    element_bytes = torch.tensor([], dtype=dtype).element_size()
    dtype_name = {
        torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16",
    }.get(dtype, str(dtype))

    a = torch.randn(size, device="npu", dtype=dtype)
    b = torch.randn(size, device="npu", dtype=dtype)

    # 数值精度 (persistent 模式)
    c_triton = vector_add(a, b, persistent=True)
    c_naive = vector_add(a, b, persistent=False)
    c_torch = ref_program(a, b)
    max_diff = torch.max(torch.abs(c_triton.cpu().float() - c_torch.cpu().float())).item()
    mean_diff = torch.mean(torch.abs(c_triton.cpu().float() - c_torch.cpu().float())).item()
    precision_pass = torch.allclose(
        c_triton.cpu().float(), c_torch.cpu().float(), rtol=1e-3, atol=1e-3
    )

    # 性能
    quantiles = [0.5, 0.2, 0.8]
    ms_naive, min_naive, max_naive = triton.testing.do_bench(
        lambda: vector_add(a, b, persistent=False), quantiles=quantiles, warmup=warmup, rep=rep
    )
    ms_persistent, min_persistent, max_persistent = triton.testing.do_bench(
        lambda: vector_add(a, b, persistent=True), quantiles=quantiles, warmup=warmup, rep=rep
    )
    ms_torch, min_torch, max_torch = triton.testing.do_bench(
        lambda: ref_program(a, b), quantiles=quantiles, warmup=warmup, rep=rep
    )

    bytes_moved = 3 * size * element_bytes
    bw_naive = bytes_moved / (ms_naive * 1e-6) * 1e-9
    bw_persistent = bytes_moved / (ms_persistent * 1e-6) * 1e-9
    bw_torch = bytes_moved / (ms_torch * 1e-6) * 1e-9
    speedup_naive = ms_torch / ms_naive
    speedup_persistent = ms_torch / ms_persistent

    data_size_mb = size * element_bytes / 1024 / 1024

    print("\n" + "=" * 80)
    print(f"  Vector Add Benchmark  |  size={size:,}  |  dtype={dtype_name}")
    print(f"  数据量: {data_size_mb:.2f} MB per tensor, 总搬运 {3 * data_size_mb:.2f} MB")
    print("=" * 80)

    print(f"\n  {'指标':<20} {'Naive':>12} {'Persistent':>12} {'PyTorch':>12} {'单位':>8}")
    print(f"  {'-'*64}")
    print(f"  {'中位数延迟':<20} {ms_naive:>12.4f} {ms_persistent:>12.4f} {ms_torch:>12.4f} {'ms':>8}")
    print(f"  {'最小延迟':<20} {min_naive:>12.4f} {min_persistent:>12.4f} {min_torch:>12.4f} {'ms':>8}")
    print(f"  {'最大延迟':<20} {max_naive:>12.4f} {max_persistent:>12.4f} {max_torch:>12.4f} {'ms':>8}")
    print(f"  {'带宽':<20} {bw_naive:>12.2f} {bw_persistent:>12.2f} {bw_torch:>12.2f} {'GB/s':>8}")

    print(f"\n  {'加速比 vs PyTorch':<25} Naive: {speedup_naive:.3f}x  |  Persistent: {speedup_persistent:.3f}x")
    if speedup_persistent > 1.0:
        print(f"  Persistent 比 PyTorch 快 {(speedup_persistent - 1) * 100:.1f}%")
    else:
        print(f"  Persistent 比 PyTorch 慢 {(1 - speedup_persistent) * 100:.1f}%")

    improvement = ms_naive / ms_persistent
    print(f"  Persistent vs Naive 提升: {improvement:.2f}x (固定核数 vs 大量 program)")

    print(f"\n  {'数值精度':<20} {'值':>12}")
    print(f"  {'-'*32}")
    print(f"  {'最大绝对误差':<20} {max_diff:>12.2e}")
    print(f"  {'平均绝对误差':<20} {mean_diff:>12.2e}")
    print(f"  {'精度要求':<20} {'rtol=1e-3, atol=1e-3':>12}")
    print(f"  {'是否通过':<20} {'PASS':>12}" if precision_pass else f"  {'是否通过':<20} {'FAIL':>12}")

    print("=" * 80)

    return {
        "size": size, "dtype": dtype_name,
        "ms_naive": ms_naive, "ms_persistent": ms_persistent, "ms_torch": ms_torch,
        "bw_naive": bw_naive, "bw_persistent": bw_persistent, "bw_torch": bw_torch,
        "speedup_naive": speedup_naive, "speedup_persistent": speedup_persistent,
        "max_diff": max_diff, "precision_pass": precision_pass,
    }


# ============================================================================
# Sweep benchmark (三路对比)
# ============================================================================

def run_sweep_benchmark(
    sizes=None,
    dtype: torch.dtype = torch.float32,
    warmup: int = 10,
    rep: int = 100,
):
    if sizes is None:
        sizes = [4 * 1024, 16 * 1024, 64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024]

    element_bytes = torch.tensor([], dtype=dtype).element_size()
    dtype_name = {
        torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16",
    }.get(dtype, str(dtype))

    print("\n" + "=" * 130)
    print(f"  Vector Add Sweep Benchmark  |  dtype={dtype_name}")
    print("=" * 130)
    print(f"  {'Size':>10} | {'Naive(ms)':>10} {'Persist(ms)':>11} {'Torch(ms)':>10} | "
          f"{'Naive(GB/s)':>11} {'Persist(GB/s)':>13} {'Torch(GB/s)':>11} | "
          f"{'SP-Naive':>8} {'SP-Pers':>8} | {'Diff':>10} | {'Pass':>5}")
    print(f"  {'-'*126}")

    results = []
    for size in sizes:
        a = torch.randn(size, device="npu", dtype=dtype)
        b = torch.randn(size, device="npu", dtype=dtype)

        c_triton = vector_add(a, b, persistent=True)
        c_torch = ref_program(a, b)
        max_diff = torch.max(torch.abs(c_triton.cpu().float() - c_torch.cpu().float())).item()
        precision_pass = torch.allclose(
            c_triton.cpu().float(), c_torch.cpu().float(), rtol=1e-3, atol=1e-3
        )

        ms_naive, _, _ = triton.testing.do_bench(
            lambda: vector_add(a, b, persistent=False), quantiles=[0.5, 0.2, 0.8], warmup=warmup, rep=rep
        )
        ms_persistent, _, _ = triton.testing.do_bench(
            lambda: vector_add(a, b, persistent=True), quantiles=[0.5, 0.2, 0.8], warmup=warmup, rep=rep
        )
        ms_torch, _, _ = triton.testing.do_bench(
            lambda: ref_program(a, b), quantiles=[0.5, 0.2, 0.8], warmup=warmup, rep=rep
        )

        bytes_moved = 3 * size * element_bytes
        bw_naive = bytes_moved / (ms_naive * 1e-6) * 1e-9
        bw_persistent = bytes_moved / (ms_persistent * 1e-6) * 1e-9
        bw_torch = bytes_moved / (ms_torch * 1e-6) * 1e-9
        sp_naive = ms_torch / ms_naive
        sp_persistent = ms_torch / ms_persistent

        size_str = f"{size//1024}K" if size < 1024*1024 else f"{size//1024//1024}M"
        pass_str = "PASS" if precision_pass else "FAIL"

        print(f"  {size_str:>10} | {ms_naive:>10.4f} {ms_persistent:>11.4f} {ms_torch:>10.4f} | "
              f"{bw_naive:>11.2f} {bw_persistent:>13.2f} {bw_torch:>11.2f} | "
              f"{sp_naive:>7.3f}x {sp_persistent:>7.3f}x | {max_diff:>10.2e} | {pass_str:>5}")
        results.append({
            "size": size, "ms_naive": ms_naive, "ms_persistent": ms_persistent, "ms_torch": ms_torch,
            "speedup_naive": sp_naive, "speedup_persistent": sp_persistent,
            "max_diff": max_diff, "pass": precision_pass,
        })

    print("=" * 130)
    all_pass = all(r["pass"] for r in results)
    avg_sp_naive = sum(r["speedup_naive"] for r in results) / len(results)
    avg_sp_persistent = sum(r["speedup_persistent"] for r in results) / len(results)
    print(f"\n  精度检查 (rtol=1e-3, atol=1e-3): {'ALL PASS' if all_pass else 'HAS FAIL'}")
    print(f"  平均加速比 vs PyTorch:  Naive={avg_sp_naive:.3f}x  |  Persistent={avg_sp_persistent:.3f}x")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Triton-Ascend Vector Add Benchmark (Naive vs Persistent vs PyTorch)"
    )
    parser.add_argument(
        "--size", type=int, default=1024 * 1024,
        help="Vector size (number of elements)"
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
