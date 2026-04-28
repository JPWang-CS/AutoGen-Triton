"""
Triton-Ascend 向量加法性能基准测试

四路对比：
  - Naive:    GPU 风格 grid，大量 program
  - Persistent: 固定核数 tl.range 跨步
  - Optimized: XBLOCK/XBLOCK_SUB constexpr 循环 (官方推荐)
  - PyTorch:  原生 a + b

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
    return a + b


# ============================================================================
# perf_report 可视化
# ============================================================================

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(12, 24, 1)],
        x_log=True,
        line_arg="provider",
        line_vals=["naive", "optimized", "torch"],
        line_names=["Triton-Naive", "Triton-Optimized", "PyTorch"],
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
            lambda: vector_add(a, b, mode="naive"), quantiles=quantiles
        )
    elif provider == "optimized":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: vector_add(a, b, mode="optimized"), quantiles=quantiles
        )

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
    element_bytes = torch.tensor([], dtype=dtype).element_size()
    dtype_name = {
        torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16",
    }.get(dtype, str(dtype))

    a = torch.randn(size, device="npu", dtype=dtype)
    b = torch.randn(size, device="npu", dtype=dtype)

    # 数值精度
    c_opt = vector_add(a, b, mode="optimized")
    c_torch = ref_program(a, b)
    max_diff = torch.max(torch.abs(c_opt.cpu().float() - c_torch.cpu().float())).item()
    mean_diff = torch.mean(torch.abs(c_opt.cpu().float() - c_torch.cpu().float())).item()
    precision_pass = torch.allclose(
        c_opt.cpu().float(), c_torch.cpu().float(), rtol=1e-3, atol=1e-3
    )

    # 性能
    quantiles = [0.5, 0.2, 0.8]
    ms_naive, _, _ = triton.testing.do_bench(
        lambda: vector_add(a, b, mode="naive"), quantiles=quantiles, warmup=warmup, rep=rep
    )
    ms_persistent, _, _ = triton.testing.do_bench(
        lambda: vector_add(a, b, mode="persistent"), quantiles=quantiles, warmup=warmup, rep=rep
    )
    ms_optimized, min_opt, max_opt = triton.testing.do_bench(
        lambda: vector_add(a, b, mode="optimized"), quantiles=quantiles, warmup=warmup, rep=rep
    )
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: ref_program(a, b), quantiles=quantiles, warmup=warmup, rep=rep
    )

    bytes_moved = 3 * size * element_bytes
    bw_naive = bytes_moved / (ms_naive * 1e-6) * 1e-9
    bw_persistent = bytes_moved / (ms_persistent * 1e-6) * 1e-9
    bw_optimized = bytes_moved / (ms_optimized * 1e-6) * 1e-9
    bw_torch = bytes_moved / (ms_torch * 1e-6) * 1e-9

    sp_naive = ms_torch / ms_naive
    sp_persistent = ms_torch / ms_persistent
    sp_optimized = ms_torch / ms_optimized

    data_mb = size * element_bytes / 1024 / 1024

    print("\n" + "=" * 90)
    print(f"  Vector Add Benchmark  |  size={size:,}  |  dtype={dtype_name}")
    print(f"  数据量: {data_mb:.2f} MB per tensor, 总搬运 {3 * data_mb:.2f} MB")
    print("=" * 90)

    print(f"\n  {'指标':<16} {'Naive':>10} {'Persistent':>12} {'Optimized':>12} {'PyTorch':>10} {'单位':>6}")
    print(f"  {'-'*66}")
    print(f"  {'延迟(ms)':<16} {ms_naive:>10.4f} {ms_persistent:>12.4f} {ms_optimized:>12.4f} {ms_torch:>10.4f} {'ms':>6}")
    print(f"  {'带宽(GB/s)':<16} {bw_naive:>10.1f} {bw_persistent:>12.1f} {bw_optimized:>12.1f} {bw_torch:>10.1f} {'GB/s':>6}")

    print(f"\n  加速比 vs PyTorch:")
    print(f"    Naive:      {sp_naive:.3f}x")
    print(f"    Persistent: {sp_persistent:.3f}x")
    print(f"    Optimized:  {sp_optimized:.3f}x")

    print(f"\n  Optimized vs Naive:      {ms_naive/ms_optimized:.2f}x 提升")
    print(f"  Optimized vs Persistent: {ms_persistent/ms_optimized:.2f}x 提升")

    print(f"\n  {'数值精度':<20} {'值':>12}")
    print(f"  {'-'*32}")
    print(f"  {'最大绝对误差':<20} {max_diff:>12.2e}")
    print(f"  {'平均绝对误差':<20} {mean_diff:>12.2e}")
    print(f"  {'精度要求':<20} {'rtol=1e-3, atol=1e-3':>12}")
    print(f"  {'是否通过':<20} {'PASS':>12}" if precision_pass else f"  {'是否通过':<20} {'FAIL':>12}")
    print("=" * 90)

    return {
        "size": size, "ms_naive": ms_naive, "ms_persistent": ms_persistent,
        "ms_optimized": ms_optimized, "ms_torch": ms_torch,
        "sp_optimized": sp_optimized, "max_diff": max_diff, "precision_pass": precision_pass,
    }


# ============================================================================
# Sweep benchmark
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

    print("\n" + "=" * 120)
    print(f"  Vector Add Sweep Benchmark  |  dtype={dtype_name}")
    print("=" * 120)
    print(f"  {'Size':>10} | {'Naive(ms)':>10} {'Persist(ms)':>12} {'Optim(ms)':>10} {'Torch(ms)':>10} | "
          f"{'SP-Naive':>8} {'SP-Optim':>8} | {'Opt/Naive':>9} | {'Diff':>10} | {'Pass':>5}")
    print(f"  {'-'*116}")

    results = []
    for size in sizes:
        a = torch.randn(size, device="npu", dtype=dtype)
        b = torch.randn(size, device="npu", dtype=dtype)

        c_opt = vector_add(a, b, mode="optimized")
        c_torch = ref_program(a, b)
        max_diff = torch.max(torch.abs(c_opt.cpu().float() - c_torch.cpu().float())).item()
        precision_pass = torch.allclose(
            c_opt.cpu().float(), c_torch.cpu().float(), rtol=1e-3, atol=1e-3
        )

        ms_naive, _, _ = triton.testing.do_bench(
            lambda: vector_add(a, b, mode="naive"), quantiles=[0.5, 0.2, 0.8], warmup=warmup, rep=rep
        )
        ms_persistent, _, _ = triton.testing.do_bench(
            lambda: vector_add(a, b, mode="persistent"), quantiles=[0.5, 0.2, 0.8], warmup=warmup, rep=rep
        )
        ms_optimized, _, _ = triton.testing.do_bench(
            lambda: vector_add(a, b, mode="optimized"), quantiles=[0.5, 0.2, 0.8], warmup=warmup, rep=rep
        )
        ms_torch, _, _ = triton.testing.do_bench(
            lambda: ref_program(a, b), quantiles=[0.5, 0.2, 0.8], warmup=warmup, rep=rep
        )

        sp_naive = ms_torch / ms_naive
        sp_opt = ms_torch / ms_optimized
        opt_vs_naive = ms_naive / ms_optimized
        pass_str = "PASS" if precision_pass else "FAIL"
        size_str = f"{size//1024}K" if size < 1024*1024 else f"{size//1024//1024}M"

        print(f"  {size_str:>10} | {ms_naive:>10.4f} {ms_persistent:>12.4f} {ms_optimized:>10.4f} {ms_torch:>10.4f} | "
              f"{sp_naive:>7.3f}x {sp_opt:>7.3f}x | {opt_vs_naive:>8.2f}x | {max_diff:>10.2e} | {pass_str:>5}")
        results.append({
            "size": size, "sp_optimized": sp_opt, "opt_vs_naive": opt_vs_naive,
            "max_diff": max_diff, "pass": precision_pass,
        })

    print("=" * 120)
    all_pass = all(r["pass"] for r in results)
    avg_sp_opt = sum(r["sp_optimized"] for r in results) / len(results)
    avg_opt_vs_naive = sum(r["opt_vs_naive"] for r in results) / len(results)
    print(f"\n  精度检查 (rtol=1e-3, atol=1e-3): {'ALL PASS' if all_pass else 'HAS FAIL'}")
    print(f"  Optimized 平均加速比 vs PyTorch: {avg_sp_opt:.3f}x")
    print(f"  Optimized 平均提升 vs Naive: {avg_opt_vs_naive:.2f}x")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Triton-Ascend Vector Add Benchmark (Naive vs Persistent vs Optimized vs PyTorch)"
    )
    parser.add_argument("--size", type=int, default=1024 * 1024)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--sweep", action="store_true", help="Run sweep benchmark")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--plot", action="store_true", help="Generate performance plot")
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    if args.plot:
        benchmark.run(print_data=True)
    elif args.sweep:
        run_sweep_benchmark(dtype=dtype, warmup=args.warmup, rep=args.rep)
    else:
        run_comparison_benchmark(args.size, dtype=dtype, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    main()
