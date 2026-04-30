"""
Triton-Ascend 向量加法性能基准测试

五路对比:
  - Naive:     GPU 风格 grid，大量 program
  - Persistent: 固定核数 tl.range 跨步
  - Optimized: XBLOCK/XBLOCK_SUB constexpr 循环 + UB 最优 block
  - Autotune:  @triton.autotune 自动搜索 BLOCK_SIZE + multibuffer
  - PyTorch:   原生 a + b

性能指标:
  - 执行时间 (ms)
  - 带宽 (GB/s)
  - 各方法间加速比
  - 优化阶段对比
"""

import argparse
import torch
import torch_npu
import triton
import triton.testing

from vector_add import vector_add


def ref_program(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b


ALL_MODES = ["naive", "persistent", "optimized", "autotune"]
MODE_NAMES = {"naive": "Naive", "persistent": "Persistent", "optimized": "Optimized", "autotune": "Autotune"}


# ============================================================================
# perf_report 可视化
# ============================================================================

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(12, 24, 1)],
        x_log=True,
        line_arg="provider",
        line_vals=["naive", "optimized", "autotune", "torch"],
        line_names=["Naive", "Optimized", "Autotune", "PyTorch"],
        styles=[("red", "-"), ("orange", "-"), ("blue", "-"), ("green", "-")],
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
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: vector_add(a, b, mode=provider), quantiles=quantiles
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

    # 精度
    c_opt = vector_add(a, b, mode="autotune")
    c_torch = ref_program(a, b)
    max_diff = torch.max(torch.abs(c_opt.cpu().float() - c_torch.cpu().float())).item()
    precision_pass = torch.allclose(
        c_opt.cpu().float(), c_torch.cpu().float(), rtol=1e-3, atol=1e-3
    )

    # 性能
    quantiles = [0.5, 0.2, 0.8]
    results = {}
    for mode in ALL_MODES:
        ms, mn, mx = triton.testing.do_bench(
            lambda m=mode: vector_add(a, b, mode=m), quantiles=quantiles, warmup=warmup, rep=rep
        )
        results[mode] = {"ms": ms, "min": mn, "max": mx}

    ms_torch, _, _ = triton.testing.do_bench(
        lambda: ref_program(a, b), quantiles=quantiles, warmup=warmup, rep=rep
    )

    bytes_moved = 3 * size * element_bytes

    print("\n" + "=" * 90)
    print(f"  Vector Add Benchmark  |  size={size:,}  |  dtype={dtype_name}")
    print(f"  数据量: {size * element_bytes / 1024 / 1024:.2f} MB per tensor")
    print("=" * 90)

    print(f"\n  {'模式':<14} {'延迟(ms)':>10} {'带宽(GB/s)':>12} {'vs PyTorch':>12}")
    print(f"  {'-'*48}")

    for mode in ALL_MODES:
        ms = results[mode]["ms"]
        bw = bytes_moved / (ms * 1e-6) * 1e-9
        sp = ms_torch / ms
        print(f"  {MODE_NAMES[mode]:<14} {ms:>10.4f} {bw:>12.1f} {sp:>11.3f}x")

    bw_torch = bytes_moved / (ms_torch * 1e-6) * 1e-9
    print(f"  {'PYTORCH':<14} {ms_torch:>10.4f} {bw_torch:>12.1f} {'1.000x':>12}")

    print(f"\n  各优化阶段提升:")
    ms_naive = results["naive"]["ms"]
    ms_persist = results["persistent"]["ms"]
    ms_opt = results["optimized"]["ms"]
    ms_atune = results["autotune"]["ms"]
    print(f"    Naive → Persistent (固定核数):       {ms_naive / ms_persist:.2f}x")
    print(f"    Persistent → Optimized (UB最优):     {ms_persist / ms_opt:.2f}x")
    print(f"    Optimized → Autotune (自动搜索):     {ms_opt / ms_atune:.2f}x")
    print(f"    Naive → Autotune (总体提升):         {ms_naive / ms_atune:.2f}x")
    print(f"\n  精度 (rtol=1e-3): {'PASS' if precision_pass else 'FAIL'}, max_diff={max_diff:.2e}")
    print("=" * 90)


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

    print("\n" + "=" * 130)
    print(f"  Vector Add Sweep  |  dtype={dtype_name}")
    print("=" * 130)
    header = (f"  {'Size':>10} | {'Naive':>8} {'Persist':>8} {'Optim':>8} "
              f"{'Atune':>8} {'Torch':>8} | {'Best BW':>7} | "
              f"{'Atune/T':>8} {'Opt/Naive':>9} {'Atune/Opt':>9} | {'Diff':>10} | {'Pass':>4}")
    print(header)
    print(f"  {'-'*126}")

    results = []
    for size in sizes:
        a = torch.randn(size, device="npu", dtype=dtype)
        b = torch.randn(size, device="npu", dtype=dtype)

        c_atune = vector_add(a, b, mode="autotune")
        c_torch = ref_program(a, b)
        max_diff = torch.max(torch.abs(c_atune.cpu().float() - c_torch.cpu().float())).item()
        precision_pass = torch.allclose(
            c_atune.cpu().float(), c_torch.cpu().float(), rtol=1e-3, atol=1e-3
        )

        ms_results = {}
        for mode in ALL_MODES:
            ms, _, _ = triton.testing.do_bench(
                lambda m=mode: vector_add(a, b, mode=m),
                quantiles=[0.5, 0.2, 0.8], warmup=warmup, rep=rep
            )
            ms_results[mode] = ms

        ms_torch, _, _ = triton.testing.do_bench(
            lambda: ref_program(a, b), quantiles=[0.5, 0.2, 0.8], warmup=warmup, rep=rep
        )

        bytes_moved = 3 * size * element_bytes
        bw_best = bytes_moved / (min(ms_results[m] for m in ALL_MODES) * 1e-6) * 1e-9
        sp_atune_torch = ms_torch / ms_results["autotune"]
        sp_opt_naive = ms_results["naive"] / ms_results["optimized"]
        sp_atune_opt = ms_results["optimized"] / ms_results["autotune"]
        size_str = f"{size//1024}K" if size < 1024*1024 else f"{size//1024//1024}M"
        pass_str = "OK" if precision_pass else "FAIL"

        print(f"  {size_str:>10} | {ms_results['naive']:>7.3f}  {ms_results['persistent']:>7.3f}  "
              f"{ms_results['optimized']:>7.3f}  {ms_results['autotune']:>7.3f}  {ms_torch:>7.3f} | "
              f"{bw_best:>6.1f}  | {sp_atune_torch:>7.3f}x {sp_opt_naive:>8.3f}x {sp_atune_opt:>8.3f}x | "
              f"{max_diff:>10.2e} | {pass_str:>4}")

        results.append({
            "sp_atune_torch": sp_atune_torch, "sp_opt_naive": sp_opt_naive,
            "sp_atune_opt": sp_atune_opt, "pass": precision_pass,
        })

    print("=" * 130)
    all_pass = all(r["pass"] for r in results)
    avg_atune_torch = sum(r["sp_atune_torch"] for r in results) / len(results)
    avg_opt_naive = sum(r["sp_opt_naive"] for r in results) / len(results)
    avg_atune_opt = sum(r["sp_atune_opt"] for r in results) / len(results)
    print(f"  精度: {'ALL PASS' if all_pass else 'HAS FAIL'}")
    print(f"  平均提升:  Autotune vs PyTorch={avg_atune_torch:.3f}x  |  "
          f"Optimized vs Naive={avg_opt_naive:.2f}x  |  "
          f"Autotune vs Optimized={avg_atune_opt:.3f}x")


def main():
    parser = argparse.ArgumentParser(description="Triton-Ascend Vector Add Benchmark")
    parser.add_argument("--size", type=int, default=1024 * 1024)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--sweep", action="store_true")
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
        run_comparison_benchmark(args.size, dtype=dtype, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    main()
