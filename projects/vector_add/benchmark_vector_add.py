"""
Triton-Ascend 向量加法性能基准测试

四路对比：
  - Naive:     GPU 风格 grid，大量 program
  - Persistent: 固定核数 tl.range 跨步
  - Optimized: XBLOCK/XBLOCK_SUB constexpr 循环 + UB 最优 block
  - PyTorch:   原生 a + b

性能指标:
  - 执行时间 (ms)
  - 带宽 (GB/s)
  - 各方法间加速比
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
        line_names=["Naive", "Optimized", "PyTorch"],
        styles=[("red", "-"), ("orange", "-"), ("green", "-")],
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

    # 精度
    c_opt = vector_add(a, b, mode="optimized")
    c_torch = ref_program(a, b)
    max_diff = torch.max(torch.abs(c_opt.cpu().float() - c_torch.cpu().float())).item()
    precision_pass = torch.allclose(
        c_opt.cpu().float(), c_torch.cpu().float(), rtol=1e-3, atol=1e-3
    )

    # 性能
    quantiles = [0.5, 0.2, 0.8]
    results = {}
    for mode in ["naive", "persistent", "optimized"]:
        ms, mn, mx = triton.testing.do_bench(
            lambda m=mode: vector_add(a, b, mode=m), quantiles=quantiles, warmup=warmup, rep=rep
        )
        results[mode] = {"ms": ms, "min": mn, "max": mx}

    ms_torch, _, _ = triton.testing.do_bench(
        lambda: ref_program(a, b), quantiles=quantiles, warmup=warmup, rep=rep
    )

    bytes_moved = 3 * size * element_bytes

    print("\n" + "=" * 80)
    print(f"  Vector Add Benchmark  |  size={size:,}  |  dtype={dtype_name}")
    print(f"  数据量: {size * element_bytes / 1024 / 1024:.2f} MB per tensor")
    print("=" * 80)

    print(f"\n  {'模式':<14} {'延迟(ms)':>10} {'带宽(GB/s)':>12} {'vs PyTorch':>12}")
    print(f"  {'-'*48}")

    for mode in ["naive", "persistent", "optimized"]:
        ms = results[mode]["ms"]
        bw = bytes_moved / (ms * 1e-6) * 1e-9
        sp = ms_torch / ms
        print(f"  {mode.upper():<14} {ms:>10.4f} {bw:>12.1f} {sp:>11.3f}x")

    bw_torch = bytes_moved / (ms_torch * 1e-6) * 1e-9
    print(f"  {'PYTORCH':<14} {ms_torch:>10.4f} {bw_torch:>12.1f} {'1.000x':>12}")

    print(f"\n  各优化阶段提升:")
    ms_naive = results["naive"]["ms"]
    ms_persist = results["persistent"]["ms"]
    ms_opt = results["optimized"]["ms"]
    print(f"    Naive → Persistent (固定核数):   {ms_naive / ms_persist:.2f}x")
    print(f"    Persistent → Optimized (UB最优):  {ms_persist / ms_opt:.2f}x")
    print(f"    Naive → Optimized (总体提升):     {ms_naive / ms_opt:.2f}x")
    print(f"\n  精度 (rtol=1e-3): {'PASS' if precision_pass else 'FAIL'}, max_diff={max_diff:.2e}")
    print("=" * 80)


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

    print("\n" + "=" * 110)
    print(f"  Vector Add Sweep  |  dtype={dtype_name}")
    print("=" * 110)
    print(f"  {'Size':>10} | {'Naive':>8} {'Persist':>8} {'Optim':>8} {'Torch':>8} | "
          f"{'Opt BW':>7} | {'Opt/Torch':>9} {'Opt/Naive':>9} {'Pst/Naive':>9} | {'Diff':>10} | {'Pass':>4}")
    print(f"  {'-'*106}")

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
        ms_persist, _, _ = triton.testing.do_bench(
            lambda: vector_add(a, b, mode="persistent"), quantiles=[0.5, 0.2, 0.8], warmup=warmup, rep=rep
        )
        ms_opt, _, _ = triton.testing.do_bench(
            lambda: vector_add(a, b, mode="optimized"), quantiles=[0.5, 0.2, 0.8], warmup=warmup, rep=rep
        )
        ms_torch, _, _ = triton.testing.do_bench(
            lambda: ref_program(a, b), quantiles=[0.5, 0.2, 0.8], warmup=warmup, rep=rep
        )

        bytes_moved = 3 * size * element_bytes
        bw_opt = bytes_moved / (ms_opt * 1e-6) * 1e-9
        sp_opt_torch = ms_torch / ms_opt
        sp_opt_naive = ms_naive / ms_opt
        sp_pst_naive = ms_naive / ms_persist
        size_str = f"{size//1024}K" if size < 1024*1024 else f"{size//1024//1024}M"
        pass_str = "OK" if precision_pass else "FAIL"

        print(f"  {size_str:>10} | {ms_naive:>7.3f}  {ms_persist:>7.3f}  {ms_opt:>7.3f}  {ms_torch:>7.3f} | "
              f"{bw_opt:>6.1f}  | {sp_opt_torch:>8.3f}x {sp_opt_naive:>8.3f}x {sp_pst_naive:>8.3f}x | {max_diff:>10.2e} | {pass_str:>4}")
        results.append({
            "sp_opt_torch": sp_opt_torch, "sp_opt_naive": sp_opt_naive,
            "sp_pst_naive": sp_pst_naive, "pass": precision_pass,
        })

    print("=" * 110)
    all_pass = all(r["pass"] for r in results)
    avg_opt_torch = sum(r["sp_opt_torch"] for r in results) / len(results)
    avg_opt_naive = sum(r["sp_opt_naive"] for r in results) / len(results)
    avg_pst_naive = sum(r["sp_pst_naive"] for r in results) / len(results)
    print(f"  精度: {'ALL PASS' if all_pass else 'HAS FAIL'}")
    print(f"  平均提升:  Optimized vs PyTorch={avg_opt_torch:.3f}x  |  Optimized vs Naive={avg_opt_naive:.2f}x  |  Persistent vs Naive={avg_pst_naive:.2f}x")


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
