"""
Triton-Ascend 向量加法性能基准测试

五路对比：
  - Naive:     GPU 风格 grid，大量 program
  - Persistent: 固定核数 tl.range 跨步
  - Optimized: XBLOCK/XBLOCK_SUB constexpr 循环 + mask
  - Pipelined: constexpr 循环 + 无 mask + 对齐 (最大带宽)
  - PyTorch:   原生 a + b

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
        line_vals=["naive", "optimized", "pipelined", "torch"],
        line_names=["Naive", "Optimized", "Pipelined", "PyTorch"],
        styles=[("red", "-"), ("orange", "-"), ("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="vector-add-pipeline",
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
    elif provider == "pipelined":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: vector_add(a, b, mode="pipelined"), quantiles=quantiles
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
    c_pipe = vector_add(a, b, mode="pipelined")
    c_torch = ref_program(a, b)
    max_diff = torch.max(torch.abs(c_pipe.cpu().float() - c_torch.cpu().float())).item()
    precision_pass = torch.allclose(
        c_pipe.cpu().float(), c_torch.cpu().float(), rtol=1e-3, atol=1e-3
    )

    # 性能
    quantiles = [0.5, 0.2, 0.8]
    results = {}
    for mode in ["naive", "persistent", "optimized", "pipelined"]:
        ms, mn, mx = triton.testing.do_bench(
            lambda m=mode: vector_add(a, b, mode=m), quantiles=quantiles, warmup=warmup, rep=rep
        )
        results[mode] = {"ms": ms, "min": mn, "max": mx}

    ms_torch, _, _ = triton.testing.do_bench(
        lambda: ref_program(a, b), quantiles=quantiles, warmup=warmup, rep=rep
    )

    bytes_moved = 3 * size * element_bytes

    print("\n" + "=" * 95)
    print(f"  Vector Add Pipeline Benchmark  |  size={size:,}  |  dtype={dtype_name}")
    print(f"  数据量: {size * element_bytes / 1024 / 1024:.2f} MB per tensor")
    print("=" * 95)

    print(f"\n  {'模式':<14} {'延迟(ms)':>10} {'带宽(GB/s)':>12} {'vs PyTorch':>12}")
    print(f"  {'-'*48}")

    for mode in ["naive", "persistent", "optimized", "pipelined"]:
        ms = results[mode]["ms"]
        bw = bytes_moved / (ms * 1e-6) * 1e-9
        sp = ms_torch / ms
        label = mode.upper()
        print(f"  {label:<14} {ms:>10.4f} {bw:>12.1f} {sp:>11.3f}x")

    bw_torch = bytes_moved / (ms_torch * 1e-6) * 1e-9
    print(f"  {'PYTORCH':<14} {ms_torch:>10.4f} {bw_torch:>12.1f} {'1.000x':>12}")

    # 流水线提升
    pipe_speedup = results["naive"]["ms"] / results["pipelined"]["ms"]
    opt_speedup = results["naive"]["ms"] / results["optimized"]["ms"]
    print(f"\n  流水线提升:")
    print(f"    Pipelined vs Naive:     {pipe_speedup:.2f}x")
    print(f"    Optimized vs Naive:     {opt_speedup:.2f}x")
    print(f"    Pipelined vs Optimized: {results['optimized']['ms'] / results['pipelined']['ms']:.2f}x (无 mask 增益)")

    print(f"\n  精度 (rtol=1e-3): {'PASS' if precision_pass else 'FAIL'}, max_diff={max_diff:.2e}")
    print("=" * 95)


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
    print(f"  Vector Add Pipeline Sweep  |  dtype={dtype_name}")
    print("=" * 130)
    print(f"  {'Size':>10} | {'Naive':>8} {'Optim':>8} {'Pipe':>8} {'Torch':>8} | "
          f"{'Pipe BW':>8} {'Torch BW':>9} | {'SP-Pipe':>7} {'Pipe/Naive':>10} | {'MaxDiff':>10} | {'Pass':>5}")
    print(f"  {'-'*140}")

    results = []
    for size in sizes:
        a = torch.randn(size, device="npu", dtype=dtype)
        b = torch.randn(size, device="npu", dtype=dtype)

        c_pipe = vector_add(a, b, mode="pipelined")
        c_torch = ref_program(a, b)
        max_diff = torch.max(torch.abs(c_pipe.cpu().float() - c_torch.cpu().float())).item()
        precision_pass = torch.allclose(
            c_pipe.cpu().float(), c_torch.cpu().float(), rtol=1e-3, atol=1e-3
        )

        ms_naive, _, _ = triton.testing.do_bench(
            lambda: vector_add(a, b, mode="naive"), quantiles=[0.5, 0.2, 0.8], warmup=warmup, rep=rep
        )
        ms_opt, _, _ = triton.testing.do_bench(
            lambda: vector_add(a, b, mode="optimized"), quantiles=[0.5, 0.2, 0.8], warmup=warmup, rep=rep
        )
        ms_pipe, _, _ = triton.testing.do_bench(
            lambda: vector_add(a, b, mode="pipelined"), quantiles=[0.5, 0.2, 0.8], warmup=warmup, rep=rep
        )
        ms_torch, _, _ = triton.testing.do_bench(
            lambda: ref_program(a, b), quantiles=[0.5, 0.2, 0.8], warmup=warmup, rep=rep
        )

        bytes_moved = 3 * size * element_bytes
        bw_pipe = bytes_moved / (ms_pipe * 1e-6) * 1e-9
        bw_torch = bytes_moved / (ms_torch * 1e-6) * 1e-9
        sp_pipe = ms_torch / ms_pipe
        pipe_vs_naive = ms_naive / ms_pipe
        size_str = f"{size//1024}K" if size < 1024*1024 else f"{size//1024//1024}M"

        print(f"  {size_str:>10} | {ms_naive:>7.3f}  {ms_opt:>7.3f}  {ms_pipe:>7.3f}  {ms_torch:>7.3f} | "
              f"{bw_pipe:>7.1f}  {bw_torch:>8.1f} | {sp_pipe:>6.3f}x {pipe_vs_naive:>9.2f}x | {max_diff:>10.2e} | {'PASS' if precision_pass else 'FAIL':>5}")
        results.append({"sp_pipe": sp_pipe, "pipe_vs_naive": pipe_vs_naive, "pass": precision_pass})

    print("=" * 130)
    all_pass = all(r["pass"] for r in results)
    avg_sp = sum(r["sp_pipe"] for r in results) / len(results)
    avg_pipe_naive = sum(r["pipe_vs_naive"] for r in results) / len(results)
    print(f"\n  精度: {'ALL PASS' if all_pass else 'HAS FAIL'}")
    print(f"  Pipelined 平均加速 vs PyTorch: {avg_sp:.3f}x")
    print(f"  Pipelined 平均提升 vs Naive:   {avg_pipe_naive:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Triton-Ascend Vector Add Pipeline Benchmark")
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
