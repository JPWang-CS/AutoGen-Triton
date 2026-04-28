"""
Triton-Ascend Reduce Sum 性能基准测试

三路对比:
  - Naive:     每行一个 program
  - Optimized: XBLOCK/XBLOCK_SUB/RBLOCK 三层切分 + constexpr 循环
  - PyTorch:   torch.sum()

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

from reduce_sum import reduce_sum


def ref_program(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    return torch.sum(x, dim=axis)


# ============================================================================
# perf_report 可视化
# ============================================================================

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_cols"],
        x_vals=[128, 256, 512, 1024, 2048, 4096],
        x_log=False,
        line_arg="provider",
        line_vals=["naive", "optimized", "torch"],
        line_names=["Triton-Naive", "Triton-Optimized", "PyTorch"],
        styles=[("red", "-"), ("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="reduce-sum-performance",
        args={"num_rows": 4096},
    )
)
def benchmark(num_rows, num_cols, provider):
    x = torch.randn(num_rows, num_cols, device="npu", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: ref_program(x), quantiles=quantiles
        )
    elif provider == "naive":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: reduce_sum(x, axis=-1, mode="naive"), quantiles=quantiles
        )
    elif provider == "optimized":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: reduce_sum(x, axis=-1, mode="optimized"), quantiles=quantiles
        )

    # 带宽: 读 input (rows*cols) + 写 output (rows) = (rows*cols + rows) * dtype_size
    gbps = lambda ms: (num_rows * num_cols + num_rows) * 4 / (ms * 1e-6) * 1e-9
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# ============================================================================
# 详细对比 benchmark
# ============================================================================

def run_comparison_benchmark(
    num_rows: int, num_cols: int,
    dtype: torch.dtype = torch.float32,
    warmup: int = 10, rep: int = 100,
):
    element_bytes = torch.tensor([], dtype=dtype).element_size()
    dtype_name = {
        torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16",
    }.get(dtype, str(dtype))

    x = torch.randn(num_rows, num_cols, device="npu", dtype=dtype)

    # 精度
    c_opt = reduce_sum(x, axis=-1, mode="optimized")
    c_torch = ref_program(x)
    max_diff = torch.max(torch.abs(c_opt.cpu().float() - c_torch.cpu().float())).item()
    precision_pass = torch.allclose(
        c_opt.cpu().float(), c_torch.cpu().float(), rtol=1e-3, atol=1e-3
    )

    # 性能
    quantiles = [0.5, 0.2, 0.8]
    ms_naive, _, _ = triton.testing.do_bench(
        lambda: reduce_sum(x, axis=-1, mode="naive"), quantiles=quantiles, warmup=warmup, rep=rep
    )
    ms_optimized, _, _ = triton.testing.do_bench(
        lambda: reduce_sum(x, axis=-1, mode="optimized"), quantiles=quantiles, warmup=warmup, rep=rep
    )
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: ref_program(x), quantiles=quantiles, warmup=warmup, rep=rep
    )

    bytes_moved = (num_rows * num_cols + num_rows) * element_bytes
    bw_naive = bytes_moved / (ms_naive * 1e-6) * 1e-9
    bw_optimized = bytes_moved / (ms_optimized * 1e-6) * 1e-9
    bw_torch = bytes_moved / (ms_torch * 1e-6) * 1e-9

    sp_opt = ms_torch / ms_optimized

    print("\n" + "=" * 80)
    print(f"  Reduce Sum Benchmark  |  ({num_rows}, {num_cols})  |  dtype={dtype_name}")
    print(f"  数据量: {num_rows * num_cols * element_bytes / 1024 / 1024:.2f} MB")
    print("=" * 80)
    print(f"\n  {'指标':<16} {'Naive':>10} {'Optimized':>12} {'PyTorch':>10} {'单位':>6}")
    print(f"  {'-'*54}")
    print(f"  {'延迟(ms)':<16} {ms_naive:>10.4f} {ms_optimized:>12.4f} {ms_torch:>10.4f} {'ms':>6}")
    print(f"  {'带宽(GB/s)':<16} {bw_naive:>10.1f} {bw_optimized:>12.1f} {bw_torch:>10.1f} {'GB/s':>6}")
    print(f"\n  加速比 Optimized vs PyTorch: {sp_opt:.3f}x")
    print(f"  提升 Optimized vs Naive:     {ms_naive/ms_optimized:.2f}x")
    print(f"\n  最大绝对误差: {max_diff:.2e}  |  精度 (rtol=1e-3): {'PASS' if precision_pass else 'FAIL'}")
    print("=" * 80)


# ============================================================================
# Sweep benchmark
# ============================================================================

def run_sweep_benchmark(
    dtype: torch.dtype = torch.float32,
    warmup: int = 10, rep: int = 100,
):
    shapes = [(128, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 4096)]
    element_bytes = torch.tensor([], dtype=dtype).element_size()
    dtype_name = {
        torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16",
    }.get(dtype, str(dtype))

    print("\n" + "=" * 110)
    print(f"  Reduce Sum Sweep Benchmark  |  dtype={dtype_name}")
    print("=" * 110)
    print(f"  {'Shape':>14} | {'Naive(ms)':>10} {'Optim(ms)':>10} {'Torch(ms)':>10} | "
          f"{'Opt(GB/s)':>10} {'Torch(GB/s)':>11} | {'SP-Opt':>7} | {'Diff':>10} | {'Pass':>5}")
    print(f"  {'-'*106}")

    results = []
    for rows, cols in shapes:
        x = torch.randn(rows, cols, device="npu", dtype=dtype)
        c_opt = reduce_sum(x, axis=-1, mode="optimized")
        c_torch = ref_program(x)
        max_diff = torch.max(torch.abs(c_opt.cpu().float() - c_torch.cpu().float())).item()
        precision_pass = torch.allclose(c_opt.cpu().float(), c_torch.cpu().float(), rtol=1e-3, atol=1e-3)

        ms_naive, _, _ = triton.testing.do_bench(
            lambda: reduce_sum(x, axis=-1, mode="naive"), quantiles=[0.5, 0.2, 0.8], warmup=warmup, rep=rep
        )
        ms_opt, _, _ = triton.testing.do_bench(
            lambda: reduce_sum(x, axis=-1, mode="optimized"), quantiles=[0.5, 0.2, 0.8], warmup=warmup, rep=rep
        )
        ms_torch, _, _ = triton.testing.do_bench(
            lambda: ref_program(x), quantiles=[0.5, 0.2, 0.8], warmup=warmup, rep=rep
        )

        bytes_moved = (rows * cols + rows) * element_bytes
        bw_opt = bytes_moved / (ms_opt * 1e-6) * 1e-9
        bw_torch = bytes_moved / (ms_torch * 1e-6) * 1e-9
        sp_opt = ms_torch / ms_opt
        pass_str = "PASS" if precision_pass else "FAIL"

        print(f"  ({rows:>4}, {cols:>4})   | {ms_naive:>10.4f} {ms_opt:>10.4f} {ms_torch:>10.4f} | "
              f"{bw_opt:>10.1f} {bw_torch:>11.1f} | {sp_opt:>6.3f}x | {max_diff:>10.2e} | {pass_str:>5}")
        results.append({"shape": (rows, cols), "sp_opt": sp_opt, "max_diff": max_diff, "pass": precision_pass})

    print("=" * 110)
    all_pass = all(r["pass"] for r in results)
    avg_sp = sum(r["sp_opt"] for r in results) / len(results)
    print(f"\n  精度 (rtol=1e-3): {'ALL PASS' if all_pass else 'HAS FAIL'}")
    print(f"  Optimized 平均加速比 vs PyTorch: {avg_sp:.3f}x")


def main():
    parser = argparse.ArgumentParser(description="Triton-Ascend Reduce Sum Benchmark")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=1024)
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
        run_comparison_benchmark(args.rows, args.cols, dtype=dtype, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    main()
