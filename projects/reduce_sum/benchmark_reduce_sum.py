"""
Triton-Ascend Reduce Sum 性能基准测试

三路对比:
  - Naive:     每行一个 program
  - Optimized: XBLOCK/XBLOCK_SUB/RBLOCK 三层切分 + constexpr 循环
  - PyTorch:   torch.sum()
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
        line_names=["Naive", "Optimized", "PyTorch"],
        styles=[("red", "-"), ("orange", "-"), ("green", "-")],
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

    gbps = lambda ms: (num_rows * num_cols + num_rows) * 4 / (ms * 1e-6) * 1e-9
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# ============================================================================
# 详细对比
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
    results = {}
    for mode in ["naive", "optimized"]:
        ms, _, _ = triton.testing.do_bench(
            lambda m=mode: reduce_sum(x, axis=-1, mode=m), quantiles=quantiles, warmup=warmup, rep=rep
        )
        results[mode] = ms

    ms_torch, _, _ = triton.testing.do_bench(
        lambda: ref_program(x), quantiles=quantiles, warmup=warmup, rep=rep
    )

    bytes_moved = (num_rows * num_cols + num_rows) * element_bytes

    print("\n" + "=" * 75)
    print(f"  Reduce Sum Benchmark  |  ({num_rows}, {num_cols})  |  dtype={dtype_name}")
    print("=" * 75)

    print(f"\n  {'模式':<14} {'延迟(ms)':>10} {'带宽(GB/s)':>12} {'vs PyTorch':>12}")
    print(f"  {'-'*48}")

    for mode in ["naive", "optimized"]:
        ms = results[mode]
        bw = bytes_moved / (ms * 1e-6) * 1e-9
        sp = ms_torch / ms
        print(f"  {mode.upper():<14} {ms:>10.4f} {bw:>12.1f} {sp:>11.3f}x")

    bw_torch = bytes_moved / (ms_torch * 1e-6) * 1e-9
    print(f"  {'PYTORCH':<14} {ms_torch:>10.4f} {bw_torch:>12.1f} {'1.000x':>12}")

    opt_speedup = results["naive"] / results["optimized"]
    print(f"\n  Optimized vs Naive: {opt_speedup:.2f}x")
    print(f"  精度 (rtol=1e-3): {'PASS' if precision_pass else 'FAIL'}, max_diff={max_diff:.2e}")
    print("=" * 75)


# ============================================================================
# Sweep
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

    print("\n" + "=" * 90)
    print(f"  Reduce Sum Sweep  |  dtype={dtype_name}")
    print("=" * 90)
    print(f"  {'Shape':>14} | {'Naive':>8} {'Optim':>8} {'Torch':>8} | "
          f"{'Optim BW':>9} | {'Speedup':>7} | {'Diff':>10} | {'Pass':>4}")
    print(f"  {'-'*86}")

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
        speedup = ms_torch / ms_opt
        pass_str = "OK" if precision_pass else "FAIL"

        print(f"  ({rows:>4}, {cols:>4})   | {ms_naive:>7.3f}  {ms_opt:>7.3f}  {ms_torch:>7.3f} | "
              f"{bw_opt:>8.1f}  | {speedup:>6.3f}x | {max_diff:>10.2e} | {pass_str:>4}")
        results.append({"speedup": speedup, "pass": precision_pass})

    print("=" * 90)
    all_pass = all(r["pass"] for r in results)
    avg_sp = sum(r["speedup"] for r in results) / len(results)
    print(f"  精度: {'ALL PASS' if all_pass else 'HAS FAIL'}  |  Optimized 平均加速 vs PyTorch: {avg_sp:.3f}x")


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
