"""
Triton-Ascend Reduce Sum 性能基准测试

对比维度:
  - 模式:  Naive (每行 1 program) / Optimized (XBLOCK/XBLOCK_SUB/RBLOCK)
  - 精度:  simple (直接 tl.sum 累加) / pairwise (RBLOCK=64 逐元素累加, 最后 tl.sum(64))
  - 基准:  PyTorch torch.sum()
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
        line_vals=["naive", "opt-simple", "opt-pairwise", "torch"],
        line_names=["Naive", "Opt-Simple", "Opt-Pairwise", "PyTorch"],
        styles=[("red", "-"), ("orange", "-"), ("blue", "-"), ("green", "-")],
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
            lambda: reduce_sum(x, axis=-1, mode="naive", precision="pairwise"),
            quantiles=quantiles,
        )
    elif provider == "opt-simple":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: reduce_sum(x, axis=-1, mode="optimized", precision="simple"),
            quantiles=quantiles,
        )
    elif provider == "opt-pairwise":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: reduce_sum(x, axis=-1, mode="optimized", precision="pairwise"),
            quantiles=quantiles,
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

    quantiles = [0.5, 0.2, 0.8]
    variants = [
        ("Naive",         "naive",     "pairwise"),
        ("Opt-Simple",    "optimized", "simple"),
        ("Opt-Pairwise",  "optimized", "pairwise"),
    ]

    # 性能
    results = {}
    for label, mode, prec in variants:
        ms, _, _ = triton.testing.do_bench(
            lambda m=mode, p=prec: reduce_sum(x, axis=-1, mode=m, precision=p),
            quantiles=quantiles, warmup=warmup, rep=rep,
        )
        results[label] = ms

    ms_torch, _, _ = triton.testing.do_bench(
        lambda: ref_program(x), quantiles=quantiles, warmup=warmup, rep=rep
    )

    # 精度
    bytes_moved = (num_rows * num_cols + num_rows) * element_bytes

    print("\n" + "=" * 85)
    print(f"  Reduce Sum Benchmark  |  ({num_rows}, {num_cols})  |  dtype={dtype_name}")
    print("=" * 85)

    print(f"\n  {'模式':<16} {'延迟(ms)':>10} {'带宽(GB/s)':>12} {'vs PyTorch':>12} {'Diff':>12}")
    print(f"  {'-'*62}")

    for label, mode, prec in variants:
        ms = results[label]
        bw = bytes_moved / (ms * 1e-6) * 1e-9
        sp = ms_torch / ms
        c = reduce_sum(x, axis=-1, mode=mode, precision=prec)
        diff = torch.max(torch.abs(c.cpu().float() - ref_program(x).cpu().float())).item()
        print(f"  {label:<16} {ms:>10.4f} {bw:>12.1f} {sp:>11.3f}x {diff:>12.2e}")

    bw_torch = bytes_moved / (ms_torch * 1e-6) * 1e-9
    print(f"  {'PYTORCH':<16} {ms_torch:>10.4f} {bw_torch:>12.1f} {'1.000x':>12} {'0.00e+00':>12}")

    # 精度策略对比
    ref_cpu = ref_program(x).cpu().float()
    print(f"\n  精度策略对比:")
    print(f"  {'策略':<16} {'MaxDiff':>12} {'MeanDiff':>12} {'Pass':>6}")
    print(f"  {'-'*46}")
    for label, mode, prec in variants:
        c = reduce_sum(x, axis=-1, mode=mode, precision=prec).cpu().float()
        max_d = torch.max(torch.abs(c - ref_cpu)).item()
        mean_d = torch.mean(torch.abs(c - ref_cpu)).item()
        passed = torch.allclose(c, ref_cpu, rtol=1e-3, atol=1e-3)
        print(f"  {label:<16} {max_d:>12.2e} {mean_d:>12.2e} {'OK' if passed else 'FAIL':>6}")

    print("=" * 85)


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

    print("\n" + "=" * 110)
    print(f"  Reduce Sum Sweep  |  dtype={dtype_name}")
    print("=" * 110)
    print(f"  {'Shape':>14} | {'Simple':>8} {'Pairwise':>8} {'Torch':>8} | "
          f"{'PW BW':>7} | {'PW/Torch':>9} {'PW/Simple':>10} | {'SimDiff':>10} {'PWDiff':>10} | {'Pass':>4}")
    print(f"  {'-'*106}")

    results = []
    for rows, cols in shapes:
        x = torch.randn(rows, cols, device="npu", dtype=dtype)
        ref_cpu = ref_program(x).cpu().float()

        ms_simple, _, _ = triton.testing.do_bench(
            lambda: reduce_sum(x, axis=-1, mode="optimized", precision="simple"),
            quantiles=[0.5, 0.2, 0.8], warmup=warmup, rep=rep,
        )
        ms_pairwise, _, _ = triton.testing.do_bench(
            lambda: reduce_sum(x, axis=-1, mode="optimized", precision="pairwise"),
            quantiles=[0.5, 0.2, 0.8], warmup=warmup, rep=rep,
        )
        ms_torch, _, _ = triton.testing.do_bench(
            lambda: ref_program(x), quantiles=[0.5, 0.2, 0.8], warmup=warmup, rep=rep,
        )

        c_sim = reduce_sum(x, axis=-1, mode="optimized", precision="simple").cpu().float()
        c_pw = reduce_sum(x, axis=-1, mode="optimized", precision="pairwise").cpu().float()
        diff_sim = torch.max(torch.abs(c_sim - ref_cpu)).item()
        diff_pw = torch.max(torch.abs(c_pw - ref_cpu)).item()
        pass_sim = torch.allclose(c_sim, ref_cpu, rtol=1e-3, atol=1e-3)
        pass_pw = torch.allclose(c_pw, ref_cpu, rtol=1e-3, atol=1e-3)

        bytes_moved = (rows * cols + rows) * element_bytes
        bw_pw = bytes_moved / (ms_pairwise * 1e-6) * 1e-9
        sp_pw_torch = ms_torch / ms_pairwise
        sp_pw_simple = ms_simple / ms_pairwise
        pass_str = "OK" if (pass_sim and pass_pw) else "FAIL"

        print(f"  ({rows:>4}, {cols:>4})   | {ms_simple:>7.3f}  {ms_pairwise:>7.3f}  {ms_torch:>7.3f} | "
              f"{bw_pw:>6.1f}  | {sp_pw_torch:>8.3f}x {sp_pw_simple:>9.3f}x | "
              f"{diff_sim:>10.2e} {diff_pw:>10.2e} | {pass_str:>4}")
        results.append({
            "sp_pw_torch": sp_pw_torch, "sp_pw_simple": sp_pw_simple,
            "pass": pass_sim and pass_pw,
        })

    print("=" * 110)
    all_pass = all(r["pass"] for r in results)
    avg_pw_torch = sum(r["sp_pw_torch"] for r in results) / len(results)
    avg_pw_simple = sum(r["sp_pw_simple"] for r in results) / len(results)
    print(f"  精度: {'ALL PASS' if all_pass else 'HAS FAIL'}")
    print(f"  平均:  Pairwise vs PyTorch={avg_pw_torch:.3f}x  |  Pairwise vs Simple 开销={1/avg_pw_simple:.2f}x")


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
