"""
Triton-Ascend 向量加法性能基准测试

四路对比:
  - Naive:     GPU 风格 grid
  - Optimized: XBLOCK/XBLOCK_SUB constexpr 循环 + UB 安全 block
  - Autotune:  @triton.autotune 自动搜索 BLOCK_SIZE + multibuffer
  - PyTorch:   原生 a + b

分析维度:
  - 延迟 (ms)
  - 带宽 (GB/s) — 注意: do_bench 返回 ms, 转秒需 ×1e-3
  - vs PyTorch 加速比
  - 带宽利用率

msprof 分析建议:
  对于内存 bound 操作, 用 msprof 确认瓶颈:
    msprof op --output=./prof_out python -c "..."
  查看 MTE2 占比: 若 MTE2 >> Vector, 说明是带宽瓶颈, kernel 优化空间有限
"""

import argparse
import os
import torch
import torch_npu
import triton
import triton.testing

from vector_add import vector_add

os.environ.setdefault("TRITON_BENCH_METHOD", "npu")


def ref_program(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b


ALL_MODES = ["naive", "optimized", "autotune"]
MODE_NAMES = {
    "naive": "Naive", "optimized": "Optimized", "autotune": "Autotune",
}


def calc_bandwidth_gbs(n_elements: int, element_bytes: int, ms: float) -> float:
    """3 个 I/O 张量 × n_elements × element_bytes / 时间(秒) → GB/s"""
    bytes_total = 3 * n_elements * element_bytes
    return bytes_total / (ms * 1e-3) * 1e-9


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

    gbps = lambda t: calc_bandwidth_gbs(size, 4, t)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


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
        sizes = [4 * 1024, 16 * 1024, 64 * 1024, 256 * 1024,
                 1024 * 1024, 4 * 1024 * 1024]

    element_bytes = torch.tensor([], dtype=dtype).element_size()
    dtype_name = {
        torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16",
    }.get(dtype, str(dtype))

    print("\n" + "=" * 100)
    print(f"  Vector Add Sweep  |  dtype={dtype_name}  |  TRITON_BENCH_METHOD={os.environ.get('TRITON_BENCH_METHOD', 'default')}")
    print("=" * 100)
    print(f"  {'Size':>8} | {'Naive':>8} {'Optim':>8} {'Atune':>8} {'Torch':>8} | "
          f"{'BW_best':>7} {'BW_torch':>8} | {'Atune/T':>8} {'Opt/Nv':>7} | {'Diff':>10} | OK")
    print(f"  {'-'*96}")

    all_results = []
    for size in sizes:
        a = torch.randn(size, device="npu", dtype=dtype)
        b = torch.randn(size, device="npu", dtype=dtype)

        # 精度
        c_atune = vector_add(a, b, mode="autotune")
        c_torch = ref_program(a, b)
        max_diff = torch.max(torch.abs(c_atune.cpu().float() - c_torch.cpu().float())).item()
        precision_pass = torch.allclose(
            c_atune.cpu().float(), c_torch.cpu().float(), rtol=1e-3, atol=1e-3
        )

        # 性能
        ms_map = {}
        for mode in ALL_MODES:
            ms, _, _ = triton.testing.do_bench(
                lambda m=mode: vector_add(a, b, mode=m),
                quantiles=[0.5, 0.2, 0.8], warmup=warmup, rep=rep,
            )
            ms_map[mode] = ms

        ms_torch, _, _ = triton.testing.do_bench(
            lambda: ref_program(a, b), quantiles=[0.5, 0.2, 0.8],
            warmup=warmup, rep=rep,
        )

        bw_best = calc_bandwidth_gbs(size, element_bytes, min(ms_map.values()))
        bw_torch = calc_bandwidth_gbs(size, element_bytes, ms_torch)
        sp_atune_t = ms_torch / ms_map["autotune"]
        sp_opt_nv = ms_map["naive"] / ms_map["optimized"]
        size_str = f"{size//1024}K" if size < 1024*1024 else f"{size//1024//1024}M"

        print(f"  {size_str:>8} | {ms_map['naive']:>7.3f}  {ms_map['optimized']:>7.3f}  "
              f"{ms_map['autotune']:>7.3f}  {ms_torch:>7.3f} | "
              f"{bw_best:>6.1f}  {bw_torch:>7.1f} | "
              f"{sp_atune_t:>7.3f}x {sp_opt_nv:>6.3f}x | "
              f"{max_diff:>10.2e} | {'OK' if precision_pass else 'FAIL'}")

        all_results.append({
            "sp_atune_t": sp_atune_t, "sp_opt_nv": sp_opt_nv, "pass": precision_pass,
        })

    print("=" * 100)
    all_pass = all(r["pass"] for r in all_results)
    avg_at = sum(r["sp_atune_t"] for r in all_results) / len(all_results)
    avg_on = sum(r["sp_opt_nv"] for r in all_results) / len(all_results)
    print(f"  精度: {'ALL PASS' if all_pass else 'HAS FAIL'}")
    print(f"  平均:  Autotune/Torch={avg_at:.3f}x  |  Optimized/Naive={avg_on:.3f}x")
    print(f"\n  注: vector_add 是纯内存 bound 操作 (2load+1store+1add)")
    print(f"      kernel 结构优化对内存 bound 操作收益有限，瓶颈在 GM 带宽")
    print(f"      建议用 msprof op 确认 MTE2 占比，若 MTE2>>Vector 则为带宽瓶颈")


# ============================================================================
# 详细对比 benchmark (单 size)
# ============================================================================

def run_comparison_benchmark(
    size: int,
    dtype: torch.dtype = torch.float32,
    warmup: int = 10,
    rep: int = 100,
):
    element_bytes = torch.tensor([], dtype=dtype).element_size()
    dtype_name = {
        torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16",
    }.get(dtype, str(dtype))

    a = torch.randn(size, device="npu", dtype=dtype)
    b = torch.randn(size, device="npu", dtype=dtype)

    # 精度
    c_auto = vector_add(a, b, mode="autotune")
    c_torch = ref_program(a, b)
    max_diff = torch.max(torch.abs(c_auto.cpu().float() - c_torch.cpu().float())).item()

    # 性能
    quantiles = [0.5, 0.2, 0.8]
    ms_map = {}
    for mode in ALL_MODES:
        ms, mn, mx = triton.testing.do_bench(
            lambda m=mode: vector_add(a, b, mode=m),
            quantiles=quantiles, warmup=warmup, rep=rep,
        )
        ms_map[mode] = {"ms": ms, "min": mn, "max": mx}

    ms_torch, _, _ = triton.testing.do_bench(
        lambda: ref_program(a, b), quantiles=quantiles, warmup=warmup, rep=rep
    )

    print("\n" + "=" * 80)
    print(f"  Vector Add  |  size={size:,}  |  dtype={dtype_name}")
    print(f"  数据量: {size * element_bytes / 1024 / 1024:.2f} MB/tensor × 3 = "
          f"{3 * size * element_bytes / 1024 / 1024:.2f} MB")
    print("=" * 80)

    print(f"\n  {'模式':<12} {'延迟(ms)':>10} {'带宽(GB/s)':>11} {'vs Torch':>10}")
    print(f"  {'-'*43}")

    for mode in ALL_MODES:
        ms = ms_map[mode]["ms"]
        bw = calc_bandwidth_gbs(size, element_bytes, ms)
        sp = ms_torch / ms
        print(f"  {MODE_NAMES[mode]:<12} {ms:>10.4f} {bw:>11.1f} {sp:>9.3f}x")

    bw_t = calc_bandwidth_gbs(size, element_bytes, ms_torch)
    print(f"  {'PYTORCH':<12} {ms_torch:>10.4f} {bw_t:>11.1f} {'1.000x':>10}")

    ms_nv = ms_map["naive"]["ms"]
    ms_opt = ms_map["optimized"]["ms"]
    ms_at = ms_map["autotune"]["ms"]
    print(f"\n  优化阶段:")
    print(f"    Naive → Optimized (constexpr+UB安全):  {ms_nv / ms_opt:.3f}x")
    print(f"    Optimized → Autotune (自动搜索):        {ms_opt / ms_at:.3f}x")
    print(f"    Naive → Autotune (总体):                {ms_nv / ms_at:.3f}x")
    print(f"\n  精度: max_diff={max_diff:.2e}")
    print("=" * 80)


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
