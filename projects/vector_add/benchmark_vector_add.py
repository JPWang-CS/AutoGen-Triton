"""
Triton-Ascend 向量加法性能基准测试

四路对比:
  - Naive:     GPU 风格 grid
  - Optimized: XBLOCK/XBLOCK_SUB constexpr 循环 + UB 安全 block
  - Autotune:  @triton.autotune 自动搜索 BLOCK_SIZE + multibuffer
  - PyTorch:   原生 a + b

所有 benchmark 先静默执行，结果暂存变量，最后统一打印。
"""

import argparse
import os
import sys
from io import StringIO

os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"
os.environ.setdefault("TRITON_BENCH_METHOD", "npu")

import torch
import torch_npu
import triton
import triton.testing

from vector_add import vector_add


def ref_program(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b


ALL_MODES = ["naive", "optimized", "autotune"]
MODE_NAMES = {
    "naive": "Naive", "optimized": "Optimized", "autotune": "Autotune",
}


def calc_bandwidth_gbs(n_elements: int, element_bytes: int, ms: float) -> float:
    return 3 * n_elements * element_bytes / (ms * 1e-3) * 1e-9


def _silent_bench(fn, quantiles=None, warmup=10, rep=100):
    """静默执行 do_bench，抑制所有中间输出。"""
    if quantiles is None:
        quantiles = [0.5, 0.2, 0.8]
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    try:
        result = triton.testing.do_bench(fn, quantiles=quantiles, warmup=warmup, rep=rep)
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
    return result


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

    print(f"\n  Running sweep ({len(sizes)} sizes × {len(ALL_MODES)+1} modes) ...", end="", flush=True)

    rows = []
    for size in sizes:
        a = torch.randn(size, device="npu", dtype=dtype)
        b = torch.randn(size, device="npu", dtype=dtype)

        # 精度
        c_auto = vector_add(a, b, mode="autotune")
        c_torch = ref_program(a, b)
        max_diff = torch.max(torch.abs(c_auto.cpu().float() - c_torch.cpu().float())).item()
        precision_pass = torch.allclose(
            c_auto.cpu().float(), c_torch.cpu().float(), rtol=1e-3, atol=1e-3
        )

        # 性能 — 静默采集
        ms_map = {}
        for mode in ALL_MODES:
            ms, _, _ = _silent_bench(
                lambda m=mode: vector_add(a, b, mode=m), warmup=warmup, rep=rep
            )
            ms_map[mode] = ms

        ms_torch, _, _ = _silent_bench(
            lambda: ref_program(a, b), warmup=warmup, rep=rep
        )

        print(".", end="", flush=True)

        bw_best = calc_bandwidth_gbs(size, element_bytes, min(ms_map.values()))
        bw_torch = calc_bandwidth_gbs(size, element_bytes, ms_torch)

        rows.append({
            "size": size, "ms_map": ms_map, "ms_torch": ms_torch,
            "bw_best": bw_best, "bw_torch": bw_torch,
            "max_diff": max_diff, "pass": precision_pass,
        })

    # 统一打印
    print(f"\n\n{'=' * 100}")
    print(f"  Vector Add Sweep  |  dtype={dtype_name}")
    print(f"{'=' * 100}")
    print(f"  {'Size':>8} | {'Naive':>8} {'Optim':>8} {'Atune':>8} {'Torch':>8} | "
          f"{'BW_best':>7} {'BW_torch':>8} | {'Atune/T':>8} {'Opt/Nv':>7} | {'Diff':>10} | OK")
    print(f"  {'-' * 96}")

    for r in rows:
        sp_at = r["ms_torch"] / r["ms_map"]["autotune"]
        sp_on = r["ms_map"]["naive"] / r["ms_map"]["optimized"]
        sz = r["size"]
        sz_str = f"{sz//1024}K" if sz < 1024*1024 else f"{sz//1024//1024}M"
        ok = "OK" if r["pass"] else "FAIL"

        print(f"  {sz_str:>8} | {r['ms_map']['naive']:>7.3f}  {r['ms_map']['optimized']:>7.3f}  "
              f"{r['ms_map']['autotune']:>7.3f}  {r['ms_torch']:>7.3f} | "
              f"{r['bw_best']:>6.1f}  {r['bw_torch']:>7.1f} | "
              f"{sp_at:>7.3f}x {sp_on:>6.3f}x | "
              f"{r['max_diff']:>10.2e} | {ok}")

    print(f"{'=' * 100}")
    all_pass = all(r["pass"] for r in rows)
    avg_at = sum(r["ms_torch"] / r["ms_map"]["autotune"] for r in rows) / len(rows)
    avg_on = sum(r["ms_map"]["naive"] / r["ms_map"]["optimized"] for r in rows) / len(rows)
    print(f"  精度: {'ALL PASS' if all_pass else 'HAS FAIL'}")
    print(f"  平均:  Autotune/Torch={avg_at:.3f}x  |  Optimized/Naive={avg_on:.3f}x")


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

    print(f"\n  Running benchmark (size={size:,}) ...", end="", flush=True)

    a = torch.randn(size, device="npu", dtype=dtype)
    b = torch.randn(size, device="npu", dtype=dtype)

    # 精度
    c_auto = vector_add(a, b, mode="autotune")
    c_torch = ref_program(a, b)
    max_diff = torch.max(torch.abs(c_auto.cpu().float() - c_torch.cpu().float())).item()

    # 性能 — 静默采集
    ms_map = {}
    for mode in ALL_MODES:
        ms, mn, mx = _silent_bench(
            lambda m=mode: vector_add(a, b, mode=m), warmup=warmup, rep=rep
        )
        ms_map[mode] = {"ms": ms, "min": mn, "max": mx}

    ms_torch, _, _ = _silent_bench(
        lambda: ref_program(a, b), warmup=warmup, rep=rep
    )

    print(" done\n")

    # 打印结果
    print(f"{'=' * 80}")
    print(f"  Vector Add  |  size={size:,}  |  dtype={dtype_name}")
    print(f"  数据量: {size * element_bytes / 1024 / 1024:.2f} MB/tensor × 3 = "
          f"{3 * size * element_bytes / 1024 / 1024:.2f} MB")
    print(f"{'=' * 80}")

    print(f"\n  {'模式':<12} {'延迟(ms)':>10} {'带宽(GB/s)':>11} {'vs Torch':>10}")
    print(f"  {'-' * 43}")

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
    print(f"{'=' * 80}")


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
        ms, min_ms, max_ms = _silent_bench(lambda: ref_program(a, b), quantiles=quantiles)
    else:
        ms, min_ms, max_ms = _silent_bench(lambda: vector_add(a, b, mode=provider), quantiles=quantiles)

    gbps = lambda t: calc_bandwidth_gbs(size, 4, t)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


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
