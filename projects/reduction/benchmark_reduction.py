"""
Triton-Ascend Reduction Operators 性能基准测试

对比: Triton 实现 vs PyTorch 原生实现
  - sum, max, min, argmax, argmin, xor_sum, prod
  - Sweep 不同 shape
  - 带宽 / 延迟 / 加速比
"""

import argparse
import torch
import triton
import triton.testing

from reduction import (
    reduce_sum, reduce_max, reduce_min,
    reduce_argmax, reduce_argmin,
    reduce_xor, reduce_prod,
    ref_sum, ref_max, ref_min,
    ref_argmax, ref_argmin,
    ref_xor, ref_prod,
)


# ============================================================================
# 单算子 benchmark
# ============================================================================

def benchmark_op(name, triton_fn, ref_fn, shape, dtype,
                 warmup=10, rep=100, tol=1e-3):
    x = torch.randn(*shape, device="npu", dtype=dtype) \
        if dtype.is_floating_point \
        else torch.randint(-1000, 1000, shape, device="npu", dtype=dtype)
    if dtype.is_floating_point and name == "prod":
        x = x * 0.5

    quantiles = [0.5, 0.2, 0.8]
    ms_triton, _, _ = triton.testing.do_bench(
        lambda: triton_fn(x), quantiles=quantiles, warmup=warmup, rep=rep,
    )
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: ref_fn(x), quantiles=quantiles, warmup=warmup, rep=rep,
    )

    ref = ref_fn(x)
    result = triton_fn(x)

    if name in ("argmax", "argmin"):
        passed = torch.equal(result.cpu(), ref.cpu())
        diff = 0.0
    elif name == "xor":
        passed = torch.equal(result.cpu(), ref.cpu())
        diff = 0.0
    else:
        diff = torch.max(torch.abs(result.cpu().float() - ref.cpu().float())).item()
        passed = torch.allclose(result.cpu().float(), ref.cpu().float(), rtol=tol, atol=tol)

    rows, cols = shape if len(shape) == 2 else (1, shape[0])
    bytes_moved = rows * cols * x.element_size() + rows * 4
    bw_triton = bytes_moved / (ms_triton * 1e-6) * 1e-9
    bw_torch = bytes_moved / (ms_torch * 1e-6) * 1e-9

    return {
        "name": name, "shape": shape,
        "ms_triton": ms_triton, "ms_torch": ms_torch,
        "bw_triton": bw_triton, "bw_torch": bw_torch,
        "speedup": ms_torch / ms_triton,
        "diff": diff, "passed": passed,
    }


# ============================================================================
# Sweep all operators
# ============================================================================

def run_sweep(dtype=torch.float32, warmup=10, rep=100):
    shapes = [(128, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 4096)]
    dtype_name = {
        torch.float32: "float32", torch.float16: "float16",
        torch.bfloat16: "bfloat16", torch.int32: "int32",
    }.get(dtype, str(dtype))

    ops = [
        ("sum",    reduce_sum,  ref_sum,  1e-3),
        ("max",    reduce_max,  ref_max,  1e-3),
        ("min",    reduce_min,  ref_min,  1e-3),
        ("argmax", reduce_argmax, ref_argmax, None),
        ("argmin", reduce_argmin, ref_argmin, None),
        ("xor",    reduce_xor,  ref_xor,  None),
        ("prod",   reduce_prod, ref_prod, 1e-2),
    ]

    print("\n" + "=" * 120)
    print(f"  Reduction Operators Sweep  |  dtype={dtype_name}")
    print("=" * 120)
    print(f"  {'Op':<9} {'Shape':>14} | {'Triton(ms)':>10} {'Torch(ms)':>10} | "
          f"{'BW(GB/s)':>9} | {'Torch BW':>9} | {'Speedup':>8} | {'Diff':>10} | {'Pass':>4}")
    print(f"  {'-'*114}")

    for op_name, triton_fn, ref_fn, tol in ops:
        for shape in shapes:
            op_dtype = torch.int32 if op_name == "xor" else dtype
            r = benchmark_op(op_name, triton_fn, ref_fn, shape, op_dtype,
                             warmup=warmup, rep=rep, tol=tol or 1e-3)
            status = "OK" if r["passed"] else "FAIL"
            print(f"  {r['name']:<9} ({shape[0]:>4},{shape[1]:>4})  | "
                  f"{r['ms_triton']:>10.4f} {r['ms_torch']:>10.4f} | "
                  f"{r['bw_triton']:>8.1f}  {r['bw_torch']:>8.1f}  | "
                  f"{r['speedup']:>7.3f}x | {r['diff']:>10.2e} | {status:>4}")
        print(f"  {'-'*114}")

    print("=" * 120)


# ============================================================================
# 单算子详细对比
# ============================================================================

def run_comparison(shape=(4096, 1024), dtype=torch.float32,
                   warmup=10, rep=100):
    dtype_name = {
        torch.float32: "float32", torch.float16: "float16",
        torch.bfloat16: "bfloat16", torch.int32: "int32",
    }.get(dtype, str(dtype))

    rows, cols = shape
    print(f"\n  Reduction Comparison  |  shape={shape}  |  dtype={dtype_name}")
    print(f"  {'='*70}\n")

    ops = [
        ("sum",    reduce_sum,  ref_sum,  1e-3),
        ("max",    reduce_max,  ref_max,  1e-3),
        ("min",    reduce_min,  ref_min,  1e-3),
        ("argmax", reduce_argmax, ref_argmax, None),
        ("argmin", reduce_argmin, ref_argmin, None),
        ("xor",    reduce_xor,  ref_xor,  None),
        ("prod",   reduce_prod, ref_prod, 1e-2),
    ]

    print(f"  {'Op':<10} {'Triton(ms)':>10} {'Torch(ms)':>10} {'Speedup':>9} {'Diff':>12} {'Pass':>6}")
    print(f"  {'-'*57}")

    for op_name, triton_fn, ref_fn, tol in ops:
        op_dtype = torch.int32 if op_name == "xor" else dtype
        r = benchmark_op(op_name, triton_fn, ref_fn, shape, op_dtype,
                         warmup=warmup, rep=rep, tol=tol or 1e-3)
        status = "OK" if r["passed"] else "FAIL"
        print(f"  {r['name']:<10} {r['ms_triton']:>10.4f} {r['ms_torch']:>10.4f} "
              f"{r['speedup']:>8.3f}x {r['diff']:>12.2e} {status:>6}")

    print(f"  {'='*70}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Triton-Ascend Reduction Benchmark")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=100)
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    if args.sweep:
        run_sweep(dtype=dtype, warmup=args.warmup, rep=args.rep)
    else:
        run_comparison(shape=(args.rows, args.cols), dtype=dtype,
                       warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    main()
