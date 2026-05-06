"""
Triton-Ascend Reduce Sum 性能基准测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import triton
import triton.testing
from sum import reduce_sum, ref_program as ref_sum


def run_sweep(dtype=torch.float32, warmup=10, rep=100):
    shapes = [(128, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 4096)]

    print(f"\n  Reduce Sum Sweep  |  dtype={dtype}")
    print(f"  {'Shape':>14} | {'Triton(ms)':>10} {'Torch(ms)':>10} | {'Speedup':>8} | {'Diff':>10} | {'Pass':>4}")
    print(f"  {'-'*70}")

    for shape in shapes:
        x = torch.randn(*shape, device="npu", dtype=dtype)
        ms_t, _, _ = triton.testing.do_bench(lambda: reduce_sum(x), warmup=warmup, rep=rep)
        ms_r, _, _ = triton.testing.do_bench(lambda: ref_sum(x), warmup=warmup, rep=rep)

        ref = ref_sum(x)
        result = reduce_sum(x)
        diff = torch.max(torch.abs(result.cpu().float() - ref.cpu().float())).item()
        passed = torch.allclose(result.cpu().float(), ref.cpu().float(), rtol=1e-3, atol=1e-3)

        print(f"  ({shape[0]:>4},{shape[1]:>4})  | {ms_t:>10.4f} {ms_r:>10.4f} | {ms_r/ms_t:>7.3f}x | {diff:>10.2e} | {'OK' if passed else 'FAIL':>4}")

    print(f"  {'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Reduce Sum Benchmark")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=100)
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    if args.sweep:
        run_sweep(dtype=dtype, warmup=args.warmup, rep=args.rep)
    else:
        shape = (args.rows, args.cols)
        x = torch.randn(*shape, device="npu", dtype=dtype)
        ms_t, _, _ = triton.testing.do_bench(lambda: reduce_sum(x), warmup=args.warmup, rep=args.rep)
        ms_r, _, _ = triton.testing.do_bench(lambda: ref_sum(x), warmup=args.warmup, rep=args.rep)
        print(f"  Reduce Sum  shape={shape}  dtype={args.dtype}")
        print(f"  Triton: {ms_t:.4f} ms  Torch: {ms_r:.4f} ms  Speedup: {ms_r/ms_t:.3f}x")


if __name__ == "__main__":
    main()
