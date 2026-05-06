"""
Triton-Ascend Reduce XOR 性能基准测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import triton
import triton.testing
from reduce_xor import reduce_xor, ref_program as ref_xor


def run_sweep(warmup=10, rep=100):
    shapes = [(128, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 4096)]

    print(f"\n  Reduce XOR Sweep  |  dtype=int32")
    print(f"  {'Shape':>14} | {'Triton(ms)':>10} {'Torch(ms)':>10} | {'Speedup':>8} | {'Pass':>4}")
    print(f"  {'-'*60}")

    for shape in shapes:
        x = torch.randint(-1000, 1000, shape, device="npu", dtype=torch.int32)
        ms_t, _, _ = triton.testing.do_bench(lambda: reduce_xor(x), warmup=warmup, rep=rep)
        ms_r, _, _ = triton.testing.do_bench(lambda: ref_xor(x), warmup=warmup, rep=rep)

        ref = ref_xor(x)
        result = reduce_xor(x)
        passed = torch.equal(result.cpu(), ref.cpu())

        print(f"  ({shape[0]:>4},{shape[1]:>4})  | {ms_t:>10.4f} {ms_r:>10.4f} | {ms_r/ms_t:>7.3f}x | {'OK' if passed else 'FAIL':>4}")

    print(f"  {'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Reduce XOR Benchmark")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=100)
    args = parser.parse_args()

    if args.sweep:
        run_sweep(warmup=args.warmup, rep=args.rep)
    else:
        shape = (args.rows, args.cols)
        x = torch.randint(-1000, 1000, shape, device="npu", dtype=torch.int32)
        ms_t, _, _ = triton.testing.do_bench(lambda: reduce_xor(x), warmup=args.warmup, rep=args.rep)
        ms_r, _, _ = triton.testing.do_bench(lambda: ref_xor(x), warmup=args.warmup, rep=args.rep)
        print(f"  Reduce XOR  shape={shape}")
        print(f"  Triton: {ms_t:.4f} ms  Torch: {ms_r:.4f} ms  Speedup: {ms_r/ms_t:.3f}x")


if __name__ == "__main__":
    main()
