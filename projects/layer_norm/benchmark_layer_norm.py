"""
Triton-Ascend Layer Normalization 性能基准测试

对比 Triton kernel 与 PyTorch layer_norm：
  - 执行时间 (ms)
  - 带宽 (GB/s)
  - 加速比 (Triton / PyTorch)
  - 数值精度 (是否通过 rtol=1e-3, atol=1e-3)
"""

import argparse
import torch
import torch_npu
import triton
import triton.testing

from layer_norm import layer_norm


def ref_program(
    x: torch.Tensor,
    gamma: torch.Tensor = None,
    beta: torch.Tensor = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """PyTorch 参考实现"""
    if gamma is not None and beta is not None:
        return torch.nn.functional.layer_norm(
            x, [x.shape[-1]], weight=gamma, bias=beta, eps=eps
        )
    else:
        return torch.nn.functional.layer_norm(x, [x.shape[-1]], eps=eps)


# ============================================================================
# perf_report 可视化 benchmark
# ============================================================================

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_cols"],
        x_vals=[128, 256, 512, 1024, 2048, 4096],
        x_log=False,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="layer-norm-performance",
        args={"num_rows": 1024},
    )
)
def benchmark(num_rows, num_cols, provider):
    x = torch.randn(num_rows, num_cols, device="npu", dtype=torch.float32)
    gamma = torch.randn(num_cols, device="npu", dtype=torch.float32)
    beta = torch.randn(num_cols, device="npu", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: ref_program(x, gamma, beta), quantiles=quantiles
        )
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: layer_norm(x, gamma, beta), quantiles=quantiles
        )

    # 带宽: 读 input + gamma + beta + 写 output = 4 * rows * cols * 4 bytes (float32)
    gbps = lambda ms: 4 * num_rows * num_cols * 4 / (ms * 1e-6) * 1e-9
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
    gamma = torch.randn(num_cols, device="npu", dtype=dtype)
    beta = torch.randn(num_cols, device="npu", dtype=dtype)

    # 数值精度
    c_triton = layer_norm(x, gamma, beta)
    c_torch = ref_program(x, gamma, beta)
    max_diff = torch.max(torch.abs(c_triton.cpu().float() - c_torch.cpu().float())).item()
    mean_diff = torch.mean(torch.abs(c_triton.cpu().float() - c_torch.cpu().float())).item()
    rtol, atol = 1e-3, 1e-3
    precision_pass = torch.allclose(
        c_triton.cpu().float(), c_torch.cpu().float(), rtol=rtol, atol=atol
    )

    # 性能
    quantiles = [0.5, 0.2, 0.8]
    ms_triton, min_triton, max_triton = triton.testing.do_bench(
        lambda: layer_norm(x, gamma, beta), quantiles=quantiles, warmup=warmup, rep=rep
    )
    ms_torch, min_torch, max_torch = triton.testing.do_bench(
        lambda: ref_program(x, gamma, beta), quantiles=quantiles, warmup=warmup, rep=rep
    )

    # 带宽: input + gamma + beta + output = 4 * total * element_bytes
    total_elements = num_rows * num_cols
    bytes_moved = (total_elements + num_cols + num_cols + total_elements) * element_bytes
    bw_triton = bytes_moved / (ms_triton * 1e-6) * 1e-9
    bw_torch = bytes_moved / (ms_torch * 1e-6) * 1e-9
    speedup = ms_torch / ms_triton

    data_mb = total_elements * element_bytes / 1024 / 1024

    print("\n" + "=" * 70)
    print(f"  LayerNorm Benchmark  |  ({num_rows}, {num_cols})  |  dtype={dtype_name}")
    print(f"  数据量: {data_mb:.2f} MB per tensor, 总搬运 {bytes_moved / 1024 / 1024:.2f} MB")
    print("=" * 70)
    print(f"\n  {'指标':<20} {'Triton':>12} {'PyTorch':>12} {'单位':>8}")
    print(f"  {'-'*52}")
    print(f"  {'中位数延迟':<20} {ms_triton:>12.4f} {ms_torch:>12.4f} {'ms':>8}")
    print(f"  {'最小延迟':<20} {min_triton:>12.4f} {min_torch:>12.4f} {'ms':>8}")
    print(f"  {'最大延迟':<20} {max_triton:>12.4f} {max_torch:>12.4f} {'ms':>8}")
    print(f"  {'带宽':<20} {bw_triton:>12.2f} {bw_torch:>12.2f} {'GB/s':>8}")
    print(f"\n  {'加速比 (Triton/PyTorch)':<30} {speedup:.3f}x")
    if speedup > 1.0:
        print(f"  Triton 比 PyTorch 快 {(speedup - 1) * 100:.1f}%")
    else:
        print(f"  Triton 比 PyTorch 慢 {(1 - speedup) * 100:.1f}%")
    print(f"\n  {'数值精度':<20} {'值':>12}")
    print(f"  {'-'*32}")
    print(f"  {'最大绝对误差':<20} {max_diff:>12.2e}")
    print(f"  {'平均绝对误差':<20} {mean_diff:>12.2e}")
    print(f"  {'精度要求':<20} {'rtol=1e-3, atol=1e-3':>12}")
    print(f"  {'是否通过':<20} {'PASS':>12}" if precision_pass else f"  {'是否通过':<20} {'FAIL':>12}")
    print("=" * 70)

    return {
        "shape": (num_rows, num_cols), "ms_triton": ms_triton, "ms_torch": ms_torch,
        "bw_triton": bw_triton, "bw_torch": bw_torch,
        "speedup": speedup, "max_diff": max_diff, "precision_pass": precision_pass,
    }


def run_sweep_benchmark(dtype: torch.dtype = torch.float32, warmup: int = 10, rep: int = 100):
    shapes = [(128, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 4096)]
    element_bytes = torch.tensor([], dtype=dtype).element_size()
    dtype_name = {torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16"}.get(dtype, str(dtype))

    print("\n" + "=" * 90)
    print(f"  LayerNorm Sweep  |  dtype={dtype_name}")
    print("=" * 90)
    print(f"  {'Shape':>14} | {'Triton':>8} {'Torch':>8} | "
          f"{'BW(GB/s)':>9} | {'Speedup':>7} | {'Diff':>10} | {'Pass':>4}")
    print(f"  {'-'*86}")

    results = []
    for num_rows, num_cols in shapes:
        x = torch.randn(num_rows, num_cols, device="npu", dtype=dtype)
        gamma = torch.randn(num_cols, device="npu", dtype=dtype)
        beta = torch.randn(num_cols, device="npu", dtype=dtype)

        c_triton = layer_norm(x, gamma, beta)
        c_torch = ref_program(x, gamma, beta)
        max_diff = torch.max(torch.abs(c_triton.cpu().float() - c_torch.cpu().float())).item()
        precision_pass = torch.allclose(c_triton.cpu().float(), c_torch.cpu().float(), rtol=1e-3, atol=1e-3)

        ms_triton, _, _ = triton.testing.do_bench(lambda: layer_norm(x, gamma, beta), quantiles=[0.5, 0.2, 0.8], warmup=warmup, rep=rep)
        ms_torch, _, _ = triton.testing.do_bench(lambda: ref_program(x, gamma, beta), quantiles=[0.5, 0.2, 0.8], warmup=warmup, rep=rep)

        total_elements = num_rows * num_cols
        bytes_moved = (total_elements + num_cols + num_cols + total_elements) * element_bytes
        bw_triton = bytes_moved / (ms_triton * 1e-6) * 1e-9
        speedup = ms_torch / ms_triton
        pass_str = "OK" if precision_pass else "FAIL"

        print(f"  ({num_rows:>4}, {num_cols:>4})   | {ms_triton:>7.3f}  {ms_torch:>7.3f} | "
              f"{bw_triton:>8.1f}  | {speedup:>6.3f}x | {max_diff:>10.2e} | {pass_str:>4}")
        results.append({"speedup": speedup, "pass": precision_pass})

    print("=" * 90)
    all_pass = all(r["pass"] for r in results)
    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    print(f"  精度: {'ALL PASS' if all_pass else 'HAS FAIL'}  |  平均加速比: {avg_speedup:.3f}x")
    return results


def main():
    parser = argparse.ArgumentParser(description="Triton-Ascend LayerNorm Benchmark")
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--sweep", action="store_true", help="Run sweep benchmark")
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
