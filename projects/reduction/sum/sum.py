"""
Triton-Ascend Reduce Sum

output = sum(x, axis=-1)

提供两种实现:
  - simple:  直接累加 acc += x，r_loops 多时精度损失
  - kahan:   Kahan 补偿求和，补偿每次累加的截断误差
"""

import triton
import triton.language as tl
import triton.runtime.driver as driver
import torch

_UB_MAX_INPUT_ELEMENTS = 12 * 1024


def _get_num_vector_cores() -> int:
    device = torch.npu.current_device()
    props = driver.active.utils.get_device_properties(device)
    return props["num_vectorcore"]


def _safe_rblock(n_cols: int, xblock_sub: int, num_tensors: int = 1) -> int:
    max_elements = (_UB_MAX_INPUT_ELEMENTS // max(num_tensors, 1)) // max(xblock_sub, 1)
    rblock = triton.next_power_of_2(min(n_cols, max_elements))
    return max(rblock, 1)


@triton.jit
def _reduce_sum_kernel(
    in_ptr, out_ptr,
    n_rows, n_cols,
    stride_in_row, stride_out_row,
    XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr, RBLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    rbase = tl.arange(0, RBLOCK)
    x_loops: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    r_loops: tl.constexpr = (n_cols + RBLOCK - 1) // RBLOCK

    for x_loop in range(x_loops):
        row_idx = xoffset + x_loop * XBLOCK_SUB + tl.arange(0, XBLOCK_SUB)
        row_mask = row_idx < n_rows
        acc = tl.zeros([XBLOCK_SUB, RBLOCK], dtype=tl.float32)

        for r_loop in range(r_loops):
            col_start = r_loop * RBLOCK
            rindex = col_start + rbase
            rmask = rindex < n_cols
            ptrs = in_ptr + row_idx[:, None] * stride_in_row + rindex[None, :]
            block_mask = row_mask[:, None] & rmask[None, :]
            x = tl.load(ptrs, mask=block_mask, other=0.0).to(tl.float32)
            acc = acc + x

        result = tl.sum(acc, axis=1)
        tl.store(out_ptr + row_idx * stride_out_row, result, mask=row_mask)


@triton.jit
def _reduce_sum_kahan_kernel(
    in_ptr, out_ptr,
    n_rows, n_cols,
    stride_in_row, stride_out_row,
    XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr, RBLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    rbase = tl.arange(0, RBLOCK)
    x_loops: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    r_loops: tl.constexpr = (n_cols + RBLOCK - 1) // RBLOCK

    for x_loop in range(x_loops):
        row_idx = xoffset + x_loop * XBLOCK_SUB + tl.arange(0, XBLOCK_SUB)
        row_mask = row_idx < n_rows
        acc = tl.zeros([XBLOCK_SUB, RBLOCK], dtype=tl.float32)
        c = tl.zeros([XBLOCK_SUB, RBLOCK], dtype=tl.float32)

        for r_loop in range(r_loops):
            col_start = r_loop * RBLOCK
            rindex = col_start + rbase
            rmask = rindex < n_cols
            ptrs = in_ptr + row_idx[:, None] * stride_in_row + rindex[None, :]
            block_mask = row_mask[:, None] & rmask[None, :]
            x = tl.load(ptrs, mask=block_mask, other=0.0).to(tl.float32)

            # Kahan: 补偿上次累加的截断误差
            y = x - c
            t = acc + y
            c = (t - acc) - y
            acc = t

        result = tl.sum(acc, axis=1)
        tl.store(out_ptr + row_idx * stride_out_row, result, mask=row_mask)


def reduce_sum(x: torch.Tensor, axis: int = -1, mode: str = "kahan") -> torch.Tensor:
    assert x.is_contiguous(), "Input must be contiguous"
    assert mode in ("simple", "kahan"), f"mode must be 'simple' or 'kahan', got '{mode}'"
    original_shape = x.shape
    n_cols = x.shape[-1]
    n_rows = x.numel() // n_cols
    x_2d = x.reshape(n_rows, n_cols)
    out_shape = original_shape[:-1]
    output = torch.empty(n_rows, device=x.device, dtype=x.dtype)

    num_cores = _get_num_vector_cores()
    xblock = triton.cdiv(n_rows, num_cores)
    xblock_sub = min(xblock, 8)
    # kahan 多一个 c 补偿变量 [xsub, RBLOCK]，需多算一份
    num_ub_tensors = 3 if mode == "kahan" else 2
    rblock = _safe_rblock(n_cols, xblock_sub, num_tensors=num_ub_tensors)

    kernel = _reduce_sum_kahan_kernel if mode == "kahan" else _reduce_sum_kernel
    kernel[(num_cores, 1, 1)](
        x_2d, output, n_rows, n_cols,
        x_2d.stride(0), output.stride(0),
        XBLOCK=xblock, XBLOCK_SUB=xblock_sub, RBLOCK=rblock,
    )
    return output.reshape(out_shape)


def ref_program(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    return torch.sum(x, dim=axis)


def main():
    torch.npu.set_device(0)
    num_cores = _get_num_vector_cores()
    print(f"Vector Cores: {num_cores}")

    tests = [
        ("2D small",      (128, 512),   torch.float32),
        ("2D large cols", (256, 8192),  torch.float32),
        ("2D huge cols",  (64, 32768),  torch.float32),
    ]

    for name, shape, dtype in tests:
        x = torch.randn(*shape, device="npu", dtype=dtype)
        ref = ref_program(x)

        result_simple = reduce_sum(x, mode="simple")
        result_kahan = reduce_sum(x, mode="kahan")

        diff_simple = torch.max(torch.abs(result_simple.cpu() - ref.cpu())).item()
        diff_kahan = torch.max(torch.abs(result_kahan.cpu() - ref.cpu())).item()

        print(f"  {name:<16} simple_diff={diff_simple:.2e}  kahan_diff={diff_kahan:.2e}  "
              f"improve={diff_simple / max(diff_kahan, 1e-12):.1f}x")

    print("\nAll checks done!")


if __name__ == "__main__":
    main()
