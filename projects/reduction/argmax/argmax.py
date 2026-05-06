"""
Triton-Ascend Reduce Argmax

output = argmax(x, axis=-1) — 返回最大值索引 (int32)
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
def _reduce_argmax_kernel(
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
        best_val = tl.full([XBLOCK_SUB], float('-inf'), dtype=tl.float32)
        best_idx = tl.zeros([XBLOCK_SUB], dtype=tl.int32)

        for r_loop in range(r_loops):
            col_start = r_loop * RBLOCK
            rindex = col_start + rbase
            rmask = rindex < n_cols
            ptrs = in_ptr + row_idx[:, None] * stride_in_row + rindex[None, :]
            block_mask = row_mask[:, None] & rmask[None, :]
            x = tl.load(ptrs, mask=block_mask, other=float('-inf')).to(tl.float32)

            local_max = tl.max(x, axis=1)
            local_idx = tl.argmax(x, axis=1).to(tl.int32) + col_start

            is_better = local_max > best_val
            best_val = tl.where(is_better, local_max, best_val)
            best_idx = tl.where(is_better, local_idx, best_idx)

        tl.store(out_ptr + row_idx * stride_out_row, best_idx, mask=row_mask)


def reduce_argmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    assert x.is_contiguous(), "Input must be contiguous"
    original_shape = x.shape
    n_cols = x.shape[-1]
    n_rows = x.numel() // n_cols
    x_2d = x.reshape(n_rows, n_cols)
    out_shape = original_shape[:-1]
    output = torch.empty(n_rows, device=x.device, dtype=torch.int32)

    num_cores = _get_num_vector_cores()
    xblock = triton.cdiv(n_rows, num_cores)
    xblock_sub = min(xblock, 8)
    rblock = _safe_rblock(n_cols, xblock_sub)

    _reduce_argmax_kernel[(num_cores, 1, 1)](
        x_2d, output, n_rows, n_cols,
        x_2d.stride(0), output.stride(0),
        XBLOCK=xblock, XBLOCK_SUB=xblock_sub, RBLOCK=rblock,
    )
    return output.reshape(out_shape)


def ref_program(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    return torch.argmax(x, dim=axis)


def main():
    torch.npu.set_device(0)
    num_cores = _get_num_vector_cores()
    print(f"Vector Cores: {num_cores}")

    tests = [
        ("2D small",      (128, 512),   torch.float32),
        ("2D large cols", (256, 8192),  torch.float32),
    ]

    for name, shape, dtype in tests:
        x = torch.randn(*shape, device="npu", dtype=dtype)
        ref = ref_program(x)
        result = reduce_argmax(x)
        diff = (result.cpu() != ref.cpu()).sum().item()
        passed = diff == 0
        print(f"  {name:<16} mismatch={diff} {'PASS' if passed else 'FAIL'}")

    print("\nAll checks done!")


if __name__ == "__main__":
    main()
