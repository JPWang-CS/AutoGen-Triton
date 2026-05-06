"""
Triton-Ascend Reduce Prod 正确性测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pytest
from prod import reduce_prod, ref_program as ref_prod


SHAPES_2D = [(16, 64), (128, 256), (256, 512), (512, 1024)]
DTYPES = [torch.float32]


class TestReduceProd:

    @pytest.mark.parametrize("shape", SHAPES_2D)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_correctness(self, shape, dtype):
        x = torch.randn(*shape, device="npu", dtype=dtype) * 0.5
        out = reduce_prod(x)
        ref = ref_prod(x)
        assert torch.allclose(out.cpu().float(), ref.cpu().float(), rtol=1e-2, atol=1e-2), \
            f"reduce_prod {shape} {dtype} failed, max_diff={torch.max(torch.abs(out.cpu().float() - ref.cpu().float()))}"

    def test_1d(self):
        x = torch.randn(64, device="npu", dtype=torch.float32) * 0.5
        out = reduce_prod(x)
        ref = ref_prod(x)
        assert torch.allclose(out.cpu().float(), ref.cpu().float(), rtol=1e-2, atol=1e-2)

    def test_ones(self):
        x = torch.ones(32, 16, device="npu", dtype=torch.float32)
        out = reduce_prod(x)
        ref = ref_prod(x)
        assert torch.allclose(out.cpu(), ref.cpu(), rtol=1e-3, atol=1e-3)


def run_all():
    torch.npu.set_device(0)
    all_pass = True
    for shape in [(128, 64), (64, 256)]:
        x = torch.randn(*shape, device="npu", dtype=torch.float32) * 0.5
        ref = ref_prod(x)
        result = reduce_prod(x)
        passed = torch.allclose(result.cpu().float(), ref.cpu().float(), rtol=1e-2, atol=1e-2)
        diff = torch.max(torch.abs(result.cpu().float() - ref.cpu().float())).item()
        if not passed:
            all_pass = False
        print(f"  shape={str(shape):<18} diff={diff:.2e}  {'PASS' if passed else 'FAIL'}")
    print(f"\n{'ALL PASS' if all_pass else 'HAS FAILURES'}")


if __name__ == "__main__":
    run_all()
