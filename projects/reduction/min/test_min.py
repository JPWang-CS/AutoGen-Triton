"""
Triton-Ascend Reduce Min 正确性测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pytest
from min import reduce_min, ref_program as ref_min


SHAPES_2D = [(16, 64), (128, 256), (256, 512), (512, 1024)]
DTYPES = [torch.float32]


class TestReduceMin:

    @pytest.mark.parametrize("shape", SHAPES_2D)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_correctness(self, shape, dtype):
        x = torch.randn(*shape, device="npu", dtype=dtype)
        out = reduce_min(x)
        ref = ref_min(x)
        assert torch.allclose(out.cpu().float(), ref.cpu().float(), rtol=1e-3, atol=1e-3)

    def test_1d(self):
        x = torch.randn(1024, device="npu", dtype=torch.float32)
        out = reduce_min(x)
        ref = ref_min(x)
        assert torch.allclose(out.cpu().float(), ref.cpu().float(), rtol=1e-3, atol=1e-3)

    def test_all_positive(self):
        x = torch.rand(64, 512, device="npu", dtype=torch.float32)
        out = reduce_min(x)
        ref = ref_min(x)
        assert torch.allclose(out.cpu().float(), ref.cpu().float(), rtol=1e-3, atol=1e-3)


def run_all():
    torch.npu.set_device(0)
    all_pass = True
    for shape in [(128, 512), (64, 8192)]:
        x = torch.randn(*shape, device="npu", dtype=torch.float32)
        ref = ref_min(x)
        result = reduce_min(x)
        passed = torch.allclose(result.cpu().float(), ref.cpu().float(), rtol=1e-3, atol=1e-3)
        diff = torch.max(torch.abs(result.cpu().float() - ref.cpu().float())).item()
        if not passed:
            all_pass = False
        print(f"  shape={str(shape):<18} diff={diff:.2e}  {'PASS' if passed else 'FAIL'}")
    print(f"\n{'ALL PASS' if all_pass else 'HAS FAILURES'}")


if __name__ == "__main__":
    run_all()
