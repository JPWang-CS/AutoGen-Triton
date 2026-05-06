"""
Triton-Ascend Reduce Argmax 正确性测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pytest
from argmax import reduce_argmax, ref_program as ref_argmax


SHAPES_2D = [(16, 64), (128, 256), (256, 512), (512, 1024)]
DTYPES = [torch.float32]


class TestReduceArgmax:

    @pytest.mark.parametrize("shape", SHAPES_2D)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_correctness(self, shape, dtype):
        x = torch.randn(*shape, device="npu", dtype=dtype)
        out = reduce_argmax(x)
        ref = ref_argmax(x)
        assert torch.equal(out.cpu(), ref.cpu()), \
            f"reduce_argmax {shape} {dtype} failed"

    def test_1d(self):
        x = torch.randn(1024, device="npu", dtype=torch.float32)
        out = reduce_argmax(x)
        ref = ref_argmax(x)
        assert torch.equal(out.cpu(), ref.cpu())

    def test_known_value(self):
        x = torch.zeros(4, 8, device="npu", dtype=torch.float32)
        x[0, 3] = 1.0
        x[1, 7] = 5.0
        x[2, 0] = -1.0
        x[3, 5] = 10.0
        out = reduce_argmax(x)
        ref = ref_argmax(x)
        assert torch.equal(out.cpu(), ref.cpu())


def run_all():
    torch.npu.set_device(0)
    all_pass = True
    for shape in [(128, 512), (64, 8192)]:
        x = torch.randn(*shape, device="npu", dtype=torch.float32)
        ref = ref_argmax(x)
        result = reduce_argmax(x)
        passed = torch.equal(result.cpu(), ref.cpu())
        if not passed:
            all_pass = False
        print(f"  shape={str(shape):<18} {'PASS' if passed else 'FAIL'}")
    print(f"\n{'ALL PASS' if all_pass else 'HAS FAILURES'}")


if __name__ == "__main__":
    run_all()
