"""
Triton-Ascend Reduce XOR 正确性测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pytest
from xor import reduce_xor, ref_program as ref_xor


SHAPES_2D = [(16, 64), (128, 256), (256, 512), (512, 1024)]
DTYPES = [torch.int32]


class TestReduceXor:

    @pytest.mark.parametrize("shape", SHAPES_2D)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_correctness(self, shape, dtype):
        x = torch.randint(-1000, 1000, shape, device="npu", dtype=dtype)
        out = reduce_xor(x)
        ref = ref_xor(x)
        assert torch.equal(out.cpu(), ref.cpu()), \
            f"reduce_xor {shape} {dtype} failed"

    def test_1d(self):
        x = torch.randint(-1000, 1000, (1024,), device="npu", dtype=torch.int32)
        out = reduce_xor(x)
        ref = ref_xor(x)
        assert torch.equal(out.cpu(), ref.cpu())

    def test_zeros(self):
        x = torch.zeros(64, 128, device="npu", dtype=torch.int32)
        out = reduce_xor(x)
        assert torch.equal(out.cpu(), torch.zeros(64, dtype=torch.int32))

    def test_self_xor(self):
        x = torch.randint(0, 100, (32, 64), device="npu", dtype=torch.int32)
        y = x.clone()
        out = reduce_xor(torch.stack([x, y], dim=1).reshape(32, 128))
        assert torch.equal(out.cpu(), torch.zeros(32, dtype=torch.int32))


def run_all():
    torch.npu.set_device(0)
    all_pass = True
    for shape in [(128, 512), (64, 8192)]:
        x = torch.randint(-1000, 1000, shape, device="npu", dtype=torch.int32)
        ref = ref_xor(x)
        result = reduce_xor(x)
        passed = torch.equal(result.cpu(), ref.cpu())
        if not passed:
            all_pass = False
        print(f"  shape={str(shape):<18} {'PASS' if passed else 'FAIL'}")
    print(f"\n{'ALL PASS' if all_pass else 'HAS FAILURES'}")


if __name__ == "__main__":
    run_all()
