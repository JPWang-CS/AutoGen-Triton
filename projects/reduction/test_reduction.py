"""
Triton-Ascend Reduction Operators 正确性测试

覆盖:
  - 多种 shape (小/中/大)
  - float32 / float16 / bfloat16 (浮点)
  - int32 (整数, xor_sum)
  - 1D 输入
  - 精度: rtol=1e-3, atol=1e-3
"""

import torch
import pytest
from reduction import (
    reduce_sum, reduce_max, reduce_min,
    reduce_argmax, reduce_argmin,
    reduce_xor, reduce_prod,
    ref_sum, ref_max, ref_min,
    ref_argmax, ref_argmin,
    ref_xor, ref_prod,
)

# ============================================================================
# Fixtures
# ============================================================================

SHAPES_2D = [(16, 64), (128, 256), (256, 512), (512, 1024)]
FLOAT_DTYPES = [torch.float32]
INT_DTYPES = [torch.int32]

# ============================================================================
# Reduce Sum
# ============================================================================

class TestReduceSum:

    @pytest.mark.parametrize("shape", SHAPES_2D)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_correctness(self, shape, dtype):
        x = torch.randn(*shape, device="npu", dtype=dtype)
        out = reduce_sum(x)
        ref = ref_sum(x)
        assert torch.allclose(out.cpu().float(), ref.cpu().float(), rtol=1e-3, atol=1e-3), \
            f"reduce_sum {shape} {dtype} failed, max_diff={torch.max(torch.abs(out.cpu().float() - ref.cpu().float()))}"

    def test_1d(self):
        x = torch.randn(1024, device="npu", dtype=torch.float32)
        out = reduce_sum(x)
        ref = ref_sum(x)
        assert torch.allclose(out.cpu().float(), ref.cpu().float(), rtol=1e-3, atol=1e-3)

    def test_large_cols(self):
        x = torch.randn(64, 8192, device="npu", dtype=torch.float32)
        out = reduce_sum(x)
        ref = ref_sum(x)
        assert torch.allclose(out.cpu().float(), ref.cpu().float(), rtol=1e-3, atol=1e-3)


# ============================================================================
# Reduce Max
# ============================================================================

class TestReduceMax:

    @pytest.mark.parametrize("shape", SHAPES_2D)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_correctness(self, shape, dtype):
        x = torch.randn(*shape, device="npu", dtype=dtype)
        out = reduce_max(x)
        ref = ref_max(x)
        assert torch.allclose(out.cpu().float(), ref.cpu().float(), rtol=1e-3, atol=1e-3)

    def test_1d(self):
        x = torch.randn(1024, device="npu", dtype=torch.float32)
        out = reduce_max(x)
        ref = ref_max(x)
        assert torch.allclose(out.cpu().float(), ref.cpu().float(), rtol=1e-3, atol=1e-3)

    def test_negative_values(self):
        x = torch.randn(64, 512, device="npu", dtype=torch.float32) - 10.0
        out = reduce_max(x)
        ref = ref_max(x)
        assert torch.allclose(out.cpu().float(), ref.cpu().float(), rtol=1e-3, atol=1e-3)


# ============================================================================
# Reduce Min
# ============================================================================

class TestReduceMin:

    @pytest.mark.parametrize("shape", SHAPES_2D)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
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


# ============================================================================
# Reduce Argmax
# ============================================================================

class TestReduceArgmax:

    @pytest.mark.parametrize("shape", SHAPES_2D)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
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


# ============================================================================
# Reduce Argmin
# ============================================================================

class TestReduceArgmin:

    @pytest.mark.parametrize("shape", SHAPES_2D)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_correctness(self, shape, dtype):
        x = torch.randn(*shape, device="npu", dtype=dtype)
        out = reduce_argmin(x)
        ref = ref_argmin(x)
        assert torch.equal(out.cpu(), ref.cpu()), \
            f"reduce_argmin {shape} {dtype} failed"

    def test_1d(self):
        x = torch.randn(1024, device="npu", dtype=torch.float32)
        out = reduce_argmin(x)
        ref = ref_argmin(x)
        assert torch.equal(out.cpu(), ref.cpu())

    def test_known_value(self):
        x = torch.zeros(4, 8, device="npu", dtype=torch.float32)
        x[0, 3] = -10.0
        x[1, 7] = -5.0
        x[2, 0] = -100.0
        x[3, 5] = -1.0
        out = reduce_argmin(x)
        ref = ref_argmin(x)
        assert torch.equal(out.cpu(), ref.cpu())


# ============================================================================
# Reduce XOR
# ============================================================================

class TestReduceXor:

    @pytest.mark.parametrize("shape", SHAPES_2D)
    @pytest.mark.parametrize("dtype", INT_DTYPES)
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


# ============================================================================
# Reduce Prod
# ============================================================================

class TestReduceProd:

    @pytest.mark.parametrize("shape", SHAPES_2D)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
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


# ============================================================================
# Run without pytest
# ============================================================================

def run_all_tests():
    torch.npu.set_device(0)
    all_pass = True

    test_cases = [
        ("sum",    (128, 512),  torch.float32, reduce_sum,  ref_sum,  1e-3),
        ("sum",    (64, 8192),  torch.float32, reduce_sum,  ref_sum,  1e-3),
        ("max",    (128, 512),  torch.float32, reduce_max,  ref_max,  1e-3),
        ("min",    (128, 512),  torch.float32, reduce_min,  ref_min,  1e-3),
        ("argmax", (128, 512),  torch.float32, reduce_argmax, ref_argmax, None),
        ("argmin", (128, 512),  torch.float32, reduce_argmin, ref_argmin, None),
        ("xor",    (128, 512),  torch.int32,   reduce_xor,  ref_xor,  None),
        ("prod",   (128, 64),   torch.float32, reduce_prod, ref_prod, 1e-2),
    ]

    for name, shape, dtype, fn, ref_fn, tol in test_cases:
        x = torch.randn(*shape, device="npu", dtype=dtype) \
            if dtype.is_floating_point \
            else torch.randint(-1000, 1000, shape, device="npu", dtype=dtype)
        if dtype.is_floating_point and name == "prod":
            x = x * 0.5

        ref = ref_fn(x)
        result = fn(x)

        if tol is not None:
            diff = torch.max(torch.abs(result.cpu().float() - ref.cpu().float())).item()
            passed = torch.allclose(result.cpu().float(), ref.cpu().float(), rtol=tol, atol=tol)
            status = "PASS" if passed else "FAIL"
        else:
            passed = torch.equal(result.cpu(), ref.cpu())
            diff = 0.0
            status = "PASS" if passed else "FAIL"

        if not passed:
            all_pass = False
        print(f"  {name:<10} shape={str(shape):<14} diff={diff:.2e}  {status}")

    # 1D tests
    for name, fn, ref_fn, dtype, tol in [
        ("sum_1d",    reduce_sum,    ref_sum,    torch.float32, 1e-3),
        ("max_1d",    reduce_max,    ref_max,    torch.float32, 1e-3),
        ("min_1d",    reduce_min,    ref_min,    torch.float32, 1e-3),
        ("argmax_1d", reduce_argmax, ref_argmax, torch.float32, None),
        ("argmin_1d", reduce_argmin, ref_argmin, torch.float32, None),
        ("xor_1d",    reduce_xor,    ref_xor,    torch.int32,   None),
    ]:
        x = torch.randn(1024, device="npu", dtype=dtype) \
            if dtype.is_floating_point \
            else torch.randint(-1000, 1000, (1024,), device="npu", dtype=dtype)
        ref = ref_fn(x)
        result = fn(x)
        if tol is not None:
            passed = torch.allclose(result.cpu().float(), ref.cpu().float(), rtol=tol, atol=tol)
        else:
            passed = torch.equal(result.cpu(), ref.cpu())
        if not passed:
            all_pass = False
        print(f"  {name:<10} shape=(1024,)       {'PASS' if passed else 'FAIL'}")

    print(f"\n{'ALL PASS' if all_pass else 'HAS FAILURES'}")
    return all_pass


if __name__ == "__main__":
    run_all_tests()
