"""
Triton-Ascend Reduce Sum 测试

覆盖场景:
- 1D 全量 sum (axis=None)
- 2D/3D last axis sum (naive / optimized)
- 不同数据类型 (float32, float16)
- 非对齐维度
- 大 reduce axis (RBLOCK 分块循环)
- 输出形状验证
"""

import torch
import pytest

from reduce_sum import reduce_sum


def npu_available():
    try:
        import torch_npu
        return torch.npu.is_available()
    except ImportError:
        return False


def setup_npu():
    import torch_npu
    torch.npu.set_device(0)


def ref(x, axis=-1):
    return torch.sum(x, dim=axis)


@pytest.mark.skipif(not npu_available(), reason="NPU not available")
class TestReduceSum:

    def setup_method(self):
        setup_npu()

    # --- 1D 全量 sum ---

    def test_1d_full_sum(self):
        x = torch.randn(1024, device="npu", dtype=torch.float32)
        result = reduce_sum(x, axis=None)
        expected = ref(x, axis=None)
        torch.testing.assert_close(result.cpu(), expected.cpu(), rtol=1e-4, atol=1e-4)

    def test_1d_full_sum_small(self):
        x = torch.randn(37, device="npu", dtype=torch.float32)
        result = reduce_sum(x, axis=None)
        expected = ref(x, axis=None)
        torch.testing.assert_close(result.cpu(), expected.cpu(), rtol=1e-4, atol=1e-4)

    # --- 2D last axis ---

    @pytest.mark.parametrize("mode", ["naive", "optimized"])
    def test_2d_basic(self, mode):
        x = torch.randn(128, 512, device="npu", dtype=torch.float32)
        result = reduce_sum(x, axis=-1, mode=mode)
        expected = ref(x, axis=-1)
        torch.testing.assert_close(result.cpu(), expected.cpu(), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("mode", ["naive", "optimized"])
    @pytest.mark.parametrize("shape", [
        (64, 64), (256, 1024), (1024, 2048),
    ])
    def test_2d_shapes(self, mode, shape):
        x = torch.randn(*shape, device="npu", dtype=torch.float32)
        result = reduce_sum(x, axis=-1, mode=mode)
        expected = ref(x, axis=-1)
        torch.testing.assert_close(result.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3)

    # --- 3D ---

    @pytest.mark.parametrize("mode", ["naive", "optimized"])
    def test_3d(self, mode):
        x = torch.randn(8, 128, 64, device="npu", dtype=torch.float32)
        result = reduce_sum(x, axis=-1, mode=mode)
        expected = ref(x, axis=-1)
        torch.testing.assert_close(result.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3)

    # --- dtype ---

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_dtypes(self, dtype):
        x = torch.randn(128, 512, device="npu", dtype=dtype)
        result = reduce_sum(x, axis=-1, mode="optimized")
        expected = ref(x, axis=-1)
        rtol = 1e-4 if dtype == torch.float32 else 1e-3
        torch.testing.assert_close(result.cpu().float(), expected.cpu().float(), rtol=rtol, atol=rtol)

    # --- 非对齐维度 ---

    def test_non_power_of_2_cols(self):
        x = torch.randn(128, 65, device="npu", dtype=torch.float32)
        result = reduce_sum(x, axis=-1, mode="optimized")
        expected = ref(x, axis=-1)
        torch.testing.assert_close(result.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3)

    # --- 大 reduce axis (RBLOCK 分块) ---

    @pytest.mark.parametrize("mode", ["naive", "optimized"])
    def test_large_reduce_axis(self, mode):
        x = torch.randn(256, 8192, device="npu", dtype=torch.float32)
        result = reduce_sum(x, axis=-1, mode=mode)
        expected = ref(x, axis=-1)
        torch.testing.assert_close(result.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3)

    # --- 输出形状 ---

    def test_output_shape_2d(self):
        x = torch.randn(128, 512, device="npu", dtype=torch.float32)
        result = reduce_sum(x, axis=-1)
        assert result.shape == (128,), f"Expected (128,), got {result.shape}"

    def test_output_shape_3d(self):
        x = torch.randn(8, 128, 64, device="npu", dtype=torch.float32)
        result = reduce_sum(x, axis=-1)
        assert result.shape == (8, 128), f"Expected (8, 128), got {result.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
