"""
Triton-Ascend 向量加法测试

覆盖三种模式: naive / persistent / optimized
"""

import torch
import pytest

from vector_add import vector_add


def npu_available():
    try:
        import torch_npu
        return torch.npu.is_available()
    except ImportError:
        return False


def setup_npu():
    import torch_npu
    torch.npu.set_device(0)


@pytest.mark.skipif(not npu_available(), reason="NPU not available")
class TestVectorAdd:

    def setup_method(self):
        setup_npu()

    @pytest.mark.parametrize("mode", ["naive", "persistent", "optimized"])
    def test_basic_correctness(self, mode):
        a = torch.randn(1024, device="npu", dtype=torch.float32)
        b = torch.randn(1024, device="npu", dtype=torch.float32)
        c = vector_add(a, b, mode=mode)
        ref_c = a.cpu() + b.cpu()
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("mode", ["naive", "persistent", "optimized"])
    @pytest.mark.parametrize("size", [64, 128, 256, 1024, 4096, 65536, 1048576])
    def test_various_sizes(self, mode, size):
        a = torch.randn(size, device="npu", dtype=torch.float32)
        b = torch.randn(size, device="npu", dtype=torch.float32)
        c = vector_add(a, b, mode=mode)
        ref_c = a.cpu() + b.cpu()
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("mode", ["naive", "persistent", "optimized"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_dtypes(self, mode, dtype):
        a = torch.randn(1024, device="npu", dtype=dtype)
        b = torch.randn(1024, device="npu", dtype=dtype)
        c = vector_add(a, b, mode=mode)
        ref_c = a.cpu() + b.cpu()
        rtol = 1e-6 if dtype == torch.float32 else 1e-3
        atol = 1e-6 if dtype == torch.float32 else 1e-3
        torch.testing.assert_close(c.cpu(), ref_c, rtol=rtol, atol=atol)

    def test_multidimensional(self):
        a = torch.randn(32, 64, device="npu", dtype=torch.float32)
        b = torch.randn(32, 64, device="npu", dtype=torch.float32)
        c = vector_add(a, b, mode="optimized")
        ref_c = a.cpu() + b.cpu()
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("xblock_sub", [256, 512, 1024, 2048])
    def test_xblock_sub_sizes(self, xblock_sub):
        a = torch.randn(4096, device="npu", dtype=torch.float32)
        b = torch.randn(4096, device="npu", dtype=torch.float32)
        c = vector_add(a, b, mode="optimized", xblock_sub=xblock_sub)
        ref_c = a.cpu() + b.cpu()
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
