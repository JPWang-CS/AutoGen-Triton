"""
Triton-Ascend 向量加法测试

覆盖测试场景：
- Naive 和 Persistent 两种模式
- 基础正确性
- 不同向量规模
- 不同数据类型
- 多维输入
- 不同 BLOCK_SIZE

所有计算在 NPU 上执行，精度对比在 CPU 上进行。
"""

import torch
import pytest

from vector_add import vector_add


# ============================================================================
# 测试辅助函数
# ============================================================================

def npu_available():
    """检查 NPU 是否可用"""
    try:
        import torch_npu
        return torch.npu.is_available()
    except ImportError:
        return False


def setup_npu():
    """设置 NPU 设备"""
    import torch_npu
    torch.npu.set_device(0)


# ============================================================================
# 测试类
# ============================================================================

@pytest.mark.skipif(not npu_available(), reason="NPU not available")
class TestVectorAdd:
    """向量加法 NPU 测试类"""

    def setup_method(self):
        setup_npu()

    def test_basic_correctness_persistent(self):
        """Persistent 模式基础正确性"""
        a = torch.randn(1024, device="npu", dtype=torch.float32)
        b = torch.randn(1024, device="npu", dtype=torch.float32)
        c = vector_add(a, b, persistent=True)
        ref_c = a.cpu() + b.cpu()
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-6, atol=1e-6)

    def test_basic_correctness_naive(self):
        """Naive 模式基础正确性"""
        a = torch.randn(1024, device="npu", dtype=torch.float32)
        b = torch.randn(1024, device="npu", dtype=torch.float32)
        c = vector_add(a, b, persistent=False)
        ref_c = a.cpu() + b.cpu()
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("persistent", [True, False])
    @pytest.mark.parametrize("size", [
        64, 128, 256, 512, 1024, 4096, 65536, 1048576,
    ])
    def test_various_sizes(self, persistent, size):
        """不同向量规模测试 (两种模式)"""
        a = torch.randn(size, device="npu", dtype=torch.float32)
        b = torch.randn(size, device="npu", dtype=torch.float32)
        c = vector_add(a, b, persistent=persistent)
        ref_c = a.cpu() + b.cpu()
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("persistent", [True, False])
    @pytest.mark.parametrize("dtype", [
        torch.float32,
        torch.float16,
    ])
    def test_dtypes(self, persistent, dtype):
        """不同数据类型测试 (两种模式)"""
        a = torch.randn(1024, device="npu", dtype=dtype)
        b = torch.randn(1024, device="npu", dtype=dtype)
        c = vector_add(a, b, persistent=persistent)
        ref_c = a.cpu() + b.cpu()
        rtol = 1e-6 if dtype == torch.float32 else 1e-3
        atol = 1e-6 if dtype == torch.float32 else 1e-3
        torch.testing.assert_close(c.cpu(), ref_c, rtol=rtol, atol=atol)

    def test_multidimensional_persistent(self):
        """多维输入测试 (persistent)"""
        a = torch.randn(32, 64, device="npu", dtype=torch.float32)
        b = torch.randn(32, 64, device="npu", dtype=torch.float32)
        c = vector_add(a, b, persistent=True)
        ref_c = a.cpu() + b.cpu()
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("block_size", [256, 512, 1024, 2048])
    def test_block_sizes_persistent(self, block_size):
        """不同 BLOCK_SIZE 测试 (persistent)"""
        a = torch.randn(4096, device="npu", dtype=torch.float32)
        b = torch.randn(4096, device="npu", dtype=torch.float32)
        c = vector_add(a, b, block_size=block_size, persistent=True)
        ref_c = a.cpu() + b.cpu()
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
