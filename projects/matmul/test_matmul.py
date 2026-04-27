"""
Triton-Ascend 矩阵乘法测试

覆盖测试场景：
- 基础正确性
- 方阵（不同规模）
- 非方阵
- 不同数据类型
- 小矩阵和边界情况

所有计算在 NPU 上执行，精度对比在 CPU 上进行。
"""

import torch
import pytest

from matmul import matmul


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
class TestMatmul:
    """矩阵乘法 NPU 测试类"""

    def setup_method(self):
        setup_npu()

    def test_basic_correctness(self):
        """基础正确性测试"""
        M, N, K = 128, 128, 128
        a = torch.randn(M, K, device="npu", dtype=torch.float16)
        b = torch.randn(K, N, device="npu", dtype=torch.float16)

        c = matmul(a, b)
        ref_c = a.cpu() @ b.cpu()
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("M,N,K", [
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ])
    def test_square_shapes(self, M, N, K):
        """方阵测试"""
        a = torch.randn(M, K, device="npu", dtype=torch.float16)
        b = torch.randn(K, N, device="npu", dtype=torch.float16)

        c = matmul(a, b)
        ref_c = a.cpu() @ b.cpu()
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("M,N,K", [
        (1024, 512, 256),
        (512, 1024, 256),
        (256, 512, 1024),
        (2048, 128, 512),
        (128, 2048, 512),
    ])
    def test_non_square(self, M, N, K):
        """非方阵测试"""
        a = torch.randn(M, K, device="npu", dtype=torch.float16)
        b = torch.randn(K, N, device="npu", dtype=torch.float16)

        c = matmul(a, b)
        ref_c = a.cpu() @ b.cpu()
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", [
        torch.float16,
        torch.float32,
    ])
    def test_dtypes(self, dtype):
        """不同数据类型测试"""
        M, N, K = 256, 256, 256
        a = torch.randn(M, K, device="npu", dtype=dtype)
        b = torch.randn(K, N, device="npu", dtype=dtype)

        c = matmul(a, b)
        ref_c = a.cpu() @ b.cpu()
        rtol = 1e-3 if dtype == torch.float32 else 1e-2
        atol = 1e-3 if dtype == torch.float32 else 1e-2
        torch.testing.assert_close(c.cpu(), ref_c, rtol=rtol, atol=atol)

    def test_small_matrices(self):
        """小矩阵测试"""
        for size in [16, 32, 48, 64]:
            a = torch.randn(size, size, device="npu", dtype=torch.float16)
            b = torch.randn(size, size, device="npu", dtype=torch.float16)
            c = matmul(a, b)
            ref_c = a.cpu() @ b.cpu()
            torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
