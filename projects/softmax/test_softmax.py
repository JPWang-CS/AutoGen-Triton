"""
Triton-Ascend Fused Softmax 测试

覆盖测试场景：
- 基础正确性
- 不同矩阵规模
- 不同数据类型
- 高维输入（3D/4D）
- 数值稳定性（含大/小值）
- 输出行和为 1

所有计算在 NPU 上执行，精度对比在 CPU 上进行。
"""

import torch
import pytest

from softmax import softmax


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
class TestSoftmax:
    """Fused Softmax NPU 测试类"""

    def setup_method(self):
        setup_npu()

    def test_basic_correctness(self):
        """基础正确性测试"""
        x = torch.randn(128, 512, device="npu", dtype=torch.float32)
        output = softmax(x)
        ref_output = torch.nn.functional.softmax(x.cpu(), dim=-1)
        torch.testing.assert_close(output.cpu(), ref_output, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("num_rows,num_cols", [
        (1, 64),
        (16, 128),
        (64, 256),
        (128, 512),
        (256, 1024),
    ])
    def test_various_shapes(self, num_rows, num_cols):
        """不同矩阵规模测试"""
        x = torch.randn(num_rows, num_cols, device="npu", dtype=torch.float32)
        output = softmax(x)
        ref_output = torch.nn.functional.softmax(x.cpu(), dim=-1)
        torch.testing.assert_close(output.cpu(), ref_output, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("dtype", [
        torch.float32,
        torch.float16,
    ])
    def test_dtypes(self, dtype):
        """不同数据类型测试"""
        x = torch.randn(64, 256, device="npu", dtype=dtype)
        output = softmax(x)
        ref_output = torch.nn.functional.softmax(x.cpu(), dim=-1)
        rtol = 1e-5 if dtype == torch.float32 else 1e-3
        atol = 1e-5 if dtype == torch.float32 else 1e-3
        torch.testing.assert_close(output.cpu(), ref_output, rtol=rtol, atol=atol)

    def test_3d_input(self):
        """3D 输入测试 (batch, seq, dim)"""
        x = torch.randn(4, 32, 128, device="npu", dtype=torch.float32)
        output = softmax(x)
        ref_output = torch.nn.functional.softmax(x.cpu(), dim=-1)
        torch.testing.assert_close(output.cpu(), ref_output, rtol=1e-5, atol=1e-5)

    def test_4d_input(self):
        """4D 输入测试 (batch, heads, seq, dim)"""
        x = torch.randn(2, 8, 32, 64, device="npu", dtype=torch.float32)
        output = softmax(x)
        ref_output = torch.nn.functional.softmax(x.cpu(), dim=-1)
        torch.testing.assert_close(output.cpu(), ref_output, rtol=1e-5, atol=1e-5)

    def test_row_sums_to_one(self):
        """验证每行和为 1"""
        x = torch.randn(32, 128, device="npu", dtype=torch.float32)
        output = softmax(x)
        row_sums = output.cpu().sum(dim=-1)
        expected = torch.ones(32, dtype=torch.float32)
        torch.testing.assert_close(row_sums, expected, rtol=1e-5, atol=1e-5)

    def test_numerical_stability(self):
        """数值稳定性测试（含大/小值）"""
        x = torch.zeros(4, 64, device="npu", dtype=torch.float32)
        x[0, :] = 100.0   # 大值
        x[1, :] = -100.0  # 小值
        x[2, :] = 0.0     # 零
        x[3, 0] = 1000.0  # 单个大值
        x[3, 1:] = -1000.0

        output = softmax(x)
        ref_output = torch.nn.functional.softmax(x.cpu(), dim=-1)
        torch.testing.assert_close(output.cpu(), ref_output, rtol=1e-4, atol=1e-4)

    def test_all_equal_values(self):
        """所有值相同时输出应均匀分布"""
        x = torch.ones(4, 64, device="npu", dtype=torch.float32) * 5.0
        output = softmax(x)
        expected = torch.ones(4, 64, dtype=torch.float32) / 64.0
        torch.testing.assert_close(output.cpu(), expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
