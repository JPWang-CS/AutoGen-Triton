"""
Triton-Ascend Layer Normalization 测试

覆盖测试场景：
- 基础正确性（不带 gamma/beta）
- 带 gamma/beta 的仿射变换
- 不同矩阵规模
- 不同数据类型
- 高维输入（3D）
- 数值稳定性
- 输出均值为 0、方差为 1

所有计算在 NPU 上执行，精度对比在 CPU 上进行。
"""

import torch
import pytest

from layer_norm import layer_norm


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
class TestLayerNorm:
    """Layer Normalization NPU 测试类"""

    def setup_method(self):
        setup_npu()

    def test_basic_correctness(self):
        """基础正确性测试（不带 gamma/beta）"""
        x = torch.randn(128, 512, device="npu", dtype=torch.float32)
        output = layer_norm(x)
        ref_output = torch.nn.functional.layer_norm(x.cpu(), [512])
        torch.testing.assert_close(output.cpu(), ref_output, rtol=1e-4, atol=1e-4)

    def test_with_gamma_beta(self):
        """带 gamma/beta 的正确性测试"""
        batch_size, hidden_dim = 64, 256
        x = torch.randn(batch_size, hidden_dim, device="npu", dtype=torch.float32)
        gamma = torch.randn(hidden_dim, device="npu", dtype=torch.float32)
        beta = torch.randn(hidden_dim, device="npu", dtype=torch.float32)

        output = layer_norm(x, gamma, beta)
        ref_output = torch.nn.functional.layer_norm(
            x.cpu(), [hidden_dim], weight=gamma.cpu(), bias=beta.cpu()
        )
        torch.testing.assert_close(output.cpu(), ref_output, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("num_rows,num_cols", [
        (16, 128),
        (64, 256),
        (128, 512),
        (256, 1024),
    ])
    def test_various_shapes(self, num_rows, num_cols):
        """不同矩阵规模测试"""
        x = torch.randn(num_rows, num_cols, device="npu", dtype=torch.float32)
        output = layer_norm(x)
        ref_output = torch.nn.functional.layer_norm(x.cpu(), [num_cols])
        torch.testing.assert_close(output.cpu(), ref_output, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("dtype", [
        torch.float32,
        torch.float16,
    ])
    def test_dtypes(self, dtype):
        """不同数据类型测试"""
        x = torch.randn(64, 256, device="npu", dtype=dtype)
        output = layer_norm(x)
        ref_output = torch.nn.functional.layer_norm(x.cpu(), [256])
        rtol = 1e-4 if dtype == torch.float32 else 1e-2
        atol = 1e-4 if dtype == torch.float32 else 1e-2
        torch.testing.assert_close(output.cpu(), ref_output, rtol=rtol, atol=atol)

    def test_3d_input(self):
        """3D 输入测试 (batch, seq, hidden)"""
        x = torch.randn(4, 32, 256, device="npu", dtype=torch.float32)
        output = layer_norm(x)
        ref_output = torch.nn.functional.layer_norm(x.cpu(), [256])
        torch.testing.assert_close(output.cpu(), ref_output, rtol=1e-4, atol=1e-4)

    def test_output_statistics(self):
        """验证输出均值为 0、方差为 1（不带 gamma/beta）"""
        x = torch.randn(64, 256, device="npu", dtype=torch.float32)
        output = layer_norm(x)

        # 每行的均值应接近 0
        row_means = output.cpu().mean(dim=-1)
        expected_zero = torch.zeros(64, dtype=torch.float32)
        torch.testing.assert_close(row_means, expected_zero, rtol=1e-4, atol=1e-4)

        # 每行的方差应接近 1
        row_vars = output.cpu().var(dim=-1, unbiased=False)
        expected_one = torch.ones(64, dtype=torch.float32)
        torch.testing.assert_close(row_vars, expected_one, rtol=1e-3, atol=1e-3)

    def test_different_eps(self):
        """不同 eps 值测试"""
        x = torch.randn(32, 128, device="npu", dtype=torch.float32)
        for eps in [1e-5, 1e-3, 1e-1]:
            output = layer_norm(x, eps=eps)
            ref_output = torch.nn.functional.layer_norm(x.cpu(), [128], eps=eps)
            torch.testing.assert_close(output.cpu(), ref_output, rtol=1e-4, atol=1e-4)

    def test_gamma_only(self):
        """仅带 gamma（不带 beta）"""
        x = torch.randn(32, 128, device="npu", dtype=torch.float32)
        gamma = torch.randn(128, device="npu", dtype=torch.float32)
        output = layer_norm(x, gamma=gamma)
        ref_output = torch.nn.functional.layer_norm(
            x.cpu(), [128], weight=gamma.cpu()
        )
        torch.testing.assert_close(output.cpu(), ref_output, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
