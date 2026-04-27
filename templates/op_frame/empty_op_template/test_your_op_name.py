"""
Triton-Ascend 空算子测试模板

基于 pytest 框架的单元测试，可作为其他算子测试的参考。
测试内容覆盖：基础正确性、不同 shape、不同数据类型。
所有计算在 NPU 上执行，精度对比在 CPU 上进行。
"""

import torch
import pytest

from your_op_name import your_op


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


def ref_your_op_name(x: torch.Tensor) -> torch.Tensor:
    """PyTorch 参考实现（CPU 上执行）"""
    # TODO: 根据实际计算逻辑修改参考实现
    return x


# ============================================================================
# 测试类
# ============================================================================

@pytest.mark.skipif(not npu_available(), reason="NPU not available")
class TestYourOpName:
    """空算子 NPU 测试类"""

    def setup_method(self):
        setup_npu()

    def test_basic_correctness(self):
        """基础正确性测试"""
        x = torch.randn(1024, device="npu", dtype=torch.float32)
        output = your_op(x)
        ref_output = ref_your_op_name(x.cpu())
        torch.testing.assert_close(output.cpu(), ref_output, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("shape", [
        (64,),
        (128,),
        (256,),
        (512,),
        (1024,),
        (4096,),
    ])
    def test_various_shapes(self, shape):
        """参数化测试不同 shape"""
        x = torch.randn(*shape, device="npu", dtype=torch.float32)
        output = your_op(x)
        ref_output = ref_your_op_name(x.cpu())
        torch.testing.assert_close(output.cpu(), ref_output, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", [
        torch.float32,
        torch.float16,
    ])
    def test_various_dtypes(self, dtype):
        """参数化测试不同数据类型"""
        x = torch.randn(1024, device="npu", dtype=dtype)
        output = your_op(x)
        ref_output = ref_your_op_name(x.cpu())
        rtol = 1e-2 if dtype == torch.float32 else 1e-1
        atol = 1e-2 if dtype == torch.float32 else 1e-1
        torch.testing.assert_close(output.cpu(), ref_output, rtol=rtol, atol=atol)

    def test_multidimensional_input(self):
        """多维输入测试"""
        x = torch.randn(32, 32, device="npu", dtype=torch.float32)
        output = your_op(x)
        ref_output = ref_your_op_name(x.cpu())
        torch.testing.assert_close(output.cpu(), ref_output, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
