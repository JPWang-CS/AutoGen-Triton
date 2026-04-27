---
name: triton-op-test
description: 为Triton-Ascend算子生成单元测试文件与用例。用户提出"生成测试/生成UT/补充测试用例"时使用本技能。
---

# Triton-Ascend 算子测试生成

当用户提出"生成Triton算子测试/生成UT/补充测试用例"时使用本技能。

## 工作流

1. 确认目标算子路径与算子名称。
2. 读取算子实现文件，理解输入/输出/参数。
3. 生成测试文件 `test_{op_name}.py`。
4. 包含以下测试类别：
   - 正确性测试（与 PyTorch 参考实现对比）
   - 边界测试（最小/最大 shape、非对齐 shape）
   - 数据类型测试（不同 dtype）
   - 可选参数测试（如有）
   - NPU 设备适配测试

## 测试文件结构

```python
import pytest
import torch
import torch_npu
import triton
import triton.language as tl

# 导入被测试的kernel
from your_op_name import your_op_name


def ref_your_op_name(*args, **kwargs):
    """PyTorch 参考实现"""
    # 实现参考逻辑
    pass


def accuracy_comparison(y_cal, y_ref):
    """
    精度比对函数：根据数据类型选择合适的比对策略。
    """
    assert y_cal.dtype == y_ref.dtype, f"dtype mismatch: {y_cal.dtype} vs {y_ref.dtype}"
    tensor_dtype = y_cal.dtype

    # 将张量移动到 CPU 进行精度比对
    y_cal = y_cal.cpu()
    y_ref = y_ref.cpu()

    if tensor_dtype == torch.float16:
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-3, atol=1e-3, equal_nan=True)
    elif tensor_dtype == torch.bfloat16:
        # bfloat16 精度较低，建议转为 float32 再比较
        torch.testing.assert_close(
            y_ref.to(torch.float32), y_cal.to(torch.float32),
            rtol=1e-3, atol=1e-3, equal_nan=True
        )
    elif tensor_dtype == torch.float32:
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-4, atol=1e-4, equal_nan=True)
    elif tensor_dtype in [torch.int64, torch.int32, torch.int16, torch.int8]:
        assert torch.equal(y_cal, y_ref), f"Integer tensors not equal for {tensor_dtype}"
    elif tensor_dtype == torch.bool:
        assert torch.equal(y_cal.cpu(), y_ref.cpu()), "Boolean tensors not equal"
    else:
        raise ValueError(f'Unsupported dtype: {tensor_dtype}')


class TestYourOpName:
    """YourOpName 算子测试类"""

    def test_basic_correctness(self):
        """基础正确性测试"""
        torch.manual_seed(0)
        # 准备输入数据，使用 device='npu'
        x = torch.randn(1024, device='npu', dtype=torch.float16)
        y = torch.randn(1024, device='npu', dtype=torch.float16)

        # PyTorch 参考结果
        ref_result = x + y  # 或其他参考实现

        # Triton 算子结果
        triton_result = your_op_name(x, y)

        # 精度比对
        accuracy_comparison(triton_result, ref_result)

    def test_different_dtypes(self):
        """不同数据类型测试"""
        for dtype in [torch.float16, torch.bfloat16]:
            x = torch.randn(1024, device='npu', dtype=dtype)
            y = torch.randn(1024, device='npu', dtype=dtype)
            ref_result = x + y
            triton_result = your_op_name(x, y)
            accuracy_comparison(triton_result, ref_result)

    def test_edge_cases(self):
        """边界情况测试"""
        # 测试非对齐 shape（非 BLOCK_SIZE 整数倍）
        for size in [1, 7, 63, 127, 98432]:
            x = torch.randn(size, device='npu', dtype=torch.float16)
            y = torch.randn(size, device='npu', dtype=torch.float16)
            ref_result = x + y
            triton_result = your_op_name(x, y)
            accuracy_comparison(triton_result, ref_result)

    @pytest.mark.parametrize("M,N,K", [
        (16, 16, 16),
        (64, 64, 64),
        (128, 128, 128),
        (1024, 1024, 1024),
    ])
    def test_various_shapes(self, M, N, K):
        """参数化测试不同 shape"""
        x = torch.randn((M, K), device='npu', dtype=torch.float16)
        y = torch.randn((K, N), device='npu', dtype=torch.float16)
        ref_result = torch.matmul(x, y)
        triton_result = your_op_name(x, y)
        accuracy_comparison(triton_result, ref_result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## 测试生成规则

### 1. 设备适配

Triton-Ascend 测试中关键差异点：

```python
# 使用 'npu' 设备而非 'cuda'
x = torch.randn(size, device='npu', dtype=torch.float16)

# 导入 torch_npu
import torch_npu

# 不需要 GPU 设备一致性校验
# x.device == 'cuda' 检查不需要
```

### 2. 正确性验证

- 使用 `torch.testing.assert_close` 进行结果对比
- 按数据类型设置合理的 `rtol` 和 `atol`

| 数据类型 | rtol | atol | 说明 |
|---------|------|------|------|
| float32 | 1e-4 | 1e-4 | 高精度，严格容差 |
| float16 | 1e-3 | 1e-3 | 中等精度 |
| bfloat16 | 1e-3 | 1e-3 | 低精度，建议转 fp32 再比较 |
| int8/16/32/64 | - | - | 使用 `torch.equal` 严格相等 |
| bool | - | - | 在 CPU 上用 `torch.equal` 比较 |

### 3. 参考实现

- 使用 PyTorch 原生操作作为参考实现
- 对于复杂算子，使用 `torch.nn.functional` 提供的函数
- 对于 Flash Attention 类算子，可使用 `torch_npu.npu_fusion_attention` 作为参考

### 4. 输入数据生成

```python
# 浮点类型：使用 torch.randn
x = torch.randn(shape, dtype=dtype, device='npu')

# 整数类型：使用 torch.randint
x = torch.randint(low=0, high=2000, size=shape, dtype=dtype, device='npu')

# int8 类型：注意范围
x = torch.randint(low=0, high=127, size=shape, dtype=torch.int8, device='npu')
```

### 5. Triton kernel 调用方式

```python
# 方式1：使用 lambda grid 函数
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
add_kernel[grid](x, y, output, n_elements)

# 方式2：使用固定核数（推荐，NPU 优化写法）
import triton.runtime.driver as driver
properties = driver.active.utils.get_device_properties(torch.npu.current_device())
aicore_num = properties["num_aicore"]
grid = (aicore_num,)
my_kernel[grid](..., NUM_CORE=aicore_num)

# 方式3：使用 triton.testing.do_bench
# 需要设置 TRITON_BENCH_METHOD="npu" 环境变量
```

### 6. 矩阵乘法测试特殊约束

对于使用 `tl.dot` 的算子，测试 shape 需满足分形对齐要求：

```python
@pytest.mark.parametrize("M,N,K", [
    (16, 16, 16),    # 最小分形
    (32, 32, 32),
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
    (1024, 1024, 1024),
])
def test_matmul_shapes(self, M, N, K):
    # BLOCK_M, BLOCK_N, BLOCK_K 必须是 16 的倍数（FP16/BF16）
    # 或 32 的倍数（INT8）
    ...
```

### 7. Flash Attention 测试

```python
@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM, causal, dtype, BM, BN", [
    (1, 1, 128, 128, False, torch.float16, 32, 128),
    (1, 2, 256, 256, False, torch.bfloat16, 32, 256),
    (4, 32, 1024, 64, False, torch.float16, 64, 128),
])
def test_attention(self, Z, H, N_CTX, HEAD_DIM, causal, dtype, BM, BN):
    if N_CTX % BM != 0 or N_CTX % BN != 0 or HEAD_DIM % 16 != 0:
        pytest.skip("Skipping non-divisible case")

    q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device='npu').normal_(0.0, 0.5)
    k = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device='npu').normal_(0.0, 0.5)
    v = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device='npu').normal_(0.0, 0.5)

    triton_out = attention(q, k, v, causal, sm_scale, BM, BN)
    ref_out = torch_npu.npu_fusion_attention(
        q, k, v, H, padding_mask=None, atten_mask=None,
        scale=sm_scale, keep_prob=1.0, input_layout="BNSD",
    )[0]

    torch.testing.assert_close(ref_out, triton_out, atol=1e-2, rtol=1e-2, equal_nan=True)
```

## 调试辅助

### 解释器模式

设置 `TRITON_INTERPRET=1` 在 CPU 上运行 kernel，作为精度基准：

```bash
TRITON_INTERPRET=1 python test_your_op.py
```

### device_print 调试

```python
@triton.jit
def debug_kernel(...):
    x = tl.load(...)
    tl.device_print("x = ", x)  # 运行时打印，需设置 TRITON_DEVICE_PRINT=1
    result = x + y
    tl.store(...)
```

```bash
TRITON_DEVICE_PRINT=1 python test_your_op.py
```

### IR 调试

```bash
TRITON_DEBUG=1 TRITON_DISABLE_CACHE=1 python test_your_op.py
# 查看 ~/.triton/dump/ 下的 kernel.ttir.mlir 和 kernel.ttadapter.mlir
```

## 输出要求

- 列出生成的文件路径
- 说明已覆盖的测试用例类别
- 提示需要用户补充的特殊测试场景

## 约束

- 仅生成测试代码，不修改算子实现
- 遵循 pytest 测试框架规范
- 确保测试可以直接在 NPU 环境运行
- 注意 NPU 与 GPU 的差异（device 名、核数限制、数据类型支持等）

## 参考

- Triton-Ascend 向量加法示例: `triton-ascend/docs/zh/examples/01_vector_add_example.md`
- Triton-Ascend 精度比对示例: `triton-ascend/docs/zh/examples/07_accuracy_comparison_example.md`
- Triton-Ascend 矩阵乘法示例: `triton-ascend/docs/zh/examples/05_matrix_multiplication_example.md`
- Triton-Ascend Flash Attention 示例: `triton-ascend/docs/zh/examples/04_fused_attention_example.md`
- Triton-Ascend 调试指南: `triton-ascend/docs/zh/debug_guide/debugging.md`
