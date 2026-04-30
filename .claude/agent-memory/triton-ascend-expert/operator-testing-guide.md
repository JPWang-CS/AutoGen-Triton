---
name: operator-testing-guide
description: Triton-Ascend 算子测试核心知识，涵盖测试模式、dtype 精度容差、Fractal 对齐要求、边界用例设计、设备适配。来源于 skills 知识库 (triton-op-test, triton-op-benchmark)。
type: reference
---

# Triton-Ascend 算子测试指南

来源: `skills/triton-op-test/SKILL.md`, `skills/triton-op-benchmark/SKILL.md`

## 1. 设备初始化

```python
import torch
import torch_npu

torch.npu.set_device(0)
DEVICE = "npu"
```

**注意**:
- 使用 `device="npu"` 而非 `"cuda"`
- 结果移回 CPU 再比较: `result.cpu()` vs `ref.cpu()`

## 2. dtype 精度容差标准

不同数据类型有不同的比较策略:

### 浮点类型

| dtype | rtol | atol | 比较方式 |
|-------|------|------|---------|
| float32 | 1e-4 | 1e-4 | `torch.allclose(result.cpu(), ref.cpu(), rtol, atol)` |
| float16 | 1e-3 | 1e-3 | 同上 |
| bfloat16 | 1e-3 | 1e-3 | 先转 fp32 再比较 |

```python
# bfloat16 比较模式
result_f = result.cpu().float()
ref_f = ref.cpu().float()
passed = torch.allclose(result_f, ref_f, rtol=1e-3, atol=1e-3)
```

### 整数类型

```python
# 严格相等比较
passed = torch.equal(result.cpu(), ref.cpu())
# 或逐元素检查
mismatch = (result.cpu() != ref.cpu()).sum().item()
```

### 布尔类型

```python
passed = torch.equal(result.cpu(), ref.cpu())
```

### 索引类型 (argmax/argmin)

```python
# 索引比较不需要浮点容差
diff = (result.cpu() != ref.cpu()).sum().item()
passed = diff == 0
```

## 3. 测试分类

### 3.1 基本正确性测试

验证算子在标准形状和 dtype 下的正确性:

```python
def test_basic_correctness():
    x = torch.randn(128, 512, device="npu", dtype=torch.float32)
    ref = torch.sum(x, dim=-1)
    result = reduce_sum(x)
    assert torch.allclose(result.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)
```

### 3.2 多 dtype 测试

每种支持的 dtype 分别测试:

```python
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dtypes(dtype):
    x = torch.randn(128, 512, device="npu", dtype=dtype)
    ref = torch.sum(x, dim=-1)
    result = reduce_sum(x)
    rtol = 1e-3 if dtype != torch.float32 else 1e-4
    assert torch.allclose(result.cpu().float(), ref.cpu().float(), rtol=rtol, atol=rtol)
```

### 3.3 边界用例

| 场景 | 示例 |
|------|------|
| 小尺寸 | `(1,)`, `(2,)`, `(4,)` |
| 非 2 的幂 | `(3, 100)`, `(7, 333)` |
| 单行 | `(1, 1024)` |
| 单列 | `(1024, 1)` |
| 大列宽 | `(16, 65536)` |
| 空/退化 | `(0,)`（需 kernel 处理） |

```python
@pytest.mark.parametrize("shape", [(1,), (2, 3), (128, 512), (1, 1024), (1024, 1), (16, 65536)])
def test_shapes(shape):
    x = torch.randn(*shape, device="npu", dtype=torch.float32)
    ...
```

## 4. GEMM 测试约束

### Fractal 对齐要求

矩阵乘法的测试 shape 必须满足 fractal 对齐:
- FP16/BF16: M, N, K 必须为 **16 的倍数**
- INT8: M, N, K 必须为 **16 或 32 的倍数**

```python
# 正确: 16 的倍数
shapes = [(128, 128, 64), (256, 256, 128), (64, 64, 32)]

# 错误: 非 16 倍数会导致边界处理复杂
# shapes = [(100, 100, 50)]  # 避免
```

### 精度容差

GEMM 累加误差较大，需放宽容差:
```python
# FP16 GEMM
rtol, atol = 1e-2, 1e-2

# BF16 GEMM
rtol, atol = 1e-2, 1e-2

# FP32 GEMM
rtol, atol = 1e-3, 1e-3
```

## 5. 参考实现选择

| 算子类型 | PyTorch 参考实现 |
|---------|----------------|
| Element-wise | `torch.add`, `torch.mul`, `torch.nn.functional.gelu` 等 |
| Reduce | `torch.sum`, `torch.max`, `torch.argmax` 等 |
| GEMM | `torch.matmul`, `a @ b` |
| Flash Attention | `torch_npu.npu_fusion_attention` |
| 自定义 | 手写 PyTorch 等价实现 |

## 6. 测试模板

```python
import torch
import torch_npu
import pytest

torch.npu.set_device(0)

# === 被测算子 ===
from my_op import my_kernel, ref_program

class TestMyOp:

    def test_basic(self):
        x = torch.randn(128, 512, device="npu", dtype=torch.float32)
        ref = ref_program(x)
        result = my_kernel(x)
        assert torch.allclose(result.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_dtypes(self, dtype):
        x = torch.randn(128, 512, device="npu", dtype=dtype)
        ref = ref_program(x)
        result = my_kernel(x)
        rtol = 1e-3 if dtype != torch.float32 else 1e-4
        assert torch.allclose(result.cpu().float(), ref.cpu().float(), rtol=rtol, atol=rtol)

    @pytest.mark.parametrize("shape", [(1, 64), (4, 100), (128, 512), (1, 1024)])
    def test_shapes(self, shape):
        x = torch.randn(*shape, device="npu", dtype=torch.float32)
        ref = ref_program(x)
        result = my_kernel(x)
        assert torch.allclose(result.cpu(), ref.cpu(), rtol=1e-3, atol=1e-3)

    def test_contiguity(self):
        x = torch.randn(128, 512, device="npu").T.T  # 确保连续
        ref = ref_program(x)
        result = my_kernel(x)
        assert torch.allclose(result.cpu(), ref.cpu(), rtol=1e-3, atol=1e-3)
```

## 7. 整数类型测试注意

```python
# int8 范围: -128 ~ 127
x_int8 = torch.randint(-127, 127, shape, device="npu", dtype=torch.int8)

# int32 范围较大，但 XOR 等位运算需注意
x_int32 = torch.randint(-100, 100, shape, device="npu", dtype=torch.int32)

# 整数比较用 torch.equal (严格相等)
ref = torch.argmax(x, dim=-1)
result = reduce_argmax(x)
assert torch.equal(result.cpu(), ref.cpu())
```

## 8. 常见测试陷阱

1. **忘设 device**: 确保 `device="npu"`
2. **GPU 上比较**: NPU 张量需 `.cpu()` 再比较
3. **bf16 直接 allclose**: bf16 精度低，先转 fp32
4. **非对齐 shape 测 GEMM**: 不满足 fractal 对齐会出错
5. **忘记 contiguity**: `assert x.is_contiguous()` 或显式 `.contiguous()`
6. **整数用 allclose**: 整数用 `torch.equal` 而非 `allclose`
