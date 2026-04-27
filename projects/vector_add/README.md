# Vector Add (Triton-Ascend)

Triton-Ascend 实现的向量加法算子：C = A + B。

这是最基础的 Triton 算子，演示了 element-wise 操作的标准模式。

## 算子描述

对两个相同形状的向量/张量执行逐元素加法运算。每个 program 处理一段连续的 BLOCK_SIZE 个元素，使用 mask 处理边界情况。

## 接口说明

### 函数签名

```python
def vector_add(
    a: torch.Tensor,
    b: torch.Tensor,
    block_size: int = 1024,
) -> torch.Tensor:
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| a | torch.Tensor | - | 输入向量 A，形状 (N,) 或任意形状 |
| b | torch.Tensor | - | 输入向量 B，形状与 A 相同 |
| block_size | int | 1024 | 每个 program 处理的元素数 |

### 输入/输出

**输入:**
- `A`: Tensor，形状 (N,)，dtype 为 float32 或 float16
- `B`: Tensor，形状 (N,)，dtype 与 A 相同

**输出:**
- `C`: Tensor，形状 (N,)，dtype 与输入相同

## 使用示例

```python
import torch
import torch_npu
from vector_add import vector_add

torch.npu.set_device(0)

a = torch.randn(1024, device="npu", dtype=torch.float32)
b = torch.randn(1024, device="npu", dtype=torch.float32)

c = vector_add(a, b)

ref_c = a.cpu() + b.cpu()
torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-6, atol=1e-6)
```

## 文件结构

```
vector_add/
├── vector_add.py           # 核心 kernel 实现
├── test_vector_add.py      # 单元测试
├── benchmark_vector_add.py # 性能基准测试
└── README.md               # 本文档
```

## 运行测试

```bash
# 运行单元测试
python test_vector_add.py

# 使用 pytest
pytest test_vector_add.py -v

# 运行性能测试
python benchmark_vector_add.py --size 1048576
```

## 参考

- [Triton-Ascend 文档](https://triton-ascend.readthedocs.io/zh-cn/latest/)
- [Triton 官方 Fused Add 论文](https://arxiv.org/abs/2305.17005)
