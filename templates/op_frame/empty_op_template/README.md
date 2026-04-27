# your_op_name (Triton-Ascend)

Triton-Ascend 实现的 your_op_name 算子，运行在华为昇腾 NPU 上。

## 算子描述

请在此处添加算子的详细描述。

所有计算在 NPU 上执行，精度对比在 CPU 上进行。

## Triton 编程规范

| 特性 | Triton 语法 |
|------|-------------|
| Kernel 定义 | `@triton.jit` 装饰器 |
| 并行索引 | `tl.program_id(axis)` |
| 偏移量生成 | `tl.arange(0, BLOCK_SIZE)` |
| 内存读取 | `tl.load(ptr + offsets, mask=mask)` |
| 内存写入 | `tl.store(ptr + offsets, value, mask=mask)` |
| 矩阵乘法 | `tl.dot(a, b)` |
| Grid 计算 | `triton.cdiv(total, BLOCK_SIZE)` |
| 自动调优 | `@triton.autotune(configs=[...], key=[...])` |

## 接口说明

### 函数签名

```python
def your_op_name(
    x: torch.Tensor,
) -> torch.Tensor:
    ...
```

### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| x | torch.Tensor | 输入张量，形状任意 |

### 输入/输出

**输入:**
- `x`: Tensor，形状 (N,)，dtype 为 float32 或 float16

**输出:**
- `output`: Tensor，形状与输入相同

## 使用示例

```python
import torch
import torch_npu
from your_op_name import your_op

torch.npu.set_device(0)

# 准备输入数据 (NPU)
x = torch.randn(1024, device="npu", dtype=torch.float32)

# 调用 kernel
output = your_op(x)

# 精度对比在 CPU 上进行
ref_output = x.cpu()  # TODO: 替换为实际参考实现
torch.testing.assert_close(output.cpu(), ref_output, rtol=1e-2, atol=1e-2)
```

## 文件结构

```
your_op_name/
├── your_op_name.py           # 核心 kernel 实现
├── test_your_op_name.py      # 单元测试
├── benchmark_your_op_name.py # 性能基准测试
└── README.md                 # 本文档
```

## 运行测试

```bash
# 运行单元测试
python test_your_op_name.py

# 使用 pytest 运行
pytest test_your_op_name.py -v

# 运行性能测试
python benchmark_your_op_name.py --size 1048576
```

## BLOCK_SIZE 选择指南

| 输入规模 | 推荐 BLOCK_SIZE | 说明 |
|----------|-----------------|------|
| 小 (< 1K) | 256 | 减少尾部浪费 |
| 中 (1K ~ 64K) | 1024 | 平衡并行度和利用率 |
| 大 (> 64K) | 2048 | 最大化内存吞吐 |

## 参考

- [Triton-Ascend 文档](https://triton-ascend.readthedocs.io/zh-cn/latest/)
- [Triton 官方文档](https://triton-lang.org/)
