# GEMM 参考实现 (Triton-Ascend)

Triton-Ascend 实现的 GEMM（矩阵乘法）参考实现，使用 `tl.dot` 和 `@triton.autotune`。

所有计算在 NPU 上执行，精度对比在 CPU 上进行。

## Triton 编程规范

本算子遵循 Triton 编程模型：

| 特性 | Triton 语法 |
|------|-------------|
| Kernel 定义 | `@triton.jit` |
| 矩阵乘法 | `tl.dot(a, b)` |
| 自动调优 | `@triton.autotune(configs=[...], key=[...])` |
| 并行索引 | `tl.program_id(axis=0/1)` |
| 编译时常量 | `BLOCK_SIZE: tl.constexpr` |
| 内存操作 | `tl.load` / `tl.store` |

## 功能特性

- 分块矩阵乘法 (`gemm`)
- `@triton.autotune` 自动调优，覆盖多种 BLOCK 配置
- 支持 float16 / float32 数据类型
- 完整的单元测试

## 文件结构

```
gemm_reference/
├── gemm.py        # 核心 kernel 实现（含 autotune）
├── test_gemm.py   # 单元测试
└── README.md      # 本文档
```

## 使用示例

```python
import torch
import torch_npu
from gemm import gemm

torch.npu.set_device(0)

M, N, K = 1024, 1024, 1024
a = torch.randn(M, K, device="npu", dtype=torch.float16)
b = torch.randn(K, N, device="npu", dtype=torch.float16)

c = gemm(a, b)

# 精度对比在 CPU 上进行
ref_c = a.cpu() @ b.cpu()
torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)
```

## GEMM 分块策略

GEMM kernel 采用 2D 分块策略：

```
输出矩阵 C (M x N)
┌─────────────────┐
│ (BLOCK_M,BLOCK_N)│  <-- 每个 program 负责一个块
│                  │
│  沿 K 维分块累加  │  <-- 循环 tl.dot(a_block, b_block)
│                  │
└─────────────────┘
```

数据流：
1. 每个 program 根据 `tl.program_id(0/1)` 定位到输出块 (pid_m, pid_n)
2. 沿 K 维循环加载 A 和 B 的分块
3. 使用 `tl.dot` 进行块矩阵乘法
4. 累加到 accumulator（float32 精度）
5. 写回全局内存

## 运行测试

```bash
# 运行所有测试
python test_gemm.py

# 使用 pytest
pytest test_gemm.py -v
```

## 自动调优配置

`@triton.autotune` 会自动测试以下配置并选择最优：

| BLOCK_M | BLOCK_N | BLOCK_K |
|---------|---------|---------|
| 128 | 256 | 64 |
| 64 | 256 | 32 |
| 128 | 128 | 32 |
| 128 | 64 | 32 |
| 64 | 128 | 32 |
| 128 | 32 | 32 |
| 64 | 32 | 32 |
| 32 | 64 | 32 |

> 注意: `num_stages` 和 `num_warps` 是 GPU (CUDA) 特有参数，在 NPU 上无效，已移除。

## 参考

- [Triton-Ascend 文档](https://triton-ascend.readthedocs.io/zh-cn/latest/)
- [Triton 官方文档](https://triton-lang.org/)
