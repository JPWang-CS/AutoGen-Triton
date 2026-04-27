# Triton-Ascend 快速入门教程（中文版）

> 本教程基于 [Triton-Ascend](https://gitee.com/ascend/triton-ascend) 编写，旨在帮助中文用户快速掌握 Triton 在华为昇腾 NPU 上的核心概念与使用方法。

## 什么是 Triton-Ascend？

Triton-Ascend 是 OpenAI Triton 在华为昇腾 NPU 上的适配版本。它保留了标准 Triton 的编程模型和 API，使开发者可以使用熟悉的 `@triton.jit`、`tl.load`、`tl.store`、`tl.dot` 等 API 在 NPU 上编写高性能算子。

### 核心特性

- **标准 Triton API**：使用 `@triton.jit`、`tl.load/store`、`tl.dot`、`@triton.autotune` 等标准 API
- **NPU 后端**：将 Triton IR 编译为昇腾 CANN 可执行的指令
- **多核并行**：利用 NPU 的 AI Core（Cube + Vector）实现数据并行
- **自动调优**：`@triton.autotune` 自动搜索最优 BLOCK 配置
- **昇腾扩展**：提供 `tl.dot_scaled`、`tl.parallel`、`tl.sync_block` 等 NPU 特有 API

### 昇腾 NPU 架构 (Ascend 910B)

910B 的 AIC（Cube Core）和 AIV（Vector Core）是**隔离**的，各自拥有独立片上存储，L2 Cache 共享。

```
                        +-----------------------+
                        |   Global Memory (HBM)  |
                        +-----------------------+
                                    |
                                    v
                        +-----------------------+
                        |   L2 Cache (共享)      |
                        +-----------------------+
                           /                \
                          v                  v
   +----------------------------+    +----------------------------+
   |   AIC (Cube Core)          |    |   AIV (Vector Core)        |
   |   矩阵乘法 (tl.dot)        |    |   向量计算 (tl.load/store) |
   |                            |    |                            |
   |  L1 (~1MB) <-> L0A/L0B/L0C |    |   UB (~192KB/A2: ~1MB)    |
   +----------------------------+    +----------------------------+
```

## 教程目录

| 序号 | 章节 | 说明 |
|------|------|------|
| 01 | [安装与环境配置](01_安装与环境配置.md) | 安装 Triton-Ascend、配置昇腾环境、验证安装 |
| 02 | [语言基础](02_语言基础.md) | SPMD 编程模型、Kernel 定义、内存操作、Grid 配置 |
| 03 | [核心 API 详解](03_核心API详解.md) | tl.load/store、tl.dot、归约操作、数学函数、数据类型 |
| 04 | [控制流与并行](04_控制流与并行.md) | 1D/2D Grid、K 维循环、行级并行、多核并行策略 |
| 05 | [类型系统与精度](05_类型系统与精度.md) | 数据类型、累加器精度、类型转换、NPU 对齐、数值稳定性 |
| 06 | [自动调优](06_自动调优.md) | @triton.autotune、triton.Config、NPU 调优注意事项 |
| 07 | [实战：GEMM 矩阵乘法](07_实战示例_GEMM.md) | Naive → 2D 分块 → Autotune → 持久化内核 |
| 08 | [实战：注意力机制](08_实战示例_注意力机制.md) | Flash Attention：在线 Softmax、QKV 分块、UB 管理 |
| 09 | [高级特性与调试](09_高级特性与调试.md) | 编译选项、昇腾扩展 API、调试工具、Profiling、GPU→NPU 迁移 |

## 快速体验

```python
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

def vector_add(x, y):
    output = torch.empty_like(x)
    n = output.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
    return output

# 运行
x = torch.randn(1024, device="npu")
y = torch.randn(1024, device="npu")
result = vector_add(x, y)
diff = torch.max(torch.abs(result - (x + y))).item()
print(f"最大误差: {diff}")  # 应为 0.0
print("Kernel Output Match!")
```

## 参考资源

- [Triton-Ascend 在线文档](https://triton-ascend.readthedocs.io/zh-cn/latest/index.html)
- [Triton-Ascend 源码仓库 (Gitee)](https://gitee.com/ascend/triton-ascend)
- [Triton-Ascend 源码仓库 (GitHub)](https://github.com/ascend/triton-ascend)
- [昇腾 CANN 文档](https://www.hiascend.com/document)
- [标准 Triton 文档](https://triton-lang.org/)
