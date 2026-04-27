# Triton-Ascend AutoGen

Triton-Ascend 算子自动生成工具集，提供基于 Claude Code 的 skill 提示词和相关模板，用于在华为昇腾 NPU 上高效开发和优化 Triton 算子。

## 概述

本仓库提供了一套完整的 skill 提示词，用于自动生成 Triton-Ascend 算子的完整实现，包括：
- 核心 kernel 实现（基于 `@triton.jit` 编程模型）
- 单元测试（正确性验证与精度对比）
- 性能基准测试（使用 `triton.testing.do_bench`）
- 硬件约束检查（内存对齐、片上内存容量等）

Triton-Ascend 是面向华为昇腾 NPU 的 Triton 编译框架，开发者仅需关注 block/tile 级的计算逻辑，编译器自动完成内存分配、数据搬运和流水并行优化。

## 快速开始

### 环境准备

```bash
# 安装 torch-npu (请根据 CANN 版本选择)
pip install torch-npu

# 安装 Triton-Ascend (稳定版)
pip install triton-ascend

# 或安装 nightly 版本
pip install -i https://test.pypi.org/simple/ triton-ascend
```

配套 CANN 版本：建议使用 **CANN 8.5.0**。支持的硬件：Atlas A2/A3 系列。

详见 [安装指南](https://triton-ascend.readthedocs.io/zh-cn/latest/installation_guide/index.html)。

### 新增算子

使用 `triton-op-pipeline` skill 生成新的算子框架：

```
请使用 triton-op-pipeline 生成一个名为 fused_softmax 的融合 softmax 算子
```

### 生成测试

使用 `triton-op-test` skill 生成测试用例：

```
请为 fused_softmax 生成单元测试
```

### 性能测试

使用 `triton-op-benchmark` skill 生成 benchmark：

```
请为 fused_softmax 生成性能基准测试
```

## 目录结构

```
Triton/
├── skills/                              # Skill 提示词目录
│   ├── triton-op-pipeline/              # 新增算子（入口分发器）
│   ├── triton-op-vector/                # 纯向量算子（Vector Core）
│   ├── triton-op-cube/                  # 纯矩阵算子（Cube Core）
│   ├── triton-op-fused/                 # 融合算子（Cube + Vector）
│   ├── triton-op-edit/                  # 修改算子
│   ├── triton-op-rename/                # 重命名算子
│   ├── triton-op-test/                  # 生成测试
│   ├── triton-op-benchmark/             # 性能测试
│   └── triton-op-hardware-constraints/  # NPU 硬件约束检查
├── templates/op_frame/                  # 模板文件
│   ├── empty_op_template/               # 空白算子模板
│   └── gemm_reference/                  # GEMM 参考实现
├── projects/                            # 生成的算子项目
├── .claude/                             # Claude Agent 配置
├── CLAUDE.md                            # AI Agent 项目说明
└── README.md                            # 本文件
```

## 技能列表

| 技能 | 描述 | 适用场景 |
|------|------|---------|
| triton-op-pipeline | 算子生成入口，自动分发到子技能 | 新增任何算子 |
| triton-op-vector | 纯向量算子（add, relu, softmax, layernorm 等） | 无矩阵乘法的算子 |
| triton-op-cube | 纯矩阵乘法（GEMM, GroupGEMM 等） | 仅矩阵乘法的算子 |
| triton-op-fused | 融合算子（Matmul+Bias, Flash Attention 等） | 矩阵乘法 + 后处理 |
| triton-op-edit | 修改现有算子参数或实现 | 调整算子逻辑 |
| triton-op-rename | 重命名算子及其引用 | 重命名算子 |
| triton-op-test | 生成单元测试用例 | 验证算子正确性 |
| triton-op-benchmark | 生成性能基准测试 | 测量算子性能 |
| triton-op-hardware-constraints | 检查 NPU 硬件约束 | 验证内存对齐和容量 |

## Triton-Ascend 核心概念

### 编程模型

Triton-Ascend 使用标准 Triton 编程范式。以下是一个向量加法示例：

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

### NPU 多核并行

与 GPU 的海量逻辑线程不同，NPU 使用物理核绑定模式。推荐将分核数设为硬件物理核数：

```python
import triton.runtime.driver as driver

properties = driver.active.utils.get_device_properties(device)
vectorcore_num = properties["num_vectorcore"]  # Vector 核数量
aicore_num = properties["num_aicore"]          # Cube 核数量

# 纯 Vector 算子使用 vectorcore_num 个核
kernel[vectorcore_num](...)
```

### Autotune 自动调优

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
    ],
    key=['n_elements'],
)
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr):
    ...
```

## 示例项目

以下算子类型可作为参考实现：

| 算子类型 | Triton-Ascend 教程 | 说明 |
|---------|-------------------|------|
| Vector Add | `triton-ascend/python/tutorials/01-vector-add.py` | 基础向量运算 |
| Fused Softmax | `triton-ascend/python/tutorials/02-fused-softmax.py` | 归约与融合优化 |
| Matmul | `triton-ascend/python/tutorials/03-matrix-multiplication.py` | 块级矩阵乘法 |
| Dropout | `triton-ascend/python/tutorials/04-low-memory-dropout.py` | 随机数与掩码 |
| Layer Norm | `triton-ascend/python/tutorials/05-layer-norm.py` | 前向与反向实现 |
| Flash Attention | `triton-ascend/python/tutorials/06-fused-attention.py` | Cube+Vector 融合 |
| Grouped GEMM | `triton-ascend/python/tutorials/08-grouped-gemm.py` | 批量矩阵乘法 |

## 开发规范

### 代码风格

- 遵循 Python PEP 8 编码规范
- 使用类型注解标注函数签名
- 使用有意义的变量名（`BLOCK_SIZE` 而非 `BS`）
- 为公开函数添加 docstring
- 关键逻辑处添加注释

### Triton-Ascend 特定规范

- `BLOCK_SIZE` 等编译参数使用 `tl.constexpr` 声明
- 使用 `tl.load` / `tl.store` 进行内存操作，始终携带 `mask` 防止越界
- 使用 `tl.program_id` 获取并行信息
- 尾轴数据对齐：VV 类算子 32B，CV 类算子 512B
- 核数建议设为物理核数（`vectorcore_num` 或 `aicore_num`）

### 性能优化建议

- 使用 `triton.autotune` 自动搜索最优 BLOCK_SIZE
- 合理利用二级切分（XBLOCK + XBLOCK_SUB）充分利用片上内存
- 借助 `multibuffer` 编译选项实现存算并行
- 尾轴对齐以避免硬件自动补齐带来的性能损失
- 对于离散访存场景，先搬运到 UB 再 select

## 贡献指南

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/my-new-op`)
3. 使用 skill 生成算子框架并完善实现
4. 添加测试用例并确保通过
5. 提交 Pull Request

### Skill 贡献

每个 skill 位于 `skills/` 目录下的独立子目录中，包含一个 `SKILL.md` 文件描述 skill 的触发条件、工作流程和输出规范。

## 参考资料

- [Triton-Ascend 在线文档](https://triton-ascend.readthedocs.io/zh-cn/latest/index.html)
- [Triton-Ascend GitCode](https://gitcode.com/Ascend/triton-ascend)
- [Triton-Ascend 安装指南](https://triton-ascend.readthedocs.io/zh-cn/latest/installation_guide/index.html)
- [Triton-Ascend 算子开发指南](https://triton-ascend.readthedocs.io/zh-cn/latest/programming_guide/index.html)
- [Triton-Ascend 算子迁移指南](https://triton-ascend.readthedocs.io/zh-cn/latest/migration_guide/migrate_from_gpu.html)
- [OpenAI Triton](https://github.com/triton-lang/triton)

## 许可证

MIT License
