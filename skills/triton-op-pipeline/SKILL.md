---
name: triton-op-pipeline
description: 用户新增Triton-Ascend算子时的入口技能。根据算子类型（纯向量/纯矩阵/融合）自动分发到对应子技能。基于 triton-ascend 官方文档、programming_guide、migration_guide 和 third_party/ascend/tutorials/ 实际示例。
---

# 新增 Triton-Ascend 算子通路技能 (入口)

基于 `triton-ascend/docs/en/programming_guide.md`、`migration_guide`、`architecture_design_and_core_features.md` 和 `third_party/ascend/tutorials/` 实际示例。

当用户提出"新增一个Triton算子/生成Triton-Ascend算子通路/按模板生成Triton算子"时使用本技能。

本技能是**入口分发器**，根据算子类型自动路由到对应的子技能。

## 生成前必读: Ascend NPU 硬件约束

生成算子前**必须**注意以下 NPU 硬件约束，确保:

1. **内存对齐**: VV 算子（纯 Vector）尾轴需 32 字节对齐；CV 算子（含 tl.dot）尾轴需 512 字节对齐
2. **UB容量限制**: Atlas A2 系列 UB 为 192 KB (1,572,864 bits)；超出会报 `ub overflow` 错误。开启 multibuffer 后容量减半
3. **coreDim限制**: grid维度不能超过 65535；超出需增大BLOCK_SIZE或启用 `TRITON_ALL_BLOCKS_PARALLEL=1`
4. **物理核心数**: VV 算子 grid = Vector Core 数；CV 算子 grid = AI Core 数（可通过 `driver.active.utils.get_device_properties` 获取）
5. **数据类型限制**: 支持 float16, float32, bfloat16, int8, int32。注意：Vector ADD 不支持 int64；Vector CMP 不支持 int64/int32（会退化为标量运算）
6. **编译选项**: 通过 `triton.Config` 传递 NPU 编译选项（如 `{'BLOCK_SIZE': 128, 'multibuffer': True}`）

### 获取硬件核心数

```python
import torch
import torch_npu
import triton.runtime.driver as driver

device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
vectorcore_num = properties["num_vectorcore"]  # 向量核心数
aicore_num = properties["num_aicore"]          # AI核心数 (Cube核心数)
```

## 算子类型判断与分发

### 判断流程

```
算子是否包含矩阵乘法 (tl.dot)?
|-- 否 --> 纯向量算子
|   └--> skills/triton-op-vector/SKILL.md
|
\-- 是 --> 矩阵乘法后是否有向量后处理?
    |-- 否 (仅 A @ B = C) --> 纯矩阵算子
    |   └--> skills/triton-op-cube/SKILL.md
    |
    \-- 是 (如 +bias, relu, softmax) --> 融合算子
        └--> skills/triton-op-fused/SKILL.md
```

### 分发表

| 算子特征 | 子技能 | 说明 |
|---------|--------|------|
| **纯向量计算** (无矩阵乘法) | `triton-op-vector` | add, mul, relu, softmax, layernorm, sigmoid |
| **纯矩阵乘法** (无后处理) | `triton-op-cube` | GEMM: C = A @ B, Batch GEMM, GEMV |
| **矩阵乘法 + 向量后处理** | `triton-op-fused` | Matmul+Bias, Matmul+ReLU, Flash Attention |

### 常见算子分类示例

| 算子 | 类型 | 子技能 |
|------|------|--------|
| vector_add | 纯向量 | `triton-op-vector` |
| element-wise mul/div | 纯向量 | `triton-op-vector` |
| relu / sigmoid / gelu | 纯向量 | `triton-op-vector` |
| softmax | 纯向量 | `triton-op-vector` |
| layernorm / rmsnorm | 纯向量 | `triton-op-vector` |
| GEMM (C = A @ B) | 纯矩阵 | `triton-op-cube` |
| GEMM with transpose (C = A @ B^T) | 纯矩阵 | `triton-op-cube` |
| Matmul + Bias (C = A@B + bias) | 融合 | `triton-op-fused` |
| Matmul + ReLU (C = relu(A@B)) | 融合 | `triton-op-fused` |
| Flash Attention | 融合 | `triton-op-fused` |

## Triton-Ascend 核心 API 速查

### 编程模型

| API | 说明 |
|-----|------|
| `@triton.jit` | 将Python函数编译为Triton内核 |
| `@triton.autotune(configs=[...], key=[...])` | 自动调优，遍历预设参数配置选择最优 |
| `tl.program_id(axis)` | 获取当前program在指定轴上的ID |
| `tl.num_programs(axis)` | 获取指定轴上的总program数 |
| `triton.cdiv(a, b)` | 向上取整除法 |
| `tl.constexpr` | 编译期常量参数标记 |

### 数据搬运

| API | 说明 |
|-----|------|
| `tl.load(pointer, mask=None, other=0.0)` | 从全局内存加载数据到片上内存 |
| `tl.store(pointer, value, mask=None)` | 将数据从片上内存写回全局内存 |
| `tl.arange(start, end)` | 生成连续整数范围 [start, end) |
| `tl.full(shape, value, dtype)` | 创建填充指定值的张量 |
| `tl.zeros(shape, dtype)` | 创建零张量 |

### 矩阵运算

| API | 说明 |
|-----|------|
| `tl.dot(a, b, allow_tf32=True)` | 块级矩阵乘法 (映射到NPU Cube Core) |
| `tl.trans(x)` | 转置 |

### 元素级数学运算

| API | 说明 |
|-----|------|
| `tl.exp(x)` / `tl.log(x)` / `tl.sqrt(x)` | 指数、对数、平方根 |
| `tl.abs(x)` / `tl.sin(x)` / `tl.cos(x)` | 绝对值、正弦、余弦 |
| `tl.sigmoid(x)` | Sigmoid激活 |
| `tl.where(condition, x, y)` | 条件选择 |
| `tl.maximum(x, y)` / `tl.minimum(x, y)` | 逐元素最大/最小 |
| `tl.clamp(x, min_val, max_val)` | 截断 |

### 归约运算

| API | 说明 |
|-----|------|
| `tl.sum(x, axis=None)` | 求和归约 |
| `tl.max(x, axis=None)` | 最大值归约 |
| `tl.min(x, axis=None)` | 最小值归约 |

### Ascend扩展API

| API | 说明 |
|-----|------|
| `tl.insert_slice(full, src, offsets, sizes, strides)` | 将张量插入目标张量 |
| `tl.extract_slice(full, offsets, sizes, strides)` | 从张量中提取切片 |
| `tl.get_element(source, offset)` | 读取张量中的单个元素 |
| `tl.custom_op` | Ascend自定义算子扩展 (index_select, gather_out_to_ub等) |
| `tl.compile_hint` | 硬件特定的编译提示 |
| `tl.sync_block_wait(sender, receiver, event_id)` | 等待块同步信号 |
| `tl.sync_block_set(sender, receiver, event_id)` | 设置块同步信号 |
| `tl.sync_block_all(mode, event_id)` | 全局块同步 |

### 编译器选项 (NPU专用)

| 选项 | 说明 |
|------|------|
| `multibuffer` | 启用/禁用乒乓流水线 (默认True) |
| `enable_auto_bind_sub_block` | CV融合算子自动绑定子块 |
| `enable_hivm_auto_cv_balance` | 自动CV平衡 |
| `sync_solver` | 同步求解器 |
| `inject_barrier_all` | 自动注入屏障 |
| `enable_linearize` | 线性化Pass |
| `stream` | NPU流 |

### 环境变量

| 变量 | 说明 |
|------|------|
| `TRITON_ALL_BLOCKS_PARALLEL=1` | 自动优化逻辑核心数为物理核心数 |
| `TRITON_DEBUG=1` | 输出中间IR dump |
| `TRITON_INTERPRET=1` | CPU解释器模式运行内核 |
| `TRITON_DEVICE_PRINT=1` | 启用 `tl.device_print` 和 `tl.static_print` |
| `TRITON_DISABLE_CACHE=1` | 禁用编译缓存 |
| `MLIR_ENABLE_DUMP=1` | Dump MLIR高层IR |
| `TRITON_PRINT_AUTOTUNING=1` | 输出autotune最优参数 |

## Ascend NPU 与 GPU 的关键差异

| 维度 | Ascend NPU | NVIDIA GPU |
|------|-----------|------------|
| **核心结构** | 多个AI Core, 分Cube Core (矩阵乘) + Vector Core (向量计算) | 多个SM, 含CUDA Core + Tensor Core |
| **并发任务** | 最大65535, 建议等于物理核心数 | 大量线程块自动调度 |
| **内存对齐** | 向量算子尾轴32B对齐, CV算子尾轴512B对齐 | 通常无需显式对齐 |
| **Grid** | 推荐1D grid; 2D会合并为1D | 1D/2D/3D灵活 |
| **片上内存** | UB (192KB on A2) + L1 + L0A/L0B/L0C | Shared Memory + Registers |
| **设备标识** | `device='npu'` | `device='cuda'` |
| **Python库** | `import torch_npu` | `import torch` (自带CUDA) |

## 数据类型

**dtype**: 使用标准 Triton dtype: `tl.float16`, `tl.float32`, `tl.int32`, `tl.int8`, `tl.bfloat16`

PyTorch侧: `torch.float16`, `torch.float32`, `torch.int32`, `torch.int8`

## 数据创建与验证

```python
import torch
import torch_npu
import triton
import triton.language as tl

torch.manual_seed(0)

# float16 数据
a = torch.randn(M, K, device='npu').half()
b = torch.randn(K, N, device='npu').half()

# float32 数据
a = torch.randn(M, K, device='npu')
b = torch.randn(K, N, device='npu')

# 调用 kernel
c = triton_add(a, b)

# 精度对比在CPU上进行
ref = a.cpu() + b.cpu()
torch.testing.assert_close(c.cpu(), ref, rtol=1e-2, atol=1e-2)
```

## 性能分析

```bash
# 板端 Profiling
msprof op --kernel-name=target_kernel_name --output=$HOME/projects/output python3 test_op.py

# 仿真 Profiling
msprof op simulator --kernel-name=target_kernel --soc-version=Ascend910B3 python3 test_op.py
```

## 工作流

1. 确认需要生成的算子类型（纯向量/纯矩阵/融合）。
2. **分发到对应子技能**:
   - 纯向量 --> `skills/triton-op-vector/SKILL.md`
   - 纯矩阵 --> `skills/triton-op-cube/SKILL.md`
   - 融合 --> `skills/triton-op-fused/SKILL.md`
3. 确认目标子项目路径。
4. 根据子技能的模板生成算子完整通路。
5. 检查生成目录及文件的完整性。
6. 输出改动说明与假设。

## 子技能索引

| 子技能 | 路径 | 说明 |
|--------|------|------|
| **triton-op-vector** | `skills/triton-op-vector/SKILL.md` | 纯向量计算 (Vector Core) |
| **triton-op-cube** | `skills/triton-op-cube/SKILL.md` | 纯矩阵乘法 (Cube Core via tl.dot) |
| **triton-op-fused** | `skills/triton-op-fused/SKILL.md` | 融合算子 (Cube + Vector) |

## 约束

- **矩阵乘法使用 `tl.dot()`**: 标准Triton API, 编译器自动映射到Cube Core
- **device使用 `'npu'`**: 禁止使用 `'cuda'`
- **必须导入 `torch_npu`**: 在 `import torch` 之后导入
- **Grid推荐1D**: NPU上2D grid会合并为1D形式
- **物理核心数优先**: grid值应等于物理核心数 (vector算子用vectorcore_num, dot算子用aicore_num)
- **BLOCK_SIZE对齐**: 推荐使用2的幂 (128, 256, 512, 1024 等), 确保满足32字节对齐要求
- **UB溢出处理**: 出现 `ub overflow` 错误时, 减小BLOCK_SIZE或引入子块循环
- **coreDim限制**: 确保 `triton.cdiv(N, BLOCK_SIZE) <= 65535`

## 参考

- 编程指南: `triton-ascend/docs/en/programming_guide.md`
- 架构设计: `triton-ascend/docs/en/architecture_design_and_core_features.md`
- 迁移指南: `triton-ascend/docs/en/migration_guide/migrate_from_gpu.md`
- 调试指南: `triton-ascend/docs/en/debug_guide/debugging.md`
- 性能分析: `triton-ascend/docs/en/debug_guide/profiling.md`
- 环境变量: `triton-ascend/docs/en/environment_variable_reference.md`
- 向量加法示例: `triton-ascend/third_party/ascend/tutorials/01-vector-add.py`
- 在线文档: https://triton-ascend.readthedocs.io/zh-cn/latest/index.html
