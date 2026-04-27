---
name: triton-op-hardware-constraints
description: 为Triton-Ascend算子提供NPU硬件约束检查，确保算子搬运和计算符合不同SOC版本（910B、910A、310P等）的硬件上限，防止地址越界和溢出
---

# Triton-Ascend NPU 硬件约束技能

当用户开发 Triton-Ascend 算子时，本技能提供 NPU 硬件约束检查和优化建议，确保算子搬运和计算符合不同 SOC 版本的硬件上限，**防止硬件地址越界或溢出**。

## 触发条件

- 用户提出"检查算子硬件约束/验证算子是否符合NPU限制/SOC硬件上限检查"
- 用户开发算子时需要考虑不同 NPU 型号的适配
- 算子出现硬件资源超限问题（如 UB OVERFLOW、coreDim 超限）
- 算子出现地址越界或溢出错误
- 用户需要选择合适的 BLOCK_SIZE 以适配片上内存

## NPU 架构版本对照

| SOC 版本 | 代号 | NPU_ARCH | 定位 | 主要用途 |
|---------|------|----------|------|---------|
| **Ascend 910** | DAV_TBE | 200x | 训练 | 大模型训练 |
| **Ascend 910A** | DAV_2201 | 200x | 训练 | 大模型训练 |
| **Ascend 910B** | DAV_3510 | 220x | 训练 | 大模型训练（当前主流） |
| **Ascend 310** | DAV_MINI | 200x | 推理 | 边缘推理 |
| **Ascend 310P** | DAV_310P | 200x | 推理 | 边缘推理 |

### NPU_ARCH 架构差异

| NPU_ARCH | 架构类型 | 对应产品 | 特点 |
|----------|---------|---------|------|
| **200x** | Cube+Vector同核 | Atlas 推理系列产品 | Cube和Vector共享Scalar单元 |
| **220x** | Cube+Vector分离 | Atlas A2/A3 训练/推理系列 | AIC(矩阵)和AIV(向量)独立，各自有Scalar单元 |

> **重要**: 不同架构版本的存储单元对齐要求和核数有差异，生成算子时必须明确目标架构。

## 多核并行约束

### 物理核数获取

```python
import torch
import torch_npu
import triton.runtime.driver as driver

device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
vectorcore_num = properties["num_vectorcore"]
aicore_num = properties["num_aicore"]
```

### 核数选择规则

- **纯 Vector 算子**（无 `tl.dot`）：分核数 = **Vector 核数**（如 910B 为 48）
- **CV 融合算子**（含 `tl.dot`）：分核数 = **AI Core 核数**（如 910B 为 24）
- grid 优先用 1D，2D 也会被 NPU 合并为 1D
- 最大并发任务数不超过 65535

### 物理核绑定模式

NPU 是**物理核强绑定模式**，与 GPU 逻辑维度并行模式形成核心差异：

| 维度 | GPU (NVIDIA) | NPU (Ascend) |
|------|-------------|-------------|
| grid 本质 | 逻辑任务维度，和物理核解耦 | 物理核组映射（绑定 AI Core 拓扑） |
| 核数/维度限制 | grid 维度/大小无硬限制 | grid 大小 <= AI Core 总数，2D 需匹配拓扑 |
| 超核数行为 | 硬件自动调度 | 分批调度，产生额外开销 |

### 推荐的分核写法

```python
@triton.jit
def my_kernel(..., NUM_CORE: tl.constexpr):
    pid = tl.program_id(0)
    NUM_BLOCKS = tl.cdiv(n, BLOCK_SIZE)  # 或手动计算
    # 跨步分配任务：每个核从自己的ID开始，按总核数跨步
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        # 处理 block_idx 对应的数据
        ...

# 调用时固定核数
grid = (aicore_num,)  # 或 (vectorcore_num,)
my_kernel[grid](..., NUM_CORE=aicore_num)
```

### TRITON_ALL_BLOCKS_PARALLEL 环境变量

当逻辑核数大于物理核数时，设置 `TRITON_ALL_BLOCKS_PARALLEL=1` 可让编译器自动调整逻辑核数量为物理核数，减少调度开销。

**限制**: kernel 的逻辑必须对执行顺序不敏感才能开启，否则可能导致死锁。

## 存储单元约束

### 存储层次规格

| 组件 | 910/910A | 910B | 310/310P | 用途 |
|------|----------|------|----------|------|
| **HBM/DDR** | 32GB/64GB | 64GB | 8GB/16GB | 全局内存 |
| **L2 Cache** | 共享 | 共享 | 共享 | 芯片级缓存 |
| **L1 Buffer** | ~1MB | ~1MB | ~512KB | 矩阵计算数据缓存 |
| **UB (片上内存)** | ~2MB | 192KB (A2) | ~1MB | 向量计算输入输出 |

> **注意**: Atlas 800T/I A2 产品片上内存容量为 **192KB**（1572864 bits），开启 double buffer 后容量减半。

### 存储单元对齐要求

#### 200x 架构

| 存储单元 | 对齐要求 | 说明 |
|---------|---------|------|
| **UB** | 32字节 | Vector 计算数据来源，必须 32 字节对齐 |
| **L1 Buffer** | 32字节 | 矩阵计算数据缓存 |
| **L0A Buffer** | 512字节 | Cube 左矩阵输入，按分形大小对齐 |
| **L0B Buffer** | 512字节 | Cube 右矩阵输入，按分形大小对齐 |
| **L0C Buffer** | 64字节 | Cube 矩阵乘输出 |

#### 220x 架构

| 核 | 存储单元 | 对齐要求 | 说明 |
|----|---------|---------|------|
| **AIV** | UB | 32字节 | 向量计算单元数据来源 |
| **AIC** | L1 Buffer | 32字节 | 矩阵计算数据缓存 |
| **AIC** | L0A Buffer | 512字节 | Cube 左矩阵输入 |
| **AIC** | L0B Buffer | 512字节 | Cube 右矩阵输入 |
| **AIC** | L0C Buffer | 64字节 | Cube 矩阵乘输出 |

### UB OVERFLOW 问题

当 BLOCK_SIZE 过大导致片上内存不足时，编译会报错：

```
E [ConvertLinalgRToBinary] encounters error:
E ub overflow, requires xxxx bits while 1572864 bits available!
```

**解决方案**:
1. 减小 BLOCK_SIZE
2. 引入 BLOCK_SIZE_SUB 进行核内二次分块（Tiling）
3. 使用 `@triton.autotune` 自动搜索最优 BLOCK_SIZE

```python
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(0)
    base_offset = pid * BLOCK_SIZE
    num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)
    for sub_block_idx in range(num_sub_blocks):
        sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N
        data = tl.load(ptr + offsets, mask=mask)
        # ... 处理数据 ...
        tl.store(out_ptr + offsets, result, mask=mask)
```

### coreDim 超限问题

NPU 的 coreDim 参数不能超过 UINT16_MAX（65535）。当处理大规模数据时，过小的 BLOCK_SIZE 会导致 coreDim 超限。

**错误信息**: `coreDim=xxxx can't be greater than UINT16_MAX`

**解决方案**:
1. 设置 `TRITON_ALL_BLOCKS_PARALLEL=1`
2. 动态计算合适的 BLOCK_SIZE：

```python
min_block_size = triton.next_power_of_2(triton.cdiv(N, 65535))
BLOCK_SIZE = max(32768, min_block_size)
```

## 分形（Fractal）约束

### 分形大小规格

| 数据类型 | 分形大小 (MxN) | 单个分形字节数 |
|---------|---------------|---------------|
| **FP16** | 16x16 | 512字节 |
| **BF16** | 16x16 | 512字节 |
| **INT8** | 32x16 或 16x32 | 1024字节 |

**BLOCK_SIZE 应为分形大小的整数倍**，特别是使用 `tl.dot` 时：

```python
def check_fractal_constraint(block_M, block_N, block_K, dtype):
    fractal_size = 16 if dtype in ["float16", "bfloat16"] else 32
    assert block_M % fractal_size == 0, f"block_M={block_M} 必须是 {fractal_size} 的倍数"
    assert block_N % fractal_size == 0, f"block_N={block_N} 必须是 {fractal_size} 的倍数"
    assert block_K % fractal_size == 0, f"block_K={block_K} 必须是 {fractal_size} 的倍数"
```

## 数据对齐与访存优化

### 尾轴对齐约束

- **VV 类算子**（纯 Vector）：Tensor 尾轴大小必须能被 **32 字节** 整除
- **CV 类算子**（含 Cube+Vector）：Tensor 尾轴大小必须能被 **512 字节** 整除
- 若尾轴长度不足，硬件会自动补齐，导致性能恶化

### 尾轴不对齐的优化技巧：借轴转置

适用于 `tensor.numel() % 256 Byte == 0` 的场景：

```python
# conv_state = tensor([2048, 3], bfloat16)
# 当成1D tensor load，由于numel对齐，不会自动补齐
conv_state = tl.load(conv_state_ptr + offsets)
# 长轴(2048)裂出一根对齐轴(16)借给短轴(3)，让两个轴都对齐
conv_state_T = conv_state.reshape(128, 16 * 3).trans().reshape(16, 3 * 128).trans().reshape(3 * 2048,)
```

### 离散访存优化

在离散场景下，先将数据搬运到 UB，再从 UB 中 select 目标值：

```python
@triton.jit
def pick_kernel(x_ptr, idx_ptr, y_ptr, M: tl.constexpr, N: tl.constexpr):
    pid = tl.program_id(0)
    rn = tl.arange(0, N)
    idx = tl.load(idx_ptr + rn * stride_idx)
    mask = idx < M
    # 优化：先加载整块数据到UB，再用gather选取
    rm = tl.arange(0, M)
    x_shared = tl.load(x_ptr + rm * stride_x)
    val = tl.gather(x_shared, idx, 0)
    tl.store(y_ptr + rn * stride_y, val, mask=mask)
```

## 数据类型约束

### tl.dot 数据类型支持

| 输入类型 | Ascend A2/A3 | 说明 |
|---------|-------------|------|
| int8 | supported | - |
| int16 | supported | - |
| int32 | supported | - |
| fp16 | supported | - |
| fp32 | supported | - |
| bf16 | supported | - |
| uint8/uint16/uint32/uint64/fp64 | NOT supported | 硬件限制 |

**关键限制**:
- `acc` 不支持 fp16，硬件默认 fp32 累加
- `max_num_imprecise_acc` 暂不支持
- `out_dtype` 不支持 int8 和 fp16

### Vector 类型限制

| 操作 | 不支持的数据类型 | 影响 |
|------|---------------|------|
| Vector ADD | int64 | 退化为标量运算 |
| Vector CMP | int64/int32 | 退化为标量运算 |

**优化**: 在 `tl.where` 等使用比较的场合，将 int32/int64 转为 fp32 以利用 Vector 操作：

```python
# 优化前：cols < N 可能退化为scalar
xbar = tl.where(cols < N, x - mean, 0.0)

# 优化后：转为fp32利用vector
cols_cmp = cols.to(tl.float32)
xbar = tl.where(cols_cmp < N, x - mean, 0.0)
```

## 计算能力规格

| SOC 版本 | FP16 算力 | INT8 算力 | BF16 算力 | FP32 算力 |
|---------|----------|----------|----------|----------|
| **910** | 256 TFLOPS | 512 TOPS | - | 64 TFLOPS |
| **910A** | 256 TFLOPS | 512 TOPS | - | 64 TFLOPS |
| **910B** | ~320 TFLOPS | ~640 TOPS | ~320 TFLOPS | ~80 TFLOPS |
| **310** | 8 TFLOPS | 16 TOPS | - | 2 TFLOPS |
| **310P** | 8 TFLOPS | 22 TOPS | - | 2 TFLOPS |

### AICore 架构参数

| 参数 | 910/910A | 910B | 310/310P |
|------|----------|------|----------|
| **AIC 数量 (矩阵核)** | 32 | 24 | 2-4 |
| **AIV 数量 (向量核)** | 64 (2个/AIC) | 48 (2个/AIC) | 4-8 (2个/AIC) |
| **Cube Unit** | 有 | 有 | 有 |
| **Vector Unit** | 2个/AIC | 2个/AIC | 2个/AIC |

## 编译优化选项

以下选项可在 `triton.Config` 中配置：

| 选项 | 能力 | 默认值 | 说明 |
|------|------|--------|------|
| `multibuffer` | 开启流水并行数据搬运 | true | true/false |
| `unit_flag` | Cube 搬出优化 | None | true/false |
| `limit_auto_multi_buffer_only_for_local_buffer` | CV 算子优化 | None | true/false |
| `limit_auto_multi_buffer_of_local_buffer` | Cube double buffer scope | None | ["no-limit", "no-l0c"] |
| `set_workspace_multibuffer` | workspace 配置 | None | 如 [2,4] |
| `enable_hivm_auto_cv_balance` | CV 均衡 | None | true/false |
| `tile_mix_vector_loop` | Vector 切分 | None | 如 [2,4,8] |
| `tile_mix_cube_loop` | Cube 切分 | None | 如 [2,4,8] |
| `auto_blockify_size` | ALL_BLOCKS_PARALLEL 优化 | 1 | 如 [2,4,8] |

使用示例：

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024, 'multibuffer': True}),
        triton.Config({'BLOCK_SIZE': 2048, 'multibuffer': False}),
    ],
    key=['n_elements'],
)
@triton.jit
def my_kernel(...):
    ...
```

## 约束检查工作流

1. **识别目标 SOC 版本** - 默认使用 910B (DAV_3510)
2. **提取算子参数** - BLOCK_SIZE, BLOCK_SIZE_SUB, dtype 等
3. **逐项检查约束**:
   - 片上内存容量检查
   - 数据对齐检查（32B/512B/64B）
   - 分形大小检查（tl.dot 相关）
   - coreDim 超限检查
   - 数据类型兼容性检查
4. **输出检查报告** - 列出约束违反项、优化建议、推荐参数值

## 检查报告模板

```markdown
## 硬件约束检查报告

### 基本信息
- 算子名称: {op_name}
- 目标SOC: {soc_version}
- NPU架构: {npu_arch} (200x/220x)
- 数据类型: {dtype}

### 约束检查结果

| 检查项 | 状态 | 详情 |
|-------|------|------|
| UB容量 | PASS/FAIL | 当前使用: {ub_used}, 上限: {ub_max} |
| L1容量 | PASS/FAIL | 当前使用: {l1_used}, 上限: {l1_max} |
| 尾轴对齐(VV) | PASS/FAIL | 要求: 32B, 实际: {actual_align} |
| 尾轴对齐(CV) | PASS/FAIL | 要求: 512B, 实际: {actual_align} |
| coreDim限制 | PASS/FAIL | 当前: {core_dim}, 上限: 65535 |
| 分形约束(dot) | PASS/FAIL | Tile大小: {tile_size}, 分形: {fractal_size} |
| 数据类型支持 | PASS/FAIL | {dtype}: {support_status} |

### 优化建议
1. {suggestion_1}
2. {suggestion_2}

### 推荐参数
- BLOCK_SIZE: {recommended_value}
- BLOCK_SIZE_SUB: {recommended_value}
```

## 参考资源

- [Triton-Ascend 官方文档](https://triton-ascend.readthedocs.io/zh-cn/latest/index.html)
- [Triton-Ascend 编程指南 - 多核并行](file://M:/Desktop/tmp/AgentTest/triton-ascend/docs/zh/programming_guide.md)
- [Triton-Ascend 架构差异](file://M:/Desktop/tmp/AgentTest/triton-ascend/docs/zh/migration_guide/architecture_difference.md)
- [Triton-Ascend 性能优化指南](file://M:/Desktop/tmp/AgentTest/triton-ascend/docs/zh/migration_guide/performance_guidelines.md)
- [昇腾社区文档](https://www.hiascend.com/document)
