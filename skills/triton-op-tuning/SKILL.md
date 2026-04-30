---
name: triton-op-tuning
description: Triton-Ascend 算子性能调优技能，涵盖算子分类分析（memory/compute/UB bound）、profiling 工具、瓶颈定位、Tiling 优化、UB 实用上限（48KB）、autotune 配置范围、编译器选项调优。
---

# Triton-Ascend 算子调优技能

基于 Triton-Ascend 官方文档：
- `docs/zh/programming_guide.md` — 算子开发相关（多核并行、数据搬运、计算、autotune）
- `docs/zh/debug_guide/profiling.md` — 性能分析（msprof、瓶颈定位、优化案例）
- `docs/zh/migration_guide/architecture_difference.md` — 架构差异（Tiling 策略、编译选项）

当用户需要**分析或优化** Triton-Ascend 算子性能时使用本技能。

## 触发条件

- 用户提出"算子调优/性能优化/分析性能瓶颈"
- 算子性能不达预期，需要定位瓶颈
- 用户需要使用 msprof 工具分析算子
- 用户需要选择合适的 Tiling 参数
- 用户需要配置编译器优化选项

---

## 零、算子分类与优化策略

在调优前，先判断算子的瓶颈类型，不同类型适用不同优化手段:

| 算子类型 | 特征 | 瓶颈 | 有效的优化手段 | 代表算子 |
|---------|------|------|-------------|---------|
| **内存 bound** | 2~3 I/O, 极少计算 | GM 带宽 | 物理核绑定、大 BLOCK | add, copy, assign |
| **计算 bound** | 多步计算、归约 | Vector 计算 | 标量退化避免、constexpr 循环、multibuffer | softmax, layernorm, gelu |
| **UB bound** | 大中间张量 | UB 容量 | XBLOCK/XBLOCK_SUB 二级切分、rblock 规划 | reduce_sum, reduce_max |
| **CV 融合** | tl.dot + 后处理 | Cube/Vector 协调 | enable_auto_bind_sub_block、CV 负载均衡 | flash-attn, matmul+bias |

### 内存 bound 算子的优化效果 (vector_add 实测)

内存 bound 算子: kernel 结构优化（constexpr 循环、XBLOCK/XBLOCK_SUB）收益有限，效果主要来自物理核绑定减少调度开销。

| 数据量 | Naive(GPU grid) | Optimized(物理核绑定) | 提升 |
|-------|-----------------|---------------------|------|
| ≤16K | 更快（调度开销可忽略） | 较慢 | - |
| 256K | 4.074ms | 3.495ms | 1.17x |
| 1M | 10.733ms | 5.014ms | **2.14x** |
| 4M | 38.675ms | 11.035ms | **3.51x** |

**结论**: 小数据量（<64K）用 Naive 即可，大数据量必须物理核绑定。Autotune 可达 PyTorch 95% 性能。

---

## 一、性能分析工具

### msprof 工具概述

| 命令 | 用途 | 说明 |
|------|------|------|
| `msprof op` | 板上性能采集 | 在 NPU 上运行并采集实际性能数据 |
| `msprof op simulator` | 指令级仿真 | 无需 NPU 设备，分析指令流水线和瓶颈 |

### msprof op 使用流程

```bash
# 1. 采集性能数据
msprof op my_op_script.py

# 2. 查看结果
# 输出目录: ./profiler_data/
# 关键文件: msprof_*.json
```

### msprof op simulator 使用流程

```bash
# 指令级仿真分析
msprof op simulator my_op_script.py

# 输出包含:
# - 指令流水线可视化
# - MTE2/MTE3/Vector/Scalar 各阶段耗时
# - Bubble 分析（空闲等待）
```

### 性能数据关键指标

| 指标 | 含义 | 优化方向 |
|------|------|---------|
| **MTE2 时间** | 数据搬运 GM→UB | 减小搬运量 / 增大 BLOCK / 开启 multibuffer |
| **Vector 时间** | 向量计算耗时 | 优化计算逻辑 / 避免标量退化 |
| **Scalar 时间** | 标量计算耗时 | 将整数比较转为 float |
| **MTE3 时间** | 数据搬运 UB→GM | 减少写回次数 / 合并写入 |
| **Bubble 时间** | 流水线空闲 | 调整 Tiling / 开启 multibuffer 流水 |

---

## 二、性能瓶颈定位

### 典型瓶颈模式与解决方案

#### 瓶颈 1: 搬运受限 (MTE2 Bound)

**特征**: MTE2 时间占比高，Vector 利用率低。

**原因**: BLOCK_SIZE 太小，每轮搬运数据量不足。

**解决方案**:
```python
# 增大 BLOCK_SIZE，但注意 UB 限制
BLOCK_SIZE = min(4096, max_ub_elements)
# 开启 multibuffer 流水搬运
triton.Config({'BLOCK_SIZE': 2048}, multibuffer=True)
```

#### 瓶颈 2: 计算受限 (Vector Bound)

**特征**: Vector 时间占比高，但流水线存在 Bubble。

**原因**: 计算密集但搬运与计算串行。

**解决方案**: 开启 multibuffer 实现搬运/计算重叠。

#### 瓶颈 3: 标量退化 (Scalar Fallback)

**特征**: Scalar 时间异常高，Vector 利用率低。

**原因**: 整数比较操作（i32/i64）退化为标量执行。

**解决方案**:
```python
# 问题代码
xbar = tl.where(cols < N, x - mean, 0.0)  # cols 是 int32, 退化为 scalar

# 优化: 转为 float32 触发向量比较
cols_f = cols.to(tl.float32)
xbar = tl.where(cols_f < N, x - mean, 0.0)  # 现在使用 vector 比较
```

#### 瓶颈 4: UB 溢出

**特征**: 编译报错 `ub overflow, requires xxxx bits while 1572864 bits available!`

**解决方案**:
1. 减小 BLOCK_SIZE
2. 使用 XBLOCK/XBLOCK_SUB 二级切分
3. 使用 `@triton.autotune` 自动搜索

---

## 三、Tiling 优化策略

### 数据切分层次 (NPU 三级 Tiling)

来源: `architecture_difference.md`

```
总数据量 N
  → ncore 份 (核间切分)     → 每个 core 处理 xblock = N / ncore
    → xblock_sub 份 (核内切分) → 每次处理 xblock_sub 个元素
      → rblock 份 (reduce 轴切分) → 归约轴分块 (reduce 算子专用)
```

#### 核间切分 (ncore)

```python
num_cores = _get_num_vector_cores()  # 纯 Vector 算子
# 或
num_cores = properties["num_aicore"]  # 含 tl.dot 的算子

xblock = triton.cdiv(n_rows, num_cores)
grid = (num_cores, 1, 1)
```

**关键**: grid 大小 = 物理核数，不多不少。超过物理核数会产生调度开销。

#### 核内切分 (xblock / xblock_sub)

```python
@triton.jit
def kernel(..., XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    x_loops: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB

    for x_loop in range(x_loops):
        row_idx = xoffset + x_loop * XBLOCK_SUB + tl.arange(0, XBLOCK_SUB)
        row_mask = row_idx < n_rows
        # ... 处理 row_idx 对应的数据 ...
```

**xblock_sub 选择**:
- 默认推荐: 4~16
- UB 受限场景: 减小 xblock_sub 以降低单次 UB 占用
- UB 充裕场景: 增大 xblock_sub 以减少循环次数

#### 归约轴切分 (rblock)

仅用于 reduce 类算子（sum, max, min, argmax 等）：

```python
def _safe_rblock(n_cols, xblock_sub, num_tensors=1):
    """计算 UB 安全的 RBLOCK"""
    max_elements = (_UB_MAX_INPUT_ELEMENTS // num_tensors) // max(xblock_sub, 1)
    rblock = triton.next_power_of_2(min(n_cols, max_elements))
    return max(rblock, 1)
```

**num_tensors 参数**:
- `num_tensors=1`: 只需要输入 `[xsub, RBLOCK]`（max, min, argmax, argmin）
- `num_tensors=2`: 需要输入 + 累加器 `[xsub, RBLOCK] × 2`（sum pairwise, xor）

### UB 容量规划

Atlas A2 UB = 192 KB = 196,608 bytes = 49,152 个 float32 元素。

**重要: 实用上限 48KB，不是 96KB（doublebuffer slot）或 192KB（总量）**

编译器需要中间变量空间，填满 UB 会导致 overflow。实测验证的安全上限:
```python
_UB_PRACTICAL_LIMIT = 48 * 1024  # 48KB 实用上限
# element-wise 3 I/O 张量: 48KB / 12B = 4096 元素 (不是 8192)
```

**开启 multibuffer (doublebuffer) 后容量再减半**: 可用约 24KB = 6,144 个 float32。

**安全 block size 计算**:
```python
def _safe_block_size(element_bytes, num_tensors=3):
    """使用 48KB 实用上限，而非理论 96KB"""
    max_elements = _UB_PRACTICAL_LIMIT // (num_tensors * element_bytes)
    return max(1 << (max_elements.bit_length() - 1), 64)
```

**UB 使用量估算**:
```python
ub_elements = 0
ub_elements += xblock_sub * rblock  # 输入块
ub_elements += xblock_sub * rblock  # 累加器 (pairwise)
ub_elements += xblock_sub           # 输出标量
# 确保 ub_elements * element_size <= UB_CAPACITY
```

### Autotune BLOCK_SIZE 搜索范围

根据算子类型选择搜索范围 (实测验证):

| 算子类型 | BLOCK_SIZE 范围 | 说明 |
|---------|----------------|------|
| element-wise (add, relu) | 512 ~ 4096 | 受 UB 实用上限约束 |
| 归约 (sum, max) | 256 ~ 1024 | rblock 受 UB 约束更紧 |
| LayerNorm/Softmax | next_power_of_2(N) | 需覆盖完整行 |
| GEMM | 64/128 (fractal 倍数) | FP16/BF16: 16 的倍数 |

---

## 四、编译器优化选项

### 通用优化选项

来源: `architecture_difference.md` 和 `programming_guide.md`

| 选项 | 说明 | 默认值 | 使用场景 |
|------|------|--------|---------|
| `multibuffer` | 流水并行（搬运/计算重叠） | True | 几乎所有算子，除非 UB 紧张 |
| `unit_flag` | Cube 搬出优化 | None | CV 融合算子 |
| `auto_blockify_size` | ALL_BLOCKS_PARALLEL 优化 | 1 | 核数超限时 |

### CV 融合算子专用选项

| 选项 | 说明 | 使用场景 |
|------|------|---------|
| `enable_auto_bind_sub_block` | 自动绑定子块 (CV融合) | CV 融合算子 |
| `enable_hivm_auto_cv_balance` | 自动 CV 负载均衡 | Cube/Vector 工作不均衡 |
| `tile_mix_vector_loop` | Vector 循环 Tiling | CV 融合，控制 Vector 切分 |
| `tile_mix_cube_loop` | Cube 循环 Tiling | CV 融合，控制 Cube 切分 |
| `sync_solver` | 自动优化同步点 | CV 融合，减少同步开销 |

### Autotune 配置示例

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, multibuffer=True),
        triton.Config({'BLOCK_SIZE': 2048}, multibuffer=True),
        triton.Config({'BLOCK_SIZE': 4096}, multibuffer=False),  # UB 受限时关闭
    ],
    key=['n_elements'],
)
@triton.jit
def my_kernel(...):
    ...
```

### CV 融合 Autotune 示例

```python
@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
            enable_auto_bind_sub_block=True,
            enable_hivm_auto_cv_balance=True,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
            enable_auto_bind_sub_block=True,
            enable_hivm_auto_cv_balance=True,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_kernel(...):
    ...
```

---

## 五、标量退化避免

### 问题描述

NPU 的 Vector 单元原生不支持 i32/i64 比较运算。当 Triton kernel 中出现整数比较（如 `tl.where(cols < N, ...)`），编译器会退化为 Scalar 标量执行，性能大幅下降。

### 受影响的场景

| 操作 | 触发条件 | 影响 |
|------|---------|------|
| `tl.where(int_cond, a, b)` | 条件为 int32/int64 | 逐元素标量比较 |
| `tl.where(cols < N, ...)` | cols 为 tl.arange (int32) | 最常见的退化场景 |
| 整数比较 in mask | `rmask = rindex < n_cols` | 通常 OK（load mask 不走 Vector） |

### 解决方案

```python
# 方案 1: 将索引转为 float 再比较
cols_f = cols.to(tl.float32)
xbar = tl.where(cols_f < N, x - mean, 0.0)

# 方案 2: 直接使用 float 比较（load 的 other 参数）
x = tl.load(ptr, mask=rmask, other=0.0).to(tl.float32)
# mask 中的整数比较通常由 MTE2 引擎处理，不退化为 scalar
```

### LayerNorm 优化案例

来源: `profiling.md`

```python
# 优化前: Scalar 占比高
xbar = tl.where(cols < N, x - mean, 0.0)  # cols < N 退化为 scalar

# 优化后: Vector 充分利用
cols_f = cols.to(tl.float32)
xbar = tl.where(cols_f < N, x - mean, 0.0)  # float 比较，走 vector
```

---

## 六、优化工作流

### 调优流程

```
1. 基准测试 → triton.testing.do_bench 测量延迟和带宽
     ↓
2. 性能采集 → msprof op 采集板上数据
     ↓
3. 瓶颈分析 → 查看 MTE2/Vector/Scalar/Bubble 占比
     ↓
4. 定位优化点:
   - MTE2 高 → 增大 BLOCK_SIZE / 开 multibuffer
   - Vector 高 → 优化计算逻辑
   - Scalar 高 → 检查整数比较，转为 float
   - Bubble 高 → 调整 Tiling / 开 multibuffer
     ↓
5. 验证 → 重新 benchmark 对比
     ↓
6. 迭代 → autotune 搜索最优参数
```

### 快速优化检查清单

- [ ] grid 绑定物理核数（vectorcore_num / aicore_num）
- [ ] BLOCK_SIZE 为 2 的幂
- [ ] 尾轴 32B 对齐（VV 算子）/ 512B 对齐（CV 算子）
- [ ] multibuffer 开启（UB 充裕时）
- [ ] 避免整数比较退化为标量（`.to(tl.float32)`）
- [ ] UB 使用量不超限（考虑 doublebuffer 减半）
- [ ] 使用 autotune 搜索最优参数
- [ ] 连续访存优先（避免离散索引）

---

## 参考资源

- [Triton-Ascend 编程指南](https://triton-ascend.readthedocs.io/zh-cn/latest/programming_guide.html) — 多核并行、数据搬运、计算、autotune
- [Triton-Ascend 性能分析](https://triton-ascend.readthedocs.io/zh-cn/latest/debug_guide/profiling.html) — msprof 工具、瓶颈定位、优化案例
- [Triton-Ascend 架构差异](https://triton-ascend.readthedocs.io/zh-cn/latest/migration_guide/architecture_difference.html) — Tiling 策略、编译选项
- 本地文档: `triton-ascend/docs/zh/programming_guide.md`
- 本地文档: `triton-ascend/docs/zh/debug_guide/profiling.md`
- 本地文档: `triton-ascend/docs/zh/migration_guide/architecture_difference.md`
