---
name: operator-development-guide
description: Triton-Ascend 算子开发核心知识，涵盖多核并行策略、数据搬运优化、计算模式（Vector/GEMM/Fused）、Tiling 策略、UB 容量规划、硬件架构详情、Fractal 对齐约束。综合官方文档和 skills 知识库。
type: reference
---

# Triton-Ascend 算子开发指南

来源: Triton-Ascend 官方文档 (2026-04-29 采集)

## 1. 多核并行策略

### 核心原则
- NPU 是**物理核强绑定模式**，grid 大小应等于物理核数
- GPU 的 "大量逻辑线程自动调度" 模式不适用于 NPU
- 超过物理核数的 grid 会产生调度开销（分批执行）

### 核数获取
```python
import triton.runtime.driver as driver
device = torch.npu.current_device()
props = driver.active.utils.get_device_properties(device)
vectorcore_num = props["num_vectorcore"]  # 纯 Vector 算子
aicore_num = props["num_aicore"]          # 含 tl.dot 的算子
```

### Atlas A2 (910B) 核数
- AI Core = 24
- Vector Core = 48 (每个 AI Core 有 2 个 Vector Core)
- AI Core = 1 Cube Unit + 2 Vector Units

### 三种并行模式

**模式 1: 物理核绑定（推荐）**
```python
grid = (num_cores, 1, 1)
# kernel 内循环处理多个 block
for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
    ...
```

**模式 2: TRITON_ALL_BLOCKS_PARALLEL**
```bash
export TRITON_ALL_BLOCKS_PARALLEL=1
```
- 编译器自动将逻辑核数调整为物理核数
- 仅适用于 kernel 逻辑对执行顺序不敏感的情况

**模式 3: 传统 2D Grid**
```python
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
```
- NPU 会将 2D grid 合并为 1D
- 可能产生调度开销

## 2. 数据搬运优化

### 尾轴对齐要求
- VV 类算子（纯 Vector）: 尾轴 32B 对齐
- CV 类算子（含 Cube+Vector）: 尾轴 512B 对齐

### 连续访存 vs 离散访存
- **连续**: `tl.load(ptr + offsets)` — 最优，硬件自动合并
- **离散**: `tl.load(ptr + offsets * stride)` — stride=1 时等同于连续
- **非常离散**: 使用 `tl.gather` 先加载整块再选取

### 计算存储并行 (Multibuffer)
- 默认开启 (multibuffer=True)
- 实现 GM↔UB 搬运与计算的流水重叠（乒乓缓冲）
- **代价**: UB 可用容量减半（doublebuffer）
- UB 紧张时可关闭: `triton.Config({'BLOCK_SIZE': 2048}, multibuffer=False)`

## 3. 计算模式

### 归约操作
| API | 返回类型 | 适用场景 |
|-----|---------|---------|
| `tl.sum(x, axis)` | 同输入类型 | 求和 |
| `tl.max(x, axis)` | 同输入类型 | 最大值 |
| `tl.min(x, axis)` | 同输入类型 | 最小值 |
| `tl.argmax(x, axis)` | int32 | 最大值索引 |
| `tl.argmin(x, axis)` | int32 | 最小值索引 |
| `tl.xor_sum(x, axis)` | 同输入类型 | 整数 XOR 归约 |
| `tl.reduce(x, axis, combine_fn)` | 自定义 | 通用归约（自定义合并函数） |

### Pairwise vs Simple 累加策略（Reduce Sum）

**Simple**: `acc += tl.sum(x, axis=1)` — 每轮做一次归约
- 优点: 快，UB 占用少（只需 1D 累加器）
- 缺点: 每轮归约引入误差，多次叠加

**Pairwise**: 逐元素累加到 `[xsub, RBLOCK]`，最后一次 `tl.sum(RBLOCK)`
- 优点: 精度更好（RBLOCK 越大精度越好）
- 缺点: UB 占用翻倍（需要 2D 累加器）

**RBLOCK 越大 → r_loops 越少 → 精度越好**

### tl.reduce 自定义归约
```python
@triton.jit
def _product_combine(a, b):
    return a * b

# 在 kernel 中使用
tile_result = tl.reduce(x, 1, _product_combine)
acc = _product_combine(acc, tile_result)
```

## 4. Tiling 策略（三级切分）

来源: architecture_difference.md

```
总数据量 (n_rows, n_cols)
  ↓ 核间切分
ncore 份 → 每核 xblock = n_rows / ncore
  ↓ 核内切分
xblock_sub 份 → 每次处理 xblock_sub 行
  ↓ 归约轴切分 (reduce 算子)
rblock 份 → 每次处理 rblock 列
```

### 推荐参数范围
| 参数 | 范围 | 说明 |
|------|------|------|
| num_cores | 物理核数 | 固定值 |
| xblock | cdiv(n_rows, num_cores) | 自动计算 |
| xblock_sub | 4~16 | UB 受限减小，充裕增大 |
| rblock | next_power_of_2(n_cols) | 受 UB 限制 |

## 5. UB 容量规划

### Atlas A2 规格
- 总 UB = 192 KB
- 实用上限 = ~48 KB (余量给中间变量)
- `_UB_MAX_INPUT_ELEMENTS = 12 * 1024 = 12,288` (float32)

### UB 安全 RBLOCK 计算
```python
def _safe_rblock(n_cols, xblock_sub, num_tensors=1):
    """num_tensors: 需占用 [xsub, RBLOCK] 的张量数"""
    max_elements = (UB_MAX // num_tensors) // max(xblock_sub, 1)
    rblock = triton.next_power_of_2(min(n_cols, max_elements))
    return max(rblock, 1)
```

- `num_tensors=1`: max/min/argmax/argmin（只需输入缓冲）
- `num_tensors=2`: sum pairwise/xor（输入 + 累加器）

## 6. Autotune 使用

### 基本模式
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
    ],
    key=['n_elements'],  # 根据哪些参数变化重新调优
)
@triton.jit
def kernel(...):
    ...
```

### 带 NPU 编译选项
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, multibuffer=True),
        triton.Config({'BLOCK_SIZE': 2048}, multibuffer=True),
    ],
    key=['n_elements'],
)
```

### GEMM Autotune
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}),
    ],
    key=['M', 'N', 'K'],
)
```

## 7. GEMM / Cube 算子开发

来源: `skills/triton-op-cube/SKILL.md`

### Fractal 对齐约束

Cube Core (tl.dot) 的块大小必须为 fractal 大小的整数倍:

| 数据类型 | Fractal 尺寸 | 每块字节数 |
|---------|-------------|-----------|
| FP16 | 16×16 | 512B |
| BF16 | 16×16 | 512B |
| INT8 | 32×16 或 16×32 | 1024B |

**规则**: BLOCK_M、BLOCK_N 必须是 fractal 尺寸的倍数（FP16/BF16: 16 的倍数，INT8: 16 或 32 的倍数）。BLOCK_K 同理。

### GEMM Kernel 模板

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_m_blocks
    pid_n = pid // num_m_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)
```

### 持久化 Kernel 模式

大矩阵推荐使用持久化 kernel，绑定 `aicore_num` 核数，核内循环处理:

```python
grid = (aicore_num,)
num_blocks = tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N)

for block_idx in range(pid, num_blocks, aicore_num):
    # 解析 block_idx → (pid_m, pid_n)
    pid_m = block_idx % num_m_blocks
    pid_n = block_idx // num_m_blocks
    # ... 执行 tl.dot ...
```

### GEMV 模式 (M=1)

当 M=1 时退化为向量内积:
```python
grid = (triton.cdiv(N, BLOCK_N),)
# 核内: tl.sum(a[:, None] * b, axis=0) 做内积
```

### 转置 GEMM

B 矩阵需要转置时，交换 stride 索引:
```python
# B^T: 原 B 布局为 [K, N]，需要 [N, K] 访问模式
b_ptrs = b_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk
```

### tl.dot 数据类型约束

| 特性 | 说明 |
|------|------|
| 支持输入 | int8, int16, int32, fp16, fp32, bf16 |
| 不支持输入 | uint8, uint16, uint32, uint64, fp64 |
| 累加器 | 必须 fp32（不支持 fp16 累加） |
| out_dtype | 不支持 int8/fp16 |

## 8. Fused 算子开发 (Cube + Vector)

来源: `skills/triton-op-fused/SKILL.md`

### CV 融合架构

```
GM → UB → tl.dot (Cube Core) → vector 后处理 → GM
          ↑ MTE2 搬入          ↑ Vector 处理    ↑ MTE3 写回
```

编译器自动分离 Cube 和 Vector 操作，Cube 处理 `tl.dot`，Vector 处理 element-wise 后处理（激活函数、归一化等）。

### Grid 选择

| 矩阵规模 | Grid 模式 | 核数 |
|---------|----------|------|
| 小矩阵 | 2D grid: `(M/BLOCK_M, N/BLOCK_N)` | 自动 |
| 大矩阵 | 1D 持久化: `(aicore_num,)` | 物理核数 |

### 同步原语

CV 融合算子中 Cube 和 Vector 的同步:
```python
tl.sync_block_wait("mte2")   # 等待 GM→UB 搬运完成
tl.sync_block_set("mte3")    # 通知 UB→GM 写回完成
tl.sync_block_all()           # 全局同步
```

流水线名称: `"mte1"`, `"mte2"`, `"mte3"`, `"m"`, `"v"`, `"fix"`

### Flash Attention 模板

在线 softmax + 多次 tl.dot:
```python
# QK^T → Score → Score@V
qk = tl.dot(q, k_trans)                    # Cube
score = tl.softmax(qk * scale, axis=-1)    # Vector
out = tl.dot(score, v)                     # Cube
```

数值稳定性: softmax 前减去行最大值。

## 9. 硬件架构详情

来源: `skills/triton-op-hardware-constraints/SKILL.md`

### SOC 架构代际

| 代际 | 架构 | Cube+Vector | 型号 |
|------|------|------------|------|
| 200x | Cube+Vector 共享 Scalar | Atlas 310/310P/910/910A |
| 220x | Cube+Vector 分离 | Atlas A2/A3 (910B) |

### 存储层级规格

| 资源 | 容量 | 对齐 |
|------|------|------|
| UB (Atlas A2) | 192 KB | VV: 32B, CV L0A/L0B: 512B, L0C: 64B |
| L1 (910B) | ~1 MB | - |
| L1 (310P) | ~512 KB | - |

### 数据类型操作约束

| 操作 | 不支持的类型 |
|------|------------|
| Vector ADD | int64 → 标量退化 |
| Vector CMP | int64, int32 → 标量退化 |
| tl.dot | uint8/16/32/64, fp64 |
| tl.dot out_dtype | int8, fp16 |

### 常见错误及修复

| 错误信息 | 原因 | 修复 |
|---------|------|------|
| `ub overflow, requires xxxx bits while 1572864 bits available!` | BLOCK 太大 | 减小 BLOCK_SIZE 或使用 XBLOCK_SUB 二级切分 |
| `coreDim=xxxx can't be greater than UINT16_MAX` | grid 超 65535 | 增大 BLOCK_SIZE 或设 `TRITON_ALL_BLOCKS_PARALLEL=1` |
| Scalar 占比高 | 整数比较退化 | `cols.to(tl.float32)` 转浮点比较 |

## 10. NPU 特有注意事项汇总

### 设备初始化
```python
import torch_npu
device = torch.npu.current_device()  # 注意: 不是 torch.cuda
```

### Autotune NPU 限制
- **不支持** `num_warps` 参数（GPU 独有概念）
- **不支持** `num_stages` 参数
- 支持 NPU 专属选项: `multibuffer`, `enable_auto_bind_sub_block` 等

### 离散访存优化
优先连续访存 (`ptr + offsets`)。离散访问时先加载整块到 UB，再用 `tl.gather` 选取，避免逐元素标量访问。
