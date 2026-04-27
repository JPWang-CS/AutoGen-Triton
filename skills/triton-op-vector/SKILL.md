---
name: triton-op-vector
description: 生成纯向量算子（元素级计算、激活函数、归约、归一化），使用 Triton 标准API映射到 Ascend NPU Vector Core。基于 triton-ascend 官方文档、programming_guide 中 vector_add 和 layernorm 示例、以及 profiling.md 中的向量优化建议。支持 autotune、多核并行、UB溢出处理等 NPU 优化。
---

# 纯向量算子技能 (Vector Core)

基于 `triton-ascend/docs/en/programming_guide.md`、`third_party/ascend/tutorials/01-vector-add.py`、`profiling.md` 中的向量优化建议和 `migration_guide/migrate_from_gpu.md` 中的迁移示例。

当用户需要生成**纯向量计算**算子时使用本技能。此类算子仅使用 Vector Core 进行元素级运算、归约、激活函数、归一化等。

## 核心 API

### 数据搬运

| API | 说明 |
|-----|------|
| `tl.load(ptr + offsets, mask=None, other=0.0)` | 从全局内存加载数据到片上 (UB) |
| `tl.store(ptr + offsets, value, mask=None)` | 将数据从片上 (UB) 写回全局内存 |
| `tl.arange(start, end)` | 生成连续整数索引 [start, end) |

### 元素级数学运算

| API | 说明 |
|-----|------|
| `x + y`, `x - y`, `x * y`, `x / y` | 算术运算 |
| `tl.exp(x)` | 指数 |
| `tl.log(x)` | 自然对数 |
| `tl.sqrt(x)` | 平方根 |
| `tl.rsqrt(x)` | 1/sqrt(x) |
| `tl.abs(x)` | 绝对值 |
| `tl.sin(x)`, `tl.cos(x)` | 三角函数 |
| `tl.sigmoid(x)` | Sigmoid: 1/(1+exp(-x)) |
| `tl.where(cond, x, y)` | 条件选择 |
| `tl.maximum(x, y)`, `tl.minimum(x, y)` | 逐元素最大/最小 |
| `tl.clamp(x, min_val, max_val)` | 截断到 [min_val, max_val] |

### 归约运算

| API | 说明 |
|-----|------|
| `tl.sum(x, axis=None)` | 沿指定轴求和 |
| `tl.max(x, axis=None)` | 沿指定轴求最大值 |
| `tl.min(x, axis=None)` | 沿指定轴求最小值 |
| `tl.argmax(x, axis=None)` | 沿指定轴求最大值索引 |
| `tl.argmin(x, axis=None)` | 沿指定轴求最小值索引 |

### 类型转换

| API | 说明 |
|-----|------|
| `x.to(tl.float32)` | 转换为 float32 |
| `x.to(tl.float16)` | 转换为 float16 |
| `x.to(tl.int32)` | 转换为 int32 |

## Vector Add 模板 (基础)

来源: `triton-ascend/third_party/ascend/tutorials/01-vector-add.py`

```python
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,         # 输入向量 x 的指针
    y_ptr,         # 输入向量 y 的指针
    output_ptr,    # 输出向量指针
    n_elements,    # 向量长度
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


# 验证
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='npu')
y = torch.rand(size, device='npu')
output_torch = x + y
output_triton = triton_add(x, y)
print(f'Max diff: {torch.max(torch.abs(output_torch - output_triton))}')
```

## Vector Add 模板 (多核优化)

使用物理核心数绑定, 并在 kernel 内循环处理多个 block。

来源: `triton-ascend/docs/en/programming_guide.md`

```python
import triton.runtime.driver as driver

device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
vectorcore_num = properties["num_vectorcore"]

@triton.jit
def add_kernel_opt(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    NUM_CORE = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(n_elements, BLOCK_SIZE)

    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        block_start = block_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)


def triton_add_opt(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = (vectorcore_num,)
    add_kernel_opt[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

## Autotune 模板

来源: `triton-ascend/docs/en/programming_guide.md` autotune 示例

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['n_elements'],
)
@triton.jit
def add_kernel_autotune(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

## Element-wise 激活函数模板

### ReLU

```python
@triton.jit
def relu_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    NUM_CORE = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(n_elements, BLOCK_SIZE)

    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
        result = tl.where(x > 0, x, 0.0)
        tl.store(output_ptr + offsets, result, mask=mask)
```

### Sigmoid

```python
@triton.jit
def sigmoid_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    NUM_CORE = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(n_elements, BLOCK_SIZE)

    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
        result = tl.sigmoid(x)
        tl.store(output_ptr + offsets, result, mask=mask)
```

### GELU (近似)

```python
@triton.jit
def gelu_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    NUM_CORE = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(n_elements, BLOCK_SIZE)

    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
        # GELU 近似: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = 0.7978845608028654
        coeff = 0.044715
        inner = sqrt_2_over_pi * (x + coeff * x * x * x)
        result = 0.5 * x * (1.0 + tl.where(inner > 20.0, 1.0,
                           tl.where(inner < -20.0, -1.0,
                                    tl.exp(2.0 * inner) - 1.0) /
                                    (tl.exp(2.0 * inner) + 1.0)))
        tl.store(output_ptr + offsets, result, mask=mask)
```

### SiLU (Swish)

```python
@triton.jit
def silu_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    NUM_CORE = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(n_elements, BLOCK_SIZE)

    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
        result = x * tl.sigmoid(x)  # SiLU = x * sigmoid(x)
        tl.store(output_ptr + offsets, result, mask=mask)
```

## Softmax 模板

```python
@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    N,  # softmax 沿最后一个维度的长度
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    NUM_CORE = tl.num_programs(0)

    # 每个 core 处理一行: 单行 softmax
    row_idx = pid
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(input_ptr + row_idx * N + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # 数值稳定: 减去最大值
    x_max = tl.max(x, axis=0)
    exp_x = tl.exp(x - x_max)
    sum_exp = tl.sum(exp_x, axis=0)
    result = exp_x / sum_exp

    tl.store(output_ptr + row_idx * N + offs, result, mask=mask)
```

## Reduce (归约) 模板

```python
@triton.jit
def reduce_sum_kernel(
    input_ptr, output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    result = tl.sum(x, axis=0)

    # 原子加法累加到输出 (多核心场景)
    tl.atomic_add(output_ptr, result)
```

## LayerNorm 模板

来源: `triton-ascend/docs/en/debug_guide/profiling.md` 中的 layernorm 示例

```python
@triton.jit
def layernorm_kernel(
    X,        # 输入 [M, N]
    W,        # 权重 gamma [N]
    B,        # 偏置 beta [N]
    Mean,     # 输出均值 [M]
    Rstd,     # 输出标准差倒数 [M]
    Out,      # 输出 [M, N]
    stride_x, stride_out,
    M, N,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)

    # 计算偏移
    x_ptr = X + row * stride_x
    out_ptr = Out + row * stride_out

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # 加载一行数据
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # 计算均值
    mean = tl.sum(x, axis=0) / N
    tl.store(Mean + row, mean)

    # 计算方差 (注意: 使用 float32 进行 cols 的比较以启用向量计算)
    cols_f = cols.to(tl.float32)
    xbar = tl.where(cols_f < N, x - mean, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)

    # 加载 gamma 和 beta
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm: out = (x - mean) * rstd * gamma + beta
    out = (x - mean) * rstd * w + b
    tl.store(out_ptr + cols, out, mask=mask)


def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    M, N = x.shape
    out = torch.empty_like(x)
    mean = torch.empty((M,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((M,), device=x.device, dtype=torch.float32)

    grid = (M,)
    layernorm_kernel[grid](
        x, weight, bias, mean, rstd, out,
        x.stride(0), out.stride(0),
        M, N, eps,
        BLOCK_N=triton.next_power_of_2(N),
    )
    return out
```

## RMSNorm 模板

```python
@triton.jit
def rmsnorm_kernel(
    X, W, Out,
    stride_x, stride_out,
    M, N,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = X + row * stride_x
    out_ptr = Out + row * stride_out

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # RMS: sqrt(mean(x^2))
    x2 = x * x
    mean_x2 = tl.sum(x2, axis=0) / N
    rrms = 1.0 / tl.sqrt(mean_x2 + eps)

    # RMSNorm: out = x * rrms * weight
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = x * rrms * w
    tl.store(out_ptr + cols, out, mask=mask)
```

## UB 溢出处理: 子块划分模板

当 BLOCK_SIZE 过大导致 UB 溢出时, 使用 BLOCK_SIZE + BLOCK_SIZE_SUB 模式。

来源: `triton-ascend/docs/en/programming_guide.md` 和 `migration_guide/migrate_from_gpu.md`

```python
@triton.jit
def elementwise_kernel_subblock(
    input_ptr, output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    base_offset = pid * BLOCK_SIZE

    # 计算需要处理的子块数
    num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)

    for sub_block_idx in range(num_sub_blocks):
        sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N

        x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        result = tl.where(x > 0, x, 0.0)  # 示例: ReLU
        tl.store(output_ptr + offsets, result, mask=mask)


def elementwise_subblock(x: torch.Tensor):
    N = x.numel()
    out = torch.empty_like(x)
    MAIN_BLOCK_SIZE = 8192
    SUB_BLOCK_SIZE = 1024

    grid = lambda meta: (triton.cdiv(N, MAIN_BLOCK_SIZE),)
    elementwise_kernel_subblock[grid](
        x, out, N,
        BLOCK_SIZE=MAIN_BLOCK_SIZE,
        BLOCK_SIZE_SUB=SUB_BLOCK_SIZE,
    )
    return out
```

## 尾轴对齐优化

来源: `triton-ascend/docs/en/programming_guide.md`

当张量尾轴不是32字节的倍数时 (如 shape=(2048, 3), dtype=bfloat16), 可以通过"借轴转置"避免自动填充:

```python
@triton.jit
def aligned_process_kernel(
    input_ptr, output_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 如果 N=3 (不对齐), 可以将数据作为1D加载, 然后reshape
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)
    total = M * N
    NUM_BLOCKS = tl.cdiv(total, BLOCK_M)
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        offsets = block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = offsets < total
        data = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        # 处理...
        result = tl.where(data > 0, data, 0.0)
        tl.store(output_ptr + offsets, result, mask=mask)
```

## 性能优化指南

### BLOCK_SIZE 选择

| 场景 | 推荐 BLOCK_SIZE | 说明 |
|------|----------------|------|
| 元素级运算 | 1024, 2048, 4096 | 尽量大, 不超出 UB |
| 归约运算 | 256, 512, 1024 | 取决于归约维度 |
| LayerNorm | triton.next_power_of_2(N) | 覆盖完整行 |
| Softmax | triton.next_power_of_2(N) | 覆盖完整行 |

### NPU 向量化要点

1. **避免 i64/i32 比较**: 使用 `cols.to(tl.float32)` 替代 `cols < N` 用于 `tl.where` 中的条件, 以启用向量计算
2. **连续内存访问**: `offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` 是连续的
3. **32字节对齐**: BLOCK_SIZE * sizeof(dtype) 应为 32 的倍数
4. **多核绑定**: 纯向量算子 grid 设为 `vectorcore_num`

### NPU 特有: 离散访存优化

来源: `triton-ascend/docs/en/programming_guide.md` "Transferring Data to the UB"

当使用离散索引访问时, 先将整块数据加载到UB, 再用 `tl.gather` 选择:

```python
@triton.jit
def pick_kernel(
    x_ptr, idx_ptr, y_ptr,
    stride_x, stride_idx, stride_y,
    M: tl.constexpr, N: tl.constexpr,
):
    pid = tl.program_id(0)
    rn = tl.arange(0, N)
    idx = tl.load(idx_ptr + rn * stride_idx)
    mask = idx < M

    # 优化: 先加载整块到UB, 再gather
    rm = tl.arange(0, M)
    x_shared = tl.load(x_ptr + rm * stride_x)  # [M]
    val = tl.gather(x_shared, idx, 0)

    tl.store(y_ptr + rn * stride_y, val, mask=mask)
```

## 约束

- **grid绑定vectorcore_num**: 纯向量算子应使用 `vectorcore_num` 作为 grid 大小
- **BLOCK_SIZE对齐**: BLOCK_SIZE * sizeof(dtype) >= 32 bytes
- **UB容量**: Atlas A2 UB = 192 KB; 确保所有中间张量总量不超限
- **避免标量退化**: 在 `tl.where` 中避免直接使用 int64 索引做比较, 应先 `.to(tl.float32)`
- **连续访问**: 优先使用连续偏移模式; 离散访问用 gather 模式
- **mask保护**: 所有 load/store 必须使用 mask 防止越界

## 参考

- 编程指南 (含vector_add、matmul、autotune、tiling): `triton-ascend/docs/en/programming_guide.md`
- 性能分析 (含layernorm、向量优化): `triton-ascend/docs/en/debug_guide/profiling.md`
- 迁移指南 (含UB溢出、coreDim处理): `triton-ascend/docs/en/migration_guide/migrate_from_gpu.md`
- 向量加法示例: `triton-ascend/third_party/ascend/tutorials/01-vector-add.py`
- 在线文档: https://triton-ascend.readthedocs.io/zh-cn/latest/index.html
