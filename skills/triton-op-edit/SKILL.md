---
name: triton-op-edit
description: 修改现有Triton-Ascend算子时，生成 claude.local.md 输入模板并据此新增/删除/修改参数（shape/dtype/BLOCK_SIZE等）
---

# 修改 Triton-Ascend 算子技能

当用户提出"修改Triton算子/调整参数/变更 shape 或 dtype/优化 BLOCK_SIZE"时使用本技能。

## 工作流

1. 确认目标算子路径与算子名称。
2. 仅在触发本技能时才生成 `claude.local.md`。
3. 若 `claude.local.md` 不存在，则创建并写入"简化的修改输入模板"。
4. 支持用户用一句话描述变更，必要时再追问补充。
5. 根据输入执行修改：
   - 新增参数（输入/输出/属性）
   - 删除参数（输入/输出/属性）
   - 修改参数的 shape / dtype 约束
   - 调整 BLOCK_SIZE / BLOCK_SIZE_SUB（Tiling 参数）
   - 修改流水线参数（multibuffer 等）
   - 添加/移除优化选项（autotune、编译选项等）
6. 输出改动说明与假设。

## 修改范围（默认）

- 主要修改：
  - 核心 kernel 实现文件（`*.py`）
  - 测试文件（`test_*.py`）
  - 基准测试文件（`benchmark_*.py`）
  - README 文档（如有参数变更）

## 常见修改场景

### 1. 修改数据类型

Triton-Ascend 支持的数据类型及注意事项：

```python
# 修改 dtype 参数
dtype = "float16"    # 支持
dtype = "bfloat16"   # 支持
dtype = "float32"    # 支持
dtype = "int8"       # tl.dot 支持
dtype = "int32"      # 支持
# 注意：uint8/uint16/uint32/uint64/fp64 不被 tl.dot 支持

# 修改 dtype 时需要同步修改：
# 1. torch tensor 创建时的 dtype
# 2. kernel 参数声明
# 3. 精度比对的 rtol/atol

# float16 -> bfloat16 示例
# torch 端
x = torch.randn(shape, dtype=torch.bfloat16, device='npu')

# kernel 端（如果 dtype 是 constexpr 参数）
# 通常 Triton kernel 通过指针参数自动适配 dtype
```

### 2. 调整 BLOCK_SIZE / Tiling 参数

```python
# 修改前
def my_kernel(..., BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ...

# 修改后：添加 BLOCK_SIZE_SUB 实现核内二次分块
def my_kernel(..., BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(0)
    base_offset = pid * BLOCK_SIZE
    num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)
    for sub_block_idx in range(num_sub_blocks):
        offsets = base_offset + sub_block_idx * BLOCK_SIZE_SUB + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < n_elements
        data = tl.load(ptr + offsets, mask=mask)
        # ... 处理 ...
        tl.store(out_ptr + offsets, result, mask=mask)
```

**BLOCK_SIZE 选择指导**:

| 场景 | 推荐 BLOCK_SIZE | 说明 |
|------|----------------|------|
| Vector 算子（小 shape） | 128 ~ 1024 | 普通元素级运算 |
| Vector 算子（大 shape） | 8192 ~ 32768 | 大规模数据 |
| 矩阵乘 (tl.dot) | BLOCK_M/N/K 为 16 的倍数 | FP16/BF16 分形对齐 |
| UB OVERFLOW 时 | 减小 BLOCK_SIZE 或引入 SUB | 控制片上内存 |

### 3. 修改分核策略

```python
# 修改前：使用默认 grid（可能核数过多）
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
my_kernel[grid](...)

# 修改后：固定为物理核数（推荐）
import triton.runtime.driver as driver
properties = driver.active.utils.get_device_properties(torch.npu.current_device())
aicore_num = properties["num_aicore"]
grid = (aicore_num,)
my_kernel[grid](..., NUM_CORE=aicore_num)
```

### 4. 修改流水线/编译优化选项

```python
# 使用 @triton.autotune 配置编译选项
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024, 'multibuffer': True}),
        triton.Config({'BLOCK_SIZE': 2048, 'multibuffer': False}),
        triton.Config({'BLOCK_SIZE': 4096, 'multibuffer': True}),
    ],
    key=['n_elements'],
)
@triton.jit
def my_kernel(...):
    ...
```

### 5. 添加新的输入 tensor

```python
# 修改前
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    ...

# 修改后：添加 bias
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,  # 新增 bias_ptr
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    ...
    # 加载 bias
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    # 加到结果上
    acc = acc + bias[None, :]
    ...
```

### 6. 修改访存优化

```python
# 添加 care_padding=False 提升并行度（当不关心 padding 内容时）
data = tl.load(input + idx, mask=mask, care_padding=False)

# 优化离散访存：先搬后选
rm = tl.arange(0, M)
x_shared = tl.load(x_ptr + rm * stride_x)  # 先加载整块
val = tl.gather(x_shared, idx, 0)            # 再 gather
```

### 7. 修复数据类型导致的 Scalar 退化

```python
# 修改前：int64 比较退化为标量
xbar = tl.where(cols < N, x - mean, 0.0)  # cols 是 int64

# 修改后：转为 fp32 利用 Vector
cols_cmp = cols.to(tl.float32)
xbar = tl.where(cols_cmp < N, x - mean, 0.0)
```

## NPU 修改注意事项

### GPU -> NPU 迁移常见修改

1. **设备名**: `device='cuda'` -> `device='npu'`
2. **导入**: 添加 `import torch_npu`
3. **核数**: 固定 grid 为物理核数
4. **BLOCK_SIZE**: 考虑 NPU 片上内存限制（A2 为 192KB）
5. **数据类型**: 注意 NPU 不支持 uint8/uint16/uint32/uint64/fp64

### UB OVERFLOW 修复

当修改 BLOCK_SIZE 后出现 UB 溢出：

```
ub overflow, requires xxxx bits while 1572864 bits available!
```

解决方案：
1. 减小 BLOCK_SIZE
2. 引入 BLOCK_SIZE_SUB 二次分块
3. 减少同时驻留在 UB 中的 tensor 数量

### coreDim 修复

```
coreDim=xxxx can't be greater than UINT16_MAX
```

解决方案：
1. 设置 `TRITON_ALL_BLOCKS_PARALLEL=1`
2. 增大 BLOCK_SIZE 使 `ceil(N / BLOCK_SIZE) <= 65535`

## 约束

- 默认最小改动，不引入新依赖。
- 未经用户明确要求，不改变目录结构。
- 若输入信息不足，必须先列出假设。
- 严格按照用户明确要求的参数变更进行修改，不得自行添加或删减任何参数。
- 修改后需要确保测试能够通过。
- 注意 NPU 与 GPU 的行为差异，修改后需验证正确性。

## 生成的 claude.local.md 模板（简化）

```
# 修改算子输入模板（简化）

一句话说明你要改什么（允许最简描述）：
- 例如："给 Matmul 新增一个 bias 输入，dtype 支持 FP16/BF16"
- 例如："将 FlashAttention 的 BLOCK_M 从 32 改为 128"
- 例如："修复 Vector CMP 退化为 Scalar 的性能问题"

必要信息（可选补充）：
- 算子名：
- 算子路径：
- 变更动作（新增/删除/修改）：
- 参数名与类型：
- shape 约束（如有）：
- dtype 约束（如有）：
- 性能优化参数变更（如有）：
```

## 参考

- Triton-Ascend 编程指南: `triton-ascend/docs/zh/programming_guide.md`
- Triton-Ascend GPU 迁移指南: `triton-ascend/docs/zh/migration_guide/migrate_from_gpu.md`
- Triton-Ascend 性能优化指南: `triton-ascend/docs/zh/migration_guide/performance_guidelines.md`
- Triton-Ascend 调试指南: `triton-ascend/docs/zh/debug_guide/debugging.md`
