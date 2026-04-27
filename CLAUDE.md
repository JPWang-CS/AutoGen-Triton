# Triton-Ascend AutoGen 仓库 (NPU专用)

本仓库提供Triton-Ascend NPU算子自动生成的skill提示词和相关模板。所有算子基于Triton编程模型运行在华为昇腾NPU上。

## 权威参考源

- **Triton-Ascend 源码仓库**: `M:\Desktop\tmp\AgentTest\triton-ascend` -- 编译器后端实现与测试用例
- **官方在线文档**: `https://triton-ascend.readthedocs.io/zh-cn/latest/index.html` -- API参考与迁移指南
- **Triton-Ascend 示例**: `triton-ascend/python/tutorials/` -- 可运行的参考代码（vector-add, softmax, matmul, layer-norm, fused-attention 等）
- **Triton-Ascend Skills**: `M:\Desktop\tmp\AgentTest\Triton\skills` -- 算子生成skill提示词
- **标准 Triton (CUDA)**: Triton-Ascend 的编程模型与标准 Triton 高度一致，但存在昇腾平台特定的差异和限制

## 编程模型概述

Triton-Ascend 使用标准 Triton 的编程范式 (`@triton.jit`, `tl.load`, `tl.store`, `tl.program_id` 等)，开发者编写基于 block/tile 的 kernel，编译器自动完成内存分配、数据搬运和流水并行。与标准 Triton (CUDA) 的核心区别在于底层硬件架构差异导致的多核并行策略、内存对齐要求和编译优化选项。

### Kernel 基本结构

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

# 启动 kernel
def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

## NPU 多核并行策略

NPU与GPU在多核并行上存在核心差异：

| 维度 | GPU (NVIDIA) | 昇腾 (Ascend) |
|------|-------------|--------------|
| grid 本质 | 逻辑任务维度（和物理核解耦） | 物理核组映射（绑定 AI Core 拓扑） |
| 核数限制 | grid 维度/大小无硬限制 | grid 大小 <= AI Core 总数 |
| 核数获取 | N/A | 通过 driver API 获取 |

### 获取物理核数

```python
import torch_npu
import triton.runtime.driver as driver

device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
vectorcore_num = properties["num_vectorcore"]
aicore_num = properties["num_aicore"]
```

### 推荐的分核模式

```python
NUM_CORE = vectorcore_num  # 纯Vector算子使用vectorcore_num
grid = (NUM_CORE,)
kernel[grid](...)  # 核内通过循环处理多个分块

@triton.jit
def kernel(..., NUM_CORE: tl.constexpr):
    pid = tl.program_id(0)
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        # 跨步分配任务，确保均匀分布到各个核
        ...
```

## 核心API参考

### 编程模型

| API | 说明 |
|-----|------|
| `tl.program_id(axis)` | 获取当前 program 的 ID |
| `tl.num_programs(axis)` | 获取总 program 数量 |
| `tl.arange(start, end)` | 生成连续整数序列 |
| `tl.cdiv(a, b)` | 向上取整除法 |

### 内存操作

| API | 说明 |
|-----|------|
| `tl.load(ptr, mask=None, other=0.)` | 从全局内存加载数据到片上 |
| `tl.store(ptr, value, mask=None)` | 将数据写回全局内存 |

### 数据类型

| Triton 类型 | 说明 |
|------------|------|
| `tl.float32` / `tl.float16` / `tl.bfloat16` | 浮点类型 |
| `tl.int1` / `tl.int8` / `tl.int32` / `tl.int64` | 整数类型 |
| `tl.constexpr` | 编译时常量参数 |

### 数学与归约操作

| API | 说明 |
|-----|------|
| `tl.dot(a, b)` | 块级矩阵乘法 |
| `tl.sum(x, axis)` | 沿轴求和 |
| `tl.max(x, axis)` | 沿轴最大值 |
| `tl.min(x, axis)` | 沿轴最小值 |
| `tl.where(mask, a, b)` | 条件选择 |
| `tl.exp / tl.log / tl.sqrt / tl.erf` | 数学函数 |
| `tl.trans()` | 转置 |

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

## 昇腾扩展 API

Triton-Ascend 在标准 Triton 基础上提供了昇腾特有的扩展操作：

| API | 说明 |
|-----|------|
| `tl.dot_scaled(a, b, ...)` | 带缩放的矩阵乘法 |
| `tl.inline_asm_elementwise(...)` | 内联汇编操作 |
| `tl.parallel(...)` | 并行循环提示 |
| `tl.sync_block()` | 块内同步 |
| `tl.gather(src, idx, axis)` | 按索引收集元素 |
| `tl.scatter_ub_to_out(...)` | 从UB到GM的高效数据分散 |

### 编译优化选项

在 autotune 的 `triton.Config` 中可配置的昇腾特有编译选项：

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `multibuffer` | 开启流水并行数据搬运 | True |
| `unit_flag` | Cube搬出优化项 | None |
| `limit_auto_multi_buffer_only_for_local_buffer` | CV算子优化项 | None |
| `tile_mix_vector_loop` | CV算子Vector切分份数 | None |
| `tile_mix_cube_loop` | CV算子Cube切分份数 | None |
| `auto_blockify_size` | TRITON_ALL_BLOCKS_PARALLEL优化 | 1 |

## NPU 硬件约束

### 内存层级

```
GM (全局内存, HBM)
  <-> 片上内存 (UB/Cache)
     <-> 计算单元 (Vector Core / Cube Core)
```

### 片上内存限制

| 资源 | 容量 | 对齐要求 |
|------|------|---------|
| **片上内存 (Atlas A2)** | 192 KB | - |
| **UB (Vector Core)** | 受片上内存限制 | 尾轴 32 字节对齐 |
| **CV 算子** | 受片上内存限制 | 尾轴 512 字节对齐 |

### 关键约束

- BLOCK_SIZE 等参数建议为 **2 的幂次**
- 尾轴数据对齐：VV 类算子需 32B 对齐，CV 类算子需 512B 对齐
- 默认开启 doublebuffer 后片上可用容量减半
- grid 核数不超过 AI Core 物理核数时性能最优
- 当前版本核数需 <= 65535

## 与标准 Triton (CUDA) 的关键差异

### 编程模型差异

| 方面 | Triton (CUDA) | Triton-Ascend (NPU) |
|------|-------------|-------------------|
| 启动设备 | `torch.cuda` | `torch_npu` |
| 设备获取 | `triton.runtime.driver.active.get_active_torch_device()` | 同左 (自动适配) |
| 多核策略 | 大量逻辑线程, 硬件自动调度 | 物理核绑定, 需手动控制核数 |
| 数据切分 | 依赖 block 级自动切分 | 推荐 XBLOCK + XBLOCK_SUB 二级切分 |
| 内存对齐 | warp 对齐 (4/8 字节) | UB 32B / CV 512B 尾轴对齐 |

### 推荐的 NPU kernel 模式

```python
@triton.jit
def npu_kernel(in_ptr, out_ptr, xnumel,
               XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)
        xmask = x_index < xnumel
        x = tl.load(in_ptr + x_index, xmask)
        ret = x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))  # GELU
        tl.store(out_ptr + x_index, ret, xmask)

# 调用: 核数=物理核数, XBLOCK=核间切分, XBLOCK_SUB=核内切分
npu_kernel[vectorcore_num, 1, 1](x, out, x.numel(), xblock, xblock_sub)
```

## 仓库结构

```
Triton/
├── skills/                          # Skill提示词目录
│   ├── triton-op-pipeline/          # 算子生成入口 (分发器)
│   ├── triton-op-vector/            # 纯向量算子 (Vector Core Only)
│   ├── triton-op-cube/              # 纯矩阵算子 (Cube Core Only)
│   ├── triton-op-fused/             # 融合算子 (Cube + Vector)
│   ├── triton-op-edit/              # 修改算子技能
│   ├── triton-op-rename/            # 重命名算子技能
│   ├── triton-op-test/              # 生成测试技能
│   ├── triton-op-benchmark/         # 生成benchmark技能
│   └── triton-op-hardware-constraints/  # NPU硬件约束技能
├── templates/op_frame/              # 模板目录
│   ├── empty_op_template/           # 空白算子模板
│   └── gemm_reference/              # GEMM参考实现
├── projects/                        # 生成的算子项目目录
├── .claude/                         # Claude Agent 配置
│   ├── agents/                      # Agent 定义 (coder, expert, reviewer, pm)
│   └── agent-memory/                # Agent 持久化记忆
├── CLAUDE.md                        # 本文件
└── README.md                        # 仓库说明
```

## 可用技能

### triton-op-pipeline (入口分发器)
新增Triton-Ascend算子时的入口技能。根据算子类型自动分发：
- **无矩阵乘法** -> `triton-op-vector` (纯向量计算)
- **仅矩阵乘法** -> `triton-op-cube` (纯 GEMM)
- **矩阵乘法+后处理** -> `triton-op-fused` (融合算子)

### triton-op-vector (纯向量算子)
生成仅使用 Vector Core 的算子: add, mul, relu, softmax, layernorm 等。
- 核心模式: `tl.load` + 标量/向量运算 + `tl.store`
- 分核策略: 使用 `vectorcore_num` 核数

### triton-op-cube (纯矩阵乘法)
生成使用 Cube Core 的 GEMM 算子: C = A @ B, 含 GroupGEMM 和持久化变体。
- 核心模式: `tl.dot` 进行块级矩阵乘
- 分核策略: 使用 `aicore_num` 核数

### triton-op-fused (融合算子)
生成 Cube + Vector 融合算子: Matmul+Bias, Flash Attention 等。
- 核心模式: `tl.dot` + 向量后处理 (softmax, activation 等)
- 编译选项: `multibuffer`, `tile_mix_vector_loop` 等

### triton-op-edit (修改算子)
修改现有算子参数或实现逻辑。

### triton-op-rename (重命名算子)
重命名算子及其相关引用。

### triton-op-test (测试生成)
为算子生成测试用例，包含正确性验证和精度对比。

### triton-op-benchmark (性能测试)
为算子生成性能基准测试，使用 `triton.testing.do_bench` 进行测量。

### triton-op-hardware-constraints (硬件约束)
检查算子是否满足 NPU 硬件约束条件（内存对齐、片上内存容量等）。

## Agent 生态

| Agent | 职责 |
|-------|------|
| **triton-ascend-coder** | 代码生成与调试 |
| **triton-ascend-expert** | 技术研究与方案分析 |
| **triton-ascend-reviewer** | 代码审查 (不写代码) |
| **triton-ascend-pm** | 工作流编排与任务调度 |

## 开发流程

1. 使用 `triton-op-pipeline` 技能生成算子框架
2. 确定算子类型（Vector / Cube / Fused）
3. 检查硬件约束 (`triton-op-hardware-constraints`)
4. 根据需求修改算子实现 (`triton-op-edit`)
5. 使用 `triton-op-test` 生成测试用例
6. 使用 `triton-op-benchmark` 进行性能测试

## 数据创建与验证

```python
import torch
import torch_npu

torch.manual_seed(0)

# 在 NPU 上创建张量
DEVICE = triton.runtime.driver.active.get_active_torch_device()
a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)

# 调用 kernel
output = my_kernel(a, b)

# 精度对比
ref = a @ b
torch.testing.assert_close(output, ref, rtol=1e-2, atol=1e-2)
```

## 参考资料

- **Triton-Ascend 源码仓库**: `M:\Desktop\tmp\AgentTest\triton-ascend` (权威)
- **Triton-Ascend 教程**: `triton-ascend/python/tutorials/`
  - 01-vector-add.py, 02-fused-softmax.py, 03-matrix-multiplication.py
  - 04-low-memory-dropout.py, 05-layer-norm.py, 06-fused-attention.py
  - 07-extern-functions.py, 08-grouped-gemm.py, 09-persistent-matmul.py
- **算子开发指南**: `triton-ascend/docs/zh/programming_guide.md`
- **GPU迁移指南**: `triton-ascend/docs/zh/migration_guide/migrate_from_gpu.md`
- **架构差异说明**: `triton-ascend/docs/zh/migration_guide/architecture_difference.md`
- **调试指南**: `triton-ascend/docs/zh/debug_guide/debugging.md`
- **性能调优指南**: `triton-ascend/docs/zh/debug_guide/profiling.md`
- **API参考文档**: `triton-ascend/docs/zh/triton_api/` (原子操作、数学操作、线性代数等)
- **在线文档**: https://triton-ascend.readthedocs.io/zh-cn/latest/index.html
- [Triton-Ascend GitCode](https://gitcode.com/Ascend/triton-ascend)
