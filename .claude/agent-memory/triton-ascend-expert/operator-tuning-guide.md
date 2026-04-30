---
name: operator-tuning-guide
description: Triton-Ascend 算子调优核心知识，涵盖 msprof profiling、benchmark 方法论、性能瓶颈定位、标量退化避免、编译器选项调优、SOC 版本信息、优化工作流。综合官方文档和 skills 知识库。
type: reference
---

# Triton-Ascend 算子调优指南

来源: Triton-Ascend 官方文档 (2026-04-29 采集)

## 1. Profiling 工具

### msprof op（板上采集）
- 在 NPU 上运行算子并采集真实性能数据
- 输出包含: 算子执行时间、核利用率、搬运/计算/写回各阶段耗时

### msprof op simulator（指令级仿真）
- 无需 NPU 设备即可分析
- 提供: 指令流水线可视化、MTE2/MTE3/Vector/Scalar 各阶段耗时、Bubble 分析
- 适合开发阶段快速迭代

## 2. 性能瓶颈模式

### MTE2 Bound（搬运受限）
- **特征**: MTE2 占比 >> Vector 占比
- **原因**: BLOCK_SIZE 太小，搬运量不足
- **方案**: 增大 BLOCK_SIZE, 开启 multibuffer

### Vector Bound（计算受限）
- **特征**: Vector 占比高
- **方案**: 优化计算逻辑，减少冗余运算

### Scalar Fallback（标量退化）
- **特征**: Scalar 占比异常高
- **原因**: 整数比较（i32/i64）退化为标量
- **方案**: `.to(tl.float32)` 转为浮点比较

### Bubble（流水线空闲）
- **特征**: Bubble 占比高
- **原因**: 搬运和计算串行执行
- **方案**: 开启 multibuffer 实现流水重叠

## 3. 标量退化详解

### 退化触发条件
NPU Vector 单元不支持 i32/i64 比较运算。以下场景会退化:
- `tl.where(int_index < limit, a, b)` — 整数条件
- `tl.where(cols < N, ...)` — cols 是 tl.arange (int32)

### 不退化的场景
- `tl.load(ptr, mask=rmask, ...)` — mask 中的整数比较由 MTE2 引擎处理
- `tl.store(ptr, val, mask=rmask)` — 同上

### LayerNorm 优化案例
```python
# Before: Scalar 退化
xbar = tl.where(cols < N, x - mean, 0.0)  # cols < N → scalar

# After: Vector 比较
cols_f = cols.to(tl.float32)
xbar = tl.where(cols_f < N, x - mean, 0.0)  # float 比较 → vector
```

**效果**: Scalar 时间大幅降低，Vector 利用率显著提升。

## 4. 编译器优化选项

### 通用选项
| 选项 | 默认值 | 说明 |
|------|--------|------|
| `multibuffer` | True | 流水并行（乒乓缓冲），开启后 UB 容量减半 |
| `unit_flag` | None | Cube 搬出优化 |
| `auto_blockify_size` | 1 | TRITON_ALL_BLOCKS_PARALLEL 优化 |

### CV 融合专用选项
| 选项 | 说明 |
|------|------|
| `enable_auto_bind_sub_block` | 自动绑定子块 (CV融合) |
| `enable_hivm_auto_cv_balance` | 自动 CV 负载均衡 |
| `tile_mix_vector_loop` | Vector 循环 Tiling 切分数 |
| `tile_mix_cube_loop` | Cube 循环 Tiling 切分数 |
| `sync_solver` | 自动优化同步点 |
| `limit_auto_multi_buffer_only_for_local_buffer` | CV 算子优化 |
| `set_workspace_multibuffer` | workspace 配置 |

### 在 autotune 中使用
```python
triton.Config(
    {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32},
    multibuffer=True,
    enable_auto_bind_sub_block=True,
    enable_hivm_auto_cv_balance=True,
)
```

## 5. 优化工作流

```
Step 1: 基准测试
  triton.testing.do_bench(lambda: my_kernel(x))

Step 2: 性能采集
  msprof op my_benchmark.py

Step 3: 瓶颈分析
  查看 MTE2 / Vector / Scalar / Bubble 占比

Step 4: 针对性优化
  - MTE2 高 → 增大 BLOCK_SIZE / multibuffer
  - Scalar 高 → 检查整数比较
  - Bubble 高 → 调整 Tiling / multibuffer

Step 5: Autotune 搜索
  配置多个 BLOCK_SIZE 和编译选项，自动选择最优

Step 6: 验证
  重新 benchmark，确认性能提升且精度不变
```

## 6. 快速优化检查清单

1. **grid = 物理核数**: vectorcore_num (Vector) / aicore_num (CV)
2. **BLOCK_SIZE = 2^n**: 使用 `triton.next_power_of_2()`
3. **尾轴对齐**: 32B (VV) / 512B (CV)
4. **multibuffer**: UB 充裕时开启（默认已开）
5. **避免标量退化**: 整数比较前 `.to(tl.float32)`
6. **UB 容量**: 使用 48KB 实用上限（不是 96KB doublebuffer slot）
7. **连续访存**: 优先 `ptr + offsets` 模式
8. **mask 保护**: 所有 load/store 必须有 mask
9. **autotune**: 搜索 BLOCK_SIZE + multibuffer 组合
10. **精度**: 累加器使用 float32，最终转换回目标 dtype
11. **先分类**: 判断算子是 memory/compute/UB bound，针对性优化

## 7. Benchmark 方法论

来源: `skills/triton-op-benchmark/SKILL.md`

### triton.testing.do_bench

```python
import os
# 必须在 import torch_npu 之前设置
os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"  # 抑制 CANN profiler 日志
os.environ["TRITON_BENCH_METHOD"] = "npu"    # NPU 精确计时

# 标准用法
triton.testing.do_bench(
    lambda: my_kernel(x),
    warmup=10,
    rep=100,
)
```

**噪音抑制**: `TRITON_BENCH_METHOD="npu"` 会触发 CANN profiler，每次 do_bench 打印 3-4 行日志。用 `_silent_bench` 包裹:

```python
import sys
from io import StringIO

def _silent_bench(fn, quantiles=None, warmup=10, rep=100):
    """静默执行 do_bench，抑制所有中间输出。"""
    if quantiles is None:
        quantiles = [0.5, 0.2, 0.8]
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    try:
        result = triton.testing.do_bench(fn, quantiles=quantiles, warmup=warmup, rep=rep)
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
    return result
```

**带宽计算**: `do_bench` 返回毫秒，转秒用 `×1e-3`（不是 `×1e-6`）:

```python
def calc_bandwidth_gbs(n_elements, element_bytes, ms):
    return 3 * n_elements * element_bytes / (ms * 1e-3) * 1e-9
```

**关键**: NPU 上必须设 `TRITON_BENCH_METHOD="npu"` 才能获得准确计时。

### Autotune 集成

**标准模式**: 手动提供 Config 列表
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, multibuffer=True),
        triton.Config({'BLOCK_SIZE': 2048}, multibuffer=True),
    ],
    key=['n_elements'],
)
```

**NPU Autotune 限制**:
- **不支持** `num_warps`（GPU 线程束概念，NPU 无对应）
- **不支持** `num_stages`（GPU 流水线概念）
- 支持 NPU 专属选项: `multibuffer`, `enable_auto_bind_sub_block` 等

**高级模式**: 自动搜索
```python
import triton.backends.ascend.runtime
# 空 configs + hints 让编译器自动生成候选
```

**Auto-profiling**:
```python
triton.autotune(..., auto_profile_dir="./autotune_profile")
```

**Autotune BLOCK_SIZE 搜索范围** (实测验证):

| 算子类型 | BLOCK_SIZE 范围 | multibuffer | Config 数量 |
|---------|----------------|-------------|-----------|
| element-wise (add, relu) | 512 ~ 4096 | True + False | 6~8 |
| 归约 (sum, max) | 256 ~ 1024 | True | 3~4 |
| GEMM | 64/128 (fractal 倍数) | True | 3~4 |
| LayerNorm/Softmax | next_power_of_2(N) | True | 1 |

推荐在 element-wise autotune 中同时搜索 multibuffer=True 和 False:
```python
configs=[
    triton.Config({'BLOCK_SIZE': 1024}, multibuffer=True),
    triton.Config({'BLOCK_SIZE': 2048}, multibuffer=True),
    triton.Config({'BLOCK_SIZE': 4096}, multibuffer=True),
    triton.Config({'BLOCK_SIZE': 2048}, multibuffer=False),  # UB 翻倍
    triton.Config({'BLOCK_SIZE': 4096}, multibuffer=False),
    triton.Config({'BLOCK_SIZE': 8192}, multibuffer=False),
]
```

### FLOPs 计算

| 算子类型 | FLOPs 公式 |
|---------|-----------|
| Element-wise | `2 * numel` (read + write) |
| GEMM (M×K @ K×N) | `2 * M * N * K` |
| Reduce Sum | `n_elements - n_outputs` |
| Softmax | `~5 * n_elements` (max + sub + exp + sum + div) |
| Flash Attention | `~4 * B * H * S^2 * D` |

### Benchmark 结构模板

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[128, 256, 512, 1024, 2048, 4096],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'PyTorch'],
        plot_name='performance',
        args={},
    ))
def benchmark(size, provider, device='npu'):
    x = torch.randn(size, device=device, dtype=torch.float32)
    if provider == 'torch':
        return triton.testing.do_bench(lambda: torch.sum(x, dim=-1))
    elif provider == 'triton':
        return triton.testing.do_bench(lambda: reduce_sum(x))
```

## 8. SOC 版本与型号

### 支持 SOC 列表

| SOC | 设备系列 | 架构 |
|-----|---------|------|
| Ascend910A | Atlas 300T | 200x |
| Ascend910B | Atlas A2 | 220x |
| Ascend910B1/B2/B3 | Atlas A2 子型号 | 220x |
| Ascend310 | Atlas 200 | 200x |
| Ascend310P1-P5 | Atlas 200I | 200x |
| Ascend310B1-B4 | Atlas 200I DK | 220x |

### msprof SOC 指定

```bash
# 仿真模式需指定 SOC 版本
msprof op simulator --soc-version=Ascend910B3 python my_bench.py

# 板上采集自动识别 SOC
msprof op python my_bench.py
```

### msprof 详细指标

来源: `skills/triton-op-benchmark/SKILL.md`

| 指标名 | 含义 | 理想占比 |
|--------|------|---------|
| `aiv_vec_ratio` | Vector 计算占比 | 高（>60%） |
| `aiv_scalar_ratio` | Scalar 计算占比 | 低（<5%） |
| `aiv_mte2_time` | GM→UB 搬运时间 | 适中 |
| `aiv_mte2_ratio` | 搬运占比 | 与 Vector 平衡 |

```bash
# 采集特定 kernel
msprof op --kernel-name=my_kernel --output=./prof_out python script.py

# 仿真模式（无需设备）
msprof op simulator --kernel-name=my_kernel --soc-version=Ascend910B3 python script.py
```

## 9. 优化决策树

```
性能不达预期
  ├── msprof 采集数据
  │   ├── MTE2 高?
  │   │   ├── 增大 BLOCK_SIZE
  │   │   ├── 确认 multibuffer=True
  │   │   └── 检查连续访存模式
  │   ├── Scalar 高?
  │   │   └── 查找整数比较，转 .to(tl.float32)
  │   ├── Bubble 高?
  │   │   ├── 开启/调整 multibuffer
  │   │   └── 调整 xblock_sub / rblock
  │   └── Vector 高?
  │       └── 检查计算逻辑是否冗余
  ├── UB overflow?
  │   ├── 减小 BLOCK_SIZE
  │   ├── 使用 XBLOCK_SUB 二级切分
  │   └── 关闭 multibuffer（UB 翻倍但取消流水）
  └── coreDim 超限?
      ├── 增大 BLOCK_SIZE 减少 block 数
      └── 设 TRITON_ALL_BLOCKS_PARALLEL=1
```
