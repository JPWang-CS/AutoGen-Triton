---
name: triton-op-benchmark
description: 为Triton-Ascend算子生成性能基准测试文件。包含噪音抑制、静默采集模式、带宽计算修正。用户提出"生成benchmark/性能测试/性能对比"时使用本技能。
---

# Triton-Ascend 算子性能基准测试生成

当用户提出"生成Triton算子benchmark/性能测试/性能对比"时使用本技能。

## 工作流

1. 确认目标算子路径与算子名称。
2. 读取算子实现文件，理解计算模式和参数。
3. 生成 benchmark 文件 `benchmark_{op_name}.py`。
4. 包含以下内容：
   - 性能测试（延迟、吞吐量）
   - 与 PyTorch / torch_npu 参考实现的性能对比
   - 不同配置的性能测试（BLOCK_SIZE、multibuffer 等）
   - 结果输出和数据汇总

## 噪音抑制与静默采集 (重要)

`TRITON_BENCH_METHOD="npu"` 会触发 CANN profiler，每次 `do_bench` 打印 3-4 行日志。在 sweep 场景下产生大量噪音，淹没结果表格。

### 必须设置的环境变量

```python
import os
# 必须在 import torch_npu 之前设置
os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
os.environ["TRITON_BENCH_METHOD"] = "npu"
```

### 静默采集模式: 先执行后打印

所有 `do_bench` 调用用 `_silent_bench` 包裹，结果暂存变量，最后统一打印:

```python
import sys
from io import StringIO

def _silent_bench(fn, quantiles=None, warmup=10, rep=100):
    """静默执行 do_bench，抑制 CANN profiler 日志。"""
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

# 使用: 先采集所有数据
rows = []
for size in sizes:
    # ... 准备数据 ...
    for mode in modes:
        ms, _, _ = _silent_bench(lambda: my_kernel(...))
        ms_map[mode] = ms
    rows.append(...)

# 最后统一打印
for row in rows:
    print(f"  {row['size']:>8} | {row['naive']:>7.3f}  ...")
```

### 带宽计算

```python
def calc_bandwidth_gbs(n_elements, element_bytes, ms):
    """正确: do_bench 返回 ms, 转 秒用 ×1e-3"""
    return 3 * n_elements * element_bytes / (ms * 1e-3) * 1e-9

# 常见错误: ms * 1e-6 (偏高 1000 倍!)
```

## Benchmark 文件结构

```python
import argparse
import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver


def ref_program(*args, **kwargs):
    """PyTorch 参考实现"""
    pass


def calculate_flops(*args, **kwargs):
    """计算 FLOPs"""
    pass


def get_device_properties():
    """获取 NPU 设备属性"""
    device = torch_npu.npu.current_device()
    properties = driver.active.utils.get_device_properties(device)
    return {
        "aicore_num": properties["num_aicore"],
        "vectorcore_num": properties["num_vectorcore"],
    }


def benchmark_config(M, N, K, config, warmup=10, rep=100):
    """单配置 benchmark"""
    # 准备输入数据
    a = torch.randn((M, K), device='npu', dtype=torch.float16)
    b = torch.randn((K, N), device='npu', dtype=torch.float16)
    c = torch.empty((M, N), device='npu', dtype=torch.float16)

    # 构造 grid
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    # 性能测量
    # 方式1：使用 triton.testing.do_bench
    # 需要设置 TRITON_BENCH_METHOD="npu"
    from triton.testing import do_bench

    def kernel_call():
        matmul_kernel[grid](a, b, c, M, N, K,
                            a.stride(0), a.stride(1),
                            b.stride(0), b.stride(1),
                            c.stride(0), c.stride(1),
                            BLOCK_M=config['BLOCK_M'],
                            BLOCK_N=config['BLOCK_N'],
                            BLOCK_K=config['BLOCK_K'])

    tl_latency = do_bench(kernel_call, warmup=warmup, rep=rep)

    # 参考实现性能
    def ref_call():
        torch.matmul(a, b, out=c)

    ref_latency = do_bench(ref_call, warmup=warmup, rep=rep)

    return tl_latency, ref_latency


def main():
    parser = argparse.ArgumentParser(description="YourOpName Benchmark")
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--autotune", action="store_true")
    args = parser.parse_args()

    # 默认配置 (NPU 推荐)
    default_config = {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 32,
    }

    if args.autotune:
        best_config = run_autotune(args.M, args.N, args.K)
        config = best_config
    else:
        config = default_config

    tl_latency, ref_latency = benchmark_config(
        args.M, args.N, args.K, config,
        warmup=args.warmup, rep=args.rep
    )

    flops = calculate_flops(args.M, args.N, args.K)

    print(f"Configuration: {config}")
    print(f"Triton Latency: {tl_latency:.3f} ms")
    print(f"Triton TFlops: {flops / tl_latency * 1e-9:.2f}")
    print(f"PyTorch Latency: {ref_latency:.3f} ms")
    print(f"PyTorch TFlops: {flops / ref_latency * 1e-9:.2f}")
    print(f"Speedup: {ref_latency / tl_latency:.2f}x")


def run_autotune(M, N, K):
    """运行自动调优"""
    configs = get_tune_configs()

    best_latency = float('inf')
    best_config = None

    for config in configs:
        try:
            latency, _ = benchmark_config(M, N, K, config)
            if latency < best_latency:
                best_latency = latency
                best_config = config
        except Exception as e:
            print(f"Config {config} failed: {e}")
            continue

    return best_config


def get_tune_configs():
    """获取 NPU 调优配置空间"""
    import itertools

    block_M = [64, 128]
    block_N = [64, 128]
    block_K = [32, 64]

    configs = []
    for bm, bn, bk in itertools.product(block_M, block_N, block_K):
        configs.append({
            "BLOCK_M": bm,
            "BLOCK_N": bn,
            "BLOCK_K": bk,
        })
    return configs


if __name__ == "__main__":
    main()
```

## 使用 @triton.autotune 进行自动调优

### 社区 autotune 标准用法

Triton-Ascend 完全兼容社区 `@triton.autotune` 使用方法：

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1 * 128, 'multibuffer': True}),
        triton.Config({'BLOCK_SIZE': 12 * 1024, 'multibuffer': True}),
        triton.Config({'BLOCK_SIZE': 12 * 1024, 'multibuffer': False}),
        triton.Config({'BLOCK_SIZE': 8 * 1024, 'multibuffer': True}),
    ],
    key=["numel"],  # 当 numel 变化时触发 autotune
)
@triton.jit
def my_kernel(out_ptr, in_ptr0, in_ptr1, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    msk = idx < numel
    tmp0 = tl.load(in_ptr0 + idx, mask=msk, other=0.0)
    tmp1 = tl.load(in_ptr1 + idx, mask=msk, other=0.0)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr + idx, tmp2, mask=msk)
```

**说明**:
- 当前 Triton-Ascend autotune 支持 block_size、multibuffer 等参数
- 由于硬件架构差异，**不支持** `num_warps`、`num_stages` 参数
- 设置 `TRITON_PRINT_AUTOTUNING=1` 可打印最优配置信息

### 进阶 autotune 用法

用户无需提供切分轴/tiling 轴等信息，autotune 自动解析并生成候选配置：

```python
import triton.backends.ascend.runtime  # 进阶模式需要此导入

@triton.autotune(
    configs=[],  # 空列表触发自动生成
    key=["n_elements"],  # 或使用 Dict 类型: key={"x": "n_elements"}
    hints={
        "split_params": {"x": "BLOCK_SIZE"},     # 切分轴
        "tiling_params": {},                       # 分块轴
        "low_dim_axes": ["x"],                     # 低维轴
        "reduction_axes": [],                      # 规约轴
    }
)
@triton.jit  # 注意：autotune 必须直接装饰在 jit 之上，中间不能有其他装饰器
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

**进阶 autotune 注意事项**:
1. 进阶模式需 `import triton.backends.ascend.runtime`
2. 若 `configs=[]`，autotune 必须直接装饰在 `@triton.jit` 之上
3. 若 `configs` 不为空且 `hints.auto_gen_config=True`，则合并用户配置和自动配置
4. 进阶用法仅针对 Vector 类算子，不支持 Cube 类算子
5. 设置 `TRITON_BENCH_METHOD="npu"` 使用 `torch_npu.profiler.profile` 方式采集时间（更准确但更慢）

### Profiling 结果自动生成

```python
@triton.autotune(
    auto_profile_dir="./profile_result",  # 自动生成最优配置的 profiling 结果
    ...
)
```

## 使用 msprof 进行性能分析

### 上板 Profiling

```bash
msprof op --kernel-name=target_kernel_name --output=$HOME/projects/output python3 test_op.py
```

### 算子仿真流水图

```bash
# 设置 simulator 路径
export LD_LIBRARY_PATH=$HOME/CANN/Install_CANN/Ascend/ascend_toolkit/latest/tools/simulator/Ascend910B3/lib:$LD_LIBRARY_PATH

# 执行仿真
msprof op simulator --kernel-name=_layer_norm_fwd_fused --soc-version=Ascend910B3 python3 test_op.py
```

### SOC 版本对应表

| Ascend 910 系列 | Ascend 310/310P 系列 | Ascend 310B 系列 |
|:---:|:---:|:---:|
| Ascend910A | Ascend310 | Ascend310B1 |
| Ascend910B | Ascend310P1 | Ascend310B2 |
| Ascend910B1 | Ascend310P2 | Ascend310B3 |
| Ascend910B2 | Ascend310P3 | Ascend310B4 |
| Ascend910B3 | Ascend310P5 | - |

### 性能分析方法

1. **查看 op_summary CSV** - 分析各流水线利用率
2. **查看仿真流水图** - 用 Chrome (`chrome://tracing`) 或 MindStudio Insight 打开 trace.json
3. **代码热点分析** - visualize_data.bin + MindStudio Insight

### 关键性能指标

| 指标 | 含义 | 理想值 |
|------|------|--------|
| `aiv_vec_ratio` | Vector 流水利用率 | 接近 100% |
| `aiv_scalar_ratio` | Scalar 流水利用率 | 低 |
| `aiv_mte2_time` | 数据加载时间 | 与理论值接近 |
| `aiv_mte2_ratio` | MTE2 流水利用率 | 高 |

## 性能指标

### 1. 延迟 (Latency)

- 单次 kernel 执行时间（毫秒）
- 使用 `triton.testing.do_bench` 测量

### 2. 吞吐量 (TFlops)

- 每秒浮点运算次数（万亿次）
- 计算公式：`FLOPs / latency * 1e-9`

### 3. 加速比 (Speedup)

- 相对于参考实现的性能提升
- 计算公式：`ref_latency / tl_latency`

## FLOPs 计算

### GEMM

```python
flops = 2 * M * N * K  # 乘法和加法各 K 次
```

### Flash Attention

```python
# QK^T
flops = 2 * batch * heads * seq_q * seq_kv * dim
# AV
flops += 2 * batch * heads * seq_q * dim * seq_kv
```

### Element-wise

```python
# 单次运算
flops = numel
# 复合运算（如 GELU）
flops = numel * num_ops
```

## NPU 性能优化方向

### 1. 指令并行优化

- 使用 `care_padding=False` 减少 load 时的同步开销
- 使用 for 循环增加 Tiling 提升并行度
- `multibuffer=True`（默认）开启存算并行

```python
# 优化：不关心 padding 内容时去掉默认填充，增加并行度
data = tl.load(input + idx, mask=mask, care_padding=False)
```

### 2. Tiling 优化

- BLOCK_SIZE 应在不超过片上内存时尽可能大
- 使用 `@triton.autotune` 自动搜索最优 BLOCK_SIZE
- 引入 BLOCK_SIZE_SUB 处理大块数据

### 3. 数据类型优化

- 避免使用 int64 进行加法（退化为标量）
- 避免 int32/int64 进行比较（退化为标量）
- 转为 fp32 利用 Vector 操作

### 4. 合并 Grid 分核

- 设置 `TRITON_ALL_BLOCKS_PARALLEL=1` 减少调度开销
- 或手动固定核数为物理核数

## 输出要求

- 清晰的性能数据表格
- 与参考实现的对比
- 不同配置的性能差异
- 最优配置推荐

## 约束

- 确保正确性验证通过后再进行性能测试
- 提供充足的 warmup 轮次
- 多次测量取平均值
- 注意 NPU 与 GPU 的 benchmark 差异

## 参考

- Triton-Ascend autotune 示例: `triton-ascend/docs/zh/examples/06_autotune_example.md`
- Triton-Ascend Profiling 指南: `triton-ascend/docs/zh/debug_guide/profiling.md`
- Triton-Ascend 性能优化: `triton-ascend/docs/zh/migration_guide/performance_guidelines.md`
- Triton-Ascend 环境变量: `triton-ascend/docs/zh/environment_variable_reference.md`
