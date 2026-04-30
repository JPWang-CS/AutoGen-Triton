---
name: vector-add-optimization-case
description: vector_add 算子优化实战案例，包含 benchmark 数据、memory-bound 分析、UB 计算修正、噪音抑制技巧、各优化模式效果对比。2026-04-30 实测。
type: project
---

# Vector Add 优化实战案例

日期: 2026-04-30 | 设备: Atlas A2 (910B) | 48 Vector Cores

## 实测数据 (fp32, warmup=10, rep=100)

| Size | Naive(ms) | Optimized(ms) | Autotune(ms) | PyTorch(ms) | Opt/Naive | Atune/Torch |
|------|-----------|---------------|-------------|-------------|-----------|-------------|
| 4K   | 1.487     | 2.688         | 2.527       | 1.379       | 0.55x     | 0.55x       |
| 16K  | 1.498     | 2.673         | 2.551       | 1.854       | 0.56x     | 0.73x       |
| 64K  | 1.962     | 3.139         | 2.966       | 3.008       | 0.63x     | 1.01x       |
| 256K | 4.074     | 3.495         | 3.519       | 3.368       | 1.17x     | 0.96x       |
| 1M   | 10.733    | 5.014         | 5.000       | 4.732       | **2.14x** | 0.95x       |
| 4M   | 38.675    | 11.035        | 11.017      | 10.500      | **3.51x** | 0.95x       |

## 关键发现

### 1. 物理核绑定在大数据量时效果显著

Naive 模式 grid = cdiv(n, BLOCK_SIZE)，远超 48 个物理核，产生调度开销。
Optimized 模式 grid = 48，核内 constexpr 循环处理。

- 小数据 (≤16K): kernel launch 开销占比大，Naive 反而更快（调度开销可忽略）
- 大数据 (≥256K): 调度开销凸显，Optimized 快 1.17x~3.51x
- 4M 时: 38.675ms → 11.035ms，提升 **3.5x**

### 2. vector_add 是纯内存 bound 操作

计算: 2 load + 1 store + 1 add，计算量极小。

**结论**: kernel 内部结构优化（constexpr loop vs tl.range、XBLOCK/XBLOCK_SUB）对纯内存 bound 操作收益有限。优化效果主要来自物理核绑定减少调度开销。

**各类算子的优化策略**:

| 算子类型 | 瓶颈 | 有效的优化手段 |
|---------|------|-------------|
| 内存 bound (add, copy) | GM 带宽 | 物理核绑定、大 BLOCK |
| 计算 bound (softmax, layernorm) | Vector 计算 | 标量退化避免、constexpr 循环、multibuffer |
| UB bound (reduce, GEMM) | UB 容量 | XBLOCK/XBLOCK_SUB 二级切分、rblock 规划 |
| CV 融合 (flash-attn) | Cube/Vector 协调 | enable_auto_bind_sub_block、CV 负载均衡 |

### 3. UB 实用上限是 48KB，不是 96KB

- 理论 doublebuffer slot: 96KB
- 实际编译器需要中间变量空间
- 安全做法: `UB_PRACTICAL_LIMIT = 48KB`
- element-wise 3 I/O 张量: 48KB / 12B = 4096 元素 (而非 8192)

### 4. Autotune Config 范围

- element-wise 推荐: 512 ~ 4096 (skill 原始建议 128~4096)
- 搜索 multibuffer True/False 组合
- **不应**超过 UB 实用上限

### 5. Benchmark 噪音问题

`TRITON_BENCH_METHOD="npu"` 触发 CANN profiler，每次 do_bench 打印 3-4 行日志。
- 解决: `os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"` + `_silent_bench()` 重定向 stdout/stderr
- Autotune 警告 `DO NOT tune args ['num_warps', ...]` 来自 Triton-Ascend 内部，无法脚本层抑制

### 6. 带宽计算修复

```python
# 错误 (高 1000 倍)
bw = bytes / (ms * 1e-6) * 1e-9
# 正确
bw = bytes / (ms * 1e-3) * 1e-9  # ms → 秒用 1e-3
```

4M 实测带宽: Optimized ≈ 4.6 GB/s, PyTorch ≈ 4.8 GB/s (修正后)。

## 最终优化方案

三个模式:
1. **naive**: GPU 风格 baseline，小数据时快
2. **optimized**: XBLOCK/XBLOCK_SUB constexpr 循环 + UB 安全 block (48KB 上限)，大数据快 3.5x
3. **autotune**: @triton.autotune 7 个 Config 自动搜索，达到 PyTorch 95% 性能
