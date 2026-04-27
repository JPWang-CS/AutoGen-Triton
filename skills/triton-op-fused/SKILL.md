---
name: triton-op-fused
description: 生成融合算子（tl.dot 矩阵乘法 + 向量后处理），如 Matmul+Bias、Matmul+ReLU、Matmul+Activation、Flash Attention、Softmax Fusion。基于 triton-ascend 官方文档、programming_guide 和 architecture_design_and_core_features.md。同时使用 Cube Core (tl.dot) 和 Vector Core (元素级/归约运算)。
---

# 融合算子技能 (Cube + Vector Fusion)

基于 `triton-ascend/docs/en/programming_guide.md`、`architecture_design_and_core_features.md` 和 `migration_guide/migrate_from_gpu.md`。

当用户需要生成**融合算子**时使用本技能。此类算子结合 `tl.dot()` (Cube Core) 矩阵乘法和后续的向量运算 (Vector Core), 在一个 kernel 中完成多步计算。

## 融合算子架构

在 Triton-Ascend 上, 融合算子的数据流如下:

```
GM --(tl.load)--> UB/片上内存
                     |
            tl.dot(a, b) --(Cube Core)--> acc (片上累加器)
                     |
            向量后处理 (element-wise, reduce, activation)
            在同一 kernel 中通过 Triton 表达式完成
                     |
            tl.store --(GM)--> 结果写回
```

**关键**: Triton 编译器自动将 `tl.dot()` 映射到 Cube Core, 将元素级运算映射到 Vector Core。用户只需按标准 Triton 语法编写, 编译器负责 CV 分离和同步。

## Ascend 扩展同步 API (高级场景)

来源: `triton-ascend/docs/en/architecture_design_and_core_features.md`

在高级融合场景中, 可以使用 Ascend 扩展的同步原语:

| API | 说明 |
|-----|------|
| `tl.sync_block_wait(sender, receiver, event_id)` | 等待块同步信号 |
| `tl.sync_block_set(sender, receiver, event_id)` | 设置块同步信号 |
| `tl.sync_block_all(mode, event_id)` | 全局块同步 |

**管道名称**: "mte1", "mte2", "mte3", "m", "v", "fix"

这些 API 在需要精细控制 Cube/Vector 流水线同步时使用。标准融合算子中, 编译器自动处理同步。

## Ascend 编译器选项 (CV融合专用)

来源: `triton-ascend/docs/en/architecture_design_and_core_features.md`

可在 autotune config 中设置 CV 融合专用选项:

| 选项 | 说明 |
|------|------|
| `enable_auto_bind_sub_block` | 自动绑定子块 (CV融合) |
| `enable_hivm_auto_cv_balance` | 自动CV负载均衡 |
| `sync_solver` | 同步求解器 |
| `tile_mix_vector_loop` | 向量循环Tiling |
| `tile_mix_cube_loop` | Cube循环Tiling |
| `disable_auto_inject_block_sync` | 禁用自动块同步注入 |
| `enable_nd2nz_on_vector` | 启用 ND 到 NZ 布局转换 |

使用示例:

```python
triton.Config(
    {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
    enable_auto_bind_sub_block=True,
    enable_hivm_auto_cv_balance=True,
    sync_solver=True,
)
```

## Matmul + Bias 模板

```python
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def matmul_bias_kernel(
    A, B, Bias, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_bias,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 累加器初始化
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K 轴循环: 矩阵乘法 (Cube Core)
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            A + (offs_m[:, None] * stride_am + (offs_k[None, :] + k) * stride_ak),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K),
            other=0.0
        )
        b = tl.load(
            B + ((offs_k[:, None] + k) * stride_bk + offs_n[None, :] * stride_bn),
            mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N),
            other=0.0
        )
        acc += tl.dot(a, b)  # Cube Core

    # 向量后处理: 加偏置 (Vector Core)
    bias = tl.load(Bias + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]  # 广播加法

    # 写回
    c_ptr = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptr, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul_bias(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )
    matmul_bias_kernel[grid](
        A, B, bias, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        bias.stride(0),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C
```

## Matmul + ReLU 模板

```python
@triton.jit
def matmul_relu_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 矩阵乘法 (Cube Core)
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            A + (offs_m[:, None] * stride_am + (offs_k[None, :] + k) * stride_ak),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K),
            other=0.0
        )
        b = tl.load(
            B + ((offs_k[:, None] + k) * stride_bk + offs_n[None, :] * stride_bn),
            mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N),
            other=0.0
        )
        acc += tl.dot(a, b)

    # 向量后处理: ReLU (Vector Core)
    acc = tl.where(acc > 0, acc, 0.0)

    # 写回
    c_ptr = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptr, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

## Matmul + Bias + ReLU (三合一融合)

```python
@triton.jit
def matmul_bias_relu_kernel(
    A, B, Bias, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_bias,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(
            A + (offs_m[:, None] * stride_am + (offs_k[None, :] + k) * stride_ak),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K),
            other=0.0
        )
        b = tl.load(
            B + ((offs_k[:, None] + k) * stride_bk + offs_n[None, :] * stride_bn),
            mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N),
            other=0.0
        )
        acc += tl.dot(a, b)

    # +Bias
    bias = tl.load(Bias + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    # ReLU
    acc = tl.where(acc > 0, acc, 0.0)

    c_ptr = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptr, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

## Matmul + GELU 模板

```python
@triton.jit
def matmul_gelu_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(
            A + (offs_m[:, None] * stride_am + (offs_k[None, :] + k) * stride_ak),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K),
            other=0.0
        )
        b = tl.load(
            B + ((offs_k[:, None] + k) * stride_bk + offs_n[None, :] * stride_bn),
            mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N),
            other=0.0
        )
        acc += tl.dot(a, b)

    # GELU 近似
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    inner = sqrt_2_over_pi * (acc + coeff * acc * acc * acc)
    # tanh 近似
    tanh_val = tl.where(inner > 20.0, 1.0,
               tl.where(inner < -20.0, -1.0,
                        (tl.exp(2.0 * inner) - 1.0) / (tl.exp(2.0 * inner) + 1.0)))
    acc = 0.5 * acc * (1.0 + tanh_val)

    c_ptr = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptr, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

## Flash Attention 模板 (简化版)

Flash Attention 是典型的融合算子: 多次 `tl.dot` (QK^T, Score@V) + softmax (向量运算)。

```python
import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver
import math

@triton.jit
def flash_attn_fwd_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z: tl.constexpr, H: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)

    NUM_BLOCKS_M = tl.cdiv(N_CTX, BLOCK_M)
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H

    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        # 恢复 multi-dim 索引
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H

        # QKV 偏移
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

        # Q block offsets
        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)

        # 加载 Q block [BLOCK_M, HEAD_DIM]
        q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM), other=0.0)

        # 初始化累加器和在线 softmax 状态
        acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
        m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)  # running max
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)                # running sum(exp)

        # 沿 N 轴循环
        for start_n in range(0, N_CTX, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)

            # 加载 K block [BLOCK_N, HEAD_DIM]
            k_ptrs = K + qvk_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
            k = tl.load(k_ptrs, mask=(offs_n[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM), other=0.0)

            # QK^T: [BLOCK_M, BLOCK_N]
            qk = tl.dot(q, tl.trans(k)) * SCALE
            qk_mask = (offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX)
            qk = tl.where(qk_mask, qk, float('-inf'))

            # 在线 softmax: 更新 m 和 l
            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)

            # 修正因子
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(m_ij - m_new)
            l_i = l_i * alpha + tl.sum(tl.where(qk_mask, beta, 0.0), axis=1)

            # 修正累加器
            acc = acc * alpha[:, None]

            # 加载 V block [BLOCK_N, HEAD_DIM]
            v_ptrs = V + qvk_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
            v = tl.load(v_ptrs, mask=(offs_n[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM), other=0.0)

            # Score * V
            p = tl.where(qk_mask, tl.exp(qk - m_new[:, None]), 0.0)
            acc += tl.dot(p.to(v.dtype), v)

            m_i = m_new

        # 最终输出: acc / l_i
        acc = acc / l_i[:, None]

        # 写回
        out_ptrs = Out + qvk_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
        tl.store(out_ptrs, acc, mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM))


def flash_attn_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float = None):
    # q, k, v: [Z, H, N_CTX, HEAD_DIM]
    Z, H, N_CTX, HEAD_DIM = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(HEAD_DIM)

    o = torch.empty_like(q)

    device = torch_npu.npu.current_device()
    properties = driver.active.utils.get_device_properties(device)
    aicore_num = properties["num_aicore"]

    BLOCK_M = 64
    BLOCK_N = 64

    grid = (aicore_num,)
    flash_attn_fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Z=Z, H=H, N_CTX=N_CTX, HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        SCALE=scale,
    )
    return o
```

## Autotune 融合算子模板

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
                      enable_auto_bind_sub_block=True,
                      enable_hivm_auto_cv_balance=True),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32},
                      enable_auto_bind_sub_block=True,
                      enable_hivm_auto_cv_balance=True),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
                      enable_auto_bind_sub_block=True,
                      enable_hivm_auto_cv_balance=True),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bias_relu_autotuned(
    A, B, Bias, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_bias,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(
            A + (offs_m[:, None] * stride_am + (offs_k[None, :] + k) * stride_ak),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K),
            other=0.0
        )
        b = tl.load(
            B + ((offs_k[:, None] + k) * stride_bk + offs_n[None, :] * stride_bn),
            mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N),
            other=0.0
        )
        acc += tl.dot(a, b)

    bias = tl.load(Bias + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    acc = tl.where(acc > 0, acc, 0.0)

    c_ptr = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptr, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

## 融合算子 + 多核持久化模式

对于需要充分利用所有 AI Core 的融合算子:

```python
def get_aicore_num():
    device = torch_npu.npu.current_device()
    properties = driver.active.utils.get_device_properties(device)
    return properties["num_aicore"]

def launch_fused_kernel(kernel, args, M, N, BLOCK_M, BLOCK_N, **kwargs):
    """
    融合算子通用启动函数:
    - 小矩阵: 标准 2D grid
    - 大矩阵: 1D 持久化 grid
    """
    NUM_BLOCKS_M = triton.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = triton.cdiv(N, BLOCK_N)
    total_blocks = NUM_BLOCKS_M * NUM_BLOCKS_N
    aicore_num = get_aicore_num()

    if total_blocks <= aicore_num:
        # 小矩阵: 标准 2D grid
        grid = (NUM_BLOCKS_M, NUM_BLOCKS_N)
    else:
        # 大矩阵: 绑定物理核心数
        grid = (aicore_num,)

    kernel[grid](*args, **kwargs)
```

## 性能优化指南

### CV 融合专用优化

| 策略 | 说明 | 配置方法 |
|------|------|---------|
| 自动CV分离 | 编译器自动将tl.dot和向量操作分配到不同核心 | autotune config: `enable_auto_bind_sub_block=True` |
| CV负载均衡 | 自动平衡Cube和Vector的工作量 | autotune config: `enable_hivm_auto_cv_balance=True` |
| 同步求解器 | 自动优化同步点 | autotune config: `sync_solver=True` |
| 乒乓流水线 | 默认启用, 数据搬运与计算并行 | autotune config: `multibuffer=True` (默认) |

### Block Size 选择

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| BLOCK_M | 64, 128 | M方向分块 |
| BLOCK_N | 64, 128 | N方向分块 |
| BLOCK_K | 32, 64 | K方向累加步长 |
| HEAD_DIM (FA) | 64, 128 | 注意力头维度 |

### 精度控制

- `tl.dot()` 的累加建议使用 `tl.float32`
- softmax 中的 exp 运算建议在 float32 下进行
- 最终结果在 store 时转换为目标 dtype

### UB 溢出防范

融合算子因为同时存在矩阵乘法和向量计算的中间结果, UB 使用量更大:

1. 中间张量总量: BLOCK_M * BLOCK_N * sizeof(float32) + 其他中间结果
2. Atlas A2 UB = 192 KB
3. 如果溢出: 减小 BLOCK_M/BLOCK_N 或拆分矩阵乘法和后处理

## 约束

- **tl.dot 用于矩阵乘法**: 编译器自动映射到 Cube Core
- **grid 使用 aicore_num**: CV融合算子应使用 AI Core 数 (不是 vectorcore_num)
- **累加用 float32**: `tl.dot()` 结果用 float32 累加以保证精度
- **softmax 数值稳定**: 先减最大值再 exp, 避免溢出
- **UB 容量**: 融合算子的中间张量更多, 需特别注意 UB 限制
- **编译器自动CV分离**: 标准 Triton 语法足够, 编译器负责 Cube/Vector 分配和同步

## 参考

- 编程指南 (含matmul、vector_add、autotune): `triton-ascend/docs/en/programming_guide.md`
- 架构设计 (含CV编译选项、同步API、Ascend扩展算子): `triton-ascend/docs/en/architecture_design_and_core_features.md`
- 迁移指南 (含多核并行策略): `triton-ascend/docs/en/migration_guide/migrate_from_gpu.md`
- 性能分析: `triton-ascend/docs/en/debug_guide/profiling.md`
- 在线文档: https://triton-ascend.readthedocs.io/zh-cn/latest/index.html
