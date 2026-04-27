---
name: triton-op-cube
description: 生成纯矩阵乘法算子 (GEMM、Batch GEMM、GEMV)，使用 tl.dot 映射到 Ascend NPU Cube Core。基于 triton-ascend 官方文档、programming_guide 中 matmul_kernel 示例和 autotune 配置。支持自动调优、多核并行、转置、UB溢出处理等 Ascend NPU 优化策略。
---

# 纯矩阵乘法算子技能 (Cube Core via tl.dot)

基于 `triton-ascend/docs/en/programming_guide.md` 中的 matmul_kernel 示例和 `architecture_design_and_core_features.md` 编译选项参考。

当用户需要生成**纯矩阵乘法**算子时使用本技能。此类算子仅包含 `tl.dot()` 调用，映射到 NPU 的 Cube Core 执行。

## 核心 API

### 矩阵乘法

| API | 说明 |
|-----|------|
| `tl.dot(a, b, allow_tf32=True)` | 块级矩阵乘法 `C += a @ b`, 编译器映射到 Cube Core |
| `tl.trans(x)` | 矩阵转置 (2D张量) |

**说明**: `tl.dot()` 是 Triton 标准API, Triton-Ascend 编译器会将其转换为 NPU Cube Core 的矩阵乘指令。输入 `a` 和 `b` 通过 `tl.load` 从全局内存加载到片上，结果通过 `tl.store` 写回。

### 多核并行

| API | 说明 |
|-----|------|
| `tl.program_id(axis)` | 获取当前 program ID |
| `tl.num_programs(axis)` | 获取总 program 数 |
| `triton.cdiv(a, b)` | 向上取整除法 |
| `tl.constexpr` | 编译期常量 |

## 基础 GEMM 模板

来源: `triton-ascend/docs/en/programming_guide.md` 中的子块划分示例

```python
import torch
import torch_npu
import triton
import triton.language as tl

# ---- Kernel 定义 ----
@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # M方向的block ID
    pid_n = tl.program_id(1)  # N方向的block ID

    # 当前block的起始坐标
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 初始化累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 沿K轴循环累加
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
        acc += tl.dot(a, b)  # 映射到 Cube Core

    # 写回结果
    c_ptr = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptr, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ---- 调用函数 ----
def matmul(A: torch.Tensor, B: torch.Tensor):
    assert A.shape[1] == B.shape[0], "Incompatible dimensions"
    assert A.is_contiguous() and B.is_contiguous(), "Input must be contiguous"
    M, K = A.shape
    K, N = B.shape

    # 分配输出
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Block 参数 (需根据硬件调整)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    # Grid: 2D, 但NPU上推荐使用1D (编译器会合并)
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C


# ---- 测试验证 ----
torch.manual_seed(0)
M, K, N = 512, 512, 512
A = torch.randn(M, K, device='npu').half()
B = torch.randn(K, N, device='npu').half()
C_triton = matmul(A, B)
C_torch = A @ B
print(f"Max diff: {torch.max(torch.abs(C_triton.cpu() - C_torch.cpu()))}")
```

## NPU 优化: 多核物理核心绑定

NPU 上推荐将 grid 设为物理核心数，然后在 kernel 内部循环处理多个 block。

来源: `triton-ascend/docs/en/programming_guide.md` 多核并行示例

```python
import triton.runtime.driver as driver

# 获取物理核心数
device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
aicore_num = properties["num_aicore"]  # tl.dot 算子使用 AI Core 数

@triton.jit
def matmul_kernel_persistent(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)
    NUM_BLOCKS_M = (M + BLOCK_M - 1) // BLOCK_M
    NUM_BLOCKS_N = (N + BLOCK_N - 1) // BLOCK_N
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N

    # 每个 core 循环处理分配到的 block
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        # 恢复 2D block 索引
        pid_m = block_idx // NUM_BLOCKS_N
        pid_n = block_idx % NUM_BLOCKS_N

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

        c_ptr = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        tl.store(c_ptr, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul_persistent(A: torch.Tensor, B: torch.Tensor):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    NUM_CORE = aicore_num  # 固定为物理核心数

    grid = (NUM_CORE,)
    matmul_kernel_persistent[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C
```

## Autotune 模板

使用 `@triton.autotune` 自动选择最优 BLOCK 参数。

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_autotuned(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # kernel body 与基础模板相同
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

    c_ptr = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptr, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

## GEMV 模板 (矩阵-向量乘)

当 M=1 时退化为 GEMV，使用 1D grid 即可。

```python
@triton.jit
def gemv_kernel(
    A, B, C,
    N, K,
    stride_ak, stride_bn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)

    for n_start in range(pid * BLOCK_N, N, NUM_CORE * BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            a = tl.load(A + offs_k, mask=mask_k, other=0.0)  # [K] 向量
            b = tl.load(
                B + offs_k[:, None] * stride_ak + offs_n[None, :] * stride_bn,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0
            )  # [BLOCK_K, BLOCK_N]
            acc += tl.sum(a[:, None] * b, axis=0)  # 内积

        tl.store(C + offs_n, acc.to(C.dtype.element_ty), mask=mask_n)
```

## Batch GEMM 模板

在额外维度 (batch) 上展平，使用 1D grid + 循环。

```python
@triton.jit
def batch_matmul_kernel(
    A, B, C,
    batch_stride_a, batch_stride_b, batch_stride_c,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    BATCH: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)

    NUM_BLOCKS_M = (M + BLOCK_M - 1) // BLOCK_M
    NUM_BLOCKS_N = (N + BLOCK_N - 1) // BLOCK_N
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N * BATCH

    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        # 恢复 batch, m, n 索引
        batch_idx = block_idx // (NUM_BLOCKS_M * NUM_BLOCKS_N)
        remaining = block_idx % (NUM_BLOCKS_M * NUM_BLOCKS_N)
        pid_m = remaining // NUM_BLOCKS_N
        pid_n = remaining % NUM_BLOCKS_N

        # batch 偏移
        a_base = A + batch_idx.to(tl.int64) * batch_stride_a
        b_base = B + batch_idx.to(tl.int64) * batch_stride_b
        c_base = C + batch_idx.to(tl.int64) * batch_stride_c

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            a = tl.load(
                a_base + (offs_m[:, None] * stride_am + (offs_k[None, :] + k) * stride_ak),
                mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K),
                other=0.0
            )
            b = tl.load(
                b_base + ((offs_k[:, None] + k) * stride_bk + offs_n[None, :] * stride_bn),
                mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N),
                other=0.0
            )
            acc += tl.dot(a, b)

        tl.store(
            c_base + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn),
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
        )
```

## 转置 GEMM (C = A @ B^T)

```python
@triton.jit
def matmul_transB_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,  # B的stride是转置的
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
        # B^T: 交换行列索引和stride
        b = tl.load(
            B + (offs_n[None, :] * stride_bn + (offs_k[:, None] + k) * stride_bk),
            mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N),
            other=0.0
        )
        acc += tl.dot(a, b)

    c_ptr = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptr, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

## 性能优化指南

### Block Size 选择

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| BLOCK_M | 64, 128 | M方向分块大小 |
| BLOCK_N | 64, 128 | N方向分块大小 |
| BLOCK_K | 32, 64 | K方向累加步长 |

**原则**:
- BLOCK_M * BLOCK_K * sizeof(dtype) <= UB容量 (192KB on A2)
- BLOCK_N 同理
- 使用 `@triton.autotune` 自动选择最优组合
- 推荐使用2的幂

### 多核并行策略

| 策略 | 适用场景 | Grid设置 |
|------|---------|---------|
| 标准2D | 小矩阵, block数 <= 核心数 | `(M/BLOCK_M, N/BLOCK_N)` |
| 持久化1D+循环 | 大矩阵, 充分利用所有核心 | `(aicore_num,)` |
| TRITON_ALL_BLOCKS_PARALLEL | 自动模式, 要求kernel无执行顺序依赖 | `(triton.cdiv(M, BM) * triton.cdiv(N, BN),)` + 环境变量 |

### 内存对齐

- 确保输入输出张量 `contiguous` (`tensor.is_contiguous()`)
- BLOCK_SIZE 应确保 32 字节对齐 (例如 BLOCK * sizeof(dtype) 是 32 的倍数)
- float16: BLOCK 应为 16 的倍数; float32: BLOCK 应为 8 的倍数

### UB 溢出处理

如果出现 `ub overflow` 错误:

1. **减小 BLOCK_M / BLOCK_N / BLOCK_K**
2. **引入子块循环** (BLOCK_SIZE + BLOCK_SIZE_SUB 模式):

```python
@triton.jit
def matmul_kernel_subblock(
    A, B, C, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_K_SUB: tl.constexpr,  # K方向的子块大小
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K_SUB):
        offs_k = k + tl.arange(0, BLOCK_K_SUB)
        a = tl.load(
            A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0
        )
        b = tl.load(
            B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn),
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0
        )
        acc += tl.dot(a, b)

    c_ptr = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptr, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

### 编译器 NPU 选项

可在 autotune config 中设置 NPU 特定选项:

```python
triton.Config(
    {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
    multibuffer=True,                       # 启用乒乓流水线 (默认True)
    inject_barrier_all=True,                # 自动注入屏障
    enable_linearize=True,                  # 线性化Pass
    auto_blockify_size=128,                 # AutoBlockify大小
)
```

## 约束

- **使用 `tl.dot()`**: 本技能模板使用 `tl.dot()`；如需带缩放的矩阵乘法可使用 `tl.dot_scaled()`（Triton-Ascend 已支持）
- **输入contiguous**: 确保 `A.is_contiguous()` 和 `B.is_contiguous()`, 否则结果不正确
- **累加精度**: `tl.dot` 输出建议用 `tl.float32` 累加, 最后转换为目标dtype
- **Grid限制**: NPU上 grid 总维度不超过 65535
- **Block size**: BLOCK_M, BLOCK_N, BLOCK_K 推荐为2的幂
- **多核优化**: 大矩阵场景务必使用持久化模式, 将 grid 绑定到物理核心数

## 参考

- 编程指南 (含matmul示例): `triton-ascend/docs/en/programming_guide.md`
- 架构设计 (含编译选项): `triton-ascend/docs/en/architecture_design_and_core_features.md`
- 迁移指南 (含coreDim和UB溢出处理): `triton-ascend/docs/en/migration_guide/migrate_from_gpu.md`
- 性能分析: `triton-ascend/docs/en/debug_guide/profiling.md`
- 在线文档: https://triton-ascend.readthedocs.io/zh-cn/latest/index.html
