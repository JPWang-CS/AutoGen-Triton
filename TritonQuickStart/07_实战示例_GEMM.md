# 07 实战示例：GEMM 矩阵乘法

## 本章学习目标

- 从零实现完整的高性能 GEMM kernel
- 理解从基础到优化的逐步改进过程
- 掌握持久化内核模式在 GEMM 中的应用
- 学会性能分析和 BLOCK_SIZE 选择

---

## 7.1 GEMM 问题定义

矩阵乘法 `C = A @ B`，其中 A 是 [M, K]，B 是 [K, N]，C 是 [M, N]。

```
A [M x K]    B [K x N]    C [M x N]
+------+     +------+     +------+
|      |  x  |      |  =  |      |
|      |     |      |     |      |
+------+     +------+     +------+
```

---

## 7.2 第一步：Naive 1D GEMM

最简单的实现：每个 program 计算输出矩阵的一个元素。

```python
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def naive_gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                      stride_am, stride_ak, stride_bk, stride_bn,
                      stride_cm, stride_cn,
                      BLOCK_SIZE: tl.constexpr):
    # 每个 program 计算一个输出元素
    pid = tl.program_id(0)
    row = pid // N
    col = pid % N

    if row < M and col < N:
        # 沿 K 维循环累加
        acc = 0.0
        for k in range(0, K, BLOCK_SIZE):
            k_off = k + tl.arange(0, BLOCK_SIZE)
            mask = k_off < K
            a_val = tl.load(a_ptr + row * stride_am + k_off * stride_ak, mask=mask, other=0.0)
            b_val = tl.load(b_ptr + k_off * stride_bk + col * stride_bn, mask=mask, other=0.0)
            acc += tl.sum(a_val * b_val)
        tl.store(c_ptr + row * stride_cm + col * stride_cn, acc)

# 启动 M*N 个 program
grid = (M * N,)
naive_gemm_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1), BLOCK_SIZE=256)
```

**问题**：每个 program 只计算 1 个元素，无法利用 `tl.dot` 的矩阵运算能力，性能极差。

---

## 7.3 第二步：2D 分块 GEMM（推荐）

利用 `tl.dot` 进行分块矩阵乘法，每个 program 计算输出的一个分块。

```python
@triton.jit
def tiled_gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                      stride_am, stride_ak, stride_bk, stride_bn,
                      stride_cm, stride_cn,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr):
    # 获取 program 在 2D grid 中的位置
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 输出分块的行列偏移
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 初始化累加器
    accumulator = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # K 维分块循环
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # 加载 A 分块 [BLOCK_M, BLOCK_K]
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # 加载 B 分块 [BLOCK_K, BLOCK_N]
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # 矩阵乘法累加
        accumulator += tl.dot(a, b)

    # 存储结果
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def tiled_gemm(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    tiled_gemm_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c
```

---

## 7.4 第三步：带 Autotune 的优化 GEMM

在 2D 分块基础上添加自动调优：

```python
def get_gemm_configs():
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, multibuffer=True),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32}, multibuffer=True),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, multibuffer=True),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, multibuffer=True),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, multibuffer=True),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, multibuffer=True),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, multibuffer=True),
    ]

@triton.autotune(configs=get_gemm_configs(), key=["M", "N", "K"])
@triton.jit
def optimized_gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                          stride_am, stride_ak, stride_bk, stride_bn,
                          stride_cm, stride_cn,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                          BLOCK_K: tl.constexpr):
    # 与 tiled_gemm_kernel 相同的实现
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

---

## 7.5 第四步：持久化内核 GEMM

固定核数为 AI Core 数量，通过内部循环分配任务。减少核启动开销：

```python
import triton.runtime.driver as driver

device = torch.npu.current_device()
props = driver.active.utils.get_device_properties(device)
aicore_num = props["num_aicore"]  # 如 24

@triton.jit
def persistent_gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                           stride_am, stride_ak, stride_bk, stride_bn,
                           stride_cm, stride_cn,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                           BLOCK_K: tl.constexpr, NUM_CORES: tl.constexpr):
    pid = tl.program_id(0)

    # 计算总任务数
    num_m = (M + BLOCK_M - 1) // BLOCK_M
    num_n = (N + BLOCK_N - 1) // BLOCK_N
    total_blocks = num_m * num_n

    # 跨步分配任务
    for block_idx in range(pid, total_blocks, NUM_CORES):
        # 从一维索引还原二维坐标
        pid_m = block_idx // num_n
        pid_n = block_idx % num_n

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        accumulator = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
            b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                        mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
            accumulator += tl.dot(a, b)

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# 固定核数启动
grid = (aicore_num,)
persistent_gemm_kernel[grid](a, b, c, M, N, K,
                             a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                             c.stride(0), c.stride(1),
                             BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
                             NUM_CORES=aicore_num)
```

---

## 7.6 性能分析

### Fractal 对齐要求

NPU Cube Core 执行矩阵乘法时，数据按"分形"（fractal）格式排列：

| 数据类型 | 分形大小 | BLOCK 约束 |
|---------|---------|-----------|
| FP16 | 16×16 | BLOCK_M/N/K 必须是 16 的倍数 |
| BF16 | 16×16 | 同 FP16 |
| INT8 | 32×16 或 16×32 | BLOCK 必须是 16 或 32 的倍数 |

**推荐**: BLOCK_M、BLOCK_N、BLOCK_K 均选择 16 的倍数（如 32, 64, 128, 256）。

### tf.dot 数据类型约束

| 特性 | 说明 |
|------|------|
| 支持输入 | int8, int16, int32, fp16, fp32, bf16 |
| 不支持输入 | uint8/16/32/64, fp64 |
| 累加器 | 必须 fp32（不支持 fp16 累加） |

### TFLOPS 计算

```python
# GEMM 的浮点运算量 = 2 * M * N * K
flops = 2 * M * N * K
# 例如 M=N=K=1024, float16: flops = 2 * 1024^3 = 2,147,483,648

# 测量时间
ms = triton.testing.do_bench(lambda: matmul(a, b))

# 计算 TFLOPS
tflops = flops / (ms * 1e-3) / 1e12
print(f"性能: {tflops:.2f} TFLOPS")
```

### BLOCK_SIZE 选择参考

| M x N x K | 推荐 BLOCK_M | 推荐 BLOCK_N | 推荐 BLOCK_K |
|-----------|-------------|-------------|-------------|
| 128x128x128 | 64 | 64 | 32 |
| 512x512x512 | 128 | 128 | 32 |
| 1024x1024x1024 | 128 | 128 | 64 |
| 4096x4096x4096 | 128 | 256 | 64 |

---

## 练习题

1. 比较三种 GEMM 实现的性能差异，记录 TFLOPS 数据。
2. 尝试在持久化内核中添加 autotune，观察效果。
3. 实现 `C = A @ B.T`（B 需要转置），提示：修改 B 的 stride 即可。
4. 思考：为什么持久化内核模式在大矩阵上性能更好？
