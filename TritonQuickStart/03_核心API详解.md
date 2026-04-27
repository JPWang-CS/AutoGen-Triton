# 03 核心API详解

## 本章学习目标

- 掌握 Triton 核心 API 的用法和参数
- 理解每个 API 在 NPU 上的行为特点
- 通过代码片段熟悉 API 调用模式

---

## 3.1 内存操作 API

### tl.load —— 从 Global Memory 读取数据

```python
# 基本用法
data = tl.load(ptr + offsets)

# 带 mask（防止越界访问）
data = tl.load(ptr + offsets, mask=mask)

# 指定越界位置的默认值
data = tl.load(ptr + offsets, mask=mask, other=0.0)

# 指定内存对齐提示（提升性能）
data = tl.load(ptr + offsets, mask=mask, eviction_policy='evict_last')
```

**参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `ptr` | tensor | 指针地址（整数或整数张量） |
| `mask` | tensor[bool] | 可选，True 表示有效读取 |
| `other` | scalar | 可选，mask=False 位置的默认值，默认 0 |
| `eviction_policy` | str | 可选，缓存策略 |

### tl.store —— 向 Global Memory 写入数据

```python
# 基本用法
tl.store(ptr + offsets, value)

# 带 mask
tl.store(ptr + offsets, value, mask=mask)
```

**完整示例：带边界保护的向量操作**

```python
@triton.jit
def vector_op_kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # 读取，越界位置返回 0.0
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # 计算
    y = x * 2.0 + 1.0
    # 写入，只在有效位置写入
    tl.store(y_ptr + offsets, y, mask=mask)
```

---

## 3.2 数学运算 API

### 基础运算

```python
# 算术运算（元素级）
c = a + b      # 加法
c = a - b      # 减法
c = a * b      # 乘法
c = a / b      # 除法

# 标量运算
c = a * 2.0
c = a + 1.0
```

### 数学函数

```python
# 指数与对数
y = tl.exp(x)     # e^x
y = tl.log(x)     # ln(x)
y = tl.log2(x)    # log2(x)
y = tl.exp2(x)    # 2^x

# 幂与根号
y = tl.sqrt(x)    # √x
y = tl.abs(x)     # |x|

# 三角函数
y = tl.sin(x)
y = tl.cos(x)

# 取整
y = tl.floor(x)
y = tl.ceil(x)

# 其他
y = tl.erf(x)        # 误差函数
y = tl.clamp(x, lo, hi)  # 限制范围
y = tl.fma(a, b, c)  # a * b + c（融合乘加）
```

**示例：GELU 激活函数**

```python
@triton.jit
def gelu_kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    y = 0.5 * x * (1.0 + tl.erf(x / 1.4142135))
    tl.store(y_ptr + offsets, y, mask=mask)
```

### 比较与逻辑

```python
# 比较（返回 bool 张量）
eq = a == b
gt = a > b
lt = a < b
ge = a >= b
le = a <= b
ne = a != b

# 条件选择
result = tl.where(condition, value_if_true, value_if_false)
```

**示例：ReLU**

```python
@triton.jit
def relu_kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.where(x > 0, x, 0.0)
    tl.store(y_ptr + offsets, y, mask=mask)
```

---

## 3.3 归约操作 API

### tl.sum —— 求和归约

```python
# 对所有元素求和
total = tl.sum(x)               # 标量

# 沿某个轴归约
row_sum = tl.sum(x, axis=0)     # 对 axis=0 归约
```

### tl.max / tl.min —— 最大/最小值

```python
row_max = tl.max(x)             # 全局最大值
row_max = tl.max(x, axis=0)     # 沿 axis 归约
row_min = tl.min(x, axis=0)
```

**示例：Softmax 的核心计算**

```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = input_ptr + row_idx * row_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    row = tl.load(row_start + offsets, mask=mask, other=float('-inf')).to(tl.float32)

    # 第 1 步：求行最大值（数值稳定性）
    row_max = tl.max(row, axis=0)

    # 第 2 步：减去最大值，求指数
    numerator = tl.exp(row - row_max)

    # 第 3 步：求分母（指数之和）
    denominator = tl.sum(numerator, axis=0)

    # 第 4 步：归一化
    softmax_output = numerator / denominator

    output_start = output_ptr + row_idx * row_stride
    tl.store(output_start + offsets, softmax_output, mask=mask)
```

---

## 3.4 线性代数 API

### tl.dot —— 矩阵乘法

```python
# 2D 矩阵乘法，返回 2D 结果
# a: [BLOCK_M, BLOCK_K], b: [BLOCK_K, BLOCK_N] -> c: [BLOCK_M, BLOCK_N]
c = tl.dot(a, b)

# 控制累加精度
c = tl.dot(a, b, allow_tf32=False)  # 在 NPU 上默认使用 float32 累加
```

**注意事项（NPU）：**
- `tl.dot` 要求输入是 2D 张量
- BLOCK_K 维度必须匹配
- 累加器自动使用 float32 精度
- FP16/BF16 输入的分形对齐要求为 16x16

---

## 3.5 创建操作 API

```python
# 创建范围向量
offsets = tl.arange(0, N)           # [0, 1, 2, ..., N-1]

# 创建全零张量
z = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

# 创建填充张量
f = tl.full([BLOCK_M, BLOCK_N], value=1.0, dtype=tl.float32)
```

**示例：矩阵乘法中的累加器**

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ...,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 初始化累加器为 0
    accumulator = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # K 维循环累加
    for k in range(0, K, BLOCK_K):
        a = tl.load(...)  # [BLOCK_M, BLOCK_K]
        b = tl.load(...)  # [BLOCK_K, BLOCK_N]
        accumulator += tl.dot(a, b)

    tl.store(c_ptr + ..., accumulator)
```

---

## 3.6 数据类型

| 类型 | Triton 名称 | PyTorch 对应 | 字节数 |
|------|-------------|-------------|--------|
| 32位浮点 | `tl.float32` | `torch.float32` | 4 |
| 16位浮点 | `tl.float16` | `torch.float16` | 2 |
| BFloat16 | `tl.bfloat16` | `torch.bfloat16` | 2 |
| 32位整数 | `tl.int32` | `torch.int32` | 4 |
| 8位整数 | `tl.int8` | `torch.int8` | 1 |

```python
# 类型转换
x_float = x.to(tl.float32)       # 转为 float32
x_half = x.to(tl.float16)        # 转为 float16

# 创建张量时指定类型
z = tl.zeros([128, 128], dtype=tl.float32)
```

---

## 3.7 原子操作 API

```python
# 原子加法
tl.atomic_add(ptr + offsets, value, mask=mask)

# 原子最大值
tl.atomic_max(ptr + offsets, value, mask=mask)

# 原子最小值
tl.atomic_min(ptr + offsets, value, mask=mask)
```

**使用场景：** 当多个 program 需要写入同一内存位置时使用原子操作。但在 NPU 上应尽量避免原子操作，因为它会影响性能。推荐让每个 program 写入独立的位置。

---

## 练习题

1. 实现一个 Sigmoid kernel：`sigmoid(x) = 1 / (1 + exp(-x))`。
2. 实现一个向量 L2Norm kernel：对每个元素除以向量的 L2 范数。
3. 使用 `tl.where` 实现一个 LeakyReLU kernel：`leaky_relu(x) = x if x > 0 else 0.01 * x`。
4. 思考：为什么 softmax 示例中要先减去最大值再求 exp？不这样做会有什么问题？
