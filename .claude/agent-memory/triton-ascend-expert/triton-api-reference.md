---
name: triton-api-reference
description: Triton-Ascend 全量 API 参考索引（~90个 API），按 18 个分类组织，含函数签名、数据类型支持、Ascend 约束和文档 URL 路径。来源于官方在线文档 triton_api/index.html 及各详情页。
type: reference
---

# Triton-Ascend API 全量参考

来源: https://triton-ascend.readthedocs.io/zh-cn/latest/triton_api/index.html (2026-04-30 采集)

## URL 路径模式

```
https://triton-ascend.readthedocs.io/zh-cn/latest/triton_api/{Category}_Ops/{api_name}.html
```

**已知 404**: `Memory_Ops/` 目录下的 load, store, load_tensor_descriptor, make_block_ptr 页面无法访问。

---

## 一、Extension_Ops (昇腾扩展)

### tl.dot_scaled
- **路径**: `Extension_Ops/dot_scaled.html`
- **签名**: `tl.dot_scaled(a, b, scale_a=None, scale_b=None, out_dtype=None)`
- **说明**: 带缩放的矩阵乘法，用于 FP8 精度场景
- **Ascend 支持**: 支持 fp16/bf16 输入, 输出 fp32/fp16/bf16

### tl.sync_block
- **路径**: `Extension_Ops/sync_block.html`
- **签名**: `tl.sync_block()`
- **说明**: 块内同步屏障，等待所有线程完成
- **约束**: 仅在单个 block 内有效

### tl.sync_block_wait
- **路径**: `Extension_Ops/sync_block_wait.html`
- **说明**: 等待同步完成

### tl.sync_block_set
- **路径**: `Extension_Ops/sync_block_set.html`
- **说明**: 设置同步标志

### tl.sync_block_all
- **路径**: `Extension_Ops/sync_block_all.html`
- **说明**: 全局同步（所有 block）

### tl.compile_hint
- **路径**: `Extension_Ops/compile_hint.html`
- **签名**: `tl.compile_hint(strategy)`
- **说明**: 向编译器提供优化提示
- **Ascend 专属策略**: 见编译器选项章节

### tl.custom_op
- **路径**: `Extension_Ops/custom_op.html`
- **说明**: 自定义算子注册

### tl.inline_asm_elementwise
- **路径**: `Extension_Ops/inline_asm_elementwise.html`
- **签名**: `tl.inline_asm_elementwise(asm: str, constraints: str, args, dtype, is_pure=True, pack=1)`
- **说明**: 内联 Ascend Vector 汇编指令
- **参数**:
  - `asm`: 汇编指令字符串
  - `constraints`: 输入输出约束
  - `args`: 输入张量列表
  - `dtype`: 输出数据类型
  - `is_pure`: 是否无副作用
  - `pack`: 打包因子 (1/2/4)
- **Ascend 约束**: 仅支持 Vector 指令集，不支持 Cube 指令

### tl.parallel
- **路径**: `Extension_Ops/parallel.html`
- **签名**: `tl.parallel(range, num_stages=None)`
- **说明**: 并行循环提示，指示编译器将循环展开为并行流水线
- **Ascend 行为**: 编译器尝试生成流水并行指令

### tl.insert_slice
- **路径**: `Extension_Ops/insert_slice.html`
- **签名**: `tl.insert_slice(src, dst, offset)`
- **说明**: 将张量切片插入目标张量的指定位置
- **Ascend 约束**: offset 必须是编译时常量

### tl.extract_slice
- **路径**: `Extension_Ops/extract_slice.html`
- **签名**: `tl.extract_slice(src, offset)`
- **说明**: 从源张量提取切片
- **Ascend 约束**: offset 必须是编译时常量

### tl.get_element
- **路径**: `Extension_Ops/get_element.html`
- **签名**: `tl.get_element(tensor, index)`
- **说明**: 从张量中获取指定索引的标量元素

---

## 二、Atomic_Ops (原子操作)

### tl.atomic_add
- **路径**: `Atomic_Ops/atomic_add.html`
- **签名**: `tl.atomic_add(pointer, value, mask=None, sem=None, scope=None)`
- **Ascend 约束**:
  - `sem` 仅支持 `"acq_rel"`
  - `scope` 仅支持 `"gpu"`
  - 支持 dtype: fp16, bf16, fp32, int32

### tl.atomic_cas
- **路径**: `Atomic_Ops/atomic_cas.html`
- **签名**: `tl.atomic_cas(pointer, cmp, val, mask=None, sem=None, scope=None)`
- **说明**: 原子比较并交换 (Compare-And-Swap)
- **Ascend 约束**: 同 atomic_add 限制

### tl.atomic_and
- **路径**: `Atomic_Ops/atomic_and.html`
- **签名**: `tl.atomic_and(pointer, value, mask=None, sem=None, scope=None)`
- **说明**: 原子按位与

### tl.atomic_or
- **路径**: `Atomic_Ops/atomic_or.html`
- **说明**: 原子按位或

### tl.atomic_xor
- **路径**: `Atomic_Ops/atomic_xor.html`
- **说明**: 原子按位异或

### tl.atomic_max
- **路径**: `Atomic_Ops/atomic_max.html`
- **说明**: 原子最大值

### tl.atomic_min
- **路径**: `Atomic_Ops/atomic_min.html`
- **说明**: 原子最小值

### tl.atomic_xchg
- **路径**: `Atomic_Ops/atomic_xchg.html`
- **签名**: `tl.atomic_xchg(pointer, value, mask=None, sem=None, scope=None)`
- **说明**: 原子交换

**原子操作通用限制**:
- NPU 不支持 `sem="release"` / `sem="acquire"`，仅 `sem="acq_rel"`
- NPU 不支持 `scope="cta"` / `scope="sys"`，仅 `scope="gpu"`
- 无 fp64 支持

---

## 三、Comparison_Ops (比较操作)

### tl.eq
- **路径**: `Comparison_Ops/eq.html`
- **签名**: `tl.eq(a, b)` / `a == b`

### tl.ne
- **路径**: `Comparison_Ops/ne.html`
- **签名**: `tl.ne(a, b)` / `a != b`

### tl.lt
- **路径**: `Comparison_Ops/lt.html`
- **签名**: `tl.lt(a, b)` / `a < b`

### tl.le
- **路径**: `Comparison_Ops/le.html`
- **签名**: `tl.le(a, b)` / `a <= b`

### tl.gt
- **路径**: `Comparison_Ops/gt.html`
- **签名**: `tl.gt(a, b)` / `a > b`

### tl.ge
- **路径**: `Comparison_Ops/ge.html`
- **签名**: `tl.ge(a, b)` / `a >= b`

**比较操作 Ascend 注意**:
- i32/i64 比较在 `tl.where` 中退化为标量 (Scalar Fallback)
- 建议 `.to(tl.float32)` 后再比较
- `tl.load` 的 mask 参数中的整数比较由 MTE2 引擎处理，不退化

---

## 四、Compiler_Hint_Ops (编译器提示)

### tl.multiple_of
- **路径**: `Compiler_Hint_Ops/multiple_of.html`
- **签名**: `tl.multiple_of(expr, value)`
- **说明**: 提示编译器 expr 的值是 value 的倍数

### tl.max_contiguous
- **路径**: `Compiler_Hint_Ops/max_contiguous.html`
- **签名**: `tl.max_contiguous(values, size)`
- **说明**: 提示编译器 values 中最多有 size 个连续值

### tl.max_constancy
- **路径**: `Compiler_Hint_Ops/max_constancy.html`
- **签名**: `tl.max_constancy(values, size)`
- **说明**: 提示编译器 values 中最多有 size 个相同值

### tl.assume
- **路径**: `Compiler_Hint_Ops/assume.html`
- **签名**: `tl.assume(condition)`
- **说明**: 假设条件恒为真，用于编译器优化

---

## 五、Creation_Ops (创建操作)

### tl.arange
- **路径**: `Creation_Ops/arange.html`
- **签名**: `tl.arange(start, end)`
- **说明**: 创建连续整数序列 [start, end)
- **注意**: 返回 int32 类型

### tl.range
- **路径**: `Creation_Ops/range.html`
- **签名**: `tl.range(start, end)`
- **说明**: 与 arange 类似，用于动态范围

### tl.static_range
- **路径**: `Creation_Ops/static_range.html`
- **说明**: 编译时常量范围迭代器

### tl.full
- **路径**: `Creation_Ops/full.html`
- **签名**: `tl.full(shape, value, dtype=None)`
- **说明**: 创建填充指定值的张量

### tl.zeros
- **路径**: `Creation_Ops/zeros.html`
- **签名**: `tl.zeros(shape, dtype=None)`
- **说明**: 创建全零张量

### tl.zeros_like
- **路径**: `Creation_Ops/zeros_like.html`
- **签名**: `tl.zeros_like(input)`
- **说明**: 创建与输入形状和类型相同的全零张量

### tl.rand
- **路径**: `Creation_Ops/rand.html`
- **签名**: `tl.rand(seed, offset, shape, dtype=None)`
- **说明**: 生成 [0, 1) 均匀分布随机数
- **Ascend 支持**: 支持 fp16/bf16/fp32

### tl.randn
- **路径**: `Creation_Ops/randn.html`
- **签名**: `tl.randn(seed, offset, shape, dtype=None)`
- **说明**: 生成标准正态分布随机数
- **Ascend 支持**: 支持 fp16/bf16/fp32

### tl.randint
- **路径**: `Creation_Ops/randint.html`
- **签名**: `tl.randint(seed, offset, shape, lo, hi, dtype=None)`
- **说明**: 生成 [lo, hi) 范围内均匀分布整数随机数

### tl.randint4x
- **路径**: `Creation_Ops/randint4x.html`
- **说明**: 生成 4 组随机整数（用于 dropout 等）

### tl.cast
- **路径**: `Creation_Ops/cast.html`
- **签名**: `tl.cast(input, dtype)` / `input.to(dtype)`
- **说明**: 类型转换
- **Ascend 支持**: fp16↔bf16↔fp32, int8↔int32, fp16→fp32→fp16
- **不支持**: fp64, uint16/uint32/uint64

### tl.cat
- **路径**: `Creation_Ops/cat.html`
- **签名**: `tl.cat(a, b, axis)`
- **说明**: 沿指定轴拼接两个张量

---

## 六、Debug_Ops (调试操作)

### tl.static_print
- **路径**: `Debug_Ops/static_print.html`
- **签名**: `tl.static_print(*args, **kwargs)`
- **说明**: 编译时打印（打印 constexpr 值）

### tl.static_assert
- **路径**: `Debug_Ops/static_assert.html`
- **签名**: `tl.static_assert(condition, msg=None)`
- **说明**: 编译时断言

### tl.device_print
- **路径**: `Debug_Ops/device_print.html`
- **签名**: `tl.device_print(*args, **kwargs)`
- **说明**: 运行时设备端打印

### tl.device_assert
- **路径**: `Debug_Ops/device_assert.html`
- **签名**: `tl.device_assert(condition, msg=None)`
- **说明**: 运行时设备端断言

### tl.debug_barrier
- **路径**: `Debug_Ops/debug_barrier.html`
- **签名**: `tl.debug_barrier()`
- **说明**: 调试用同步屏障
- **Ascend 约束**: 可能影响性能，仅在调试时使用

---

## 七、Indexing_Ops (索引操作)

### tl.gather
- **路径**: `Indexing_Ops/gather.html`
- **签名**: `tl.gather(src, idx, axis=0)`
- **说明**: 按索引从源张量收集元素
- **Ascend 支持**: 支持 1D 和 2D 索引

### tl.scatter_ub_to_out
- **路径**: `Indexing_Ops/scatter_ub_to_out.html`
- **说明**: 从 UB 到 GM 的高效数据分散（Ascend 扩展）

### tl.flip
- **路径**: `Indexing_Ops/flip.html`
- **签名**: `tl.flip(input, dims)`
- **说明**: 沿指定维度翻转张量

### tl.swizzle2d
- **路径**: `Indexing_Ops/swizzle2d.html`
- **签名**: `tl.swizzle2d(i, j, size_i, size_j, size_g)`
- **说明**: 2D 交错索引变换，用于优化访存模式

---

## 八、InlineAssembly_Ops (内联汇编)

### tl.inline_asm_elementwise
- **路径**: 见 Extension_Ops 章节（已在上面列出）

---

## 九、Iterator_Ops (迭代器)

### tl.advance
- **路径**: `Iterator_Ops/advance.html`
- **签名**: `tl.advance(block_ptr, offsets)`
- **说明**: 推进 block pointer 的偏移量

### tl.make_block_ptr
- **路径**: `Memory_Ops/make_block_ptr.html` (注意: 此页面 404)
- **签名**: `tl.make_block_ptr(base, shape, strides, offsets, block_shape, order=None)`
- **说明**: 创建 block pointer 描述符，用于结构化访存

### tl.load_tensor_descriptor
- **路径**: `Memory_Ops/load_tensor_descriptor.html` (注意: 此页面 404)
- **说明**: 加载张量描述符

---

## 十、LinearAlgebra_Ops (线性代数)

### tl.dot
- **路径**: `LinearAlgebra_Ops/dot.html`
- **签名**: `tl.dot(a, b, allow_tf32=None)`
- **说明**: 块级矩阵乘法 (A[M,K] @ B[K,N] → C[M,N])
- **Ascend 支持**:
  - 输入: fp16, bf16, int8
  - 输出: fp32 (默认), fp16, bf16
  - 自动映射到 Cube Core
- **约束**:
  - 最内层两个维度必须是 [M,K] @ [K,N] 形状
  - K 维度 (内积轴) 推荐 2 的幂
  - Atlas A2: 尾轴 512B 对齐
- **CV 融合**: dot 后接 element-wise 自动触发 CV 融合

### tl.dot_scaled
- **路径**: 见 Extension_Ops 章节（已在上面列出）

### tl.trans
- **路径**: `LinearAlgebra_Ops/trans.html`
- **签名**: `tl.trans(input)` / `tl.trans(input, dims)`
- **说明**: 张量转置
- **Ascend 约束**:
  - 仅支持相邻维度转置（如 dims=[0,1] 或 dims=[1,0]）
  - **不支持**非相邻维度转置（如 dims=[2,1,0] 会报错）
  - 最大支持 8 维
  - 推荐: 2D 矩阵转置用 `tl.trans(a)`

### tl.matmul
- **路径**: `LinearAlgebra_Ops/matmul.html`
- **说明**: 高层矩阵乘法接口

---

## 十一、Logic_Ops (逻辑操作)

### tl.and_
- **路径**: `Logic_Ops/and.html`
- **签名**: `tl.and_(a, b)` / `a & b`

### tl.or_
- **路径**: `Logic_Ops/or.html`
- **签名**: `tl.or_(a, b)` / `a | b`

### tl.not_
- **路径**: `Logic_Ops/not.html`
- **签名**: `tl.not_(a)` / `~a`

### tl.logical_and
- **路径**: `Logic_Ops/logical_and.html`
- **签名**: `tl.logical_and(a, b)`

### tl.logical_or
- **路径**: `Logic_Ops/logical_or.html`
- **签名**: `tl.logical_or(a, b)`

---

## 十二、Math_Ops (数学操作)

### tl.add
- **路径**: `Math_Ops/add.html`
- **签名**: `tl.add(a, b)` / `a + b`

### tl.sub
- **路径**: `Math_Ops/sub.html`
- **签名**: `tl.sub(a, b)` / `a - b`

### tl.mul
- **路径**: `Math_Ops/mul.html`
- **签名**: `tl.mul(a, b)` / `a * b`

### tl.div
- **路径**: `Math_Ops/div.html`
- **签名**: `tl.div(a, b)` / `a / b`

### tl.abs
- **路径**: `Math_Ops/abs.html`
- **签名**: `tl.abs(input)`

### tl.neg
- **路径**: `Math_Ops/neg.html`
- **签名**: `tl.neg(input)` / `-input`

### tl.cdiv
- **路径**: `Math_Ops/cdiv.html`
- **签名**: `tl.cdiv(a, b)`
- **说明**: 向上取整除法 (a + b - 1) // b

### tl.ceil
- **路径**: `Math_Ops/ceil.html`
- **签名**: `tl.ceil(input)`

### tl.floor
- **路径**: `Math_Ops/floor.html`
- **签名**: `tl.floor(input)`

### tl.clamp
- **路径**: `Math_Ops/clamp.html`
- **签名**: `tl.clamp(input, min=None, max=None)`
- **说明**: 将值裁剪到 [min, max] 范围

### tl.exp
- **路径**: `Math_Ops/exp.html`
- **签名**: `tl.exp(input)`
- **Ascend 支持**: fp16, bf16, fp32
- **不支持**: fp64

### tl.log
- **路径**: `Math_Ops/log.html`
- **签名**: `tl.log(input)`

### tl.log2
- **路径**: `Math_Ops/log2.html`
- **签名**: `tl.log2(input)`

### tl.cos
- **路径**: `Math_Ops/cos.html`
- **签名**: `tl.cos(input)`

### tl.sqrt
- **路径**: `Math_Ops/sqrt.html`
- **签名**: `tl.sqrt(input)`
- **Ascend 支持**: fp16, bf16, fp32

### tl.rsqrt
- **路径**: `Math_Ops/rsqrt.html`
- **签名**: `tl.rsqrt(input)` — 1/sqrt(input)
- **Ascend 支持**: fp16, bf16, fp32

### tl.sigmoid
- **路径**: `Math_Ops/sigmoid.html`
- **签名**: `tl.sigmoid(input)` — 1/(1+exp(-x))
- **Ascend 支持**: fp16, bf16, fp32

### tl.erf
- **路径**: `Math_Ops/erf.html`
- **签名**: `tl.erf(input)`
- **Ascend 支持**: fp16, bf16, fp32
- **用途**: GELU 激活函数: 0.5 * x * (1 + erf(x / sqrt(2)))

### tl.fma
- **路径**: `Math_Ops/fma.html`
- **签名**: `tl.fma(a, b, c)` — a*b+c (融合乘加)

### tl.maximum
- **路径**: `Math_Ops/maximum.html`
- **签名**: `tl.maximum(a, b)`

### tl.minimum
- **路径**: `Math_Ops/minimum.html`
- **签名**: `tl.minimum(a, b)`

### tl.mod
- **路径**: `Math_Ops/mod.html`
- **签名**: `tl.mod(a, b)` / `a % b`

### tl.umulhi
- **路径**: `Math_Ops/umulhi.html`
- **签名**: `tl.umulhi(a, b)`
- **说明**: 无符号乘法高位

### tl.div_rn
- **路径**: `Math_Ops/div_rn.html`
- **说明**: 舍入到最近的除法

### tl.fdiv
- **路径**: `Math_Ops/fdiv.html`
- **说明**: 快速浮点除法

### tl.where
- **路径**: `Math_Ops/where.html`
- **签名**: `tl.where(condition, a, b)`
- **说明**: 条件选择 — condition 为 True 选 a，否则选 b
- **Ascend 关键约束**:
  - 整数条件 (i32/i64) 退化为标量执行！
  - **修复**: `condition.to(tl.float32)` 转为浮点比较
  - `tl.where(cols < N, a, b)` 中 cols 为 int32 → 退化
  - `tl.where(cols_f < N, a, b)` 中 cols_f = cols.to(tl.float32) → 正常

---

## 十三、Memory_Ops (内存操作)

> **注意**: 此分类下所有在线文档页面返回 404，信息来源于 CLAUDE.md、本地文档和标准 Triton 知识。

### tl.load
- **路径**: `Memory_Ops/load.html` (404)
- **签名**: `tl.load(pointer, mask=None, other=0.0, boundary_check=None, padding_option=None)`
- **说明**: 从全局内存加载数据到片上 (UB)
- **关键参数**:
  - `mask`: 布尔掩码，越界保护
  - `other`: mask=False 位置的填充值
  - `boundary_check`: 边界检查的维度
- **Ascend 约束**:
  - 尾轴 32B 对齐 (VV 算子)
  - 尾轴 512B 对齐 (CV 算子)
  - mask 中的整数比较由 MTE2 引擎处理，**不退化**为标量

### tl.store
- **路径**: `Memory_Ops/store.html` (404)
- **签名**: `tl.store(pointer, value, mask=None, boundary_check=None)`
- **说明**: 将数据从片上写回全局内存
- **Ascend 约束**: 同 load 对齐要求

### tl.make_tensor
- **路径**: `Memory_Ops/tensor.html` (404)
- **签名**: `tl.tensor(data, dtype)`
- **说明**: 从原始数据创建张量

### tl.make_block_ptr
- **路径**: `Memory_Ops/make_block_ptr.html` (404)
- **签名**: `tl.make_block_ptr(base, shape, strides, offsets, block_shape, order=None)`
- **说明**: 创建 block pointer，用于结构化分块访存

---

## 十四、Programming_Model_Ops (编程模型)

### tl.program_id
- **路径**: `Programming_Model_Ops/program_id.html`
- **签名**: `tl.program_id(axis)`
- **说明**: 获取当前 program 在指定轴的 ID
- **NPU 用法**: `pid = tl.program_id(0)` — 获取当前核 ID

### tl.num_programs
- **路径**: `Programming_Model_Ops/num_programs.html`
- **签名**: `tl.num_programs(axis)`
- **说明**: 获取指定轴的 program 总数
- **NPU 等价**: 即 grid 大小

---

## 十五、Random_Ops (随机操作)

> 详细说明见 Creation_Ops 中的 rand, randn, randint, randint4x

### tl.rand
- **路径**: `Creation_Ops/rand.html`
- **签名**: `tl.rand(seed, offset, shape, dtype=None)`

### tl.randn
- **路径**: `Creation_Ops/randn.html`
- **签名**: `tl.randn(seed, offset, shape, dtype=None)`

### tl.randint
- **路径**: `Creation_Ops/randint.html`

### tl.randint4x
- **路径**: `Creation_Ops/randint4x.html`

---

## 十六、Reduction_Ops (归约操作)

### tl.sum
- **路径**: `Reduction_Ops/sum.html`
- **签名**: `tl.sum(input, axis=None)`
- **说明**: 沿轴求和
- **Ascend 支持**: fp16, bf16, fp32, int32
- **返回类型**: 与输入类型相同

### tl.max
- **路径**: `Reduction_Ops/max.html`
- **签名**: `tl.max(input, axis=None)`
- **说明**: 沿轴取最大值

### tl.min
- **路径**: `Reduction_Ops/min.html`
- **签名**: `tl.min(input, axis=None)`
- **说明**: 沿轴取最小值

### tl.argmax
- **路径**: `Reduction_Ops/argmax.html`
- **签名**: `tl.argmax(input, axis)`
- **说明**: 沿轴取最大值索引
- **返回类型**: int32

### tl.argmin
- **路径**: `Reduction_Ops/argmin.html`
- **签名**: `tl.argmin(input, axis)`
- **说明**: 沿轴取最小值索引
- **返回类型**: int32

### tl.xor_sum
- **路径**: `Reduction_Ops/xor_sum.html`
- **签名**: `tl.xor_sum(input, axis)`
- **说明**: 沿轴异或归约
- **适用**: 整数类型

### tl.reduce
- **路径**: `Reduction_Ops/reduce.html`
- **签名**: `tl.reduce(input, axis, combine_fn)`
- **说明**: 自定义归约操作
- **用法**: 需定义 `@triton.jit` 的 `combine_fn`
```python
@triton.jit
def _product_combine(a, b):
    return a * b
result = tl.reduce(x, axis=1, combine_fn=_product_combine)
```

### tl.softmax
- **路径**: `Reduction_Ops/softmax.html`
- **签名**: `tl.softmax(input, axis)`
- **说明**: 沿轴计算 softmax
- **Ascend 支持**: fp16, bf16, fp32
- **数值稳定性**: 内部使用 max 减法防止溢出

---

## 十七、Scan_Sort_Ops (扫描与排序)

### tl.associative_scan
- **路径**: `Scan_Sort_Ops/associative_scan.html`
- **签名**: `tl.associative_scan(input, axis, combine_fn, reverse=False)`
- **说明**: 前缀扫描（前缀和、前缀积等）
- **用法**:
```python
@triton.jit
def _add_combine(a, b):
    return a + b
prefix_sum = tl.associative_scan(x, axis=1, combine_fn=_add_combine)
```

### tl.cumsum
- **路径**: `Scan_Sort_Ops/cumsum.html`
- **签名**: `tl.cumsum(input, axis)`
- **说明**: 累积求和（前缀和）
- **Ascend 支持**: fp16, bf16, fp32

### tl.cumprod
- **路径**: `Scan_Sort_Ops/cumprod.html`
- **签名**: `tl.cumprod(input, axis)`
- **说明**: 累积求积

### tl.sort
- **路径**: `Scan_Sort_Ops/sort.html`
- **签名**: `tl.sort(input, axis)`
- **说明**: 沿轴排序（升序）

### tl.histogram
- **路径**: `Scan_Sort_Ops/histogram.html`
- **说明**: 直方图统计

---

## 十八、Shape_Ops (形状操作)

### tl.reshape
- **路径**: `Shape_Ops/reshape.html`
- **签名**: `tl.reshape(input, shape)`
- **说明**: 改变张量形状
- **Ascend 约束**: shape 必须是编译时常量

### tl.broadcast
- **路径**: `Shape_Ops/broadcast.html`
- **签名**: `tl.broadcast(input, shape)`
- **说明**: 将张量广播到目标形状
- **Ascend 约束**: 广播维度大小必须为 1

### tl.permute
- **路径**: `Shape_Ops/permute.html`
- **签名**: `tl.permute(input, dims)`
- **说明**: 维度重排
- **Ascend 约束**:
  - 仅支持相邻维度交换
  - **不支持**非相邻维度重排（如 [2,1,0] 报错）
  - 与 tl.trans 共享底层实现

### tl.expand_dims
- **路径**: `Shape_Ops/expand_dims.html`
- **签名**: `tl.expand_dims(input, axis)`
- **说明**: 在指定位置插入大小为 1 的维度

### tl.interleave
- **路径**: `Shape_Ops/interleave.html`
- **说明**: 交错合并

### tl.join
- **路径**: `Shape_Ops/join.html`
- **说明**: 沿新维度拼接

### tl.ravel
- **路径**: `Shape_Ops/ravel.html`
- **说明**: 展平为一维

### tl.split
- **路径**: `Shape_Ops/split.html`
- **说明**: 沿轴拆分

### tl.view
- **路径**: `Shape_Ops/view.html`
- **签名**: `tl.view(input, dtype)`
- **说明**: 以目标类型重新解释内存（不进行数据转换）

---

## 通用 Ascend 限制速查

### 数据类型支持矩阵

| 类型 | 数学运算 | 归约 | 线性代数 | 原子操作 | 类型转换 |
|------|---------|------|---------|---------|---------|
| fp32 | ✅ | ✅ | ✅ (累加) | ✅ | 源/目标 |
| fp16 | ✅ | ✅ | ✅ | ✅ | 源/目标 |
| bf16 | ✅ | ✅ | ✅ | ✅ | 源/目标 |
| int8 | ❌ | ❌ | ✅ (dot) | ❌ | 源 |
| int32 | ❌ | ✅ | ❌ | ✅ | 源/目标 |
| int64 | ❌ | ❌ | ❌ | ❌ | ❌ |
| fp64 | ❌ | ❌ | ❌ | ❌ | ❌ |

### 关键约束汇总

1. **无 fp64 支持**: 所有操作均不支持双精度浮点
2. **标量退化**: `tl.where(int_condition, ...)` 中整数比较退化为标量
3. **转置限制**: 仅支持相邻维度交换
4. **原子操作**: 仅 sem="acq_rel", scope="gpu"
5. **内存对齐**: VV 算子 32B, CV 算子 512B
6. **UB 容量**: 192KB (Atlas A2), multibuffer 减半
7. **grid 限制**: 推荐 = 物理核数, 最大 65535

### 编译选项 (autotune Config 参数)

| 参数 | 类型 | 说明 |
|------|------|------|
| `multibuffer` | bool | 流水并行搬运 (默认 True) |
| `unit_flag` | int | Cube 搬出优化 |
| `enable_auto_bind_sub_block` | bool | CV 融合自动子块绑定 |
| `enable_hivm_auto_cv_balance` | bool | CV 自动负载均衡 |
| `tile_mix_vector_loop` | int | Vector 循环切分数 |
| `tile_mix_cube_loop` | int | Cube 循环切分数 |
| `sync_solver` | bool | 自动优化同步点 |
| `auto_blockify_size` | int | ALL_BLOCKS_PARALLEL 优化 |

---

## 按使用频率排序的 API 快查

### 高频 (几乎每个 kernel 都用)
- `tl.program_id`, `tl.arange`, `tl.load`, `tl.store`
- `tl.sum`, `tl.max`, `tl.where`
- `tl.constexpr` (参数声明)

### 中频 (特定模式使用)
- `tl.dot`, `tl.trans`, `tl.reshape`, `tl.broadcast`
- `tl.exp`, `tl.sqrt`, `tl.rsqrt`, `tl.erf`
- `tl.zeros`, `tl.full`, `tl.cast`
- `tl.reduce`, `tl.associative_scan`

### 低频 (特殊场景)
- `tl.inline_asm_elementwise`, `tl.compile_hint`, `tl.parallel`
- `tl.gather`, `tl.scatter_ub_to_out`
- `tl.insert_slice`, `tl.extract_slice`, `tl.get_element`
- `tl.dot_scaled`, `tl.custom_op`
- `tl.atomic_*`, `tl.debug_barrier`

---

## 参考

- 官方 API 索引: https://triton-ascend.readthedocs.io/zh-cn/latest/triton_api/index.html
- 本地 API 文档: `triton-ascend/docs/zh/triton_api/`
- 标准 Triton 参考: https://triton-lang.org/main/python-api/triton.language.html
