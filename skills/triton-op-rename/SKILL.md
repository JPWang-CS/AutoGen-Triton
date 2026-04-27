---
name: triton-op-rename
description: 用户需要修改Triton-Ascend算子名字时，提供旧的算子名以及新的算子名，对算子名字做全量的修改
---

# 重命名 Triton-Ascend 算子

当用户提出"修改Triton算子名称/重命名Triton算子"时使用本技能。

## 工作流

1. 确认算子**旧的名字**和算子**新的名字**。
2. 确认需要修改的算子的路径，在用户确认后再进行修改。
3. 执行重命名：
   - 全局替换**原有算子名**为**新算子名**（同时处理文件名与内容）。
   - 修改文件名（如 `old_name.py` -> `new_name.py`）
   - 修改函数名（如 `def old_name(...)` -> `def new_name(...)`）
   - 修改 `@triton.jit` 装饰的 kernel 函数名
   - 修改测试文件中的引用
   - 修改 benchmark 文件中的引用
   - 修改 README 中的引用
   - 修改 autograd.Function 子类名（如有）
4. 输出改动说明与假设。

## 重命名范围

### 需要重命名的文件

- `{old_name}.py` -> `{new_name}.py`
- `test_{old_name}.py` -> `test_{new_name}.py`
- `benchmark_{old_name}.py` -> `benchmark_{new_name}.py`

### 需要重命名的内容

#### 1. Kernel 函数名

```python
# 修改前
@triton.jit
def old_name_kernel(...):
    ...

# 修改后
@triton.jit
def new_name_kernel(...):
    ...
```

#### 2. Wrapper 函数名

```python
# 修改前
def old_name(x, y):
    ...
    old_name_kernel[grid](...)
    return output

# 修改后
def new_name(x, y):
    ...
    new_name_kernel[grid](...)
    return output
```

#### 3. autograd.Function 类名（如有）

```python
# 修改前
class _old_name(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ...):
        ...

old_name = _old_name.apply

# 修改后
class _new_name(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ...):
        ...

new_name = _new_name.apply
```

#### 4. 测试类名和方法名

```python
# 修改前
class TestOldName:
    def test_old_name_basic(self):
        ...

# 修改后
class TestNewName:
    def test_new_name_basic(self):
        ...
```

#### 5. Benchmark 中的引用

```python
# 修改前
from old_name import old_name_kernel, old_name

# 修改后
from new_name import new_name_kernel, new_name
```

#### 6. 文档中的引用

- README 文件中的算子名称
- 注释中的函数名引用
- 使用说明中的示例代码

## 重命名检查清单

在执行重命名后，检查以下所有项：

- [ ] 文件名已更新
- [ ] `@triton.jit` kernel 函数名已更新
- [ ] wrapper 函数名已更新
- [ ] kernel 调用处（`kernel_name[grid]`）已更新
- [ ] 测试文件中的 import 和调用已更新
- [ ] 测试类名已更新
- [ ] benchmark 文件中的 import 和调用已更新
- [ ] autograd 类名已更新（如有）
- [ ] 文档引用已更新（如有）
- [ ] 代码注释中的名称引用已更新

## 约束

- 只涉及到名称变更，不进行其他修改。
- 重命名后需要确保所有引用都已更新。
- 重命名后需要确保测试仍然能够运行。
- 注意 Triton kernel 名称在 `@triton.jit` 装饰后即为编译后的 kernel 名称，调用处使用 `kernel_name[grid]` 语法，需要同步更新。
- 不修改环境变量名称或 Triton 框架相关的名称。

## 示例

```
用户输入：
- 旧名称：vector_add
- 新名称：vector_sum
- 路径：projects/elementwise/vector_add

修改内容：
1. vector_add.py -> vector_sum.py
2. test_vector_add.py -> test_vector_sum.py
3. benchmark_vector_add.py -> benchmark_vector_sum.py
4. kernel 函数名：add_kernel -> sum_kernel
5. wrapper 函数名：vector_add -> vector_sum
6. 测试类名：TestVectorAdd -> TestVectorSum
7. kernel 调用：add_kernel[grid] -> sum_kernel[grid]
```

## 参考

- Triton-Ascend 编程指南: `triton-ascend/docs/zh/programming_guide.md`
- Triton-Ascend 示例: `triton-ascend/docs/zh/examples/`
