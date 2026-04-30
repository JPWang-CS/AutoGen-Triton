# Triton-Ascend Expert Agent Memory

## Project Structure

### Main Work Directory: `M:\Desktop\tmp\AgentTest\Triton\`
- `.claude/agents/` - 4 agent definitions:
  - `triton-ascend-expert.md` - Research & technical analysis (me)
  - `triton-ascend-coder.md` - Code generation
  - `triton-ascend-reviewer.md` - Review only, never writes code
  - `triton-ascend-pm.md` - Workflow orchestration
- `.claude/agent-memory/triton-ascend-expert/` - My persistent memory (this file)
- `skills/` - 10 Triton-Ascend skills:
  - `triton-op-pipeline/SKILL.md` - Entry dispatcher
  - `triton-op-vector/SKILL.md` - Vector operations
  - `triton-op-cube/SKILL.md` - Matrix multiplication (tl.dot)
  - `triton-op-fused/SKILL.md` - Fused operators (Cube+Vector)
  - `triton-op-edit/SKILL.md` - Modify existing operators
  - `triton-op-rename/SKILL.md` - Rename operators
  - `triton-op-test/SKILL.md` - Test generation
  - `triton-op-benchmark/SKILL.md` - Performance benchmarking
  - `triton-op-hardware-constraints/SKILL.md` - NPU constraint verification
  - `triton-op-tuning/SKILL.md` - Performance tuning

### Key External Knowledge: `M:\Desktop\tmp\AgentTest\triton-ascend\`
- Source code and documentation repository
- Online docs: https://triton-ascend.readthedocs.io/zh-cn/latest/index.html

## Critical Distinction: Triton-Ascend vs TileLang-Ascend

- **Triton-Ascend** uses standard Triton API: `@triton.jit`, `tl.load/store`, `tl.dot`, `tl.program_id`
- **TileLang-Ascend** uses: `T.gemm_v0`, `T.alloc_L1`, `T.alloc_ub`, `T.tile.*`, `T.Scope("C"/"V")`
- These are DIFFERENT DSLs. Do NOT mix patterns.

## Triton-Ascend Key Architecture

### Compilation Flow
```
Python Kernel -> ttir.mlir -> ttadapter.mlir -> triton_xxx_kernel.o
```
- ttir: platform-independent Triton IR
- ttadapter: Ascend-adapted IR using memref/linalg/scf dialects
- BiSheng Compiler generates NPU binary

### NPU Hardware Constraints
- UB size: 192 KB on Atlas A2 (1,572,864 bits)
- Memory alignment: 32B for vector ops, 512B for CV fusion ops
- Grid limit: max 65,535 concurrent tasks
- Physical cores: vectorcore_num (vector ops), aicore_num (tl.dot ops)
- AI Core = 1 Cube Core + 2 Vector Cores

### Key API Patterns
- `tl.dot(a, b)` maps to Cube Core automatically
- Element-wise ops map to Vector Core automatically
- `@triton.autotune` supports NPU-specific options: `multibuffer`, `enable_auto_bind_sub_block`, etc.
- Ascend extensions: `tl.sync_block_wait/set/all`, `tl.custom_op`, `tl.compile_hint`
- `tl.insert_slice`, `tl.extract_slice`, `tl.get_element` for Ascend-specific tensor operations

### Common Error Patterns
- `ub overflow`: reduce BLOCK_SIZE or use BLOCK_SIZE_SUB sub-block loop
- `coreDim can't be greater than UINT16_MAX`: increase BLOCK_SIZE or set TRITON_ALL_BLOCKS_PARALLEL=1
- Scalar fallback: avoid i64/i32 comparison in tl.where, use `.to(tl.float32)` first

### Documentation Structure (Local)
- `docs/en/programming_guide.md` - Multi-core, data transfer, data compute, autotune
- `docs/en/architecture_design_and_core_features.md` - Architecture, compiler options, Ascend extensions
- `docs/en/migration_guide/migrate_from_gpu.md` - GPU-to-NPU migration guide
- `docs/en/debug_guide/debugging.md` - Debugging guide, IR dump, interpreter mode
- `docs/en/debug_guide/profiling.md` - Performance analysis, optimization examples
- `docs/en/environment_variable_reference.md` - All env vars
- `docs/en/FAQ.md` - Common questions
- `third_party/ascend/tutorials/01-vector-add.py` - Vector add example

### Online Docs Access
- Index page works: https://triton-ascend.readthedocs.io/zh-cn/latest/
- Most sub-page URLs return 404 via webReader - may need specific URL patterns
- WebSearch also has difficulty finding triton-ascend content (rate limiting or low indexing)

## Triton-Ascend Version Info (from README)
- Current: 3.2.0, based on Triton 3.2
- CANN version: 8.5.0
- 2026 plan: upgrade to Triton 3.5
- Hardware: Atlas A2/A3 series
- Python: 3.9-3.11

## Skills Generated (Session 2026-04-27)
All 10 SKILL.md files generated based on:
1. Local docs from triton-ascend/docs/en/
2. TileLang reference skills structure (adapted from TileLang to Triton API)
3. Online docs index page
4. README.md version and architecture info

## Learning Library (Session 2026-04-29 ~ 2026-04-30)

综合官方文档 + 10 个 skills 知识库:

### Learning Entries
- `operator-development-guide.md` — 多核并行、数据搬运、Vector/GEMM/Fused 计算模式、Tiling、UB 容量规划、Fractal 对齐、CV 融合、硬件架构详情
- `operator-tuning-guide.md` — msprof profiling、benchmark 方法论、瓶颈定位、标量退化避免、编译器选项、SOC 版本、优化工作流
- `operator-testing-guide.md` — 测试模式、dtype 精度容差、Fractal 对齐测试、边界用例设计、GEMM 测试约束
- `triton-api-reference.md` — 全量 API 参考（~90个 API，18 分类），含签名、dtype 支持、Ascend 约束、文档 URL 路径
- `vector-add-optimization-case.md` — vector_add 优化实战: 物理核绑定 3.5x 提升、内存 bound 分析、UB 48KB 实用上限验证、噪音抑制技巧

### Key Knowledge Learned
1. **NPU 分核原则**: grid 必须固定为物理核数（vectorcore_num / aicore_num），核内循环处理
2. **三级 Tiling**: ncore → xblock → xblock_sub → rblock，逐层切分数据
3. **UB 容量规划**: Atlas A2 UB=192KB, 实用上限 ~48KB, multibuffer 减半
4. **标量退化**: i32/i64 比较退化为 scalar，用 `.to(tl.float32)` 转为 vector
5. **Multibuffer**: 默认开启，实现搬运/计算流水重叠，代价是 UB 减半
6. **msprof**: `msprof op` 板上采集 + `msprof op simulator` 指令仿真
7. **Pairwise 精度**: RBLOCK 越大 → r_loops 越少 → 精度越好，但 UB 占用翻倍
8. **CV 融合编译选项**: enable_auto_bind_sub_block, enable_hivm_auto_cv_balance, tile_mix_vector/cube_loop
9. **Fractal 对齐**: FP16/BF16=16x16(512B), INT8=32x16(1KB), BLOCK 必须为 fractal 倍数
10. **持久化 Kernel**: 大矩阵用 grid=(aicore_num,) 核内循环，避免 coreDim 超限
11. **CV 同步原语**: tl.sync_block_wait/set/all, 流水线名 "mte1"/"mte2"/"mte3"/"m"/"v"/"fix"
12. **SOC 架构代际**: 200x(共享Scalar) vs 220x(Cube+Vector分离)
13. **Autotune NPU 限制**: 不支持 num_warps/num_stages, 需设 TRITON_BENCH_METHOD="npu"
14. **测试容差**: fp32=1e-4, fp16/bf16=1e-3, 整数=torch.equal, GEMM=1e-2

### NPU Grid 和分核核心要点
- GPU SM: 几十~几百量级，逻辑线程自动调度
- NPU AI Core: 几十量级，物理核绑定
- 运行时并发上限: 65535，但超物理核部分按轮次下发，引入额外开销
- 推荐: grid = 物理核数，核内做细致数据分块
- 纯 Vector: vectorcore_num (48 on 910B)
- CV 融合: aicore_num (24 on 910B)，1 Cube:2 Vector 比例调用
