# Triton-Ascend Expert Agent Memory

## Project Structure

### Main Work Directory: `M:\Desktop\tmp\AgentTest\Triton\`
- `.claude/agents/` - 4 agent definitions:
  - `triton-ascend-expert.md` - Research & technical analysis (me)
  - `triton-ascend-coder.md` - Code generation
  - `triton-ascend-reviewer.md` - Review only, never writes code
  - `triton-ascend-pm.md` - Workflow orchestration
- `.claude/agent-memory/triton-ascend-expert/` - My persistent memory (this file)
- `skills/` - 4 Triton-Ascend skills (generated in this session):
  - `triton-op-pipeline/SKILL.md` - Entry dispatcher
  - `triton-op-cube/SKILL.md` - Matrix multiplication (tl.dot)
  - `triton-op-vector/SKILL.md` - Vector operations
  - `triton-op-fused/SKILL.md` - Fused operators

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
All 4 SKILL.md files were generated based on:
1. Local docs from triton-ascend/docs/en/
2. TileLang reference skills structure (adapted from TileLang to Triton API)
3. Online docs index page
4. README.md version and architecture info
