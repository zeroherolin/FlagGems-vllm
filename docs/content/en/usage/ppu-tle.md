---
title: PPU TLE implementation selection
---

PPU implementation defaults are centralized in `runtime/backend/_thead/tle_config.py`. No environment variables are needed for normal execution. Availability, valid inputs and measured performance are separate decisions.

Operators with interchangeable implementations accept `FLAGGEMS_<OP>_BACKEND=auto|triton|tle`. `auto` applies the PPU policy and falls back when optional capabilities are absent. `triton` selects the ordinary Triton implementation. Existing input validation remains in effect. Explicit `tle` raises `NotImplementedError` when the requested candidate is unavailable. Invalid values raise `ValueError`. PPU dispatch does not catch kernel execution errors to silently change implementations.

Supported OP names: `FLASHMLA_DECODE`, `FLASHMLA_SPARSE`, `CHUNK_GATED_DELTA_RULE`, `CHUNK_GDR_RECOMPUTE`, `CHUNK_GLA`, `PARALLEL_NSA`, `CHUNK_KDA`, `TOP_K_PER_ROW_DECODE`, `TOP_K_PER_ROW_PREFILL`, `GROUPED_TOPK`, `TOPK_SOFTPLUS_SQRT`, `SPARSE_MLA` (DSA), `BUCKET_SORT_TOPK`, `MOE_ALIGN_BLOCK_SIZE`, and `FLASH_ATTN_VARLEN_INT8` (eligible asynchronous K loads).

| Operation | PPU automatic policy |
| --- | --- |
| Sparse MLA prefill | Warp-specialized TLE, inline-assembly KV ring |
| Dense MLA decode, GDR, recompute | Ordinary Triton |
| chunk_gla | TLE for sequence length ≥4096 and K dimension >64; Triton otherwise |
| parallel_nsa | TLE for K dimension >64; Triton otherwise |
| PPU chunk_kda | TLE for D≤64 or total token count ≤32; Triton otherwise |
| topk_softplus_sqrt | TLE for hash gathers; Triton for ordinary selection |
| DSA sparse_mla_fwd_interface, bucket_sort_topk | TLE |
| Per-row TopK, grouped TopK, MoE align | Ordinary Triton |
| INT8 attention | Optional asynchronous loads on eligible paged GQA tiles |

Per-operator overrides, including explicit `auto`, take precedence over legacy settings. Existing `*_TLE=0/false/off/no` flags disable the corresponding path; enable values allow automatic selection. `FLA_GLA_TLE` and `FLA_NSA_TLE` remain aliases. `FLAGGEMS_FORCE_TLE` retains only its existing diagnostic consumers (dense decode, GDR, recompute and per-row TopK); it is not a library-wide switch. `CHUNK_KDA` retains the `generic_tle` / `triton_fuse` aliases. See the Chinese PPU guide for the complete alias mapping.

The PPU sparse MLA KV-ring variant is independent: `FLAGGEMS_FLASHMLA_SPARSE_RING=auto|inline_asm|pipe`. Both rings use TLE. Automatic selection uses inline assembly; explicit pipe selection requires compiler support. The old `FLAGGEMS_FLASHMLA_SPARSE_PIPE` remains an alias when the new variable is absent.

PPU persistent_topk has no interchangeable ordinary-Triton implementation; no override is added. Existing attn_res tuning candidates, AIU shape specializations and architecture-specific WGMMA/TMA guards remain in place. Other vendors retain their own automatic defaults.

These policies select measured implementations over validated input classes on PPU-ZW810E, not a proof of a global optimum over every possible input. Input classes without consistent gains use ordinary Triton; stable cross-shape differences use input-based routing. Known per-row TLE TopK stability failures also reproduce with the unmodified operator library and baseline compiler, so that path is excluded from PPU automatic selection.

The paired timing set has 70 inputs: six GLA, six NSA, twenty KDA, twenty softplus, four bucket TopK, three grouped TopK, three DSA sparse MLA and eight INT8 attention. Outputs are compared before timing on an idle device, using the median of five 40 ms CUDA-graph runs. A fixed GEMM is checked before and after timing with a 3% acceptance band to detect device interference; this is not a confidence interval for operator latency. The D=64 KDA class benefits from TLE overall; the dense T=2048 case has a measured difference of about 1% and retains the same class policy. For D=128, T=32 favors TLE while T=64/128/256 and longer sequences favor the ordinary implementation.

Primitive support and limits, raw timings and baseline-failure comparisons are documented in [PPU compiler optimization and related operators](https://jwolpxeehx.feishu.cn/docx/WB3IdPhdDonaOPxXKZgcdiLNnMh) and its validation attachment. Other devices and input classes require their own validation and calibration.
