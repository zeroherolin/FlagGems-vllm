---
title: PPU TLE 实现选择
---

PPU 算子的默认实现由 `runtime/backend/_thead/tle_config.py` 集中选择。正常运行无需设置环境变量。选择依据为正确性、设备能力、输入条件和 PPU-ZW810E 上的受控性能测试；TLE 可用不代表 TLE 更快。

存在两条独立实现的算子使用相同的覆盖形式：

```bash
FLAGGEMS_<OP>_BACKEND=auto   # 默认：使用后端的性能策略
FLAGGEMS_<OP>_BACKEND=triton # 普通 Triton 实现
FLAGGEMS_<OP>_BACKEND=tle    # TLE 实现，编译器或输入不支持时明确报错
```

选择发生在 host dispatch；不会捕获 PPU kernel 执行错误并静默切换实现。无效设置会报 `ValueError`。`auto` 仅在能力和输入条件满足时采用 TLE，否则使用普通 Triton。

| OP | 旧变量（兼容） |
| --- | --- |
| `FLASHMLA_DECODE` | `FLAGGEMS_FLASHMLA_DECODE_TLE` |
| `FLASHMLA_SPARSE` | `FLAGGEMS_FLASHMLA_SPARSE_TLE` |
| `CHUNK_GATED_DELTA_RULE` | `FLAGGEMS_CHUNK_GATED_DELTA_RULE_TLE` |
| `CHUNK_GDR_RECOMPUTE` | `FLAGGEMS_CHUNK_GDR_RECOMPUTE_TLE` |
| `CHUNK_GLA` | `FLA_GLA_TLE` |
| `PARALLEL_NSA` | `FLA_NSA_TLE` |
| `CHUNK_KDA` | 已使用相同的 `FLAGGEMS_CHUNK_KDA_BACKEND` |
| `TOP_K_PER_ROW_DECODE` / `TOP_K_PER_ROW_PREFILL` | `FLAGGEMS_FORCE_TLE` |
| `GROUPED_TOPK` | 无 |
| `TOPK_SOFTPLUS_SQRT` | 无 |
| `SPARSE_MLA` | 无；指 DSA 的 `sparse_mla_fwd_interface` |
| `BUCKET_SORT_TOPK` | 无 |
| `MOE_ALIGN_BLOCK_SIZE` | 无 |
| `FLASH_ATTN_VARLEN_INT8` | 无；控制满足条件的 INT8 K 异步加载 |

新的单算子变量优先于旧变量，即使设置为 `auto` 也是如此。旧 `*_TLE=0/false/off/no` 禁用该路径；旧启用值表示允许自动选择，不保证一定使用 TLE。旧 `FLAGGEMS_FORCE_TLE=1/true/on/yes` 仅兼容原有 dense decode、GDR、recompute 和 per-row TopK 调试入口，不是全库开关；`0` 也不是全库禁用。新脚本应使用单算子变量。`CHUNK_KDA` 保留 `generic_tle` 和 `triton_fuse` 别名。

Sparse MLA 的 TLE 路径还有两种 KV 环实现，使用独立选项：

```bash
FLAGGEMS_FLASHMLA_SPARSE_RING=auto       # 内联汇编
FLAGGEMS_FLASHMLA_SPARSE_RING=inline_asm
FLAGGEMS_FLASHMLA_SPARSE_RING=pipe
```

`pipe` 需要编译器支持对应 TLE 原语；显式选择不满足条件时报错。旧 `FLAGGEMS_FLASHMLA_SPARSE_PIPE` 保留兼容，新变量优先。两种环都属于 TLE 实现，不能用 `BACKEND` 来区分。

`persistent_topk` 的 PPU 专用共享内存实现和仅有单一路径的算子不增加虚假的备选开关。`attn_res` 的异步加载属于已有 autotune 候选，由原调优器选择。FP8 WGMMA/TMA 等依赖其他架构的路径保留原硬件限制。厂商的 `tle_enabled` 和编译器原语支持表不作为全算子性能开关。

性能策略只作用于 PPU；其他厂商的默认实现和调优配置保持各自规则。测试和基准应显式固定待比较路径及输入，记录编译器、设备和缓存条件。

PPU-ZW810E 的自动选择如下；能力和输入合法性检查始终优先：

| 路径 | 自动选择 |
| --- | --- |
| sparse MLA prefill | TLE warp specialization，内联汇编 KV 环 |
| dense MLA decode、GDR、recompute | 普通 Triton |
| `chunk_gla` | 序列长度至少 4096 且 K 维大于 64 时用 TLE，否则普通 Triton |
| `parallel_nsa` | K 维大于 64 时用 TLE，否则普通 Triton |
| PPU `chunk_kda` | D 不超过 64 或总 token 数不超过 32 时用 TLE，否则普通 Triton |
| `topk_softplus_sqrt` | hash gather 用 TLE，普通选择用 Triton |
| DSA `sparse_mla_fwd_interface`、`bucket_sort_topk` | TLE |
| per-row TopK、grouped TopK、MoE align | 普通 Triton |
| INT8 attention 的 K 加载 | 按后端条件选择异步加载；只影响符合条件的 paged GQA tile |

TLE per-row TopK 的稳定性测试在原始算子库和基线编译器上也失败，故不纳入 PPU 自动选择。MoE align 的协作 TLE 路径需要 PPU 不支持的 `device_mesh`，自动选择不尝试该实现。单一 TLE 共享内存实现、原有 AIU 形状优化与 autotune 继续使用已有策略。

默认策略是对已验证输入范围的性能选择，不代表对所有可能形状的全局最优证明。没有稳定收益的输入类别使用普通 Triton；跨形状存在稳定差异时按输入路由。

性能对照覆盖 GLA 的 6 个形状、NSA 的 6 个形状、KDA 的 20 个 dense / varlen 输入、softplus 的 20 个 hash / 非 hash 输入，以及 grouped TopK、DSA、INT8 attention 的代表负载。使用同输入两路数值检查、空闲设备、GEMM 校准和 CUDA graph 5×40 ms 中位数计时。KDA 的 D=64 类别总体支持 TLE；其中 T=2048 的 dense 输入差异约 1%，未显示显著回归。D=128 的短序列 T=32 使用 TLE，T=64/128/256 及更长序列使用普通实现。

受控计时及已有失败的基线对照见[PPU 技术分析](https://jwolpxeehx.feishu.cn/docx/DiTTdZq1poyI2hxpUDncUDe9n37)的验证附件。无 TLE、不同硬件或不同输入类别应重新验证和标定，不能套用本设备的性能结论。
