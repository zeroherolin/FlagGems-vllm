# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""PPU defaults for operators with interchangeable TLE and Triton paths."""

TLE_DEFAULTS = {
    "flashmla_sparse": True,
    "flashmla_decode": False,
    "chunk_gated_delta_rule": False,
    "chunk_gdr_recompute": False,
    "top_k_per_row_decode": False,
    "top_k_per_row_prefill": False,
    "grouped_topk": False,
    "sparse_mla": True,
    "bucket_sort_topk": True,
    "moe_align_block_size": False,
    "flash_attn_varlen_int8": True,
}


def prefer_tle(op, *, default=True, **shape):
    if op == "chunk_gla":
        return shape.get("T", 0) >= 4096 and shape.get("D", 0) > 64
    if op == "parallel_nsa":
        return shape.get("D", 0) > 64
    if op == "chunk_kda":
        return shape.get("D", 128) <= 64 or shape.get("T", 33) <= 32
    if op == "topk_softplus_sqrt":
        # Shared staging pays off for indexed hash gathers, not a full row scan.
        return shape.get("hash_mode", False)
    return TLE_DEFAULTS.get(op, default)
