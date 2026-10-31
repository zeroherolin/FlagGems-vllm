# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Optional API contracts used by TLE kernels, without modifying the runtime."""

import os


def _has_gpu_apis(language, names):
    gpu = getattr(language, "gpu", None)
    return (
        gpu is not None
        and hasattr(gpu, "smem")
        and all(callable(getattr(gpu, name, None)) for name in names)
    )


def supports_topk_tle(language):
    return callable(getattr(language, "cumsum", None)) and _has_gpu_apis(
        language, ("alloc", "local_ptr")
    )


def supports_sparse_mla_tle(language):
    return callable(getattr(language, "pipe", None)) and _has_gpu_apis(
        language, ("alloc", "local_ptr", "copy", "warp_specialize")
    )


def vendor_takes_tle_pipe_kernels():
    """Whether this vendor takes the Hopper TLE kernels built on tle.pipe.

    T-Head PPU compiles them, but its own Triton paths are faster for dense
    decode, gated delta rule and recompute, so it stays on those.
    FLAGGEMS_FORCE_TLE=1 selects the TLE kernels anyway; any other value leaves
    the choice to the vendor, so the setting that switches off the top-k TLE
    kernels does not switch off these on other vendors.
    """
    if os.environ.get("FLAGGEMS_FORCE_TLE", "").lower() in {"1", "true", "on", "yes"}:
        return True
    try:
        from flaggems_vllm import runtime

        return runtime.device.vendor_name != "thead"
    except Exception:  # noqa: BLE001 - never let detection break the import
        return True
