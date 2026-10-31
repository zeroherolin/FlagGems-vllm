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
    """Select Hopper TLE pipe kernels, keeping faster Triton paths on T-Head.

    FLAGGEMS_FORCE_TLE=1 overrides the vendor choice. False values leave other
    vendors' defaults intact; they only disable TLE in the separate top-k gate.
    """
    if os.environ.get("FLAGGEMS_FORCE_TLE", "").lower() in {"1", "true", "on", "yes"}:
        return True
    try:
        from flaggems_vllm import runtime

        return runtime.device.vendor_name != "thead"
    except Exception:  # noqa: BLE001 - never let detection break the import
        return True
