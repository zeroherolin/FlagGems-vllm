# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Host-side selection of optional TLE implementations."""

import os
from functools import lru_cache


@lru_cache(maxsize=None)
def supports_tle_primitives(primitives):
    try:
        from triton.experimental.tle import is_tle_supported
    except ImportError:
        return False
    return is_tle_supported(primitives)


def get_tle_backend(op, *, legacy_env=None, legacy_force=False):
    """Read the per-operator override; old enable flags retain auto semantics."""
    name = f"FLAGGEMS_{op.upper()}_BACKEND"
    value = os.environ.get(name, "auto").strip().lower()
    if op == "chunk_kda":
        value = {"generic_tle": "tle", "triton_fuse": "triton"}.get(value, value)
    if value not in {"auto", "triton", "tle"}:
        raise ValueError(f"invalid {name}={value!r}; expected auto, triton or tle")
    if name in os.environ:
        return value
    if legacy_env and legacy_env in os.environ:
        value = os.environ[legacy_env].strip().lower()
        if value in {"0", "false", "off", "no"}:
            return "triton"
    if legacy_force:
        value = os.environ.get("FLAGGEMS_FORCE_TLE", "").strip().lower()
        if value in {"1", "true", "on", "yes"}:
            return "tle"
        if op.startswith("top_k_per_row_") and value in {"0", "false", "off", "no"}:
            return "triton"
    return "auto"


def use_tle(
    op,
    *,
    available,
    supported=True,
    default=True,
    legacy_env=None,
    legacy_force=False,
    primitives=(),
    **shape,
):
    """Choose a valid implementation without catching kernel execution errors."""
    backend = get_tle_backend(op, legacy_env=legacy_env, legacy_force=legacy_force)
    if backend == "triton":
        return False
    if available and primitives:
        available = supports_tle_primitives(tuple(primitives))
    if backend == "tle":
        if not available or not supported:
            if f"FLAGGEMS_{op.upper()}_BACKEND" not in os.environ:
                return False
            raise NotImplementedError(
                f"FLAGGEMS_{op.upper()}_BACKEND=tle is unavailable for this "
                "compiler, device or input"
            )
        return True
    if not available or not supported:
        return False

    from flaggems_vllm import runtime

    if runtime.device.vendor_name == "thead":
        from flaggems_vllm.runtime.backend._thead.tle_config import prefer_tle

        return prefer_tle(op, default=default, **shape)
    return default
