# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

import pytest

from flaggems_vllm.utils.tle_dispatch import get_tle_backend, use_tle


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    import os

    for name in os.environ:
        if name.startswith(("FLAGGEMS_", "FLA_")) and (
            name.endswith(("_BACKEND", "_TLE")) or name == "FLAGGEMS_FORCE_TLE"
        ):
            monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("vendor", ["thead", "nvidia", "mthreads"])
@pytest.mark.parametrize("requested", ["auto", "triton", "tle"])
def test_override_and_vendor_defaults(monkeypatch, vendor, requested):
    from flaggems_vllm import runtime

    monkeypatch.setattr(runtime.device, "vendor_name", vendor)
    monkeypatch.setenv("FLAGGEMS_FLASHMLA_DECODE_BACKEND", requested)
    assert use_tle("flashmla_decode", available=True) is (
        requested == "tle" or requested == "auto" and vendor != "thead"
    )


@pytest.mark.parametrize("available,supported", [(False, True), (True, False)])
def test_auto_falls_back_but_explicit_tle_rejects(monkeypatch, available, supported):
    assert not use_tle("chunk_gla", available=available, supported=supported)
    monkeypatch.setenv("FLAGGEMS_CHUNK_GLA_BACKEND", "tle")
    with pytest.raises(NotImplementedError, match="compiler, device or input"):
        use_tle("chunk_gla", available=available, supported=supported)
    monkeypatch.setenv("FLAGGEMS_CHUNK_GLA_BACKEND", "triton")
    assert not use_tle("chunk_gla", available=available, supported=supported)


def test_operator_override_wins_over_legacy_flags(monkeypatch):
    monkeypatch.setenv("FLA_GLA_TLE", "0")
    monkeypatch.setenv("FLAGGEMS_FORCE_TLE", "1")
    assert (
        get_tle_backend("chunk_gla", legacy_env="FLA_GLA_TLE", legacy_force=True)
        == "triton"
    )
    monkeypatch.setenv("FLAGGEMS_CHUNK_GLA_BACKEND", "auto")
    assert (
        get_tle_backend("chunk_gla", legacy_env="FLA_GLA_TLE", legacy_force=True)
        == "auto"
    )


@pytest.mark.parametrize("value", ["0", "false", "off", "no"])
def test_legacy_disable(monkeypatch, value):
    monkeypatch.setenv("FLA_GLA_TLE", value)
    assert not use_tle("chunk_gla", available=True, legacy_env="FLA_GLA_TLE")


def test_legacy_enable_retains_per_operator_defaults(monkeypatch):
    from flaggems_vllm import runtime

    monkeypatch.setattr(runtime.device, "vendor_name", "thead")
    monkeypatch.setenv("FLAGGEMS_FLASHMLA_DECODE_TLE", "1")
    assert not use_tle(
        "flashmla_decode", available=True, legacy_env="FLAGGEMS_FLASHMLA_DECODE_TLE"
    )
    monkeypatch.setenv("FLAGGEMS_FORCE_TLE", "1")
    assert use_tle(
        "flashmla_decode",
        available=True,
        legacy_env="FLAGGEMS_FLASHMLA_DECODE_TLE",
        legacy_force=True,
    )


def test_legacy_global_flag_is_not_a_new_global_switch(monkeypatch):
    from flaggems_vllm import runtime

    monkeypatch.setattr(runtime.device, "vendor_name", "nvidia")
    monkeypatch.setenv("FLAGGEMS_FORCE_TLE", "0")
    assert use_tle("flashmla_decode", available=True, legacy_force=True)
    monkeypatch.setenv("FLAGGEMS_FORCE_TLE", "1")
    assert get_tle_backend("chunk_gla") == "auto"


@pytest.mark.parametrize("value", ["", "1", "enabled", "strict_tle"])
def test_invalid_override_is_rejected(monkeypatch, value):
    monkeypatch.setenv("FLAGGEMS_CHUNK_GLA_BACKEND", value)
    with pytest.raises(ValueError, match="expected auto, triton or tle"):
        use_tle("chunk_gla", available=False)


def test_unsupported_primitives_are_not_attempted(monkeypatch):
    from flaggems_vllm.utils import tle_dispatch

    monkeypatch.setattr(tle_dispatch, "supports_tle_primitives", lambda names: False)
    assert not use_tle(
        "moe_align_block_size", available=True, primitives=("device_mesh",)
    )
    monkeypatch.setenv("FLAGGEMS_MOE_ALIGN_BLOCK_SIZE_BACKEND", "tle")
    with pytest.raises(NotImplementedError):
        use_tle("moe_align_block_size", available=True, primitives=("device_mesh",))


@pytest.mark.parametrize("phase", ["decode", "prefill"])
def test_topk_override_is_read_at_call_time(monkeypatch, phase):
    from importlib import import_module

    module = import_module(f"flaggems_vllm.ops.top_k_per_row_{phase}")
    monkeypatch.setattr(module, "HAS_TLE", True)

    class Selected(Exception):
        pass

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                raise Selected("tle")

            return launch

    monkeypatch.setattr(module, f"tle_top_k_per_row_{phase}", Kernel(), raising=False)
    monkeypatch.setenv(f"FLAGGEMS_TOP_K_PER_ROW_{phase.upper()}_BACKEND", "tle")
    from types import SimpleNamespace

    logits = SimpleNamespace(shape=(1, 64))
    args = (
        (logits, 1, None, None, 1, 64, 1, 8)
        if phase == "decode"
        else (logits, None, None, None, 1, 64, 1, 8)
    )
    with pytest.raises(Selected, match="tle"):
        getattr(module, f"top_k_per_row_{phase}")(*args)


def test_sparse_ring_selection_is_independent_of_tle_backend(monkeypatch):
    from importlib import import_module

    module = import_module("flaggems_vllm.runtime.backend._thead.fused.flashmla_sparse")
    monkeypatch.setattr(module, "_has_tle_pipe", lambda: False)
    monkeypatch.setenv("FLAGGEMS_FLASHMLA_SPARSE_RING", "pipe")
    with pytest.raises(NotImplementedError, match="pipe support"):
        module._use_pipe()
    monkeypatch.setenv("FLAGGEMS_FLASHMLA_SPARSE_PIPE", "1")
    monkeypatch.setenv("FLAGGEMS_FLASHMLA_SPARSE_RING", "auto")
    assert not module._use_pipe()
    monkeypatch.setenv("FLAGGEMS_FLASHMLA_SPARSE_RING", "invalid")
    with pytest.raises(ValueError):
        module._use_pipe()


def test_explicit_auto_ignores_legacy_topk_disable(monkeypatch):
    monkeypatch.setenv("FLAGGEMS_FORCE_TLE", "0")
    assert get_tle_backend("top_k_per_row_decode", legacy_force=True) == "triton"
    monkeypatch.setenv("FLAGGEMS_TOP_K_PER_ROW_DECODE_BACKEND", "auto")
    assert get_tle_backend("top_k_per_row_decode", legacy_force=True) == "auto"


@pytest.mark.parametrize(
    "op,shape,expected",
    [
        ("parallel_nsa", {"D": 64}, False),
        ("parallel_nsa", {"D": 128}, True),
        ("chunk_gla", {"T": 1024, "D": 128}, False),
        ("chunk_gla", {"T": 4096, "D": 128}, True),
        ("chunk_kda", {"D": 64}, True),
        ("chunk_kda", {"D": 128, "T": 64}, False),
        ("chunk_kda", {"D": 128, "T": 32}, True),
        ("chunk_kda", {"D": 128, "T": 2048}, False),
        ("topk_softplus_sqrt", {"hash_mode": False}, False),
        ("topk_softplus_sqrt", {"hash_mode": True}, True),
    ],
)
def test_ppu_auto_can_choose_by_input_without_changing_other_vendors(
    monkeypatch, op, shape, expected
):
    from flaggems_vllm import runtime

    monkeypatch.setattr(runtime.device, "vendor_name", "thead")
    assert use_tle(op, available=True, **shape) is expected
    monkeypatch.setattr(runtime.device, "vendor_name", "nvidia")
    assert use_tle(op, available=True, **shape)


def test_sparse_mla_explicit_tle_rejects_unsupported_topk(monkeypatch):
    import importlib
    from types import SimpleNamespace

    module = importlib.import_module("flaggems_vllm.ops.flashmla_sparse")
    monkeypatch.setattr(module, "HAS_TLE_FLASHMLA_SPARSE", True)
    q = SimpleNamespace(device=SimpleNamespace(type="cuda"), shape=(1, 64, 576))
    kv = SimpleNamespace(shape=(512, 1, 576))
    indices = SimpleNamespace(shape=(1, 1, 127))
    assert not module._can_use_tle_flash_mla_sparse_fwd(q, kv, indices, 512)
    monkeypatch.setenv("FLAGGEMS_FLASHMLA_SPARSE_BACKEND", "tle")
    with pytest.raises(NotImplementedError):
        module._can_use_tle_flash_mla_sparse_fwd(q, kv, indices, 512)
