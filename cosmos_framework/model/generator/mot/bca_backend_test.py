# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.model.generator.mot import bca_attention as backend

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture(autouse=True)
def clear_cache():
    backend._inference_flash_backend.cache_clear()
    yield
    backend._inference_flash_backend.cache_clear()


@pytest.mark.parametrize("nvidia_available", [True, False])
def test_hopper_prefers_installed_flash3_and_normalizes_lse(monkeypatch, nvidia_available):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    monkeypatch.setenv("COSMOS_BCA_FLASH_BACKEND", "flash3")
    attempts, calls = [], []

    def function(q, k, v, softmax_scale, causal, return_attn_probs=False):
        calls.append((softmax_scale, causal, return_attn_probs))
        return q + 2, torch.ones(q.shape[0], q.shape[2], q.shape[1]), None

    def load(name):
        attempts.append(name)
        if not nvidia_available and "_nv" in name:
            raise ImportError("package absent")
        return SimpleNamespace(flash_attn_func=function)

    monkeypatch.setattr(backend.importlib, "import_module", load)
    q = torch.randn(2, 3, 4, 32)
    out, lse = backend.inference_attention_with_lse(q, q, q)
    torch.testing.assert_close(out, q + 2)
    assert lse.shape == q.shape[:-1]
    assert calls == [(32**-0.5, False, True)]
    assert attempts[0] == "flash_attn_3_nv.flash_attn_interface"
    assert len(attempts) == (1 if nvidia_available else 2)


def test_forced_flash3_does_not_silently_fall_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))

    def missing(name):
        raise ImportError("package absent")

    monkeypatch.setattr(backend.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="could not load Hopper FlashAttention-3"):
        backend._inference_flash_backend(torch.device("cuda:0"), "flash3")
    with pytest.warns(RuntimeWarning, match="Falling back"):
        assert backend._inference_flash_backend(torch.device("cuda:0"), "auto")[0] == "aten"


def test_non_hopper_does_not_load_flash3(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 6))

    def forbidden(name):
        pytest.fail("FA3 import attempted on Ampere")

    monkeypatch.setattr(backend.importlib, "import_module", forbidden)
    assert backend._inference_flash_backend(torch.device("cuda:0"), "auto")[0] == "aten"
    with pytest.raises(RuntimeError, match="requires a Hopper"):
        backend._inference_flash_backend(torch.device("cuda:0"), "flash3")
