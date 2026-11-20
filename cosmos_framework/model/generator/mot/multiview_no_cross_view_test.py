# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""The ablation must remove cross-camera data dependencies and kernel launches."""

from pathlib import Path

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing.runtime import sequence_pack_from_packed_sequence
from cosmos_framework.model.generator.mot import multiview_maskless_attention as dense
from cosmos_framework.utils.step_timing import diffusion_step, record_diffusion_steps

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _attention(q, k, v, *, cumulative_seqlen_Q, cumulative_seqlen_KV, **kwargs):
    output = torch.zeros_like(q)
    lse = torch.full(q.shape[:-1], -torch.inf, dtype=q.dtype)
    for qs, qe, ks, ke in zip(
        cumulative_seqlen_Q[:-1],
        cumulative_seqlen_Q[1:],
        cumulative_seqlen_KV[:-1],
        cumulative_seqlen_KV[1:],
        strict=True,
    ):
        scores = torch.einsum("bqhd,bkhd->bhqk", q[:, qs:qe], k[:, ks:ke]) / q.shape[-1] ** 0.5
        output[:, qs:qe] = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v[:, ks:ke])
        lse[:, qs:qe] = scores.logsumexp(-1).transpose(1, 2)
    return output, lse


def _merge(*, outputs, lse_tensors, **kwargs):
    lse = torch.logsumexp(torch.stack(lse_tensors), dim=0)
    result = sum(output * (part - lse).exp().unsqueeze(-1) for output, part in zip(outputs, lse_tensors, strict=True))
    return result, lse


def _pack(tensor):
    return sequence_pack_from_packed_sequence(
        tensor,
        attn_modes=["causal", "full"],
        split_lens=[2, 8],
        sample_lens=[10],
        packed_und_token_indexes=torch.arange(2),
        packed_gen_token_indexes=torch.arange(2, 10),
    )


@torch.no_grad()
def test_no_cross_view_ignores_other_camera_but_retains_own_temporal_control_and_text(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(dense, "attention", _attention)
    monkeypatch.setattr(dense, "merge_attentions", _merge)

    def forbidden(*args, **kwargs):
        raise AssertionError("A cross-view kernel was launched")

    monkeypatch.setattr(dense, "_aligned_cross_view_attention", forbidden)
    # A loaded BCA module must not resurrect cross-view work in the ablation.
    monkeypatch.setattr("cosmos_framework.model.generator.mot.bca.active_bca", lambda: forbidden)
    plan = dense.build_multiview_maskless_plan(
        [2, 2],
        [(4, 1, 1), (4, 1, 1)],
        device=torch.device("cpu"),
        items_per_sample=[2],
        is_control=[False, True],
        control_attends_sensor=True,
        attention_scope="same_view",
        captions=[[(0, 1), (1, 1)]],
        padded_gen_tokens=9,  # SequencePack appends its one-token GEN pad segment.
    )
    assert plan.cross_view_empty
    torch.manual_seed(0)
    q, k, v = [torch.randn(10, 2, 4, dtype=torch.float64) for _ in range(3)]

    def run(values):
        return dense.multiview_maskless_gen_attention(_pack(q), _pack(k), _pack(values), plan=plan)

    path = tmp_path / "same-view.csv"
    with record_diffusion_steps(path, {}, attention=True) as recorder:
        with diffusion_step(0, 1):
            baseline = run(v)
    # GEN order: camera-0 frames, camera-1 frames, then each camera's controls.
    own = [0, 1, 4, 5]
    changed = v.clone()
    changed[[1, 4, 5, 8, 9]] += 50  # Other camera's caption, RGB and WSM values.
    torch.testing.assert_close(run(changed)[own], baseline[own], rtol=0, atol=0)
    for token in (0, 3, 6):  # Own caption, other own-camera frame, own WSM.
        changed = v.clone()
        changed[token] += 5
        assert not torch.allclose(run(changed)[0], baseline[0])
    regions = recorder.steps[0]["regions"]
    assert regions["cross_view"]["calls"] == 0
    assert regions["same_view"]["calls"] == regions["caption"]["calls"] == 1
