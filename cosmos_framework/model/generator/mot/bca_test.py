# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from cosmos_framework.model.generator.mot.bca import (
    BudgetCrossView,
    DifferentiablePartitionBridge,
    active_bca,
    bca_layer,
    dense_teacher,
    importance_indices,
    record_plan,
)
from cosmos_framework.model.generator.mot.bca_distillation import pool_block_features, teacher_student_forward
from cosmos_framework.model.generator.mot.bca_geometry import local_routes
from cosmos_framework.model.generator.mot.merge_attention import merge_attentions_ac_safe
from cosmos_framework.utils.generator.lora import LoraInjectedLinear

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def config():
    calibration = json.loads(
        (Path(__file__).resolve().parents[3] / "configs/simulation/assets/bca_daytime_calibration.json").read_text()
    )
    cameras = list(calibration["cameras"])[:2]
    return SimpleNamespace(
        bca_calibration_json=json.dumps(calibration),
        bca_camera_keys=cameras,
        bca_local_budget=3,
        bca_global_budget=4,
        bca_global_pool=2,
        bca_inference_ratio=0.35,
        bca_query_chunk=3,
        bca_min_depth=1.0,
        bca_max_depth=120.0,
        bca_depth_samples=8,
    )


def plan():
    return SimpleNamespace(
        token_shapes=((4, 3, 4), (4, 3, 4)),
        num_views=(2, 2),
        view_axis=(0, 0),
        items_per_sample=(2,),
        is_control=(True, False),
        num_gen_tokens=96,
        padded_gen_tokens=96,
        seconds_per_frame=(4 / 30, 4 / 30),
        cross_view_temporal_window=SimpleNamespace(bounds=(-0.4, 0.0), includes_first_frame=False),
    )


def test_importance_retains_representatives_and_outliers_without_changing_features():
    values = torch.zeros((64, 2, 4))
    # In one representative cell at most one of these can itself be the
    # representative, so the remaining large residuals must also be retained.
    values[0, :, 0] = 100
    values[1, :, 0] = -100
    original = values.clone()
    indices = importance_indices(values, (4, 4, 4), 0.35, seed=3)
    assert indices.shape == (2, 23)
    assert all({0, 1} <= set(head.tolist()) for head in indices)
    assert torch.equal(values, original)
    assert torch.equal(indices, importance_indices(values, (4, 4, 4), 0.35, seed=3))
    assert len(indices[0].unique()) == 23


def test_calibrated_routes_are_bounded_deduplicated_and_cross_view_only():
    cfg = config()
    routes, valid = local_routes(cfg.bca_calibration_json, tuple(cfg.bca_camera_keys), 3, 4, 32)
    assert routes.shape == valid.shape == (2, 12, 32)
    for view in range(2):
        assert not (routes[view][valid[view]] // 12 == view).any()
        for row, mask in zip(routes[view], valid[view], strict=True):
            assert len(row[mask].unique()) == int(mask.sum())


def test_reusable_source_registers_match_joint_neighbor_softmax():
    operator = BudgetCrossView(4, config(), 10)
    key, value = [torch.randn(2, 7, 5, 2, 4) for _ in range(2)]
    source_k, source_v, lse = operator._source_registers(key, value)
    neighbors = [0, 2, 6]
    weight = lse[neighbors].softmax(0)[..., None]
    actual_k, actual_v = (weight * source_k[neighbors]).sum(0), (weight * source_v[neighbors]).sum(0)
    k, v = key[:, neighbors].reshape(-1, 2, 4), value[:, neighbors].reshape(-1, 2, 4)
    logits = torch.einsum("rhd,khd->hrk", operator.bca_register_queries, k) / 2
    expected_k = torch.einsum("hrk,khd->rhd", logits.softmax(-1), k)
    expected_v = torch.einsum("hrk,khd->rhd", logits.softmax(-1), v)
    torch.testing.assert_close(actual_k, expected_k)
    torch.testing.assert_close(actual_v, expected_v)


def test_bca_forward_backward_and_inference_residual_selection():
    operator = BudgetCrossView(4, config(), 10)
    query, key, value = [torch.randn(96, 2, 4, requires_grad=True) for _ in range(3)]
    # This call is isolated, so the bridge receives its own O/LSE unchanged.
    output, lse = operator(query, key, value, plan())
    assert output.shape == (1, 96, 2, 4) and lse.shape == (1, 96, 2)
    assert torch.isfinite(output).all() and torch.isfinite(lse).all()
    output.square().mean().backward()
    assert operator.bca_register_queries.grad is not None
    assert operator.bca_register_queries.grad.abs().sum() > 0
    operator.eval()
    with torch.no_grad():
        inference_output, _ = operator(query, key, value, plan())
    assert torch.isfinite(inference_output).all()
    assert not torch.allclose(output, inference_output)


@pytest.mark.parametrize("checkpointed", [False, True])
def test_storage_patch_bridge_matches_joint_softmax_gradients(checkpointed):
    torch.manual_seed(1)
    q = torch.randn(1, 3, 2, 4, requires_grad=True)
    k = torch.randn(1, 5, 2, 4, requires_grad=True)
    v = torch.randn(1, 5, 2, 4, requires_grad=True)
    other_logits = torch.randn(1, 3, 2, 7)
    other_values = torch.randn(1, 7, 2, 4)
    other_out = torch.einsum("bqhk,bkhd->bqhd", other_logits.softmax(-1), other_values)

    def forward(query, key, value):
        logits = torch.einsum("bqhd,bkhd->bqhk", query, key) / 2
        output = torch.einsum("bqhk,bkhd->bqhd", logits.softmax(-1), value)
        bridged, lse = DifferentiablePartitionBridge.apply(output, logits.logsumexp(-1))
        return merge_attentions_ac_safe([other_out.clone(), bridged], [other_logits.logsumexp(-1), lse])[0]

    if checkpointed:
        from torch.utils.checkpoint import checkpoint

        actual = checkpoint(forward, q, k, v, use_reentrant=False)
    else:
        actual = forward(q, k, v)
    logits = torch.einsum("bqhd,bkhd->bqhk", q, k) / 2
    joint_logits = torch.cat((other_logits, logits), -1)
    reference = torch.einsum("bqhk,bkhd->bqhd", joint_logits.softmax(-1), torch.cat((other_values, v), 1))
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-6)
    actual_grad = torch.autograd.grad(actual.square().sum(), (q, k, v), retain_graph=True)
    expected_grad = torch.autograd.grad(reference.square().sum(), (q, k, v))
    for observed, expected in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(observed, expected, atol=3e-6, rtol=3e-6)


def test_dense_teacher_disables_and_restores_student_lora():
    layer = LoraInjectedLinear(torch.nn.Linear(4, 4), 2, 2)
    with torch.no_grad():
        layer.lora_A.weight.fill_(0.5)
        layer.lora_B.weight.fill_(0.5)
    x = torch.randn(3, 4)
    student = layer(x)
    with dense_teacher():
        torch.testing.assert_close(layer(x), F.linear(x, layer.weight, layer.bias))
    torch.testing.assert_close(layer(x), student)


def test_block_feature_pool_preserves_frames_and_excludes_controls():
    features = torch.arange(96 * 2, dtype=torch.float32).reshape(96, 2)
    pooled = pool_block_features(features, plan(), 2)
    target = features[48:].reshape(4, 3, 4, 2).permute(0, 3, 1, 2)
    torch.testing.assert_close(pooled, F.avg_pool2d(target, 2, 2, ceil_mode=True).flatten())


@torch.no_grad()
def test_budget_sources_cannot_leak_future_camera_values():
    operator = BudgetCrossView(4, config(), 10).eval()
    q, k, v = [torch.randn(96, 2, 4) for _ in range(3)]
    baseline, _ = operator(q, k, v, plan())
    changed = v.reshape(2, 2, 2, 12, 2, 4).clone()
    changed[:, :, 1] += 100
    actual, _ = operator(q, k, changed.reshape_as(v), plan())
    first = torch.arange(96).reshape(2, 2, 2, 12)[:, :, 0].flatten()
    torch.testing.assert_close(actual[:, first], baseline[:, first], rtol=0, atol=0)
    assert not torch.allclose(actual, baseline)


@torch.no_grad()
def test_native_maskless_dispatch_replaces_only_cross_view_partition(monkeypatch):
    from cosmos_framework.data.generator.sequence_packing.runtime import sequence_pack_from_packed_sequence
    from cosmos_framework.model.generator.mot import multiview_maskless_attention as native
    from cosmos_framework.model.generator.mot.multiview_no_cross_view_test import _attention, _merge

    monkeypatch.setattr(native, "attention", _attention)
    monkeypatch.setattr(native, "merge_attentions", _merge)

    def forbidden(*args, **kwargs):
        raise AssertionError("Dense cross-view attention was invoked for the budget student")

    monkeypatch.setattr(native, "_aligned_cross_view_attention", forbidden)
    camera_plan = native.build_multiview_maskless_plan(
        [2, 2],
        [(4, 3, 4), (4, 3, 4)],
        device=torch.device("cpu"),
        items_per_sample=[2],
        is_control=[True, False],
        control_attends_sensor=True,
        attention_scope="decomposed",
        seconds_per_frame=[4 / 30, 4 / 30],
        padded_gen_tokens=97,
    )

    def pack(tensor):
        return sequence_pack_from_packed_sequence(
            tensor,
            attn_modes=["causal", "full"],
            split_lens=[2, 96],
            sample_lens=[98],
            packed_und_token_indexes=torch.arange(2),
            packed_gen_token_indexes=torch.arange(2, 98),
        )

    operator = BudgetCrossView(4, config(), 10).eval()
    tensors = [pack(torch.randn(98, 2, 4)) for _ in range(3)]
    with bca_layer(operator):
        actual = native.multiview_maskless_gen_attention(*tensors, plan=camera_plan)
    assert actual.shape == (97, 8) and torch.isfinite(actual).all()
    assert operator.last_plan is camera_plan


def test_ten_teacher_student_updates_preserve_teacher_and_train_lora_and_registers():
    """Exercise the actual distillation helper/merge backward on a tiny CPU network."""

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = torch.nn.Module()
            self.self_attn.bca_attention = BudgetCrossView(4, config(), 10)
            self.projections = torch.nn.ModuleList([LoraInjectedLinear(torch.nn.Linear(8, 8), 2, 2) for _ in range(4)])
            for layer in self.projections:
                torch.nn.init.zeros_(layer.lora_B.weight)

        def forward(self, inputs):
            q, k, v = [layer(inputs).reshape(96, 2, 4) for layer in self.projections[:3]]
            logits = torch.einsum("qhd,khd->qhk", q, k) / 2
            operator = self.self_attn.bca_attention
            with bca_layer(operator):
                record_plan(plan())
                if active_bca() is None:
                    result = torch.einsum("qhk,khd->qhd", logits.softmax(-1), v)
                else:
                    view = torch.arange(96) // 24 % 2
                    same = logits.masked_fill(view[:, None, None] != view[None, None, :], -torch.inf)
                    same_out = torch.einsum("qhk,khd->qhd", same.softmax(-1), v)
                    own_out, own_lse = DifferentiablePartitionBridge.apply(
                        same_out.unsqueeze(0), same.logsumexp(-1).unsqueeze(0)
                    )
                    cross_out, cross_lse = operator(q, k, v, plan())
                    result, _ = merge_attentions_ac_safe([own_out, cross_out], [own_lse, cross_lse])
                    result = result.squeeze(0)
            result = self.projections[3](result.flatten(1)) + inputs
            return {"full_only_seq": result}, None, None

    class Model:
        def __init__(self):
            self.net = torch.nn.Sequential(Block())
            self.config = SimpleNamespace(bca_feature_pool=2)
            self.parallel_dims = SimpleNamespace(cp_enabled=False)

        def denoise(self, *, data_batch_packed, memory):
            result = self.net(memory)[0]["full_only_seq"]
            return {"preds_vision": list(result.split(48))}

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(7)
        model = Model()
        for name, parameter in model.net.named_parameters():
            parameter.requires_grad_("lora_" in name or "bca_" in name)
        original = {name: p.detach().clone() for name, p in model.net.named_parameters()}
        optimizer = torch.optim.Adam([p for p in model.net.parameters() if p.requires_grad], lr=1e-2)
        packed = SimpleNamespace(vision=SimpleNamespace(condition_mask=[torch.ones(48).bool(), torch.zeros(48).bool()]))
        noisy_inputs, target = torch.randn(96, 8), torch.randn(48, 8)
        with torch.no_grad(), dense_teacher():
            teacher_before = model.denoise(data_batch_packed=packed, memory=noisy_inputs)["preds_vision"][1]
        for _ in range(10):
            optimizer.zero_grad()
            student, losses = teacher_student_forward(model, packed, noisy_inputs)
            loss = F.mse_loss(student["preds_vision"][1], target)
            loss = loss + losses["bca_prediction_loss"] + 0.5 * losses["bca_feature_loss"]
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
        for name, parameter in model.net.named_parameters():
            if not parameter.requires_grad:
                torch.testing.assert_close(parameter, original[name], rtol=0, atol=0)
        registers = "0.self_attn.bca_attention.bca_register_queries"
        assert not torch.equal(dict(model.net.named_parameters())[registers], original[registers])
        assert not torch.equal(model.net[0].projections[3].lora_B.weight, original["0.projections.3.lora_B.weight"])
        with torch.no_grad(), dense_teacher():
            teacher_after = model.denoise(data_batch_packed=packed, memory=noisy_inputs)["preds_vision"][1]
        torch.testing.assert_close(teacher_after, teacher_before, rtol=0, atol=0)
        assert model.net.training
    finally:
        torch.set_num_threads(previous_threads)
