# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""CUDA parity tests for the actual BCA kernels; no model weights are required."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from cosmos_framework.model.generator.mot.bca import (
    BudgetCrossView,
    batched_importance_indices,
    importance_indices,
)
from cosmos_framework.model.generator.mot.bca_attention import attention_with_lse, indexed_attention
from cosmos_framework.model.generator.mot.merge_attention import merge_attentions_ac_safe

pytestmark = [
    pytest.mark.L0,
    pytest.mark.GPU,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def _close(actual, expected, *, atol=0.025):
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=0.03)
    # Also constrain aggregate error; absolute tolerances alone hide weak gradients.
    delta = (actual.float() - expected.float()).norm()
    assert delta <= 0.025 * expected.float().norm() + 1e-5


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("gqa", [1, 2])
@pytest.mark.parametrize("lse_only", [False, True])
@pytest.mark.parametrize("dim", [32, 128])
def test_flash_output_and_lse_vjp(dtype, gqa, lse_only, dim):
    torch.manual_seed(27)
    tensors = [
        torch.randn(2, n, h, dim, device="cuda", dtype=dtype, requires_grad=True)
        for n, h in [(13, 2 * gqa), (19, 2), (19, 2)]
    ]
    out, lse = attention_with_lse(*tensors)
    reference = [x.detach().float().requires_grad_() for x in tensors]
    q, k, v = reference
    scores = torch.einsum("bqhd,bkhd->bqhk", q, k.repeat_interleave(gqa, 2)) / dim**0.5
    expected = torch.einsum("bqhk,bkhd->bqhd", scores.softmax(-1), v.repeat_interleave(gqa, 2))
    _close(out, expected)
    _close(lse, scores.logsumexp(-1))
    dout, dlse = torch.randn_like(out), torch.randn_like(lse)
    if lse_only:
        actual_grad = torch.autograd.grad(lse, tensors, dlse)
        expected_grad = torch.autograd.grad(scores.logsumexp(-1), reference[:2], dlse)
        assert actual_grad[2].count_nonzero() == 0
    else:
        actual_grad = torch.autograd.grad((out, lse), tensors, (dout, dlse))
        expected_grad = torch.autograd.grad((expected, scores.logsumexp(-1)), reference, (dout.float(), dlse))
    for actual, expected in zip(actual_grad, expected_grad):
        _close(actual, expected)


@pytest.mark.parametrize("empty", [False, True])
def test_indexed_attention_gqa_gradients_and_empty_rows(empty):
    torch.manual_seed(31)
    views, roles, pixels, heads, kv_heads, dim, budget, offset = 3, 2, 5, 4, 2, 64, 7, 2
    q = torch.randn(views, roles * pixels, heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k, v = [torch.randn(100, kv_heads, dim, device="cuda", dtype=q.dtype, requires_grad=True) for _ in range(2)]
    routes = torch.randint(0, 98, (views, pixels, budget), device="cuda")
    valid = torch.rand_like(routes.float()) > 0.2
    valid[:, 0] = False
    if empty:
        valid[:] = False
    out, lse = indexed_attention(q, k, v, routes, valid, roles, offset)
    reference = [x.detach().float().cpu().requires_grad_() for x in (q, k, v)]
    expected, expected_lse = indexed_attention(*reference, routes.cpu(), valid.cpu(), roles, offset)
    _close(out.cpu(), expected)
    torch.testing.assert_close(lse.cpu(), expected_lse, atol=2e-6, rtol=2e-6)
    dout, dlse = torch.randn_like(out), torch.randn_like(lse)
    actual_grad = torch.autograd.grad((out, lse), (q, k, v), (dout, dlse))
    expected_grad = torch.autograd.grad((expected, expected_lse), reference, (dout.cpu().float(), dlse.cpu()))
    for actual, expected in zip(actual_grad, expected_grad, strict=True):
        _close(actual.cpu(), expected)


@pytest.mark.parametrize("shape,ratio", [((3, 5, 6), 0.35), ((4, 4, 4), 0.35), ((3, 5, 6), 0.0), ((3, 5, 6), 1.0)])
def test_fused_importance_matches_reference(shape, ratio):
    torch.manual_seed(33)
    values = torch.randn(4, shape[0] * shape[1] * shape[2], 2, 64, device="cuda", dtype=torch.bfloat16)
    seeds = (13, 17, 29, 101)
    expected = torch.stack([importance_indices(v, shape, ratio, seed=s) for v, s in zip(values, seeds, strict=True)])
    actual = batched_importance_indices(values, shape, ratio, seeds)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # Cache reuse must never reuse input-dependent scores or selected K/V.
    changed = values * -3
    expected = torch.stack([importance_indices(v, shape, ratio, seed=s) for v, s in zip(changed, seeds, strict=True)])
    torch.testing.assert_close(batched_importance_indices(changed, shape, ratio, seeds), expected, atol=0, rtol=0)


def _operator_and_plan(views=7):
    calibration = json.loads(
        (Path(__file__).resolve().parents[3] / "configs/simulation/assets/bca_daytime_calibration.json").read_text()
    )
    cfg = SimpleNamespace(
        bca_calibration_json=json.dumps(calibration),
        bca_camera_keys=list(calibration["cameras"])[:views],
        bca_local_budget=32,
        bca_global_budget=16,
        bca_global_pool=2,
        bca_inference_ratio=0.35,
        bca_query_chunk=5,
        bca_min_depth=1.0,
        bca_max_depth=120.0,
        bca_depth_samples=8,
    )
    plan = SimpleNamespace(
        token_shapes=((views * 3, 3, 4),) * 2,
        num_views=(views, views),
        view_axis=(0, 0),
        items_per_sample=(2,),
        num_gen_tokens=views * 3 * 3 * 4 * 2,
        seconds_per_frame=(4 / 30,) * 2,
        cross_view_temporal_window=SimpleNamespace(bounds=(-0.4, 0.0), includes_first_frame=True),
    )
    return BudgetCrossView(32, cfg, 10, device="cuda").to(torch.bfloat16), plan


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("checkpoint_mode", ["none", "full", "selective"])
def test_complete_bca_matches_reference_with_native_merge_and_checkpoint(training, checkpoint_mode):
    torch.manual_seed(41)
    operator, plan = _operator_and_plan()
    operator.train(training)
    tensors = [
        torch.randn(plan.num_gen_tokens + 3, h, 32, device="cuda", dtype=torch.bfloat16, requires_grad=training)
        for h in (4, 2, 2)
    ]
    # A nontrivial second partition exercises the native storage-patching VJP.
    other_out = torch.randn_like(tensors[0][: plan.num_gen_tokens]).unsqueeze(0)
    other_lse = torch.randn(other_out.shape[:-1], device="cuda")

    def forward(fn, *args):
        out, lse = fn(*args, plan)
        assert not out[:, plan.num_gen_tokens :].count_nonzero()
        assert lse[:, plan.num_gen_tokens :].isneginf().all()
        # Pad same-view output/normalizer to include unused sequence padding.
        other = torch.nn.functional.pad(other_out, (0, 0, 0, 0, 0, 3))
        olse = torch.nn.functional.pad(other_lse, (0, 0, 0, 3), value=0.0)
        return merge_attentions_ac_safe([other, out], [olse, lse])[0]

    with torch.set_grad_enabled(training):
        if checkpoint_mode != "none":
            from functools import partial

            from torch.utils.checkpoint import create_selective_checkpoint_contexts

            kwargs = {}
            if checkpoint_mode == "selective":
                kwargs["context_fn"] = partial(
                    create_selective_checkpoint_contexts,
                    [torch.ops.aten.mm.default, torch.ops.aten._scaled_dot_product_flash_attention.default],
                )
            actual = checkpoint(lambda *x: forward(operator.forward_optimized, *x), *tensors, use_reentrant=False)
        else:
            actual = forward(operator.forward_optimized, *tensors)
        expected = forward(operator.forward_reference, *tensors)
        _close(actual, expected)
        if training:
            dout = torch.randn_like(actual)
            params = (*tensors, operator.bca_register_queries)
            actual_grad = torch.autograd.grad(actual, params, dout, retain_graph=True)
            expected_grad = torch.autograd.grad(expected, params, dout)
            for actual, expected in zip(actual_grad, expected_grad, strict=True):
                _close(actual, expected, atol=0.06)


@torch.no_grad()
def test_fused_path_cannot_see_future_frames():
    operator, plan = _operator_and_plan(2)
    operator.eval()
    tensors = [torch.randn(plan.num_gen_tokens, h, 32, device="cuda", dtype=torch.bfloat16) for h in (4, 2, 2)]
    original, _ = operator(*tensors, plan)
    modified = tensors[2].reshape(2, 2, 3, 12, 2, 32).clone()
    modified[:, :, 2] += 100
    changed, _ = operator(tensors[0], tensors[1], modified.reshape_as(tensors[2]), plan)
    first = torch.arange(plan.num_gen_tokens, device="cuda").reshape(2, 2, 3, 12)[:, :, :2].flatten()
    torch.testing.assert_close(original[:, first], changed[:, first], atol=0, rtol=0)
    assert not torch.equal(original, changed)


def test_route_cache_from_inference_mode_is_safe_for_later_training():
    from cosmos_framework.model.generator.mot.bca import _device_routes

    _device_routes.cache_clear()
    operator, plan = _operator_and_plan(2)
    tensors = [
        torch.randn(plan.num_gen_tokens, h, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for h in (4, 2, 2)
    ]
    with torch.inference_mode():
        operator.eval()(*tensors, plan)
    output, _ = operator.train()(*tensors, plan)
    output.square().mean().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in tensors)
