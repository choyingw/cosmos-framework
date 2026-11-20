# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""Budgeted cross-view K/V: calibrated local routes, learned summaries, exact residuals.

Paper 5919, equations (2)-(7), adapted to the native Cosmos3 maskless partition
and capture-time visibility policy. Dense same-view and caption passes stay native.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmos_framework.configs.base.defaults.multiview_attention import DECOMPOSED_TEMPORAL_WINDOW_EPS
from cosmos_framework.model.generator.mot.bca_geometry import PHYSICAL_NEIGHBORS, local_routes

_ACTIVE: ContextVar[BudgetCrossView | None] = ContextVar("bca_layer", default=None)
_TEACHER: ContextVar[bool] = ContextVar("bca_dense_teacher", default=False)


@contextmanager
def bca_layer(operator):
    token = _ACTIVE.set(operator)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


@contextmanager
def dense_teacher():
    from cosmos_framework.utils.generator.lora import disable_lora

    token = _TEACHER.set(True)
    try:
        with disable_lora():
            yield
    finally:
        _TEACHER.reset(token)


def active_bca():
    return None if _TEACHER.get() else _ACTIVE.get()


def record_plan(plan):
    operator = _ACTIVE.get()
    if operator is not None:
        operator.last_plan = plan


def importance_indices(values: torch.Tensor, shape: tuple[int, int, int], ratio: float, *, seed: int = 0):
    """Per-head representatives plus top residual ||v_j-v_representative||².

    Return [heads,ceil(ratio*N)] indices; K and V use exactly these same rows.
    No token merging/averaging is performed. Torch RNG is private and repeatable
    under activation-checkpoint recomputation.
    """
    frames, height, width = shape
    count, heads, _ = values.shape
    if count != frames * height * width or not 0 <= ratio <= 1:
        raise ValueError("BCA importance selection requires a matching T,H,W grid and a ratio in [0,1]")
    keep = math.ceil(count * ratio)
    if keep == 0:
        return torch.empty((heads, 0), device=values.device, dtype=torch.long)
    if keep == count:
        return torch.arange(count, device=values.device).expand(heads, -1)
    t, y, x = torch.meshgrid(
        torch.arange(frames, device=values.device),
        torch.arange(height, device=values.device),
        torch.arange(width, device=values.device),
        indexing="ij",
    )
    cells = (
        (t // 2) * math.ceil(height / 2) * math.ceil(width / 2) + (y // 2) * math.ceil(width / 2) + x // 2
    ).flatten()
    n_cells = math.ceil(frames / 2) * math.ceil(height / 2) * math.ceil(width / 2)
    if keep < n_cells:
        raise ValueError("Residual retention is below the protected 2x2x2 representative-cell floor")
    generator = torch.Generator(device=values.device).manual_seed(seed)
    random = torch.rand((heads, count), generator=generator, device=values.device)
    maxima = random.new_full((heads, n_cells), -1)
    maxima.scatter_reduce_(1, cells.expand(heads, -1), random, reduce="amax", include_self=True)
    representatives = torch.zeros((heads, n_cells), device=values.device, dtype=torch.long)
    token_ids = torch.arange(count, device=values.device).expand(heads, -1)
    winners = random == maxima.gather(1, cells.expand(heads, -1))
    representatives.scatter_reduce_(1, cells.expand(heads, -1), torch.where(winners, token_ids, 0), reduce="amax")
    features = values.detach().float().permute(1, 0, 2)
    reference = features.gather(1, representatives[:, cells].unsqueeze(-1).expand(-1, -1, features.shape[-1]))
    scores = (features - reference).square().sum(-1)
    scores.scatter_(1, representatives, float("inf"))
    return scores.topk(keep, dim=-1).indices.sort(dim=-1).values


def head_gather(tensor, indices):
    return tensor.permute(1, 0, 2).gather(1, indices.unsqueeze(-1).expand(-1, -1, tensor.shape[-1])).permute(1, 0, 2)


class DifferentiablePartitionBridge(torch.autograd.Function):
    """Translate native FA storage-patching merges into ordinary PyTorch VJPs.

    The native merge patches each partition's saved O/LSE with the joint
    output/normalizer and forwards an unscaled output gradient. PyTorch's
    differentiable attention needs both its scaled O gradient and an LSE VJP.
    Clones keep the raw partition values intact until that VJP is computed.
    """

    @staticmethod
    def forward(ctx, output, lse):
        patched_output, patched_lse = output.clone(), lse.clone()
        ctx.save_for_backward(output, lse, patched_output, patched_lse)
        return patched_output, patched_lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse):
        output, lse, joint_output, joint_lse = ctx.saved_tensors
        mass = torch.exp(lse - joint_lse).nan_to_num(0)
        lse_gradient = mass * ((output.float() - joint_output.float()) * grad_output.float()).sum(-1)
        if grad_lse is not None:
            lse_gradient = lse_gradient + mass * grad_lse
        return grad_output * mass[..., None], lse_gradient


class BudgetCrossView(nn.Module):
    def __init__(self, head_dim: int, config, layer_index: int, *, device=None):
        super().__init__()
        self.head_dim, self.layer_index = head_dim, layer_index
        self.calibration = config.bca_calibration_json
        self.cameras = tuple(config.bca_camera_keys)
        self.local_budget, self.global_budget = config.bca_local_budget, config.bca_global_budget
        self.pool = config.bca_global_pool
        self.inference_ratio = config.bca_inference_ratio
        self.query_chunk = config.bca_query_chunk
        self.depth_min, self.depth_max = config.bca_min_depth, config.bca_max_depth
        self.depth_samples = config.bca_depth_samples
        self.bca_register_queries = nn.Parameter(torch.empty(self.global_budget, 1, head_dim, device=device))
        self.last_plan = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.bca_register_queries, std=self.head_dim**-0.5)

    def _summaries(self, key, value, shape):
        # [roles,views,T*H*W,heads,D] -> pooled tokens, preserving each frame.
        frames, height, width = shape
        roles, views, _, heads, dim = key.shape

        def pool(tensor):
            tensor = tensor.reshape(roles * views * frames, height, width, heads * dim).permute(0, 3, 1, 2)
            tensor = F.avg_pool2d(tensor, self.pool, self.pool, ceil_mode=True)
            return tensor.permute(0, 2, 3, 1).reshape(roles, views, -1, heads, dim)

        return pool(key), pool(value)

    def _attend(self, query, local_key, local_value, local_valid, shared_key, shared_value):
        heads, kv_heads = query.shape[1], local_key.shape[2]
        if heads % kv_heads:
            raise ValueError("BCA Q heads must be divisible by K/V heads")
        groups = heads // kv_heads
        local_key = local_key.repeat_interleave(groups, dim=2)
        local_value = local_value.repeat_interleave(groups, dim=2)
        shared_key = shared_key.repeat_interleave(groups, dim=1)
        shared_value = shared_value.repeat_interleave(groups, dim=1)
        local_logits = torch.einsum("qhd,qkhd->qhk", query.float(), local_key.float()) * self.head_dim**-0.5
        local_logits = local_logits.masked_fill(~local_valid[:, None, :], float("-inf"))
        shared_logits = torch.einsum("qhd,khd->qhk", query.float(), shared_key.float()) * self.head_dim**-0.5
        logits = torch.cat((local_logits, shared_logits), dim=-1)
        present = torch.isfinite(logits).any(-1)
        safe = torch.where(present[..., None], logits, torch.zeros_like(logits))
        weights = torch.softmax(safe, dim=-1).to(query.dtype)
        result = torch.einsum("qhk,qkhd->qhd", weights[..., : self.local_budget], local_value) + torch.einsum(
            "qhk,khd->qhd", weights[..., self.local_budget :], shared_value
        )
        result = torch.where(present[..., None], result, torch.zeros_like(result))
        lse = torch.where(present, torch.logsumexp(safe, dim=-1), safe.new_full((), -float("inf")))
        return result, lse

    def _source_registers(self, key, value):
        """Reusable per-source summaries; a small LSE merge selects neighbors."""
        roles, views, _, heads, dim = key.shape
        key = key.permute(1, 0, 2, 3, 4).reshape(views, -1, heads, dim)
        value = value.permute(1, 0, 2, 3, 4).reshape_as(key)
        scores = torch.einsum("rhd,vkhd->vhrk", self.bca_register_queries.to(key.dtype), key) * dim**-0.5
        weights = scores.float().softmax(-1).to(key.dtype)
        return (
            torch.einsum("vhrk,vkhd->vrhd", weights, key),
            torch.einsum("vhrk,vkhd->vrhd", weights, value),
            scores.float().logsumexp(-1).transpose(1, 2),
        )

    def forward(self, query, key, value, plan):
        self.last_plan = plan
        views = len(self.cameras)
        # Explicitly support the seven-camera, aligned control+target experiment.
        # Refuse mixed samples/sensors rather than silently routing the wrong rig.
        if (
            len(plan.items_per_sample) != 1
            or len(set(plan.token_shapes)) != 1
            or any(v != views for v in plan.num_views)
        ):
            raise ValueError("BCA requires one sample of synchronized camera-only items with a fixed rig/grid")
        if any(axis != 0 for axis in plan.view_axis):
            raise ValueError("BCA requires the native camera-major temporal layout")
        total_frames, height, width = plan.token_shapes[0]
        frames = total_frames // views
        per_view, roles = frames * height * width, len(plan.token_shapes)
        count = roles * views * per_view
        if count != plan.num_gen_tokens:
            raise ValueError("BCA packed camera grid does not cover the real generation tokens")
        if len(set(plan.seconds_per_frame)) != 1:
            raise ValueError("BCA requires synchronized camera rates")
        q = query[:count].reshape(roles, views, frames, height * width, query.shape[1], self.head_dim)
        k = key[:count].reshape(roles, views, frames, height * width, key.shape[1], self.head_dim)
        v = value[:count].reshape_as(k)
        route, valid = local_routes(
            self.calibration,
            self.cameras,
            height,
            width,
            self.local_budget,
            self.depth_samples,
            self.depth_min,
            self.depth_max,
        )
        route, valid = route.to(query.device), valid.to(query.device)
        output, normalizers = (
            torch.zeros_like(query),
            query.new_full(query.shape[:2], -float("inf"), dtype=torch.float32),
        )
        bounds = plan.cross_view_temporal_window.bounds
        epsilon = DECOMPOSED_TEMPORAL_WINDOW_EPS
        source_role = torch.arange(self.local_budget, device=query.device) % roles
        for frame in range(frames):
            # Retain Cosmos3's capture-time visibility; compression never
            # introduces a future key forbidden by the native dense teacher.
            allowed = [
                i
                for i in range(frames)
                if (
                    i == frame
                    if bounds is None
                    else bounds[0] - epsilon <= (i - frame) * plan.seconds_per_frame[0] <= bounds[1] + epsilon
                )
            ]
            if plan.cross_view_temporal_window.includes_first_frame and 0 not in allowed:
                allowed.insert(0, 0)
            if not allowed:
                raise ValueError("BCA requires a cross-view time window containing camera frames")
            pooled_k, pooled_v = self._summaries(
                k[:, :, allowed].reshape(roles, views, -1, k.shape[-2], self.head_dim),
                v[:, :, allowed].reshape(roles, views, -1, v.shape[-2], self.head_dim),
                (len(allowed), height, width),
            )
            source_k, source_v, source_lse = self._source_registers(pooled_k, pooled_v)
            residuals = {}
            if not self.training:
                for other in range(views):
                    selected_k, selected_v = [], []
                    for role in range(roles):
                        rk = k[role, other, allowed].reshape(-1, k.shape[-2], self.head_dim)
                        rv = v[role, other, allowed].reshape_as(rk)
                        indices = importance_indices(
                            rv,
                            (len(allowed), height, width),
                            self.inference_ratio,
                            seed=10009 * self.layer_index + 101 * other + role,
                        )
                        selected_k.append(head_gather(rk, indices))
                        selected_v.append(head_gather(rv, indices))
                    residuals[other] = (torch.cat(selected_k), torch.cat(selected_v))
            for view in range(views):
                neighbors = PHYSICAL_NEIGHBORS[view] if views == 7 else tuple(i for i in range(views) if i != view)
                # Exact softmax over neighbors' pooled source tokens, without
                # repeating register/source dot products for each target camera.
                weights = source_lse[list(neighbors)].softmax(0).to(source_k.dtype)[..., None]
                global_k = (weights * source_k[list(neighbors)]).sum(0)
                global_v = (weights * source_v[list(neighbors)]).sum(0)
                residual_k = [pair[0] for other, pair in residuals.items() if other != view]
                residual_v = [pair[1] for other, pair in residuals.items() if other != view]
                shared_k = torch.cat([global_k, *residual_k])
                shared_v = torch.cat([global_v, *residual_v])
                # Local candidates are same-frame, split across control/target
                # source roles while the TOTAL local budget stays exactly 32.
                source_view = route[view] // (height * width)
                source_pixel = route[view] % (height * width)
                local_k = k[source_role[None], source_view, frame, source_pixel]
                local_v = v[source_role[None], source_view, frame, source_pixel]
                for role in range(roles):
                    offset = (role * views * frames + view * frames + frame) * height * width
                    for start in range(0, height * width, self.query_chunk):
                        stop = min(start + self.query_chunk, height * width)
                        out, lse = self._attend(
                            q[role, view, frame, start:stop],
                            local_k[start:stop],
                            local_v[start:stop],
                            valid[view, start:stop],
                            shared_k,
                            shared_v,
                        )
                        output[offset + start : offset + stop] = out
                        normalizers[offset + start : offset + stop] = lse
        output, normalizers = output.unsqueeze(0), normalizers.unsqueeze(0)
        if torch.is_grad_enabled():
            return DifferentiablePartitionBridge.apply(output, normalizers)
        return output, normalizers


def install_bca(net, config):
    if not config.bca_enabled:
        return
    if (
        config.compile.enabled
        or config.video_temporal_causal
        or config.joint_attn_implementation != "multiview"
        or config.multiview_attention.backend != "maskless"
        or config.multiview_attention.geometry_position_encoding != "baseline"
        or config.multiview_attention.mask.attention_scope != "decomposed"
        or not config.multiview_attention.mask.control_attends_sensor
        or config.attention_io_layout != "sequence_sharded"
    ):
        raise ValueError(
            "BCA requires bidirectional camera maskless attention with sequence-sharded I/O and native mRoPE"
        )
    if not config.bca_calibration_json:
        raise ValueError("BCA requires matching calibrated camera intrinsics/extrinsics")
    selected = set(range(config.bca_start_block, config.bca_end_block + 1))
    found = set()
    for module in list(net.modules()):
        if hasattr(module, "q_proj_moe_gen") and getattr(module, "layer_idx", -1) in selected:
            module.bca_attention = BudgetCrossView(
                module.head_dim, config, module.layer_idx, device=module.q_proj_moe_gen.weight.device
            )
            found.add(module.layer_idx)
    if found != selected:
        raise ValueError(f"BCA blocks are absent from this model: {sorted(selected - found)}")


def initialize_bca(net):
    for module in net.modules():
        if isinstance(module, BudgetCrossView):
            module.reset_parameters()
