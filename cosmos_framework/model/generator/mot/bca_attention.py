# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""BCA primitives with ordinary output AND log-normalizer derivatives.

The enclosing BCA partition bridge alone handles native storage-patching merges.
Internal FlashAttention LSEs need their own VJP: silently detaching them gives
incorrect register and local/shared partition training gradients.
"""

import importlib
import inspect
import os
from functools import lru_cache

import torch


def fused_supported(tensor):
    return tensor.is_cuda and tensor.dtype in (torch.float16, torch.bfloat16) and tensor.shape[-1] <= 256


@lru_cache(maxsize=8)
def _inference_flash_backend(device, preference):
    """Use the installed Hopper backend, as Bosch does; never hide execution errors."""
    if preference not in ("auto", "aten", "flash3"):
        raise ValueError("COSMOS_BCA_FLASH_BACKEND must be auto, aten or flash3")
    if preference == "aten":
        return "aten", None, False
    hopper = torch.cuda.get_device_capability(device)[0] == 9
    if not hopper:
        if preference == "flash3":
            raise RuntimeError("BCA FlashAttention-3 requires a Hopper GPU (e.g. H100/H200)")
        return "aten", None, False
    failures = []
    for package in ("flash_attn_3_nv.flash_attn_interface", "flash_attn_3.flash_attn_interface"):
        try:
            module = importlib.import_module(package)
        except (ImportError, OSError) as exc:
            failures.append(f"{package}: {exc}")
            continue
        function = module.flash_attn_func
        return package, function, "return_attn_probs" in inspect.signature(function).parameters
    reason = "BCA could not load Hopper FlashAttention-3: " + "; ".join(failures)
    if preference == "flash3":
        raise RuntimeError(reason)
    import warnings

    warnings.warn(
        reason + ". Falling back to PyTorch FlashAttention; set COSMOS_BCA_FLASH_BACKEND=flash3 to require FA3.",
        RuntimeWarning,
        stacklevel=2,
    )
    return "aten", None, False


def inference_attention_with_lse(query, key, value):
    _, function, return_probs = _inference_flash_backend(
        query.device, os.environ.get("COSMOS_BCA_FLASH_BACKEND", "auto")
    )
    if function is not None:
        kwargs = {"return_attn_probs": True} if return_probs else {}
        out, lse = function(
            q=query.contiguous(),
            k=key.contiguous(),
            v=value.contiguous(),
            softmax_scale=query.shape[-1] ** -0.5,
            causal=False,
            **kwargs,
        )[:2]
        return out, lse.transpose(1, 2)
    # FlashAttention accepts these transposed VIEWS. Making them head-major
    # contiguous copied every Q/K/V bank in the previous implementation.
    result = torch.ops.aten._scaled_dot_product_flash_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), 0.0, False, False
    )
    return result[0].transpose(1, 2), result[1].transpose(1, 2)


class _FlashWithLSE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value):
        # ATen exposes FlashAttention without depending on a separately compiled
        # flash-attn extension. Its layouts are [B,H,S,D], including GQA.
        q, k, v = [x.transpose(1, 2).contiguous() for x in (query, key, value)]
        result = torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, False, False)
        out, lse, cu_q, cu_k, max_q, max_k, rng, unused, _ = result
        ctx.save_for_backward(q, k, v, out, lse, cu_q, cu_k, rng, unused)
        ctx.max_q, ctx.max_k = max_q, max_k
        ctx.set_materialize_grads(False)
        return out.transpose(1, 2), lse.transpose(1, 2)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output, grad_lse):
        from cosmos_framework.model.generator.mot.bca_kernels import lse_backward

        q, k, v, out, lse, cu_q, cu_k, rng, unused = ctx.saved_tensors
        if grad_output is not None:
            dq, dk, dv = torch.ops.aten._scaled_dot_product_flash_attention_backward(
                grad_output.transpose(1, 2).contiguous(),
                q,
                k,
                v,
                out,
                lse,
                cu_q,
                cu_k,
                ctx.max_q,
                ctx.max_k,
                0.0,
                False,
                rng,
                unused,
            )
        else:
            dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
        if grad_lse is not None:
            batch, heads, qlen, dim = q.shape
            hk, klen = k.shape[1:3]
            dl = grad_lse.transpose(1, 2).contiguous()
            # FP32 scratch; GQA query-head partials are reduced after the kernel.
            extra_q = torch.empty_like(q, dtype=torch.float32)
            extra_k = torch.empty((batch, heads, klen, dim), device=k.device, dtype=torch.float32)
            args = dict(QLEN=qlen, KLEN=klen, HQ=heads, HK=hk, D=dim, BD=max(16, 1 << (dim - 1).bit_length()), BLOCK=32)
            with torch.cuda.device(q.device):
                lse_backward[((qlen + 31) // 32, batch * heads)](
                    q, k, lse.contiguous(), dl, extra_q, QUERY_GRAD=True, **args
                )
                lse_backward[((klen + 31) // 32, batch * heads)](
                    q, k, lse.contiguous(), dl, extra_k, QUERY_GRAD=False, **args
                )
            dq = (dq.float() + extra_q).to(q.dtype)
            dk = (dk.float() + extra_k.reshape(batch, hk, heads // hk, klen, dim).sum(2)).to(k.dtype)
        return dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2)


def attention_with_lse(query, key, value):
    """[batch,sequence,heads,dim], with GQA and no score matrix on CUDA."""
    if fused_supported(query):
        if not torch.is_grad_enabled():
            return inference_attention_with_lse(query, key, value)
        return _FlashWithLSE.apply(query, key, value)
    groups = query.shape[2] // key.shape[2]
    k, v = [x.repeat_interleave(groups, dim=2) for x in (key, value)]
    scores = torch.einsum("bqhd,bkhd->bqhk", query.float(), k.float()) * query.shape[-1] ** -0.5
    return torch.einsum("bqhk,bkhd->bqhd", scores.softmax(-1).to(query.dtype), v), scores.logsumexp(-1)


def _indexed_forward(query, key, value, routes, valid, roles, offset, base=None, base_lse=None, frame_ids=None):
    from cosmos_framework.model.generator.mot.bca_kernels import indexed_forward

    views, queries, heads, dim = query.shape
    pixels, budget = routes.shape[1:]
    tile = 4 if budget <= 32 and dim <= 128 else 1
    bd, bl = 1 << (dim - 1).bit_length(), 1 << (budget - 1).bit_length()
    out = torch.empty_like(query)
    lse = torch.empty(query.shape[:-1], device=query.device, dtype=torch.float32)
    with torch.cuda.device(query.device):
        indexed_forward[((views * queries + tile - 1) // tile, heads)](
            query,
            key,
            value,
            routes,
            valid,
            out,
            lse,
            query if base is None else base.contiguous(),
            lse if base_lse is None else base_lse.contiguous(),
            routes if frame_ids is None else frame_ids,
            routes.shape[0],
            frame_ids is not None,
            views * queries,
            pixels,
            roles,
            heads,
            key.shape[1],
            dim,
            budget,
            offset,
            bd,
            bl,
            tile,
            base is not None,
            num_warps=8 if tile * bd * bl > 8192 else 4,
        )
    return out, lse


def indexed_attention_with_shared(query, key, value, routes, valid, roles, offset, shared, shared_lse):
    if fused_supported(query) and not torch.is_grad_enabled():
        return _indexed_forward(
            query.contiguous(), key.contiguous(), value.contiguous(), routes, valid, roles, offset, shared, shared_lse
        )
    local, local_lse = indexed_attention(query, key, value, routes, valid, roles, offset)
    return merge_partitions(local, local_lse, shared, shared_lse)


class _IndexedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, routes, valid, roles, offset):
        query, key, value = [x.contiguous() for x in (query, key, value)]
        out, lse = _indexed_forward(query, key, value, routes, valid, roles, offset)
        ctx.save_for_backward(query, key, value, routes, valid, out, lse)
        ctx.roles, ctx.offset = roles, offset
        return out, lse

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output, grad_lse):
        from cosmos_framework.model.generator.mot.bca_kernels import indexed_backward

        q, k, v, routes, valid, out, lse = ctx.saved_tensors
        views, queries, heads, dim = q.shape
        pixels, budget = routes.shape[1:]
        dq = torch.empty_like(q)
        dk, dv = [torch.zeros_like(x, dtype=torch.float32) for x in (k, v)]
        with torch.cuda.device(q.device):
            indexed_backward[(views * queries, heads)](
                q,
                k,
                v,
                routes,
                valid,
                out,
                lse,
                grad_output.contiguous(),
                grad_lse.contiguous(),
                dq,
                dk,
                dv,
                pixels,
                ctx.roles,
                heads,
                k.shape[1],
                dim,
                budget,
                ctx.offset,
                1 << (dim - 1).bit_length(),
                1 << (budget - 1).bit_length(),
            )
        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None, None


def indexed_attention(query, key, value, routes, valid, roles, offset):
    if fused_supported(query):
        return _IndexedAttention.apply(query, key, value, routes, valid, roles, offset)
    views, queries, heads, dim = query.shape
    pixels, budget = routes.shape[1:]
    # Small-grid correctness fallback, intentionally independent of CUDA kernels.
    rows = (routes + offset)[:, None].expand(-1, roles, -1, -1).reshape(views, queries, budget)
    k, v = [x[rows].repeat_interleave(heads // x.shape[1], dim=3) for x in (key, value)]
    mask = valid[:, None].expand(-1, roles, -1, -1).reshape(views, queries, budget)
    scores = torch.einsum("bqhd,bqlhd->bqhl", query.float(), k.float()) * dim**-0.5
    scores = scores.masked_fill(~mask[:, :, None], -torch.inf)
    present = mask.any(-1)[:, :, None]
    safe = torch.where(present[..., None], scores, 0.0)
    out = torch.einsum("bqhl,bqlhd->bqhd", safe.softmax(-1).to(query.dtype), v)
    return torch.where(present[..., None], out, 0.0), torch.where(present, safe.logsumexp(-1), -torch.inf)


def merge_partitions(local, local_lse, shared, shared_lse):
    # The shared bank is always nonempty. Avoid -inf/-inf and preserve all LSE
    # gradients, including rows with no valid geometric routes.
    lse = torch.logaddexp(local_lse, shared_lse)
    local_mass = (local_lse - lse).exp()[..., None]
    shared_mass = (shared_lse - lse).exp()[..., None]
    return (local.float() * local_mass + shared.float() * shared_mass).to(shared.dtype), lse
