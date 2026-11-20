# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compare isolated BCA operators without loading model weights or generating video.

One untimed warmup and one measured call per backend. CSV goes to stdout on
rank zero only. Training includes backward and defaults to two latent frames
because the reference implementation retains very large intermediate tensors.
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from cosmos_framework.model.generator.mot.bca import BudgetCrossView


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inference", "training"), default="inference")
    parser.add_argument("--latent-frames", type=int)
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--width", type=int, default=26)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()
    if int(os.environ.get("RANK", "0")) != 0:
        return
    training = args.mode == "training"
    frames = args.latent_frames if args.latent_frames is not None else (2 if training else 16)
    if min(frames, args.height, args.width, args.query_heads, args.kv_heads, args.head_dim) < 1:
        parser.error("All dimensions must be positive")
    if args.query_heads % args.kv_heads:
        parser.error("Query heads must be divisible by K/V heads")
    if not torch.cuda.is_available():
        parser.error("A CUDA GPU is required")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.manual_seed(20)
    calibration = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/simulation/assets/bca_daytime_calibration.json").read_text()
    )
    config = SimpleNamespace(
        bca_calibration_json=json.dumps(calibration),
        bca_camera_keys=list(calibration["cameras"]),
        bca_local_budget=32,
        bca_global_budget=512,
        bca_global_pool=8,
        bca_inference_ratio=0.35,
        bca_query_chunk=128,
        bca_min_depth=1.0,
        bca_max_depth=120.0,
        bca_depth_samples=8,
    )
    views, roles = 7, 2
    count = roles * views * frames * args.height * args.width
    plan = SimpleNamespace(
        token_shapes=((views * frames, args.height, args.width),) * roles,
        num_views=(views,) * roles,
        view_axis=(0,) * roles,
        items_per_sample=(roles,),
        num_gen_tokens=count,
        seconds_per_frame=(4 / 30,) * roles,
        cross_view_temporal_window=SimpleNamespace(bounds=(-0.4, 0.0), includes_first_frame=False),
    )
    operator = BudgetCrossView(args.head_dim, config, 10, device="cuda").to(torch.bfloat16).train(training)
    tensors = [
        torch.randn(count, heads, args.head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=training)
        for heads in (args.query_heads, args.kv_heads, args.kv_heads)
    ]
    writer = csv.writer(sys.stdout)
    writer.writerow(
        ("backend", "mode", "latent_frames", "height", "width", "wall_ms", "cuda_elapsed_ms", "peak_extra_mib")
    )
    for backend in ("reference", "optimized"):
        forward = getattr(operator, f"forward_{backend}")

        def run():
            with torch.set_grad_enabled(training):
                output, lse = forward(*tensors, plan)
                if training:
                    return torch.autograd.grad(output.square().mean(), (*tensors, operator.bca_register_queries))
                return output, lse

        warmup = run()
        torch.cuda.synchronize()
        del warmup
        gc.collect()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start.record()
        result = run()
        end.record()
        torch.cuda.synchronize()
        writer.writerow(
            (
                backend,
                args.mode,
                frames,
                args.height,
                args.width,
                (time.perf_counter() - wall_start) * 1000,
                start.elapsed_time(end),
                (torch.cuda.max_memory_allocated() - baseline) / 2**20,
            )
        )
        sys.stdout.flush()
        del result


if __name__ == "__main__":
    main()
