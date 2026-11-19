#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run the native Transfer-Auto notebook workflow without Jupyter.

Invoke from the Framework root with python -m cosmos_framework.scripts.run_transfer_auto.
Uses the checkout's helpers for sample checksums, camera mapping and validation.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def write_window_captions(request: dict, output: Path):
    """Copy the initial caption interval into this run, leaving downloaded assets intact."""
    from helpers.wsm_helpers import caption_window

    frames = request["num_frames"]
    multiview = request["multiview"]
    entries = [(f"view_{index:02d}", view) for index, view in enumerate(multiview["views"])]
    if multiview.get("caption_path"):
        entries.append(("shared", multiview))
    prepared = []
    for name, entry in entries:
        source = Path(entry["caption_path"])
        caption = caption_window(json.loads(source.read_text()), frames, request["fps"])
        document = {
            "caption_structured": {
                f"chunk_0_{frames}": {
                    "start_frame": 0,
                    "end_frame": frames,
                    "caption": caption,
                }
            }
        }
        prepared.append((output / "captions" / f"{name}.json", entry, document))
    for path, entry, document in prepared:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2) + "\n")
        entry["caption_path"] = str(path)
    multiview["use_first_chunk_only"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=("camera", "text2video", "joint"), default="camera")
    parser.add_argument("--resolution", choices=("480", "720"), default="480")
    parser.add_argument("--frames", type=int, help="Generate the first N frames per view (5 or more, 4k+1)")
    parser.add_argument("--gpus", type=int, help="Default: all visible GPUs")
    parser.add_argument("--checkpoint", type=Path, help="Reuse a complete local model snapshot")
    parser.add_argument("--revision", default="main", help="Model revision when downloading")
    parser.add_argument("--sample-root", type=Path, help="Reuse a downloaded sample_data folder")
    parser.add_argument("--request", type=Path, help="Custom single-sample JSON/JSONL request")
    parser.add_argument("--output", type=Path, help="Fresh output directory")
    parser.add_argument(
        "--captions", choices=("captions", "captions_with_timestamp"), default="captions_with_timestamp"
    )
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=None, help="Otherwise preserve request seed")
    parser.add_argument("--profile-steps", action="store_true", help="Write timings for every denoising step")
    parser.add_argument("--profile-step-attention", action="store_true", help="Also time maskless attention regions")
    parser.add_argument(
        "--disable-cross-view-attention", action="store_true", help="Run the same-view-only Auto ablation"
    )
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--num-iterations", type=int, default=1)
    parser.add_argument("--profile", action="store_true", help="Also export a PyTorch Chrome trace")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare/check inputs without downloading model weights or using GPUs",
    )
    args = parser.parse_args()
    if args.steps < 1 or (args.gpus is not None and args.gpus < 1):
        parser.error("--steps and --gpus must be positive")
    if args.frames is not None and (args.frames < 5 or (args.frames - 1) % 4):
        parser.error("--frames must be 4k+1 and at least 5, e.g. 61, 121, 201 or 297")

    if args.profile_step_attention and not args.profile_steps:
        parser.error("--profile-step-attention requires --profile-steps")
    if args.warmup < 0 or args.num_iterations < 1:
        parser.error("--warmup must be nonnegative and --num-iterations must be positive")
    repo = args.repo.expanduser().absolute()
    auto = repo / ".agents/skills/cookbooks/cosmos3/simulation/auto"
    if not (repo / "cosmos_framework").is_dir() or not (auto / "helpers/daytime_sample.py").is_file():
        parser.error("Run from the Cosmos Framework root, or pass --repo /path/to/cosmos-framework")
    for binary in ("ffmpeg", "ffprobe"):
        if not shutil.which(binary):
            parser.error(f"Install {binary} and put it on PATH")
    os.environ.update(COSMOS_INTERNAL="0", COSMOS_TRAINING="0", TOKENIZERS_PARALLELISM="false")
    sys.path[:0] = [str(auto), str(repo)]
    from helpers.auto_helpers import probe_video, validate_framework_outputs, validate_rgb_conditions
    from helpers.daytime_sample import fetch_sample, native_request
    from helpers.lidar_rangemap import index_sweeps

    if args.request:
        source = args.request.expanduser().absolute()
    else:
        local_sample = args.sample_root or os.environ.get("AUTO_SAMPLE_ROOT")
        source = fetch_sample(local_sample) / "input-transfer-auto.jsonl"
    request = native_request(source, mode=args.mode, caption_folder=None if args.request else args.captions)
    request.update(
        resolution=args.resolution, aspect_ratio="16,9", num_steps=args.steps, guidance=args.guidance, shift=args.shift
    )
    if args.frames is not None:
        request.update(num_frames=args.frames, num_video_frames_per_chunk=args.frames)
    if args.seed is not None:
        request["seed"] = args.seed
    negatives = repo / ".agents/skills/cosmos3-nano-transfer-auto/inference/assets/negative-prompt.txt"
    request["per_view_negative_prompt"] = negatives.read_text().strip()

    views = request["multiview"]["views"]
    keys = [view["camera_key"] for view in views]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("Camera keys must be nonempty and unique")
    for view in views:
        info = probe_video(Path(view["control_path"]))
        if info["fps"] != request["fps"] or info["frames"] < request["num_frames"]:
            raise ValueError(f"WSM timeline does not cover the request: {view['camera_key']}")
    if request["multiview"].get("condition_video_as_image"):
        validate_rgb_conditions(views, source.parent)
    if "lidar" in request and request["lidar"]["control_path"].endswith(".tar"):
        with tarfile.open(request["lidar"]["control_path"]) as archive:
            _, sweeps = index_sweeps(archive)
        count = round(request["num_frames"] * 10 / request["fps"])
        if list(sweeps)[:count] != [i * round(request["fps"] / 10) for i in range(count)]:
            raise ValueError("LiDAR controls do not cover the camera timeline")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or repo / "outputs/transfer_auto" / stamp).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    if args.frames is not None:
        write_window_captions(request, output)
    prepared = output / "request.json"
    prepared.write_text(json.dumps(request, indent=2) + "\n")
    print(f"Python: {sys.executable}", flush=True)
    print(f"{len(views)} views; {request['num_frames']} frames; {args.resolution}p; {args.mode}", flush=True)
    print(f"Prepared request: {prepared}", flush=True)
    if args.prepare_only:
        return

    import torch

    available = torch.cuda.device_count()
    gpus = args.gpus if args.gpus is not None else available
    if not 1 <= gpus <= available:
        raise ValueError(f"Requested {gpus} GPUs; this environment sees {available}")
    if args.checkpoint:
        checkpoint = args.checkpoint.expanduser().absolute()
    else:
        from huggingface_hub import HfApi, snapshot_download

        model = "nvidia/Cosmos3-Nano-Transfer-Auto"
        revision = HfApi().model_info(model, revision=args.revision).sha
        print(f"Model revision: {revision}", flush=True)
        checkpoint = Path(snapshot_download(model, revision=revision, ignore_patterns=["assets/**"]))
    if not (checkpoint / "config.json").is_file():
        raise ValueError(f"Expected a complete model snapshot at {checkpoint}")
    generated = output / "generated"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={gpus}",
        "--module",
        "cosmos_framework.scripts.inference",
        "-i",
        str(prepared),
        "-o",
        str(generated),
        "--checkpoint-path",
        str(checkpoint),
        "--cp-size",
        str(gpus),
        "--cfgp-size",
        "1",
        "--dp-shard-size",
        "1",
        "--dp-replicate-size",
        "1",
        "--no-use-torch-compile",
        "--no-use-cuda-graphs",
        "--no-diffusion-cache",
    ]
    for flag in ("profile_steps", "profile_step_attention", "disable_cross_view_attention", "profile"):
        if getattr(args, flag):
            command.append("--" + flag.replace("_", "-"))
    command.extend(["--warmup", str(args.warmup), "--num-iterations", str(args.num_iterations)])
    if args.num_iterations > 1:
        command.append("--benchmark")
    (output / "command.txt").write_text(shlex.join(command) + "\n")
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=repo, env=os.environ.copy(), check=True)
    result = validate_framework_outputs(generated, request)
    for camera, path in result["cameras"].items():
        print(f"{camera}: {path}")
    if "lidar" in result:
        print(f"LiDAR: {result['lidar']}")
    print(f"Validated output: {generated}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
