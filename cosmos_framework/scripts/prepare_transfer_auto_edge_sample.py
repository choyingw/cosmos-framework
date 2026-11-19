# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""Use run_transfer_auto's single daytime sample for edge preparation or dummy training."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from cosmos_framework.data.generator.local_simulation.transfer_auto_assets import (
    validate_pretrained_initialization as validate_pretrained_initialization,
)
from cosmos_framework.data.generator.multiview.rig_catalog import MADS_AML_ALT_RIG

SAMPLE_REPO = "nvidia/Cosmos3-Nano-Transfer-Auto"


def inference_sample_helpers():
    """Reuse the exact sample downloader/checksums/caption window used by run_transfer_auto."""
    cookbook = Path(__file__).resolve().parents[2] / ".agents/skills/cookbooks/cosmos3/simulation/auto"
    if str(cookbook) not in sys.path:
        sys.path.insert(0, str(cookbook))
    from helpers import daytime_sample

    return daytime_sample


def resolve_sample_root(directory: Path | None = None) -> Path:
    """Find existing assets only; never initiate a download or select generated output."""
    if directory is None and os.environ.get("AUTO_SAMPLE_ROOT"):
        directory = Path(os.environ["AUTO_SAMPLE_ROOT"])
    if directory is None:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(SAMPLE_REPO, "assets/sample_data/input-transfer-auto.jsonl")
        if isinstance(cached, str):
            # Keep the snapshot path, rather than resolving its symlink into blobs/.
            directory = Path(cached).parent
    if directory is None:
        raise FileNotFoundError("Set AUTO_SAMPLE_ROOT or --sample-root to your downloaded assets/sample_data directory")
    directory = directory.expanduser().absolute()
    for candidate in (directory, directory / "assets/sample_data", directory.parent):
        if (candidate / "daytime_scene/rgb").is_dir():
            return candidate
    raise FileNotFoundError(f"No downloaded daytime_scene/rgb under {directory}")


def prepare_sample(sample_root: Path, tokenizer: Path, output: Path, *, captions: str, hardlink: bool = False) -> Path:
    """Use the same first 61 frames/caption window as run_transfer_auto --frames 61."""
    from cosmos_framework.data.generator.local_simulation.auto_bundle import build_bundle

    sample_root = resolve_sample_root(sample_root)
    tokenizer = tokenizer.expanduser().absolute()
    output = output.expanduser().absolute()
    if not tokenizer.is_dir():
        raise FileNotFoundError(f"Matching local text tokenizer directory is missing: {tokenizer}")
    assets = {}
    for camera in MADS_AML_ALT_RIG:
        video = sample_root / "daytime_scene/rgb" / f"{camera}.mp4"
        caption = sample_root / "daytime_scene" / captions / f"{camera}.json"
        for path in (video, caption):
            if not path.is_file():
                raise FileNotFoundError(path)
        document = json.loads(caption.read_text())
        if not isinstance(document, dict) or not document.get("caption_structured"):
            raise ValueError(f"Expected supplied structured captions: {caption}")
        assets[f"video_{camera}"] = str(video)
        assets[f"caption_{camera}"] = document
    # WSM/LiDAR entries in the release's joint training index are deliberately
    # excluded: camera-only edge adaptation needs neither those files nor targets.
    output.mkdir(parents=True, exist_ok=False)
    (output / "captions").mkdir()
    for camera in MADS_AML_ALT_RIG:
        key = f"caption_{camera}"
        selected = inference_sample_helpers().caption_window(assets[key], 61, 30)
        caption = output / "captions" / f"{camera}.json"
        caption.write_text(
            json.dumps({"caption_structured": {"chunk_0_61": dict(start_frame=0, end_frame=61, caption=selected)}})
            + "\n"
        )
        assets[key] = str(caption)
    index = output / "training-index.jsonl"
    index.write_text(json.dumps(dict(id="daytime_scene", stratum="480", assets=assets)) + "\n")
    bundle = output / "bundle"
    build_bundle(index, tokenizer, bundle, list(MADS_AML_ALT_RIG), control_type="edge", hardlink=hardlink)
    environment = output / "train.env"
    environment.write_text(
        "".join(
            f"export {key}={shlex.quote(str(value))}\n"
            for key, value in (
                ("AUTO_SAMPLE_ROOT", sample_root),
                ("EDGE_INDEX", index),
                ("COSMOS_TRANSFER_AUTO_LOCAL_BUNDLE", bundle),
            )
        )
    )
    return environment


def dummy_training_command(*, gpus: int, steps: int) -> list[str]:
    """Repeat the one-sequence bundle through the native infinite training loader."""
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={gpus}",
        "--module",
        "cosmos_framework.scripts.train_experiment",
        "--config",
        # get_config_module expects a path relative to the repository root;
        # the subprocess below already runs with that root as its cwd.
        "cosmos_framework/configs/simulation/config.py",
        "--experiment-module",
        "cosmos_framework.configs.simulation.experiment.auto.transfer_auto_edge",
        "--",
        "experiment=transfer_auto_edge_61_480",
        "job.name=edge_dummy",
        f"trainer.max_iter={steps}",
        f"checkpoint.save_iter={steps}",
        "checkpoint.dcp_async_mode_enabled=False",
        "model.config.compile.enabled=False",
        "scheduler.warm_up_steps=[1]",
        f"scheduler.cycle_lengths=[{max(steps, 2)}]",
        f"model_parallel.context_parallel_size={gpus}",
        f"model.config.parallelism.context_parallel_shard_degree={gpus}",
        f"model.config.parallelism.data_parallel_shard_degree={gpus}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-root", type=Path, help="sample_data, daytime_scene, or its download/snapshot root")
    parser.add_argument("--tokenizer", type=Path, help="Matching local text_tokenizer directory")
    parser.add_argument(
        "--checkpoint", type=Path, help="Reuse a complete local pretrained snapshot; otherwise use HF cache"
    )
    parser.add_argument(
        "--output", type=Path, default=Path.home() / "cosmos_data/edge_daytime_sample", help="New preparation directory"
    )
    parser.add_argument(
        "--captions", choices=("captions", "captions_with_timestamp"), default="captions_with_timestamp"
    )
    parser.add_argument(
        "--hardlink", action="store_true", help="Same filesystem only; source assets must remain immutable"
    )
    parser.add_argument(
        "--train-dummy", action="store_true", help="Train on this one sample repeatedly after preparation"
    )
    parser.add_argument("--steps", type=int, default=10, help="Dummy training updates (default: 10)")
    parser.add_argument("--gpus", type=int, help="Dummy training GPU count; defaults to all visible GPUs")
    args = parser.parse_args()
    if args.steps < 1 or (args.gpus is not None and args.gpus < 1):
        parser.error("--steps and --gpus must be positive")
    try:
        training_assets = {}
        if args.train_dummy:
            import torch

            gpus = args.gpus if args.gpus is not None else torch.cuda.device_count()
            if not 1 <= gpus <= torch.cuda.device_count():
                parser.error(f"Requested {gpus} GPUs; this environment sees {torch.cuda.device_count()}")
            if args.output.expanduser().exists():
                parser.error(f"Use a fresh output directory: {args.output}")
            from cosmos_framework.data.generator.local_simulation.transfer_auto_assets import resolve_training_assets

            training_assets = resolve_training_assets(checkpoint=args.checkpoint, tokenizer=args.tokenizer)
        local_sample = args.sample_root or os.environ.get("AUTO_SAMPLE_ROOT")
        sample = inference_sample_helpers().fetch_sample(
            resolve_sample_root(Path(local_sample)) if local_sample else None
        )
        tokenizer = training_assets.get("LOCAL_QWEN_TOKENIZER") or args.tokenizer
        if tokenizer is None and os.environ.get("TRANSFER_AUTO_HF"):
            tokenizer = Path(os.environ["TRANSFER_AUTO_HF"]) / "text_tokenizer"
        if tokenizer is None:
            # Standard HF download/cache layout: snapshot/assets/sample_data.
            tokenizer = sample.parent.parent / "text_tokenizer"
        environment = prepare_sample(
            sample, Path(tokenizer), args.output, captions=args.captions, hardlink=args.hardlink
        )
        if training_assets:
            with environment.open("a") as handle:
                handle.writelines(f"export {key}={shlex.quote(value)}\n" for key, value in training_assets.items())
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(f"Downloaded sample: {sample}")
    print("Training: seven views, 61 frames per view, 480x832; online Canny controls")
    print(f"Load these training paths: source {shlex.quote(str(environment))}")
    if args.train_dummy:
        initialization = training_assets["COSMOS_TRANSFER_AUTO_EDGE_INIT_DCP"]
        command = dummy_training_command(gpus=gpus, steps=args.steps)
        output = environment.parent
        (output / "command.txt").write_text(shlex.join(command) + "\n")
        training_environment = os.environ.copy()
        training_environment.update(
            COSMOS_TRAINING="1",
            COSMOS_INTERNAL="0",
            COSMOS_TRANSFER_AUTO_LOCAL_BUNDLE=str(output / "bundle"),
            IMAGINAIRE_OUTPUT_ROOT=str(output / "training"),
            **training_assets,
        )
        print(
            f"Pretrained initialization: {initialization} (strict regular-weight load; fresh optimizer/EMA)", flush=True
        )
        print("Trainable: full shared generation experts, time embedding, video adapters and rig embeddings; LoRA off")
        print(f"Dummy training: repeat daytime_scene for {args.steps} updates on {gpus} GPUs", flush=True)
        print(shlex.join(command), flush=True)
        subprocess.run(command, cwd=Path(__file__).resolve().parents[2], env=training_environment, check=True)


if __name__ == "__main__":
    main()
