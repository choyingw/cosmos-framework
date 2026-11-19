# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""Resolve released Transfer-Auto assets and cache weights-only training initialization."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

MODEL_REPO = "nvidia/Cosmos3-Nano-Transfer-Auto"
VIDEO_VAE_REPO = "Wan-AI/Wan2.2-TI2V-5B"
VIDEO_VAE_FILE = "Wan2.2_VAE.pth"


def validate_pretrained_initialization(directory: Path) -> Path:
    """Refuse absent/incomplete regular weights before launching a dummy fine-tune."""
    from torch.distributed.checkpoint import FileSystemReader

    directory = directory.expanduser().absolute()
    model = directory / "model"
    if not (model / ".metadata").is_file():
        raise FileNotFoundError(
            f"Missing pretrained DCP: {model / '.metadata'}. Convert the released Transfer-Auto snapshot "
            "with cosmos_framework.scripts.prepare_sim_auto_checkpoint first."
        )
    weights = FileSystemReader(model).read_metadata().state_dict_metadata
    regular = [name for name in weights if name.startswith("net.")]
    missing = [
        group
        for group in ("moe_gen", "time_embedder", "vae2llm", "llm2vae", "rig_view_embed")
        if not any(group in name for name in regular)
    ]
    if missing:
        raise ValueError(f"Pretrained DCP lacks regular model weights for: {', '.join(missing)}")
    return directory


def snapshot_weights(snapshot: Path) -> list[Path]:
    """Validate complete main weights, including every shard named by an index."""
    for name in ("model.safetensors.index.json", "transformer/diffusion_pytorch_model.safetensors.index.json"):
        index = snapshot / name
        if index.is_file():
            mapping = json.loads(index.read_text()).get("weight_map", {})
            weights = [index.parent / name for name in sorted(set(mapping.values()))]
            if weights and all(path.is_file() for path in weights):
                return weights
            raise FileNotFoundError(f"Incomplete pretrained weight shards in {snapshot}")
    for name in (
        "model.safetensors",
        "diffusion_pytorch_model.safetensors",
        "transformer/diffusion_pytorch_model.safetensors",
    ):
        if (snapshot / name).is_file():
            return [snapshot / name]
    raise FileNotFoundError(f"No pretrained model weights in {snapshot}")


def validate_snapshot(snapshot: Path) -> Path:
    for name in (
        "config.json",
        "text_tokenizer/tokenizer_config.json",
        "lidar_vae/diffusion_pytorch_model.safetensors",
    ):
        if not (snapshot / name).is_file():
            raise FileNotFoundError(f"Incomplete pretrained snapshot: {snapshot / name}")
    snapshot_weights(snapshot)
    return snapshot


def resolve_snapshot(checkpoint: Path | None = None) -> Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    explicit = checkpoint or os.environ.get("TRANSFER_AUTO_HF")
    if explicit:
        return validate_snapshot(Path(explicit).expanduser().absolute())
    cached = None
    try:
        cached = Path(snapshot_download(MODEL_REPO, local_files_only=True))
        return validate_snapshot(cached)
    except (LocalEntryNotFoundError, FileNotFoundError):
        pass  # A sample-only cached snapshot is not a complete model.
    print("Preparing released Transfer-Auto weights (reusing Hugging Face cache)", flush=True)
    options = {"ignore_patterns": ["assets/**"]}
    # Finish the already selected snapshot rather than mixing its tokenizer and
    # weights with a newer main revision.
    if cached is not None and len(cached.name) == 40 and all(char in "0123456789abcdef" for char in cached.name):
        options["revision"] = cached.name
    return validate_snapshot(Path(snapshot_download(MODEL_REPO, **options)))


def existing_file(path: str | Path) -> Path:
    result = Path(path).expanduser().absolute()
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def resolve_training_assets(*, checkpoint: Path | None = None, tokenizer: Path | None = None) -> dict[str, str]:
    """Honor explicit paths; otherwise obtain the same released model used by inference."""
    from filelock import FileLock
    from huggingface_hub import hf_hub_download
    from huggingface_hub.constants import HF_HOME

    initialization = os.environ.get("COSMOS_TRANSFER_AUTO_EDGE_INIT_DCP")
    video = os.environ.get("COSMOS_TRANSFER_AUTO_VIDEO_VAE") or os.environ.get("WAN_VAE_PATH")
    lidar = os.environ.get("COSMOS_TRANSFER_AUTO_LIDAR_VAE")
    tokenizer = tokenizer or os.environ.get("LOCAL_QWEN_TOKENIZER")
    snapshot = None
    if checkpoint or not all((initialization, tokenizer, lidar)):
        snapshot = resolve_snapshot(checkpoint)
    if tokenizer is None:
        tokenizer = snapshot / "text_tokenizer"
    tokenizer = Path(tokenizer).expanduser().absolute()
    if not tokenizer.is_dir():
        raise FileNotFoundError(f"Matching local text tokenizer is missing: {tokenizer}")
    if video is None:
        local_vae = Path(__file__).resolve().parents[4] / "examples/checkpoints/wan22_vae" / VIDEO_VAE_FILE
        video = local_vae if local_vae.is_file() else hf_hub_download(VIDEO_VAE_REPO, VIDEO_VAE_FILE)
    video = existing_file(video)
    lidar = existing_file(lidar or snapshot / "lidar_vae/diffusion_pytorch_model.safetensors")
    if initialization:
        initialization = validate_pretrained_initialization(Path(initialization))
    else:
        weights = snapshot_weights(snapshot)
        identity = json.dumps(
            [
                str(snapshot.absolute()),
                (snapshot / "config.json").read_text(),
                [(str(path.relative_to(snapshot)), path.stat().st_size, path.stat().st_mtime_ns) for path in weights],
            ]
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
        cache = Path(HF_HOME) / "cosmos_training/transfer_auto_init"
        cache.mkdir(parents=True, exist_ok=True)
        initialization = cache / digest
        with FileLock(str(initialization) + ".lock"):
            if initialization.exists():
                validate_pretrained_initialization(initialization)
            else:
                # Publish only a successful, validated conversion; interrupted
                # conversion attempts cannot be reused as a training checkpoint.
                with tempfile.TemporaryDirectory(prefix="converting-", dir=cache) as temporary:
                    destination = Path(temporary) / "checkpoint"
                    environment = os.environ.copy()
                    environment.update(COSMOS_INTERNAL="0", COSMOS_TRAINING="0", COSMOS_DEVICE="cpu")
                    command = [
                        sys.executable,
                        "-m",
                        "cosmos_framework.scripts.prepare_sim_auto_checkpoint",
                        "--checkpoint-path",
                        str(snapshot),
                        "--output-path",
                        str(destination),
                        "--video-vae",
                        str(video),
                    ]
                    print(f"Converting pretrained weights to reusable training DCP: {initialization}", flush=True)
                    subprocess.run(command, cwd=Path(__file__).resolve().parents[4], env=environment, check=True)
                    validate_pretrained_initialization(destination)
                    (destination / "source.json").write_text(
                        json.dumps(dict(checkpoint=str(snapshot), identity=digest))
                    )
                    destination.rename(initialization)
    result = {
        "COSMOS_TRANSFER_AUTO_EDGE_INIT_DCP": str(initialization),
        "COSMOS_TRANSFER_AUTO_VIDEO_VAE": str(video),
        "COSMOS_TRANSFER_AUTO_LIDAR_VAE": str(lidar),
        "LOCAL_QWEN_TOKENIZER": str(tokenizer),
    }
    if snapshot is not None:
        result["TRANSFER_AUTO_HF"] = str(snapshot)
    return result
