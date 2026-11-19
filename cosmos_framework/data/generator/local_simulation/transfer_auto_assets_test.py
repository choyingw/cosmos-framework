# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp

from cosmos_framework.data.generator.local_simulation import transfer_auto_assets as assets

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _snapshot(root):
    root.mkdir()
    for name in ("text_tokenizer", "lidar_vae"):
        (root / name).mkdir()
    (root / "config.json").write_text("{}")
    (root / "text_tokenizer/tokenizer_config.json").write_text("{}")
    (root / "model.safetensors").write_bytes(b"checked-by-the-native-converter")
    (root / "lidar_vae/diffusion_pytorch_model.safetensors").write_bytes(b"vae")
    return root


def _save_dcp(directory):
    dcp.save(
        {
            f"net.{name}.weight": torch.ones(2, 2)
            for name in ("q_proj_moe_gen", "time_embedder", "vae2llm", "llm2vae", "rig_view_embed")
        },
        checkpoint_id=directory / "model",
    )


@pytest.fixture
def automatic_assets(tmp_path, monkeypatch):
    for key in (
        "TRANSFER_AUTO_HF",
        "LOCAL_QWEN_TOKENIZER",
        "WAN_VAE_PATH",
        "COSMOS_TRANSFER_AUTO_EDGE_INIT_DCP",
        "COSMOS_TRANSFER_AUTO_VIDEO_VAE",
        "COSMOS_TRANSFER_AUTO_LIDAR_VAE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("huggingface_hub.constants.HF_HOME", str(tmp_path / "hf_cache"))
    snapshot = _snapshot(tmp_path / "snapshot")
    video = tmp_path / assets.VIDEO_VAE_FILE
    video.write_bytes(b"video-vae")
    calls = []
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *args, **kwargs: str(snapshot))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *args, **kwargs: str(video))

    def convert(command, **kwargs):
        calls.append((command, kwargs))
        _save_dcp(Path(command[command.index("--output-path") + 1]))

    monkeypatch.setattr(assets.subprocess, "run", convert)
    return snapshot, video, calls


def test_missing_env_automatically_resolves_and_converts_pretrained_assets(automatic_assets):
    snapshot, video, calls = automatic_assets
    result = assets.resolve_training_assets()
    initialization = Path(result["COSMOS_TRANSFER_AUTO_EDGE_INIT_DCP"])
    assert assets.validate_pretrained_initialization(initialization) == initialization
    assert result["COSMOS_TRANSFER_AUTO_VIDEO_VAE"] == str(video)
    assert result["COSMOS_TRANSFER_AUTO_LIDAR_VAE"] == str(snapshot / "lidar_vae/diffusion_pytorch_model.safetensors")
    assert result["LOCAL_QWEN_TOKENIZER"] == str(snapshot / "text_tokenizer")
    command, options = calls[0]
    assert command[command.index("--checkpoint-path") + 1] == str(snapshot)
    assert command[command.index("--video-vae") + 1] == str(video)
    assert options["env"]["COSMOS_DEVICE"] == "cpu"
    assert options["env"]["COSMOS_TRAINING"] == "0"
    assert json.loads((initialization / "source.json").read_text())["checkpoint"] == str(snapshot)


def test_conversion_is_reused_and_model_change_gets_a_new_cache(automatic_assets):
    snapshot, _, calls = automatic_assets
    first = assets.resolve_training_assets()
    assert assets.resolve_training_assets() == first
    assert len(calls) == 1
    (snapshot / "config.json").write_text('{"changed":true}')
    second = assets.resolve_training_assets()
    assert second["COSMOS_TRANSFER_AUTO_EDGE_INIT_DCP"] != first["COSMOS_TRANSFER_AUTO_EDGE_INIT_DCP"]
    assert len(calls) == 2


def test_sample_only_cache_is_completed_at_the_same_revision(tmp_path, monkeypatch):
    monkeypatch.delenv("TRANSFER_AUTO_HF", raising=False)
    partial = tmp_path / ("a" * 40)
    partial.mkdir()
    complete = _snapshot(tmp_path / "complete")
    calls = []

    def download(*args, **kwargs):
        calls.append(kwargs)
        return str(partial if kwargs.get("local_files_only") else complete)

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    assert assets.resolve_snapshot() == complete
    assert calls[1]["revision"] == partial.name


def test_failed_conversion_is_not_published(automatic_assets, monkeypatch):
    _, _, _ = automatic_assets

    def fail(*args, **kwargs):
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(assets.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="conversion failed"):
        assets.resolve_training_assets()
    from huggingface_hub.constants import HF_HOME

    cache = Path(HF_HOME) / "cosmos_training/transfer_auto_init"
    assert not [path for path in cache.iterdir() if path.is_dir()]


def test_snapshot_requires_every_pretrained_shard(tmp_path):
    snapshot = _snapshot(tmp_path / "sharded")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer1": "model-00001.safetensors", "layer2": "model-00002.safetensors"}})
    )
    (snapshot / "model-00001.safetensors").touch()
    with pytest.raises(FileNotFoundError, match="Incomplete pretrained weight shards"):
        assets.validate_snapshot(snapshot)
