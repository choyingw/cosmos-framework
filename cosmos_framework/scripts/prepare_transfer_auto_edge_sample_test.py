# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import shlex
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp

from cosmos_framework.data.generator.multiview.rig_catalog import MADS_AML_ALT_RIG
from cosmos_framework.scripts import prepare_transfer_auto_edge_sample as setup
from cosmos_framework.scripts.prepare_transfer_auto_edge_sample import prepare_sample, resolve_sample_root

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _sample(tmp_path):
    root = tmp_path / "download/assets/sample_data"
    scene = root / "daytime_scene"
    (scene / "rgb").mkdir(parents=True)
    (scene / "captions_with_timestamp").mkdir()
    for camera in MADS_AML_ALT_RIG:
        (scene / "rgb" / f"{camera}.mp4").write_bytes(b"recorded-rgb-validated-on-decode")
        (scene / "captions_with_timestamp" / f"{camera}.json").write_text(
            json.dumps(
                {
                    "caption_structured": {
                        "chunk_0_300": dict(start_frame=0, end_frame=300, caption="Supplied recording caption.")
                    }
                }
            )
        )
    return root


def _pretrained(tmp_path, *, omit=None):
    directory = tmp_path / "pretrained"
    state = {
        f"net.{name}.weight": torch.full((2, 2), 0.125)
        for name in ("q_proj_moe_gen", "time_embedder", "vae2llm", "llm2vae", "rig_view_embed")
        if name != omit
    }
    dcp.save(state, checkpoint_id=directory / "model")
    return directory


def test_sample_preparation_builds_only_recorded_rgb_and_caption_assets(tmp_path):
    root = _sample(tmp_path)
    tokenizer = tmp_path / "text_tokenizer"
    tokenizer.mkdir()
    output = tmp_path / "training with spaces"
    environment = prepare_sample(root, tokenizer, output, captions="captions_with_timestamp")
    index = json.loads((output / "training-index.jsonl").read_text())
    assert len(index["assets"]) == 14
    assert all(key.startswith(("video_", "caption_")) for key in index["assets"])
    assert all(
        str(root / "daytime_scene/rgb") in path for key, path in index["assets"].items() if key.startswith("video_")
    )
    for camera in MADS_AML_ALT_RIG:
        selected = json.loads((output / "captions" / f"{camera}.json").read_text())["caption_structured"]["chunk_0_61"]
        assert selected["start_frame"] == 0 and selected["end_frame"] == 61
        original = json.loads((root / "daytime_scene/captions_with_timestamp" / f"{camera}.json").read_text())
        assert original["caption_structured"]["chunk_0_300"]["end_frame"] == 300
    manifest = json.loads((output / "bundle/manifest.json").read_text())
    policy = manifest["transforms"]["480"]
    assert policy["num_video_frames"] == 61 and policy["resolution_hw"] == [480, 832]
    assert policy["camera_control_type"] == "edge" and not policy["load_lidar"]
    variables = dict(shlex.split(line)[1].split("=", 1) for line in environment.read_text().splitlines())
    assert variables["AUTO_SAMPLE_ROOT"] == str(root)
    assert variables["COSMOS_TRANSFER_AUTO_LOCAL_BUNDLE"] == str(output / "bundle")


def test_resolve_existing_download_layouts_and_hf_cache(tmp_path, monkeypatch):
    root = _sample(tmp_path)
    for argument in (root, root / "daytime_scene", tmp_path / "download"):
        assert resolve_sample_root(argument) == root
    monkeypatch.setenv("AUTO_SAMPLE_ROOT", str(root))
    assert resolve_sample_root() == root
    monkeypatch.delenv("AUTO_SAMPLE_ROOT")
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *args: str(root / "input-transfer-auto.jsonl"))
    assert resolve_sample_root() == root


def test_missing_sample_assets_fail_before_creating_bundle(tmp_path):
    root = _sample(tmp_path)
    (root / "daytime_scene/rgb" / f"{MADS_AML_ALT_RIG[-1]}.mp4").unlink()
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    output = tmp_path / "output"
    with pytest.raises(FileNotFoundError):
        prepare_sample(root, tokenizer, output, captions="captions_with_timestamp")
    assert not output.exists()


def test_dummy_launch_uses_the_inference_sample_and_one_sequence(tmp_path, monkeypatch):
    root = _sample(tmp_path)
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    output = tmp_path / "dummy"
    calls = []
    helpers = setup.inference_sample_helpers()
    monkeypatch.setattr(
        setup,
        "inference_sample_helpers",
        lambda: SimpleNamespace(
            fetch_sample=lambda source: calls.append(source) or root,
            caption_window=helpers.caption_window,
        ),
    )
    monkeypatch.delenv("AUTO_SAMPLE_ROOT", raising=False)
    pretrained = _pretrained(tmp_path)
    vae = tmp_path / "vae.safetensors"
    vae.touch()
    monkeypatch.setenv("COSMOS_TRANSFER_AUTO_EDGE_INIT_DCP", str(pretrained))
    for key in ("COSMOS_TRANSFER_AUTO_VIDEO_VAE", "COSMOS_TRANSFER_AUTO_LIDAR_VAE"):
        monkeypatch.setenv(key, str(vae))
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    launched = []
    monkeypatch.setattr(setup.subprocess, "run", lambda command, **kwargs: launched.append((command, kwargs)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "setup",
            "--train-dummy",
            "--steps",
            "3",
            "--tokenizer",
            str(tokenizer),
            "--output",
            str(output),
        ],
    )
    setup.main()
    assert calls == [None]  # Exactly fetch_sample(None), as in run_transfer_auto's default path.
    samples = json.loads((output / "bundle/samples.json").read_text())
    assert len(samples) == 1 and samples[0]["identity"] == "daytime_scene"
    command, options = launched[0]
    assert "trainer.max_iter=3" in command and "checkpoint.save_iter=3" in command
    assert "--nproc-per-node=2" in command
    assert "model_parallel.context_parallel_size=2" in command
    assert options["env"]["COSMOS_TRAINING"] == "1"
    assert options["env"]["COSMOS_TRANSFER_AUTO_LOCAL_BUNDLE"] == str(output / "bundle")
    assert options["env"]["COSMOS_TRANSFER_AUTO_EDGE_INIT_DCP"] == str(pretrained)
    assert options["env"]["IMAGINAIRE_OUTPUT_ROOT"] == str(output / "training")


def test_pretrained_preflight_requires_real_dcp_and_all_adapted_modules(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing pretrained DCP"):
        setup.validate_pretrained_initialization(tmp_path / "missing")
    incomplete = _pretrained(tmp_path, omit="q_proj_moe_gen")
    with pytest.raises(ValueError, match="moe_gen"):
        setup.validate_pretrained_initialization(incomplete)


def test_pretrained_preflight_accepts_covered_regular_weights(tmp_path):
    pretrained = _pretrained(tmp_path)
    assert setup.validate_pretrained_initialization(pretrained) == pretrained
