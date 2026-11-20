# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""Exercise real config/DCP persistence with a lightweight stand-in for the loaded network."""

import json
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp

from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel
from cosmos_framework.scripts.prepare_sim_auto_checkpoint import main

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_converter_saves_and_reloads_public_config_with_local_tokenizer(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "text_tokenizer").mkdir()
    (source / "text_tokenizer/tokenizer_config.json").write_text("{}")
    (source / "lidar_vae").mkdir()
    (source / "lidar_vae/diffusion_pytorch_model.safetensors").touch()
    video_vae = tmp_path / "Wan2.2_VAE.pth"
    video_vae.touch()
    # Public metadata sets _use_public_model_config, the path that failed when
    # the converter replaced the tokenizer with a local directory loader.
    config = Cosmos3OmniConfig(
        model={
            "_target": "omni_mot_model",
            "config": {
                "vlm_config": {"pretrained_weights": {"enabled": True}},
                "diffusion_expert_config": {"load_weights_from_pretrained": True},
                "tokenizer": {"_target": "wan2pt2_vae_interface", "vae_path": "original"},
                "lidar_tokenizer": {"_target": "lidar_tokenizer_v1_interface"},
            },
        }
    )
    config.save_pretrained(source)
    model = torch.nn.Module()
    model.net = torch.nn.ModuleDict(
        {name: torch.nn.Linear(2, 2) for name in ("lidar2llm", "llm2lidar", "rig_view_embed")}
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.125)
    monkeypatch.setattr("cosmos_framework.inference.common.init.init_script", lambda **kwargs: None)
    monkeypatch.setattr(Cosmos3OmniModel, "from_pretrained_dcp", lambda *args, **kwargs: SimpleNamespace(model=model))
    output = tmp_path / "dcp"
    monkeypatch.setattr(
        sys,
        "argv",
        ["convert", "--checkpoint-path", str(source), "--output-path", str(output), "--video-vae", str(video_vae)],
    )
    main()
    public = json.loads((output / "model/config.json").read_text())
    assert public["model"]["config"]["vlm_config"]["tokenizer"]["_target"] == "load_bundle_tokenizer"
    restored = Cosmos3OmniConfig.from_pretrained(output / "model", local_files_only=True)
    tokenizer = restored.model["config"]["vlm_config"]["tokenizer"]
    assert tokenizer["_target_"] == "cosmos_framework.data.generator.local_simulation.dataset.load_bundle_tokenizer"
    assert tokenizer["directory"] == str(source / "text_tokenizer")
    assert restored.model["config"]["tokenizer"]["vae_path"] == str(video_vae)
    state = {name: torch.empty_like(parameter) for name, parameter in model.state_dict().items()}
    dcp.load(state, checkpoint_id=output / "model")
    for value in state.values():
        torch.testing.assert_close(value, torch.full_like(value, 0.125), rtol=0, atol=0)
