# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from cosmos_framework.configs.simulation.common.transfer_auto_bca import make_transfer_auto_bca_experiment
from cosmos_framework.inference.common.public_model_config import (
    build_public_model_config,
    load_model_config_from_hf_config,
)
from cosmos_framework.scripts.prepare_transfer_auto_edge_sample import dummy_training_command

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_bca_recipe_matches_paper_and_survives_export_metadata(monkeypatch):
    calibration = (
        Path(__file__).resolve().parents[2] / "configs/simulation/assets/bca_daytime_calibration.json"
    ).read_text()
    monkeypatch.setenv("COSMOS_BCA_CALIBRATION_JSON", calibration)
    monkeypatch.setenv("COSMOS_TRANSFER_AUTO_LOCAL_BUNDLE", "/training/bundle")
    monkeypatch.setenv("COSMOS_TRANSFER_AUTO_EDGE_INIT_DCP", "/teacher/dcp")
    monkeypatch.setenv("COSMOS_TRANSFER_AUTO_VIDEO_VAE", "/video.pth")
    monkeypatch.setenv("COSMOS_TRANSFER_AUTO_LIDAR_VAE", "/lidar.safetensors")
    config = make_transfer_auto_bca_experiment()
    model = config.model.config
    assert config.trainer.max_iter == config.checkpoint.save_iter == 10
    assert model.lora_enabled and model.lora_rank == model.lora_alpha == 16
    assert list(config.optimizer.keys_to_select) == ["lora_", "bca_"]
    assert model.bca_local_budget == 32 and model.bca_global_budget == 512
    assert model.bca_start_block == 10 and model.bca_end_block == 20
    assert model.bca_prediction_weight == 1.0 and model.bca_feature_weight == 0.5
    assert model.bca_inference_ratio == 0.35
    assert model.correct_cp_gradients
    assert model.bca_native_cross_view
    assert not config.checkpoint.load_training_state and config.checkpoint.strict_resume
    assert "bca_" in config.checkpoint.keys_to_skip_loading
    plain = OmegaConf.to_container(config.model, resolve=True)
    # Exercise public metadata without relying on the training dataloader path.
    public = build_public_model_config(plain)
    restored = load_model_config_from_hf_config(json.loads(json.dumps({"model": public})))
    assert restored["config"]["bca_calibration_json"] == calibration
    assert restored["config"]["bca_inference_ratio"] == 0.35
    assert restored["config"]["bca_native_cross_view"] is True
    command = dummy_training_command(gpus=8, steps=10, bca=True)
    assert "experiment=transfer_auto_bca_61_480" in command and "job.name=bca_dummy" in command
