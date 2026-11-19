# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU tests of edge adaptation; GPU loss/backward and checkpoint readback remain separate."""

import json

import cv2
import numpy as np
import pytest
import torch

from cosmos_framework.configs.simulation.common.transfer_auto_edge import make_transfer_auto_edge_experiment
from cosmos_framework.data.generator.augmentors.multiview_edge import MultiviewEdgeControl
from cosmos_framework.data.generator.augmentors.text_tokenizer import _SYSTEM_PROMPTS, TextTokenizerTransformForTransfer
from cosmos_framework.data.generator.local_simulation.auto_bundle import build_bundle, default_transforms
from cosmos_framework.data.generator.local_simulation.core import decode_config
from cosmos_framework.data.generator.local_simulation.dataset import _model_constructors
from cosmos_framework.data.generator.multiview.local_pipeline import make_auto_augmentations
from cosmos_framework.data.generator.multiview.rig_catalog import MADS_AML_ALT_RIG
from cosmos_framework.data.generator.multiview.transforms import ExtractFramesAndCaptions
from cosmos_framework.inference.common.multiview_captions import per_view_caption_export_fields
from cosmos_framework.utils.edge_control import canny_control
from cosmos_framework.utils.generator.optimizer import _build_params_with_metadata

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_edges_match_bosch_rgb_canny_and_keep_target():
    video = torch.randint(0, 256, (3, 4, 16, 32), dtype=torch.uint8)
    original = video.clone()
    sample = MultiviewEdgeControl(random_thresholds=False)({"video": video})
    expected = np.stack([cv2.Canny(frame, 100, 200) for frame in video.permute(1, 2, 3, 0).contiguous().numpy()])
    assert torch.equal(sample["control_input"][0], torch.from_numpy(expected))
    assert torch.equal(sample["control_input"][0], sample["control_input"][2])
    assert torch.equal(video, original)
    assert sample["video"] is video


def test_random_training_thresholds_match_bosch(monkeypatch):
    draws = iter([24, 80])
    monkeypatch.setattr(np.random, "randint", lambda low, high: next(draws))
    video = torch.randint(0, 256, (3, 2, 16, 32), dtype=torch.uint8)
    sample = MultiviewEdgeControl()({"video": video})
    assert torch.equal(sample["control_input"], canny_control(video, 24, 104))


def test_seven_synchronized_views_61_frames_and_no_wsm(monkeypatch):
    draws = []

    def decode(video, indices, resolution, **kwargs):
        draws.append(list(indices))
        frames = torch.zeros((len(indices), 3, *resolution), dtype=torch.uint8)
        frames[:, :, 4:12, 8:24] = 255
        return frames, 30.0, resolution

    monkeypatch.setattr(ExtractFramesAndCaptions, "_extract_frames", staticmethod(decode))
    policy = default_transforms(MADS_AML_ALT_RIG, 480, control_type="edge")
    assert policy["resolution_hw"] == [480, 832]
    policy.update(resolution_hw=[16, 32], edge_random_thresholds=False)
    sample = {"__key__": "scene", "__url__": "local://test"}
    for camera in MADS_AML_ALT_RIG:
        sample[f"video_{camera}"] = b"decoded-by-test"
        sample[f"caption_{camera}"] = {
            "caption_structured": {"chunk": dict(start_frame=0, end_frame=120, caption="A car drives on a dry road.")}
        }
    for transform in make_auto_augmentations(decode_config(policy, constructors=_model_constructors("auto"))):
        sample = transform(sample)
        assert sample is not None
    controls, targets = sample["video"]
    assert controls.shape == targets.shape == (3, 7 * 61, 16, 32)
    assert len(draws) == 7 and all(draw == draws[0] for draw in draws)
    assert len(draws[0]) == 61
    assert torch.equal(controls, canny_control(targets))
    assert sample["num_video_frames_per_view"].item() == 61
    assert sample["sample_n_views"].item() == 7
    assert sample["enable_per_camera_vae_encoding"]


def test_edge_bundle_requires_only_rgb_captions_and_tokenizer(tmp_path):
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    video = tmp_path / "rgb.mp4"
    video.write_bytes(b"media-validated-by-loader")
    caption = tmp_path / "caption.json"
    caption.write_text(
        json.dumps({"caption_structured": {"chunk": dict(start_frame=0, end_frame=61, caption="Daytime.")}})
    )
    assets = {}
    for camera in MADS_AML_ALT_RIG:
        assets[f"video_{camera}"] = video.name
        assets[f"caption_{camera}"] = caption.name
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps(dict(id="scene", stratum="480", assets=assets)) + "\n")
    output = tmp_path / "bundle"
    build_bundle(index, tokenizer, output, MADS_AML_ALT_RIG, control_type="edge")
    manifest = json.loads((output / "manifest.json").read_text())
    assert set(manifest["transforms"]) == {"480"}
    policy = manifest["transforms"]["480"]
    assert policy["camera_control_key_mapping"] is None
    assert policy["require_full_num_frames"] and policy["num_video_frames"] == 61
    assert not policy["load_lidar"] and not policy["joint_cam_lidar_transfer"]
    assert policy["camera_sampler"] is None
    with pytest.raises(ValueError, match="seven"):
        default_transforms(MADS_AML_ALT_RIG[:6], 480, control_type="edge")


def test_dense_generation_recipe_fresh_state_and_export_prompt(monkeypatch):
    monkeypatch.setenv("COSMOS_TRANSFER_AUTO_EDGE_INIT_DCP", "/models/transfer_auto_dcp")
    config = make_transfer_auto_edge_experiment()
    assert config.model.config.lora_enabled is False
    assert list(config.optimizer.keys_to_select) == ["moe_gen", "time_embedder", "vae2llm", "llm2vae", "rig_view_embed"]
    assert config.checkpoint.load_path == "/models/transfer_auto_dcp"
    assert config.checkpoint.strict_resume and not config.checkpoint.load_training_state
    assert list(config.checkpoint.keys_to_skip_loading) == ["net_ema."]
    assert list(config.checkpoint.keys_not_to_resume) == []
    assert config.model.config.ema.iteration_shift == 0
    assert config.trainer.callbacks.initialize_edge_ema.source_checkpoint == "/models/transfer_auto_dcp"
    assert config.model.config.multiview_attention.mask.attention_scope == "decomposed"
    fields, _, _ = per_view_caption_export_fields(config)
    assert fields["per_view_captions"] is True
    assert fields["per_view_caption_task"] == "av_multiview_edge_transfer"
    policy = default_transforms(MADS_AML_ALT_RIG, 480, control_type="edge")
    policy = decode_config(policy, constructors=_model_constructors("auto"))
    policy["tokenizer_config"] = {"_target_": dict}
    tokenizer = next(t for t in make_auto_augmentations(policy) if isinstance(t, TextTokenizerTransformForTransfer))
    assert tokenizer.args["task"] == fields["per_view_caption_task"]
    assert "Canny edge" in _SYSTEM_PROMPTS[fields["per_view_caption_task"]]
    assert "WSM" not in _SYSTEM_PROMPTS[fields["per_view_caption_task"]]


def test_native_optimizer_freezes_language_and_lidar_but_keeps_dense_generation():
    model = torch.nn.Module()
    model.net = torch.nn.ModuleDict(
        {
            key: torch.nn.Linear(2, 2)
            for key in (
                "q_proj_moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
                "rig_view_embed",
                "q_proj",
                "lidar2llm",
                "llm2lidar",
            )
        }
    )
    config = make_transfer_auto_edge_experiment()
    selected = _build_params_with_metadata(
        model, list(config.optimizer.keys_to_select), {}, config.optimizer.lr, 0.05, False
    )
    assert len(selected) == 10  # Full weights and biases of all five selected modules.
    for key, layer in model.net.items():
        expected = key not in ("q_proj", "lidar2llm", "llm2lidar")
        assert all(parameter.requires_grad == expected for parameter in layer.parameters())
