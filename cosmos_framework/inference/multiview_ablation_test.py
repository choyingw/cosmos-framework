# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

import copy
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from cosmos_framework.inference.multiview_ablation import without_cross_view_attention

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_ablation_preserves_own_view_controls_captions_and_checkpoint_metadata():
    model = {
        "config": {
            "multiview_attention": {
                "backend": "maskless",
                "geometry_position_encoding": "prope_cross_view",
                "mask": {
                    "attention_scope": "decomposed",
                    "control_attends_sensor": True,
                    "lidar_attends_captions": False,
                    "cross_view_temporal_window": {"bounds": [-0.4, 0.0], "includes_first_frame": True},
                    "same_view_temporal_window": {"sensor_to_sensor_bounds": [-0.8, 0.8]},
                },
                "maskless_cross_view_band_radius": 2,
                "maskless_cross_view_attends_action": True,
            },
            "tokenizer": {"path": "unchanged"},
        }
    }
    source = copy.deepcopy(model)
    result = without_cross_view_attention(model)
    attention = result["config"]["multiview_attention"]
    assert model == source
    assert attention["mask"]["attention_scope"] == "same_view"
    assert attention["mask"]["control_attends_sensor"] is True
    assert attention["mask"]["lidar_attends_captions"] is False
    assert attention["mask"]["same_view_temporal_window"] == {"sensor_to_sensor_bounds": [-0.8, 0.8]}
    assert attention["mask"]["cross_view_temporal_window"] == {"bounds": None, "includes_first_frame": False}
    assert attention["geometry_position_encoding"] == "baseline"
    assert attention["maskless_cross_view_band_radius"] == 0
    assert not attention["maskless_cross_view_attends_action"]
    assert result["config"]["tokenizer"] == source["config"]["tokenizer"]


def test_causal_student_ablation_is_rejected():
    with pytest.raises(ValueError, match="not causal Sim-Auto"):
        without_cross_view_attention({"config": {"video_temporal_causal": True}})


def test_cli_flags_parse_and_round_trip():
    import tyro

    from cosmos_framework.inference.args import OmniSetupOverrides

    setup = tyro.cli(
        OmniSetupOverrides,
        args=[
            "--checkpoint-path",
            "unused-test-checkpoint",
            "--profile-steps",
            "--profile-step-attention",
            "--disable-cross-view-attention",
            "--no-use-torch-compile",
            "--no-use-cuda-graphs",
        ],
    )
    setup._build_setup()
    restored = OmniSetupOverrides(**setup.model_dump())
    assert restored.profile_steps and restored.profile_step_attention and restored.disable_cross_view_attention
    assert not restored.use_torch_compile and not restored.use_cuda_graphs


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"profile_step_attention": True}, "requires profile_steps"),
        ({"profile_steps": True, "profile_step_attention": True, "use_torch_compile": True}, "no-use-torch-compile"),
        ({"profile_steps": True, "profile_step_attention": True, "use_cuda_graphs": True}, "no-use-cuda-graphs"),
    ],
)
def test_detail_profiling_rejects_uninstrumented_execution(overrides, message):
    from cosmos_framework.inference.args import OmniSetupOverrides

    with pytest.raises(ValueError, match=message):
        OmniSetupOverrides(checkpoint_path="unused-test-checkpoint", **overrides)._build_setup()


def test_warmup_measured_and_rank_reports_are_separate(tmp_path: Path, monkeypatch):
    from cosmos_framework.inference.common import inference as module
    from cosmos_framework.utils.step_timing import diffusion_step

    monkeypatch.setattr(module, "sync_distributed_errors", nullcontext)
    monkeypatch.setattr(module.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(module.torch.distributed, "get_rank", lambda: 3)
    sample = SimpleNamespace(name="seven_views", output_dir=tmp_path)
    saves = []

    class Inference(module.Inference):
        @property
        def model_config(self):
            return None

        @classmethod
        def _create(cls, *args, **kwargs):
            raise NotImplementedError

        def create_batches(self, samples):
            yield samples, {}

        def generate_batch(self, samples, data_batch, *, save_outputs=True):
            saves.append(save_outputs)
            for i in range(2):
                with diffusion_step(i, 2):
                    pass
            return []

    setup = SimpleNamespace(
        profile=False,
        warmup=1,
        num_iterations=2,
        profile_steps=True,
        profile_step_attention=False,
        disable_cross_view_attention=True,
    )
    pipe = Inference(setup_args=setup, model=None, guardrails=None, _timer=None)
    pipe.generate([sample])
    assert saves == [True, False, False]
    paths = sorted(tmp_path.glob("*.json"))
    assert len(paths) == len(list(tmp_path.glob("*.csv"))) == 3
    for path in paths:
        report = json.loads(path.read_text())
        assert len(report["steps"]) == 2
        assert report["metadata"]["rank"] == 3
        assert report["metadata"]["disable_cross_view_attention"] is True
    assert [json.loads(path.read_text())["metadata"]["phase"] for path in paths].count("warmup") == 1
