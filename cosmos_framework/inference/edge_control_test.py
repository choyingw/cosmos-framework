# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""Exercise native edge request preparation with real video decode/encode."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from cosmos_framework.data.generator.multiview.rig_catalog import MADS_AML_ALT_RIG
from cosmos_framework.inference.edge_control import write_edge_controls
from cosmos_framework.scripts.run_transfer_auto import main
from cosmos_framework.utils.edge_control import canny_control
from cosmos_framework.utils.generator.torchcodec_video import TorchCodecVideoReader
from cosmos_framework.utils.generator.video_preprocess import resize_and_center_crop_tensor

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_prepare_61_frames_seven_views_text2video_lossless_edges(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg tools are required for real media roundtrip")
    for key, value in (("COSMOS_INTERNAL", "0"), ("COSMOS_TRAINING", "0"), ("TOKENIZERS_PARALLELISM", "false")):
        monkeypatch.setenv(key, value)
    video = tmp_path / "source.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x32:rate=30",
            "-frames:v",
            "61",
            "-c:v",
            "ffv1",
            str(video),
        ],
        check=True,
    )
    caption = tmp_path / "caption.json"
    caption.write_text(
        json.dumps(
            {"caption_structured": {"chunk_0_61": dict(start_frame=0, end_frame=61, caption="A test pattern moves.")}}
        )
    )
    source = tmp_path / "input.json"
    source.write_text(
        json.dumps(
            dict(
                fps=30,
                num_frames=61,
                wsm={},
                multiview=dict(
                    views=[
                        dict(camera_key=key, vision_path=video.name, caption_path=caption.name)
                        for key in MADS_AML_ALT_RIG
                    ]
                ),
            )
        )
    )
    output = tmp_path / "run"
    repo = str(Path(__file__).resolve().parents[2])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_transfer_auto",
            "--repo",
            repo,
            "--control-type",
            "edge",
            "--request",
            str(source),
            "--mode",
            "text2video",
            "--prepare-only",
            "--output",
            str(output),
        ],
    )
    # Bound Torch's CPU thread pool for this small codec/preprocessing test.
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        main()
    finally:
        torch.set_num_threads(previous)
    request = json.loads((output / "request.json").read_text())
    assert request["num_frames"] == request["num_video_frames_per_chunk"] == 61
    assert request["resolution"] == "480" and request["edge"] == {}
    assert "wsm" not in request and "lidar" not in request
    assert request["num_conditional_frames"] == 0
    assert not request["multiview"]["condition_video_as_image"]
    rgb = TorchCodecVideoReader(str(video)).get_frames_tchw_uint8([0, 30, 60])
    expected = canny_control(resize_and_center_crop_tensor(rgb, target_h=480, target_w=832).permute(1, 0, 2, 3))
    for view in request["multiview"]["views"]:
        assert "vision_path" not in view
        reader = TorchCodecVideoReader(view["control_path"])
        assert len(reader) == 61 and reader.get_avg_fps() == pytest.approx(30, abs=0.01)
        edges = reader.get_frames_tchw_uint8([0, 30, 60]).permute(1, 0, 2, 3)
        assert torch.equal(edges, expected)


def test_edge_inputs_reject_wrong_resolution(tmp_path):
    with pytest.raises(ValueError, match="resolution"):
        write_edge_controls({"resolution": "720"}, tmp_path)
