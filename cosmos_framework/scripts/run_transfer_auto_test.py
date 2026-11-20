# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cosmos_framework.data.generator.multiview.rig_catalog import MADS_AML_ALT_RIG
from cosmos_framework.scripts.run_transfer_auto import align_bca_camera_order, write_window_captions

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_bca_alignment_keeps_camera_assets_and_prompts_together():
    views = [
        dict(
            camera_key=key,
            vision_path=f"{key}/rgb.mp4",
            control_path=f"{key}/edge.mp4",
            caption_path=f"{key}/caption.json",
            prompt=f"Caption for {key}",
        )
        for key in reversed(MADS_AML_ALT_RIG)
    ]
    original = {view["camera_key"]: view for view in views}
    request = {"multiview": {"views": views}}
    metadata = {"model": {"config": {"bca_enabled": True, "bca_camera_keys": list(MADS_AML_ALT_RIG)}}}
    align_bca_camera_order(request, metadata)
    assert [view["camera_key"] for view in views] == MADS_AML_ALT_RIG
    for view in views:
        assert view is original[view["camera_key"]]
    align_bca_camera_order(request, metadata)  # Already aligned is unchanged.
    assert [view["camera_key"] for view in views] == MADS_AML_ALT_RIG


@pytest.mark.parametrize("cameras", [MADS_AML_ALT_RIG[:-1], MADS_AML_ALT_RIG + ["unknown"], MADS_AML_ALT_RIG * 2])
def test_bca_alignment_rejects_missing_extra_or_duplicate_cameras(cameras):
    request = {"multiview": {"views": [dict(camera_key=key) for key in cameras]}}
    metadata = {"model": {"config": {"bca_enabled": True, "bca_camera_keys": list(MADS_AML_ALT_RIG)}}}
    with pytest.raises(ValueError, match="calibrated training rig|unique"):
        align_bca_camera_order(request, metadata)


def test_dense_checkpoint_keeps_original_request_order():
    request = {"multiview": {"views": [dict(camera_key=key) for key in reversed(MADS_AML_ALT_RIG)]}}
    align_bca_camera_order(request, {"model": {"config": {"bca_enabled": False}}})
    assert [view["camera_key"] for view in request["multiview"]["views"]] == list(reversed(MADS_AML_ALT_RIG))


def test_frame_window_copies_captions_and_clips_future_timestamps(tmp_path: Path, monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo / ".agents/skills/cookbooks/cosmos3/simulation/auto"))
    original = tmp_path / "original.json"
    document = {
        "caption_structured": {
            "chunk_0_300": {
                "start_frame": 0,
                "end_frame": 300,
                "caption": "Daytime. [0s-3s]: The car turns. [3s-10s]: It stops.",
            }
        }
    }
    original.write_text(json.dumps(document))
    request = {"num_frames": 61, "fps": 30, "multiview": {"views": [{"caption_path": str(original)} for _ in range(7)]}}
    write_window_captions(request, tmp_path / "run")
    assert json.loads(original.read_text()) == document
    assert request["multiview"]["use_first_chunk_only"] is True
    for view in request["multiview"]["views"]:
        path = Path(view["caption_path"])
        assert path != original
        chunk = json.loads(path.read_text())["caption_structured"]["chunk_0_61"]
        assert chunk["start_frame"] == 0 and chunk["end_frame"] == 61
        assert "It stops" not in chunk["caption"]
        assert "[0s-2.03333s]" in chunk["caption"]


@pytest.mark.parametrize("frames", [0, 1, 60, 300])
def test_invalid_frame_counts_fail_before_download(frames):
    result = subprocess.run(
        [sys.executable, "-m", "cosmos_framework.scripts.run_transfer_auto", "--frames", str(frames)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--frames must be 4k+1" in result.stderr
