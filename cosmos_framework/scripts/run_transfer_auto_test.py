# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cosmos_framework.scripts.run_transfer_auto import write_window_captions

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


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
