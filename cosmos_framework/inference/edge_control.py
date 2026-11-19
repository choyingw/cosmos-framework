# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""Prepare lossless per-camera Canny controls from synchronized source RGB."""

import math
import subprocess
from pathlib import Path

from cosmos_framework.utils.edge_control import canny_control
from cosmos_framework.utils.generator.torchcodec_video import TorchCodecVideoReader
from cosmos_framework.utils.generator.video_preprocess import resize_and_center_crop_tensor


def write_edge_controls(request: dict, output: Path) -> None:
    """Write the first requested frames at 480x832 with Bosch medium thresholds.

    Decode in bounded batches, using the training crop/resize and shared Canny
    implementation. Lossless RGB H.264 keeps binary maps in the native MP4 format.
    """
    if request["resolution"] != "480":
        raise ValueError("The edge adaptation uses resolution=480 (480x832)")
    frames = request["num_frames"]
    folder = output / "edge_controls"
    folder.mkdir(exist_ok=False)
    for index, view in enumerate(request["multiview"]["views"]):
        if not view.get("vision_path"):
            raise ValueError(f"RGB-derived edge controls need vision_path for {view['camera_key']}")
        reader = TorchCodecVideoReader(str(view["vision_path"]))
        # MKV millisecond timestamps can yield e.g. 30.0049 for a nominal
        # 30 FPS clip. Accept that rounding without accepting a 25/30 FPS mix.
        if len(reader) < frames or not math.isclose(reader.get_avg_fps(), request["fps"], abs_tol=0.05):
            raise ValueError(f"RGB timeline must cover {frames} frames at {request['fps']} FPS: {view['camera_key']}")
        destination = folder / f"view_{index:02d}.mp4"
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-n",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            "832x480",
            "-r",
            str(request["fps"]),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264rgb",
            "-crf",
            "0",
            "-preset",
            "fast",
            str(destination),
        ]
        with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as encoder:
            try:
                for start in range(0, frames, 8):
                    rgb = reader.get_frames_tchw_uint8(list(range(start, min(start + 8, frames))))
                    rgb = resize_and_center_crop_tensor(rgb, target_h=480, target_w=832)
                    edges = canny_control(rgb.permute(1, 0, 2, 3))
                    encoder.stdin.write(edges.permute(1, 2, 3, 0).contiguous().numpy().tobytes())
                encoder.stdin.close()
                error = encoder.stderr.read().decode(errors="replace")
                if encoder.wait() != 0:
                    raise RuntimeError(f"ffmpeg edge encoding failed: {error}")
            except BaseException:
                encoder.kill()
                encoder.wait()
                destination.unlink(missing_ok=True)
                raise
        view["control_path"] = str(destination)
