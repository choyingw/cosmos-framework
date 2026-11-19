# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""Canny controls shared by multiview training and input preparation."""

import numpy as np
import torch


def canny_control(video: torch.Tensor, lower: int = 100, upper: int = 200) -> torch.Tensor:
    """CPU uint8 [3,T,H,W] RGB -> three identical uint8 edge channels.

    Match Bosch Transfer2.5's AddControlInputEdge: apply OpenCV Canny directly
    to RGB, without an extra grayscale conversion or blur. Resize/crop RGB first.
    """
    import cv2

    if video.ndim != 4 or video.shape[0] != 3 or video.dtype != torch.uint8 or video.device.type != "cpu":
        raise ValueError("Canny controls require CPU uint8 RGB [3,T,H,W]")
    if not 0 <= lower < upper:
        raise ValueError("Canny thresholds must satisfy 0 <= lower < upper")
    frames = video.permute(1, 2, 3, 0).contiguous().numpy()
    edges = np.stack([cv2.Canny(frame, lower, upper) for frame in frames])
    return torch.from_numpy(edges).unsqueeze(0).expand(3, -1, -1, -1).contiguous()
