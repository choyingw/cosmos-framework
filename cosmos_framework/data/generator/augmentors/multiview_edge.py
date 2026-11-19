# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""Derive aligned controls after the common multiview RGB crop and frame draw."""

import numpy as np

from cosmos_framework.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos_framework.utils.edge_control import canny_control


class MultiviewEdgeControl(Augmentor):
    def __init__(self, *, random_thresholds: bool = True) -> None:
        super().__init__(["video"], ["control_input"])
        self.random_thresholds = random_thresholds

    def __call__(self, data: dict) -> dict:
        # One draw for the whole synchronized sample. Bosch uses lower in
        # [20,100), upper=lower+[50,150) at training; inference uses 100/200.
        lower = int(np.random.randint(20, 100)) if self.random_thresholds else 100
        upper = lower + int(np.random.randint(50, 150)) if self.random_thresholds else 200
        data["control_input"] = canny_control(data["video"], lower, upper)
        return data
