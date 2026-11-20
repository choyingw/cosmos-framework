# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in registration: no changes to the native WSM curriculum."""

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.simulation.common.transfer_auto_bca import make_transfer_auto_bca_experiment
from cosmos_framework.configs.simulation.common.transfer_auto_edge import make_transfer_auto_edge_experiment

ConfigStore.instance().store(
    group="experiment", package="_global_", name="transfer_auto_edge_61_480", node=make_transfer_auto_edge_experiment()
)
ConfigStore.instance().store(
    group="experiment", package="_global_", name="transfer_auto_bca_61_480", node=make_transfer_auto_bca_experiment()
)
