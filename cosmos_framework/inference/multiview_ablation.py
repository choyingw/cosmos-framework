# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""Model construction overrides for the no-cross-view Transfer-Auto experiment."""

import copy


def same_view_model_overrides() -> list[str]:
    prefix = "model.config.multiview_attention"
    return [
        f"{prefix}.mask.attention_scope=same_view",
        f"{prefix}.mask.cross_view_temporal_window.bounds=null",
        f"{prefix}.mask.cross_view_temporal_window.includes_first_frame=false",
        f"{prefix}.maskless_cross_view_band_radius=0",
        f"{prefix}.maskless_cross_view_include_frame_zero=false",
        f"{prefix}.maskless_cross_view_attends_action=false",
        f"{prefix}.maskless_cross_view_action_attends_action=false",
        f"{prefix}.geometry_position_encoding=baseline",
    ]


def without_cross_view_attention(model: dict) -> dict:
    """Copy an exported architecture; leave source metadata/weights and unrelated settings intact."""
    from omegaconf import OmegaConf

    result = copy.deepcopy(model)
    if result["config"].get("video_temporal_causal", False):
        raise ValueError("disable_cross_view_attention supports bidirectional Transfer-Auto, not causal Sim-Auto")
    return OmegaConf.to_container(
        OmegaConf.merge({"model": result}, OmegaConf.from_dotlist(same_view_model_overrides())),
        resolve=False,
    )["model"]
