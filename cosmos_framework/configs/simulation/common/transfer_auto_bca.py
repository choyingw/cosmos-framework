# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""Paper 5919 student distillation from a previously edge-finetuned dense teacher."""

from cosmos_framework.configs.simulation.common.transfer_auto_edge import make_transfer_auto_edge_experiment
from cosmos_framework.data.generator.multiview.rig_catalog import MADS_AML_ALT_RIG


def make_transfer_auto_bca_experiment():
    config = make_transfer_auto_edge_experiment()
    config.job.name = "bca_dummy"
    config.model.config.update(
        bca_enabled=True,
        bca_native_cross_view=True,
        bca_distillation=True,
        bca_calibration_json="${oc.env:COSMOS_BCA_CALIBRATION_JSON}",
        bca_camera_keys=list(MADS_AML_ALT_RIG),
        bca_start_block=10,
        bca_end_block=20,
        bca_local_budget=32,
        bca_global_budget=512,
        bca_global_pool=8,
        bca_inference_ratio=0.35,
        bca_prediction_weight=1.0,
        bca_feature_weight=0.5,
        bca_feature_pool=4,
        correct_cp_gradients=True,
        lora_enabled=True,
        lora_rank=16,
        lora_alpha=16,
        lora_target_modules=(
            "q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen,"
            "mlp_moe_gen.gate_proj,mlp_moe_gen.up_proj,mlp_moe_gen.down_proj"
        ),
    )
    config.model.config.compile.enabled = False
    config.optimizer.keys_to_select = ["lora_", "bca_"]
    config.checkpoint.keys_to_skip_loading = ["lora_", "bca_", "net_ema."]
    config.checkpoint.save_iter = config.trainer.max_iter = 10
    config.checkpoint.dcp_async_mode_enabled = False
    config.scheduler.update(warm_up_steps=[1], cycle_lengths=[10])
    return config
