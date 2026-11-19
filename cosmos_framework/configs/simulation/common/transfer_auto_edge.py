# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""Dense camera generation adaptation from released WSM weights to edge controls."""

from cosmos_framework.callbacks.initialize_ema import InitializeEmaFromRegular
from cosmos_framework.configs.simulation.common.transfer_auto_recipes import (
    AUTO_BUNDLE,
    make_transfer_auto_experiment,
)
from cosmos_framework.data.generator.local_simulation.dataloader import LocalSimulationDataLoader
from cosmos_framework.data.generator.local_simulation.dataset import LocalSimulationDataset
from cosmos_framework.data.generator.multiview.rig_catalog import MADS_AML_ALT_RIG
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict


def make_transfer_auto_edge_experiment() -> LazyDict:
    """Seven simultaneous views, exactly 61 frames/view, 480x832, no LiDAR task.

    Unlike Transfer2.5, MoT has no independent ControlNet. Adapt full shared
    generation experts/video adapters, freezing the language backbone and LiDAR
    adapters. No LoRA or same-view-only attention ablation is enabled.
    """
    config = make_transfer_auto_experiment(4)
    config.job.name = "transfer_auto_edge_61_480"
    model = config.model.config
    model.lora_enabled = False
    model.max_num_tokens_after_packing = 300_000
    model.ema.iteration_shift = 0
    model.parallelism.update(data_parallel_shard_degree=8, context_parallel_shard_degree=8)
    config.model_parallel.context_parallel_size = 8
    config.optimizer.lr = 1e-5
    config.optimizer.keys_to_select = ["moe_gen", "time_embedder", "vae2llm", "llm2vae", "rig_view_embed"]
    config.scheduler.update(f_start=[0.1], f_max=[1.0], f_min=[0.1], warm_up_steps=[100], cycle_lengths=[5000])
    source = "${oc.env:COSMOS_TRANSFER_AUTO_EDGE_INIT_DCP}"
    config.checkpoint.update(
        load_path=source,
        load_training_state=False,
        load_ema_to_reg=False,
        strict_resume=True,
        keys_to_skip_loading=["net_ema."],
        keys_not_to_resume=[],
        save_iter=100,
    )
    config.trainer.max_iter = 5000
    config.trainer.callbacks.initialize_edge_ema = L(InitializeEmaFromRegular)(source_checkpoint=source)
    config.dataloader_train.restart_data_iterator_on_handoff = False
    config.dataloader_train.dataloaders = dict(
        edge_480=dict(
            ratio=1.0,
            dataloader=L(LocalSimulationDataLoader)(
                dataset=L(LocalSimulationDataset)(
                    bundle_path=AUTO_BUNDLE,
                    model="auto",
                    stratum="480",
                    seed=42,
                    transform_overrides=dict(
                        camera_keys=list(MADS_AML_ALT_RIG),
                        camera_sampler=None,
                        num_cameras_to_keep=None,
                        camera_control_type="edge",
                        camera_control_key_mapping=None,
                        edge_random_thresholds=True,
                        resolution_hw=[480, 832],
                        num_video_frames=61,
                        require_full_num_frames=True,
                        load_lidar=False,
                        lidar_transfer=False,
                        joint_cam_lidar_transfer=False,
                        lidar_dropout_rate=0.0,
                        separate_view_text_tokenization=True,
                        first_caption_paragraph_only=False,
                        conditioning_config={"__mapping__": [[0, 0.5], [1, 0.5]]},
                    ),
                ),
                batch_size=1,
                num_workers=0,
                pin_memory=False,
                seed=42,
            ),
        )
    )
    return config
