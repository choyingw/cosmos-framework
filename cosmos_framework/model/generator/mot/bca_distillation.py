# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""Frozen dense teacher plus LoRA/register student on identical packed diffusion inputs."""

from contextlib import contextmanager

import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.sequence_packing.runtime import get_gen_seq
from cosmos_framework.model.generator.mot.bca import dense_teacher


def pool_block_features(features, plan, pool, cp_mesh=None, *, correct_cp_gradients=False):
    """Pool 4x4 spatially after restoring CP's contiguous GEN sequence shards."""
    if cp_mesh is not None and cp_mesh.size() > 1:
        from cosmos_framework.model.generator.mot.context_parallel_utils import all_gather_tensor

        features = all_gather_tensor(features, 0, cp_mesh, correct_cp_gradients=correct_cp_gradients)
    if features.shape[0] < plan.num_gen_tokens:
        raise ValueError("BCA feature distillation could not restore the full packed GEN sequence")
    results, offset = [], 0
    for (frames, height, width), control in zip(plan.token_shapes, plan.is_control, strict=True):
        count = frames * height * width
        if not control:
            grid = features[offset : offset + count].reshape(frames, height, width, -1).permute(0, 3, 1, 2)
            results.append(F.avg_pool2d(grid.float(), pool, pool, ceil_mode=True).flatten())
        offset += count
    if not results:
        raise ValueError("BCA feature distillation needs generated target camera tokens")
    return torch.cat(results)


@contextmanager
def capture_features(net, pool, cp_mesh=None, *, correct_cp_gradients=False):
    captured, handles = {}, []
    for module in net.modules():
        attention = getattr(module, "self_attn", None)
        operator = getattr(attention, "bca_attention", None)
        if operator is None:
            continue

        def hook(_module, _inputs, output, operator=operator):
            if operator.last_plan is None:
                raise RuntimeError("BCA block features were captured without the camera packing plan")
            captured[operator.layer_index] = pool_block_features(
                get_gen_seq(output[0]), operator.last_plan, pool, cp_mesh, correct_cp_gradients=correct_cp_gradients
            )

        handles.append(module.register_forward_hook(hook))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def teacher_student_forward(model, packed_sequence, memory):
    net, config = model.net, model.config
    cp = model.parallel_dims.cp_mesh if model.parallel_dims.cp_enabled else None
    feature_kwargs = dict(correct_cp_gradients=getattr(config, "correct_cp_gradients", False))
    was_training = net.training
    try:
        net.eval()
        with (
            torch.no_grad(),
            dense_teacher(),
            capture_features(net, config.bca_feature_pool, cp, **feature_kwargs) as teacher_features,
        ):
            teacher = model.denoise(data_batch_packed=packed_sequence, memory=memory)
    finally:
        net.train(was_training)
    with capture_features(net, config.bca_feature_pool, cp, **feature_kwargs) as student_features:
        student = model.denoise(data_batch_packed=packed_sequence, memory=memory)
    if not teacher_features or teacher_features.keys() != student_features.keys():
        raise RuntimeError("BCA dense-teacher and budget-student block captures disagree")
    feature_loss = torch.stack(
        [F.mse_loss(student_features[key], teacher_features[key]) for key in teacher_features]
    ).mean()
    # Same target/control item ordering as the native flow-matching loss. Only
    # generated camera items contribute; control items are fully conditioned.
    from cosmos_framework.model.generator.omni_mot_model import _vision_condition_masks

    predictions = []
    for actual, reference, mask in zip(
        student["preds_vision"], teacher["preds_vision"], _vision_condition_masks(packed_sequence), strict=True
    ):
        if bool(mask.all()):
            continue
        predictions.append(F.mse_loss(actual.float(), reference.float()))
    if not predictions:
        raise ValueError("BCA prediction distillation requires generated camera targets")
    return student, {"bca_prediction_loss": torch.stack(predictions).mean(), "bca_feature_loss": feature_loss}
