# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""BCA camera projection, ported from Bosch_GenWM cross_view_budget_attn.py.

Camera models retain the source's F-theta and Waymo pinhole conventions.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Tuple

import torch
import torch.nn.functional as F

PHYSICAL_NEIGHBORS = ((1, 5, 6), (0, 6, 2), (1, 3), (2, 4), (5, 3), (0, 6, 4), (0,))


@lru_cache(maxsize=16)
def local_routes(
    calibration_json: str,
    cameras: tuple[str, ...],
    height: int,
    width: int,
    budget: int,
    depth_samples: int = 8,
    minimum: float = 1.0,
    maximum: float = 120.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU [view,pixel,budget] indices into [source_view,pixel], with validity.

    Sample eight logarithmic depths and inverse-depth-weighted neighborhoods,
    de-duplicate projected token indices, and cap each query at B_local.
    """
    calibration = json.loads(calibration_json)
    models = []
    for key in cameras:
        camera = calibration["cameras"][key]
        cls = _FThetaCamera if camera["intrinsic_model_class"] == "FThetaCameraModelParameters" else _PinholeCamera
        models.append(cls(camera))
    if budget < 1 or depth_samples < 2 or not 0 < minimum < maximum:
        raise ValueError("BCA requires a positive local budget and at least two positive depth hypotheses")
    depths = torch.logspace(math.log10(minimum), math.log10(maximum), depth_samples, dtype=torch.float64)
    counts = _allocate_depth_neighborhood_samples(depths, 128)
    offsets = _nearest_token_offsets(max(counts))
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    # The input transforms resize with preserved aspect ratio then center crop.
    pixels = torch.stack(((xx.flatten() + 0.5) * 832 / width - 0.5, (yy.flatten() + 0.5) * 480 / height - 0.5), dim=-1)
    routes = torch.zeros((len(cameras), height * width, budget), dtype=torch.long)
    valid = torch.zeros_like(routes, dtype=torch.bool)
    affines = []
    for camera in models:
        source_w, source_h = camera.resolution.tolist()
        scale = max(832 / source_w, 480 / source_h)
        resized_w, resized_h = math.ceil(source_w * scale), math.ceil(source_h * scale)
        affines.append((resized_w / source_w, resized_h / source_h, (resized_w - 832) // 2, (resized_h - 480) // 2))
    for view, target in enumerate(models):
        sx, sy, left, top = affines[view]
        image = (pixels + pixels.new_tensor([left + 0.5, top + 0.5])) / pixels.new_tensor([sx, sy]) - 0.5
        rays = target.image_points_to_rays(image)
        neighbors = (
            PHYSICAL_NEIGHBORS[view] if len(cameras) == 7 else tuple(i for i in range(len(cameras)) if i != view)
        )
        candidates = [[] for _ in range(height * width)]
        for other in neighbors:
            source = models[other]
            transform = source.rig_to_camera @ target.camera_to_rig
            for distance, count in zip(depths, counts, strict=True):
                points = (rays * distance) @ transform[:3, :3].T + transform[:3, 3]
                projected, inside = source.rays_to_image_points(points)
                sx_s, sy_s, left_s, top_s = affines[other]
                projected = (projected + 0.5) * projected.new_tensor([sx_s, sy_s]) - projected.new_tensor(
                    [left_s + 0.5, top_s + 0.5]
                )
                cell = torch.floor((projected + 0.5) / projected.new_tensor([832 / width, 480 / height])).long()
                for delta in offsets[:count]:
                    shifted = cell + delta.flip(0)  # Bosch offsets are (y,x); projected pixels are (x,y).
                    ok = (
                        inside
                        & (shifted[:, 0] >= 0)
                        & (shifted[:, 0] < width)
                        & (shifted[:, 1] >= 0)
                        & (shifted[:, 1] < height)
                    )
                    indices = other * height * width + shifted[:, 1] * width + shifted[:, 0]
                    for query in ok.nonzero().flatten().tolist():
                        candidates[query].append(int(indices[query]))
        for query, choices in enumerate(candidates):
            choices = list(dict.fromkeys(choices))
            if len(choices) > budget:
                choices = [choices[i] for i in torch.linspace(0, len(choices) - 1, budget).round().long().tolist()]
            routes[view, query, : len(choices)] = torch.tensor(choices, dtype=torch.long)
            valid[view, query, : len(choices)] = True
    return routes, valid


def _polyval(coefficients: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(x)
    for coefficient in reversed(coefficients.unbind()):
        result = result * x + coefficient
    return result


def _polyder(coefficients: torch.Tensor) -> torch.Tensor:
    if coefficients.numel() <= 1:
        return coefficients.new_zeros(1)
    powers = torch.arange(1, coefficients.numel(), dtype=coefficients.dtype, device=coefficients.device)
    return coefficients[1:] * powers


def _allocate_depth_neighborhood_samples(depths: torch.Tensor, total_samples: int) -> tuple[int, ...]:
    """Allocate a fixed candidate count across depth anchors with inverse-depth weighting."""
    if depths.ndim != 1 or depths.numel() == 0:
        raise ValueError("Depth anchors must be a non-empty 1D tensor.")
    if total_samples < depths.numel():
        raise ValueError(f"epi_neighborhood_samples={total_samples} must be at least depth_samples={depths.numel()}.")

    minimum_per_anchor = 2 if total_samples >= 2 * depths.numel() else 1
    counts = torch.full((depths.numel(),), minimum_per_anchor, dtype=torch.long)
    remaining = total_samples - minimum_per_anchor * depths.numel()
    if remaining == 0:
        return tuple(int(value) for value in counts.tolist())

    weights = depths.to(torch.float64).reciprocal()
    exact_extra = weights / weights.sum() * remaining
    floor_extra = torch.floor(exact_extra).to(torch.long)
    counts += floor_extra
    leftover = remaining - int(floor_extra.sum().item())
    if leftover:
        fractional = exact_extra - floor_extra
        # Stable tie-breaking favors closer anchors.
        order = sorted(range(depths.numel()), key=lambda index: (-float(fractional[index]), index))
        counts[order[:leftover]] += 1
    return tuple(int(value) for value in counts.tolist())


def _nearest_token_offsets(count: int) -> torch.Tensor:
    """Return integer 2D offsets ordered from the center toward the neighborhood boundary."""
    if count <= 0:
        raise ValueError("Neighborhood offset count must be positive.")
    radius = 0
    while (2 * radius + 1) ** 2 < count:
        radius += 1
    offsets = [
        (offset_y, offset_x) for offset_y in range(-radius, radius + 1) for offset_x in range(-radius, radius + 1)
    ]
    offsets.sort(
        key=lambda offset: (
            offset[0] * offset[0] + offset[1] * offset[1],
            abs(offset[0]) + abs(offset[1]),
            offset[0],
            offset[1],
        )
    )
    return torch.tensor(offsets[:count], dtype=torch.long)


class _FThetaCamera:
    """Small torch implementation of NVIDIA NCore's F-theta projection model."""

    def __init__(self, camera: dict) -> None:
        if camera.get("intrinsic_model_class") != "FThetaCameraModelParameters":
            raise ValueError(
                f"Only FThetaCameraModelParameters is supported, got {camera.get('intrinsic_model_class')}"
            )
        intrinsics = camera["intrinsics"]
        self.resolution = torch.tensor(intrinsics["resolution"], dtype=torch.float64)
        self.principal_point = torch.tensor(intrinsics["principal_point"], dtype=torch.float64)
        self.backward_poly = torch.tensor(intrinsics["pixeldist_to_angle_poly"], dtype=torch.float64)
        self.forward_poly = torch.tensor(intrinsics["angle_to_pixeldist_poly"], dtype=torch.float64)
        reference_poly = intrinsics["reference_poly"]
        if isinstance(reference_poly, str):
            reference_poly = {"PIXELDIST_TO_ANGLE": 1, "ANGLE_TO_PIXELDIST": 2}[reference_poly]
        self.reference_poly = int(reference_poly)
        self.max_angle = float(intrinsics["max_angle"])
        c, d, e = intrinsics["linear_cde"]
        self.linear = torch.tensor([[c, d], [e, 1.0]], dtype=torch.float64)
        self.linear_inverse = torch.linalg.inv(self.linear)
        self.camera_to_rig = torch.tensor(camera["T_camera_rig"], dtype=torch.float64)
        self.rig_to_camera = torch.tensor(camera["T_rig_camera"], dtype=torch.float64)
        if not torch.allclose(
            self.camera_to_rig @ self.rig_to_camera,
            torch.eye(4, dtype=torch.float64),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError("T_camera_rig and T_rig_camera are not inverse transforms.")

    @staticmethod
    def _invert_polynomial(
        reference: torch.Tensor,
        initial_inverse: torch.Tensor,
        values: torch.Tensor,
        iterations: int = 3,
    ) -> torch.Tensor:
        estimate = _polyval(initial_inverse, values)
        derivative = _polyder(reference)
        for _ in range(iterations):
            residual = _polyval(reference, estimate) - values
            slope = _polyval(derivative, estimate)
            safe_slope = torch.where(slope.abs() < 1e-12, torch.full_like(slope, 1e-12), slope)
            estimate = estimate - residual / safe_slope
        return estimate

    def image_points_to_rays(self, image_points: torch.Tensor) -> torch.Tensor:
        sensor_points = torch.einsum(
            "ij,nj->ni", self.linear_inverse, image_points.to(torch.float64) - self.principal_point
        )
        radii = torch.linalg.vector_norm(sensor_points, dim=-1, keepdim=True)
        if self.reference_poly == 1:  # PIXELDIST_TO_ANGLE
            theta = _polyval(self.backward_poly, radii)
        else:
            theta = self._invert_polynomial(self.forward_poly, self.backward_poly, radii)
        denominator = radii.clamp_min(1e-12)
        rays = torch.cat((torch.sin(theta) * sensor_points / denominator, torch.cos(theta)), dim=-1)
        principal = radii.squeeze(-1) < 1e-12
        if principal.any():
            rays[principal] = rays.new_tensor([0.0, 0.0, 1.0])
        return rays

    def rays_to_image_points(self, rays: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rays = rays.to(torch.float64)
        xy_norm = torch.linalg.vector_norm(rays[..., :2], dim=-1, keepdim=True).clamp_min(1e-12)
        theta = torch.atan2(xy_norm, rays[..., 2:3])
        if self.reference_poly == 1:  # Invert the reference backward polynomial.
            pixel_distance = self._invert_polynomial(self.backward_poly, self.forward_poly, theta)
        else:
            pixel_distance = _polyval(self.forward_poly, theta)
        sensor_points = pixel_distance * rays[..., :2] / xy_norm
        image_points = torch.einsum("ij,...j->...i", self.linear, sensor_points) + self.principal_point
        valid = (
            (theta.squeeze(-1) < self.max_angle)
            & (image_points[..., 0] >= 0.0)
            & (image_points[..., 0] < self.resolution[0])
            & (image_points[..., 1] >= 0.0)
            & (image_points[..., 1] < self.resolution[1])
        )
        return image_points, valid


class _PinholeCamera:
    """Waymo Brown-Conrady pinhole model in the dataset's forward-left-up camera frame."""

    def __init__(self, camera: dict) -> None:
        if camera.get("intrinsic_model_class") != "PinholeCameraModelParameters":
            raise ValueError(f"Expected PinholeCameraModelParameters, got {camera.get('intrinsic_model_class')!r}.")
        intrinsics = camera["intrinsics"]
        self.resolution = torch.tensor(intrinsics["resolution"], dtype=torch.float64)
        self.focal = torch.tensor(intrinsics["focal"], dtype=torch.float64)
        self.principal_point = torch.tensor(intrinsics["principal_point"], dtype=torch.float64)
        self.distortion = torch.tensor(intrinsics.get("distortion", [0.0] * 5), dtype=torch.float64)
        if any(value.shape != (2,) for value in (self.resolution, self.focal, self.principal_point)):
            raise ValueError("Pinhole resolution, focal, and principal_point must each contain two values.")
        if torch.any(self.resolution <= 0) or torch.any(self.focal <= 0):
            raise ValueError("Pinhole resolution and focal lengths must be positive.")
        if self.distortion.shape != (5,):
            raise ValueError("Pinhole distortion must contain [k1, k2, p1, p2, k3].")
        self.camera_to_rig = torch.tensor(camera["T_camera_rig"], dtype=torch.float64)
        self.rig_to_camera = torch.tensor(camera["T_rig_camera"], dtype=torch.float64)
        if not torch.allclose(
            self.camera_to_rig @ self.rig_to_camera,
            torch.eye(4, dtype=torch.float64),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError("T_camera_rig and T_rig_camera are not inverse transforms.")

    def _distort(self, xy: torch.Tensor) -> torch.Tensor:
        k1, k2, p1, p2, k3 = self.distortion
        x, y = xy.unbind(dim=-1)
        radius2 = x.square() + y.square()
        radial = 1.0 + k1 * radius2 + k2 * radius2.square() + k3 * radius2.pow(3)
        dx = x * radial + 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x.square())
        dy = y * radial + p1 * (radius2 + 2.0 * y.square()) + 2.0 * p2 * x * y
        return torch.stack((dx, dy), dim=-1)

    def image_points_to_rays(self, image_points: torch.Tensor) -> torch.Tensor:
        distorted = (image_points.to(torch.float64) - self.principal_point) / self.focal
        undistorted = distorted.clone()
        # Invert Waymo's low-order Brown-Conrady coefficients by fixed-point iteration.
        for _ in range(8):
            undistorted = undistorted + distorted - self._distort(undistorted)
        # In Waymo camera coordinates +x points forward, +y points left, and +z points up.
        return F.normalize(torch.cat((torch.ones_like(undistorted[..., :1]), -undistorted), dim=-1), dim=-1)

    def rays_to_image_points(self, rays: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rays = rays.to(torch.float64)
        forward = rays[..., 0]
        normalized_image = torch.stack(
            (-rays[..., 1] / forward.clamp_min(1e-12), -rays[..., 2] / forward.clamp_min(1e-12)), dim=-1
        )
        image_points = self._distort(normalized_image) * self.focal + self.principal_point
        valid = (
            (forward > 1e-8)
            & torch.isfinite(image_points).all(dim=-1)
            & (image_points[..., 0] >= 0.0)
            & (image_points[..., 0] < self.resolution[0])
            & (image_points[..., 1] >= 0.0)
            & (image_points[..., 1] < self.resolution[1])
        )
        return image_points, valid
