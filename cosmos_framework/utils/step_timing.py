# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in timing of complete solver steps, independent of training/inference runners."""

from __future__ import annotations

import csv
import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import torch

_RECORDER: ContextVar[StepTimingRecorder | None] = ContextVar("diffusion_step_recorder", default=None)


class StepTimingRecorder:
    def __init__(self, *, attention: bool = False):
        self.attention = attention
        self.steps: list[dict[str, Any]] = []
        self.run_index = -1
        self._regions: list[tuple[str, float, Any, Any]] | None = None
        self.device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else None

    def _event(self):
        if self.device is None:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream(self.device))
        return event

    @contextmanager
    def step(self, index: int, count: int, timestep=None):
        if index == 0:
            self.run_index += 1
        # Exclude outstanding encoding/previous-step work from this step's wall time.
        if self.device is not None:
            torch.cuda.synchronize(self.device)
        regions: list[tuple[str, float, Any, Any]] = []
        self._regions = regions
        start_event = self._event()
        start = time.perf_counter()
        status = "error"
        try:
            with torch.profiler.record_function(f"diffusion/step_{index:03d}"):
                yield
            status = "success"
        finally:
            end_event = self._event()
            if self.device is not None:
                torch.cuda.synchronize(self.device)
            elapsed = (time.perf_counter() - start) * 1000
            self._regions = None
            totals: dict[str, dict[str, Any]] = (
                {
                    name: {"calls": 0, "cpu_dispatch_ms": 0.0, "cuda_elapsed_ms": 0.0 if self.device else None}
                    for name in ("same_view", "cross_view", "caption", "attention_merge")
                }
                if self.attention
                else {}
            )
            for name, wall_ms, begin, end in regions:
                item = totals.setdefault(name, {"calls": 0, "cpu_dispatch_ms": 0.0, "cuda_elapsed_ms": None})
                item["calls"] += 1
                item["cpu_dispatch_ms"] += wall_ms
                if begin is not None:
                    item["cuda_elapsed_ms"] = (item["cuda_elapsed_ms"] or 0.0) + begin.elapsed_time(end)
            self.steps.append(
                {
                    "sampler_run": self.run_index,
                    "step_index": index,
                    "num_steps": count,
                    "timestep": float(timestep.detach().flatten()[0].cpu().item())
                    if isinstance(timestep, torch.Tensor)
                    else timestep,
                    "status": status,
                    "wall_ms": elapsed,
                    "cuda_elapsed_ms": start_event.elapsed_time(end_event) if start_event is not None else None,
                    "regions": totals,
                }
            )

    @contextmanager
    def region(self, name: str):
        begin = self._event()
        start = time.perf_counter()
        try:
            with torch.profiler.record_function(f"diffusion/{name}"):
                yield
        finally:
            end = self._event()
            assert self._regions is not None
            self._regions.append((name, (time.perf_counter() - start) * 1000, begin, end))

    def write(self, path: Path, metadata: dict[str, Any]):
        path = path.with_suffix(".csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for step in self.steps:
            row = {key: value for key, value in step.items() if key != "regions"}
            for name, values in step["regions"].items():
                row.update({f"{name}_{key}": value for key, value in values.items()})
            row.update({key: value for key, value in metadata.items() if isinstance(value, (str, int, float, bool))})
            rows.append(row)
        fields = list(dict.fromkeys(key for row in rows for key in row)) or ["step_index", "wall_ms"]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


@contextmanager
def record_diffusion_steps(path: Path, metadata: dict[str, Any], *, attention: bool = False):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else metadata.get("rank", 0)
    if rank != 0:
        yield None
        return
    recorder = StepTimingRecorder(attention=attention)
    token = _RECORDER.set(recorder)
    try:
        yield recorder
    finally:
        _RECORDER.reset(token)
        recorder.write(path, metadata)


def diffusion_step(index: int, count: int, timestep=None):
    recorder = _RECORDER.get()
    return recorder.step(index, count, timestep) if recorder is not None else nullcontext()


def attention_region(name: str, *, enabled: bool = True):
    # No ContextVar access or timer instrumentation in compiled model graphs.
    if not enabled or torch.compiler.is_compiling():
        return nullcontext()
    recorder = _RECORDER.get()
    if recorder is None or not recorder.attention or recorder._regions is None:
        return nullcontext()
    return recorder.region(name)
