# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

import csv
import time
from pathlib import Path

import pytest
import torch

from cosmos_framework.utils.step_timing import attention_region, diffusion_step, record_diffusion_steps

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_step_reports_keep_runs_and_regions_separate(tmp_path: Path):
    path = tmp_path / "steps.csv"
    with record_diffusion_steps(path, {"phase": "measured", "rank": 0}, attention=True) as recorder:
        for _ in range(2):
            for index in range(3):
                with diffusion_step(index, 3, torch.tensor(900 - 100 * index)):
                    for _ in range(2):  # Two CFG evaluations in one solver step.
                        with attention_region("same_view"):
                            time.sleep(0.001)
                    with attention_region("cross_view", enabled=False):
                        pass
    steps = recorder.steps
    assert [(step["sampler_run"], step["step_index"]) for step in steps] == [
        (run, index) for run in range(2) for index in range(3)
    ]
    for step in steps:
        assert step["status"] == "success" and step["wall_ms"] >= 2
        assert step["regions"]["same_view"]["calls"] == 2
        assert step["regions"]["cross_view"]["calls"] == 0
        assert step["timestep"] == 900 - 100 * step["step_index"]
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6 and rows[0]["same_view_calls"] == "2"
    assert rows[0]["rank"] == "0" and rows[0]["phase"] == "measured"
    assert not list(tmp_path.glob("*.json"))


def test_exception_writes_partial_step_and_restores_recorder(tmp_path: Path):
    path = tmp_path / "partial.csv"
    with pytest.raises(RuntimeError, match="failed forward"):
        with record_diffusion_steps(path, {}, attention=True) as recorder:
            with diffusion_step(0, 1):
                with attention_region("same_view"):
                    raise RuntimeError("failed forward")
    assert recorder.steps[0]["status"] == "error"
    assert recorder.steps[0]["regions"]["same_view"]["calls"] == 1
    with diffusion_step(0, 1), attention_region("same_view"):
        pass
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1 and rows[0]["status"] == "error"


@pytest.mark.parametrize("kind", ["unipc", "fixed", "edm", "component"])
def test_profiling_preserves_sampler_outputs_and_includes_scheduler(tmp_path: Path, kind: str, monkeypatch):
    from cosmos_framework.model.generator.diffusion.samplers.edm import EDMSampler
    from cosmos_framework.model.generator.diffusion.samplers.fixed_step import FixedStepSampler
    from cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler, UniPCSamplerConfig
    from cosmos_framework.model.generator.utils.component_unipc import ComponentSchedule, sample_component_unipc

    original = FlowUniPCMultistepScheduler.step

    def slow_scheduler(self, *args, **kwargs):
        time.sleep(0.002)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(FlowUniPCMultistepScheduler, "step", slow_scheduler)
    noise = torch.randn(2, 2)
    callbacks = []

    def velocity(latent, timestep):
        with attention_region("same_view"):
            if isinstance(latent, list):
                return [value * 0.1 for value in latent]
            return latent * 0.1

    def sample():
        def callback(i, n):
            callbacks.append((i, n))

        if kind == "component":
            return sample_component_unipc(
                velocity,
                [noise.flatten().clone()],
                cfg=UniPCSamplerConfig(),
                schedule=ComponentSchedule(shifts=(5.0, 10.0), lengths=[(2, 2)]),
                num_steps=3,
                step_callback=callback,
            )[0]
        if kind == "unipc":
            sampler = UniPCSampler(tensor_kwargs={"device": torch.device("cpu")})
        elif kind == "fixed":
            sampler = FixedStepSampler(t_list=[1.0, 0.6, 0.3, 0.0])
        else:
            sampler = EDMSampler()
            return sampler(velocity, noise.clone(), num_steps=3, step_callback=callback)
        return sampler(velocity, noise.clone(), num_steps=3, seed=0, step_callback=callback)

    expected = sample()
    callbacks.clear()
    path = tmp_path / "sampler.csv"
    with record_diffusion_steps(path, {}, attention=True) as recorder:
        actual = sample()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    steps = recorder.steps
    assert [(step["step_index"], step["num_steps"]) for step in steps] == callbacks
    assert len(steps) == 3
    assert all(step["regions"]["same_view"]["calls"] >= 1 for step in steps)
    if kind in ("unipc", "component"):
        assert all(step["wall_ms"] >= 2 for step in steps)


def test_cuda_events_are_read_only_after_step_synchronization(tmp_path: Path, monkeypatch):
    operations = []

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self, stream):
            operations.append("record")

        def elapsed_time(self, end):
            assert operations[-1] == "sync"
            return 1.25

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: object())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: operations.append("sync"))
    monkeypatch.setattr(torch.cuda, "Event", Event)
    path = tmp_path / "cuda.csv"
    with record_diffusion_steps(path, {}, attention=True) as recorder:
        with diffusion_step(0, 1):
            with attention_region("same_view"):
                pass
    step = recorder.steps[0]
    assert step["cuda_elapsed_ms"] == step["regions"]["same_view"]["cuda_elapsed_ms"] == 1.25
    assert step["regions"]["cross_view"]["cuda_elapsed_ms"] == 0
    assert operations.count("sync") == 2


def test_nonzero_rank_neither_times_nor_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)
    monkeypatch.setattr(
        "cosmos_framework.utils.step_timing.StepTimingRecorder",
        lambda **kwargs: pytest.fail("Nonzero rank created a timer"),
    )
    with record_diffusion_steps(tmp_path / "rank3.csv", {}, attention=True) as recorder:
        assert recorder is None
        with diffusion_step(0, 1), attention_region("same_view"):
            pass
    assert not list(tmp_path.glob("*.csv")) and not list(tmp_path.glob("*.json"))
