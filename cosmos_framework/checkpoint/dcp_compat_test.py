# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

import errno
import pickle
import traceback
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import utils as dcp_utils
from torch.distributed.checkpoint.api import CheckpointException

from cosmos_framework.checkpoint.dcp_compat import portable_dcp_tracebacks

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class _FrameWithCode(traceback.FrameSummary):
    """Emulate Python 3.13's bytecode reference when running tests on older Python."""


def _old_wrap_exception(error):
    summary = traceback.StackSummary()
    for original in traceback.extract_tb(error.__traceback__):
        frame = _FrameWithCode(original.filename, original.lineno, original.name, line=original.line)
        frame._code = _old_wrap_exception.__code__
        summary.append(frame)
    return error, summary


def test_portable_error_preserves_cause_and_locations_and_restores_wrapper(monkeypatch):
    monkeypatch.setattr(dcp_utils, "_wrap_exception", _old_wrap_exception)
    try:
        raise OSError(errno.ENOSPC, "No space left on device", "/checkpoint/model/__1_0.distcp")
    except OSError as error:
        with pytest.raises(TypeError, match="code"):
            pickle.dumps(_old_wrap_exception(error))
        with portable_dcp_tracebacks():
            exception, summary = dcp_utils._wrap_exception(error)
            assert exception is error
            restored, frames = pickle.loads(pickle.dumps((exception, summary)))
            assert restored.errno == errno.ENOSPC
            assert restored.filename == error.filename
            assert frames[0].filename == summary[0].filename
            assert frames[0].lineno == summary[0].lineno
        assert dcp_utils._wrap_exception is _old_wrap_exception
    with pytest.raises(RuntimeError, match="scope failed"):
        with portable_dcp_tracebacks():
            with portable_dcp_tracebacks():
                raise RuntimeError("scope failed")
    assert dcp_utils._wrap_exception is _old_wrap_exception


def _distributed_save_worker(rank, rendezvous, root):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    original = dcp_utils._wrap_exception
    dcp_utils._wrap_exception = _old_wrap_exception
    try:
        # Exercise the same real collectives as the user's eight-rank save.
        class FailingWriter(dcp.FileSystemWriter):
            def write_data(self, plan, planner):
                if rank == 1:
                    raise OSError(errno.ENOSPC, "injected disk full", root + "/failed/__1_0.distcp")
                return super().write_data(plan, planner)

        with pytest.raises(CheckpointException) as caught:
            with portable_dcp_tracebacks():
                dcp.save({f"weight_{rank}": torch.arange(4)}, storage_writer=FailingWriter(root + "/failed"))
        error, frames = caught.value.failures[1]
        assert error.errno == errno.ENOSPC
        assert "injected disk full" in str(caught.value)
        assert frames and all(frame._code is None for frame in frames)
        assert dcp_utils._wrap_exception is _old_wrap_exception

        # Failure still permits a later successful save/load; no state is dropped.
        source = {f"weight_{rank}": torch.arange(4) + rank}
        with portable_dcp_tracebacks():
            dcp.save(source, checkpoint_id=root + "/success")
        actual = {f"weight_{rank}": torch.zeros(4, dtype=torch.long)}
        dcp.load(actual, checkpoint_id=root + "/success")
        torch.testing.assert_close(actual[f"weight_{rank}"], source[f"weight_{rank}"])
    finally:
        dcp_utils._wrap_exception = original
        dist.destroy_process_group()


def test_distributed_write_error_reports_original_failure_on_every_rank(tmp_path):
    torch.multiprocessing.spawn(
        _distributed_save_worker,
        args=((tmp_path / "rendezvous").as_uri(), str(tmp_path)),
        nprocs=2,
        join=True,
    )
