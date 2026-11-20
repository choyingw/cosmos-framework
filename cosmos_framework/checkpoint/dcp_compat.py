# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""Keep DCP failure tracebacks pickleable on Python 3.13 and older PyTorch."""

from contextlib import contextmanager
from threading import RLock

from torch.distributed.checkpoint import utils as dcp_utils

_ERROR_WRAPPER_LOCK = RLock()


@contextmanager
def portable_dcp_tracebacks():
    """Backport PyTorch's FrameSummary._code cleanup for this DCP operation.

    DCP gathers wrapped exceptions across ranks. Python 3.13's frame summaries
    retain code objects, so older PyTorch fails to pickle the error, hiding the
    original disk/write failure. Preserve its exception, locations and source
    lines, clearing only the unpickleable bytecode reference as upstream does.
    Successful saves and checkpoint data are unchanged. The private wrapper is
    restored on exit; the lock also makes overlapping/nested scopes safe.
    """
    with _ERROR_WRAPPER_LOCK:
        original = dcp_utils._wrap_exception

        def wrap_exception(error):
            exception, summary = original(error)
            for frame in summary:
                if hasattr(frame, "_code"):
                    object.__setattr__(frame, "_code", None)
            return exception, summary

        dcp_utils._wrap_exception = wrap_exception
        try:
            yield
        finally:
            dcp_utils._wrap_exception = original
