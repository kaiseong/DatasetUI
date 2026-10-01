"""Cooperative running-job cancellation with bounded child-process cleanup."""

from __future__ import annotations

import ctypes
import logging
import os
import signal
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol


logger = logging.getLogger("datasetui.worker")


class JobCancellationRequested(RuntimeError):
    """Raised inside a worker after its owner requested cancellation."""


class _CancellationDatabase(Protocol):
    def is_job_cancellation_requested(self, job_id: str, *, worker_id: str) -> bool: ...


@dataclass(frozen=True)
class _ProcessIdentity:
    pid: int
    start_time: str
    state: str = field(compare=False)
    parent_pid: int = field(compare=False)


def _identity(pid: int, proc_root: Path) -> _ProcessIdentity | None:
    try:
        value = (proc_root / str(pid) / "stat").read_text()
    except (FileNotFoundError, OSError, PermissionError):
        return None
    closing = value.rfind(")")
    if closing < 0:
        return None
    fields = value[closing + 1 :].split()
    if len(fields) <= 19:
        return None
    try:
        parent_pid = int(fields[1])
    except ValueError:
        return None
    return _ProcessIdentity(
        pid=pid,
        start_time=fields[19],
        state=fields[0],
        parent_pid=parent_pid,
    )


def _children(pid: int, proc_root: Path) -> list[int]:
    task_root = proc_root / str(pid) / "task"
    try:
        tasks = list(task_root.iterdir())
    except (FileNotFoundError, OSError, PermissionError):
        return []
    children: set[int] = set()
    for task in tasks:
        try:
            value = (task / "children").read_text()
        except (FileNotFoundError, OSError, PermissionError):
            continue
        children.update(int(item) for item in value.split() if item.isdecimal())
    return list(children)


def _descendants(parent_pid: int, proc_root: Path) -> list[_ProcessIdentity]:
    result: list[_ProcessIdentity] = []
    pending = _children(parent_pid, proc_root)
    seen: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        identity = _identity(pid, proc_root)
        if identity is None:
            continue
        result.append(identity)
        pending.extend(_children(pid, proc_root))
    return result


def _same_process(identity: _ProcessIdentity, proc_root: Path) -> bool:
    current = _identity(identity.pid, proc_root)
    return current is not None and current.start_time == identity.start_time


def _active_identities(
    identities: set[_ProcessIdentity], proc_root: Path
) -> list[_ProcessIdentity]:
    active: list[_ProcessIdentity] = []
    for identity in identities:
        current = _identity(identity.pid, proc_root)
        if current is not None and current.start_time == identity.start_time:
            active.append(current)
    return active


def _reap_adopted_zombies(
    identities: list[_ProcessIdentity], *, parent_pid: int, proc_root: Path
) -> None:
    if proc_root != Path("/proc"):
        return
    for identity in identities:
        if identity.parent_pid != parent_pid or identity.state != "Z":
            continue
        try:
            os.waitpid(identity.pid, os.WNOHANG)
        except (ChildProcessError, ProcessLookupError):
            pass


def _enable_child_subreaper() -> int | None:
    """Adopt orphaned job descendants until cancellation cleanup completes."""

    if sys.platform != "linux":
        logger.warning("hard subprocess cancellation requires Linux child subreapers")
        return None
    libc = ctypes.CDLL(None, use_errno=True)
    current = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(current), 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, "unable to read child subreaper state")
    if current.value == 0 and libc.prctl(36, 1, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, "unable to enable child subreaper")
    return current.value


def _restore_child_subreaper(previous: int | None) -> None:
    if previous != 0:
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 0, 0, 0, 0) != 0:
        logger.error(
            "unable to restore child subreaper state: errno=%s", ctypes.get_errno()
        )


def terminate_job_subprocesses(
    parent_pid: int,
    *,
    grace_seconds: float = 2.0,
    proc_root: Path = Path("/proc"),
) -> None:
    """Terminate verified descendants, including children spawned during cleanup."""

    tracked: set[_ProcessIdentity] = set()
    term_signalled: set[_ProcessIdentity] = set()
    deadline = time.monotonic() + max(0.0, grace_seconds)
    while time.monotonic() < deadline:
        descendants = _descendants(parent_pid, proc_root)
        tracked.update(descendants)
        active = _active_identities(tracked, proc_root)
        _reap_adopted_zombies(active, parent_pid=parent_pid, proc_root=proc_root)
        tracked.update(_descendants(parent_pid, proc_root))
        active = _active_identities(tracked, proc_root)
        if not active:
            return
        for identity in reversed(descendants):
            if identity not in active or identity in term_signalled:
                continue
            try:
                os.kill(identity.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            term_signalled.add(identity)
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    kill_deadline = time.monotonic() + 1.0
    while True:
        tracked.update(_descendants(parent_pid, proc_root))
        active = _active_identities(tracked, proc_root)
        _reap_adopted_zombies(active, parent_pid=parent_pid, proc_root=proc_root)
        tracked.update(_descendants(parent_pid, proc_root))
        active = _active_identities(tracked, proc_root)
        if not active:
            return
        for identity in reversed(list(active)):
            try:
                os.kill(identity.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        if time.monotonic() >= kill_deadline:
            return
        time.sleep(0.05)


@contextmanager
def cancellation_monitor(
    database: _CancellationDatabase,
    *,
    job_id: str,
    worker_id: str,
    poll_seconds: float = 0.25,
    child_grace_seconds: float = 2.0,
) -> Iterator[None]:
    """Interrupt the main job and reap descendants after a persisted request."""

    stop = threading.Event()
    observed = threading.Event()
    children_cleaned = threading.Event()
    cleanup_lock = threading.Lock()
    previous_subreaper = _enable_child_subreaper()
    main_thread = threading.current_thread() is threading.main_thread()
    previous_handler = None

    def requested() -> bool:
        try:
            return database.is_job_cancellation_requested(job_id, worker_id=worker_id)
        except Exception:
            return False

    def interrupt(_signum, _frame) -> None:
        if not stop.is_set() and observed.is_set():
            raise JobCancellationRequested(job_id)

    def cleanup_children() -> None:
        with cleanup_lock:
            if children_cleaned.is_set():
                return
            terminate_job_subprocesses(
                os.getpid(), grace_seconds=child_grace_seconds
            )
            children_cleaned.set()

    if main_thread:
        previous_handler = signal.getsignal(signal.SIGUSR1)
        signal.signal(signal.SIGUSR1, interrupt)

    def watch() -> None:
        while not stop.wait(poll_seconds):
            if not requested():
                continue
            if stop.is_set():
                return
            observed.set()
            if main_thread and not stop.is_set():
                os.kill(os.getpid(), signal.SIGUSR1)
            else:
                cleanup_children()
            return

    thread = threading.Thread(
        target=watch,
        name=f"datasetui-cancel-{job_id}",
        daemon=True,
    )
    thread.start()
    try:
        yield
        if requested():
            raise JobCancellationRequested(job_id)
    except JobCancellationRequested:
        stop.set()
        observed.set()
        cleanup_children()
        raise
    finally:
        stop.set()
        thread.join()
        if main_thread and previous_handler is not None:
            signal.signal(signal.SIGUSR1, previous_handler)
        _restore_child_subreaper(previous_subreaper)
