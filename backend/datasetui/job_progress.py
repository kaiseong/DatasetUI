from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol


class _ProgressDatabase(Protocol):
    def update_job_progress(
        self, job_id: str, *, worker_id: str, progress: dict[str, Any]
    ) -> None: ...


class JobProgressReporter:
    """Persist measured job progress without turning hot loops into DB writes."""

    def __init__(
        self,
        database: _ProgressDatabase,
        *,
        job_id: str,
        worker_id: str,
        output_name: str = "",
        interval_seconds: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._database = database
        self._job_id = job_id
        self._worker_id = worker_id
        self._output_name = output_name
        self._interval_seconds = interval_seconds
        self._clock = clock
        self._started = clock()
        self._last_emit: float | None = None
        self._last_stage: str | None = None

    def __call__(self, progress: dict[str, Any]) -> None:
        event = dict(progress)
        force = bool(event.pop("_force", False))
        now = self._clock()
        stage = str(event["stage"])
        completed = int(event["completed"])
        total = int(event["total"])
        stage_changed = stage != self._last_stage
        stage_finished = total > 0 and completed >= total
        interval_elapsed = (
            self._last_emit is None or now - self._last_emit >= self._interval_seconds
        )
        if not (force or stage_changed or stage_finished or interval_elapsed):
            return
        payload = {
            "stage": stage,
            "completed": completed,
            "total": total,
            "unit": str(event["unit"]),
            "elapsed_seconds": round(now - self._started, 3),
            "output_name": self._output_name,
        }
        current_item = event.get("current_item")
        if current_item:
            payload["current_item"] = str(current_item)
        overall = event.get("overall")
        if overall is not None:
            payload["overall"] = round(min(1.0, max(0.0, float(overall))), 4)
        self._database.update_job_progress(
            self._job_id,
            worker_id=self._worker_id,
            progress=payload,
        )
        self._last_emit = now
        self._last_stage = stage


class WeightedProgress:
    """Whole-job progress from a plan of stages weighted by expected work.

    ``plan`` maps stage name -> expected cost (any unit, e.g. frame-equivalents),
    in execution order. Each event's fraction of its own stage is combined with
    the completed stages before it. Stages missing from the plan pass through
    without an overall value; ``complete`` is always 100 %. Estimates only grow
    (``grow``) so the bar never moves backwards when work exceeds the plan.
    """

    def __init__(self, reporter: Callable[[dict[str, Any]], None], plan: dict[str, float]):
        self._reporter = reporter
        self._plan = {stage: max(0.0, float(cost)) for stage, cost in plan.items()}
        self._best = 0.0

    def grow(self, stage: str, cost: float) -> None:
        if stage in self._plan and cost > self._plan[stage]:
            self._plan[stage] = float(cost)

    def __call__(self, progress: dict[str, Any]) -> None:
        event = dict(progress)
        stage = str(event["stage"])
        if stage == "complete":
            event["overall"] = 1.0
        elif stage in self._plan:
            total = int(event.get("total") or 0)
            completed = int(event.get("completed") or 0)
            if total > 0 and completed > total:
                self.grow(stage, self._plan[stage] * completed / total)
            fraction = min(1.0, completed / total) if total > 0 else 0.0
            stages = list(self._plan)
            before = sum(self._plan[name] for name in stages[: stages.index(stage)])
            whole = sum(self._plan.values()) or 1.0
            self._best = max(self._best, (before + self._plan[stage] * fraction) / whole)
            event["overall"] = min(0.99, self._best)
        self._reporter(event)


ProgressCallback = Callable[[dict[str, Any]], None]


def report_progress(
    callback: ProgressCallback | None,
    *,
    stage: str,
    completed: int,
    total: int,
    unit: str,
    current_item: str | None = None,
    force: bool = False,
) -> None:
    if callback is None:
        return
    progress: dict[str, Any] = {
        "stage": stage,
        "completed": completed,
        "total": total,
        "unit": unit,
    }
    if current_item is not None:
        progress["current_item"] = current_item
    if force:
        progress["_force"] = True
    callback(progress)
