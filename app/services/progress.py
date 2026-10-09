"""Overall completion percentage of one leasing request, for the job progress API.

The request runs in stages, one after another; inside a stage every document is processed at the
same time. Each stage has a fixed share of the 100%, and within a stage the share fills as each
document finishes it - so four documents in the extraction stage move the bar a quarter of that
stage's share each, whichever order they finish in.

Only a job (see app.services.jobs) installs a tracker. Everywhere else `report` does nothing, so
the pipeline can report progress unconditionally.
"""

from contextvars import ContextVar

# (stage, share of the bar). The shares follow how long each stage usually takes and add up to 100.
STAGES: tuple[tuple[str, int], ...] = (
    ("download", 10),
    ("extraction", 30),
    ("pairing", 5),
    ("self_verification", 20),
    ("cross_verification", 10),
    ("crop_rotate_upload", 25),
)
_STAGE_START = {}
_offset = 0
for _name, _share in STAGES:
    _STAGE_START[_name] = (_offset, _share)
    _offset += _share


class ProgressTracker:
    def __init__(self) -> None:
        self._percent = 0.0
        self._finished = False

    @property
    def percent(self) -> int:
        """0-99 while running; 100 only once the request has fully ended (success or failure)."""
        if self._finished:
            return 100
        return min(99, int(self._percent))

    def report(self, stage: str, done: int, total: int) -> None:
        if stage not in _STAGE_START:
            return
        start, share = _STAGE_START[stage]
        fraction = 1.0 if total <= 0 else min(done, total) / total
        # Never moves backwards, so a bar fed from this only ever grows.
        self._percent = max(self._percent, start + share * fraction)

    def finish(self) -> None:
        self._finished = True


_current: ContextVar[ProgressTracker | None] = ContextVar("leasing_progress", default=None)


def install(tracker: ProgressTracker) -> None:
    _current.set(tracker)


def report(stage: str, done: int = 1, total: int = 1) -> None:
    """`done` of `total` documents have finished `stage`. A no-op outside a job."""
    tracker = _current.get()
    if tracker is not None:
        tracker.report(stage, done, total)
