"""Running one stage's documents concurrently, with a cap and progress reporting."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from app.services import progress

# Documents one stage of one request works on at once. Each is one Gemini call (or one download)
# in flight, so the cap keeps a large request from tripping the provider's per-minute rate limit.
MAX_PARALLEL_DOCUMENTS = 4


_T = TypeVar("_T")


async def run_concurrently(
    jobs: list[Callable[[], Awaitable[_T]]],
    *,
    stage: str | None = None,
    failed: Callable[[_T], bool] | None = None,
) -> list[_T | None]:
    """Runs every job, at most `MAX_PARALLEL_DOCUMENTS` at a time; results come back in job order.

    Each finished job moves `stage`'s share of the progress bar on. When `failed` says a result is
    a failure, every job still running or waiting is cancelled at once - its result is None - so a
    request stops at its first problem instead of finishing work that can no longer matter.
    """
    if not jobs:
        if stage:
            progress.report(stage)
        return []
    limit = asyncio.Semaphore(MAX_PARALLEL_DOCUMENTS)
    finished = 0

    async def run(job: Callable[[], Awaitable[_T]]) -> _T:
        nonlocal finished
        async with limit:
            result = await job()
        finished += 1
        if stage:
            progress.report(stage, finished, len(jobs))
        return result

    tasks = [asyncio.create_task(run(job)) for job in jobs]
    position = {task: index for index, task in enumerate(tasks)}
    results: list[_T | None] = [None] * len(jobs)
    pending = set(tasks)
    try:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            stop = False
            for task in done:
                result = task.result()
                results[position[task]] = result
                if failed is not None and failed(result):
                    stop = True
            if stop:
                break
    finally:
        # Cancels whatever is left - after a failure, or if a job raised.
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    return results
