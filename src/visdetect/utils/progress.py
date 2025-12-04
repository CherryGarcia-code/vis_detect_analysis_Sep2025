from __future__ import annotations

import os
import sys
from typing import Iterable, Iterator, Optional, TypeVar, Union

T = TypeVar("T")

try:  # optional dependency
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - fallback when tqdm not installed
    tqdm = None  # type: ignore


class Progress:
    """Unified progress utility.

    Existing code instantiates Progress directly; we now optionally wrap a
    tqdm progress bar while preserving the previous API (start/update/close/iter).

    Behavior:
    - If tqdm is available AND total is known AND environment variable
      PROGRESS_SIMPLE is not set, uses a tqdm bar.
    - Else falls back to simple newline printing every ~total/ticks steps.
    - update(i) expects the absolute current index (1-based or 0-based both okay);
      we internally track and advance the tqdm bar appropriately.
    """

    def __init__(self, desc: str, total: Optional[int], ticks: int = 20, stream=None, use_tqdm: Optional[bool] = None) -> None:
        self.desc = desc
        self.total = int(total) if (total is not None) else 0
        self.ticks = max(int(ticks), 1)
        self.stream = stream or sys.stdout
        self._printed_start = False
        self._last_i = 0
        self._bar = None  # type: ignore
        # Decide whether to use tqdm
        if use_tqdm is None:
            # auto decision: only if tqdm import succeeded and total > 0 and not disabled
            auto = (tqdm is not None) and (self.total > 0) and ("PROGRESS_SIMPLE" not in os.environ)
            use_tqdm = auto
        if use_tqdm and tqdm is not None and self.total > 0:
            # use stdout so bars are visible in typical terminals, allow dynamic width
            # and keep the final bar visible (`leave=True`) for easier inspection.
            self._bar = tqdm(
                total=self.total,
                desc=self.desc,
                unit="it",
                leave=True,
                dynamic_ncols=True,
                file=self.stream,
            )
        if self.total > 0:
            self._step = max(1, self.total // self.ticks)
        else:
            self._step = 0

    def _print(self, i: int) -> None:
        if self._bar is not None:
            # advance bar to i (absolute); tqdm expects increments
            delta = i - self._bar.n
            if delta > 0:
                self._bar.update(delta)
            return
        if self.total <= 0:
            return
        pct = 100.0 * i / self.total
        print(f"{self.desc}: {i}/{self.total} ({pct:4.1f}%)", file=self.stream, flush=True)

    def start(self) -> None:
        if self._bar is not None:
            # tqdm shows immediately on first update; we keep start semantics optional
            pass
        elif self.total > 0 and not self._printed_start:
            self._print(0)
            self._printed_start = True

    def update(self, i: int) -> None:
        if self.total <= 0:
            return
        if not self._printed_start:
            self.start()
        # Always forward to tqdm; else newline strategy
        if self._bar is not None:
            self._print(i)
            self._last_i = i
        else:
            if i == 1 or i == self.total or (self._step and (i % self._step == 0)):
                if i != self._last_i:
                    self._print(i)
                    self._last_i = i

    def close(self) -> None:
        if self._bar is not None:
            # ensure bar completes
            if self._bar.n < self.total:
                self._print(self.total)
            self._bar.close()
            return
        if self.total > 0 and self._last_i != self.total:
            self._print(self.total)

    def iter(self, iterable: Iterable[T], start: int = 1) -> Iterator[T]:
        self.start()
        for i, item in enumerate(iterable, start):
            self.update(i)
            yield item
        self.close()


def progress_iter(iterable: Iterable[T], desc: str = "", total: Optional[int] = None, use_tqdm: Optional[bool] = None) -> Iterator[T]:
    """Convenience wrapper: iterate with a Progress instance.

    Usage:
        for item in progress_iter(data, desc="Processing", total=len(data)):
            ...
    """
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except Exception:
            total = None
    prog = Progress(desc=desc, total=total, use_tqdm=use_tqdm)
    yield from prog.iter(iterable)


def progress_range(n: int, desc: str = "", use_tqdm: Optional[bool] = None) -> Iterator[int]:
    """Range wrapper with progress (0..n-1)."""
    prog = Progress(desc=desc, total=n, use_tqdm=use_tqdm)
    for i in range(1, n + 1):  # maintain previous 1-based update semantics
        prog.update(i)
        yield i - 1
    prog.close()


__all__ = ["Progress", "progress_iter", "progress_range"]
