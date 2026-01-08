from __future__ import annotations

import atexit
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import IO, Any, Iterator

_RESULTS_LOG_PATH = Path("results.log")
_results_stream: IO[str] | None = None


def get_results_stream() -> IO[str]:
    """Return a line-buffered stream for writing to results.log.

    Opened lazily and kept for the process lifetime.
    """
    global _results_stream
    if _results_stream is None or _results_stream.closed:
        _results_stream = _RESULTS_LOG_PATH.open("a", encoding="utf-8", buffering=1)
        atexit.register(_results_stream.close)
    return _results_stream


def results_print(*args: Any, **kwargs: Any) -> None:
    """Like print(), but defaults to writing into results.log instead of stdout."""
    if "file" not in kwargs or kwargs["file"] is None:
        kwargs["file"] = get_results_stream()
    print(*args, **kwargs)


@contextmanager
def redirect_stdout_to_results() -> Iterator[None]:
    """Context manager that redirects sys.stdout into results.log."""
    with redirect_stdout(get_results_stream()):
        yield
