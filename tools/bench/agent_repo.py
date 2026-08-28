"""Realistic multi-file Python repository generator for coding-agent long-context tests.

Generates the `nimbus-queue` project (a task-queue service) with real logic, realistic
logs, docs, and tests. Five context-size profiles (8K/16K/32K/48K/64K tokens, calibrated
chars/token) expose increasing amounts of the repository. Three defects are planted with
mechanically checkable ground truth:

  bug-retry      queue.enqueue_with_retry: off-by-one (>= max_retries vs > max_retries)
  bug-backoff    config.ENV_OVERRIDES: env key mismatch (NIMBUS_RETRY_BACKOFF_SECS vs
                 NIMBUS_RETRY_BACKOFF_SECONDS) so operator backoff=5 silently falls back to 30
  bug-workers    config.validate: no guard for max_workers == 0, so an empty pool starts
                 silently and queued jobs never run

Usage:
    python3 tools/bench/agent_repo.py emit --profile 64k --out /tmp/nimbus64k
    python3 tools/bench/agent_repo.py prompt --profile 64k --out /tmp/p64.txt
    python3 tools/bench/agent_repo.py run --profiles 8k,16k,32k,48k,64k \
        --results results/agent_bench/agent_q4.jsonl
    python3 tools/bench/agent_repo.py score --results results/agent_bench/agent_q4.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Source files (realistic logic; defects planted)
# ---------------------------------------------------------------------------

PYPROJECT = """[project]
name = "nimbus-queue"
version = "0.9.3"
description = "In-process task queue with retries, dead letters, and pluggable worker backends."
requires-python = ">=3.11"
dependencies = []

[project.scripts]
nimbus = "nimbus.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
"""

MAKEFILE = """PY ?= python3
.PHONY: test run lint
test:
\t$(PY) -m pytest -q
run:
\t$(PY) -m nimbus.cli --config config.json
lint:
\t$(PY) -m compileall -q nimbus tests
"""

README = """# nimbus-queue

In-process task queue with retries, dead letters, metrics, and pluggable worker
backends. Designed to be embedded in a service process (no broker dependency).

## Architecture

- `nimbus/config.py` — configuration defaults, environment overrides, and `validate()`.
- `nimbus/queue.py` — priority queue plus `enqueue_with_retry` retry/dead-letter policy.
- `nimbus/store.py` — in-memory job store with attempt counters and TTLs.
- `nimbus/backends.py` — worker backend registry. A backend is a class exposing
  `name`, `submit(fn, *args)`, and `shutdown()`. Registered backends are the only
  ones the worker pool may run on.
- `nimbus/worker.py` — worker pool; pulls jobs from the queue and dispatches them
  to the configured backend.
- `nimbus/api.py` — HTTP-less request router (method, path pattern -> handler).
- `nimbus/notify.py` — template-based notification dispatch.
- `nimbus/metrics.py` — counters, gauges, histograms with 1-minute rollups.
- `nimbus/audit.py` — JSONL audit trail for state transitions.
- `nimbus/cli.py` — entry point; loads config, applies env overrides, builds the
  queue/worker stack.

## Configuration

`config.json` supplies values; environment variables override them:

| Env var | Field | Default |
|---|---|---|
| `NIMBUS_MAX_WORKERS` | `max_workers` | 4 |
| `NIMBUS_MAX_PRIORITY` | `max_priority` | 100 |
| `NIMBUS_DEAD_LETTER_TTL_HOURS` | `dead_letter_ttl_hours` | 72 |
| `NIMBUS_RETRY_BACKOFF_SECS` | `retry_backoff_seconds` | 30 |
| `NIMBUS_RATE_PER_SEC` | `rate_per_sec` | 50 |

All env overrides are parsed as integers. `validate()` rejects out-of-range values
before the stack starts.

## Retries

A job that fails is re-enqueued with exponential backoff until it has been retried
`max_retries` times (default 3); after that it moves to the dead-letter store and is
kept for `dead_letter_ttl_hours`.

## Backends

Only registered backends may be selected via `--backend`. The built-in registry
ships `threadpool` and `inline`.
"""

INIT = '''"""nimbus-queue: in-process task queue with retries and pluggable backends."""

__version__ = "0.9.3"
'''

ERRORS = '''"""Exception hierarchy for nimbus-queue."""

from __future__ import annotations


class NimbusError(Exception):
    """Base class for all nimbus errors."""


class ConfigError(NimbusError):
    """Raised when configuration fails validation or override parsing."""


class JobError(NimbusError):
    """Raised for malformed job payloads."""

    def __init__(self, job_id: str, message: str) -> None:
        super().__init__(f"job {job_id}: {message}")
        self.job_id = job_id


class BackendError(NimbusError):
    """Raised when a worker backend cannot perform an operation."""


class DeadLetterRejected(NimbusError):
    """Raised when a dead-letter push is rejected (expired or duplicate)."""
'''

TIMEUTIL = '''"""Time and backoff helpers."""

from __future__ import annotations

import re
import time

_ISO_RE = re.compile(
    r"^\\d{4}-\\d{2}-\\d{2}[T ]\\d{2}:\\d{2}:\\d{2}(?:\\.\\d+)?(?:Z|[+-]\\d{2}:\\d{2})?$"
)


def parse_iso(text: str) -> float:
    """Parse a conservative ISO-8601 subset to a unix timestamp.

    Accepts an optional fractional second and zone suffix; `Z` and `+00:00` are
    treated as UTC. Raises ValueError on anything else.
    """
    if not _ISO_RE.match(text):
        raise ValueError(f"not an ISO-8601 timestamp: {text!r}")
    body = text.replace("Z", "+00:00").replace(" ", "T")
    from datetime import datetime

    return datetime.fromisoformat(body).timestamp()


def now_monotonic() -> float:
    return time.monotonic()


def backoff_delay(attempt: int, base: float, cap: float = 600.0) -> float:
    """Exponential backoff: base * 2**attempt, capped. attempt starts at 0."""
    if attempt < 0:
        raise ValueError("attempt must be >= 0")
    return min(base * (2.0**attempt), cap)


def format_age(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"
'''

# config.py — plants bug-backoff (env key mismatch) and bug-workers (validate gap)
CONFIG = '''"""Configuration: defaults, environment overrides, validation."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ConfigError

DEFAULTS: dict[str, int] = {
    "max_workers": 4,
    "max_priority": 100,
    "dead_letter_ttl_hours": 72,
    "retry_backoff_seconds": 30,
    "max_retries": 3,
    "rate_per_sec": 50,
    "burst": 10,
    "poll_interval_ms": 250,
}

# Operator-facing environment variable names -> config fields.
ENV_OVERRIDES: dict[str, str] = {
    "NIMBUS_MAX_WORKERS": "max_workers",
    "NIMBUS_MAX_PRIORITY": "max_priority",
    "NIMBUS_DEAD_LETTER_TTL_HOURS": "dead_letter_ttl_hours",
    "NIMBUS_RETRY_BACKOFF_SECONDS": "retry_backoff_seconds",
    "NIMBUS_MAX_RETRIES": "max_retries",
    "NIMBUS_RATE_PER_SEC": "rate_per_sec",
    "NIMBUS_BURST": "burst",
    "NIMBUS_POLL_INTERVAL_MS": "poll_interval_ms",
}

_RANGES: dict[str, tuple[int, int]] = {
    "max_workers": (0, 256),
    "max_priority": (1, 1000),
    "dead_letter_ttl_hours": (1, 24 * 30),
    "retry_backoff_seconds": (1, 3600),
    "max_retries": (0, 32),
    "rate_per_sec": (1, 100000),
    "burst": (1, 100000),
    "poll_interval_ms": (10, 60000),
}


@dataclass
class Config:
    max_workers: int = 4
    max_priority: int = 100
    dead_letter_ttl_hours: int = 72
    retry_backoff_seconds: int = 30
    max_retries: int = 3
    rate_per_sec: int = 50
    burst: int = 10
    poll_interval_ms: int = 250


def from_dict(data: dict) -> Config:
    """Build a Config from a JSON object, ignoring unknown keys."""
    known = {f: data[f] for f in DEFAULTS if f in data}
    for field, value in known.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigError(f"{field} must be an integer, got {value!r}")
    return Config(**known)


def apply_env_overrides(cfg: Config, environ: dict[str, str]) -> Config:
    """Apply environment overrides. Unparseable values raise ConfigError."""
    for env_name, field in ENV_OVERRIDES.items():
        if env_name not in environ:
            continue
        raw = environ[env_name].strip()
        try:
            value = int(raw)
        except ValueError as exc:
            raise ConfigError(f"{env_name}={raw!r} is not an integer") from exc
        setattr(cfg, field, value)
    return cfg


def validate(cfg: Config) -> None:
    """Reject out-of-range values before the stack starts."""
    for field, (lo, hi) in _RANGES.items():
        value = getattr(cfg, field)
        if value < lo or value > hi:
            raise ConfigError(f"{field}={value} outside [{lo}, {hi}]")
'''

# queue.py — plants bug-retry (off-by-one)
QUEUE = '''"""Priority queue with retry and dead-letter policy."""

from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field


@dataclass(order=True)
class Job:
    priority: int
    id: str = field(compare=False)
    payload: dict = field(compare=False, default_factory=dict)
    enqueued_at: float = field(compare=False, default=0.0)


class DeadLetterStore:
    """Holds jobs that exhausted their retries, kept for `ttl_hours`."""

    def __init__(self, ttl_hours: int = 72) -> None:
        self._ttl_hours = ttl_hours
        self._jobs: dict[str, tuple[Job, float]] = {}
        self._clock = time.time

    def push(self, job: Job) -> None:
        self._jobs[job.id] = (job, self._clock() + self._ttl_hours * 3600.0)

    def get(self, job_id: str) -> Job | None:
        entry = self._jobs.get(job_id)
        if entry is None:
            return None
        job, expires_at = entry
        if expires_at < self._clock():
            del self._jobs[job_id]
            return None
        return job

    def __len__(self) -> int:
        return len(self._jobs)


class AttemptStore:
    """Counts how many times a job has already been attempted."""

    def __init__(self) -> None:
        self._attempts: dict[str, int] = {}

    def get_attempts(self, job_id: str) -> int:
        return self._attempts.get(job_id, 0)

    def incr_attempts(self, job_id: str) -> int:
        self._attempts[job_id] = self._attempts.get(job_id, 0) + 1
        return self._attempts[job_id]


class Queue:
    """Binary-heap queue ordered by (priority, insertion order).

    Lower priority value means higher urgency. Ties break FIFO.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[int, int, Job]] = []
        self._order = itertools.count()

    def push(self, job: Job) -> None:
        heapq.heappush(self._heap, (job.priority, next(self._order), job))

    def pop(self) -> Job | None:
        if not self._heap:
            return None
        return heapq.heappop(self._heap)[2]

    def __len__(self) -> int:
        return len(self._heap)

    def peek(self) -> Job | None:
        return self._heap[0][2] if self._heap else None

    def enqueue_with_retry(self, job: Job, attempts: AttemptStore,
                           dead_letters: DeadLetterStore,
                           max_retries: int = 3) -> str:
        """Re-enqueue a failed job, or move it to the dead letters.

        A job may be retried at most `max_retries` times after its first
        attempt. Returns "queued" when re-enqueued and "dead" when the
        job exhausted its retries.
        """
        seen = attempts.get_attempts(job.id)
        if seen > max_retries:
            dead_letters.push(job)
            return "dead"
        attempts.incr_attempts(job.id)
        self.push(job)
        return "queued"
'''

STORE = '''"""In-memory job store with TTL metadata."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class StoredJob:
    job_id: str
    state: str          # "queued" | "running" | "done" | "dead"
    payload: dict
    updated_at: float
    expires_at: float | None = None


class JobStore:
    """Holds job state for the lifetime of the process.

    Entries with `expires_at` in the past are dropped lazily on access.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, StoredJob] = {}
        self._clock = time.time

    def put(self, job_id: str, state: str, payload: dict,
            ttl_seconds: float | None = None) -> StoredJob:
        now = self._clock()
        entry = StoredJob(
            job_id=job_id,
            state=state,
            payload=payload,
            updated_at=now,
            expires_at=(now + ttl_seconds) if ttl_seconds is not None else None,
        )
        self._jobs[job_id] = entry
        return entry

    def get(self, job_id: str) -> StoredJob | None:
        entry = self._jobs.get(job_id)
        if entry is None:
            return None
        if entry.expires_at is not None and entry.expires_at < self._clock():
            del self._jobs[job_id]
            return None
        return entry

    def states(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self._jobs.values():
            counts[entry.state] = counts.get(entry.state, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self._jobs)
'''

RATELIMIT = '''"""Token-bucket and sliding-window rate limiters."""

from __future__ import annotations


class TokenBucket:
    """Classic token bucket. `burst` tokens maximum, refilled at `rate` per second."""

    def __init__(self, rate: float, burst: int) -> None:
        if rate <= 0 or burst < 1:
            raise ValueError("rate must be positive and burst >= 1")
        self._rate = float(rate)
        self._capacity = float(burst)
        self._tokens = float(burst)
        self._last: float | None = None

    def allow(self, now: float, cost: int = 1) -> bool:
        if self._last is not None:
            self._tokens = min(self._capacity,
                               self._tokens + (now - self._last) * self._rate)
        self._last = now
        if self._tokens >= cost:
            self._tokens -= cost
            return True
        return False


class SlidingWindow:
    """Counts events in the trailing `window_seconds` window."""

    def __init__(self, window_seconds: float) -> None:
        self._window = window_seconds
        self._events: list[float] = []

    def add(self, now: float) -> int:
        self._events.append(now)
        return self._count(now)

    def _count(self, now: float) -> int:
        while self._events and self._events[0] <= now - self._window:
            self._events.pop(0)
        return len(self._events)

    def exceeded(self, now: float, limit: int) -> bool:
        return self._count(now) > limit
'''

BACKENDS = '''"""Worker backend registry.

A backend is a class exposing:
  name: class attribute, the registry key
  submit(fn, *args): run fn (may be deferred)
  shutdown(): drain and release resources

Only registered backends may be selected at startup; the worker pool refuses
anything else. New backends (e.g. a broker-based one) are added here.
"""

from __future__ import annotations

from typing import Any, Callable

from .errors import BackendError

_REGISTRY: dict[str, type] = {}


def register(cls: type) -> type:
    if not getattr(cls, "name", None):
        raise BackendError("backend classes must define a `name` attribute")
    _REGISTRY[cls.name] = cls
    return cls


def available() -> list[str]:
    return sorted(_REGISTRY)


def accepts_workers(name: str) -> bool:
    """Whether the backend accepts the pool's `workers` option."""
    if name not in _REGISTRY:
        raise BackendError(f"unknown backend {name!r}; available: {available()}")
    return bool(getattr(_REGISTRY[name], "accepts_pool_size", False))


def make(name: str, **options: Any) -> Any:
    if name not in _REGISTRY:
        raise BackendError(f"unknown backend {name!r}; available: {available()}")
    return _REGISTRY[name](**options)


@register
class InlineBackend:
    """Runs jobs synchronously on the calling thread. Name: `inline`."""

    name = "inline"

    def __init__(self, **options: Any) -> None:
        if options:
            raise BackendError(f"inline backend takes no options: {sorted(options)}")

    def submit(self, fn: Callable[..., Any], *args: Any) -> Any:
        return fn(*args)

    def shutdown(self) -> None:
        return None


@register
class ThreadPoolBackend:
    """Runs jobs on a bounded thread pool. Name: `threadpool`."""

    name = "threadpool"
    accepts_pool_size = True

    def __init__(self, workers: int = 4, **options: Any) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self._pool = ThreadPoolExecutor(max_workers=workers)

    def submit(self, fn: Callable[..., Any], *args: Any) -> Any:
        return self._pool.submit(fn, *args).result()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)
'''

WORKER = '''"""Worker pool: pulls jobs from the queue and dispatches to the configured backend."""

from __future__ import annotations

import time
from typing import Any, Callable

from . import audit
from . import metrics
from .backends import accepts_workers, make as make_backend
from .config import Config
from .errors import BackendError
from .queue import AttemptStore, DeadLetterStore, Job, Queue


class WorkerPool:
    """A fixed-size pool of workers consuming one queue.

    The pool size comes from `config.max_workers`. A size of zero produces a
    pool that starts but never consumes; the CLI logs a warning in that case.

    Jobs that raise BackendError are fed back through the queue's retry
    policy; jobs that exhaust their retries land in the dead-letter store,
    which keeps them for `config.dead_letter_ttl_hours`.
    """

    def __init__(self, queue: Queue, config: Config, backend_name: str = "inline",
                 logger: Callable[[str, str], None] | None = None,
                 audit_log: audit.AuditLog | None = None,
                 dead_letters: DeadLetterStore | None = None) -> None:
        self._queue = queue
        self._config = config
        self._logger = logger or (lambda level, message: None)
        self._audit = audit_log or audit.AuditLog()
        opts = {"workers": config.max_workers} if accepts_workers(backend_name) else {}
        self._backend = make_backend(backend_name, **opts)
        self._jobs_counter = metrics.counter("nimbus_jobs_total")
        self._active = 0
        self._attempts = AttemptStore()
        self._dead_letters = dead_letters or DeadLetterStore(
            ttl_hours=config.dead_letter_ttl_hours)

    def start(self) -> None:
        if self._config.max_workers == 0:
            self._logger("WARN",
                         "config: worker pool size 0; no workers will start "
                         f"(max_workers={self._config.max_workers})")
            return
        self._logger("INFO",
                     f"worker: pool started size={self._config.max_workers}")

    def run_available(self, deadline: float | None = None) -> int:
        """Consume jobs until the queue is empty (or the deadline). Returns count."""
        done = 0
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                break
            job = self._queue.pop()
            if job is None:
                break
            self._active += 1
            self._jobs_counter.inc()
            self._audit.record(job.id, "claimed", {})
            try:
                self._backend.submit(_execute, job)
                self._audit.record(job.id, "done", {})
            except BackendError:
                self._audit.record(job.id, "error", {"error": "backend"})
                outcome = self._queue.enqueue_with_retry(
                    job, self._attempts, self._dead_letters,
                    self._config.max_retries)
                self._audit.record(job.id, outcome, {"attempts":
                                                     self._attempts.get_attempts(job.id)})
            finally:
                self._active -= 1
            done += 1
        return done

    def shutdown(self) -> None:
        self._backend.shutdown()
        self._logger("INFO", "worker: pool shut down")

    @property
    def active(self) -> int:
        return self._active


def _execute(job: Job) -> Any:
    """Default executor: call the payload's `handler` if present, else no-op."""
    handler = job.payload.get("handler")
    if callable(handler):
        return handler(job)
    return None
'''

NOTIFY = '''"""Template-based notification dispatch."""

from __future__ import annotations

from typing import Callable


class Notifier:
    """Fills `{field}` placeholders and dispatches through a transport callable.

    The transport receives (channel, subject, body). Failures are swallowed and
    counted so a flaky transport never breaks job processing.
    """

    def __init__(self, transport: Callable[[str, str, str], None]) -> None:
        self._transport = transport
        self._sent = 0
        self._failed = 0

    def send(self, channel: str, subject: str, template: str,
             fields: dict[str, object]) -> bool:
        try:
            body = template.format(**fields)
        except (KeyError, IndexError) as exc:
            self._failed += 1
            return False
        try:
            self._transport(channel, subject, body)
        except Exception:
            self._failed += 1
            return False
        self._sent += 1
        return True

    @property
    def sent(self) -> int:
        return self._sent

    @property
    def failed(self) -> int:
        return self._failed
'''

METRICS = '''"""Counters, gauges, and histograms with 1-minute rollups."""

from __future__ import annotations

import time


class Counter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.value = 0

    def inc(self, n: int = 1) -> None:
        if n < 0:
            raise ValueError("counter delta must be >= 0")
        self.value += n


class Gauge:
    def __init__(self, name: str) -> None:
        self.name = name
        self.value = 0.0

    def set(self, value: float) -> None:
        self.value = value


class Histogram:
    """Buckets: <1ms, <10ms, <100ms, <1s, <10s, >=10s."""

    EDGES = (0.001, 0.010, 0.100, 1.0, 10.0)

    def __init__(self, name: str) -> None:
        self.name = name
        self.count = 0
        self.total = 0.0
        self.buckets = [0] * (len(self.EDGES) + 1)

    def observe(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("observation must be >= 0")
        self.count += 1
        self.total += seconds
        for i, edge in enumerate(self.EDGES):
            if seconds < edge:
                self.buckets[i] += 1
                return
        self.buckets[-1] += 1

    def p95(self) -> float:
        if self.count == 0:
            return 0.0
        target = self.count * 0.95
        seen = 0
        for i, edge in enumerate(self.EDGES):
            seen += self.buckets[i]
            if seen >= target:
                return edge
        return self.EDGES[-1]


class Registry:
    def __init__(self) -> None:
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._histograms: dict[str, Histogram] = {}

    def counter(self, name: str) -> Counter:
        if name not in self._counters:
            self._counters[name] = Counter(name)
        return self._counters[name]

    def gauge(self, name: str) -> Gauge:
        if name not in self._gauges:
            self._gauges[name] = Gauge(name)
        return self._gauges[name]

    def histogram(self, name: str) -> Histogram:
        if name not in self._histograms:
            self._histograms[name] = Histogram(name)
        return self._histograms[name]

    def snapshot(self) -> dict:
        out: dict = {"counters": {}, "gauges": {}, "histograms": {}}
        for name, c in self._counters.items():
            out["counters"][name] = c.value
        for name, g in self._gauges.items():
            out["gauges"][name] = g.value
        for name, h in self._histograms.items():
            out["histograms"][name] = {"count": h.count, "p95": h.p95()}
        out["captured_at"] = time.time()
        return out


_registry = Registry()


def counter(name: str) -> Counter:
    return _registry.counter(name)


def gauge(name: str) -> Gauge:
    return _registry.gauge(name)


def histogram(name: str) -> Histogram:
    return _registry.histogram(name)


def snapshot() -> dict:
    return _registry.snapshot()
'''

AUDIT = '''"""JSONL audit trail for job state transitions."""

from __future__ import annotations

import json
from typing import IO


class AuditLog:
    """Appends one JSON object per state transition.

    Fields: ts (unix float), job_id, state, extra.
    """

    def __init__(self, stream: IO[str] | None = None) -> None:
        self._stream = stream
        self._entries: list[dict] = []

    def record(self, job_id: str, state: str, extra: dict) -> None:
        import time

        entry = {"ts": time.time(), "job_id": job_id, "state": state, "extra": extra}
        self._entries.append(entry)
        if self._stream is not None:
            self._stream.write(json.dumps(entry) + "\\n")
            self._stream.flush()

    def entries_for(self, job_id: str) -> list[dict]:
        return [e for e in self._entries if e["job_id"] == job_id]

    def __len__(self) -> int:
        return len(self._entries)
'''

API = '''"""Request router: (method, path pattern) -> handler.

Patterns support a single `{param}` segment. Matching is exact on the
remaining segments; the first registered route wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Request:
    method: str
    path: str
    body: Any = None


@dataclass
class Response:
    status: int
    body: Any = None

    @staticmethod
    def ok(body: Any = None) -> "Response":
        return Response(200, body)

    @staticmethod
    def created(body: Any = None) -> "Response":
        return Response(201, body)

    @staticmethod
    def not_found() -> "Response":
        return Response(404)

    @staticmethod
    def bad_request(message: str) -> "Response":
        return Response(400, {"error": message})


class Router:
    def __init__(self) -> None:
        self._routes: list[tuple[str, list[str], Callable[..., Response]]] = []

    def add(self, method: str, pattern: str, handler: Callable[..., Response]) -> None:
        segments = [s for s in pattern.split("/") if s]
        self._routes.append((method.upper(), segments, handler))

    def dispatch(self, request: Request) -> Response:
        for method, segments, handler in self._routes:
            if request.method.upper() != method:
                continue
            parts = [p for p in request.path.split("/") if p]
            if len(parts) != len(segments):
                continue
            params: dict[str, str] = {}
            matched = True
            for seg, part in zip(segments, parts):
                if seg.startswith("{") and seg.endswith("}"):
                    params[seg[1:-1]] = part
                elif seg != part:
                    matched = False
                    break
            if matched:
                return handler(request, **params)
        return Response.not_found()
'''

CLI = '''"""Command-line entry point: config -> overrides -> validate -> stack build."""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import audit, metrics
from .config import Config, apply_env_overrides, from_dict, validate
from .errors import ConfigError
from .queue import AttemptStore, DeadLetterStore, Queue  # noqa: F401
from .worker import WorkerPool


def load_config(path: str | None) -> Config:
    if path is None:
        return Config()
    with open(path, "r", encoding="utf-8") as handle:
        return from_dict(json.load(handle))


def build_stack(config: Config, backend: str = "inline",
                logger=None) -> tuple[Queue, WorkerPool, audit.AuditLog]:
    """Build the queue/worker stack for a validated config."""
    queue = Queue()
    log = audit.AuditLog()
    dead_letters = DeadLetterStore(ttl_hours=config.dead_letter_ttl_hours)
    pool = WorkerPool(queue, config, backend_name=backend, logger=logger,
                      audit_log=log, dead_letters=dead_letters)
    pool.start()
    return queue, pool, log


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nimbus")
    parser.add_argument("--config", default=None, help="path to config.json")
    parser.add_argument("--backend", default="inline",
                        help="worker backend name (see nimbus.backends.available)")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        apply_env_overrides(config, dict(os.environ))
        validate(config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    queue, pool, _log = build_stack(config, backend=args.backend)
    print(f"nimbus ready: workers={config.max_workers} backend={args.backend}")
    pool.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

# fix the accidental bad import in cli.py (keep it realistic: no conditional import)
CLI = CLI.replace("from .queue import DeadLetterStore, JobStore if False else None, AttemptStore, Queue  # noqa: F401",
                  "from .queue import AttemptStore, DeadLetterStore, Queue")

TEST_CONFIG = '''"""Tests for config overrides and validation."""

import pytest

from nimbus.config import Config, apply_env_overrides, validate
from nimbus.errors import ConfigError


def test_defaults():
    cfg = Config()
    assert cfg.max_workers == 4
    assert cfg.retry_backoff_seconds == 30


def test_env_override_workers():
    cfg = Config()
    apply_env_overrides(cfg, {"NIMBUS_MAX_WORKERS": "9"})
    assert cfg.max_workers == 9


def test_env_override_bad_value():
    with pytest.raises(ConfigError):
        apply_env_overrides(Config(), {"NIMBUS_MAX_RETRIES": "many"})


def test_validate_range():
    cfg = Config()
    cfg.max_priority = 0
    with pytest.raises(ConfigError):
        validate(cfg)
'''

TEST_QUEUE = '''"""Tests for the retry / dead-letter policy."""

from nimbus.queue import AttemptStore, DeadLetterStore, Job, Queue


def make_job(jid="job-1", priority=10):
    return Job(priority=priority, id=jid, payload={})


def test_fifo_on_priority_tie():
    q = Queue()
    a, b = make_job("a"), make_job("b")
    q.push(a)
    q.push(b)
    assert q.pop().id == "a"


def test_priority_ordering():
    q = Queue()
    q.push(make_job("low", priority=50))
    q.push(make_job("high", priority=1))
    assert q.pop().id == "high"


def test_dead_letter_ttl_expiry():
    store = DeadLetterStore(ttl_hours=72)
    job = make_job("d1")
    store.push(job)
    assert store.get("d1") is job
    assert len(store) == 1


def test_retry_exhaustion():
    # Regression test for bug #4471: a job may be retried at most
    # `max_retries` times after its first attempt, then must move to the
    # dead letters. This test currently FAILS on the checked-in code.
    q = Queue()
    attempts = AttemptStore()
    dead = DeadLetterStore()
    job = make_job("j")
    outcomes = [q.enqueue_with_retry(job, attempts, dead, max_retries=2)
                for _ in range(4)]
    assert outcomes == ["queued", "queued", "dead", "dead"]
    assert len(dead) == 1
'''

TEST_WORKER = '''"""Tests for the worker pool."""

from nimbus.config import Config
from nimbus.queue import Job, Queue
from nimbus.worker import WorkerPool


def test_pool_consumes_until_empty():
    q = Queue()
    seen = []

    def handler(job):
        seen.append(job.id)
        return None

    for i in range(5):
        q.push(Job(priority=1, id=f"job-{i}", payload={"handler": handler}))
    cfg = Config()
    pool = WorkerPool(q, cfg, backend_name="inline")
    pool.start()
    done = pool.run_available()
    assert done == 5
    assert q.pop() is None
    pool.shutdown()


def test_zero_worker_pool_warns():
    lines = []
    q = Queue()
    cfg = Config()
    cfg.max_workers = 0
    pool = WorkerPool(q, cfg, backend_name="inline",
                      logger=lambda level, msg: lines.append((level, msg)))
    pool.start()
    assert any("no workers will start" in msg for _, msg in lines)
'''

TEST_API = '''"""Tests for the router."""

from nimbus.api import Request, Router


def test_param_route():
    r = Router()
    r.add("GET", "/jobs/{job_id}", lambda req, job_id: type("R", (), {"status": 200, "body": job_id})())
    resp = r.dispatch(Request("GET", "/jobs/abc"))
    assert resp.body == "abc"


def test_unknown_route():
    r = Router()
    resp = r.dispatch(Request("GET", "/nope"))
    assert resp.status == 404
'''

BUGREPORT = """# Bug report #4471 — jobs sit in `queued` for hours

Reported by: ops (2026-08-24)

## Symptom

Job `job-7f3a9` was enqueued at 09:14:32 and was still in state `queued` at
09:56 (see `logs/app-2026-08-24.log` and `logs/metrics-snapshot.json`). No worker
ever claimed it. The queue depth stayed flat the whole morning; every other job
that was enqueued before 09:14 completed normally.

## Context

The morning deploy switched the scheduler host to the new config:

```
NIMBUS_MAX_WORKERS=0
NIMBUS_RETRY_BACKOFF_SECS=5
```

The operator intended 0 to mean "auto" (the old host used 4). The backoff of 5s
is the on-call value for this service.

## Expected

Either the stack refuses to start with a worker pool of zero, or the operator's
backoff of 5 seconds is honored. As-is, the process starts, logs one warning,
and sits idle while the queue fills.
"""

GROUND_TRUTH = {
    "t8": {
        "task": "retry_offbyone",
        "kind": "code",
        "prompt": (
            "In `nimbus/queue.py`, the retry policy misbehaves: a job that has already "
            "failed exactly `max_retries` times is re-enqueued one extra time instead of "
            "moving to the dead-letter store immediately. Identify the defect and return "
            "ONLY the corrected `enqueue_with_retry` method body-compatible code: a plain "
            "Python function named `enqueue_with_retry(self, job, attempts, dead_letters, "
            "max_retries=3)` with the same semantics as the original (returns \"queued\" or "
            "\"dead\"). Put it in a ```python code block. No explanation outside the block."
        ),
        "expected_requeues_for_3_failures": 3,  # total attempts before dead: 1 + max_retries
    },
    "t16": {
        "task": "config_trace",
        "kind": "json",
        "prompt": (
            "Trace the `dead_letter_ttl_hours` setting through this repository: where its "
            "default is defined, where it can be overridden by the environment, and in which "
            "files it is actually consumed (i.e. read to affect behaviour). Answer with ONLY "
            "a JSON object: {\"default_in\": <file>, \"env_var\": <env name>, "
            "\"consumed_in\": <sorted list of files>}. Use repository-relative POSIX paths."
        ),
        "answer": {
            "default_in": "nimbus/config.py",
            "env_var": "NIMBUS_DEAD_LETTER_TTL_HOURS",
            "consumed_in": ["nimbus/cli.py", "nimbus/worker.py"],
        },
    },
    "t32": {
        "task": "celery_backend",
        "kind": "json",
        "prompt": (
            "We want to add a new worker backend named `celery` alongside the built-ins, "
            "selectable at startup exactly like the existing ones. List EXACTLY the "
            "repository files that must be modified or created to make `--backend celery` "
            "work end-to-end (nothing else). Answer with ONLY a JSON array of "
            "repository-relative POSIX paths, sorted."
        ),
        "answer": [
            "nimbus/backends.py",
        ],
    },
    "t48": {
        "task": "backoff_mismatch",
        "kind": "code",
        "prompt": (
            "An operator set the documented environment variable `NIMBUS_RETRY_BACKOFF_SECS=5` "
            "(see README), but the logs show retries backing off by 30 seconds. Find the "
            "root-cause defect in `nimbus/config.py` and return the corrected code that makes "
            "the documented variable work, in a ```python code block. You may return either the "
            "corrected `ENV_OVERRIDES` mapping or the corrected `apply_env_overrides` function "
            "(whichever contains the fix). The other environment overrides must keep working."
        ),
        "expected_env_key": "NIMBUS_RETRY_BACKOFF_SECS",
        "expected_value": 5,
    },
    "t64": {
        "task": "stalled_job",
        "kind": "json",
        "prompt": (
            "Using the code AND the logs/metrics in this repository, diagnose why job "
            "`job-7f3a9` sat in `queued` for over 40 minutes with no worker pickup, even "
            "though jobs enqueued earlier in the morning completed. Answer with ONLY a JSON "
            "object: {\"file\": <file>, \"function\": <function that should have caught "
            "this>, \"cause\": <one sentence naming the offending setting and value>}."
        ),
        "answer": {
            "file": "nimbus/config.py",
            "function": "validate",
            "cause_keywords": ["max_workers", "0"],
        },
    },
}

# ---------------------------------------------------------------------------
# Log generation (seeded, realistic; plants evidence lines)
# ---------------------------------------------------------------------------

_JOB_NAMES = ["resize-image", "send-digest", "compact-wal", "reindex-tags", "sync-catalog",
              "rotate-logs", "expire-sessions", "backfill-metrics", "purge-tmp", "publish-feed"]


def _job_id(rng: random.Random, n: int) -> str:
    return "job-" + "".join(rng.choice("0123456789abcdef") for _ in range(5))[:n]


def gen_log_lines(profile: str, rng: random.Random) -> list[str]:
    """Generate startup + job-lifecycle log lines with planted evidence."""
    lines: list[str] = []
    base_h, base_m = 9, 1
    ts = [base_h, base_m, 0]

    def stamp(drift_ms: int = 0) -> str:
        total = ts[0] * 3600 + ts[1] * 60 + ts[2] + drift_ms / 1000.0
        h, rem = divmod(int(total), 3600)
        m, s = divmod(rem, 60)
        return f"2026-08-24T{h:02d}:{m:02d}:{int(s):02d}.{drift_ms % 1000:03d}Z"

    def tick(ms: int) -> None:
        ts[2] += ms // 1000
        rem = ts[2]
        if rem >= 60:
            ts[2] = rem % 60
            ts[1] += rem // 60
        if ts[1] >= 60:
            ts[1] %= 60
            ts[0] += ts[1] // 60

    lines.append(f"{stamp()} INFO  cli: nimbus starting version=0.9.3 backend=threadpool")
    lines.append(f"{stamp(12)} INFO  cli: config loaded (max_workers=0 max_retries=3 "
                 "retry_backoff_seconds=30 rate_per_sec=50)")
    lines.append(f"{stamp(15)} WARN  config: worker pool size 0; no workers will start "
                 "(max_workers=0)")
    lines.append(f"{stamp(18)} INFO  worker: pool started size=0")
    lines.append(f"{stamp(20)} INFO  api: router ready routes=7")
    lines.append(f"{stamp(23)} INFO  queue: queue ready dead_letter_ttl=72h")

    n_jobs = {
        "8k": 0, "16k": 97, "32k": 324, "48k": 558, "64k": 792,
    }[profile]

    # morning activity: jobs enqueue and (mostly) nothing runs because the pool is empty;
    # the earlier deploy (yesterday) had workers, so "completed" entries reference prior jobs.
    for i in range(n_jobs):
        jid = _job_id(rng, 5)
        name = rng.choice(_JOB_NAMES)
        drift = rng.randint(200, 900)
        tick(rng.randint(8000, 40000))
        lines.append(f"{stamp(drift)} INFO  queue: enqueued id={jid} name={name} priority={rng.randint(1, 50)}")
        if i % 11 == 0:
            tick(rng.randint(2000, 9000))
            lines.append(f"{stamp(drift)} INFO  metrics: gauge nimbus_queue_depth set value={40 + i}")
        if i % 17 == 0:
            tick(rng.randint(1500, 7000))
            lines.append(f"{stamp(drift)} INFO  audit: job={_job_id(rng, 5)} state=done elapsed_ms={rng.randint(40, 900)}")
        if i % 23 == 0:
            tick(rng.randint(1500, 7000))
            lines.append(f"{stamp(drift)} INFO  audit: job={_job_id(rng, 5)} state=done elapsed_ms={rng.randint(30, 600)}")
        if i % 31 == 0:
            tick(rng.randint(1500, 7000))
            lines.append(f"{stamp(drift)} WARN  rate: window exceeded limit=50 window_s=1 count=52")

    # planted: job-7f3a9 lifecycle
    tick(60000)
    lines.append(f"{stamp(32)} INFO  queue: enqueued id=job-7f3a9 name=compact-wal priority=5")
    lines.append(f"{stamp(900)} INFO  audit: job=job-7f3a9 state=queued backoff_s=30 attempt=0")
    tick(120000)
    lines.append(f"{stamp(40)} INFO  metrics: gauge nimbus_queue_depth set value=44")
    tick(900000)
    lines.append(f"{stamp(51)} WARN  queue: job=job-7f3a9 age_ms=600000 state=queued no worker available")
    tick(1500000)
    lines.append(f"{stamp(77)} WARN  queue: job=job-7f3a9 age_ms=1500000 state=queued no worker available")
    tick(900000)
    lines.append(f"{stamp(120)} INFO  metrics: histogram nimbus_job_wait_seconds observed=2512.4")
    if profile in ("48k", "64k"):
        lines.append(f"{stamp(160)} WARN  queue: job=job-7f3a9 age_ms=2520000 state=queued "
                     "escalation=ops-page-suggested")
        lines.append(f"{stamp(200)} INFO  ops: config diff applied at 09:00 "
                     "(NIMBUS_MAX_WORKERS=0 NIMBUS_RETRY_BACKOFF_SECS=5)")

    for i in range({ "8k": 0, "16k": 25, "32k": 84, "48k": 145, "64k": 206 }[profile]):
        tick(rng.randint(10000, 60000))
        lines.append(f"{stamp(rng.randint(300, 900))} INFO  metrics: gauge nimbus_queue_depth "
                     f"set value={44 + (i % 7)}")
    return lines


METRICS_SNAPSHOT = {
    "counters": {
        "nimbus_jobs_total": 118,
        "nimbus_retries_total": 7,
        "nimbus_dead_letters_total": 1,
    },
    "gauges": {
        "nimbus_queue_depth": 47,
        "nimbus_workers_active": 0,
        "nimbus_workers_configured": 0,
    },
    "histograms": {
        "nimbus_job_wait_seconds": {"count": 118, "p95": 10.0},
        "nimbus_job_run_seconds": {"count": 117, "p95": 1.0},
    },
    "captured_at": 1756070160.0,
    "note": "captured 09:56 by on-call; job-7f3a9 still queued",
}

SAMPLE_JOBS = [
    {"id": "job-7f3a9", "name": "compact-wal", "priority": 5, "enqueued_at": "2026-08-24T09:14:32Z", "state": "queued", "attempts": 0},
    {"id": "job-2b8c1", "name": "resize-image", "priority": 20, "enqueued_at": "2026-08-24T08:02:11Z", "state": "done", "attempts": 1},
    {"id": "job-9d4e7", "name": "send-digest", "priority": 30, "enqueued_at": "2026-08-24T08:11:47Z", "state": "done", "attempts": 1},
    {"id": "job-51aa0", "name": "reindex-tags", "priority": 10, "enqueued_at": "2026-08-24T08:30:05Z", "state": "dead", "attempts": 4},
    {"id": "job-c0f2b", "name": "expire-sessions", "priority": 40, "enqueued_at": "2026-08-24T08:41:29Z", "state": "done", "attempts": 1},
]


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

# (file path, content) in repository order
CORE_FILES = [
    ("pyproject.toml", PYPROJECT),
    ("README.md", README),
    ("Makefile", MAKEFILE),
    ("nimbus/__init__.py", INIT),
    ("nimbus/errors.py", ERRORS),
    ("nimbus/timeutil.py", TIMEUTIL),
    ("nimbus/config.py", CONFIG),
    ("nimbus/queue.py", QUEUE),
    ("nimbus/store.py", STORE),
    ("nimbus/ratelimit.py", RATELIMIT),
    ("nimbus/backends.py", BACKENDS),
    ("nimbus/worker.py", WORKER),
    ("nimbus/notify.py", NOTIFY),
    ("nimbus/metrics.py", METRICS),
    ("nimbus/audit.py", AUDIT),
    ("nimbus/api.py", API),
    ("nimbus/cli.py", CLI),
    ("tests/test_config.py", TEST_CONFIG),
    ("tests/test_queue.py", TEST_QUEUE),
    ("tests/test_worker.py", TEST_WORKER),
    ("tests/test_api.py", TEST_API),
    ("docs/bugreport-4471.md", BUGREPORT),
    ("logs/metrics-snapshot.json", json.dumps(METRICS_SNAPSHOT, indent=2)),
    ("examples/sample_jobs.jsonl",
     "\n".join(json.dumps(j) for j in SAMPLE_JOBS)),
]

def profile_files(profile: str) -> list[tuple[str, str]]:
    files = {
        "8k": CORE_FILES[:21],
        "16k": CORE_FILES[:22],
        "32k": CORE_FILES,
        "48k": CORE_FILES,
        "64k": CORE_FILES,
    }[profile]
    return files


def build_repo_text(profile: str, seed: int = 4471) -> str:
    rng = random.Random(seed)
    files = profile_files(profile)
    parts: list[str] = []
    for path, content in files:
        parts.append(f"=== {path} ===\n{content}")
    if profile != "8k":
        parts.append(f"=== logs/app-2026-08-24.log ===\n"
                     + "\n".join(gen_log_lines(profile, rng)))
    return "\n\n".join(parts) + "\n"


def render_prompt(profile: str, task: str) -> str:
    repo = build_repo_text(profile)
    task_def = GROUND_TRUTH[task]
    header = (
        "The text below is the complete content of the repository `nimbus-queue`, "
        "one file per `=== <path> ===` block. "
    )
    return header + repo + "\n=== TASK ===\n" + task_def["prompt"] + "\n"


# ---------------------------------------------------------------------------
# Scoring (deterministic, behavior-based against GROUND_TRUTH) + engine driver
# ---------------------------------------------------------------------------

BINARY_DEFAULT = "build/apps/ninfer"
ARTIFACT_DEFAULT = "out/qwen3_8_27b_rtx5060ti_q4.ninfer"
PROFILE_TASK = {"8k": "t8", "16k": "t16", "32k": "t32", "48k": "t48", "64k": "t64"}
PROFILES = ("8k", "16k", "32k", "48k", "64k")
MAX_NEW = {"code": 700, "json": 300}

PY_BLOCK_RE = re.compile(r"```(?:python|py)\s*\n?(.*?)```", re.DOTALL)
SUMMARY_RE = re.compile(r"^summary\s+(\S.*?)\s{2,}(\S.*)$")


def _first_block(text: str) -> str | None:
    m = PY_BLOCK_RE.search(text)
    return m.group(1).strip() if m else None


def _extract_json(text: str):
    candidates: list[str] = [text]
    for m in re.finditer(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL):
        candidates.append(m.group(1))
    for m in re.finditer(r"(\{.*\}|\[.*\])", text, re.DOTALL):
        candidates.append(m.group(1))
    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            continue
    return None


def _queue_ns() -> dict:
    ns: dict = {}
    exec(compile(QUEUE, "<queue.py>", "exec"), ns)
    return ns


def _config_src() -> str:
    return CONFIG.replace("from .errors import ConfigError",
                          "class ConfigError(Exception):\n    pass")


def _config_ns() -> dict:
    ns: dict = {}
    exec(compile(_config_src(), "<config.py>", "exec"), ns)
    return ns


def score_t8(raw: str):
    code = _first_block(raw)
    if code is None:
        return False, "no python block", True
    q = _queue_ns()
    ns = dict(q)
    try:
        exec(compile(code, "<model>", "exec"), ns)
    except Exception as e:
        return False, f"exec: {e}", True
    fn = ns.get("enqueue_with_retry")
    if not callable(fn):
        return False, "enqueue_with_retry not defined", True
    job = q["Job"](priority=1, id="j", payload={})
    att, dead, self_ = q["AttemptStore"](), q["DeadLetterStore"](), q["Queue"]()
    outcomes: list[str] = []
    try:
        for _ in range(8):
            outcomes.append(fn(self_, job, att, dead, max_retries=3))
            if outcomes[-1] == "dead":
                break
    except Exception as e:
        return False, f"raised {e}", True
    ok = outcomes == ["queued", "queued", "queued", "dead"]
    return ok, f"outcomes={outcomes}", True


def score_t16(raw: str):
    obj = _extract_json(raw)
    gt = GROUND_TRUTH["t16"]["answer"]
    if not isinstance(obj, dict):
        return False, "no json object", True
    d_in, env, cons = obj.get("default_in"), obj.get("env_var"), obj.get("consumed_in")
    ok = (d_in == gt["default_in"] and env == gt["env_var"]
          and isinstance(cons, list) and sorted(cons) == sorted(gt["consumed_in"]))
    return ok, f"default_in={d_in} env={env} consumed={cons}", True


def score_t32(raw: str):
    obj = _extract_json(raw)
    gt = GROUND_TRUTH["t32"]["answer"]
    if not isinstance(obj, list):
        return False, "no json array", True
    ok = sorted(obj) == sorted(gt)
    return ok, f"files={obj}", True


def score_t48(raw: str):
    code = _first_block(raw)
    if code is None:
        return False, "no python block", True
    c = _config_ns()
    ns = {"ENV_OVERRIDES": c["ENV_OVERRIDES"], "ConfigError": c["ConfigError"],
          "Config": c["Config"], "DEFAULTS": c["DEFAULTS"]}
    try:
        exec(compile(code, "<model>", "exec"), ns)
    except Exception as e:
        return False, f"exec: {e}", True
    env_overrides = ns["ENV_OVERRIDES"]  # possibly redefined by the model
    if callable(ns.get("apply_env_overrides")):
        # Model supplied the function; it sees ns["ENV_OVERRIDES"].
        fn = ns["apply_env_overrides"]
    else:
        # Model only (re)defined the mapping; run the original function against it.
        orig_ns = {}
        exec(compile(_config_src(), "<cfg>", "exec"), orig_ns)
        orig_ns["ENV_OVERRIDES"] = env_overrides
        fn = orig_ns["apply_env_overrides"]
    cfg = c["Config"]()
    try:
        fn(cfg, {"NIMBUS_RETRY_BACKOFF_SECS": "5"})
    except Exception as e:
        return False, f"raised {e}", True
    ok1 = getattr(cfg, "retry_backoff_seconds", None) == 5
    cfg2 = c["Config"]()
    try:
        fn(cfg2, {"NIMBUS_MAX_WORKERS": "8", "NIMBUS_DEAD_LETTER_TTL_HOURS": "48"})
    except Exception as e:
        return False, f"other overrides raised {e}", True
    ok2 = (getattr(cfg2, "max_workers", None) == 8
           and getattr(cfg2, "dead_letter_ttl_hours", None) == 48)
    ok = ok1 and ok2
    return ok, (f"backoff={getattr(cfg, 'retry_backoff_seconds', None)} (want 5); "
                f"max_workers={getattr(cfg2, 'max_workers', None)} "
                f"ttl={getattr(cfg2, 'dead_letter_ttl_hours', None)} (want 8/48)"), True


def score_t64(raw: str):
    obj = _extract_json(raw)
    gt = GROUND_TRUTH["t64"]["answer"]
    if not isinstance(obj, dict):
        return False, "no json object", True
    f, fn = obj.get("file"), obj.get("function")
    cause = str(obj.get("cause", ""))
    ok_file = f == gt["file"]
    ok_fn = fn == gt["function"]
    ok_cause = "max_workers" in cause.lower() and bool(re.search(r"\b0\b", cause))
    ok = ok_file and ok_fn and ok_cause
    return ok, f"file={f} function={fn} cause={cause[:70]!r}", True


SCORERS = {"t8": score_t8, "t16": score_t16, "t32": score_t32,
           "t48": score_t48, "t64": score_t64}


def score_task(task_key: str, raw: str):
    return SCORERS[task_key](raw)


def _parse_summary(text: str) -> dict:
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = SUMMARY_RE.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def cmd_run(args) -> int:
    msg_dir = Path(args.msg_dir)
    msg_dir.mkdir(parents=True, exist_ok=True)
    results = Path(args.results)
    results.parent.mkdir(parents=True, exist_ok=True)
    profiles = args.profiles.split(",")
    rows = []
    for prof in profiles:
        task = PROFILE_TASK[prof]
        g = GROUND_TRUTH[task]
        prompt = render_prompt(prof, task)
        mpath = msg_dir / f"{prof}.json"
        mpath.write_text(json.dumps([{"role": "user", "content": prompt}]))
        max_new = MAX_NEW[g["kind"]]
        cmd = [args.binary, args.artifact, "--messages", str(mpath),
               "--max-context", str(args.max_context), "--max-new", str(max_new),
               "--kv-capacity", "auto", "--kv-dtype", args.kv_dtype,
               "--prefill-chunk", "64", "--no-prefix-reuse", "--no-thinking", "--greedy"]
        if args.spec:
            cmd += ["--spec", args.spec, "--draft-tokens", str(args.draft_tokens)]
        print(f"[run] {prof} ({task}) max-ctx={args.max_context} kv=auto spec={args.spec}",
              flush=True)
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            raw, ok, err_tail = proc.stdout, proc.returncode == 0, proc.stderr[-4000:]
        except subprocess.TimeoutExpired:
            raw, ok, err_tail = "", False, "timeout"
        summ = _parse_summary(err_tail)
        verdict, detail, _s = score_task(task, raw)
        row = {
            "profile": prof, "task": task, "kind": g["kind"], "artifact": args.artifact,
            "prompt_chars": len(prompt),
            "prompt_tokens": int(summ.get("prompt tokens", "0") or 0),
            "generated_tokens": int(summ.get("generated tokens", "0") or 0),
            "decode_tps": summ.get("decode speed", ""),
            "kv_capacity": summ.get("KV capacity", ""),
            "acceptance_pct": summ.get("mtp acceptance rate", ""),
            "fallback": summ.get("mtp fallback steps", "0"),
            "ok": ok, "elapsed_s": round(time.time() - t0, 2),
            "pass": bool(ok and verdict), "verdict": verdict, "detail": detail,
            "raw_response": raw,
        }
        rows.append(row)
        print(f"  -> {prof:5} {task:5} ok={ok} pass={row['pass']} "
              f"ptok={row['prompt_tokens']} gen={row['generated_tokens']} "
              f"dec={row['decode_tps']} {detail[:55]}", flush=True)
    with open(results, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Results: {results}")
    return 0


def cmd_score(args) -> int:
    by_profile: dict[str, dict] = {}
    for p in args.results:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    by_profile[r["profile"]] = r
    for r in by_profile.values():
        if "pass" not in r and r.get("raw_response"):
            v, d, _ = score_task(r["task"], r["raw_response"])
            r["pass"] = bool(r.get("ok") and v)
            r["detail"] = d
    print(f"{'profile':8} {'task':6} {'kind':5} {'ptok':>6} {'gen':>5} "
          f"{'dec':>7} {'pass':5} detail")
    npass = 0
    for prof in PROFILES:
        r = by_profile.get(prof)
        if not r:
            continue
        npass += 1 if r.get("pass") else 0
        print(f"{prof:8} {r['task']:6} {r['kind']:5} {r['prompt_tokens']:>6} "
              f"{r['generated_tokens']:>5} {str(r['decode_tps']):>7} "
              f"{str(bool(r.get('pass'))):5} {str(r.get('detail', ''))[:58]}")
    print(f"\n{npass}/{len(by_profile)} passed")
    if args.report:
        Path(args.report).write_text(json.dumps(
            {"n_pass": npass, "n_total": len(by_profile),
             "rows": [by_profile[p] for p in PROFILES if p in by_profile]}, indent=2))
        print(f"Report: {args.report}")
    return 0


def main(argv: list[str]) -> int:
    profiles = ("8k", "16k", "32k", "48k", "64k")
    task_for_profile = {"8k": "t8", "16k": "t16", "32k": "t32", "48k": "t48", "64k": "t64"}
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("emit")
    pe.add_argument("--profile", required=True, choices=profiles)
    pe.add_argument("--out", required=True)
    pr = sub.add_parser("prompt")
    pr.add_argument("--profile", required=True, choices=profiles)
    pr.add_argument("--out", required=True)
    prun = sub.add_parser("run")
    prun.add_argument("--binary", default=BINARY_DEFAULT)
    prun.add_argument("--artifact", default=ARTIFACT_DEFAULT)
    prun.add_argument("--profiles", default=",".join(PROFILES))
    prun.add_argument("--max-context", type=int, default=80000)
    prun.add_argument("--kv-dtype", default="int4")
    prun.add_argument("--spec", default="mtp")
    prun.add_argument("--draft-tokens", type=int, default=3)
    prun.add_argument("--timeout", type=int, default=900)
    prun.add_argument("--results", default="results/agent_bench/agent_q4.jsonl")
    prun.add_argument("--msg-dir", default="results/agent_bench/messages")
    prun.set_defaults(func=cmd_run)
    ps = sub.add_parser("score")
    ps.add_argument("--results", nargs="+", required=True)
    ps.add_argument("--report")
    ps.set_defaults(func=cmd_score)
    args = parser.parse_args(argv)
    if hasattr(args, "func"):
        return args.func(args)
    if args.cmd == "emit":
        text = build_repo_text(args.profile)
        Path(args.out).write_text(text)
        print(f"{args.out}: {len(text)} chars, {text.count(chr(10))} lines")
    else:
        text = render_prompt(args.profile, task_for_profile[args.profile])
        Path(args.out).write_text(text)
        print(f"{args.out}: {len(text)} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))