"""Executing and measuring one request, for an arm of the study.

The request boundary is the same for every arm: from the moment the request is issued to the
moment its last token exists.  Session creation, prompt processing, drafting, verification and
orchestration are inside; loading models (start-up) and putting the system into the study's
declared regime (for example, clearing the expert cache) are outside and reported separately.

Every request yields a record, including the ones that fail: an out-of-memory error, a
time-out or a wrong token count is an observation, not a gap.
"""

from __future__ import annotations

import json
import signal
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any

from basp.runtime.telemetry import io_delta, io_snapshot


class RequestTimeout(TimeoutError):
    pass


@contextmanager
def deadline(seconds: float | None):
    """Raise ``RequestTimeout`` in the body after ``seconds`` (where the platform allows it:
    POSIX main thread; elsewhere the body simply runs)."""
    usable = (seconds and hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread())
    if not usable:
        yield
        return

    def expire(_signum, _frame):
        raise RequestTimeout(f"request exceeded {seconds:g} s")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class BaspArms:
    """This runtime's arms.  Models are loaded once (start-up); every request gets a fresh
    session and, under the ``cleared before every request`` policy, an empty expert cache."""

    def __init__(self, target: str, draft: str | None = None, *, prefer=("cpu",), expert_cache_bytes: int | None = None,
                 embedding: str = "resident", memory_bytes: int | None = None) -> None:  # fmt: skip
        from basp.runtime import Engine

        t0 = time.perf_counter()
        self.target = Engine(target, prefer=prefer, expert_cache_bytes=expert_cache_bytes, embedding=embedding,
                             memory_bytes=memory_bytes)  # fmt: skip
        self.draft = Engine(draft, prefer=prefer) if draft else None
        self.startup_s = time.perf_counter() - t0
        self.backend = "+".join(sorted(set(self.target.plan.assignment.values())))

    def _counters(self) -> dict[str, int]:
        t = self.target.telemetry
        d = self.draft.telemetry if self.draft is not None else None
        return {"target_rows": t.forward_rows, "target_calls": t.forward_calls,
                "draft_rows": d.forward_rows if d else 0, "expert_logical_requests": t.expert_requests,
                "expert_logical_stored_bytes": t.expert_logical_bytes, "expert_cache_misses": t.expert_misses,
                "expert_decoded_stored_bytes": t.expert_decoded_stored_bytes}  # fmt: skip

    def request(self, arm: Mapping[str, Any], prompt: list[int], max_new_tokens: int, *,
                expert_cache: str = "cleared", timeout_s: float | None = None) -> dict[str, Any]:  # fmt: skip
        from basp.runtime import generate
        from basp.runtime.speculative import (
            AdaptiveController,
            ModelDrafter,
            NgramDrafter,
            speculative_generate,
        )

        method = arm["method"]
        if method not in ("ordinary", "speculative"):  # a configuration error, not an observation
            raise ValueError(f"unknown method {method!r}")
        if method == "speculative" and arm.get("drafter", "model") == "model" and self.draft is None:
            raise ValueError("this arm drafts with a model: give a draft model")
        if expert_cache == "cleared":  # the declared regime, set up outside the request boundary
            self.target.weights.clear_cache()
        before, io0 = self._counters(), io_snapshot()
        self.target.weights.peak_resident_bytes = self.target.weights.resident_bytes
        obs: dict[str, Any] = {"ok": False, "failure": None, "tokens_returned": 0, "token_ids": None}
        drafter = None
        start = time.perf_counter()
        try:
            with deadline(timeout_s), self.target.session(len(prompt) + max_new_tokens + 16) as session:
                if method == "ordinary":
                    tokens = generate(session, prompt, max_new_tokens).tokens
                else:
                    capacity = len(prompt) + max_new_tokens + 16
                    drafter = (ModelDrafter(self.draft, capacity) if arm.get("drafter", "model") == "model"
                               else NgramDrafter())  # fmt: skip
                    k = arm.get("k", 4)
                    controller = AdaptiveController(k_max=int(arm.get("k_max", 8))) if k == "adaptive" else None
                    result = speculative_generate(session, drafter, prompt, max_new_tokens,
                                                  k=0 if controller else int(k), controller=controller)  # fmt: skip
                    tokens = result.tokens
                    summary = result.summary()
                    obs["speculation"] = {key: summary[key] for key in (
                        "proposed", "verified", "accepted", "acceptance", "target_calls", "draft_seconds",
                        "verify_seconds", "k_histogram")}  # fmt: skip
            obs.update(ok=True, tokens_returned=len(tokens), token_ids=[int(t) for t in tokens])
        except Exception as exc:  # noqa: BLE001 - any failure of a request is an observation to record
            obs["failure"] = f"{type(exc).__name__}: {exc}"[:300]
        finally:
            obs["latency_s"] = time.perf_counter() - start
            if drafter is not None and hasattr(drafter, "close"):
                drafter.close()
        after = self._counters()
        obs["work"] = {key: after[key] - before[key] for key in after}
        obs["io"] = io_delta(io0, io_snapshot())
        memory = self.target.memory_report()
        obs["memory"] = {"peak_expert_resident_bytes": memory["expert_arrays"]["peak_resident_bytes"],
                         "expert_ceiling_bytes": memory["expert_arrays"]["ceiling_bytes"],
                         "process_rss_bytes": memory["process"]["rss_bytes"],
                         "process_peak_rss_bytes": memory["process"]["peak_rss_bytes"]}  # fmt: skip
        if not obs["ok"]:  # a failure must leave the runtime usable
            try:
                self.target.weights.check_invariant()
                self.target.domain.check_invariant()
                obs["recovered"] = not self.target.admission.active
            except AssertionError as exc:
                obs["recovered"] = False
                obs["failure"] += f"; and the runtime did not recover: {exc}"
        return obs


class NativeArm:
    """A native inference server reached over its OpenAI-compatible completions endpoint.

    The server is started by the person running the study, from the pinned binary with the
    pinned command; this class only issues requests and measures them from outside, the same
    boundary as for the other arms.  NOT exercised against a real engine in development (only
    against a stand-in server in the tests): expect to adjust the request fields on first use.
    """

    def __init__(self, arm: Mapping[str, Any], *, timeout_s: float | None = None) -> None:
        self.arm = dict(arm)
        self.url = str(arm["endpoint"]).rstrip("/") + "/v1/completions"
        self.timeout_s = timeout_s

    def request(self, text: str, max_new_tokens: int) -> dict[str, Any]:
        body = {"prompt": text, "max_tokens": max_new_tokens, "temperature": 0, "stream": False,
                **self.arm.get("request", {})}  # fmt: skip
        if self.arm.get("model_name"):
            body["model"] = self.arm["model_name"]
        request = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})  # fmt: skip
        obs: dict[str, Any] = {"ok": False, "failure": None, "tokens_returned": 0, "token_ids": None}
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                data = json.loads(response.read())
            obs["latency_s"] = time.perf_counter() - start
            obs["tokens_returned"] = int(data["usage"]["completion_tokens"])
            obs["text"] = data["choices"][0].get("text", "")
            obs["server_timings"] = data.get("timings")  # reported by the server; not the primary outcome
            obs["ok"] = True
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError, IndexError) as exc:
            obs["latency_s"] = time.perf_counter() - start
            obs["failure"] = f"{type(exc).__name__}: {exc}"[:300]
        return obs
