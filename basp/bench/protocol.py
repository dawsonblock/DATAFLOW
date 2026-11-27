"""The study protocol: what is compared, on what, in what order -- fixed before results exist.

A protocol names its arms, the comparisons between them (baseline, candidate, the speed-up a
decision needs), the prompts (a development set for tuning and timing, a held-out set for the
decision), the regime every request runs in, and how trials are ordered.

``fingerprint(comparison)`` hashes everything that defines a comparison.  A comparison is
*frozen* by recording that hash under ``frozen``; a study marked ``requires_frozen`` (the decision
study) refuses to run a comparison that is not frozen, or was changed since.  Tuning happens on the development
prompts, before freezing; nothing about the held-out prompts may inform any setting.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from basp.evidence import sha256_json

ARM_KEYS = ("A", "B", "C", "D", "E")
ENGINES = ("basp", "native")
#: what a native arm must pin before it may run: the comparison is against *that* build and configuration
NATIVE_PINS = ("endpoint", "revision", "binary_sha256", "command")


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class Trial:
    """One request of one arm.  The two trials of a pair share ``pair``; ``position`` is 0 for
    the arm that runs first."""

    pair: int
    prompt_id: str
    category: str
    repetition: int
    arm: str
    position: int


class Protocol:
    def __init__(self, data: Mapping[str, Any], path: str | Path | None = None) -> None:
        self.data = json.loads(json.dumps(data))  # a private, JSON-clean copy
        self.path = None if path is None else Path(path)
        self._validate()

    @classmethod
    def load(cls, path: str | Path) -> Protocol:
        return cls(json.loads(Path(path).read_text()), path)

    # -- content ---------------------------------------------------------------------------
    @property
    def arms(self) -> dict[str, dict[str, Any]]:
        return self.data["arms"]

    @property
    def criteria(self) -> dict[str, Any]:
        return self.data["criteria"]

    def study(self, name: str) -> dict[str, Any]:
        try:
            return self.data["studies"][name]
        except KeyError:
            raise ProtocolError(f"unknown study {name!r}; known: {sorted(self.data['studies'])}") from None

    def comparison(self, name: str) -> dict[str, Any]:
        for c in self.data["comparisons"]:
            if c["name"] == name:
                return c
        raise ProtocolError(f"unknown comparison {name!r}; known: {[c['name'] for c in self.data['comparisons']]}")

    def prompts(self, study: str) -> list[dict[str, Any]]:
        return self.data["prompts"][self.study(study)["prompts"]]

    def _validate(self) -> None:
        d = self.data
        for key in ("name", "version", "arms", "comparisons", "studies", "prompts", "criteria", "order", "regime"):
            if key not in d:
                raise ProtocolError(f"protocol lacks {key!r}")
        for key, arm in d["arms"].items():
            if key not in ARM_KEYS or arm.get("engine") not in ENGINES:
                raise ProtocolError(f"arm {key!r}: key must be one of {ARM_KEYS} and engine one of {ENGINES}")
        for c in d["comparisons"]:
            if c["baseline"] not in d["arms"] or c["candidate"] not in d["arms"] or c["baseline"] == c["candidate"]:
                raise ProtocolError(f"comparison {c.get('name')!r} names unknown or identical arms")
            if not float(c["min_speedup"]) > 1.0:
                raise ProtocolError(f"comparison {c['name']!r}: min_speedup must exceed 1")
        for name, prompts in d["prompts"].items():
            ids = [p["id"] for p in prompts]
            if len(set(ids)) != len(ids):
                raise ProtocolError(f"prompt set {name!r} has duplicate ids")
            if any(not p.get("text") or not p.get("category") for p in prompts):
                raise ProtocolError(f"prompt set {name!r}: every prompt needs text and a category")
        sets = list(d["prompts"].values())
        texts = [p["text"] for prompts in sets for p in prompts]
        if len(set(texts)) != len(texts):
            raise ProtocolError("a prompt appears twice: development and held-out prompts must be disjoint")
        for name, study in d["studies"].items():
            if study["prompts"] not in d["prompts"]:
                raise ProtocolError(f"study {name!r} uses unknown prompt set {study['prompts']!r}")
            if int(study["repetitions"]) < 1 or int(study["max_new_tokens"]) < 1:
                raise ProtocolError(f"study {name!r}: repetitions and max_new_tokens must be positive")

    # -- identity and freezing -----------------------------------------------------------------
    def fingerprint(self, comparison: str | None = None) -> str:
        """Hash of everything that defines the study -- or, with ``comparison``, of everything that
        defines that comparison: its two arms, its criterion, and the regime, order, studies,
        prompts and criteria it shares.  Comments, status and freeze marks are excluded, and so
        are the other comparisons' arms: pinning a native arm later does not unfreeze a
        comparison that does not use it."""
        def strip(obj):
            if isinstance(obj, dict):
                return {k: strip(v) for k, v in obj.items() if not k.startswith("_")
                        and k not in ("status", "frozen")}  # fmt: skip
            if isinstance(obj, list):
                return [strip(v) for v in obj]
            return obj

        data = strip(self.data)
        if comparison is not None:
            c = self.comparison(comparison)
            data["comparisons"] = [strip(c)]
            data["arms"] = {k: data["arms"][k] for k in (c["baseline"], c["candidate"])}
            data.pop("tuning", None)
            if not self.uses_basp(comparison):
                data.pop("setup", None)  # a native-only comparison does not depend on this runtime's setup
        return sha256_json(data)

    def uses_basp(self, comparison: str) -> bool:
        c = self.comparison(comparison)
        return any(self.arms[k]["engine"] == "basp" for k in (c["baseline"], c["candidate"]))

    def check_setup(self, *, target, draft, backend, expert_cache_gib, embedding) -> None:
        """A frozen comparison of this runtime's arms runs on exactly the pinned files and settings."""
        from basp.evidence import file_identity

        setup = self.data.get("setup") or {}
        problems = []
        for role, path in (("target", target), ("draft", draft)):
            want = setup.get(f"{role}_sha256")
            if want and (path is None or file_identity(path)["sha256"] != want):
                problems.append(f"the {role} model is not the pinned file ({setup.get(role)})")
        for key, value in (("backend", backend), ("expert_cache_gib", expert_cache_gib), ("embedding", embedding)):
            if setup.get(key) is not None and setup[key] != value:
                problems.append(f"{key} is {value!r}, the protocol pins {setup[key]!r}")
        if problems:
            raise ProtocolError("this run does not match the frozen setup: " + "; ".join(problems))

    def frozen(self, comparison: str) -> bool:
        return (self.data.get("frozen") or {}).get(comparison) == self.fingerprint(comparison)

    def freeze(self, comparison: str) -> str:
        """Mark one comparison frozen at its present content; returns its fingerprint."""
        c = self.comparison(comparison)
        self.check_arms((c["baseline"], c["candidate"]))  # nothing unpinned may be frozen in
        if self.uses_basp(comparison):
            missing = [k for k in ("target", "target_sha256", "backend", "machine")
                       if not (self.data.get("setup") or {}).get(k)]  # fmt: skip
            if missing:
                raise ProtocolError(f"comparison {comparison!r} runs this runtime: pin setup ({', '.join(missing)}) "
                                    "before freezing")  # fmt: skip
        self.data.setdefault("frozen", {})[comparison] = self.fingerprint(comparison)
        self.data["status"] = "frozen: " + ", ".join(sorted(self.data["frozen"]))
        return self.data["frozen"][comparison]

    def save(self, path: str | Path | None = None) -> None:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ProtocolError("no path to save the protocol to")
        target.write_text(json.dumps(self.data, indent=1) + "\n")

    def check_arms(self, keys) -> None:
        """Every arm about to run must be fully specified; a native arm must pin its build."""
        for key in keys:
            arm = self.arms[key]
            if arm["engine"] == "native":
                missing = [k for k in NATIVE_PINS if not arm.get(k)]
                if missing:
                    raise ProtocolError(f"arm {key} (native engine) is not pinned: set {', '.join(missing)}")
            elif not arm.get("method"):
                raise ProtocolError(f"arm {key} has no method")

    def check_runnable(self, study: str, comparison: str) -> None:
        c = self.comparison(comparison)
        self.check_arms((c["baseline"], c["candidate"]))
        if self.study(study).get("requires_frozen") and not self.frozen(comparison):
            was = (self.data.get("frozen") or {}).get(comparison)
            state = "was changed after it was frozen" if was else "is not frozen"
            raise ProtocolError(f"study {study!r} decides something: comparison {comparison!r} {state}. Finish "
                                "tuning on the development prompts, freeze it, and only then run it.")  # fmt: skip

    # -- the order of trials ----------------------------------------------------------------------
    def schedule(self, study: str, comparison: str) -> list[Trial]:
        """Every trial, in execution order: prompts in a seeded random order per repetition, the
        two arms of a pair back to back, and which arm goes first counterbalanced (AB/BA) so that
        each prompt, and the study as a whole, has both orders as evenly as possible."""
        c = self.comparison(comparison)
        prompts = self.prompts(study)
        reps = int(self.study(study)["repetitions"])
        rng = random.Random(f"{self.data['order']['seed']}:{study}:{comparison}")
        offsets = {p["id"]: i for i, p in enumerate(rng.sample(prompts, len(prompts)))}
        trials: list[Trial] = []
        pair = 0
        for rep in range(reps):
            for p in rng.sample(prompts, len(prompts)):
                first_is_baseline = (offsets[p["id"]] + rep) % 2 == 0
                order = (c["baseline"], c["candidate"]) if first_is_baseline else (c["candidate"], c["baseline"])
                for position, arm in enumerate(order):
                    trials.append(Trial(pair, p["id"], p["category"], rep, arm, position))
                pair += 1
        return trials
