"""Predictive dataflow: prefetch tensors the model will probably need next.

The predictor estimates which experts upcoming layers will select; the
prefetcher turns confident predictions into speculative, low-priority,
rate-capped loads onto cache devices so the transfer overlaps current
execution.  Prediction never influences which experts the model selects: a
wrong guess only wastes budgeted bandwidth, and its load is cancelled (if still
in flight) or becomes the first eviction victim (if it completed unused).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from basp.core.graph import OpNode
from basp.dataflow.base import PrefetchPolicy
from basp.memory.replication import ReplicaAction, ReplicaManager
from basp.memory.residency import CopyState
from basp.scheduler.congestion import PlanState


class Prefetcher:
    def __init__(
        self,
        policy: PrefetchPolicy,
        predictor,
        replicas: ReplicaManager,
        scheduler,
        *,
        n_layers: int,
        expert_tensor: Callable[[int, int], int],
        probe_op: Callable[[int, int], OpNode],
    ) -> None:
        self.policy = policy
        self.predictor = predictor
        self.replicas = replicas
        self.scheduler = scheduler
        self.n_layers = n_layers
        self.expert_tensor = expert_tensor
        self.probe_op = probe_op
        self.outstanding: dict[tuple[int, int], tuple[int, int]] = {}  # (tensor, device) -> (layer, expert)
        self.stats: Counter = Counter()

    def propose(self, layer_done: int, st: PlanState) -> list[ReplicaAction]:
        """Speculative loads for the layers following ``layer_done``."""
        actions: list[ReplicaAction] = []
        cost = self.scheduler.cost
        for ahead in range(1, self.policy.lookahead_layers + 1):
            layer = (layer_done + ahead) % self.n_layers
            for expert, probability in self.predictor.predict(layer, self.policy.candidates):
                if probability < self.policy.min_probability:
                    break
                tensor = self.expert_tensor(layer, expert)
                op = self.probe_op(layer, expert)
                for dst in self.replicas.policy.devices:
                    if st.avail(tensor, dst) is not None or not cost.allowed(dst, op):
                        continue
                    local = cost.local_time(op, dst)
                    if local is None:
                        continue
                    saving = self.scheduler.best_time(op, st.now) - local
                    action = self.replicas.propose(
                        tensor, dst, st, cost, value=float(probability), saving_s=saving,
                        speculative=True, tag="prefetch",
                    )  # fmt: skip
                    if action is not None:
                        actions.append(action)
                        self.outstanding[(tensor, dst)] = (layer, expert)
                        self.replicas.protected.add((tensor, dst))
                        self.stats["issued"] += 1
                        self.stats["issued_bytes"] += int(action.nbytes)
        return actions

    def reveal(self, layer: int, selected: set[int]) -> list[tuple[int, int]]:
        """Routing for ``layer`` is known: settle the predictions made for it.

        A correct prediction turns the speculative copy into an ordinary replica.
        A wrong one is returned for cancellation if still in flight; if it has
        already landed it stays marked speculative and is the first eviction
        victim.  Until its layer is revealed a prefetched copy cannot be evicted.
        """
        cancel: list[tuple[int, int]] = []
        residency = self.replicas.residency
        for key, (pred_layer, expert) in list(self.outstanding.items()):
            if pred_layer != layer:
                continue
            del self.outstanding[key]
            self.replicas.protected.discard(key)
            copy = residency.copy(*key)
            if expert in selected:
                self.stats["hits"] += copy is not None
                if copy is not None:
                    copy.speculative = False
            else:
                self.stats["mispredicted"] += 1
                if copy is not None and copy.state is CopyState.VALID:
                    self.stats["wasted_bytes"] += copy.nbytes
                elif self.policy.cancel_mispredicted:
                    cancel.append(key)
        return cancel
