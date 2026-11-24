"""Planning context shared by the dataflow mode generators."""

from __future__ import annotations

from dataclasses import dataclass

from basp.core.graph import OpNode
from basp.core.topology import Topology
from basp.dataflow.base import ExecutionPolicy
from basp.memory.residency import ResidencyMap
from basp.memory.tensor_registry import TensorRegistry
from basp.scheduler.congestion import PlanState


@dataclass
class PlanContext:
    topology: Topology
    registry: TensorRegistry
    residency: ResidencyMap
    policy: ExecutionPolicy
    cost: object  # CostModel (kept untyped to avoid an import cycle)
    st: PlanState

    def allowed(self, device: int, op: OpNode) -> bool:
        return self.cost.allowed(device, op)

    def devices_for(self, op: OpNode) -> list[int]:
        """Devices that may execute ``op`` at all."""
        if op.pinned_device is not None:
            return [op.pinned_device] if self.allowed(op.pinned_device, op) else []
        return [d for d in sorted(self.topology.devices) if self.allowed(d, op)]

    def consumable(self, tensor: int, holder: int, executor: int) -> bool:
        """Can ``executor`` consume the copy of ``tensor`` stored on ``holder``?"""
        fmt = self.st.format_at(tensor, holder) or self.registry.spec(tensor).fmt
        return self.topology.devices[executor].caps.can_consume(fmt)

    def fused(self, tensor: int, holder: int, executor: int) -> bool:
        fmt = self.st.format_at(tensor, holder) or self.registry.spec(tensor).fmt
        return self.topology.devices[executor].caps.fused(fmt)

    def weights_local(self, op: OpNode, device: int) -> bool:
        return all(
            self.st.avail(w, device) is not None and self.consumable(w, device, device)
            for w in op.weights
        )

    def inputs_local(self, op: OpNode, device: int) -> bool:
        """No input of ``op`` has to cross a link to reach ``device``.

        Inputs that do not exist yet (produced earlier in the same batch) cannot
        be located and do not count against a device.
        """
        for tid in op.inputs:
            if self.st.avail(tid, device) is None and self.st.sources(tid):
                return False
        return True

    def activation_devices(self, op: OpNode) -> list[int]:
        """Devices where the operation's activations are, or where its results go.

        When no compute-capable device holds the weights at all (cold weights on
        storage), every compute device is a possible destination for a load.
        """
        if op.pinned_device is not None:
            return self.devices_for(op)
        found: set[int] = set()
        for tid in op.inputs:
            found.update(device for device, _ in self.st.sources(tid))
        if op.output_device is not None:
            found.add(op.output_device)
        cold = op.weights and not any(
            self.allowed(device, op) for w in op.weights for device, _ in self.st.sources(w)
        )
        if cold:
            found.update(self.topology.devices)
        return sorted(d for d in found if self.allowed(d, op))

    def best_source(self, tensor: int, dst: int, *, fused: bool = False) -> int | None:
        """Holder from which ``tensor`` reaches ``dst`` soonest in a format it can use.

        Comparing holders compares representations: a smaller compressed copy
        wins only if the destination decodes it faster than the link saves.
        """
        best = None
        for src, t_src in self.st.sources(tensor):
            if src == dst:
                continue
            fmt = self.st.format_at(tensor, src) or self.registry.spec(tensor).fmt
            nbytes = self.registry.nbytes(tensor, fmt)
            if fused:
                if not self.fused(tensor, src, dst):
                    continue
                estimate = self.cost.transfer_estimate(src, dst, nbytes)
            else:
                landed = self.cost.landing_format(tensor, fmt, dst)
                if landed is None:
                    continue
                estimate = self.cost.transfer_estimate(
                    src, dst, nbytes, decode_fmt=fmt if landed != fmt else None,
                    dst_bytes=self.registry.nbytes(tensor, landed),
                )
            arrive = max(t_src, self.st.now) + sum(estimate)
            if best is None or arrive < best[0]:
                best = (arrive, src)
        return None if best is None else best[1]
