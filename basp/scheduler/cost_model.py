"""Deterministic cost model: resolves choices into plans with time estimates.

For every valid execution path the runtime estimates

    T_path = T_queue + T_input + T_kernel + T_output + T_sync

with a roofline kernel term ``max(W/B, C/P, W/D)`` for fused streaming kernels.
Estimation is list scheduling over a :class:`PlanState`: transfers are
serialised per link direction, kernels per device.  The kernel descriptors
produced here are the very objects the simulator executes, so estimate and
simulation agree exactly whenever resources are uncontended.
"""

from __future__ import annotations

import math
from collections.abc import Hashable

from basp.core.graph import OpNode
from basp.core.tensor import FormatTable
from basp.core.topology import Topology
from basp.dataflow.base import (
    Choice,
    PathEstimate,
    ResolvedPlan,
    RShard,
    RTransfer,
    ShardChoice,
)
from basp.memory.residency import CopyState, ResidencyMap
from basp.memory.tensor_registry import TensorRegistry
from basp.scheduler.congestion import PlanState
from basp.simulator.compute import (
    Kernel,
    Phase,
    compute_key,
    decode_key,
    kernel_time,
    link_key,
    mem_key,
)

_ALIGN = 64


def _aligned(nbytes: float) -> int:
    return int(math.ceil(nbytes / _ALIGN) * _ALIGN)


class CostModel:
    def __init__(
        self,
        topology: Topology,
        registry: TensorRegistry,
        residency: ResidencyMap,
        formats: FormatTable | None = None,
        *,
        replicas=None,
        compute_devices: frozenset[int] | None = None,
    ) -> None:
        self.topology = topology
        self.registry = registry
        self.residency = residency
        self.formats = formats or registry.formats
        self.replicas = replicas
        self.compute_devices = compute_devices
        self.sync_s = topology.sync_latency_s
        self._energy: dict[tuple, float] = {}

    # -- helpers ------------------------------------------------------------------
    def allowed(self, device: int, op: OpNode) -> bool:
        if op.pinned_device is not None and device != op.pinned_device:
            return False
        if self.compute_devices is not None and device not in self.compute_devices:
            return False
        return self.topology.devices[device].caps.supports(op.opcode)

    def transfer_estimate(self, src: int, dst: int, nbytes: float, *, decode_fmt=None,
                          dst_bytes: float | None = None) -> tuple[float, float]:  # fmt: skip
        """(route latency, bandwidth-limited time) of an uncontended transfer.

        With ``decode_fmt`` the payload is decoded on arrival and ``dst_bytes``
        are written at the destination: the stream is then limited by
        ``max(T_read/link, T_decode, T_write)``.
        """
        topo = self.topology
        moving = nbytes / topo.transfer_rate(src, dst)
        if decode_fmt is not None:
            dev = topo.devices[dst]
            moving = max(moving, nbytes / dev.caps.decode_rate.get(decode_fmt, math.inf),
                         (dst_bytes or nbytes) / dev.mem_bandwidth)  # fmt: skip
        return topo.path_latency(src, dst), moving

    def landing_format(self, tensor: int, src_fmt, dst: int):
        """Format a copy takes when it lands on ``dst``, or ``None`` if ``dst`` cannot use it.

        A copy arriving in a non-canonical but lossless-equivalent representation
        is decoded once on arrival, so repeated use pays no further decode cost.
        """
        caps = self.topology.devices[dst].caps
        canonical = self.registry.spec(tensor).fmt
        if src_fmt is None or src_fmt == canonical:
            return canonical if caps.can_consume(canonical) else None
        if not caps.can_consume(src_fmt):
            return None
        if caps.can_consume(canonical) and self.formats.interchangeable(src_fmt, canonical):
            return canonical
        return src_fmt

    # -- energy -------------------------------------------------------------------
    def _joules_per_unit(self, key: tuple) -> float:
        """Dynamic energy of one unit of work on a resource (same model as the energy report)."""
        if key not in self._energy:
            kind = key[0]
            if kind == "compute":
                value = self.topology.devices[key[1]].power.joules_per_mac
            elif kind == "mem":
                value = self.topology.devices[key[1]].power.joules_per_mem_byte
            elif kind == "link":
                value = self.topology.links[key[1]].joules_per_byte
            else:
                value = 0.0
            self._energy[key] = value
        return self._energy[key]

    def kernel_energy(self, kernel: Kernel, *, sram_hit: bool = False) -> float:
        phases = kernel.sram_hit_phases if sram_hit and kernel.sram_hit_phases else kernel.phases
        return sum(work * self._joules_per_unit(key) for p in phases for key, work in p.demands)

    def transfer_energy(self, tr: RTransfer) -> float:
        per_byte = self._joules_per_unit(("mem", tr.src))
        for hop in self.topology.route(tr.src, tr.dst):
            per_byte += self._joules_per_unit(("link", hop.link, hop.forward))
        written = tr.nbytes if tr.dst_bytes is None else tr.dst_bytes
        return tr.nbytes * per_byte + written * self._joules_per_unit(("mem", tr.dst))

    # -- kernels ------------------------------------------------------------------
    def build_kernel(
        self,
        op: OpNode,
        shard: ShardChoice,
        weights: list[tuple[int, object, float, int | None]],
        *,
        use_sram: bool = False,
    ) -> Kernel | None:
        """Kernel descriptor for ``shard``; ``weights`` holds (tensor, format, bytes, source).

        A ``None`` source means the weight is read from the executing device's
        memory; otherwise it is streamed from the source over the route.
        Returns ``None`` when the device cannot execute the shard.
        """
        topo = self.topology
        d = shard.device
        dev = topo.devices[d]
        caps = dev.caps
        if not caps.supports(op.opcode):
            return None
        main: dict[tuple, float] = {}
        staged: dict[tuple, float] = {}
        delay = 0.0
        extra = op.extra_mem_bytes * shard.fraction
        if extra:
            main[mem_key(d)] = extra
        cacheable: list[tuple[Hashable, int]] = []
        for tid, fmt, nbytes, src in weights:
            if not caps.can_consume(fmt):
                return None
            fused = caps.fused(fmt)
            limited = caps.decode_limited(fmt)
            if src is None:
                if fused:
                    main[mem_key(d)] = main.get(mem_key(d), 0.0) + nbytes
                    if limited:
                        key = decode_key(d, fmt)
                        main[key] = main.get(key, 0.0) + nbytes
                else:
                    spec = self.registry.spec(tid)
                    expanded = (
                        self.formats.weight_bytes(spec.n_weights, caps.expanded_format)
                        * op.weight_fraction
                    )
                    staged[mem_key(d)] = staged.get(mem_key(d), 0.0) + nbytes + expanded
                    if limited:
                        key = decode_key(d, fmt)
                        staged[key] = staged.get(key, 0.0) + nbytes
                    main[mem_key(d)] = main.get(mem_key(d), 0.0) + expanded
                cacheable.append((tid, int(nbytes)))
            else:
                if not (caps.stream_ingest and fused):
                    return None
                main[mem_key(src)] = main.get(mem_key(src), 0.0) + nbytes
                for hop in topo.route(src, d):
                    key = link_key(hop.link, hop.forward)
                    main[key] = main.get(key, 0.0) + nbytes
                if limited:
                    key = decode_key(d, fmt)
                    main[key] = main.get(key, 0.0) + nbytes
                delay = max(delay, topo.path_latency(src, d))
        if shard.macs:
            main[compute_key(d)] = float(shard.macs)
        phases: list[Phase] = []
        if staged:
            phases.append(Phase(tuple(staged.items())))
        phases.append(Phase(tuple(main.items()), delay))
        sram_tensors: tuple = ()
        hit_phases = None
        if use_sram and cacheable and sum(b for _, b in cacheable) <= dev.sram_bytes:
            sram_tensors = tuple(cacheable)
            hit: dict[tuple, float] = {}
            if extra:
                hit[mem_key(d)] = extra
            if shard.macs:
                hit[compute_key(d)] = float(shard.macs)
            hit_phases = (Phase(tuple(hit.items())),)
        return Kernel(d, tuple(phases), float(shard.macs), dev.kernel_startup_s,
                      dev.warm_startup_s, sram_tensors, hit_phases)  # fmt: skip

    def local_time(self, op: OpNode, device: int) -> float | None:
        """Idle-system time of ``op`` on ``device`` if all weights were resident there.

        Activations are assumed to live on ``op.output_device`` (true for expert
        operations, whose inputs and consumers are on the trunk device).
        """
        weights = []
        for tid in op.weights:
            spec = self.registry.spec(tid)
            weights.append((tid, spec.fmt, spec.nbytes * op.weight_fraction, None))
        kernel = self.build_kernel(op, ShardChoice(device, op.weights, float(op.macs)), weights)
        if kernel is None:
            return None
        total = kernel_time(self.topology, kernel)
        home = op.output_device
        if home is not None and home != device:
            t_in = max(
                (sum(self.transfer_estimate(home, device, self.registry.spec(t).nbytes))
                 for t in op.inputs),
                default=0.0,
            )  # fmt: skip
            t_out = max(
                (sum(self.transfer_estimate(device, home, self.registry.spec(t).nbytes))
                 for t in op.outputs),
                default=0.0,
            )  # fmt: skip
            total += t_in + t_out
        return total

    # -- transfers ----------------------------------------------------------------
    def _transfer(self, key, src, dst, nbytes, st: PlanState, tag, *, ready=None, retain=False,
                  after_ops=(), fmt=None, store_fmt=None, dst_bytes=None) -> RTransfer:  # fmt: skip
        topo = self.topology
        ready = st.now if ready is None else max(st.now, ready)
        transcode = store_fmt is not None and store_fmt != fmt
        latency, moving = self.transfer_estimate(
            src, dst, nbytes, decode_fmt=fmt if transcode else None, dst_bytes=dst_bytes
        )
        start, end = st.book(topo.route(src, dst), ready, moving)
        st.link_bytes += nbytes
        return RTransfer(key, src, dst, nbytes, tag, retain, start, end + latency, tuple(after_ops),
                         fmt, store_fmt, dst_bytes if transcode else None)  # fmt: skip

    def _ensure(self, key, dst, st: PlanState, transfers, tag, *, nbytes, src=None, ready=None,
                retain=False, after_ops=(), fmt=None, store_fmt=None, dst_bytes=None,
                waits=None) -> float | None:  # fmt: skip
        """Make ``key`` available on ``dst``; returns the time it will be usable."""
        t = st.avail(key, dst)
        if t is not None:
            if waits is not None and isinstance(key, int) and (key, dst) not in st.arrivals:
                copy = st.residency.copy(key, dst)
                if copy is not None and copy.state is CopyState.PENDING:
                    waits.append((key, dst))
            return t
        if src is None:
            best = None
            for device, t_src in st.sources(key):
                latency, bw = self.transfer_estimate(device, dst, nbytes)
                arrive = max(t_src, st.now) + latency + bw
                if best is None or arrive < best[0]:
                    best = (arrive, device, t_src)
            if best is None:
                return None
            src, ready = best[1], best[2]
        tr = self._transfer(key, src, dst, nbytes, st, tag, ready=ready, retain=retain,
                            after_ops=after_ops, fmt=fmt, store_fmt=store_fmt,
                            dst_bytes=dst_bytes)  # fmt: skip
        transfers.append(tr)
        st.add_arrival(key, dst, tr.arrive)
        if store_fmt is not None:
            st.formats[(key, dst)] = store_fmt
        return tr.arrive

    def reserve(self, device: int, nbytes: float, st: PlanState, evict: list, value: float,
                *, retain: bool) -> bool:  # fmt: skip
        """Reserve replica memory on ``device``, tentatively evicting victims if needed."""
        need = _aligned(nbytes)
        free = st.free_bytes(device)
        if free < need:
            if self.replicas is None:
                return False
            victims = st.victim_list(device)
            i = st.victim_idx.get(device, 0)
            while free < need and i < len(victims):
                victim = victims[i]
                key = (victim.tensor, device)
                if key in st.locked or key in st.gone:
                    i += 1
                    continue
                if retain and not self.replicas.admits(value, victim):
                    break
                i += 1
                free += victim.alloc.size
                st.gone.add(key)
                evict.append(key)
            st.victim_idx[device] = i
            if free < need:
                return False
        st.free[device] = free - need
        return True

    def _follow_device(self, op: OpNode, st: PlanState) -> int | None:
        """Device for a weightless operation: where its inputs can be gathered soonest."""
        candidates: set[int] = set()
        for tid in op.inputs:
            candidates.update(device for device, _ in st.sources(tid))
        best = None
        for device in sorted(candidates):
            if not self.allowed(device, op):
                continue
            ready = st.device_free.get(device, st.now)
            for tid in op.inputs:
                t = st.avail(tid, device)
                if t is None:
                    nbytes = self.registry.spec(tid).nbytes
                    t = min(
                        (max(ts, st.now) + sum(self.transfer_estimate(src, device, nbytes))
                         for src, ts in st.sources(tid)),
                        default=math.inf,
                    )  # fmt: skip
                ready = max(ready, t)
            if best is None or ready < best[0]:
                best = (ready, device)
        if best is not None:
            return best[1]
        host = self.topology.host
        return host if self.allowed(host, op) else None

    # -- resolution ---------------------------------------------------------------
    def resolve(self, op: OpNode, choice: Choice, st: PlanState) -> ResolvedPlan | None:
        """Turn ``choice`` into a concrete plan, advancing ``st``.

        Returns ``None`` when the choice is infeasible in this state; ``st`` is
        then left in an undefined condition, so callers resolve on a fork.
        """
        topo = self.topology
        registry = self.registry
        now = st.now
        shard_choices = choice.shards
        if choice.follow_inputs:
            device = self._follow_device(op, st)
            if device is None:
                return None
            shard_choices = (ShardChoice(device, (), float(op.macs)),)
        group = op.attrs.get("reduce_group")
        reducer = bool(op.attrs.get("reducer"))
        multi = len(shard_choices) > 1
        # tensor-parallel shards each produce a full-size partial output that is summed
        partials_out = bool(op.attrs.get("partial_outputs"))
        link_before = st.link_bytes
        shards: list[RShard] = []
        energy = 0.0
        finish = now
        critical = (0.0, 0.0, 0.0, 0.0)
        weight_wait = 0.0

        for sc in shard_choices:
            d = sc.device
            transfers: list[RTransfer] = []
            waits: list[tuple[Hashable, int]] = []
            evict: list[tuple[int, int]] = []
            transient: list[int] = []
            local: list[int] = []
            t_in = now

            for tid in op.inputs:
                partials = st.accum_of.get(tid, ()) if reducer else ()
                if partials:
                    # the input was folded into device-local partial sums: gather those instead
                    for partial in partials:
                        ready, contributors, _rows, nbytes = st.accum[partial]
                        if partial[2] == d:
                            t = ready
                        else:
                            t = self._ensure(partial, d, st, transfers, "output", nbytes=nbytes,
                                             src=partial[2], ready=ready,
                                             after_ops=contributors)  # fmt: skip
                        if t is None:
                            return None
                        t_in = max(t_in, t)
                    continue
                t = self._ensure(tid, d, st, transfers, "activation",
                                 nbytes=registry.spec(tid).nbytes, waits=waits)  # fmt: skip
                if t is None:
                    return None
                t_in = max(t_in, t)
            t_activations = t_in

            weights: list[tuple[int, object, float, int | None]] = []
            for tid in sc.weights:
                src = sc.source(tid)
                here = st.avail(tid, d)
                if here is not None:
                    fmt = st.format_at(tid, d) or registry.spec(tid).fmt
                    copy = st.residency.copy(tid, d)
                    if (copy is not None and copy.state is CopyState.PENDING
                            and (tid, d) not in st.arrivals):  # fmt: skip
                        waits.append((tid, d))
                    weights.append((tid, fmt, registry.nbytes(tid, fmt) * op.weight_fraction, None))
                    local.append(tid)
                    t_in = max(t_in, here)
                    continue
                if src is None or st.avail(tid, src) is None:
                    return None
                fmt = st.format_at(tid, src) or registry.spec(tid).fmt
                full = registry.nbytes(tid, fmt)
                if choice.stream:
                    weights.append((tid, fmt, full * op.weight_fraction, src))
                    continue
                landed = self.landing_format(tid, fmt, d)
                if landed is None:
                    return None
                stored = registry.nbytes(tid, landed)
                value = math.inf
                if choice.retain and self.replicas is not None:
                    value = self.replicas.value(tid)
                if not self.reserve(d, stored, st, evict, value, retain=choice.retain):
                    return None
                t = self._ensure(tid, d, st, transfers, "weights", nbytes=full, src=src,
                                 ready=st.avail(tid, src), retain=choice.retain, fmt=fmt,
                                 store_fmt=landed, dst_bytes=stored)  # fmt: skip
                if not choice.retain:
                    transient.append(tid)
                weights.append((tid, landed, stored * op.weight_fraction, None))
                local.append(tid)
                t_in = max(t_in, t)

            kernel = self.build_kernel(op, sc, weights, use_sram=choice.use_sram)
            if kernel is None:
                return None
            free = st.device_free.get(d)
            t_start = t_in if free is None else max(t_in, free)
            # back-to-back with the previous kernel on this device: one batched launch
            warm = topo.devices[d].persistent or (free is not None and t_start == free)
            streamed = [(nbytes, src) for _, _, nbytes, src in weights if src is not None]
            sram_ids = [t for t, _ in kernel.sram_tensors]
            hit = bool(sram_ids) and st.sram_has(d, sram_ids)
            duration = kernel_time(topo, kernel, sram_hit=hit, warm=warm)
            stream_wait = shared = 0.0
            for nbytes, src in streamed:
                # a streamed kernel holds its links for as long as the weights flow
                moving = nbytes / topo.transfer_rate(src, d)
                t_start, link_end = st.book(topo.route(src, d), t_start, moving)
                shared = max(shared, link_end - t_start - moving)  # time lost to link sharing
                st.link_bytes += nbytes
                stream_wait = max(stream_wait, topo.path_latency(src, d) + moving)
            duration += shared
            t_done = t_start + duration
            st.device_free[d] = t_done
            if sram_ids:
                st.sram_add(d, sram_ids)
            for tid in transient:
                st.remove_arrival(tid, d)  # freed after this kernel: invisible to later plans
                st.formats.pop((tid, d), None)
            weight_wait += (t_in - t_activations) + stream_wait

            outputs: list[RTransfer] = []
            t_out = t_done
            partial_key = None
            if not multi:
                for tid in op.outputs:
                    st.add_arrival(tid, d, t_done)
            if choice.accumulate and group is not None and (not multi or partials_out):
                partial_key = ("partial", group, d)
                ready, contributors, rows, _ = st.accum.get(partial_key, (0.0, (), frozenset(), 0.0))
                rows = rows | frozenset(op.attrs.get("rows", range(op.batch)))
                per_row = sum(registry.spec(t).nbytes for t in op.outputs) / max(op.batch, 1)
                st.accum[partial_key] = (max(ready, t_done), contributors + (op.op_id,), rows,
                                         per_row * len(rows))  # fmt: skip
                for tid in op.outputs:
                    if partial_key not in st.accum_of.get(tid, ()):
                        st.accum_of[tid] = st.accum_of.get(tid, ()) + (partial_key,)
            elif op.output_device is not None and op.output_device != d:
                for tid in op.outputs:
                    nbytes = registry.spec(tid).nbytes * (1.0 if partials_out else sc.fraction)
                    tr = self._transfer(tid, d, op.output_device, nbytes, st, "output", ready=t_done)
                    outputs.append(tr)
                    st.add_arrival(tid, op.output_device, tr.arrive)
                    t_out = max(t_out, tr.arrive)
            elif multi and op.output_device is not None:
                for tid in op.outputs:
                    st.add_arrival(tid, d, t_done)

            shards.append(RShard(d, transfers, kernel, t_start, t_done, outputs, evict,
                                 tuple(local), tuple(transient), partial_key, hit, tuple(waits)))  # fmt: skip
            energy += self.kernel_energy(kernel, sram_hit=hit)
            energy += sum(self.transfer_energy(tr) for tr in (*transfers, *outputs))
            if t_out >= finish:
                finish = t_out
                critical = (t_start - t_in, t_in - now, duration, t_out - t_done)

        estimate = PathEstimate(queue=critical[0], input=critical[1], kernel=critical[2],
                                output=critical[3], sync=0.0)  # fmt: skip
        return ResolvedPlan(op, choice, shards, now, finish, estimate,
                            st.link_bytes - link_before, weight_wait, energy)  # fmt: skip
