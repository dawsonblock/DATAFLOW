"""Command emission: resolved plans -> BSP command streams.

Each plan becomes a short stream of command descriptors whose dependencies are
resolved by the devices from completion events: TRANSFER the inputs, EXECUTE (or
REDUCE) the kernel, TRANSFER the outputs.  Weight movement is bracketed by
FREE/ALLOCATE so data only lands in memory reserved for it.  Operand lists
longer than the descriptor's tensor slots live in the kernel descriptor the
command refers to.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field

from basp.backends.virtual.system import VirtualSystem
from basp.core.graph import OpCode
from basp.core.tensor import DType, TensorKind, activation_bytes
from basp.dataflow.base import ResolvedPlan, RTransfer
from basp.memory.replication import ReplicaAction
from basp.simulator.compute import Kernel
from basp.streams.commands import MAX_DEPS, MAX_TENSORS, Command, fan_in
from basp.streams.protocol import Flags, Opcode


@dataclass
class Emitted:
    cmd: Command
    kernel: Kernel | None = None
    nbytes: float | None = None
    tag: str = ""
    reads: tuple[tuple[int, int], ...] = ()
    writes: tuple[int, ...] = ()
    max_rate: float | None = None
    piece: bool = False
    dst_bytes: float | None = None
    meta: Hashable = None


@dataclass
class EmittedPlan:
    plan: ResolvedPlan
    commands: list[Emitted] = field(default_factory=list)
    final: list[int] = field(default_factory=list)  # commands whose completion finishes the plan
    executes: list[int] = field(default_factory=list)


class Emitter:
    def __init__(self, system: VirtualSystem, hidden: int, batch: int) -> None:
        self.system = system
        self.registry = system.registry
        self.residency = system.residency
        self._partial_bytes = activation_bytes(hidden * batch)
        self._partial_shape = (batch, hidden)
        self.new_step()

    def new_step(self) -> None:
        """Forget per-step bookkeeping (temporaries of the previous step are gone)."""
        self._arrival: dict[tuple[Hashable, int], tuple[int, ...]] = {}
        self._partial_id: dict[Hashable, int] = {}
        self._accum_keys: dict[int, list[Hashable]] = {}  # output -> partial sums holding it
        self._exec: dict[int, list[int]] = {}
        self.partials: list[int] = []

    # -- helpers ------------------------------------------------------------------
    def _tensor(self, key: Hashable) -> int:
        if isinstance(key, int):
            return key
        if key not in self._partial_id:
            spec = self.registry.add(
                f"partial.{key[1]}.d{key[2]}", TensorKind.INTERMEDIATE, self._partial_shape,
                DType.FP16, self._partial_bytes,
            )  # fmt: skip
            self._partial_id[key] = spec.tensor_id
            self.partials.append(spec.tensor_id)
        return self._partial_id[key]

    def _arrived(self, key: Hashable, device: int, cmd_id: int) -> None:
        self._arrival[(key, device)] = self._arrival.get((key, device), ()) + (cmd_id,)

    def _narrow(self, deps: list[int], out: list[Emitted]) -> tuple[int, ...]:
        unique = tuple(dict.fromkeys(deps))
        if len(unique) <= MAX_DEPS:
            return unique
        barriers, narrowed = fan_in(unique, self.system.new_id)
        out.extend(Emitted(b, tag="barrier") for b in barriers)
        return narrowed

    def _move(self, tr: RTransfer, out: list[Emitted], extra_deps: list[int], *, piece: bool = False,
              meta: Hashable = None) -> int:  # fmt: skip
        """Emit the commands for one transfer; returns the id of the TRANSFER."""
        tensor = self._tensor(tr.tensor)
        deps = list(extra_deps)
        deps.extend(self._arrival.get((tr.tensor, tr.src), ()))
        for op_id in tr.after_ops:
            deps.extend(self._exec.get(op_id, ()))
        flags = Flags.NONE
        if tr.tag == "weights":
            stored = tr.store_fmt if tr.store_fmt is not None else tr.fmt
            alloc = Command(
                self.system.new_id(), Opcode.ALLOCATE, tr.dst, inputs=(tensor,), deps=tuple(extra_deps),
                precision=int(stored) if stored is not None else 0,
                arg0=max(int(tr.dst_bytes or tr.nbytes), 1), arg1=max(int(tr.arrive * 1e9), 1),
            )  # fmt: skip
            out.append(Emitted(alloc, tag="alloc"))
            deps.append(alloc.op_id)
            if tr.retain:
                flags |= Flags.RETAIN
        cmd = Command(
            self.system.new_id(), Opcode.TRANSFER, tr.dst, inputs=(tensor,),
            deps=self._narrow(deps, out), version=self.registry.version(tensor), src_device=tr.src,
            flags=flags, arg0=int(tr.nbytes), precision=int(tr.fmt) if tr.fmt is not None else 0,
        )  # fmt: skip
        out.append(Emitted(cmd, nbytes=tr.nbytes, tag=tr.tag, piece=piece, meta=meta,
                           dst_bytes=tr.dst_bytes, reads=((tensor, tr.src),)))  # fmt: skip
        self._arrived(tr.tensor, tr.dst, cmd.op_id)
        return cmd.op_id

    # -- plans --------------------------------------------------------------------
    def emit_plan(self, plan: ResolvedPlan) -> EmittedPlan:
        op = plan.op
        result = EmittedPlan(plan)
        out = result.commands
        multi = len(plan.shards) > 1
        reducer = bool(op.attrs.get("reducer"))
        self._exec.setdefault(op.op_id, [])
        for shard in plan.shards:
            d = shard.device
            frees: list[int] = []
            for tensor, device in shard.evict:
                free = Command(self.system.new_id(), Opcode.FREE, device, inputs=(tensor,))
                out.append(Emitted(free, tag="evict"))
                frees.append(free.op_id)
            deps: list[int] = []
            for tr in shard.inputs:
                extra = frees if tr.tag == "weights" else []
                deps.append(self._move(tr, out, extra, meta=("in", op.op_id)))
            for key, device in shard.waits_for:
                loading = self.system.loading.get((key, device))
                if loading is not None:
                    self.system.promote(loading)
                    deps.append(loading)
            reads: list[tuple[int, int]] = []
            for tid in op.inputs:
                keys = self._accum_keys.get(tid, [tid]) if reducer else [tid]
                for key in keys:
                    deps.extend(self._arrival.get((key, d), ()))
                    reads.append((self._tensor(key), d))
            for tid in shard.local_weights:
                deps.extend(self._arrival.get((tid, d), ()))
                reads.append((tid, d))
            for sc in plan.choice.shards:
                if sc.device == d and plan.choice.stream:
                    for tid, src in sc.remote:
                        if tid not in shard.local_weights:
                            reads.append((tid, src))
            writes = list(op.outputs)
            flags = Flags.NONE
            if plan.choice.stream:
                flags |= Flags.STREAM
            if shard.kernel.sram_tensors:
                flags |= Flags.SRAM
            if len(shard.kernel.phases) > 1:
                flags |= Flags.STAGED
            if shard.partial is not None:
                flags |= Flags.ACCUMULATE
                partial = self._tensor(shard.partial)
                writes = [partial]
                for tid in op.outputs:
                    held = self._accum_keys.setdefault(tid, [])
                    if shard.partial not in held:
                        held.append(shard.partial)
            opcode = Opcode.REDUCE if op.opcode is OpCode.REDUCE else Opcode.EXECUTE
            unique_reads = tuple(dict.fromkeys(reads))
            cmd = Command(
                self.system.new_id(), opcode, d,
                inputs=tuple(dict.fromkeys(t for t, _ in unique_reads))[:MAX_TENSORS],
                outputs=tuple(writes)[:MAX_TENSORS], deps=self._narrow(deps, out), flags=flags,
                precision=int(self.registry.spec(op.weights[0]).fmt or 0) if op.weights else 0,
            )  # fmt: skip
            out.append(Emitted(cmd, kernel=shard.kernel, tag="kernel", reads=unique_reads,
                               writes=tuple(writes), meta=("op", op.op_id)))  # fmt: skip
            result.executes.append(cmd.op_id)
            self._exec[op.op_id].append(cmd.op_id)
            if shard.partial is not None:
                self._arrived(shard.partial, d, cmd.op_id)
            else:
                for tid in op.outputs:
                    self._arrived(tid, d, cmd.op_id)
            for tid in shard.transient:
                free = Command(self.system.new_id(), Opcode.FREE, d, inputs=(tid,), deps=(cmd.op_id,))
                out.append(Emitted(free, tag="scratch"))
            last = [cmd.op_id]
            if shard.outputs:
                last = [
                    self._move(tr, out, [cmd.op_id], piece=multi, meta=("out", op.op_id))
                    for tr in shard.outputs
                ]
            result.final.extend(last)
        return result

    def emit_broadcast(self, tensor: int, src: int, targets, after: list[int]) -> list[Emitted]:
        """Stream ``tensor`` from ``src`` to every target as soon as ``after`` have completed."""
        out: list[Emitted] = []
        nbytes = self.registry.spec(tensor).nbytes
        for dst in targets:
            if (tensor, dst) in self._arrival:
                continue
            cmd = Command(
                self.system.new_id(), Opcode.TRANSFER, dst, inputs=(tensor,), deps=tuple(after),
                version=self.registry.version(tensor), src_device=src, arg0=int(nbytes),
            )  # fmt: skip
            out.append(Emitted(cmd, nbytes=nbytes, tag="activation", meta=("eager", tensor),
                               reads=((tensor, src),)))  # fmt: skip
            self._arrived(tensor, dst, cmd.op_id)
        return out

    def emit_fill(self, action: ReplicaAction, eta: float) -> list[Emitted]:
        """Background replication or speculative prefetch of one tensor."""
        out: list[Emitted] = []
        frees: list[int] = []
        for tensor, device in action.evict:
            free = Command(self.system.new_id(), Opcode.FREE, device, inputs=(tensor,))
            out.append(Emitted(free, tag="evict"))
            frees.append(free.op_id)
        flags = Flags.SPECULATIVE if action.speculative else Flags.NONE
        stored = action.store_fmt if action.store_fmt is not None else action.fmt
        alloc = Command(
            self.system.new_id(), Opcode.ALLOCATE, action.dst, inputs=(action.tensor,),
            deps=tuple(frees), precision=int(stored) if stored is not None else 0,
            flags=flags, arg0=max(int(action.dst_bytes or action.nbytes), 1),
            arg1=max(int(eta * 1e9), 1),
        )  # fmt: skip
        out.append(Emitted(alloc, tag="alloc"))
        opcode = Opcode.PREFETCH if action.speculative else Opcode.TRANSFER
        move = Command(
            self.system.new_id(), opcode, action.dst, inputs=(action.tensor,), deps=(alloc.op_id,),
            priority=1, version=self.registry.version(action.tensor), src_device=action.src,
            flags=flags | Flags.RETAIN, arg0=int(action.nbytes),
            precision=int(action.fmt) if action.fmt is not None else 0,
        )  # fmt: skip
        out.append(Emitted(move, nbytes=action.nbytes, tag=action.tag,
                           dst_bytes=action.dst_bytes, reads=((action.tensor, action.src),),
                           meta=("fill", action.tensor)))  # fmt: skip
        return out
