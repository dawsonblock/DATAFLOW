"""Memory: logical registry, residency, replication and allocator (v0.1-v0.3), and the v0.4
physical memory domains -- shared accounting, admission control, safe migration and eviction,
expert-cache policies, hierarchy and prefetching."""

from .admission import Admission, AdmissionController, RequestShape
from .domains import AdmissionRefused, Block, Category, EvictionUnsafe, MemoryDomain, MemoryMap
from .expert_cache import POLICIES, CacheResult, ExpertCost, compare, simulate, trace_steps
from .hierarchy import MemoryHierarchy, Tier
from .migration import Migration, MigrationEngine
from .prefetch import Prefetcher

__all__ = [
    "POLICIES", "Admission", "AdmissionController", "AdmissionRefused", "Block", "CacheResult", "Category",
    "EvictionUnsafe", "ExpertCost", "MemoryDomain", "MemoryHierarchy", "MemoryMap", "Migration",
    "MigrationEngine", "Prefetcher", "RequestShape", "Tier", "compare", "simulate", "trace_steps",
]  # fmt: skip
