"""The physical runtime (v0.4): real GGUF models executed on real backends.

Today one backend is implemented -- NumPy float32 on the CPU -- and one model family, Qwen3
(dense and mixture-of-experts).  CUDA, Metal, GGML and remote backends are declared and report
themselves unavailable; a plan that needs them fails before anything runs.
"""

from .config import ModelConfig, UnsupportedModel
from .device_manager import DeviceManager
from .engine import Engine
from .execution_plan import ExecutionPlan, make_plan
from .generation import GenerationResult, generate
from .model import ContextOverflow, ForwardOutput, KVCache
from .session import Session
from .telemetry import Telemetry

__all__ = [
    "ContextOverflow", "DeviceManager", "Engine", "ExecutionPlan", "ForwardOutput", "GenerationResult", "KVCache",
    "ModelConfig", "Session", "Telemetry", "UnsupportedModel", "generate", "make_plan",
]  # fmt: skip
