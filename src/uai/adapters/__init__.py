"""
Provider adapters for the Universal AI Provider SDK.

Each adapter translates between the SDK's unified request/response models
and the provider's specific API format. Adapters implement the
BaseProviderAdapter contract.

Adapter classes are loaded **lazily** (Module 1.6.1 — resource footprint):
``from uai.adapters import DeepSeekAdapter`` works exactly as before via
module ``__getattr__``, but merely importing this package (or
``uai.adapters.base_adapter``, which the client imports) no longer pulls
in every provider module — an application that uses one provider never
pays the import cost of the other seven.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BaseProviderAdapter",
    "DeepSeekAdapter",
    "DoubaoAdapter",
    "GLMAdapter",
    "HunyuanAdapter",
    "KimiAdapter",
    "MiniMaxAdapter",
    "QwenAdapter",
    "StepFunAdapter",
]

# module name -> exported class name
_LAZY = {
    "BaseProviderAdapter": ("uai.adapters.base_adapter", "BaseProviderAdapter"),
    "DeepSeekAdapter": ("uai.adapters.deepseek", "DeepSeekAdapter"),
    "DoubaoAdapter": ("uai.adapters.doubao", "DoubaoAdapter"),
    "GLMAdapter": ("uai.adapters.glm", "GLMAdapter"),
    "HunyuanAdapter": ("uai.adapters.hunyuan", "HunyuanAdapter"),
    "KimiAdapter": ("uai.adapters.kimi", "KimiAdapter"),
    "MiniMaxAdapter": ("uai.adapters.minimax", "MiniMaxAdapter"),
    "QwenAdapter": ("uai.adapters.qwen", "QwenAdapter"),
    "StepFunAdapter": ("uai.adapters.stepfun", "StepFunAdapter"),
}


def __getattr__(name: str) -> Any:
    spec = _LAZY.get(name)
    if spec is None:
        raise AttributeError(f"module 'uai.adapters' has no attribute {name!r}")
    module_name, attr = spec
    import importlib

    return getattr(importlib.import_module(module_name), attr)


def __dir__() -> list[str]:
    return sorted(__all__)
