"""Strategy implementations for MJX and CPU fallback acceleration."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from .observer import EventSink, ProbeEvent


class AcceleratorStrategy(ABC):
    name: str

    @abstractmethod
    def probe(self, sink: EventSink) -> bool:
        """Return True when this strategy can run on the current machine."""

    @abstractmethod
    def execute(self, values: Sequence[float]) -> dict[str, Any]:
        """Run one minimal acceleration step."""


class CPUFallbackStrategy(AcceleratorStrategy):
    name = "cpu"

    def probe(self, sink: EventSink) -> bool:
        sink.on_event(ProbeEvent(name="probe.cpu.available"))
        return True

    def execute(self, values: Sequence[float]) -> dict[str, Any]:
        return {
            "backend": self.name,
            "count": len(values),
            "total": float(sum(values)),
        }


class MJXAcceleratorStrategy(AcceleratorStrategy):
    name = "mjx"

    def __init__(self) -> None:
        self._import_module = importlib.import_module
        self._device_count = 0
        self._jax: Any | None = None

    def probe(self, sink: EventSink) -> bool:
        sink.on_event(ProbeEvent(name="probe.mjx.started"))
        try:
            jax = self._import_module("jax")
            self._import_module("mujoco.mjx")
            devices = jax.devices() if hasattr(jax, "devices") else ()
            self._device_count = len(tuple(devices))
            self._jax = jax
        except Exception as exc:  # noqa: BLE001
            sink.on_event(
                ProbeEvent(
                    name="probe.mjx.unavailable",
                    payload={
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
            )
            return False

        sink.on_event(
            ProbeEvent(
                name="probe.mjx.available",
                payload={"device_count": self._device_count},
            )
        )
        return True

    def execute(self, values: Sequence[float]) -> dict[str, Any]:
        total = float(sum(values))
        if self._jax is not None and hasattr(self._jax, "numpy"):
            # Minimal real JAX compute path: host -> device -> reduction.
            arr = self._jax.numpy.asarray(list(values), dtype=self._jax.numpy.float32)
            total = float(self._jax.numpy.sum(arr))
        return {
            "backend": self.name,
            "count": len(values),
            "total": total,
            "device_count": self._device_count,
        }
