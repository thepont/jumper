"""Factory for selecting an acceleration strategy."""

from __future__ import annotations

from .observer import EventSink, ProbeEvent
from .strategy import CPUFallbackStrategy, AcceleratorStrategy, MJXAcceleratorStrategy

Preference = str


class AcceleratorFactory:
    """GoF Factory that encapsulates strategy choice policy."""

    def __init__(
        self,
        *,
        mjx_strategy: AcceleratorStrategy | None = None,
        cpu_strategy: AcceleratorStrategy | None = None,
    ) -> None:
        self._mjx = mjx_strategy or MJXAcceleratorStrategy()
        self._cpu = cpu_strategy or CPUFallbackStrategy()

    def create(self, preference: Preference, sink: EventSink) -> AcceleratorStrategy:
        if preference not in {"auto", "mjx", "cpu"}:
            raise ValueError(
                f"unknown accelerator preference {preference!r}; choose auto/mjx/cpu"
            )

        if preference != "cpu" and self._mjx.probe(sink):
            sink.on_event(ProbeEvent(name="selection.chosen", payload={"strategy": "mjx"}))
            return self._mjx

        if preference != "cpu":
            sink.on_event(
                ProbeEvent(
                    name="selection.fallback",
                    payload={"from": "mjx", "to": "cpu"},
                )
            )

        sink.on_event(ProbeEvent(name="selection.chosen", payload={"strategy": "cpu"}))
        return self._cpu
