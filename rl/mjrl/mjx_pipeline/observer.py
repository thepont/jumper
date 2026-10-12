"""Observer primitives for MJX probe and strategy selection events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ProbeEvent:
    name: str
    payload: dict[str, Any] = field(default_factory=dict)


class EventSink(Protocol):
    def on_event(self, event: ProbeEvent) -> None:
        """Consume one runtime event."""


class NullEventSink:
    """Default sink that keeps the pipeline side-effect free."""

    def on_event(self, event: ProbeEvent) -> None:
        _ = event


class ListEventSink:
    """Test-friendly sink that records events in emission order."""

    def __init__(self) -> None:
        self.events: list[ProbeEvent] = []

    def on_event(self, event: ProbeEvent) -> None:
        self.events.append(event)


class PrintEventSink:
    """Stream events immediately for long-running remote jobs."""

    def on_event(self, event: ProbeEvent) -> None:
        details = ", ".join(f"{k}={v}" for k, v in sorted(event.payload.items()))
        print(f"- {event.name}" + (f" ({details})" if details else ""), flush=True)


class TeeEventSink:
    """Fan out events to multiple sinks."""

    def __init__(self, *sinks: EventSink) -> None:
        self._sinks = sinks

    def on_event(self, event: ProbeEvent) -> None:
        for sink in self._sinks:
            sink.on_event(event)
