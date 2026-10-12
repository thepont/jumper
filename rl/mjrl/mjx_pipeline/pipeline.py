"""MJX acceleration pipeline scaffold."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .factory import AcceleratorFactory
from .observer import EventSink, NullEventSink, ProbeEvent


@dataclass(frozen=True)
class PipelineReport:
    requested: str
    selected: str
    result: dict[str, Any]


class MJXAccelerationPipeline:
    """Facade over strategy selection and execution."""

    def __init__(
        self,
        *,
        factory: AcceleratorFactory | None = None,
        sink: EventSink | None = None,
    ) -> None:
        self._factory = factory or AcceleratorFactory()
        self._sink = sink or NullEventSink()

    def run(self, values: Sequence[float], *, preference: str = "auto") -> PipelineReport:
        self._sink.on_event(ProbeEvent(name="pipeline.started", payload={"requested": preference}))
        strategy = self._factory.create(preference, self._sink)
        result = strategy.execute(values)
        self._sink.on_event(
            ProbeEvent(
                name="pipeline.finished",
                payload={"selected": strategy.name, "count": len(values)},
            )
        )
        return PipelineReport(requested=preference, selected=strategy.name, result=result)
