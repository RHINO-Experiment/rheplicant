"""JAX-free records used at the configuration/bootstrap boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | Sequence["JsonValue"] | Mapping[str, "JsonValue"]
Stage: TypeAlias = Literal[
    "source",
    "raw_process_entry",
    "preset_layering",
    "effective_process_entry",
    "output_preflight",
    "runtime",
    "plugins",
    "preflight",
    "axes",
    "built",
    "run_parse",
    "postflight",
    "execution",
    "serialization",
]
Status: TypeAlias = Literal["ok", "refused", "error"]
DimensionDomain: TypeAlias = Literal["config_path", "model_field", "resource_field"]


@dataclass(frozen=True, slots=True)
class LayerIdentity:
    kind: Literal["base", "variant"]
    name: str | None


@dataclass(frozen=True, slots=True)
class Origin:
    kind: Literal["user", "variant", "preset", "rheplicant-default"]
    name: str | None = None

    def __post_init__(self) -> None:
        if self.kind in ("user", "rheplicant-default"):
            if self.name is not None:
                raise ValueError(f"{self.kind} origin cannot have a name")
        elif self.kind in ("variant", "preset"):
            if not isinstance(self.name, str) or not self.name:
                raise ValueError(f"{self.kind} origin requires a non-empty name")
        else:
            raise ValueError(f"unknown origin kind: {self.kind!r}")

    def render(self) -> str:
        if self.kind in ("user", "rheplicant-default"):
            return self.kind
        assert self.name is not None
        return f"{self.kind}:n-{self.name.encode('utf-8').hex()}"


class OriginLookup(Protocol):
    """Exact per-layer lookup; absence is never silently called ``user``."""

    def __call__(self, document_path: str, /) -> Origin | None: ...


@dataclass(frozen=True, slots=True)
class RunDescriptor:
    index: int
    name: str
    kind: str
    variant: str | None


@dataclass(frozen=True, slots=True)
class CompletedBoundary:
    stage: Stage
    layer: LayerIdentity | None


class TraceSink(Protocol):
    """Layer-aware, JAX-free append seam implemented by Task 14."""

    def boundary_completed(
        self, stage: Stage, layer: LayerIdentity | None = None
    ) -> None: ...

    def record_findings(
        self,
        stage: Stage,
        layer: LayerIdentity,
        findings: Sequence[Mapping[str, str]],
    ) -> None: ...

    def record_default(
        self, layer: LayerIdentity, path: str, value: JsonValue
    ) -> None: ...

    def record_delivery(
        self,
        layer: LayerIdentity,
        destination: DestinationDescriptor,
        *,
        dtype: str,
        origin: Origin,
        unit: str | None,
    ) -> None: ...

    def record_input(
        self, layer: LayerIdentity, row: Mapping[str, JsonValue]
    ) -> None: ...

    def record_parsed_run(
        self, layer: LayerIdentity, row: Mapping[str, JsonValue]
    ) -> None: ...

    def record_run_outcome(
        self, layer: LayerIdentity, row: Mapping[str, JsonValue]
    ) -> None: ...

    def freeze_layer(
        self, layer: LayerIdentity, row: Mapping[str, JsonValue]
    ) -> None: ...

    def completed_boundaries(self) -> Sequence[CompletedBoundary]: ...


@dataclass(frozen=True, slots=True)
class DestinationDescriptor:
    document_path: str
    domain: DimensionDomain
    selector: str

    def __post_init__(self) -> None:
        if not self.document_path:
            raise ValueError("document_path must be non-empty")
        if not self.selector:
            raise ValueError("selector must be non-empty")

    def child(
        self,
        segment: str | int,
        *,
        domain: DimensionDomain | None = None,
        selector: str | int | None = None,
    ) -> DestinationDescriptor:
        document_suffix = f"[{segment}]" if isinstance(segment, int) else f".{segment}"
        chosen = segment if selector is None else selector
        if isinstance(chosen, int) or chosen == "[]":
            selector_suffix = "[]"
        else:
            selector_suffix = f".{chosen}"
        return DestinationDescriptor(
            document_path=f"{self.document_path}{document_suffix}",
            domain=self.domain if domain is None else domain,
            selector=f"{self.selector}{selector_suffix}",
        )

    def nested(self, segment: str | int) -> DestinationDescriptor:
        document_suffix = f"[{segment}]" if isinstance(segment, int) else f".{segment}"
        return DestinationDescriptor(
            document_path=f"{self.document_path}{document_suffix}",
            domain=self.domain,
            selector=self.selector,
        )


OriginSegment: TypeAlias = str | int


@dataclass(frozen=True, slots=True)
class SourceInput:
    input_bytes: bytes
    source_path: str
    source_realpath: str | None
    source_name: str
    base_dir: str
    launch_mode: Literal["cli", "embedded"]


__all__ = [
    "CompletedBoundary",
    "DestinationDescriptor",
    "DimensionDomain",
    "JsonScalar",
    "JsonValue",
    "LayerIdentity",
    "Origin",
    "OriginLookup",
    "OriginSegment",
    "RunDescriptor",
    "SourceInput",
    "Stage",
    "Status",
    "TraceSink",
]
