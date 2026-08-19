"""The document load as one orchestration: boundaries, layers, and the record.

Until this module the load was a monolith and the runs loop interleaved parse
and execute per run: a typo in the LAST run's options cost every earlier
run's execution first (measured in Plan 3A's survey), and a builder's
``ConfigError`` carried nothing the completed passes had earned.  Here the
load becomes staged (spec §7/§8):

* the canonical base plus every declared variant is enumerated ONCE, and the
  text pre-flight fans over all of it even when the caller wants one layer --
  so an unselected variant's text fault still refuses, and no unselected
  variant is built;
* every selected layer completes axes and built first; then
  :func:`parse_declared_schedules_once` handler-parses EVERY declaration
  exactly once -- the base schedule against the layer each ``variant:`` names,
  a variant's own schedule against that variant's build -- and only then does
  each layer run post-flight and freeze;
* :func:`execute_prepared` runs the base schedule in declaration order,
  stopping at the first uncaptured failure, and keeps the prior rows.

Each of the four load boundaries accumulates the earlier ones' findings, and
the raise at a boundary attaches the CUMULATIVE report (spec §8); a
``ConfigError`` thrown later -- a file reader, a builder, a kind parser --
carries the completed boundaries' report.  Before the first completed pass
(structural, layering, source errors) ``report`` stays ``None``.

``expect: refuse`` keeps its legacy scope across the parse/execute split: the
capture wraps the parse, the pre-execute and the execute of one run.  A
base-schedule declaration whose handler parse raises ``Exception`` is recorded
as a tombstone :class:`ParsedRun` whose execution view is a
:class:`_CapturedParse`, and the execution stage turns it into the legacy
captured row; a parse refusal in a validation-only schedule is captured the
same way but never reaches an executor.  A run that SUCCEEDS despite
``expect: refuse`` is the legacy failure, in the legacy words.

This module imports the Task-1 ``TraceSink`` protocol only; Task 14 supplies
the concrete ``AuditTrace``.
"""

from __future__ import annotations

import dataclasses
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from _rheplicant_bootstrap.capture import (
    CapturedInput,
    CaptureService,
    captured_input_json,
)
from _rheplicant_bootstrap.errors import DirtError
from _rheplicant_bootstrap.layering import (
    DeletionRecord,
    OriginNode,
    initial_merge,
    origins_at,
)
from _rheplicant_bootstrap.path_syntax import PATH_STEP
from _rheplicant_bootstrap.types import (
    CompletedBoundary,
    LayerIdentity,
    Origin,
    OriginLookup,
    Status,
    TraceSink,
)
from _rheplicant_bootstrap.variants import (
    LayerAttributor,
    LayerRef,
    enumerate_layers_once,
)
from rheplicant.config.context import using_resolution_audit
from rheplicant.config.dimensions import (
    DimensionEnvironment,
    dimension_environment_for,
    using_dimension_environment,
    using_dimension_registry_snapshot,
)
from rheplicant.config.document import (
    ConfiguredRun,
    _attach,
    _build_with_axes,
    _through_built,
    _through_priced,
)
from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import Report
from rheplicant.config.layering import apply_variant
from rheplicant.config.passes import sweep
from rheplicant.config.preflight import _LABEL, _SECTIONS, CHECKS, _structural
from rheplicant.config.preflight.document import _variant_text
from rheplicant.config.sections import exits  # noqa: F401  -- fills the registry
from rheplicant.config.sections.exit_support import (
    ParsedOptions,
    ParsedRun,
    handler_for,
    parse_run,
)
from rheplicant.config.sections.runs import RunResult, parse_runs

__all__ = [
    "ExecutionRecord",
    "PreparedDocument",
    "PreparedLayer",
    "RunExecution",
    "base_parsed_schedule",
    "canonical_layers",
    "complete_all_postflight",
    "deletions_for",
    "execute_one_parsed",
    "execute_prepared",
    "origins_for",
    "parse_declared_schedules_once",
    "prepare_document",
    "prepare_layer_through_built",
    "run_text_preflight_all_layers",
    "select_build_layers",
    "validate_base_variant_targets",
]


@dataclass(frozen=True, slots=True)
class PreparedLayer:
    """One canonical layer with its configured build and parsed schedule."""

    layer: LayerRef
    configured: ConfiguredRun
    declared_runs: Sequence[ParsedRun]


@dataclass(frozen=True, slots=True)
class PreparedDocument:
    """Every built layer, and the base execution schedule within them."""

    layers: Sequence[PreparedLayer]
    execution_runs: Sequence[ParsedRun]


@dataclass(frozen=True, slots=True)
class RunExecution:
    """One executed run: its product row, or the terminal failure's."""

    index: int
    parsed: ParsedRun
    result: RunResult | None
    status: Literal["ok", "refused", "error"]
    wall_time_ns: int
    error: BaseException | None
    captured_expected_refusal: bool = False


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    """What a prepared document did when its base schedule executed."""

    prepared: PreparedDocument
    runs: Sequence[RunExecution]
    results: Mapping[str, RunResult]
    status: Status
    error: BaseException | None
    completed_boundaries: Sequence[CompletedBoundary]


@dataclass(frozen=True, slots=True)
class _Canonical:
    """The enumerated layers plus their origin/deletion evidence maps."""

    layers: tuple[LayerRef, ...]
    origins: Mapping[LayerIdentity, OriginNode]
    deletions: Mapping[LayerIdentity, Sequence[DeletionRecord]]


class _CapturedParse(Mapping):
    """The tombstone execution view of an ``expect: refuse`` run whose parse
    was captured: an EMPTY mapping (no options were normalized), carrying the
    captured error for the execution stage to record.

    The resolved view of the same tombstone stays empty too, so the audit
    projection never claims defaults that were never chosen.
    """

    __slots__ = ("error",)

    def __init__(self, error: BaseException):
        self.error = error

    def __getitem__(self, key):
        raise KeyError(key)

    def __len__(self):
        return 0

    def __iter__(self):
        return iter(())


def canonical_layers(
    document: Mapping[str, object],
    *,
    supplied: Sequence[LayerRef] | None = None,
    layer_origins: Mapping[LayerIdentity, OriginNode] | None = None,
    layer_deletions: Mapping[LayerIdentity, Sequence[DeletionRecord]]
    | None = None,
) -> _Canonical:
    """The base plus every declared variant, exactly once.

    The mapping route enumerates from ``document`` after the structural gate
    (whose refusals predate the first boundary and so carry ``report=None``).
    The integration route supplies the bootstrap's own layers with their
    matching evidence maps; supplied objects are used AS THEY ARE, so the
    audit trail's origin trees keep their identity.
    """
    if supplied is None:
        if layer_origins is not None or layer_deletions is not None:
            raise ConfigError(
                "layer_origins and layer_deletions accompany supplied "
                "canonical layers; layers is None.")
        if not isinstance(document, Mapping):
            raise ConfigError(
                f"A document is a mapping of sections; got "
                f"{type(document).__name__} ({document!r}).")
        _structural(document)
        merged = initial_merge(document, origin=Origin("user"))
        enumeration = enumerate_layers_once(
            merged.document, merged.origins, merged.deletions)
        return _Canonical(layers=tuple(enumeration.layers),
                          origins=enumeration.origins,
                          deletions=enumeration.deletions)
    if layer_origins is None or layer_deletions is None:
        raise ConfigError(
            "canonical layers supplied by an integration caller require "
            "their matching layer_origins and layer_deletions maps.")
    layers = tuple(supplied)
    identities = tuple(layer.identity for layer in layers)
    if (not layers or layers[0].kind != "base"
            or len(set(identities)) != len(identities)):
        raise ConfigError(
            "supplied canonical layers begin with exactly one base layer "
            "and carry unique identities.")
    if set(layer_origins) != set(identities) or set(
            layer_deletions) != set(identities):
        raise ConfigError(
            "supplied layer_origins/layer_deletions must cover every "
            "canonical layer identity exactly once.")
    return _Canonical(layers=layers, origins=layer_origins,
                      deletions=layer_deletions)


def validate_base_variant_targets(canonical: _Canonical) -> None:
    """Every base run's ``variant:`` names a declared layer, before any build.

    Measured at the hand this off: today's bad reference survives the text
    pre-flight and is refused only when the run's turn comes --
    ``load_document(variant=...)`` inside the loop, after the base was built.
    """
    names = {layer.name for layer in canonical.layers[1:]}
    declared = (f"this document declares {sorted(names)}" if names
                else "this document declares no variants")
    specs = parse_runs(canonical.layers[0].mutable_document().get("runs"))
    for spec in specs:
        if spec.variant is not None and spec.variant not in names:
            raise ConfigError(
                f"runs[{spec.name!r}]: variant: {spec.variant!r} names no "
                f"declared variant; {declared}.")


def run_text_preflight_all_layers(
    canonical: _Canonical,
    *,
    trace: TraceSink | None = None,
    environments: Mapping[LayerIdentity, DimensionEnvironment] | None = None,
) -> Mapping[LayerIdentity, Report]:
    """Run one isolated, pass-scoped text preflight over canonical layers."""
    with using_dimension_registry_snapshot():
        return _run_text_preflight_all_layers(
            canonical, trace=trace, environments=environments
        )


def _run_text_preflight_all_layers(
    canonical: _Canonical,
    *,
    trace: TraceSink | None = None,
    environments: Mapping[LayerIdentity, DimensionEnvironment] | None = None,
) -> Mapping[LayerIdentity, Report]:
    """The text-only pre-flight over every canonical layer, attributed.

    This is the ``preflight()`` walk with per-layer reports: one
    ``LayerAttributor`` over the canonical tuple, the variant-structural
    ``_variant_text`` conversion, and the combined report raised and warned
    at the end, exactly once.  The per-layer slices it returns are what each
    layer's later boundaries accumulate onto.

    The pass COMPLETES once every layer has been walked, including when the
    combined report then refuses: every layer's findings and boundary are
    appended to the trace before ``raise_if_refused``.
    """
    attributor = LayerAttributor()
    slices: dict[LayerIdentity, Report] = {}
    combined: list = []
    for layer in canonical.layers:
        layer_document = layer.mutable_document()
        failed_structure = False
        if layer.kind == "variant":
            try:
                _structural(layer_document)
            except ConfigError:
                found = tuple(_variant_text(layer_document))
                failed_structure = True
        if not failed_structure:
            environment = (
                dimension_environment_for(layer_document)
                if environments is None
                else environments[layer.identity]
            )
            with using_dimension_environment(environment):
                found = sweep(CHECKS, layer_document, label=_LABEL,
                              sections=_SECTIONS).findings
        slice_ = attributor.attribute(layer, found)
        slices[layer.identity] = Report(findings=slice_)
        combined.extend(slice_)
        if trace is not None:
            trace.record_findings(
                "preflight", layer.identity,
                tuple(dataclasses.asdict(row) for row in slice_))
            trace.boundary_completed("preflight", layer.identity)
    report = Report(findings=tuple(combined))
    report.raise_if_refused()
    report.emit_warnings()
    return slices


def select_build_layers(
    canonical: _Canonical,
    *,
    scope: str,
    variant: str | None,
) -> tuple[LayerRef, ...]:
    """Which layers the preparation builds: all of them, or the selected one.

    The unknown-name refusals are the compatibility sentences
    ``apply_variant`` has always raised for the same request.
    """
    if scope == "all_layers":
        return tuple(canonical.layers)
    if variant is None:
        return (canonical.layers[0],)
    for layer in canonical.layers[1:]:
        if layer.name == variant:
            return (layer,)
    names = [layer.name for layer in canonical.layers[1:]]
    if not names:
        raise ConfigError(
            f"variant {variant!r} was requested but this document declares "
            "no variants.")
    raise ConfigError(
        f"variant {variant!r} is not declared; this document declares "
        f"{sorted(names)}.")


def origins_for(
    layer: LayerRef,
    origins: Mapping[LayerIdentity, OriginNode],
) -> OriginNode:
    """The one origin tree the enumeration paired with this layer."""
    return origins[layer.identity]


def deletions_for(
    layer: LayerRef,
    deletions: Mapping[LayerIdentity, Sequence[DeletionRecord]],
) -> Sequence[DeletionRecord]:
    """The one deletion list the enumeration paired with this layer."""
    return deletions[layer.identity]


def _document_path_segments(document_path: str) -> tuple[str | int, ...]:
    segments: list[str | int] = []
    for piece in document_path.split("."):
        match = PATH_STEP.fullmatch(piece)
        if match is None:
            return ()
        name = match.group("name")
        index = match.group("index")
        if name is not None:
            segments.append(name)
        if index is not None:
            segments.append(int(index))
    return tuple(segments)


def _origin_lookup_for(
    document: Mapping[str, object], origins: OriginNode
) -> OriginLookup:
    """Bind concrete value paths to the authority of their payload form."""
    from rheplicant.config.values import VALUE_FORMS

    def lookup(document_path: str, /) -> Origin | None:
        segments = _document_path_segments(document_path)
        if not segments:
            return None
        value: object = document
        try:
            for segment in segments:
                value = value[segment]  # type: ignore[index]
        except (KeyError, IndexError, TypeError):
            return None
        authority = segments
        if isinstance(value, Mapping):
            forms = [key for key in value if key in VALUE_FORMS]
            if len(forms) == 1:
                authority = (*segments, forms[0])
        try:
            return origins_at(origins, authority)
        except ConfigError:
            return None

    return lookup


def prepare_layer_through_built(
    layer: LayerRef,
    *,
    previous: Report,
    base_dir: str | None,
    trace: TraceSink | None,
    dimensions: DimensionEnvironment | None = None,
    origins: OriginNode | None = None,
    capture: CaptureService | None = None,
) -> PreparedLayer:
    """One layer through the axes and built boundaries -> its configured run.

    The layer's thawed document is what every builder reads (the layered,
    variant-applied mapping, with the process-entry sections removed);
    ``previous`` is the layer's own text pre-flight slice, so the carried
    report accumulates in pass order.
    """
    environment = dimensions or dimension_environment_for(layer.mutable_document())
    origin_lookup = (
        None
        if origins is None
        else _origin_lookup_for(layer.mutable_document(), origins)
    )
    with using_dimension_environment(environment), using_resolution_audit(
        layer.identity, trace, origin_lookup, capture
    ):
        configured = _build_with_axes(layer.mutable_document(), base_dir=base_dir,
                                      previous=previous, layer=layer.identity,
                                      trace=trace)
        configured = _through_built(configured, layer=layer.identity, trace=trace)
    return PreparedLayer(layer=layer, configured=configured,
                         declared_runs=())


def _parse_target(
    spec,
    declaring: PreparedLayer,
    built_by_identity: Mapping[LayerIdentity, PreparedLayer],
    *,
    scope: str,
) -> PreparedLayer | None:
    """The build a declaration parses against; ``None`` to stand down.

    All-layers scope, base schedule: a declaration parses against the layer
    its ``variant:`` names (the base when absent).  A variant layer's
    validation-only schedule parses its entries against that variant's OWN
    build -- the schedule answers what ``load_document(variant=...)``
    accepts -- EXCEPT an entry whose ``variant:`` names a DIFFERENT layer:
    that entry belongs to the named layer's validation, and parsing it here
    would judge it against a build it never targets (measured: an inherited
    ``variant: with_inf`` run refused on the parameterless ``unity_gain``
    layer).  Selected scope: the one built layer parses every entry of its
    own effective schedule that targets it -- ``variant:`` absent, or naming
    the selected layer itself -- and stands down on the rest: the named
    layer was never built here, so parsing against the WRONG build would
    refuse documents that are valid when it is the one being built.
    """
    if scope == "selected":
        if declaring.layer.kind == "base":
            return declaring if spec.variant is None else None
        if spec.variant in (None, declaring.layer.name):
            return declaring
        return None
    if declaring.layer.kind == "base":
        if spec.variant is None:
            return declaring
        return built_by_identity[LayerIdentity("variant", spec.variant)]
    if spec.variant in (None, declaring.layer.name):
        return declaring
    return None


def _empty_views():
    return ParsedOptions(execution=MappingProxyType({}),
                         resolved=MappingProxyType({}))


def parse_declared_schedules_once(
    built_layers: Sequence[PreparedLayer],
    *,
    scope: str,
    trace: TraceSink | None = None,
) -> tuple[PreparedLayer, ...]:
    """Handler-parse every declared run exactly once, layer by layer.

    The base schedule first, then each variant's validation-only schedule;
    every declaration is parsed against the build it names (see
    :func:`_parse_target`).  An ``expect: refuse`` declaration whose parse
    raises ``Exception`` is captured into a tombstone -- the legacy capture
    scope, moved ahead of the split.  Any other parse refusal refuses the
    document with every built layer's earned report attached.
    """
    built_by_identity = {built.layer.identity: built for built in built_layers}
    cumulative = Report(findings=tuple(
        finding for built in built_layers
        for finding in built.configured.report.findings))
    parsed_layers = []
    for built in built_layers:
        layer = built.layer
        specs = parse_runs(layer.mutable_document().get("runs"))
        parsed = []
        for index, spec in enumerate(specs):
            target = _parse_target(spec, built, built_by_identity, scope=scope)
            if target is None:
                parsed.append(ParsedRun(index=index, layer=layer, spec=spec,
                                        parsed=_empty_views()))
                continue
            if spec.expect == "refuse":
                try:
                    parsed.append(parse_run(spec, target.configured,
                                            index=index, layer=target.layer,
                                            trace=trace))
                except Exception as error:  # noqa: BLE001 -- the legacy
                    # capture wraps the whole parse/pre-execute/execute
                    # triple; parse is the part that moved stages.
                    parsed.append(ParsedRun(
                        index=index, layer=target.layer, spec=spec,
                        parsed=ParsedOptions(execution=_CapturedParse(error),
                                             resolved=MappingProxyType({}))))
                continue
            # The uncaptured route: a kind-parser refusal carries every built
            # layer's earned report (spec §8); _attach adds it in place, so
            # the parser's own args, cause and traceback survive.
            parsed.append(_attach(cumulative, parse_run, spec,
                                  target.configured, index=index,
                                  layer=target.layer, trace=trace))
        if trace is not None:
            trace.boundary_completed("run_parse", layer.identity)
        parsed_layers.append(PreparedLayer(
            layer=layer, configured=built.configured,
            declared_runs=tuple(parsed)))
    return tuple(parsed_layers)


def base_parsed_schedule(
    parsed_layers: Sequence[PreparedLayer],
) -> Sequence[ParsedRun]:
    """The base layer's parsed declarations, projected by IDENTITY.

    Never ``parse_run`` again, never copy a ``ParsedRun``, never normalize
    options a second time: the schedule that executes is the one that was
    parsed.
    """
    declared = parsed_layers[0].declared_runs
    return declared if type(declared) is tuple else tuple(declared)


def _descriptor(parsed: ParsedRun) -> dict:
    """The closed ``{index, name, kind, variant}`` projection of one run."""
    return {"index": parsed.index, "name": parsed.name, "kind": parsed.kind,
            "variant": parsed.variant}


def complete_all_postflight(
    parsed_layers: Sequence[PreparedLayer],
    *,
    execution_runs: Sequence[ParsedRun],
    trace: TraceSink | None,
    origins: Mapping[LayerIdentity, OriginNode],
    deletions: Mapping[LayerIdentity, Sequence[DeletionRecord]],
) -> tuple[PreparedLayer, ...]:
    """Each layer's post-flight, then its freeze -- in canonical order.

    Every handler parse has already completed globally, so each layer is
    frozen immediately after its OWN successful post-flight and the freeze
    never reapplies a variant patch.  A post-flight refusal leaves the
    failing layer and every later layer unfrozen while preserving the
    earlier, truthfully completed ones.
    """
    completed = []
    for built in parsed_layers:
        identity = built.layer.identity
        configured = _through_priced(built.configured, layer=identity,
                                     trace=trace)
        if trace is not None:
            trace.freeze_layer(identity, {
                "origins": origins_for(built.layer, origins),
                "deletions": tuple(deletions_for(built.layer, deletions)),
                "declared_runs": tuple(
                    _descriptor(parsed) for parsed in built.declared_runs),
                "execution_runs": tuple(
                    _descriptor(parsed) for parsed in execution_runs),
            })
        completed.append(PreparedLayer(layer=built.layer,
                                       configured=configured,
                                       declared_runs=built.declared_runs))
    return tuple(completed)


def _prepare_document_with_capture(
    document: Mapping[str, object],
    *,
    scope: Literal["selected", "all_layers"],
    variant: str | None = None,
    base_dir: str | None = None,
    layers: Sequence[LayerRef] | None = None,
    layer_origins: Mapping[LayerIdentity, OriginNode] | None = None,
    layer_deletions: Mapping[LayerIdentity, Sequence[DeletionRecord]]
    | None = None,
    trace: TraceSink | None = None,
    capture: CaptureService,
) -> PreparedDocument:
    """Prepare a document through every validation boundary -> the layers.

    ``scope="selected"`` is the mapping API's route: a requested ``variant``
    is applied to the raw mapping FIRST -- the pre-4A ``_assemble`` order --
    so the text pre-flight fans over the merged document and the selected
    layer's findings are the document's own, unprefixed.  An unselected
    variant's text fault still refuses; no unselected variant is built; the
    selected layer's effective schedule is handler-parsed once against its
    build for validation; and ``execution_runs`` is empty.
    ``scope="all_layers"`` enumerates the raw document once, builds every
    declared layer, and projects the base schedule as the execution
    schedule.
    """
    if scope not in ("selected", "all_layers"):
        raise ValueError(
            f"scope is 'selected' or 'all_layers'; got {scope!r}.")
    if scope == "selected" and layers is None and variant is not None:
        if not isinstance(document, Mapping):
            raise ConfigError(
                f"A document is a mapping of sections; got "
                f"{type(document).__name__} ({document!r}).")
        document = apply_variant(dict(document), variant)
        variant = None
    canonical = canonical_layers(document, supplied=layers,
                                 layer_origins=layer_origins,
                                 layer_deletions=layer_deletions)
    validate_base_variant_targets(canonical)
    environments = {
        layer.identity: dimension_environment_for(layer.mutable_document())
        for layer in canonical.layers
    }
    preflight_reports = run_text_preflight_all_layers(
        canonical, trace=trace, environments=environments
    )
    selected = select_build_layers(canonical, scope=scope, variant=variant)
    built_layers = tuple(
        prepare_layer_through_built(
            layer,
            previous=preflight_reports[layer.identity],
            base_dir=base_dir,
            trace=trace,
            dimensions=environments[layer.identity],
            origins=origins_for(layer, canonical.origins),
            capture=capture,
        )
        for layer in selected
    )
    parsed_layers = parse_declared_schedules_once(built_layers, scope=scope,
                                                  trace=trace)
    execution_runs = (
        base_parsed_schedule(parsed_layers) if scope == "all_layers" else ()
    )
    completed_layers = complete_all_postflight(
        parsed_layers, execution_runs=execution_runs, trace=trace,
        origins=canonical.origins, deletions=canonical.deletions)
    return PreparedDocument(completed_layers, execution_runs)


def prepare_document(
    document: Mapping[str, object],
    *,
    scope: Literal["selected", "all_layers"],
    variant: str | None = None,
    base_dir: str | None = None,
    layers: Sequence[LayerRef] | None = None,
    layer_origins: Mapping[LayerIdentity, OriginNode] | None = None,
    layer_deletions: Mapping[LayerIdentity, Sequence[DeletionRecord]]
    | None = None,
    trace: TraceSink | None = None,
    capture: CaptureService | None = None,
) -> PreparedDocument:
    """Prepare through every boundary with one invocation-owned capture root."""

    if capture is not None:
        return _prepare_document_with_capture(
            document,
            scope=scope,
            variant=variant,
            base_dir=base_dir,
            layers=layers,
            layer_origins=layer_origins,
            layer_deletions=layer_deletions,
            trace=trace,
            capture=capture,
        )

    def on_verified(layer: LayerIdentity, row: CapturedInput) -> None:
        if trace is not None:
            trace.record_input(layer, captured_input_json(row))

    root = tempfile.mkdtemp(prefix="rheplicant-capture-")
    service = CaptureService(root, on_verified=on_verified)
    try:
        return _prepare_document_with_capture(
            document,
            scope=scope,
            variant=variant,
            base_dir=base_dir,
            layers=layers,
            layer_origins=layer_origins,
            layer_deletions=layer_deletions,
            trace=trace,
            capture=service,
        )
    finally:
        service.close()


def execute_one_parsed(
    parsed: ParsedRun,
    *,
    prepared: PreparedDocument,
    prior: Mapping[str, RunResult],
    clock_ns: Callable[[], int],
    trace: TraceSink | None = None,
) -> RunExecution:
    """One parsed declaration through its handler -> one timestamped row.

    The configured build is the ONE prepared layer whose identity is
    ``parsed.layer.identity`` -- an absent, duplicated or inconsistent target
    is a wiring error, and a variant-targeted run is never defaulted back to
    the base build.  The classification is the legacy one: a captured
    ``expect: refuse`` is a successful row with the error on the result; an
    uncaptured ``ConfigError`` refuses; any other ``Exception`` is an error.
    """
    label = parsed.layer.prefix or "base"
    matches = [layer for layer in prepared.layers
               if layer.layer.identity == parsed.layer.identity]
    if not matches:
        raise ConfigError(
            f"runs[{parsed.name!r}]: the run was parsed against the {label} "
            "layer, which is not among this prepared document's layers; the "
            "two come from different enumerations.")
    if len(matches) > 1:
        raise ConfigError(
            f"runs[{parsed.name!r}]: the {label} layer appears twice in "
            "this prepared document.")
    (prepared_layer,) = matches
    if prepared_layer.layer is not parsed.layer:
        raise ConfigError(
            f"runs[{parsed.name!r}]: the prepared {label} layer is not the "
            "one the run was parsed against; the two come from different "
            "enumerations.")
    configured = prepared_layer.configured
    result = None
    error = None
    status: Status = "ok"
    captured_expected = False
    started = clock_ns()
    tombstone = parsed.parsed.execution
    if isinstance(tombstone, _CapturedParse):
        captured_error = tombstone.error
        result = RunResult(name=parsed.name, kind=parsed.kind, product=None,
                           error=captured_error, variant=parsed.variant)
        captured_expected = True
    elif parsed.expect == "refuse":
        handler = handler_for(parsed.kind)
        try:
            handler.pre_execute(parsed, configured, prior)
            product = handler.execute(parsed, configured, prior)
        except Exception as captured_error:  # noqa: BLE001 -- run-and-capture
            # is the point
            result = RunResult(name=parsed.name, kind=parsed.kind,
                               product=None, error=captured_error,
                               variant=parsed.variant)
            captured_expected = True
        else:
            error = ConfigError(
                f"runs[{parsed.name!r}]: expect: refuse, and kind: "
                f"{parsed.kind} SUCCEEDED -- the assertion this run makes "
                "about the design no longer holds.")
            status = "refused"
    else:
        handler = handler_for(parsed.kind)
        try:
            handler.pre_execute(parsed, configured, prior)
            product = handler.execute(parsed, configured, prior)
        except ConfigError as caught:
            error = caught
            status = "refused"
        except Exception as caught:
            error = caught
            status = "error"
        else:
            result = RunResult(name=parsed.name, kind=parsed.kind,
                               product=product, error=None,
                               variant=parsed.variant)
    wall = clock_ns() - started
    row = RunExecution(index=parsed.index, parsed=parsed, result=result,
                       status=status, wall_time_ns=wall, error=error,
                       captured_expected_refusal=captured_expected)
    if trace is not None:
        thrown = error
        if captured_expected:
            thrown = result.error
        trace.record_run_outcome(parsed.layer.identity, {
            "descriptor": _descriptor(parsed),
            "status": "expected_refusal" if captured_expected else status,
            "wall_time_ns": wall,
            "exception_type": (None if thrown is None else
                               f"{type(thrown).__module__}."
                               f"{type(thrown).__qualname__}"),
            "exception_message": None if thrown is None else str(thrown),
            "capture_scope": ("arbitrary_exception" if captured_expected
                              else None),
            "is_dirt_error": (None if thrown is None else
                              isinstance(thrown, DirtError)),
        })
    return row


def execute_prepared(
    prepared: PreparedDocument,
    *,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    trace: TraceSink | None = None,
) -> ExecutionRecord:
    """Execute the base schedule in declaration order, stopping at the first
    uncaptured failure.

    The accumulation an executor reads is a read-only proxy over a COPY:
    ``reuse:`` may only look backwards, and the copy is what keeps that
    promise for a view an executor retains (``runs.py``'s documented
    contract, carried over unchanged).
    """
    rows: list[RunExecution] = []
    results: dict[str, RunResult] = {}
    terminal_error: BaseException | None = None
    terminal_status: Status = "ok"
    for parsed in prepared.execution_runs:
        row = execute_one_parsed(parsed, prepared=prepared,
                                 prior=MappingProxyType(dict(results)),
                                 clock_ns=clock_ns, trace=trace)
        rows.append(row)
        if row.result is not None:
            results[row.parsed.name] = row.result
        if row.status != "ok":
            terminal_status = row.status
            terminal_error = row.error
            break
    return ExecutionRecord(
        prepared=prepared, runs=tuple(rows),
        results=MappingProxyType(dict(results)), status=terminal_status,
        error=terminal_error,
        completed_boundaries=(
            () if trace is None else tuple(trace.completed_boundaries())),
    )
