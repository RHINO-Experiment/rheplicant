"""prepare_document / execute_prepared: boundaries, layers, and the record.

Plan 4A Task 10.  The document load becomes one orchestration with explicit
pass boundaries: the canonical base plus every declared variant, the text
pre-flight fanned over all of them, axes/build/built per selected layer,
every declared schedule handler-parsed exactly once before any post-flight,
and the base schedule executed only after all of that.  Each boundary
accumulates the earlier ones' findings onto the raised ``ConfigError``
(spec §8), and ``run_document``/``load_document`` keep their exact public
contracts over the new machinery.

Every test that registers a probe id registers it into an EMPTIED registry
that is restored afterwards; the ids (``C11``/``B3``/``A9``/``C97``) are the
ones the sibling pass tests measured free of every real registry.
"""

from __future__ import annotations

import types
import warnings

import jax.numpy as jnp
import pytest

import rheplicant.config.document as document_module
from _rheplicant_bootstrap.layering import initial_merge
from _rheplicant_bootstrap.types import CompletedBoundary, LayerIdentity, Origin
from _rheplicant_bootstrap.variants import enumerate_layers_once
from rheplicant.config.document import ConfiguredRun, load_document
from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import ConfigWarning, Report, refuse, warn
from rheplicant.config.inflight import (
    AXIS_CHECKS,
    BUILT_CHECKS,
    register_axes,
    register_built,
)
from rheplicant.config.orchestration import (
    ExecutionRecord,
    PreparedDocument,
    PreparedLayer,
    canonical_layers,
    execute_prepared,
    prepare_document,
    run_text_preflight_all_layers,
    select_build_layers,
)
from rheplicant.config.postflight import CHECKS as PRICED_CHECKS
from rheplicant.config.postflight import register as register_priced
from rheplicant.config.preflight import CHECKS as TEXT_CHECKS
from rheplicant.config.preflight import preflight
from rheplicant.config.preflight import register as register_text
from rheplicant.config.sections import exits  # noqa: F401  -- fills PARSERS
from rheplicant.config.sections.exit_support import (
    DEFERRED_CHECKS,
    EXECUTORS,
    PARSERS,
    PRE_EXECUTORS,
)
from rheplicant.config.sections.runs import run_document
from tests.config.exit_helpers import conjugate_document
from tests.config.preflight_helpers import preflight_document
from tests.config.test_config_document import synthetic_document

_BASE = LayerIdentity("base", None)
_LATE = LayerIdentity("variant", "late")


def _document(**patch):
    """The shared document, with ``variants:`` stripped unless patched in."""
    if "variants" not in patch:
        patch["variants"] = None
    return preflight_document(**patch)


def _enumerate(document):
    """The canonical enumeration of a mapping, the mapping API's own route."""
    merged = initial_merge(document, origin=Origin("user"))
    return enumerate_layers_once(merged.document, merged.origins,
                                 merged.deletions)


class _Trace:
    """The Task-1 TraceSink protocol as a recording test double.

    Every method appends to its own list AND to the shared ``events`` log,
    so a test can assert cross-method order (a layer is frozen after ITS
    post-flight, before the next layer's).  ``snapshot()`` returns a fresh
    detached view each call, mirroring the audit contract's duck.
    """

    def __init__(self):
        self.events = []
        self.findings = []
        self.boundaries = []
        self.parsed_runs = []
        self.outcomes = []
        self.frozen = []

    def record_findings(self, stage, layer, findings):
        self.events.append(("findings", stage, layer))
        self.findings.append((stage, layer, tuple(findings)))

    def boundary_completed(self, stage, layer=None):
        self.events.append(("boundary", stage, layer))
        self.boundaries.append(CompletedBoundary(stage, layer))

    def record_parsed_run(self, layer, row):
        self.events.append(("parsed_run", layer))
        self.parsed_runs.append((layer, row))

    def record_run_outcome(self, layer, row):
        self.events.append(("outcome", layer))
        self.outcomes.append((layer, row))

    def freeze_layer(self, layer, row):
        self.events.append(("freeze", layer))
        self.frozen.append((layer, row))

    def completed_boundaries(self):
        return tuple(self.boundaries)

    def snapshot(self):
        return types.SimpleNamespace(
            completed_boundaries=self.completed_boundaries())


@pytest.fixture
def all_registries():
    """All four check registries emptied for one test and restored after."""
    registries = (TEXT_CHECKS, AXIS_CHECKS, BUILT_CHECKS, PRICED_CHECKS)
    saved = [dict(registry) for registry in registries]
    for registry in registries:
        registry.clear()
    try:
        yield
    finally:
        for registry, was in zip(registries, saved, strict=True):
            registry.clear()
            registry.update(was)


@pytest.fixture
def handler_tables():
    """The four live handler tables, snapshotted and restored."""
    tables = (PARSERS, PRE_EXECUTORS, EXECUTORS, DEFERRED_CHECKS)
    saved = [dict(table) for table in tables]
    try:
        yield
    finally:
        for table, was in zip(tables, saved, strict=True):
            table.clear()
            table.update(was)


def _parse_spy(events, seen, real):
    """Record one ("parse", run-name) event and the configured build seen."""
    def spy(options, context):
        events.append(("parse", context.spec.name))
        seen[context.spec.name] = context.configured_run
        return real(options, context)

    return spy


def _config_warning_texts(recorded):
    return [str(one.message) for one in recorded
            if issubclass(one.category, ConfigWarning)]


class TestTheFourBoundariesCarryTheCumulativeReport:
    """Spec §8: each load boundary's ``ConfigError`` carries every completed
    pass's findings; before the first completed pass it carries none."""

    def test_a_structural_error_before_the_first_boundary_carries_no_report(
            self):
        with pytest.raises(ConfigError) as caught:
            prepare_document({**_document(), "observations": {}},
                             scope="all_layers")
        assert caught.value.report is None

    def test_a_layering_error_before_the_first_boundary_carries_no_report(
            self):
        document = _document()
        document["variants"] = 0
        with pytest.raises(ConfigError, match="variants") as caught:
            prepare_document(document, scope="all_layers", trace=_Trace())
        assert caught.value.report is None

    def test_a_preflight_refusal_carries_the_preflight_only(
            self, all_registries):
        note = warn("C11", "runtime.seed", "text note.")
        refusal = refuse("C12", "model.gain", "text refusal.")
        register_text("C11")(lambda document: (note,))
        register_text("C12")(lambda document: (refusal,))
        trace = _Trace()
        with warnings.catch_warnings(record=True) as heard:
            warnings.simplefilter("always")
            with pytest.raises(ConfigError) as caught:
                prepare_document(_document(), scope="all_layers", trace=trace)
        # The first REFUSAL verbatim -- the warning collected ahead of it
        # must not displace it.
        assert str(caught.value) == "text refusal."
        assert caught.value.report.findings == (note, refusal)
        # The pass completed, so its boundary and findings ARE appended even
        # though the boundary then refused.
        assert trace.completed_boundaries() == (
            CompletedBoundary("preflight", _BASE),)
        # ...but a refusing boundary emits no warnings.
        assert _config_warning_texts(heard) == []

    def test_an_axes_refusal_carries_preflight_and_axes(self, all_registries):
        note = warn("C11", "runtime.seed", "text note.")
        refusal = refuse("B3", "observation.time", "axes refusal.")
        register_text("C11")(lambda document: (note,))
        register_axes("B3")(lambda facts: (refusal,))
        trace = _Trace()
        with warnings.catch_warnings(record=True) as heard:
            warnings.simplefilter("always")
            with pytest.raises(ConfigError) as caught:
                prepare_document(_document(), scope="all_layers", trace=trace)
        assert str(caught.value) == "axes refusal."
        assert caught.value.report.findings == (note, refusal)
        assert trace.completed_boundaries() == (
            CompletedBoundary("preflight", _BASE),
            CompletedBoundary("axes", _BASE))
        assert _config_warning_texts(heard) == ["text note."]

    def test_a_built_refusal_carries_three_passes(self, all_registries):
        findings = (warn("C11", "runtime.seed", "text note."),
                    warn("B3", "observation.time", "axes note."),
                    refuse("A9", "model.gain", "built refusal."))
        register_text("C11")(lambda document: (findings[0],))
        register_axes("B3")(lambda facts: (findings[1],))
        register_built("A9")(lambda run: (findings[2],))
        trace = _Trace()
        with warnings.catch_warnings(record=True) as heard:
            warnings.simplefilter("always")
            with pytest.raises(ConfigError) as caught:
                prepare_document(_document(), scope="all_layers", trace=trace)
        assert str(caught.value) == "built refusal."
        assert caught.value.report.findings == findings
        assert trace.completed_boundaries() == (
            CompletedBoundary("preflight", _BASE),
            CompletedBoundary("axes", _BASE),
            CompletedBoundary("built", _BASE))
        assert _config_warning_texts(heard) == ["text note.", "axes note."]

    def test_a_postflight_refusal_carries_all_four_and_is_itself_recorded(
            self, all_registries):
        findings = (warn("C11", "runtime.seed", "text note."),
                    warn("B3", "observation.time", "axes note."),
                    warn("A9", "model.gain", "built note."),
                    refuse("C97", "inference.parameters", "priced refusal."))
        register_text("C11")(lambda document: (findings[0],))
        register_axes("B3")(lambda facts: (findings[1],))
        register_built("A9")(lambda run: (findings[2],))
        register_priced("C97")(lambda payload: (findings[3],))
        trace = _Trace()
        with warnings.catch_warnings(record=True) as heard:
            warnings.simplefilter("always")
            with pytest.raises(ConfigError) as caught:
                prepare_document(_document(), scope="all_layers", trace=trace)
        assert str(caught.value) == "priced refusal."
        assert caught.value.report.findings == findings
        # The pass is complete once it RETURNS its Report, including when the
        # report then refuses -- so the post-flight boundary is recorded
        # before the raise, unlike a builder that throws.
        assert trace.completed_boundaries() == (
            CompletedBoundary("preflight", _BASE),
            CompletedBoundary("axes", _BASE),
            CompletedBoundary("built", _BASE),
            CompletedBoundary("run_parse", _BASE),
            CompletedBoundary("postflight", _BASE))
        assert _config_warning_texts(heard) == [
            "text note.", "axes note.", "built note."]

    def test_a_builder_error_after_a_completed_boundary_carries_the_cumulative(
            self, all_registries, monkeypatch):
        findings = (warn("C11", "runtime.seed", "text note."),
                    warn("B3", "observation.time", "axes note."))
        register_text("C11")(lambda document: (findings[0],))
        register_axes("B3")(lambda facts: (findings[1],))

        def explode(*args, **kwargs):
            raise ConfigError("the beam is gone.")

        monkeypatch.setattr(document_module, "build_resources", explode)
        trace = _Trace()
        with pytest.raises(ConfigError) as caught:
            prepare_document(_document(), scope="all_layers", trace=trace)
        # Attached in place: args, type and identity are the builder's own.
        assert str(caught.value) == "the beam is gone."
        assert caught.value.args == ("the beam is gone.",)
        assert caught.value.report is not None
        assert caught.value.report.findings == findings
        # The built boundary never ran, so it is absent -- unlike a pass
        # that returned a refusing Report.
        assert trace.completed_boundaries() == (
            CompletedBoundary("preflight", _BASE),
            CompletedBoundary("axes", _BASE))

    def test_a_parser_error_after_the_built_boundary_carries_three_passes(
            self, all_registries, handler_tables):
        findings = (warn("C11", "runtime.seed", "text note."),
                    warn("B3", "observation.time", "axes note."),
                    warn("A9", "model.gain", "built note."))
        register_text("C11")(lambda document: (findings[0],))
        register_axes("B3")(lambda facts: (findings[1],))
        register_built("A9")(lambda run: (findings[2],))

        def explode(options, context):
            raise ConfigError("parse exploded.")

        PARSERS["forward"] = explode
        trace = _Trace()
        with pytest.raises(ConfigError) as caught:
            prepare_document(_document(), scope="all_layers", trace=trace)
        assert str(caught.value) == "parse exploded."
        assert caught.value.report is not None
        assert caught.value.report.findings == findings
        assert trace.completed_boundaries() == (
            CompletedBoundary("preflight", _BASE),
            CompletedBoundary("axes", _BASE),
            CompletedBoundary("built", _BASE))

    def test_an_already_attached_report_is_never_overwritten(
            self, all_registries, handler_tables):
        """The other arm of the attach: a refusal that already carries a
        report keeps it -- the cumulative one is not pasted over it."""
        findings = (warn("C11", "runtime.seed", "text note."),)
        register_text("C11")(lambda document: (findings[0],))
        own = Report(findings=(refuse("C97", "runs[0]", "its own."),))

        def explode(options, context):
            raise ConfigError("carries its own.", report=own)

        PARSERS["forward"] = explode
        with pytest.raises(ConfigError) as caught:
            prepare_document(_document(), scope="all_layers")
        assert caught.value.report is own

    def test_a_pass_that_throws_before_returning_completes_no_boundary(
            self, all_registries, monkeypatch):
        note = warn("C11", "runtime.seed", "text note.")
        register_text("C11")(lambda document: (note,))

        def explode(facts):
            raise ConfigError("axes exploded.")

        monkeypatch.setattr(document_module, "axes", explode)
        trace = _Trace()
        with pytest.raises(ConfigError) as caught:
            prepare_document(_document(), scope="all_layers", trace=trace)
        assert str(caught.value) == "axes exploded."
        assert caught.value.report is not None
        assert caught.value.report.findings == (note,)
        assert trace.completed_boundaries() == (
            CompletedBoundary("preflight", _BASE),)


class TestTheOrchestratedTextPreflightIsThePassItself:
    """The orchestration walks the layers itself (the pass's own driver is
    not layer-reporting), so the two are pinned equal on a battery."""

    def _combined(self, document):
        canonical = canonical_layers(document)
        slices = run_text_preflight_all_layers(canonical)
        return tuple(finding for layer in canonical.layers
                     for finding in slices[layer.identity].findings)

    def test_clean_and_warning_documents_agree(self):
        assert self._combined(_document()) == preflight(_document()).findings
        warned = conjugate_document({"kind": "forward"})
        assert self._combined(warned) == preflight(warned).findings
        assert self._combined(warned)  # the battery is not vacuous

    def test_refusing_documents_agree_on_the_attached_report(self):
        documents = [
            preflight_document(inference={
                "twin": {"without": ["noise"]},
                "parameters": {"d": {"init": 0.5, "into": [
                    "global_signal.depth", "gain.gain"]}}}),
            preflight_document(variants={"bad": {"campaign": {}}}),
        ]
        for document in documents:
            with pytest.raises(ConfigError) as via_pass:
                preflight(document).raise_if_refused()
            with pytest.raises(ConfigError) as via_orchestration:
                run_text_preflight_all_layers(canonical_layers(document))
            assert (via_orchestration.value.report.findings
                    == via_pass.value.report.findings)


class TestTheGlobalOrder:
    """Plan Step 2: every build before every parse, every parse before every
    post-flight, and execution strictly after all of it."""

    def test_all_run_parsing_finishes_before_the_first_executor(
            self, handler_tables):
        parse_events = []
        seen = {}
        for kind in list(PARSERS):
            PARSERS[kind] = _parse_spy(parse_events, seen, PARSERS[kind])
        execute_events = []
        for kind in list(EXECUTORS):
            def execute_spy(parsed, configured, previous, _events=execute_events):
                _events.append(("execute", parsed.index))
                return configured.state

            EXECUTORS[kind] = execute_spy
        document = _document(runs=[
            {"name": "fwd", "kind": "forward"},
            {"name": "cov", "kind": "fisher"},
            {"name": "fit", "kind": "optimize", "optimizer": "gradient",
             "learning_rate": 0.01, "n_steps": 2},
            {"name": "est", "kind": "plan.estimate",
             "blocks": [{"names": ["g"]}]},
        ])
        record = execute_prepared(prepare_document(document,
                                                   scope="all_layers"))
        # Every parse lands in prepare (before any execute could start);
        # the executes then run in declaration order.
        assert [name for _, name in parse_events] == [
            "fwd", "cov", "fit", "est"]
        assert [index for _, index in execute_events] == [0, 1, 2, 3]
        assert record.status == "ok"
        assert set(record.results) == {"fwd", "cov", "fit", "est"}

    def test_an_invalid_later_run_prevents_the_first_executor(
            self, handler_tables):
        """Before this task the later run's refusal arrived only after the
        earlier run had EXECUTED (measured in 3A's survey)."""
        calls = []
        real = EXECUTORS["forward"]

        def spy(parsed, configured, previous):
            calls.append(parsed.name)
            return real(parsed, configured, previous)

        EXECUTORS["forward"] = spy
        # `fisher` on a document with no inference.parameters is silent at
        # text level (measured) and refuses at handler parse.
        document = synthetic_document()
        document["runs"] = [{"name": "ok", "kind": "forward"},
                            {"name": "bad", "kind": "fisher"}]
        with pytest.raises(ConfigError, match="fits latents"):
            run_document(document)
        assert calls == []

    def test_a_postflight_refusal_prevents_any_execution(
            self, all_registries, handler_tables):
        """Every post-flight finishes before execution begins: a priced
        refusal means no executor ever runs."""
        calls = []
        real = EXECUTORS["forward"]

        def spy(parsed, configured, previous):
            calls.append(parsed.name)
            return real(parsed, configured, previous)

        EXECUTORS["forward"] = spy
        register_priced("C97")(lambda payload: (
            refuse("C97", "inference.parameters", "priced refusal."),))
        with pytest.raises(ConfigError, match="priced refusal"):
            run_document(_document())
        assert calls == []

    def test_variant_runs_are_validated_but_never_replace_the_base_schedule(
            self, handler_tables):
        parse_events = []
        seen = {}
        for kind in list(PARSERS):
            PARSERS[kind] = _parse_spy(parse_events, seen, PARSERS[kind])
        document = conjugate_document({"name": "fwd", "kind": "forward"})
        document["variants"] = {"v": {"runs": [{"name": "alt",
                                                "kind": "fisher"}]}}
        prepared = prepare_document(document, scope="all_layers")
        assert [(parsed.index, parsed.name, parsed.kind, parsed.variant)
                for parsed in prepared.execution_runs] == [
            (0, "fwd", "forward", None)]
        (variant_layer,) = [layer for layer in prepared.layers
                            if layer.layer.name == "v"]
        assert [parsed.name for parsed in variant_layer.declared_runs
                ] == ["alt"]
        assert ("parse", "alt") in parse_events
        # ...and the variant's schedule is validated against ITS OWN build:
        # the same twin (the parse-time configured is the pre-post-flight
        # object; the layer's is its `_replace`d, report-carrying successor).
        assert seen["alt"].twin is variant_layer.configured.twin

    def test_every_declaration_is_parsed_once_before_any_postflight(
            self, handler_tables):
        parse_events = []
        seen = {}
        for kind in list(PARSERS):
            PARSERS[kind] = _parse_spy(parse_events, seen, PARSERS[kind])
        trace = _Trace()
        document = conjugate_document(
            {"name": "b0", "kind": "forward"},
            {"name": "b1", "kind": "fisher", "variant": "late"})
        document["variants"] = {
            "late": {"model": {"gain": {"gain": {"value": 2.0,
                                                 "unit": "dimensionless"}}},
                     "runs": [{"name": "v0", "kind": "forward"}]}}
        prepared = prepare_document(document, scope="all_layers",
                                    trace=trace)
        # The global stage order: all-layer text pre-flight, then each layer
        # through built in canonical order, then every schedule parsed, then
        # each layer's post-flight.
        assert trace.completed_boundaries() == (
            CompletedBoundary("preflight", _BASE),
            CompletedBoundary("preflight", _LATE),
            CompletedBoundary("axes", _BASE),
            CompletedBoundary("built", _BASE),
            CompletedBoundary("axes", _LATE),
            CompletedBoundary("built", _LATE),
            CompletedBoundary("run_parse", _BASE),
            CompletedBoundary("run_parse", _LATE),
            CompletedBoundary("postflight", _BASE),
            CompletedBoundary("postflight", _LATE),
        )
        counts = {}
        for _, name in parse_events:
            counts[name] = counts.get(name, 0) + 1
        assert counts == {"b0": 1, "b1": 1, "v0": 1}
        # The execution schedule IS the base layer's declarations, projected
        # without re-parsing: same objects, same order.
        base = prepared.layers[0]
        assert all(
            executed is declared
            for executed, declared in zip(prepared.execution_runs,
                                          base.declared_runs, strict=True))
        # A base declaration naming a variant parses against THAT build, and
        # `ParsedRun.layer` is the build it parsed against.
        parsed = prepared.execution_runs[1]
        assert parsed.variant == "late"
        assert parsed.layer.identity == _LATE
        assert seen["b1"].twin is prepared.layers[-1].configured.twin

    def test_a_variant_schedules_entry_naming_another_layer_stands_down(
            self, handler_tables):
        """The twin route of the variant-schedule rule: an inherited entry
        whose ``variant:`` names a DIFFERENT layer belongs to that layer's
        validation, so the untargeted layer never judges it."""
        parse_events = []
        seen = {}
        for kind in list(PARSERS):
            PARSERS[kind] = _parse_spy(parse_events, seen, PARSERS[kind])
        document = conjugate_document(
            {"name": "b0", "kind": "forward"},
            {"name": "b1", "kind": "fisher", "variant": "a"})
        document["variants"] = {"a": {"runtime": {"seed": 1}},
                                "b": {"runtime": {"seed": 2}}}
        prepared = prepare_document(document, scope="all_layers")
        # b1 parses exactly twice: in the base schedule (against a's build)
        # and in a's own schedule (same build).  Never against b's.
        assert [name for _, name in parse_events].count("b1") == 2
        (layer_b,) = [layer for layer in prepared.layers
                      if layer.layer.name == "b"]
        # b's declared record of b1 is a stand-down tombstone: empty views.
        (tombstone,) = [parsed for parsed in layer_b.declared_runs
                        if parsed.name == "b1"]
        assert dict(tombstone.parsed.execution) == {}
        assert tombstone.parsed.resolved == {}

    def test_a_base_exception_is_never_captured(self, handler_tables):
        """The capture scope is ``Exception`` -- the legacy spelling -- so a
        KeyboardInterrupt is nobody's expected refusal."""

        def fail(parsed, configured, previous):
            raise KeyboardInterrupt

        EXECUTORS["forward"] = fail
        with pytest.raises(KeyboardInterrupt):
            execute_prepared(prepare_document(_document(runs=[
                {"name": "a", "kind": "forward", "expect": "refuse"}]),
                scope="all_layers"))


class TestTheSelectedScope:
    def test_an_unselected_variants_text_fault_still_refuses_and_nothing_is_built(
            self, monkeypatch):
        calls = []
        real = document_module.build_resources
        monkeypatch.setattr(
            document_module, "build_resources",
            lambda *args, **kwargs: (calls.append(1), real(*args, **kwargs))[1])
        with pytest.raises(ConfigError, match=r"variants\.bad"):
            prepare_document(_document(variants={"bad": {"campaign": {}}}),
                             scope="selected", variant=None)
        assert calls == []

    def test_only_the_selected_layer_reaches_axes_build_and_postflight(
            self, monkeypatch):
        calls = []
        real = document_module.build_resources
        monkeypatch.setattr(
            document_module, "build_resources",
            lambda *args, **kwargs: (calls.append(1), real(*args, **kwargs))[1])
        trace = _Trace()
        prepared = prepare_document(
            _document(variants={"idle": {"runtime": {"seed": 3}}}),
            scope="selected", variant=None, trace=trace)
        assert [layer.layer.identity for layer in prepared.layers] == [_BASE]
        assert prepared.execution_runs == ()
        assert calls == [1]
        stages = trace.completed_boundaries()
        assert CompletedBoundary(
            "preflight", LayerIdentity("variant", "idle")) in stages
        assert CompletedBoundary(
            "axes", LayerIdentity("variant", "idle")) not in stages

    def test_the_selected_layer_carries_the_named_variants_build(self):
        """The compatibility order: the variant is merged FIRST, and the
        selected layer is the merged document's own base layer."""
        prepared = prepare_document(synthetic_document(), scope="selected",
                                    variant="unity_gain")
        (layer,) = prepared.layers
        assert layer.layer.identity == _BASE
        assert float(layer.configured.twin["gain"].gain) == pytest.approx(1.0)

    def test_an_unknown_variant_selection_is_the_compatibility_refusal(self):
        document = synthetic_document()
        del document["variants"]
        with pytest.raises(ConfigError) as caught:
            prepare_document(document, scope="selected", variant="nope")
        assert str(caught.value) == (
            "variant 'nope' was requested but this document declares no "
            "variants.")
        with pytest.raises(ConfigError) as caught:
            prepare_document(synthetic_document(), scope="selected",
                             variant="nope")
        assert str(caught.value) == (
            "variant 'nope' is not declared; this document declares "
            "['unity_gain'].")

    def test_a_run_naming_an_unbuilt_layer_is_not_parsed_against_the_selected(
            self):
        """The stand-down: selected scope builds ONE layer, so a base run
        targeting another layer cannot be parsed against its build -- and
        parsing it against the WRONG build would refuse valid documents."""
        document = synthetic_document()
        document["variants"]["with_inf"] = {"inference": {
            "twin": {"without": ["noise"]},
            "parameters": {"g": {"init": 1.0, "linear": True,
                                 "into": "gain.gain"}},
            "noise": {"kind": "homoscedastic",
                      "sigma": {"value": 0.05, "unit": "K"}}}}
        document["runs"] = [{"name": "cov", "kind": "fisher",
                             "variant": "with_inf"}]
        run = load_document(document)
        assert isinstance(run, ConfiguredRun)
        prepared = prepare_document(document, scope="all_layers")
        parsed = prepared.execution_runs[0]
        assert parsed.layer.identity == LayerIdentity("variant", "with_inf")
        assert parsed.options == {"space": False, "jitter": 0.0}

    def test_load_document_handler_parses_the_selected_schedule(self):
        """New breadth: the mapping API validates the schedule it builds.

        Measured before this task: ``load_document`` never parsed runs at
        all, so this document loaded clean and the refusal waited for an
        executor that only ``run_document`` would have called.
        """
        document = synthetic_document()
        document["runs"] = [{"kind": "fisher"}]
        with pytest.raises(ConfigError, match="fits latents"):
            load_document(document)


class TestTheVariantTargetExecutesAgainstWhatItParsed:
    def test_a_base_run_naming_an_undeclared_variant_is_refused_before_any_build(
            self, monkeypatch):
        """Every ``variant:`` reference is checked against the declared set
        BEFORE any build -- today a bad name survives the text pass and is
        refused only when the run's turn comes, after the base was built."""
        calls = []
        real = document_module.build_resources
        monkeypatch.setattr(
            document_module, "build_resources",
            lambda *args, **kwargs: (calls.append(1), real(*args, **kwargs))[1])
        document = synthetic_document()
        document["runs"] = [{"name": "v", "kind": "forward",
                             "variant": "nope"}]
        with pytest.raises(ConfigError) as caught:
            prepare_document(document, scope="all_layers")
        assert str(caught.value) == (
            "runs['v']: variant: 'nope' names no declared variant; this "
            "document declares ['unity_gain'].")
        assert calls == []
        # ...and through the public route, with the same precedence.
        with pytest.raises(ConfigError, match="names no declared variant"):
            run_document(document)
        assert calls == []

    def test_parse_pre_execute_and_execute_all_see_the_target_build(
            self, handler_tables):
        seen = {"parse": {}, "pre": {}, "exec": {}}
        real_parse = PARSERS["forward"]

        def parse_spy(options, context):
            seen["parse"][context.spec.name] = context.configured_run
            return real_parse(options, context)

        real_pre = PRE_EXECUTORS["forward"]

        def pre_spy(parsed, configured, previous):
            seen["pre"][parsed.name] = configured
            return real_pre(parsed, configured, previous)

        real_exec = EXECUTORS["forward"]

        def exec_spy(parsed, configured, previous):
            seen["exec"][parsed.name] = configured
            return real_exec(parsed, configured, previous)

        PARSERS["forward"] = parse_spy
        PRE_EXECUTORS["forward"] = pre_spy
        EXECUTORS["forward"] = exec_spy
        document = synthetic_document()
        document["runs"] = [
            {"name": "b", "kind": "forward"},
            {"name": "v", "kind": "forward", "variant": "unity_gain"},
        ]
        prepared = prepare_document(document, scope="all_layers")
        record = execute_prepared(prepared)
        target = prepared.layers[1]
        parsed = prepared.execution_runs[1]
        assert target.layer.identity == LayerIdentity("variant", "unity_gain")
        assert parsed.variant == target.layer.name
        # Parse ran at the run_parse stage, against the variant's build; the
        # completed layer's configured is that build's report-carrying
        # successor, so the twin (not the record) is the shared object.
        assert seen["parse"]["v"].twin is target.configured.twin
        assert seen["pre"]["v"] is target.configured
        assert seen["exec"]["v"] is target.configured
        assert record.runs[1].parsed is parsed
        # ...and it is the variant's build, measurably: gain 1.0 vs 1.1.
        assert not jnp.allclose(record.results["b"].product.data,
                                record.results["v"].product.data)

    def test_the_outcome_row_is_attributed_to_the_target_layer(self):
        trace = _Trace()
        document = synthetic_document()
        document["runs"] = [{"name": "v", "kind": "forward",
                             "variant": "unity_gain"}]
        record = execute_prepared(
            prepare_document(document, scope="all_layers"), trace=trace)
        (layer, row), = trace.outcomes
        assert layer == LayerIdentity("variant", "unity_gain")
        assert row["descriptor"] == {"index": 0, "name": "v", "kind": "forward",
                                     "variant": "unity_gain"}
        assert record.status == "ok"


class TestTheExecutionRecord:
    def test_wall_times_are_measured_per_run(self, handler_tables):
        for kind in list(EXECUTORS):
            EXECUTORS[kind] = (
                lambda parsed, configured, previous: configured.state)
        prepared = prepare_document(_document(runs=[
            {"name": "a", "kind": "forward"},
            {"name": "b", "kind": "forward"}]), scope="all_layers")
        clock = iter((100, 137, 200, 251))
        record = execute_prepared(prepared, clock_ns=clock.__next__)
        assert [row.wall_time_ns for row in record.runs] == [37, 51]
        assert [row.index for row in record.runs] == [0, 1]
        assert record.status == "ok"
        assert isinstance(record, ExecutionRecord)

    def test_a_captured_expected_refusal_is_a_successful_run(
            self, handler_tables):
        error = ValueError("boom")

        def fail(parsed, configured, previous):
            raise error

        EXECUTORS["forward"] = fail
        trace = _Trace()
        prepared = prepare_document(_document(runs=[
            {"name": "a", "kind": "forward", "expect": "refuse"}]),
            scope="all_layers")
        record = execute_prepared(prepared, trace=trace)
        (row,) = record.runs
        assert row.status == "ok"
        assert row.captured_expected_refusal is True
        assert row.error is None
        assert row.result.product is None
        assert row.result.error is error
        assert record.status == "ok"
        assert record.error is None
        assert record.results["a"].error is error
        (_, outcome), = trace.outcomes
        assert outcome["status"] == "expected_refusal"
        assert outcome["capture_scope"] == "arbitrary_exception"
        assert outcome["is_dirt_error"] is False
        assert outcome["exception_type"].endswith("ValueError")
        assert outcome["exception_message"] == "boom"

    def test_a_parse_time_refusal_is_captured_for_an_expect_refuse_run(self):
        """The capture follows the run across the parse/execute split.

        ``fisher`` with no ``inference.parameters`` refuses at handler PARSE
        since Task 7; the legacy wrapper heard it because parse ran inside
        the capture.  The tombstone is the parse-time half of that contract.
        """
        document = synthetic_document()
        document["runs"] = [{"name": "a", "kind": "fisher",
                             "variant": "unity_gain", "expect": "refuse"}]
        results = run_document(document)
        assert results["a"].product is None
        assert isinstance(results["a"].error, ConfigError)
        assert results["a"].variant == "unity_gain"
        prepared = prepare_document(document, scope="all_layers")
        (tombstone,) = prepared.execution_runs
        assert tombstone.parsed.resolved == {}
        assert dict(tombstone.parsed.execution) == {}
        # The declaration is still recorded, on both layers that carry it.
        assert [parsed.name for parsed in prepared.layers[0].declared_runs
                ] == ["a"]
        record = execute_prepared(prepared)
        (row,) = record.runs
        assert row.status == "ok"
        assert row.captured_expected_refusal is True
        assert isinstance(row.result.error, ConfigError)

    def test_an_uncaptured_config_error_refuses_the_run_and_the_document(
            self, handler_tables):
        error = ConfigError("the fit blew up.")

        def fail(parsed, configured, previous):
            if parsed.name == "bad":
                raise error
            return configured.state

        EXECUTORS["forward"] = fail

        def document():
            return _document(runs=[{"name": "good", "kind": "forward"},
                                   {"name": "bad", "kind": "forward"},
                                   {"name": "late", "kind": "forward"}])

        record = execute_prepared(prepare_document(document(),
                                                   scope="all_layers"))
        # Prior successes plus the failing row remain; later rows do not
        # exist, and the failing run has no result entry.
        assert [row.parsed.name for row in record.runs] == ["good", "bad"]
        assert record.runs[0].status == "ok"
        assert record.runs[1].status == "refused"
        assert record.runs[1].result is None
        assert record.runs[1].error is error
        assert record.status == "refused"
        assert record.error is error
        assert set(record.results) == {"good"}
        with pytest.raises(ConfigError) as caught:
            run_document(document())
        assert caught.value is error

    def test_an_unexpected_exception_is_an_error(self, handler_tables):
        error = RuntimeError("kaboom")

        def fail(parsed, configured, previous):
            raise error

        EXECUTORS["forward"] = fail
        prepared = prepare_document(_document(), scope="all_layers")
        record = execute_prepared(prepared)
        (row,) = record.runs
        assert row.status == "error"
        assert row.error is error
        assert record.status == "error"
        with pytest.raises(RuntimeError, match="kaboom"):
            run_document(_document())

    def test_the_succeeded_refusal_message_is_the_legacy_one(self):
        document = _document(runs=[{"name": "a", "kind": "forward",
                                    "expect": "refuse"}])
        record = execute_prepared(prepare_document(document,
                                                   scope="all_layers"))
        (row,) = record.runs
        assert row.status == "refused"
        assert str(row.error) == (
            "runs['a']: expect: refuse, and kind: forward SUCCEEDED -- the "
            "assertion this run makes about the design no longer holds.")
        assert record.status == "refused"
        with pytest.raises(ConfigError, match="SUCCEEDED"):
            run_document(document)

    def test_prior_results_are_read_only_and_a_retained_view_never_grows(
            self, handler_tables):
        seen = {}

        def spy(parsed, configured, previous):
            seen[parsed.name] = previous
            return configured.state

        EXECUTORS["forward"] = spy
        record = execute_prepared(prepare_document(_document(runs=[
            {"name": "a", "kind": "forward"},
            {"name": "b", "kind": "forward"}]), scope="all_layers"))
        assert record.status == "ok"
        assert sorted(seen["a"]) == []
        assert sorted(seen["b"]) == ["a"]
        with pytest.raises(TypeError):
            seen["a"]["b"] = "too late"
        # The view run "a" retained does not grow after run "b" lands --
        # the proxy wraps a COPY (reuse: may only look backwards).
        assert "b" not in seen["a"]

    def test_the_results_view_is_read_only(self, handler_tables):
        prepared = prepare_document(_document(), scope="all_layers")
        record = execute_prepared(prepared)
        with pytest.raises(TypeError):
            record.results["forward"] = None
        assert dict(record.results)

    def test_completed_boundaries_come_from_the_trace(self, handler_tables):
        prepared = prepare_document(_document(), scope="all_layers",
                                    trace=_Trace())
        record = execute_prepared(prepared)
        assert record.completed_boundaries == ()
        trace = _Trace()
        prepared = prepare_document(_document(), scope="all_layers",
                                    trace=trace)
        record = execute_prepared(prepared, trace=trace)
        assert record.completed_boundaries == trace.completed_boundaries()
        assert record.completed_boundaries != ()

    def test_outcome_rows_use_the_closed_keys(self, handler_tables):
        EXECUTORS["forward"] = (
            lambda parsed, configured, previous: configured.state)
        trace = _Trace()
        prepared = prepare_document(_document(), scope="all_layers",
                                    trace=trace)
        clock = iter((10, 42))
        record = execute_prepared(prepared, clock_ns=clock.__next__,
                                  trace=trace)
        (layer, row), = trace.outcomes
        assert layer == _BASE
        assert set(row) == {"descriptor", "status", "wall_time_ns",
                            "exception_type", "exception_message",
                            "capture_scope", "is_dirt_error"}
        assert row["descriptor"] == {"index": 0, "name": "forward",
                                     "kind": "forward", "variant": None}
        assert row["status"] == "ok"
        assert row["wall_time_ns"] == 32
        assert row["exception_type"] is None
        assert row["exception_message"] is None
        assert row["capture_scope"] is None
        assert row["is_dirt_error"] is None
        assert record.status == "ok"


class TestTheTargetLookup:
    """execute_one_parsed looks up exactly one prepared layer by identity."""

    def _prepared(self):
        return prepare_document(_document(), scope="all_layers")

    def test_an_absent_target_is_refused(self):
        prepared = self._prepared()
        orphan = PreparedDocument(layers=(),
                                  execution_runs=prepared.execution_runs)
        with pytest.raises(ConfigError) as caught:
            execute_prepared(orphan)
        assert str(caught.value) == (
            "runs['forward']: the run was parsed against the base layer, "
            "which is not among this prepared document's layers; the two "
            "come from different enumerations.")

    def test_a_duplicate_target_is_refused(self):
        prepared = self._prepared()
        layer = prepared.layers[0]
        doubled = PreparedDocument(layers=(layer, layer),
                                   execution_runs=prepared.execution_runs)
        with pytest.raises(ConfigError) as caught:
            execute_prepared(doubled)
        assert str(caught.value) == (
            "runs['forward']: the base layer appears twice in this prepared "
            "document.")

    def test_an_inconsistent_target_is_refused(self):
        prepared = self._prepared()
        foreign = _enumerate(_document()).layers[0]
        mismatched = PreparedDocument(
            layers=(PreparedLayer(layer=foreign,
                                  configured=prepared.layers[0].configured,
                                  declared_runs=prepared.layers[0]
                                  .declared_runs),),
            execution_runs=prepared.execution_runs)
        with pytest.raises(ConfigError) as caught:
            execute_prepared(mismatched)
        assert str(caught.value) == (
            "runs['forward']: the prepared base layer is not the one the "
            "run was parsed against; the two come from different "
            "enumerations.")


class TestLayerFreezing:
    def test_each_layer_is_frozen_immediately_after_its_own_postflight(self):
        trace = _Trace()
        document = synthetic_document()
        document["variants"] = {"v": {"runtime": {"seed": 5}}}
        prepare_document(document, scope="all_layers", trace=trace)
        variant = LayerIdentity("variant", "v")
        assert [layer for layer, _ in trace.frozen] == [_BASE, variant]
        events = trace.events
        postflight_base = events.index(("boundary", "postflight", _BASE))
        freeze_base = events.index(("freeze", _BASE))
        postflight_v = events.index(("boundary", "postflight", variant))
        assert postflight_base < freeze_base < postflight_v
        (base_row,) = [row for layer, row in trace.frozen if layer == _BASE]
        (variant_row,) = [row for layer, row in trace.frozen
                          if layer == variant]
        assert [row["name"] for row in base_row["declared_runs"]] == [
            "forward"]
        assert [row["name"] for row in base_row["execution_runs"]] == [
            "forward"]
        # Every completed layer records the same base execution projection.
        assert [row["name"] for row in variant_row["execution_runs"]] == [
            "forward"]
        assert variant_row["declared_runs"] == base_row["declared_runs"]

    def test_a_later_postflight_refusal_leaves_the_failing_layer_unfrozen(
            self, all_registries):
        register_priced("C97")(lambda payload: (
            (refuse("C97", "runtime.seed", "variant fault."),)
            if payload.run.document["runtime"]["seed"] == 5 else ()))
        trace = _Trace()
        variant = LayerIdentity("variant", "v")
        document = synthetic_document()
        document["variants"] = {"v": {"runtime": {"seed": 5}}}
        with pytest.raises(ConfigError, match="variant fault"):
            prepare_document(document, scope="all_layers", trace=trace)
        # The base completed and is frozen; the failing variant and any
        # later layer are not.
        assert [layer for layer, _ in trace.frozen] == [_BASE]
        boundaries = trace.completed_boundaries()
        assert CompletedBoundary("postflight", _BASE) in boundaries
        assert CompletedBoundary("postflight", variant) in boundaries


class TestSuppliedLayers:
    def test_select_build_layers_names_and_selects_from_the_enumeration(self):
        canonical = canonical_layers(synthetic_document())
        with pytest.raises(ConfigError) as caught:
            select_build_layers(canonical, scope="selected", variant="nope")
        assert str(caught.value) == (
            "variant 'nope' is not declared; this document declares "
            "['unity_gain'].")
        (layer,) = select_build_layers(canonical, scope="selected",
                                       variant="unity_gain")
        assert layer.identity == LayerIdentity("variant", "unity_gain")
        assert select_build_layers(canonical, scope="selected",
                                   variant=None)[0].identity == _BASE
        assert len(select_build_layers(canonical, scope="all_layers",
                                       variant=None)) == 2

    def test_supplied_layers_thread_their_own_evidence_maps(self):
        document = _document()
        enumeration = _enumerate(document)
        trace = _Trace()
        prepared = prepare_document(
            document, scope="all_layers", layers=enumeration.layers,
            layer_origins=enumeration.origins,
            layer_deletions=enumeration.deletions, trace=trace)
        assert [layer.layer.identity for layer in prepared.layers] == [_BASE]
        (layer, row), = trace.frozen
        assert layer == _BASE
        assert row["origins"] is enumeration.origins[_BASE]
        assert tuple(row["deletions"]) == tuple(enumeration.deletions[_BASE])

    def test_supplied_layers_require_both_evidence_maps(self):
        enumeration = _enumerate(_document())
        with pytest.raises(ConfigError) as caught:
            prepare_document(_document(), scope="all_layers",
                             layers=enumeration.layers)
        assert str(caught.value) == (
            "canonical layers supplied by an integration caller require "
            "their matching layer_origins and layer_deletions maps.")

    def test_evidence_maps_require_supplied_layers(self):
        enumeration = _enumerate(_document())
        with pytest.raises(ConfigError) as caught:
            prepare_document(_document(), scope="all_layers",
                             layer_origins=enumeration.origins,
                             layer_deletions=enumeration.deletions)
        assert str(caught.value) == (
            "layer_origins and layer_deletions accompany supplied canonical "
            "layers; layers is None.")

    def test_the_maps_must_cover_every_layer(self):
        enumeration = _enumerate(_document())
        with pytest.raises(ConfigError) as caught:
            prepare_document(_document(), scope="all_layers",
                             layers=enumeration.layers, layer_origins={},
                             layer_deletions={})
        assert str(caught.value) == (
            "supplied layer_origins/layer_deletions must cover every "
            "canonical layer identity exactly once.")

    def test_an_unknown_scope_is_a_programming_error(self):
        with pytest.raises(ValueError) as caught:
            prepare_document(_document(), scope="everything")
        assert str(caught.value) == (
            "scope is 'selected' or 'all_layers'; got 'everything'.")

    def test_supplied_layers_select_a_named_variant(self):
        """The integration route's selection: the name resolves against the
        supplied enumeration (no merge -- supplied layers ARE canonical)."""
        enumeration = _enumerate(synthetic_document())
        prepared = prepare_document(
            synthetic_document(), scope="selected", variant="unity_gain",
            layers=enumeration.layers, layer_origins=enumeration.origins,
            layer_deletions=enumeration.deletions)
        (layer,) = prepared.layers
        assert layer.layer.identity == LayerIdentity("variant", "unity_gain")
        assert float(layer.configured.twin["gain"].gain) == pytest.approx(1.0)


class TestTheParsedViewsAreTheHandlers:
    def test_the_parsed_views_carry_the_handlers_normalization(self):
        prepared = prepare_document(_document(runs=[{"name": "cov",
                                                     "kind": "fisher"}]),
                                    scope="all_layers")
        (parsed,) = prepared.execution_runs
        # fisher's parser normalizes and injects the measured defaults.
        assert parsed.options == {"space": False, "jitter": 0.0}
        assert dict(parsed.parsed.resolved) == {"space": False, "jitter": 0.0}

    def test_the_trace_records_one_parsed_run_row_per_parse(self):
        trace = _Trace()
        prepare_document(_document(), scope="all_layers", trace=trace)
        (layer, row), = trace.parsed_runs
        assert layer == _BASE
        assert set(row) == {"descriptor", "resolved_options",
                            "deferred_checks"}
        assert row["descriptor"] == {"index": 0, "name": "forward",
                                     "kind": "forward", "variant": None}
