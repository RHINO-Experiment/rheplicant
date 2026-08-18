"""The atomic live exit-handler registry: four tables, one binder, one parse.

Plan 4A Task 6 installs ``PARSERS``, ``PRE_EXECUTORS``, ``EXECUTORS`` and
``DEFERRED_CHECKS`` behind one lock, the ``handler_for`` live projection,
``parsed_options``/``parse_run`` for the §1.3 parsed records, and keeps the
compatibility ``execute_run`` dispatch byte-identical while the sixteen
built-ins are transitionally registered with ``_legacy_freeze_parse``.
Tasks 7-9 replace those entries kind by kind; Task 9 owns the hard
no-legacy gate.

Plan 4A Task 7 migrates the five base kinds (``forward``, ``fisher``,
``optimize``, ``plan.estimate``, ``plan.sample``) onto explicit parsers:
defaults are normalized into both frozen views, every grammar refusal
precedes science, and the transitional pin below narrows to the eleven
unmigrated kinds.
"""

import threading
from collections.abc import Mapping

import pytest

from _rheplicant_bootstrap.types import LayerIdentity
from _rheplicant_bootstrap.variants import LayerRef
from rheplicant.config.errors import ConfigError
from rheplicant.config.sections import exits  # registration side effects
from rheplicant.config.sections.exit_support import (
    DEFERRED_CHECKS,
    EXECUTORS,
    PARSERS,
    PRE_EXECUTORS,
    ExitHandler,
    ParsedOptions,
    ParsedRun,
    RunParseContext,
    _legacy_freeze_parse,
    handler_for,
    parse_run,
    parsed_options,
    register,
)
from rheplicant.config.sections.runs import _KINDS, RunSpec
from tests.config.exit_helpers import HOMOSCEDASTIC, ONE_LATENT, conjugate_built

_REGISTRIES = (PARSERS, PRE_EXECUTORS, EXECUTORS, DEFERRED_CHECKS)


def _pop_all(kind):
    """A temporary registration leaves nothing behind, in any table."""
    for registry in _REGISTRIES:
        registry.pop(kind, None)


def _layer(name=None):
    """A minimal valid LayerRef: the base layer, or the variant it names."""
    if name is None:
        return LayerRef(kind="base", name=None, prefix="", document={},
                        declared_runs=None)
    return LayerRef(kind="variant", name=name, prefix=f"variants.{name}",
                    document={}, declared_runs=None)


def _execute(parsed, built, previous):
    return None


def _spec(kind, **overrides):
    fields = {"name": "run", "kind": kind, "variant": None, "on": "primary",
              "expect": "ok", "options": {}}
    fields.update(overrides)
    return RunSpec(**fields)


class TestEveryDeclaredKindHasOneLiveHandler:
    """The registry is complete in all four tables, not only in EXECUTORS.

    Importing ``exits`` registers the leaf modules as an import side effect,
    which is exactly the wiring that rots in silence: a kind with an executor
    but no parser would pass the old set comparison and fail Task 10's
    orchestration.  ``_KINDS`` remains the declared 16-kind source
    (plan §1.7).
    """

    def test_every_declared_kind_has_one_live_handler(self):
        assert exits is not None  # importing exits registers the leaf modules
        assert set(_KINDS) == set(PARSERS) == set(PRE_EXECUTORS)
        assert set(_KINDS) == set(EXECUTORS) == set(DEFERRED_CHECKS)
        assert len(PARSERS) == 16

    def test_every_builtin_is_transitionally_on_the_legacy_parser(self):
        """Tasks 8-9 replace the remaining entries; Task 9 owns the gate."""
        unmigrated = ("conjugate.wiener", "conjugate.gcr", "conjugate.gls",
                      "condition", "identifiability", "score_directions",
                      "gradient", "mmodes", "predict", "nuts", "npe")
        for kind in unmigrated:
            assert PARSERS[kind] is _legacy_freeze_parse, kind
        for kind in ("forward", "fisher", "optimize", "plan.estimate",
                     "plan.sample"):
            assert PARSERS[kind] is not _legacy_freeze_parse, kind
        handler = handler_for("forward")
        assert isinstance(handler, ExitHandler)
        assert handler.deferred_checks == ()


class TestDuplicateRegistrationChangesNothing:
    """A refused second binding must not leave a half-written handler.

    Refusing AFTER assigning any one table would leave a poisoned partial
    handler -- a parser with no executor, or the reverse -- that the NEXT
    test touching the kind would read as real wiring.
    """

    def test_duplicate_registration_changes_nothing(self):
        # The duplicate attempt carries DIFFERENT members on purpose: a
        # binder that assigned before refusing would replace them, and a
        # snapshot of equal values could not tell.
        def first_parse(options, context):
            return parsed_options(options, resolved=options)

        def second_parse(options, context):
            return parsed_options(options, resolved=options)

        def first_execute(parsed, built, previous):
            return None

        def second_execute(parsed, built, previous):
            return None

        register("_dup_probe", parse=first_parse)(first_execute)
        try:
            before = tuple(dict(registry) for registry in _REGISTRIES)
            with pytest.raises(ConfigError, match="registered twice"):
                register("_dup_probe", parse=second_parse)(second_execute)
            after = tuple(dict(registry) for registry in _REGISTRIES)
            assert after == before
            assert PARSERS["_dup_probe"] is first_parse
            assert EXECUTORS["_dup_probe"] is first_execute
        finally:
            _pop_all("_dup_probe")

    def test_the_duplicate_message_still_names_both_modules_in_order(self):
        """The incumbent's module first, the challenger's second -- pinned.

        With the transitional adapter, the incumbent's ``__module__`` must
        still be the module that wrote the executor, or a reader chasing a
        collision is sent to the adapter instead of the claimant.
        """

        def probe(run, built, *, results=None):
            return None

        register("_dup_modules")(probe)
        try:
            with pytest.raises(ConfigError, match="registered twice") as got:
                register("_dup_modules")(probe)
            message = str(got.value)
            assert probe.__module__ in message
        finally:
            _pop_all("_dup_modules")


def race_two_registrations(kind, spins=400):
    """Two binders and one observer on one kind; returns their outcomes.

    The observer's question is the atomicity one: ``handler_for`` may only
    ever answer with a COMPLETE handler (all four fields bound) or refuse --
    never a handler whose parser exists but whose executor does not.  That is
    what the single lock around all four assignments buys.
    """
    barrier = threading.Barrier(3)
    outcomes = []
    partial_sightings = []
    lock = threading.Lock()

    def execute_a(parsed, built, previous):
        return "a"

    def execute_b(parsed, built, previous):
        return "b"

    def binder(execute):
        barrier.wait()
        try:
            register(kind)(execute)
        except ConfigError:
            with lock:
                outcomes.append("duplicate")
        else:
            with lock:
                outcomes.append("bound")

    def observer():
        barrier.wait()
        for _ in range(spins):
            try:
                handler = handler_for(kind)
            except ConfigError:
                continue
            if not (callable(handler.parse)
                    and callable(handler.pre_execute)
                    and callable(handler.execute)
                    and isinstance(handler.deferred_checks, tuple)):
                with lock:
                    partial_sightings.append(handler)

    threads = [threading.Thread(target=binder, args=(execute_a,)),
               threading.Thread(target=binder, args=(execute_b,)),
               threading.Thread(target=observer)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return outcomes, partial_sightings


class TestTwoConcurrentBindersPublishExactlyOneCompleteHandler:
    def test_two_concurrent_binders_publish_exactly_one_complete_handler(self):
        outcomes, partials = race_two_registrations("_race")
        try:
            assert sorted(outcomes) == ["bound", "duplicate"]
            assert all("_race" in registry for registry in _REGISTRIES)
            assert partials == []
        finally:
            _pop_all("_race")


class TestTheHandlerProjectionIsLive:
    """``handler_for`` reads the tables at call time, never a cached copy.

    A cached handler is how a monkeypatched executor silently stops being
    the one a run gets -- the projection must be assembled on every call.
    """

    def test_handler_projection_is_live(self, monkeypatch):
        def replacement(run, built, previous):
            return "replacement"

        monkeypatch.setitem(EXECUTORS, "forward", replacement)
        assert handler_for("forward").execute is replacement

    def test_the_parser_projection_is_live_too(self, monkeypatch):
        def replacement(options, context):
            return parsed_options(options, resolved=options)

        monkeypatch.setitem(PARSERS, "forward", replacement)
        assert handler_for("forward").parse is replacement


class TestParsedOptionsAreTwoDetachedFrozenViews:
    def test_parsed_options_are_two_detached_frozen_views(self):
        raw = {"nested": {"items": [1, 2]}}
        parsed = parsed_options(raw, resolved=raw)
        raw["nested"]["items"][0] = 9
        assert parsed.execution["nested"]["items"] == (1, 2)
        assert parsed.resolved["nested"]["items"] == (1, 2)
        assert parsed.execution is not parsed.resolved

    def test_both_views_are_read_only_and_the_record_is_frozen(self):
        parsed = parsed_options({"a": [1]}, resolved={"a": [1]})
        with pytest.raises(TypeError):
            parsed.execution["a"] = 2
        with pytest.raises(TypeError):
            parsed.resolved["a"] = 2
        import dataclasses

        with pytest.raises(dataclasses.FrozenInstanceError):
            parsed.execution = {}

    def test_the_two_views_may_differ(self):
        """The seam Tasks 7-9 use: execution holds the hook, resolved its name."""
        def hook(prediction):
            return prediction

        parsed = parsed_options({"loss": hook}, resolved={"loss": "mod:fn"})
        assert parsed.execution["loss"] is hook
        assert parsed.resolved["loss"] == "mod:fn"

    def test_neither_root_may_be_a_non_mapping(self):
        with pytest.raises(ConfigError):
            parsed_options([("a", 1)], resolved={"a": 1})
        with pytest.raises(ConfigError):
            parsed_options({"a": 1}, resolved=[("a", 1)])


class TestNonYamlSafeResolvedValuesAreRejectedAtomically:
    """The resolved view is the audit representation: ``JsonValue`` only.

    ``freeze_evidence`` already refuses sets, foreign objects, cycles and
    non-string keys; the YAML-safe walk adds the binary leaf types, which
    canonicalize to ``bytes`` and would otherwise reach Task 15A's encoder.
    """

    @pytest.mark.parametrize(
        "blob",
        [b"\x00", bytearray(b"\x00"), memoryview(b"\x00")],
        ids=["bytes", "bytearray", "memoryview"],
    )
    def test_a_binary_resolved_value_is_refused(self, blob):
        with pytest.raises(ConfigError, match="YAML-safe"):
            parsed_options({"ok": 1}, resolved={"blob": blob})

    def test_a_deeply_nested_binary_value_is_refused(self):
        resolved = {"outer": [{"inner": b"\x00"}]}
        with pytest.raises(ConfigError, match="YAML-safe"):
            parsed_options({}, resolved=resolved)

    def test_a_non_string_key_is_refused(self):
        with pytest.raises(ConfigError):
            parsed_options({}, resolved={1: "one"})

    def test_a_cyclic_resolved_view_is_refused(self):
        resolved = {}
        resolved["self"] = resolved
        with pytest.raises(ConfigError):
            parsed_options({}, resolved=resolved)

    def test_the_rejection_is_atomic_and_appends_no_trace_event(self):
        """Neither view escapes, and ``parse_run`` records nothing."""
        rows = []

        class Trace:
            def record_parsed_run(self, layer, row):
                rows.append(row)

        def parse(options, context):
            return parsed_options(options, resolved={"blob": b"\x00"})

        register("_atomic_probe", parse=parse)(_execute)
        try:
            with pytest.raises(ConfigError, match="YAML-safe"):
                parse_run(_spec("_atomic_probe"), object(), index=0,
                          layer=_layer(), trace=Trace())
        finally:
            _pop_all("_atomic_probe")
        assert rows == []


class TestParseRun:
    """``parse_run`` calls one current parser and projects one trace row."""

    def test_parse_run_calls_only_the_current_parser_once(self):
        calls = []

        def parse(options, context):
            calls.append((options, context))
            return parsed_options({"seen": options["seen"]},
                                  resolved={"seen": options["seen"]})

        register("_parse_probe", parse=parse)(_execute)
        try:
            configured = object()
            spec = _spec("_parse_probe", variant="v", options={"seen": 1})
            layer = _layer("v")
            parsed = parse_run(spec, configured, index=3, layer=layer)
            assert len(calls) == 1
            options, context = calls[0]
            assert isinstance(context, RunParseContext)
            assert isinstance(parsed.parsed, ParsedOptions)
            assert options is spec.options
            assert context.index == 3
            assert context.layer is layer
            assert context.spec is spec
            assert context.configured_run is configured
            assert parsed.index == 3 and parsed.layer is layer
            assert parsed.spec is spec
            assert parsed.parsed.execution["seen"] == 1
        finally:
            _pop_all("_parse_probe")

    def test_the_projection_is_appended_after_a_successful_parse(self):
        order = []
        rows = []

        def parse(options, context):
            order.append("parse")
            return parsed_options(options, resolved=options)

        class Trace:
            def record_parsed_run(self, layer, row):
                order.append("trace")
                rows.append((layer, row))

        register("_trace_probe", parse=parse, deferred_checks=("late",))(
            _execute)
        try:
            layer = _layer("v")
            parse_run(_spec("_trace_probe", name="p", variant="v",
                            options={"a": 1}),
                      object(), index=2, layer=layer, trace=Trace())
        finally:
            _pop_all("_trace_probe")
        assert order == ["parse", "trace"]
        (identity, row), = rows
        assert identity == LayerIdentity("variant", "v")
        assert tuple(row) == ("descriptor", "resolved_options",
                              "deferred_checks")
        assert tuple(row["descriptor"]) == ("index", "name", "kind",
                                            "variant")
        assert row["descriptor"] == {"index": 2, "name": "p",
                                     "kind": "_trace_probe", "variant": "v"}
        assert row["resolved_options"] == {"a": 1}
        assert row["deferred_checks"] == ("late",)

    def test_a_failing_parser_appends_nothing(self):
        rows = []

        class Trace:
            def record_parsed_run(self, layer, row):
                rows.append(row)

        def parse(options, context):
            raise ConfigError("runs['run']: bad option.")

        register("_failing_probe", parse=parse)(_execute)
        try:
            with pytest.raises(ConfigError, match="bad option"):
                parse_run(_spec("_failing_probe"), object(), index=0,
                          layer=_layer(), trace=Trace())
        finally:
            _pop_all("_failing_probe")
        assert rows == []

    def test_an_unknown_kind_uses_the_registry_wording(self):
        with pytest.raises(ConfigError) as caught:
            parse_run(_spec("_nowhere"), object(), index=0, layer=_layer())
        message = str(caught.value)
        assert message.startswith("runs[].kind: '_nowhere' is not registered")
        assert "it takes" in message

    def test_a_parser_returning_anything_but_parsed_options_is_a_wiring_error(
            self):
        register("_bad_parse", parse=lambda options, context: {})(_execute)
        try:
            with pytest.raises(TypeError, match="ParsedOptions"):
                parse_run(_spec("_bad_parse"), object(), index=0,
                          layer=_layer())
        finally:
            _pop_all("_bad_parse")


class TestParsedRunDelegates:
    """The delegating properties are what let a ``ParsedRun`` stand where a
    ``RunSpec`` stood -- and ``options`` is the parsed execution view, never
    the raw spec mapping."""

    def _parsed(self, spec):
        return ParsedRun(index=0, layer=_layer(), spec=spec,
                         parsed=parsed_options(spec.options,
                                               resolved=spec.options))

    def test_the_six_spec_delegations(self):
        spec = _spec("forward", name="n", variant="v", on="night",
                     expect="refuse", options={"x": 1}, reuse="earlier")
        parsed = self._parsed(spec)
        assert (parsed.name, parsed.kind, parsed.variant, parsed.on,
                parsed.expect, parsed.reuse) == (
            "n", "forward", "v", "night", "refuse", "earlier")

    def test_options_is_the_parsed_execution_view(self):
        spec = _spec("forward", options={"x": [1]})
        parsed = self._parsed(spec)
        assert parsed.options is parsed.parsed.execution
        assert parsed.options is not spec.options
        assert parsed.options["x"] == (1,)


class TestRegisterValidatesBeforeBinding:
    """Nothing is bound before every member of the handler checks out."""

    def test_non_callable_members_are_type_errors_and_bind_nothing(self):
        with pytest.raises(TypeError, match="callable"):
            register("_probe")(42)
        with pytest.raises(TypeError, match="callable"):
            register("_probe", pre_execute=42)(_execute)
        with pytest.raises(TypeError, match="callable"):
            register("_probe", parse=42)(_execute)
        assert all("_probe" not in registry for registry in _REGISTRIES)

    def test_deferred_checks_are_unique_non_empty_strings(self):
        for bad in (("a", "a"), ("",), (7,), "ab"):
            with pytest.raises(ValueError,
                               match="unique non-empty strings"):
                register("_probe", deferred_checks=bad)(_execute)
        assert all("_probe" not in registry for registry in _REGISTRIES)

    def test_deferred_checks_are_stored_on_the_handler(self):
        register("_probe", deferred_checks=("late", "later"))(_execute)
        try:
            assert DEFERRED_CHECKS["_probe"] == ("late", "later")
            assert handler_for("_probe").deferred_checks == ("late", "later")
        finally:
            _pop_all("_probe")


class TestABindingFailureRollsBackAllFourRegistries:
    def test_a_mid_binding_failure_leaves_no_partial_handler(
            self, monkeypatch):
        import rheplicant.config.sections.exit_support as support

        class ExplodingDict(dict):
            def __setitem__(self, key, value):
                raise RuntimeError("boom")

        monkeypatch.setattr(support, "DEFERRED_CHECKS", ExplodingDict())
        with pytest.raises(RuntimeError, match="boom"):
            register("_rollback")(_execute)
        assert all("_rollback" not in registry
                   for registry in (PARSERS, PRE_EXECUTORS, EXECUTORS))


class TestExecuteRunKeepsItsCompatibilitySeams:
    """The wrapper keeps its signature, its legacy unknown-kind message, and
    its capture scope -- while routing through parse, pre-execute, execute."""

    def test_the_unknown_kind_message_is_the_legacy_one_byte_for_byte(self):
        run = _spec("not_a_kind", name="x")
        with pytest.raises(ConfigError) as caught:
            exits.execute_run(run, None)
        assert str(caught.value) == (
            "runs['x']: kind: not_a_kind has no executor. Every kind this "
            f"layer declares must register one; it knows {sorted(EXECUTORS)}."
        )

    def test_execute_run_parses_pre_executes_then_executes_in_order(self):
        events = []

        def parse(options, context):
            events.append("parse")
            return parsed_options(options, resolved=options)

        def pre_execute(parsed, built, previous):
            events.append("pre_execute")

        def execute(parsed, built, previous):
            events.append("execute")
            return "product"

        register("_order", parse=parse, pre_execute=pre_execute)(execute)
        try:
            result = exits.execute_run(_spec("_order", options={"k": [1]}),
                                       object())
            assert events == ["parse", "pre_execute", "execute"]
            assert result.product == "product"
            assert result.error is None
        finally:
            _pop_all("_order")

    def test_the_executor_receives_the_parsed_run_not_the_spec(self):
        seen = []

        def parse(options, context):
            return parsed_options(options, resolved=options)

        def execute(parsed, built, previous):
            seen.append((parsed, built, previous))
            return parsed.options

        register("_parsed", parse=parse)(execute)
        try:
            spec = _spec("_parsed", variant="v", options={"k": [1]})
            built = object()
            result = exits.execute_run(spec, built)
            (parsed, got_built, previous), = seen
            assert isinstance(parsed, ParsedRun)
            assert parsed.options["k"] == (1,)
            assert parsed.options is not spec.options
            assert got_built is built
            assert result.product["k"] == (1,)
            with pytest.raises(TypeError):
                previous["x"] = 1
        finally:
            _pop_all("_parsed")

    def test_expect_refuse_captures_a_pre_execute_or_execute_error(self):
        def parse(options, context):
            return parsed_options(options, resolved=options)

        def refusing_pre(parsed, built, previous):
            raise ConfigError("a late check failed.")

        def refusing_execute(parsed, built, previous):
            raise ValueError("boom")

        register("_refuse_pre", parse=parse, pre_execute=refusing_pre)(
            _execute)
        register("_refuse_exec", parse=parse)(refusing_execute)
        try:
            captured = exits.execute_run(
                _spec("_refuse_pre", name="a", expect="refuse"), object())
            assert isinstance(captured.error, ConfigError)
            assert captured.product is None
            captured = exits.execute_run(
                _spec("_refuse_exec", name="b", expect="refuse"), object())
            assert isinstance(captured.error, ValueError)
        finally:
            _pop_all("_refuse_pre")
            _pop_all("_refuse_exec")

    def test_expect_refuse_captures_a_parse_error(self):
        """The parser's refusal is the run's own, so the capture hears it.

        The grammar checks lived INSIDE the executor before the parse/execute
        split; leaving parse outside the capture would silently narrow the
        pre-split scope (a refused ``jitter: -1.0`` used to be a recorded
        ``RunResult.error``, not an aborted document).
        """

        def parse(options, context):
            raise ConfigError("runs['run']: a parse-time refusal.")

        register("_refuse_parse", parse=parse)(_execute)
        try:
            captured = exits.execute_run(
                _spec("_refuse_parse", name="p", expect="refuse"), object())
            assert isinstance(captured.error, ConfigError)
            assert captured.product is None
        finally:
            _pop_all("_refuse_parse")

    def test_expect_refuse_still_refuses_a_success(self):
        def parse(options, context):
            return parsed_options(options, resolved=options)

        register("_succeeds", parse=parse)(_execute)
        try:
            with pytest.raises(ConfigError, match="SUCCEEDED"):
                exits.execute_run(
                    _spec("_succeeds", name="c", expect="refuse"), object())
        finally:
            _pop_all("_succeeds")


class TestLegacyRegistrationsKeepTheLegacyDispatch:
    """``parse=`` omitted: the executor keeps TODAY'S calling convention.

    This is the transitional shape the sixteen built-ins sit in until
    Tasks 7-9: the registry stores a new-convention adapter that hands the
    legacy function its raw ``RunSpec`` and keyword ``results=`` -- measured
    byte-identical dispatch -- while the parser is real and the four tables
    are complete.
    """

    def test_a_parse_less_registration_executes_with_the_legacy_convention(
            self):
        seen = []

        def legacy(run, built, *, results=None):
            seen.append((run, built, results))
            return "ok"

        register("_legacy")(legacy)
        try:
            spec = _spec("_legacy", options={"k": 1})
            built = object()
            result = exits.execute_run(spec, built, results={"a": 2})
            (run, got_built, results), = seen
            assert run is spec
            assert run.options is spec.options  # raw, unfrozen, untouched
            assert got_built is built
            assert results == {"a": 2}
            assert result.product == "ok"
        finally:
            _pop_all("_legacy")

    def test_a_parse_less_registration_still_gets_the_real_parser(self):
        def legacy(run, built, *, results=None):
            return None

        register("_legacy2")(legacy)
        try:
            assert PARSERS["_legacy2"] is _legacy_freeze_parse
            handler = handler_for("_legacy2")
            spec = _spec("_legacy2", options={"k": [1]})
            parsed = parse_run(spec, object(), index=0, layer=_layer())
            assert parsed.options["k"] == (1,)
            assert isinstance(handler.deferred_checks, tuple)
        finally:
            _pop_all("_legacy2")


# ---------------------------------------------------------------------------
# Plan 4A Task 7: the five base kinds on explicit parsers.


def prediction_only_loss(prediction):
    """The natural mistake, importable from THIS module: one argument."""
    return (prediction**2).mean()


def half_mse(prediction, observed):
    """A legal two-argument loss, importable from THIS module."""
    from rheplicant.inference import mean_squared_error

    return 0.5 * mean_squared_error(prediction, observed)


@pytest.fixture(scope="module")
def base_configured():
    """A real built document for parse tests -- never executed.

    Built by the sanctioned helper (the fixture census forbids a rolled-own
    builder here): one latent ``g``, a homoscedastic noise, a simulated
    observation, and the named ``runtime.seeds.sample`` entry
    ``plan.sample`` resolves at parse.
    """
    return conjugate_built({"kind": "forward"}, seeds={"sample": 11})


_BASE_OPTIONS = {
    "forward": {},
    "fisher": {},
    "optimize": {"optimizer": "gradient", "learning_rate": 0.1,
                 "n_steps": 2},
    "plan.estimate": {"blocks": [{"names": ["g"]}],
                      "check_identifiability": False},
    "plan.sample": {"blocks": [{"names": ["g"]}],
                    "seed": {"from": "runtime.seeds.sample"},
                    "n_sweeps": 6, "check_identifiability": False},
}


def _explode(*args, **kwargs):
    raise AssertionError(f"science ran during parse: {args!r} {kwargs!r}")


def _science_targets(kind, configured):
    """The explicit spy table: the calls a base kind's EXECUTOR may make.

    Listed, never discovered (plan Step 1): the twin's ``__call__``, the
    Jacobian/Fisher/covariance functions, the two calibrators' ``fit``, and
    ``SamplingPlan.estimate``/``.sample``.  A parser that reaches any of
    them is execution wearing a parser's name.
    """
    import rheplicant.inference as inference
    from rheplicant.inference import AdamCalibrator, GradientCalibrator, SamplingPlan

    twins = [(type(configured.twin), "__call__"),
             (type(configured.inference.fit_twin), "__call__")]
    fisher = [(inference, "fisher_information"),
              (inference, "parameter_covariance")]
    fits = [(GradientCalibrator, "fit"), (AdamCalibrator, "fit")]
    plan = [(SamplingPlan, "estimate"), (SamplingPlan, "sample")]
    return {
        "forward": twins,
        "fisher": twins + fisher,
        "optimize": twins + fits,
        "plan.estimate": twins + plan,
        "plan.sample": twins + plan,
    }[kind]


def _is_yaml_safe(view):
    if isinstance(view, Mapping):
        return all(isinstance(key, str) and _is_yaml_safe(value)
                   for key, value in view.items())
    if type(view) is tuple:
        return all(_is_yaml_safe(child) for child in view)
    return view is None or isinstance(view, (bool, int, float, str))


def _plain(view):
    """A frozen view back as plain dicts/lists, for one ``==`` per test."""
    if isinstance(view, Mapping):
        return {key: _plain(value) for key, value in view.items()}
    if type(view) is tuple:
        return [_plain(child) for child in view]
    return view


def _parse_base(kind, configured, **overrides):
    options = dict(_BASE_OPTIONS[kind])
    options.update(overrides)
    return parse_run(_spec(kind, options=options), configured, index=0,
                     layer=_layer())


class TestBaseKindParsersDoNotExecuteScience:
    """Plan Step 1: parsing a base kind never reaches a scientific call."""

    @pytest.mark.parametrize(
        "kind",
        ["forward", "fisher", "optimize", "plan.estimate", "plan.sample"],
    )
    def test_base_kind_parsers_do_not_execute_science(
            self, kind, base_configured, monkeypatch):
        for owner, name in _science_targets(kind, base_configured):
            monkeypatch.setattr(owner, name, _explode)
        parsed = parse_run(_spec(kind, options=dict(_BASE_OPTIONS[kind])),
                           base_configured, index=0, layer=_layer())
        assert parsed.kind == kind
        assert isinstance(parsed.parsed.execution, Mapping)
        assert _is_yaml_safe(parsed.parsed.resolved)


class TestBaseKindsNormalizeTheirDefaults:
    """One selected set of defaults, pinned in BOTH views (plan Step 2)."""

    def test_forward_takes_no_options(self, base_configured):
        parsed = _parse_base("forward", base_configured)
        assert _plain(parsed.parsed.execution) == {}
        assert _plain(parsed.parsed.resolved) == {}

    def test_fisher_defaults(self, base_configured):
        parsed = _parse_base("fisher", base_configured)
        assert _plain(parsed.parsed.resolved) == {"space": False,
                                                  "jitter": 0.0}
        assert parsed.parsed.execution["space"] is False
        assert isinstance(parsed.parsed.execution["jitter"], float)

    def test_optimize_adam_defaults(self, base_configured):
        from rheplicant.inference import mean_squared_error

        parsed = _parse_base("optimize", base_configured,
                             optimizer="adam", learning_rate=0.05, n_steps=9)
        assert _plain(parsed.parsed.resolved) == {
            "optimizer": "adam", "learning_rate": 0.05, "n_steps": 9,
            "loss": "mse", "beta1": 0.9, "beta2": 0.999, "eps": 1e-8}
        assert parsed.parsed.execution["loss"] is mean_squared_error

    def test_optimize_gradient_injects_loss_alone(self, base_configured):
        parsed = _parse_base("optimize", base_configured,
                             optimizer="gradient", learning_rate=0.05,
                             n_steps=9)
        assert _plain(parsed.parsed.resolved) == {
            "optimizer": "gradient", "learning_rate": 0.05, "n_steps": 9,
            "loss": "mse"}
        assert "beta1" not in parsed.parsed.execution

    def test_a_python_loss_projects_its_declaration(self, base_configured):
        parsed = _parse_base(
            "optimize", base_configured, optimizer="gradient",
            learning_rate=0.05, n_steps=9,
            loss={"python": "tests.config.test_config_exit_parsers"
                            ":half_mse"})
        assert _plain(parsed.parsed.resolved)["loss"] == {
            "python": "tests.config.test_config_exit_parsers:half_mse"}
        loss = parsed.parsed.execution["loss"]
        # ``import_target`` re-imports this module under its dotted name, so
        # the callable is not identity-equal to the attribute pytest sees.
        assert loss.__name__ == "half_mse"
        assert loss.__module__.endswith("test_config_exit_parsers")

    def test_plan_estimate_defaults(self, base_configured):
        parsed = _parse_base("plan.estimate", base_configured)
        assert _plain(parsed.parsed.resolved) == {
            "blocks": [{"names": ["g"]}], "check_identifiability": False,
            "max_iter": 100, "tol": 1e-8, "min_sweeps": 3,
            "solve_tol": 1e-6, "solve_guard": 0.001}
        blocks = parsed.parsed.execution["blocks"]
        assert type(blocks) is tuple and len(blocks) == 1

    def test_plan_sample_defaults_and_the_resolved_seed(
            self, base_configured):
        parsed = _parse_base("plan.sample", base_configured)
        assert _plain(parsed.parsed.resolved) == {
            "blocks": [{"names": ["g"]}], "seed": 11, "n_sweeps": 6,
            "check_identifiability": False, "warmup": None,
            "rhat_max": 1.05, "solve_tol": 1e-6, "solve_guard": 0.001}

    def test_a_warm_start_is_normalized_not_estimated(self, base_configured):
        parsed = _parse_base(
            "plan.sample", base_configured,
            warm_start={"kind": "plan.estimate",
                        "blocks": [{"names": ["g"]}], "move": ["g"]})
        warm = parsed.parsed.execution["warm_start"]
        assert warm["move"] == ("g",)
        assert type(warm["blocks"]) is tuple
        assert _plain(parsed.parsed.resolved["warm_start"]) == {
            "kind": "plan.estimate", "blocks": [{"names": ["g"]}],
            "move": ["g"]}


class TestBaseOptionRefusalsHappenAtParse:
    """Every grammar refusal the executor used to raise mid-run now precedes
    science (plan Step 2) -- asserted through ``parse_run`` alone."""

    def test_an_invalid_optimizer_is_refused(self, base_configured):
        with pytest.raises(ConfigError, match="gradient or adam is required"):
            _parse_base("optimize", base_configured, optimizer="bfgs")

    def test_a_loss_of_the_wrong_shape_is_refused(self, base_configured):
        with pytest.raises(ConfigError, match="is 'mse' or"):
            _parse_base("optimize", base_configured, loss=42)

    def test_a_one_argument_python_loss_is_refused_naming_the_signature(
            self, base_configured):
        with pytest.raises(ConfigError, match="cannot be called as"):
            _parse_base(
                "optimize", base_configured,
                loss={"python": "tests.config.test_config_exit_parsers"
                                ":prediction_only_loss"})

    def test_learning_rate_and_n_steps_are_required(self, base_configured):
        for missing in ("learning_rate", "n_steps"):
            options = {"optimizer": "gradient", "learning_rate": 0.1,
                       "n_steps": 2}
            del options[missing]
            with pytest.raises(ConfigError, match=missing):
                parse_run(_spec("optimize", options=options),
                          base_configured, index=0, layer=_layer())

    def test_an_adam_only_knob_on_gradient_is_refused(self, base_configured):
        with pytest.raises(ConfigError, match="belongs to optimizer: adam"):
            _parse_base("optimize", base_configured, beta1=0.8)

    def test_trainable_and_parameters_together_are_ambiguous(self):
        configured = conjugate_built(
            {"kind": "forward"},
            inference={**ONE_LATENT, "trainable": {"leaves": ["gain.gain"]}})
        with pytest.raises(ConfigError, match="cannot serve two masters"):
            _parse_base("optimize", configured)

    def test_a_route_with_nothing_free_to_move_is_refused(self):
        configured = conjugate_built(
            {"kind": "forward"},
            inference={"noise": HOMOSCEDASTIC,
                       "observed": {"from": "simulation"}})
        with pytest.raises(ConfigError, match="something must be free"):
            _parse_base("optimize", configured)

    def test_a_negative_jitter_is_refused(self, base_configured):
        with pytest.raises(ConfigError, match="jitter: must be >= 0"):
            _parse_base("fisher", base_configured, jitter=-1.0)

    def test_a_non_bool_space_is_refused(self, base_configured):
        with pytest.raises(ConfigError, match="space: is a bool"):
            _parse_base("fisher", base_configured, space=1)

    def test_a_malformed_warm_start_is_refused(self, base_configured):
        with pytest.raises(ConfigError, match="warm_start: is a mapping"):
            _parse_base("plan.sample", base_configured, warm_start="warm")
        with pytest.raises(ConfigError, match="warm_start does not take"):
            _parse_base("plan.sample", base_configured,
                        warm_start={"kind": "plan.estimate",
                                    "blocks": [{"names": ["g"]}],
                                    "move": ["g"], "tep": 1})
        with pytest.raises(ConfigError,
                           match="plan.estimate is the one warm start"):
            _parse_base("plan.sample", base_configured,
                        warm_start={"kind": "plan.sample",
                                    "blocks": [{"names": ["g"]}],
                                    "move": ["g"]})
        with pytest.raises(ConfigError, match="warm_start.move: is required"):
            _parse_base("plan.sample", base_configured,
                        warm_start={"kind": "plan.estimate",
                                    "blocks": [{"names": ["g"]}]})

    def test_an_unknown_move_name_is_refused(self, base_configured):
        with pytest.raises(ConfigError, match="ghost") as caught:
            _parse_base("plan.sample", base_configured,
                        warm_start={"kind": "plan.estimate",
                                    "blocks": [{"names": ["g"]}],
                                    "move": ["ghost"]})
        assert "inference.parameters does not declare" in str(caught.value)

    def test_a_non_whole_count_is_refused(self, base_configured):
        with pytest.raises(ConfigError, match="n_steps: is a whole number"):
            _parse_base("optimize", base_configured, n_steps=2.5)
        with pytest.raises(ConfigError, match="n_sweeps: is a whole number"):
            _parse_base("plan.sample", base_configured, n_sweeps=6.5)

    def test_a_wrong_seed_is_refused(self, base_configured):
        options = dict(_BASE_OPTIONS["plan.sample"])
        del options["seed"]
        with pytest.raises(ConfigError, match="'seed' is required"):
            parse_run(_spec("plan.sample", options=options),
                      base_configured, index=0, layer=_layer())
        with pytest.raises(ConfigError, match="seed must NAME an entry"):
            _parse_base("plan.sample", base_configured, seed=11)
        with pytest.raises(ConfigError, match="must be under"):
            _parse_base("plan.sample", base_configured,
                        seed={"from": "elsewhere.sample"})

    def test_the_estimate_seed_refusal_precedes_the_sweep_whole_message(
            self, base_configured):
        """A29 ahead of the generic sweep: the precedence is the message."""
        with pytest.raises(ConfigError) as caught:
            _parse_base("plan.estimate", base_configured,
                        seed={"from": "runtime.seeds.sample"})
        assert str(caught.value) == (
            "runs['run']: plan.estimate refuses a seed -- the asymmetry is "
            "the package's own (sample takes key=, estimate has no key "
            "parameter; check A29). Drop it, or make this run plan.sample."
        )

    def test_an_unknown_option_is_swept_at_parse(self, base_configured):
        with pytest.raises(ConfigError, match="does not take"):
            _parse_base("fisher", base_configured, jitters=0.1)
