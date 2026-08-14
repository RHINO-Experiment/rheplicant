"""The exit registry, the reuse seam, and the ordered execution loop."""

import pytest

from rheplicant.config import ConfigError
from rheplicant.config.sections import exits
from rheplicant.config.sections import runs as runs_module
from rheplicant.config.sections.exit_support import (
    EXECUTORS,
    _binds,
    _number,
    register,
    reuse_of,
)
from rheplicant.config.sections.runs import (
    _KINDS,
    _KINDS_2D,
    RunResult,
    RunSpec,
    parse_runs,
    run_document,
)
from tests.config.test_config_document import synthetic_document


def document(*runs, **extra):
    doc = {**synthetic_document(), **extra}
    doc["runs"] = list(runs) if runs else [{"kind": "forward"}]
    return doc


class TestTheRegistryIsComplete:
    """Importing exits must leave every declared kind runnable.

    The registration is an import side effect, which is exactly the kind of
    wiring that rots in silence.  These two assertions make the silence
    impossible: a kind declared without an executor, or an executor
    registered under a kind no document may name, fails here.
    """

    def test_every_declared_kind_has_an_executor(self):
        assert exits is not None  # importing exits registers the leaf modules
        missing = sorted(set(_KINDS) - set(EXECUTORS))
        assert not missing, f"declared but not registered: {missing}"

    def test_every_executor_is_a_declared_kind(self):
        extra = sorted(set(EXECUTORS) - set(_KINDS))
        assert not extra, f"registered but unreachable from a document: {extra}"

    def test_the_kind_tables_are_pairwise_disjoint(self):
        """A runnable kind must not also sit in a deferral tuple.

        `_one` tests the deferral tuples BEFORE `_KINDS`, so a kind left in a
        deferral tuple after being promoted is refused with "arrives with Plan
        N" no matter that it is declared, registered and unit-tested.  The two
        tests above cannot see this: the registry is complete and every
        executor is declared -- the kind is simply unreachable from any
        document.  Tasks 3-11 each performed that promotion, so this is the
        assertion that makes forgetting half of it loud.

        The tables are DISCOVERED rather than listed.  Task 11 retired
        `_KINDS_2C` outright (`predict` was its last member, and an empty
        tuple would leave `if kind in ()` in the parser), and a test that
        named its tables would have had to be edited to keep importing --
        which is the same edit that would silently drop a table a later plan
        adds.  Discovery covers both directions.
        """
        tables = {name: getattr(runs_module, name) for name in vars(runs_module)
                  if name.startswith("_KINDS")}
        assert "_KINDS" in tables and len(tables) >= 2, sorted(tables)
        assert "_KINDS_2C" not in tables, "predict was its last member"
        names = sorted(tables)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                shared = sorted(set(tables[left]) & set(tables[right]))
                assert not shared, (
                    f"{left} and {right} both claim {shared}; a kind is "
                    "runnable or deferred, never both."
                )

    def test_registering_one_kind_twice_is_a_programming_error(self):
        """Two modules claiming one kind must not silently shadow.

        Tasks 2-11 each register into this one table from their own module;
        a `register` that merely assigned would let the second import win in
        silence, and the surviving executor would depend on import order.
        """

        def probe(run, built, *, results=None):
            return None

        register("_probe_kind")(probe)
        try:
            assert EXECUTORS["_probe_kind"] is probe
            with pytest.raises(AssertionError):
                register("_probe_kind")(probe)
        finally:
            EXECUTORS.pop("_probe_kind", None)

    def test_an_unregistered_kind_refuses_in_the_layers_own_voice(self):
        """A kind in _KINDS with no executor must not escape as a KeyError.

        The config layer's contract is that every user-facing refusal is a
        ConfigError; a bare KeyError from a dict lookup breaks it.
        """
        run = RunSpec(name="x", kind="not_a_kind", variant=None, on="primary",
                      expect="ok", options={})
        with pytest.raises(ConfigError, match="no executor"):
            exits.execute_run(run, None)


class TestReuseIsParsed:
    def test_reuse_lands_on_the_spec_rather_than_in_options(self):
        (run,) = parse_runs([{"kind": "forward", "reuse": "earlier"}])
        assert run.reuse == "earlier"
        assert "reuse" not in run.options

    def test_reuse_defaults_to_none(self):
        (run,) = parse_runs([{"kind": "forward"}])
        assert run.reuse is None

    def test_reuse_is_a_name(self):
        with pytest.raises(ConfigError, match="reuse"):
            parse_runs([{"kind": "forward", "reuse": 7}])


class TestReuseResolution:
    def test_an_unnamed_reuse_is_refused(self):
        """`is required` belongs to this branch alone.

        Matching the bare `reuse:` would also match the missing-name refusal
        below, so deleting this branch outright would leave the suite green
        and the user reading `reuse: None names no earlier run` -- declaration
        order blamed for a key that was never written.
        """
        run = RunSpec(name="b", kind="forward", variant=None, on="primary",
                      expect="ok", options={}, reuse=None)
        with pytest.raises(ConfigError, match="is required"):
            reuse_of(run, {})

    def test_a_reuse_naming_no_earlier_run_is_refused(self):
        run = RunSpec(name="b", kind="forward", variant=None, on="primary",
                      expect="ok", options={}, reuse="a")
        with pytest.raises(ConfigError, match="names no earlier run"):
            reuse_of(run, {})

    def test_a_reuse_naming_a_refused_run_is_refused(self):
        """expect: refuse makes a run's product None -- reusing it is a bug."""
        earlier = RunResult(name="a", kind="fisher", product=None,
                            error=ValueError("boom"))
        run = RunSpec(name="b", kind="forward", variant=None, on="primary",
                      expect="ok", options={}, reuse="a")
        with pytest.raises(ConfigError, match="refused"):
            reuse_of(run, {"a": earlier})

    def test_a_resolved_reuse_hands_back_the_whole_result(self):
        """The RunResult, not its product -- Task 11 reads .kind off it."""
        earlier = RunResult(name="a", kind="fisher", product={"fisher": 1},
                            error=None)
        run = RunSpec(name="b", kind="forward", variant=None, on="primary",
                      expect="ok", options={}, reuse="a")
        assert reuse_of(run, {"a": earlier}) is earlier


class TestTheLoopIsOrdered:
    def test_a_later_run_can_see_an_earlier_ones_result(self):
        """run_document must accumulate, not comprehend.

        The 2B implementation built its dict in one comprehension, so no run
        could observe another.  This pins the accumulation.
        """
        seen = {}

        def spy(run, built, *, results=None):
            seen[run.name] = sorted(results or {})
            return built.state

        original = EXECUTORS["forward"]
        EXECUTORS["forward"] = spy
        try:
            run_document(document({"name": "a", "kind": "forward"},
                                  {"name": "b", "kind": "forward"}))
        finally:
            EXECUTORS["forward"] = original
        assert seen == {"a": [], "b": ["a"]}

    def test_an_executor_cannot_write_into_the_accumulation(self):
        """What an executor sees is a read-only view of the accumulation.

        Handed the live dict, an executor could rewrite an earlier run's
        recorded result or add a key no run declared.  `reuse_of` types the
        parameter as a Mapping; this is what makes that intent true, and ten
        separately-authored tasks are about to consume the seam.
        """
        refused = []

        def vandal(run, built, *, results=None):
            try:
                results["injected"] = "not a run"
            except TypeError as error:
                refused.append(str(error))
            return built.state

        original = EXECUTORS["forward"]
        EXECUTORS["forward"] = vandal
        try:
            out = run_document(document({"name": "a", "kind": "forward"},
                                        {"name": "b", "kind": "forward"}))
        finally:
            EXECUTORS["forward"] = original
        assert len(refused) == 2, f"the write was allowed: {refused}"
        assert set(out) == {"a", "b"}


class TestARunResultCarriesItsVariant:
    """Both results ``execute_run`` builds carry the run's own ``variant:``.

    A later run can only tell whether it is reading its OWN build if the
    result says which build it came from, and ``execute_run`` is the one
    place that knows.  The ``expect: refuse`` leg is a second constructor
    six lines from the first, and nothing downstream reads it today -- which
    is exactly the leg a fix written once and tested through ``predict``
    alone leaves at None forever.

    ``unity_gain`` is ``synthetic_document``'s own variant, so these three
    need no variants: of their own.
    """

    def test_the_ok_path_records_the_variant_it_ran_on(self):
        results = run_document(document({"name": "a", "kind": "forward",
                                         "variant": "unity_gain"}))
        assert results["a"].variant == "unity_gain"

    def test_a_run_declaring_no_variant_records_none(self):
        # Not vacuous: it is the other half of the pair a `variant=None`
        # hard-coded into execute_run would satisfy on its own.
        results = run_document(document({"name": "a", "kind": "forward"}))
        assert results["a"].variant is None

    def test_the_field_defaults_so_a_four_argument_construction_still_binds(
            self):
        """The DEFAULT, which no route through ``execute_run`` can reach.

        Both constructors now pass ``variant=`` explicitly, so the test above
        exercises ``run.variant`` and never the default -- the four
        construction sites in the tests, this one among them, are the only
        callers that rely on it.
        ``is None`` rather than a falsiness check: a default of ``""`` binds
        just as cleanly, is falsey, and would have every such result quietly
        claim it ran on a build named by the empty string.
        """
        result = RunResult(name="a", kind="forward", product=None, error=None)
        assert result.variant is None

    def test_the_expect_refuse_path_records_it_too(self):
        """The refusal is the product here, and it still names its build.

        `fisher` on synthetic_document refuses because the document declares
        no inference.parameters -- which is all this leg needs: a run whose
        RunResult comes back through the OTHER constructor.
        """
        results = run_document(document({"name": "a", "kind": "fisher",
                                         "variant": "unity_gain",
                                         "expect": "refuse"}))
        assert results["a"].product is None
        assert isinstance(results["a"].error, ConfigError)
        assert results["a"].variant == "unity_gain"


class TestTheDeferredKindsNameTheirPlan:
    def test_nuts_and_npe_are_plan_2d(self):
        assert _KINDS_2D == ("nuts", "npe")
        for kind in _KINDS_2D:
            with pytest.raises(ConfigError, match="2D"):
                parse_runs([{"kind": kind}])


class TestACountIsAWholeNumber:
    """``kind=int`` refuses a fractional count instead of truncating it.

    ``int(2.5)`` is 2, so ``n_draws: 2.5`` used to RUN as two draws: the
    document said one thing and the run did another, with nothing anywhere to
    notice.  Two things in this repository already refuse the same value --
    ``transforms._whole``, shipped one task later in the same plan, and the
    package itself (``n_steps must be a positive int``,
    ``tests/inference/test_inference_construction_guards.py:191``) -- so the
    layer held the only permissive reading of three.

    Reached by ``n_draw``, ``n_steps``, ``n_sweeps``, and through ``_knobs``
    by ``maxiter``, ``iterations``, ``min_reweights`` and ``max_reweights``.
    """

    class _Run:
        name = "counted"

    def test_a_fractional_count_is_refused_rather_than_truncated(self):
        with pytest.raises(ConfigError, match="is a whole number") as caught:
            _number(self._Run(), "n_draw", 2.5, kind=int, minimum=1)
        message = str(caught.value)
        assert message.startswith("runs['counted']: n_draw: ")
        # BOTH numbers, because the refusal's whole content is that they are
        # different runs.  A message naming only the declared value leaves the
        # reader to work out what it would silently have become.
        assert "2" in message and "2.5" in message

    def test_a_whole_float_is_refused_too_and_says_the_same_thing(self):
        # 2.0 truncates losslessly, so this leg is the one a "harmless"
        # relaxation would reintroduce -- and it is exactly the value the
        # package refuses for n_steps.  Refusing it keeps one rule rather
        # than a rule with an exception a reader has to remember.
        with pytest.raises(ConfigError, match="is a whole number"):
            _number(self._Run(), "n_steps", 2.0, kind=int)

    def test_an_integer_count_still_passes_through(self):
        assert _number(self._Run(), "n_draw", 3, kind=int, minimum=1) == 3

    def test_kind_float_is_untouched_by_the_rule(self):
        # The guard is on the int branch alone: a tolerance is a real number
        # and 1e-6 must not become a refusal.
        assert _number(self._Run(), "tol", 1e-6, kind=float) == 1e-6
        assert _number(self._Run(), "tol", 1, kind=float) == 1.0


class TestTheSharedCallabilityProbe:
    """``_binds`` is the one place a ``python:`` seam is checked, not three.

    ``kind: gradient``'s ``objective:``, ``kind: mmodes``' ``sky:`` and
    ``optimize``'s ``loss:`` all resolve a hook and then call it with a fixed
    number of arguments.  Each grew its own eight-line inspect/bind/except
    skeleton; this is the shape they share.
    """

    def test_a_callable_of_the_wrong_arity_does_not_bind(self):
        binds, signature = _binds(lambda prediction: prediction, object(),
                                  object())
        assert binds is False
        assert "prediction" in str(signature)

    def test_a_callable_of_the_right_arity_binds(self):
        binds, signature = _binds(lambda a, b: a, object(), object())
        assert binds is True
        assert signature is not None

    def test_defaults_and_star_args_bind_rather_than_being_counted(self):
        # Counting parameters gets all three of these wrong; binding does not.
        for fn in (lambda a, b=1: a, lambda *args: args,
                   lambda a, /, b: a):
            assert _binds(fn, object(), object())[0] is True

    def test_a_callable_inspect_cannot_describe_is_passed_through(self):
        # Some C builtins and some jax wrappers have no retrievable
        # signature.  The call is then its own check, and guessing there
        # would refuse working hooks -- so the verdict is True and the
        # signature is None, which callers must not print.
        binds, signature = _binds(print, object(), object())
        assert binds is True
        assert signature is None or isinstance(signature, object)
