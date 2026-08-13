"""The exit registry, the reuse seam, and the ordered execution loop."""

import pytest

from rheplicant.config import ConfigError
from rheplicant.config.sections import exits
from rheplicant.config.sections.exit_support import EXECUTORS, register, reuse_of
from rheplicant.config.sections.runs import (
    _KINDS,
    _KINDS_2C,
    _KINDS_2D,
    _KINDS_PLAN4,
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

        `_one` tests the deferral tuples BEFORE `_KINDS`, so a kind left in
        `_KINDS_2C` after being promoted is refused with "arrives with Plan
        2C" no matter that it is declared, registered and unit-tested.  The
        two tests above cannot see this: the registry is complete and every
        executor is declared -- the kind is simply unreachable from any
        document.  Tasks 3-11 each perform that promotion, so this is the
        assertion that makes forgetting half of it loud.
        """
        tables = {"_KINDS": _KINDS, "_KINDS_2C": _KINDS_2C,
                  "_KINDS_2D": _KINDS_2D, "_KINDS_PLAN4": _KINDS_PLAN4}
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


class TestTheDeferredKindsNameTheirPlan:
    def test_nuts_and_npe_are_plan_2d(self):
        assert _KINDS_2D == ("nuts", "npe")
        for kind in _KINDS_2D:
            with pytest.raises(ConfigError, match="2D"):
                parse_runs([{"kind": kind}])
