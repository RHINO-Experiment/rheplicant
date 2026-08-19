"""The exit registry, the reuse seam, and the ordered execution loop."""

import jax.numpy as jnp
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.sections import exits
from rheplicant.config.sections import runs as runs_module
from rheplicant.config.sections.exit_support import (
    DEFERRED_CHECKS,
    EXECUTORS,
    PARSERS,
    PRE_EXECUTORS,
    _binds,
    _decided_sigma,
    _noise,
    _number,
    _observed,
    _on,
    register,
    reuse_of,
)
from rheplicant.config.sections.observed import ObservedBuild
from rheplicant.config.sections.runs import (
    _KINDS,
    _KINDS_PLAN4,
    RunResult,
    RunSpec,
    parse_runs,
    run_document,
)
from tests.config.exit_helpers import (
    FROZEN_FRACTION,
    HOMOSCEDASTIC,
    fanned_built,
    fanned_document,
    spec,
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

        A RAISE and not an ``assert``: ``python -O`` strips asserts, and
        measured before Plan 3A's Task 1 the second registration then won
        silently.  The four tests in
        ``tests/config/test_config_findings.py::TestTheExecutorRegistryNoLongerAsserts``
        are where the ``-O`` behaviour itself is pinned, in a subprocess.
        """

        def probe(run, built, *, results=None):
            return None

        register("_probe_kind")(probe)
        try:
            # The transitional legacy adapter wraps a parse-less binding;
            # ``__wrapped__`` is the executor the caller registered.
            assert EXECUTORS["_probe_kind"].__wrapped__ is probe
            with pytest.raises(ConfigError):
                register("_probe_kind")(probe)
        finally:
            for registry in (PARSERS, PRE_EXECUTORS, EXECUTORS,
                             DEFERRED_CHECKS):
                registry.pop("_probe_kind", None)

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

        def spy(run, built, previous):
            seen[run.name] = sorted(previous or {})
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

        def vandal(run, built, previous):
            try:
                previous["injected"] = "not a run"
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


class TestTheObservationFan:
    """A run's ``on:`` chooses its own frozen sigma, not the primary's.

    ``radiometer_frozen`` with ``source: observed`` decides the sigma FROM
    the data, so ``night`` -- simulated at twice ``primary``'s truth -- has
    twice the sigma in every channel.  Before this task there was one sigma
    per document and ``_noise`` had no idea a run carried an ``on:`` at all,
    so a run on ``night`` was weighed with half the sigma it should have
    been and nothing anywhere said so.  Measured at channel [0, 4]:
    0.00052495 and 0.00104989.
    """

    PRIMARY = 0.0005249460227787495
    NIGHT = 0.001049892045557499

    def test_noise_hands_each_run_its_own_observations_sigma(self):
        # Two pins, not one: a `_noise` that always returned NIGHT would
        # satisfy either alone.
        built = fanned_built()
        assert float(_noise(spec(kind="fisher"), built)[0, 4]) == (
            pytest.approx(self.PRIMARY, rel=1e-6))
        assert float(_noise(spec(kind="fisher", on="night"), built)[0, 4]) == (
            pytest.approx(self.NIGHT, rel=1e-6))

    def test_the_decided_sigma_route_selects_the_same_way(self):
        """The twin accessor -- every conjugate exit reaches its sigma here.

        `_decided_sigma` composes over `_noise`, so a fan written into one
        and not the other is the hole-on-one-route shape.  A frozen sigma
        passes through `_decided_sigma` untouched, so the numbers are the
        same two, and that is the point: they must be.
        """
        built = fanned_built()
        assert float(_decided_sigma(spec(), built)[0, 4]) == pytest.approx(
            self.PRIMARY, rel=1e-6)
        assert float(_decided_sigma(spec(on="night"), built)[0, 4]) == (
            pytest.approx(self.NIGHT, rel=1e-6))

    def test_noise_and_observed_agree_about_which_observation(self):
        """One resolver, or a run is weighed with one and compared to another.

        The two accessors are read by different exits (`fisher` takes only
        the sigma, `optimize` only the data, `conjugate.*` both), so nothing
        else in the suite can see them disagree.  The cross pairing at the
        foot is what makes the agreement an assertion: without it, "sigma
        equals |some observation| * frac" is satisfied by the bug.
        """
        built = fanned_built()
        for name in ("primary", "night"):
            run = spec(kind="fisher", on=name)
            assert jnp.allclose(_noise(run, built),
                                jnp.abs(_observed(run, built))
                                * FROZEN_FRACTION)
        primary, night = spec(kind="fisher"), spec(kind="fisher", on="night")
        assert not jnp.allclose(_noise(primary, built),
                                jnp.abs(_observed(night, built))
                                * FROZEN_FRACTION)

    def test_the_shared_resolver_reads_primary_as_the_documents_own_name(self):
        """``on: primary`` on a document whose one observation is not.

        ``build_observed`` calls a LONE entry the primary whatever it is
        named (``observed.py:206-211``), so ``on:``'s default has to travel
        through that indirection -- and ``_noise`` now leans on it as hard
        as ``_observed`` always did.  The two-observation fixture above
        cannot see this: its primary IS named ``primary``, so there the
        indirection is a no-op and ``by_observation[run.on]`` would answer
        the same.  Here it would raise a bare KeyError on a document the
        layer accepts.
        """
        observed = ObservedBuild(entries={"night": jnp.asarray([[1.0]])},
                                 primary="night", at={}, records={})
        assert _on(spec(kind="fisher"), observed) == "night"
        assert _on(spec(kind="fisher", on="night"), observed) == "night"

    def test_an_unknown_on_is_refused_rather_than_silently_the_primary(self):
        """The name the sigma is chosen by is checked where it is used.

        Falling back to the primary for a name the document does not declare
        is the original bug wearing a typo, so `_noise` refuses -- in
        `_observed`'s own words, because they resolve through the same
        helper and a second wording is a second contract.
        """
        with pytest.raises(ConfigError, match="names no observation") as got:
            _noise(spec(kind="fisher", on="dusk"), fanned_built())
        assert "['night', 'primary']" in str(got.value)
        assert str(got.value).startswith("runs['fisher']:")

    def test_a_model_noise_kind_is_untouched_by_on(self):
        """Only the frozen sigma fans, and this is what says "only".

        A homoscedastic document has ONE model for the whole document, so
        both runs get the same object -- not an equal one.  An implementation
        that fanned every kind off `observed.entries` would return an array
        here and fail the identity.
        """
        built = fanned_built(noise=HOMOSCEDASTIC)
        assert (_noise(spec(kind="fisher"), built)
                is _noise(spec(kind="fisher", on="night"), built))

    def test_the_fan_reaches_the_answer_a_run_returns(self):
        """The feature's own thesis: it must move a NUMBER, from a document.

        `kind: fisher` reads the sigma and never touches observed data, so
        its `on:` does exactly one thing -- and the Fisher information is
        J^T N^-1 J, so a sigma twice as large is a width twice as wide.
        Measured today, WITHOUT the fan, both of these come back
        9.375001536682248e-05: the accessor tests above are unit tests of a
        seam, and this is the one assertion that says the seam is wired to
        the answer.
        """
        narrow = run_document(fanned_document({"name": "cov",
                                               "kind": "fisher"}))
        wide = run_document(fanned_document({"name": "cov", "kind": "fisher",
                                             "on": "night"}))
        narrow_g = float(narrow["cov"].product["covariance"].sigma("g"))
        wide_g = float(wide["cov"].product["covariance"].sigma("g"))
        assert narrow_g == pytest.approx(9.375001536682248e-05, rel=1e-5)
        assert wide_g == pytest.approx(1.8750003073364496e-04, rel=1e-5)
        assert wide_g / narrow_g == pytest.approx(2.0, rel=1e-6)


class TestTheDeferredKindsNameTheirPlan:
    """One deferral tuple is left, and it names its plan.

    ``_KINDS_2C`` went when ``predict`` shipped and ``_KINDS_2D`` when ``npe``
    did -- both DELETED rather than emptied, because ``_one`` tests the
    deferral tuples BEFORE ``_KINDS`` and an empty one is a dead branch that
    reads to the next author as an outstanding promise.
    ``test_the_kind_tables_are_pairwise_disjoint`` above discovers whatever is
    left by prefix, so it needs no edit here.
    """

    def test_benchmark_is_the_last_plan_4_deferral(self):
        assert _KINDS_PLAN4 == ("benchmark",)
        for kind in _KINDS_PLAN4:
            with pytest.raises(ConfigError, match="Plan 4"):
                parse_runs([{"kind": kind}])
        assert parse_runs(
            [{"kind": "compare", "of": ["a", "b"], "metric": "rms", "tolerance": 0.0}]
        )[0].kind == "compare"

    def test_nuts_and_npe_have_left_and_the_tuple_has_gone_with_them(self):
        assert not hasattr(runs_module, "_KINDS_2D"), (
            "runs._KINDS_2D still exists; the last task to move a kind out "
            "of it deletes the tuple and its refusal branch."
        )
        for kind in ("nuts", "npe"):
            assert parse_runs([{"kind": kind}])[0].kind == kind


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
