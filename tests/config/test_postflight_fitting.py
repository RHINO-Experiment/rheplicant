"""C12, C13, C19 and the C14 auto-skip -- the three checks a document pays for.

**Every assertion here goes through** ``inflight_helpers.priced_only`` /
``priced_findings`` / ``priced_run``, which reach the POST-FLIGHT pass.
``preflight_helpers.only``/``ids``/``findings`` run the text pass alone and an
assertion written with them passes against an empty implementation.

**The cost budget is this module's tightest constraint** (plan §0.1: no test
over ~4 s, because ``prior_sensitivity`` alone is 2.4-3.2 s cold).  Three
devices keep it:

* the two call-count tests drive **counting stubs**, so they assert the cost
  contract without paying it;
* every C12/C13 test runs on the one- or two-latent document, where the real
  calls are 0.05-0.5 s;
* **exactly one test pays the real cold ``prior_sensitivity`` cost** --
  :meth:`TestC19.test_C19_reports_the_shift_the_priors_caused`, on the
  ONE-latent base document.  Every other C19 test either stands down before
  the call or drives a stub.  Measured with ``--durations=20``.

**The C14 tests are UNIT tests over a DOCTORED build, and that is a
correction to the plan, not a shortcut.**  No document this config layer
accepts can build a complex or a non-floating latent:
``sections/parameters.py:162`` casts every ``init`` to ``context.dtype``,
which ``RuntimeFacts.dtype`` restricts to ``float32``/``float64``, and
``modifiers.DTYPES`` holds no integer name at all.  So the four dtype tests
doctor a built ``ParameterSpace`` directly, and
:meth:`TestC14.test_no_document_this_layer_accepts_builds_an_undifferentiable_latent`
is the end-to-end partner that goes red the day that stops being true.

**Which shape each doctored test uses is stated on the test**, because the
refusal shape depends on it: the space here binds its single latent DIRECTLY
into the float leaf ``gain.gain``, so ``ParameterSpace.validate`` refuses the
dtype KIND mismatch before ``check_linearity``'s own ``_require_inexact``
(``linear.py:517`` runs ``_isolate`` before ``:518``'s
``_require_inexact``; the ``names=`` branch is the second pair at ``:538``
and ``:539``).  Measured, that refusal reads *"Bind for ('g',) produces
complex values for `into` selector 0, but that leaf is float."* -- a sentence
naming neither the check nor the gate.
"""

import dataclasses
import importlib
import types
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.config.document import load_document
from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import REFUSE, REPORT, WARN, ConfigWarning
from rheplicant.config.gating import AUTO_SKIP, CHECK_ID
from rheplicant.config.postflight import CHECKS
from rheplicant.config.postflight import fitting as fitting_module
from rheplicant.inference.identifiability import ParameterSpaceError
from rheplicant.inference.parameters import Latent, ParameterSpace
from rheplicant.inference.uncertainty import StateValidationError
from tests.config import exit_helpers, preflight_helpers
from tests.config.inflight_helpers import (
    priced_findings,
    priced_only,
    priced_run,
)
from tests.config.preflight_helpers import preflight_document, repatch

# --------------------------------------------------------------- documents --

_TWO = exit_helpers.TWO_LATENTS


def _two_latent_document(**inference):
    """``preflight_document(inference=TWO_LATENTS)``, plus an inference patch.

    P8: **the two-latent document is this one** -- latents ``['d', 'a']`` --
    and NOT ``exit_helpers.two_latent_document()``, whose latents are ``dep``
    and ``c``.  The plan's printed margins are this document's.
    """
    return preflight_document(inference={**_TWO, **inference})


def _degenerate_document(**inference):
    """The exactly-degenerate pair, as a WHOLE inference block.

    :func:`repatch` and not :func:`preflight_document`, because a one-level
    merge would leave the base's ``g`` beside ``g1``/``g2`` and the pair would
    no longer be the whole space.
    """
    return repatch(preflight_document(),
                   inference={**preflight_helpers.T4_DEGENERATE_PAIR,
                              **inference})


# ------------------------------------------------------------- doctoring --


def _doctored(payload, dtype, *, name="g"):
    """``payload`` with one latent's dtype swapped and its bindings untouched.

    A fresh :class:`ParameterSpace` built from the real one's parts, in the
    spirit of ``tests/inference/test_identifiability.py``'s own hand-built
    literals -- the plan's ``eqx.tree_at``/``_replace`` claim about those two
    sites is false, they are constructor calls.

    ``ConfiguredRun`` and ``InferenceBuild`` are ``NamedTuple``s
    (``_replace``); ``Priced`` is a frozen dataclass whose ``gates`` proxy
    cannot be deep-copied, so ``dataclasses.replace`` is the only way to vary
    it.
    """
    space = payload.run.inference.space
    old = space.latent(name)
    doctored = ParameterSpace(
        latents=tuple(
            Latent(one.name, init=jnp.asarray(one.init, dtype=dtype),
                   prior=one.prior, linear=one.linear)
            if one.name == name else one
            for one in space.latents),
        bindings=space.bindings,
        joint_prior=space.joint_prior,
    )
    assert doctored.latent(name).init.dtype != old.init.dtype
    run = payload.run._replace(
        inference=payload.run.inference._replace(space=doctored))
    return dataclasses.replace(payload, run=run)


def _regated(payload, **modes):
    """``payload`` with named gates put into ``mode``, defaults elsewhere."""
    gates = {name: gate._replace(state=modes[name]) if name in modes else gate
             for name, gate in payload.gates.items()}
    return dataclasses.replace(payload, gates=gates)


def _found(check, payload):
    """The findings one of this module's three functions produced, as a tuple."""
    return tuple(check(payload))


# ------------------------------------------------------------------ stubs --

#: The three MODULES the package functions live in, fetched through
#: ``sys.modules`` rather than as attributes of ``rheplicant.inference``.
#: ``rheplicant/inference/__init__.py`` re-exports each function under the
#: submodule's own name, so ``rheplicant.inference.identifiability`` **is the
#: function** and ``monkeypatch.setattr("rheplicant.inference.identifiability.
#: identifiability", ...)`` dies as ``AttributeError: 'function' object ...
#: has no attribute 'identifiability'`` -- measured.  ``from a.b import c``
#: reads ``sys.modules['a.b']``, which is what these are.
_LINEAR = importlib.import_module("rheplicant.inference.linear")
_IDENTIFIABILITY = importlib.import_module("rheplicant.inference.identifiability")
_SENSITIVITY = importlib.import_module("rheplicant.inference.sensitivity")


def _linearity_stub(counter, *, raises=None):
    def stub(space, pipeline, state, name=None, **options):
        counter.append(("check_linearity", name, options))
        if raises is not None:
            raise raises
        return {0.001: 0.0, 1.0: 0.0, 1000.0: 0.0}
    return stub


def _identifiability_stub(counter, *, nullity=0, raises=None):
    def stub(space, pipeline, state, **options):
        counter.append(("identifiability", None, options))
        if raises is not None:
            raise raises
        return types.SimpleNamespace(
            n_par=2, n_data=128, rank=2 - nullity, nullity=nullity,
            rtol=1e-8, weakest_identified=0.35,
            participation=lambda index: {"d": 0.5, "a": 0.5})
    return stub


def _sensitivity_stub(counter, *, shift=1e-3, raises=None):
    def stub(space, pipeline, state, observed, noise_std, **options):
        counter.append(("prior_sensitivity", None, options))
        if raises is not None:
            raise raises
        return types.SimpleNamespace(
            n_par=1, worst=("g", 0, shift),
            for_latent=lambda name: {"criterion_std": np.array([0.0303])})
    return stub


def _stub_all(monkeypatch, counter, **options):
    """The three package functions, replaced where THEY live.

    Patched on ``rheplicant.inference.*`` and not on this module, because
    ``fitting.py`` imports all three INSIDE the function that calls them --
    ``rheplicant.config.postflight`` is reached from ``config/document.py``,
    so a module-scope import here would put the whole inference layer in every
    process that reads a config, which
    ``test_config_exits_predict.py::test_importing_the_config_layer_does_not_
    import_numpyro`` and ``test_config_preflight.py::test_the_cold_pass_drags_
    in_no_part_of_the_inference_layer`` both measure.  A function-local
    ``from x import y`` re-reads the attribute on every call, so patching the
    source module is what reaches it.
    """
    monkeypatch.setattr(_LINEAR, "check_linearity",
                        _linearity_stub(counter,
                                        raises=options.get("linearity_raises")))
    monkeypatch.setattr(
        _IDENTIFIABILITY, "identifiability",
        _identifiability_stub(
            counter, nullity=options.get("nullity", 0),
            raises=options.get("identifiability_raises")))
    monkeypatch.setattr(
        _SENSITIVITY, "prior_sensitivity",
        _sensitivity_stub(counter, shift=options.get("shift", 1e-3),
                          raises=options.get("raises")))


def _counts(counter):
    return tuple(sum(1 for one in counter if one[0] == which)
                 for which in ("check_linearity", "identifiability",
                               "prior_sensitivity"))


# ============================================================ the contract ==


class TestTheSlotsThisModuleOwns:
    def test_the_module_owns_C12_C13_and_C19_and_no_other(self):
        """C14 is a FINDING id and never a slot -- ``gating.verdict`` binds it
        on the auto-skip branch, so both ``_identifiability`` and
        ``_prior_sensitivity`` emit it with no second registration.

        **Kills** a fourth ``@register("C14")``, which would double-count in
        ``Report.checks()`` and make ``raise_if_refused``'s tail wrong."""
        mine = {slot for slot, fn in CHECKS.items()
                if fn.__module__ == fitting_module.__name__}
        assert mine == {"C12", "C13", "C19"}
        assert "C14" not in CHECKS

    def test_the_three_slots_are_three_distinct_functions(self):
        """``sweep`` de-duplicates by ``id(fn)``, so one function bound to two
        ids runs ONCE.  These three must not be that: C13 and C19 differ in
        what they call and C12 in which predicate gates it."""
        bound = {slot: CHECKS[slot] for slot in ("C12", "C13", "C19")}
        assert len({id(fn) for fn in bound.values()}) == 3


class TestTheCostContract:
    """Plan §0.1's budget, as a property rather than a timing.

    A timing threshold would be flaky in CI and would not say WHICH check
    ran.  These two drive counting stubs, so they assert the contract without
    paying for it.
    """

    def test_the_call_count_with_no_checks_declared_is_one_per_linear_latent(
            self, monkeypatch):
        """``(len(linear latents), 0, 0)`` on a document declaring **no**
        ``inference.checks:`` at all.

        **Kills** a default table quietly reversed -- ``linearity`` off, or
        either of the other two on -- which no other test in this plan can
        see, and it is this plan's whole cost contract."""
        counter = []
        _stub_all(monkeypatch, counter)
        document = _two_latent_document()
        assert "checks" not in document["inference"]
        priced_findings(document)
        assert _counts(counter) == (2, 0, 0)

    def test_the_call_count_with_all_three_declared_is_n_one_one(
            self, monkeypatch):
        """The anti-vacuity partner: a check-counter that always returns
        ``(n, 0, 0)`` -- three functions that never call anything -- passes
        the test above and fails this one."""
        counter = []
        _stub_all(monkeypatch, counter)
        priced_findings(_two_latent_document(
            checks=preflight_helpers.T4_CHECKS_ALL_REFUSE))
        assert _counts(counter) == (2, 1, 1)

    def test_C12_passes_no_rtol_and_C13_passes_one_only_when_written(
            self, monkeypatch):
        """§0.2 item 7: ``rtol`` is ``identifiability``'s alone.

        ``check_linearity`` HAS an ``rtol=`` and ``inference.checks.linearity``
        cannot express one, so C12 must pass none -- and ``identifiability``'s
        is **keyword-only with a default of 1e-08**, so passing
        ``rtol=gate.rtol`` unconditionally is a ``TypeError`` on every default
        C13 run (D-12, measured: *unsupported operand type(s) for *:
        'NoneType' and 'float'*)."""
        counter = []
        _stub_all(monkeypatch, counter)
        priced_findings(_two_latent_document(
            checks=preflight_helpers.T4_CHECKS_ALL_REFUSE))
        by_name = {one[0]: one[2] for one in counter}
        assert "rtol" not in by_name["check_linearity"]
        assert "rtol" not in by_name["identifiability"]
        assert "rtol" not in by_name["prior_sensitivity"]

        counter.clear()
        priced_findings(_two_latent_document(
            checks=preflight_helpers.T4_CHECKS_IDENTIFIABILITY_RTOL))
        by_name = {one[0]: one[2] for one in counter}
        assert by_name["identifiability"] == {"rtol": 1e-2}

    def test_the_gate_rtol_is_None_for_the_other_two_however_written(self):
        """Pinned on the GATES rather than on the calls, because
        ``gating.gates`` is where a fourth ``rtol`` would be admitted."""
        payload = priced_run(_two_latent_document(
            checks={**preflight_helpers.T4_CHECKS_ALL_REFUSE,
                    **preflight_helpers.T4_CHECKS_IDENTIFIABILITY_RTOL}))
        assert payload.gates["linearity"].rtol is None
        assert payload.gates["prior_sensitivity"].rtol is None
        assert payload.gates["identifiability"].rtol == 1e-2


# ==================================================================== C12 ==


class TestC12:
    def test_C12_runs_once_per_linear_latent(self, monkeypatch):
        """``check_linearity(space, twin, state, name=<one>)``, once per
        latent.

        **Kills** a single call with no ``name=``: measured on this document
        that raises *"Latent 'a' is not declared linear=True"* when only one
        is declared and silently checks ONE latent when several are."""
        counter = []
        _stub_all(monkeypatch, counter)
        priced_findings(_two_latent_document())
        assert [one[1] for one in counter
                if one[0] == "check_linearity"] == ["d", "a"]

    def test_C12_passes_on_the_shipped_document(self):
        """P8: the base document has exactly ONE latent, ``g``, and measured
        its margins are ``{0.001: 0.0, 1.0: 0.0, 1000.0: 0.0}``.

        **Kills** a gate that refuses everything, which every negative test in
        this class would pass."""
        assert list(priced_run(preflight_document())
                    .run.inference.space.names) == ["g"]
        assert "C12" not in {one.check
                             for one in priced_findings(preflight_document())}

    def test_C12_passes_on_the_two_latent_document_and_records_both(self):
        """P8's correction: the ``d`` row is a DIFFERENT fixture from the base
        document's ``g`` row, and **``a`` is NOT 0.0 at 1000x** --
        6.852110354059506e-08 -- so a blanket "every margin is zero" assertion
        is wrong here.  Both latents' margins reach the record, which is what
        says the loop did not stop after the first."""
        found = priced_only(
            _two_latent_document(
                checks=preflight_helpers.T4_CHECKS_LINEARITY_REPORT), "C12")
        assert found.severity == REPORT
        assert "d" in found.message and "a" in found.message
        assert found.where == "inference.parameters"

    def test_C12_refuses_a_nonlinear_latent_declared_linear(self):
        """``NONLINEAR_LATENT``'s ``w`` into ``global_signal.width`` IS
        declared ``linear: True`` and the prediction is not affine in it.

        **Kills** a check whose failure branch is unreachable."""
        found = priced_only(
            preflight_document(inference=exit_helpers.NONLINEAR_LATENT), "C12")
        assert found.severity == REFUSE
        assert found.where == "inference.parameters.w"
        assert "not affine" in found.message
        assert found.message.endswith("(check C12).")

    def test_C12_stands_down_when_the_same_model_makes_no_claim(self):
        """The anti-vacuity partner of the test above: the SAME non-affine
        ``global_signal.width``, with ``linear: true`` taken away.

        **Kills** a check that refuses the physics rather than the claim --
        which would refuse every document whose model is not globally
        linear."""
        found = priced_findings(
            preflight_document(inference=preflight_helpers.NONLINEAR_NOT_DECLARED))
        assert "C12" not in {one.check for one in found}

    def test_C12_stands_down_when_the_document_declares_no_latents(self):
        """``inference.space`` is ``None`` for a document with no
        ``inference:`` section, and ``linearity`` still ``runs()`` by default.

        **Kills** ``space.names`` on ``None`` -- ``post-flight check 'C12'
        RAISED AttributeError``, the laundered-blame shape this layer exists
        to prevent."""
        payload = priced_run(preflight_document(inference=None))
        assert payload.run.inference.space is None
        assert payload.gates["linearity"].runs()
        assert _found(fitting_module._linearity, payload) == ()

    def test_C12_records_its_margins_only_when_the_document_asks(self):
        """§2.3's table, rows 3/6/9: ``report:`` governs the numbers of a
        check that RAN and PASSED.  Without it a pass says nothing at all.

        **Kills** a check that always reports, which would put a C12 finding
        on every document in the suite."""
        quiet = {one.check for one in priced_findings(preflight_document())}
        loud = priced_only(preflight_document(
            inference={"checks": preflight_helpers.T4_CHECKS_LINEARITY_REPORT}),
            "C12")
        assert "C12" not in quiet
        assert loud.severity == REPORT
        assert "0.001" in loud.message


# ==================================================================== C13 ==


class TestC13:
    def test_C13_is_off_until_the_document_turns_it_on(self):
        """``identifiability`` defaults to ``off``, which is not ``skip``: no
        author wrote a reason, so A37 never sees it."""
        payload = priced_run(preflight_document())
        assert payload.gates["identifiability"].state == "off"
        assert not payload.gates["identifiability"].runs()
        assert "C13" not in {one.check
                             for one in priced_findings(preflight_document())}

    def test_C13_reports_the_nullity_it_found(self):
        """Measured on the two-latent document: ``n_par 2``, ``n_data 128``,
        ``rank 2``, ``nullity 0``, ``weakest_identified 0.350504``.

        **Kills** a gate that reads ``report.nullity`` and never compares it,
        and one that reports a number it did not compute -- the nullity and
        the rank both have to come off the report."""
        found = priced_only(
            _two_latent_document(
                checks=preflight_helpers.T4_CHECKS_IDENTIFIABILITY_REPORT),
            "C13")
        assert found.severity == REPORT
        assert "nullity 0" in found.message
        assert "rank 2" in found.message
        assert "2 parameters" in found.message
        assert "128 data" in found.message

    def test_C13_refuses_a_degenerate_pair(self):
        """``gain.gain = g1 + g2`` -- the data constrains the sum and nothing
        else.  Measured: ``rank 1`` of ``n_par 2``, ``nullity 1``,
        ``participation(0) == {'g1': 0.5, 'g2': 0.5}``.

        **Kills** a check that reports a rank it never compared against
        ``n_par``: rank 1 is a perfectly good number and this document is
        still not identified."""
        found = priced_only(
            _degenerate_document(
                checks=preflight_helpers.T4_CHECKS_IDENTIFIABILITY_REFUSE),
            "C13")
        assert found.severity == REFUSE
        assert "nullity 1" in found.message
        assert "g1" in found.message and "g2" in found.message
        assert found.message.endswith("(check C13).")

    def test_C13_does_not_fire_on_the_identified_document_it_shares_a_gate_with(
            self):
        """The anti-vacuity partner: the same gate at the same mode on a
        document that IS identified produces no refusal."""
        found = priced_findings(_two_latent_document(
            checks=preflight_helpers.T4_CHECKS_IDENTIFIABILITY_REFUSE))
        assert [one.check for one in found if one.severity == REFUSE] == []

    def test_C13_uses_the_documents_rtol_when_it_wrote_one(self):
        """``rtol`` is the only per-check option ``check_gates`` admits, and
        only for this check.  The recorded number must be the document's."""
        found = priced_only(
            _two_latent_document(
                checks=preflight_helpers.T4_CHECKS_IDENTIFIABILITY_RTOL),
            "C13")
        assert "1e-02" in found.message or "0.01" in found.message

    def test_C13_runs_at_the_package_default_when_the_document_wrote_none(
            self):
        """D-12: ``identifiability``'s ``rtol`` is keyword-only with a default
        of ``1e-08``, and ``rtol=None`` is a ``TypeError`` -- deliberately
        OUTSIDE this module's except set, so it would escape as ``post-flight
        check 'C13' RAISED TypeError``.

        **Kills** ``identifiability(..., rtol=gate.rtol)`` on every default
        C13 run, which is every C13 run a document does not tune."""
        found = priced_only(
            _two_latent_document(
                checks=preflight_helpers.T4_CHECKS_IDENTIFIABILITY_REPORT),
            "C13")
        assert "1e-08" in found.message

    def test_the_message_says_which_identifiability_key(self):
        """§0.2 item 8.  ``inference.checks.identifiability`` is the gate;
        ``runs[].check_identifiability`` is a ``SamplingPlan`` passthrough
        (four tuples in ``sections/exits.py``, read by
        ``inference/plan.py``'s own refusal) and has nothing to do with this
        check.  Both directions, so a message naming the wrong one is red."""
        found = priced_only(
            _degenerate_document(
                checks=preflight_helpers.T4_CHECKS_IDENTIFIABILITY_REFUSE),
            "C13")
        assert "inference.checks.identifiability" in found.message
        assert "check_identifiability" not in found.message
        assert "runs[" not in found.message


# ==================================================================== C14 ==


class TestC14:
    """The auto-skip, over a DOCTORED build -- see this module's docstring.

    Each test says which shape it doctors.  All four doctor the SAME shape:
    the base document's single latent ``g``, which binds DIRECTLY into the
    float leaf ``gain.gain``.
    """

    @pytest.fixture
    def base(self):
        return priced_run(preflight_document())

    def test_the_two_predicates_are_two(self, base):
        """The trap this task exists for.  ``identifiability`` and
        ``prior_sensitivity`` refuse complex AND non-floating
        (``_check_differentiable``, two distinct messages);
        ``check_linearity`` refuses non-floating ONLY (``_require_inexact``,
        a third).  A task that binds one predicate for all three either loses
        C12 on a legitimate complex latent or turns C13's auto-skip into a
        ``ParameterSpaceError`` out of the package."""
        complexed = _doctored(base, jnp.complex64).run.inference.space
        integered = _doctored(base, jnp.int32).run.inference.space
        assert fitting_module._undifferentiable(complexed) == {"g": "complex64"}
        assert fitting_module._unlinearisable(complexed) == {}
        assert fitting_module._undifferentiable(integered) == {"g": "int32"}
        assert fitting_module._unlinearisable(integered) == {"g": "int32"}

    def test_C14_on_a_complex_latent_auto_skips_identifiability(self, base):
        """Shape: ``g`` complex64, binding straight into ``gain.gain``.

        The gate is at ``mode: refuse``, so a check that RAN and failed would
        produce a C13 REFUSE.  It produces one C14 REPORT instead, and
        ``gating.verdict`` ignores ``report:`` on that branch: a check the
        user asked for and did not get must say so."""
        payload = _regated(_doctored(base, jnp.complex64),
                           identifiability="refuse")
        found = _found(fitting_module._identifiability, payload)
        assert [one.check for one in found] == ["C14"]
        assert found[0].severity == REPORT

    def test_C14_does_not_auto_skip_linearity_on_a_complex_latent(self, base):
        """The twin, in the direction that LOSES a check.

        ``check_linearity`` accepts a complex latent, so C12 does not
        auto-skip here -- it runs, and on this shape ``ParameterSpace``'s own
        bind-time dtype check refuses it first with *"Bind for ('g',)
        produces complex values for `into` selector 0, but that leaf is
        float."*, a ``ParameterSpaceError`` this module re-voices as C12's own
        failure.  So: a C12 finding, never a C14.

        **Kills** ``_undifferentiable`` reused for C12."""
        payload = _regated(_doctored(base, jnp.complex64), linearity="refuse")
        found = _found(fitting_module._linearity, payload)
        assert [one.check for one in found] == ["C12"]
        assert found[0].severity == REFUSE

    def test_C14_auto_skips_all_three_on_an_integer_latent(self, base):
        """The other direction: an ``int32`` latent is undefined for all
        three, so each gate that RUNS emits its own C14.

        **Kills** ``_unlinearisable`` reused for C13 -- under which C12 would
        auto-skip here and C13 would not."""
        payload = _regated(_doctored(base, jnp.int32),
                           linearity="refuse", identifiability="refuse",
                           prior_sensitivity="refuse")
        for check in (fitting_module._linearity,
                      fitting_module._identifiability,
                      fitting_module._prior_sensitivity):
            found = _found(check, payload)
            assert [one.check for one in found] == ["C14"], check.__name__
            assert found[0].severity == REPORT

    def test_C14_s_generated_reason_names_the_latent_and_its_dtype(self, base):
        """**Kills** a generic "a latent is not differentiable", which tells a
        user with forty latents nothing about which one to go and look at."""
        payload = _regated(_doctored(base, jnp.complex64),
                           identifiability="refuse")
        found = _found(fitting_module._identifiability, payload)[0]
        assert "g" in found.message
        assert "complex64" in found.message
        assert found.where == "inference.parameters.g"
        assert found.message.endswith("(check C14).")

    def test_the_reason_distinguishes_complex_from_merely_non_floating(
            self, base):
        """**The trap inside the test itself.**  A complex dtype is ALSO
        non-floating, and ``_check_differentiable``'s non-floating message
        EMBEDS the string ``complex64`` in its ``{name: dtype}`` mapping -- so
        a test matching ``"complex"`` alone is green against no complex branch
        at all.  ``tests/inference/test_identifiability.py`` records that
        exact mutation.  This asserts the two reasons are DIFFERENT
        sentences."""
        complexed = _regated(_doctored(base, jnp.complex64),
                             identifiability="refuse")
        integered = _regated(_doctored(base, jnp.int32),
                             identifiability="refuse")
        first = _found(fitting_module._identifiability, complexed)[0].message
        second = _found(fitting_module._identifiability, integered)[0].message
        assert first != second
        assert "complex64" in first and "int32" not in first
        assert "int32" in second and "complex64" not in second
        # The dtype names alone do NOT discriminate: collapse the two branches
        # into one and both messages still carry their own dtype and still
        # differ.  Measured -- that collapse SURVIVED the assertions above.
        # What has to differ is the EXPLANATION.
        assert "complex coefficients" in first
        assert "complex coefficients" not in second
        assert "not floating-point" in second
        assert "not floating-point" not in first

    def test_C14_advises_no_fix_because_a_generated_auto_skip_has_none(
            self, base):
        """The advice-loop rule, in its one negative form.  A refusal must
        name its escape; an auto-skip has none to give -- there is nothing in
        ``inference.checks:`` for the reader to change, and advising
        ``{mode: skip}`` for a check that already did not run is an advice
        loop with a straight face."""
        payload = _regated(_doctored(base, jnp.complex64),
                           identifiability="refuse")
        message = _found(fitting_module._identifiability, payload)[0].message
        assert "mode:" not in message
        assert "inference.checks.identifiability" in message

    def test_the_auto_skipped_gate_is_local_and_the_mapping_is_untouched(
            self, base):
        """``auto_skipped`` returns a NEW gate; one mapping is handed to every
        check in the pass, so a check that wrote into it would change what a
        later check sees.

        **Kills** an implementation that reaches for ``payload.gates[name] =
        ...`` -- which is a ``TypeError`` today and would be a silent
        cross-check leak if the proxy were ever dropped."""
        payload = _regated(_doctored(base, jnp.int32),
                           identifiability="refuse")
        _found(fitting_module._identifiability, payload)
        assert payload.gates["identifiability"].state == "refuse"
        assert payload.gates["identifiability"].state != AUTO_SKIP

    @pytest.mark.parametrize(
        ("label", "inference"),
        [("the base document", {}),
         ("the two-latent document", dict(_TWO)),
         ("a latent asking for a complex init",
          dict(preflight_helpers.COMPLEX_INIT_LATENT))],
    )
    def test_no_document_this_layer_accepts_builds_an_undifferentiable_latent(
            self, label, inference):
        """The end-to-end partner of the four unit tests above, and the one
        that goes RED the day the config layer admits a complex latent.

        Why it holds today, measured: there is exactly ONE ``Latent(``
        construction site under ``config/`` (``sections/parameters.py:194``)
        and a few lines above it ``init`` is cast to ``context.dtype``
        (``:162``), which ``RuntimeFacts.dtype`` restricts to
        ``float32``/``float64``.  ``dtype: complex64`` on an ``init:`` is
        therefore ACCEPTED and silently cast, with a ``ComplexWarning`` and
        not a raise; ``dtype: int32`` is not writable at all
        (``modifiers.DTYPES`` names four dtypes and no integer); a latent spec
        takes no ``dtype:`` key of its own; and ``transform:`` changes the
        BOUND value, never ``latent.init``.

        The ``ComplexWarning`` the third row earns is swallowed here rather
        than asserted, because it is the cast's evidence and not this test's
        subject -- :meth:`test_a_complex_init_is_cast_rather_than_kept` is
        where it is pinned."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            payload = priced_run(preflight_document(
                **({"inference": inference} if inference else {})))
        space = payload.run.inference.space
        assert fitting_module._undifferentiable(space) == {}, label
        assert fitting_module._unlinearisable(space) == {}, label

    def test_a_complex_init_is_cast_rather_than_kept(self):
        """The measurement every test above rests on, on its own: a latent
        asking for ``complex64`` is ACCEPTED, warned about, and arrives as a
        ``float32``.

        **Kills the premise silently changing.**  If this ever raises instead,
        or ever keeps the complex dtype, C14 becomes reachable from a document
        and the four unit tests above stop being hypothetical."""
        with pytest.warns(Warning, match="[Cc]omplex"):
            payload = priced_run(preflight_document(
                inference=preflight_helpers.COMPLEX_INIT_LATENT))
        assert str(payload.run.inference.space.latent("g").init.dtype) \
            == "float32"


# ==================================================================== C19 ==


class TestC19:
    def test_C19_stands_down_without_observed_data(self):
        """``inference.observed`` is ``None`` while ``inference.space`` is
        not, so the stand-down is C19's own and not the document's.

        **Kills** ``observed.entries[observed.primary]`` on ``None`` --
        ``post-flight check 'C19' RAISED AttributeError: 'NoneType' object has
        no attribute 'entries'``."""
        payload = priced_run(repatch(preflight_document(), inference={
            **preflight_helpers.T4_NO_OBSERVED_INFERENCE,
            "checks": preflight_helpers.T4_CHECKS_PRIOR_SENSITIVITY_REPORT}))
        assert payload.run.inference.observed is None
        assert payload.run.inference.space is not None
        assert payload.gates["prior_sensitivity"].runs()
        assert _found(fitting_module._prior_sensitivity, payload) == ()

    def test_C19_stands_down_on_noise_kind_none(self):
        """``decided_noise`` returns ``None`` for ``kind: none``, and
        ``as_noise_model(None, ...)`` inside the package is a ``TypeError``.

        **Kills** handing ``None`` over and re-voicing the package's
        ``TypeError`` as a document fault -- which would be a refusal naming
        an ``inference.parameters`` line that is perfectly correct."""
        document = preflight_document(inference={
            **preflight_helpers.T4_NOISE_NONE,
            "checks": preflight_helpers.T4_CHECKS_PRIOR_SENSITIVITY_REPORT})
        payload = priced_run(document)
        assert payload.run.inference.noise.kind == "none"
        assert payload.run.inference.observed is not None
        assert payload.gates["prior_sensitivity"].runs()
        assert _found(fitting_module._prior_sensitivity, payload) == ()

    def test_C19_reports_the_shift_the_priors_caused(self):
        """**The one test in this module that pays the real cold
        ``prior_sensitivity`` cost** -- two Newton solves plus a Jacobian plus
        an SVD, on the ONE-latent base document because it is the cheapest
        space that has priors at all.

        Anti-vacuity for every stubbed C19 test above and below: without it
        the whole body of ``_prior_sensitivity`` is only ever driven against a
        ``SimpleNamespace``."""
        found = priced_only(preflight_document(inference={
            "checks": preflight_helpers.T4_CHECKS_PRIOR_SENSITIVITY_REPORT}),
            "C19")
        assert found.severity == REPORT
        assert "sigma" in found.message
        assert "inference.checks.prior_sensitivity" in found.message
        assert found.message.endswith("(check C19).")

    def test_a_package_error_is_re_voiced_and_not_swallowed(self, monkeypatch):
        """``prior_sensitivity``'s own docstring names three ways to a
        ``StateValidationError``; the finding must carry the package's
        sentence AND the gate's escape, so a reader learns both what went
        wrong and what it costs to decline.

        Driven through a stub rather than through a real float32 prediction
        for one reason: the real route costs two Newton solves that do not
        converge, and this module's budget allows exactly one cold C19."""
        counter = []
        _stub_all(monkeypatch, counter, raises=StateValidationError(
            "prior_sensitivity could not find the mode: 100 damped Newton "
            "steps did not converge."))
        found = priced_only(_two_latent_document(
            checks=preflight_helpers.T4_CHECKS_ALL_REFUSE), "C19")
        assert found.severity == REFUSE
        assert "could not find the mode" in found.message
        assert "inference.checks.prior_sensitivity" in found.message
        assert "mode: skip" in found.message

    def test_a_parameter_space_error_is_re_voiced_too(self, monkeypatch):
        """The other half of the except set.  Both are named; neither is
        ``Exception``."""
        counter = []
        _stub_all(monkeypatch, counter, raises=ParameterSpaceError(
            "identifiability() reports rank 1 of 2 at the mode."))
        found = priced_only(_two_latent_document(
            checks=preflight_helpers.T4_CHECKS_ALL_REFUSE), "C19")
        assert found.severity == REFUSE
        assert "reports rank 1 of 2" in found.message

    def test_an_unexpected_exception_is_not_re_voiced_as_a_document_fault(
            self, monkeypatch):
        """**Kills the bare ``except Exception``.**  A ``TypeError`` out of a
        package function is a bug in THIS layer, not in the user's document,
        and must reach ``sweep``'s raise-guard -- which names the slot --
        rather than being dressed up as a refusal of
        ``inference.parameters``."""
        counter = []
        _stub_all(monkeypatch, counter,
                  raises=TypeError("unsupported operand type(s)"))
        with pytest.raises(ConfigError) as raised:
            priced_findings(_two_latent_document(
                checks=preflight_helpers.T4_CHECKS_ALL_REFUSE))
        assert str(raised.value).startswith(
            "post-flight check 'C19' RAISED TypeError: ")

    def test_C19_refuses_a_prior_that_moved_the_mode(self, monkeypatch):
        """The package's own :data:`CRITERION_SHIFT` (0.1 sigma) is the
        threshold, imported rather than restated: a second constant here is a
        second thing to retune.

        **Kills** a check that computes ``shift_sigma`` and never compares
        it."""
        counter = []
        _stub_all(monkeypatch, counter, shift=-1.75)
        found = priced_only(_two_latent_document(
            checks=preflight_helpers.T4_CHECKS_ALL_REFUSE), "C19")
        assert found.severity == REFUSE
        assert "1.75" in found.message

    def test_C19_passes_a_prior_that_did_not(self, monkeypatch):
        """The anti-vacuity partner: the same gate, the same document, a
        shift three decades below the criterion."""
        counter = []
        _stub_all(monkeypatch, counter, shift=-1e-3)
        found = priced_findings(_two_latent_document(
            checks=preflight_helpers.T4_CHECKS_ALL_REFUSE))
        assert [one.check for one in found if one.severity == REFUSE] == []


# ================================================ the gates are independent ==


class TestTheExceptSetIsThreeTimesNarrow:
    """The twin hunt: THREE functions catch, and closing one is not closing
    three.

    Measured -- with only C19's half written, widening C12's ``except`` to
    ``Exception`` SURVIVED the whole module.
    """

    @pytest.mark.parametrize(
        ("slot", "option"),
        [("C12", "linearity_raises"), ("C13", "identifiability_raises"),
         ("C19", "raises")],
    )
    def test_an_unexpected_exception_escapes_from_each_of_the_three(
            self, monkeypatch, slot, option):
        """A ``TypeError`` out of a package function is a bug in THIS layer.
        It must reach ``sweep``'s raise-guard -- which names the slot -- and
        never be dressed up as a refusal of a document line the user wrote
        correctly.

        Run order is ``sorted(CHECKS)`` = C12, C13, C19, so exactly one stub
        is armed per case: with all three armed the first would mask the
        other two."""
        counter = []
        _stub_all(monkeypatch, counter,
                  **{option: TypeError("unsupported operand type(s)")})
        with pytest.raises(ConfigError) as raised:
            priced_findings(_two_latent_document(
                checks=preflight_helpers.T4_CHECKS_ALL_REFUSE))
        assert str(raised.value).startswith(
            f"post-flight check {slot!r} RAISED TypeError: ")

    @pytest.mark.parametrize(
        ("slot", "option"),
        [("C12", "linearity_raises"), ("C13", "identifiability_raises"),
         ("C19", "raises")],
    )
    def test_a_package_error_from_each_of_the_three_becomes_a_finding(
            self, monkeypatch, slot, option):
        """The anti-vacuity partner: the SAME three call sites turn the two
        NAMED exceptions into findings rather than letting them escape."""
        counter = []
        _stub_all(monkeypatch, counter,
                  **{option: ParameterSpaceError("the package's own words")})
        found = priced_only(_two_latent_document(
            checks=preflight_helpers.T4_CHECKS_ALL_REFUSE), slot)
        assert found.severity == REFUSE
        assert "the package's own words" in found.message


class TestNoGateSilencesAnother:
    def test_no_gate_silences_another(self):
        """``identifiability: {mode: skip, reason: "..."}`` and **C12 still
        fires**.

        **Kills** a shared early return -- one ``if not gate.runs(): return``
        covering all three, which is the single most likely way three
        functions in one module come to share a stand-down."""
        found = priced_only(preflight_document(
            inference={**exit_helpers.NONLINEAR_LATENT,
                       "checks": preflight_helpers.T4_CHECKS_IDENTIFIABILITY_SKIP}),
            "C12")
        assert found.severity == REFUSE
        assert "C13" not in {one.check for one in priced_findings(
            preflight_document(inference={
                **exit_helpers.NONLINEAR_LATENT,
                "checks": preflight_helpers.T4_CHECKS_IDENTIFIABILITY_SKIP}))}

    def test_a_C14_from_one_gate_does_not_speak_for_another(self, monkeypatch):
        """Two priced checks cannot communicate through ``payload.run.report``
        (``sweep`` accumulates into a local list), so a C14 emitted by C13
        must not be read by C19 -- each recomputes its own predicate."""
        payload = _regated(
            _doctored(priced_run(preflight_document()), jnp.int32),
            identifiability="refuse", prior_sensitivity="refuse")
        assert [one.check for one in
                _found(fitting_module._identifiability, payload)] == ["C14"]
        assert [one.check for one in
                _found(fitting_module._prior_sensitivity, payload)] == ["C14"]


# ========================================================== the advice loop ==


class TestTheAdviceLoop:
    """Take a document this module refuses, apply the refusal's own advice
    LITERALLY, and assert that (i) the id is gone or downgraded and (ii)
    ``load_document`` succeeds.

    In wave 1 a refusal named two escapes and the second produced a SECOND
    refusal; the reviewer's own prescribed fix was also an advice loop.  Only
    the round that RAN it found the bug.
    """

    REFUSED = dict(exit_helpers.NONLINEAR_LATENT)

    def test_the_refusal_names_both_escapes(self):
        found = priced_only(preflight_document(inference=self.REFUSED), "C12")
        assert "inference.checks.linearity" in found.message
        assert "mode: warn" in found.message
        assert 'mode: skip, reason: "..."' in found.message

    def test_applying_mode_warn_loads_and_downgrades_to_a_warning(self):
        """(i) the severity becomes WARN, (ii) ``load_document`` returns."""
        document = preflight_document(inference={
            **self.REFUSED,
            "checks": preflight_helpers.T4_CHECKS_LINEARITY_WARN})
        assert priced_only(document, "C12").severity == WARN
        with pytest.warns(ConfigWarning, match="not affine"):
            run = load_document(document)
        assert [one.check for one in run.report.findings
                if one.severity == WARN] == ["C12"]

    def test_applying_mode_skip_loads_and_removes_the_finding(self):
        """(i) C12 is gone from the report entirely, (ii) ``load_document``
        returns -- and no SECOND refusal arrives in its place, which is the
        wave-1 failure this rule exists for.  In particular ``mode: skip``
        with a written ``reason:`` satisfies A37, and the pair is not the
        ``{mode: skip, report: true}`` shape Task 2 refuses."""
        document = preflight_document(inference={
            **self.REFUSED,
            "checks": preflight_helpers.T4_CHECKS_LINEARITY_SKIP})
        run = load_document(document)
        assert "C12" not in {one.check for one in run.report.findings}
        assert run.report.refusals() == ()

    def test_the_unadvised_document_really_is_refused(self):
        """The anti-vacuity partner of both: without the advice,
        ``load_document`` raises."""
        with pytest.raises(ConfigError, match="not affine"):
            load_document(preflight_document(inference=self.REFUSED))


# ====================================================== message discipline ==


class TestMessageDiscipline:
    def test_the_tag_is_appended_and_never_doubled(self):
        """``_tagged`` on a sentence that ALREADY ends in its tail must leave
        it alone.  Several messages this layer ships already end in their own
        tag, and a doubled tail is what a reader sees first.

        A unit test, because no message this module composes today ends in
        its tag before the append -- so the guard is unreachable from a
        document, and an unconditional append SURVIVED every other test
        here."""
        assert fitting_module._tagged("C12", "x") == "x (check C12)."
        assert fitting_module._tagged("C12", "x (check C12).") \
            == "x (check C12)."
        assert fitting_module._tagged("C13", "x (check C12).") \
            == "x (check C12). (check C13)."

    def test_every_finding_this_module_emits_ends_with_its_check_tag(self):
        """``Finding``'s own docstring: a message "ends with ``(check A30).``
        when ``check`` is set", and the tail is APPENDED rather than woven in,
        so a doubled tag is a defect."""
        seen = []
        seen.append(priced_only(
            preflight_document(inference=exit_helpers.NONLINEAR_LATENT), "C12"))
        seen.append(priced_only(
            _degenerate_document(
                checks=preflight_helpers.T4_CHECKS_IDENTIFIABILITY_REFUSE),
            "C13"))
        base = priced_run(preflight_document())
        seen.extend(_found(fitting_module._identifiability,
                           _regated(_doctored(base, jnp.complex64),
                                    identifiability="refuse")))
        for one in seen:
            assert one.message.endswith(f"(check {one.check}).")
            assert one.message.count(f"(check {one.check}).") == 1

    def test_every_gated_message_names_its_gate_and_its_escape(self):
        """§3.2(i).  C14 is the deliberate exception and has its own test."""
        for found, gate in (
            (priced_only(preflight_document(
                inference=exit_helpers.NONLINEAR_LATENT), "C12"), "linearity"),
            (priced_only(_degenerate_document(
                checks=preflight_helpers.T4_CHECKS_IDENTIFIABILITY_REFUSE),
                "C13"), "identifiability"),
        ):
            assert f"inference.checks.{gate}" in found.message
            assert "mode: warn" in found.message
            assert 'mode: skip, reason: "..."' in found.message

    def test_C13_and_C19_say_they_were_off_by_default(self):
        """§3.2(i)'s third clause, for the two checks that default to ``off``
        -- a reader has to learn that this document ASKED for the cost."""
        found = priced_only(
            _degenerate_document(
                checks=preflight_helpers.T4_CHECKS_IDENTIFIABILITY_REFUSE),
            "C13")
        assert "off by default" in found.message

    def test_C12_does_not_claim_to_be_off_by_default(self):
        """It is the one check that is ON by default (``gating.DEFAULT_MODE``),
        so the clause above must not be shared by all three."""
        found = priced_only(
            preflight_document(inference=exit_helpers.NONLINEAR_LATENT), "C12")
        assert "off by default" not in found.message

    def test_every_finding_carries_the_id_its_gate_is_bound_to(self):
        """``gating.CHECK_ID`` is the one binding of check name to schema id;
        a second table here would be a second thing to keep in step."""
        assert CHECK_ID == {"linearity": "C12", "identifiability": "C13",
                            "prior_sensitivity": "C19"}
