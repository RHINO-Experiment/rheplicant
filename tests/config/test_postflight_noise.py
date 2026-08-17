"""C18, the numeric half -- the sigma that drew the data and the sigma that
weighs it, made to agree.

``tests/config/test_preflight_gated.py`` is ``C18.kind``'s file: the two-word
family check, decided from the document's text alone, before the beam is
read. This is the OTHER slot sharing the same bare id -- the numeric
comparison, decided from the built twin and the built likelihood, which is
why it lives in ``postflight/`` and is reached through
:func:`~tests.config.inflight_helpers.priced_findings` /
:func:`~tests.config.inflight_helpers.priced_only` rather than through
``preflight_helpers.findings`` -- that helper calls the TEXT pass alone and
would make every assertion below pass against an empty implementation.

**Documents are built with** :func:`~tests.config.preflight_helpers.repatch`
**on top of** :func:`~tests.config.preflight_helpers.preflight_document`,
**never assembled from scratch.** The base document already declares
``from: simulation`` with ``twin: full`` -- the two conditions this check
needs before there is a second sigma to compare at all -- and delegates
through ``exit_helpers.conjugate_document``, so ``tests/config/
test_config_fixture_contract.py``'s census holds this module to the same
standard as every other.
"""

import pytest

from rheplicant.config.document import load_document
from rheplicant.config.errors import ConfigError
from rheplicant.config.postflight.noise import _T6_RTOL, _t6_drawn
from rheplicant.config.preflight.gated import _t2c_generated
from rheplicant.inference import RadiometerNoise
from tests.config.exit_helpers import HOMOSCEDASTIC, RADIOMETER
from tests.config.inflight_helpers import priced_findings, priced_only, priced_run
from tests.config.preflight_helpers import (
    FLOOR_DECLARED_K,
    RADIOMETER_DRAWN,
    RADIOMETER_NODE,
    SIGMA_BROADCAST_FREQ,
    SIGMA_MISMATCHED_K,
    T6_FROZEN_HUNDRED_FOLD,
    T6_RADIOMETER_TEN_FOLD,
    T6_RADIOMETER_VALUES_SWAPPED,
    preflight_document,
    repatch,
)
from tests.config.preflight_helpers import findings as text_findings
from tests.config.test_config_fixture_contract import _builders


#: The base document, patched with a ``model.noise`` and an ``inference.noise``
#: -- the one shape every test in this file needs.  Not named ``_document``:
#: ``test_config_fixture_contract._factories`` discovers ``*_document``
#: functions as builders the property walk drives with no argument, and this
#: one takes two.
def _pair(model_noise, inference_noise, **more):
    base = preflight_document()
    return repatch(base, model={**base["model"], "noise": model_noise},
                   inference={**base["inference"], "noise": inference_noise},
                   **more)


class TestTheRadiometerFamily:
    """Kills: a field-by-field comparison instead of ``.fractional``; a
    formula re-derived instead of read off the object; the ``floor`` folded
    into the comparison."""

    def test_the_agreeing_radiometer_pair_is_silent(self):
        """Both sides ``0.0007071067811865475`` -- measured, not assumed."""
        document = _pair(RADIOMETER_DRAWN, RADIOMETER)
        assert priced_findings(document) == ()

    def test_a_ten_fold_disagreement_refuses(self):
        """The measured document: ``(1 MHz, 2 s)`` against ``(100 MHz, 2 s)``."""
        document = _pair(RADIOMETER_DRAWN, T6_RADIOMETER_TEN_FOLD)
        found = priced_only(document, "C18")
        assert found.where == "model.noise"
        assert found.severity == "refuse"

    def test_swapped_width_and_time_agree(self):
        """``(1 MHz, 2 s)`` against ``(2 MHz, 1 s)`` -- VALUES swapped, UNITS
        kept.  **Kills** the field-by-field comparison, and it is the single
        most likely wrong implementation of this task: a validator that reads
        ``channel_width`` against ``channel_width`` and ``integration_time``
        against ``integration_time`` refuses this document, and the two
        products (``1e6 * 2`` and ``2e6 * 1``) are exactly equal."""
        document = _pair(RADIOMETER_DRAWN, T6_RADIOMETER_VALUES_SWAPPED)
        assert priced_findings(document) == ()

    def test_the_frozen_kind_is_compared_too(self):
        """``radiometer_frozen`` reads its numbers from ``NoiseBuild.frozen``
        and not from a ``model:``. **Kills** a check that only handles
        ``kind: radiometer``, under which the frozen route -- measured
        accepted today with a hundred-fold ``channel_width`` disagreement --
        stays open. **This is the twin of the test above.**"""
        document = _pair(RADIOMETER_DRAWN, T6_FROZEN_HUNDRED_FOLD)
        found = priced_only(document, "C18")
        assert found.severity == "refuse"
        assert "radiometer_frozen" in found.message or "kind: radiometer_frozen" in found.message

    def test_a_declared_floor_does_not_count_as_a_disagreement(self):
        """A floor on the likelihood side does not make an agreeing pair
        disagree. **Kills** folding ``floor`` into the comparison."""
        weighed = {**RADIOMETER, "floor": FLOOR_DECLARED_K}
        document = _pair(RADIOMETER_DRAWN, weighed)
        assert priced_findings(document) == ()

    def test_a_disagreement_with_a_declared_floor_names_it(self):
        """The partner above: a REAL disagreement's message still mentions
        the declared floor, so a reader is not sent looking for a difference
        this check chose to ignore."""
        weighed = {**T6_RADIOMETER_TEN_FOLD, "floor": FLOOR_DECLARED_K}
        document = _pair(RADIOMETER_DRAWN, weighed)
        found = priced_only(document, "C18")
        assert "floor" in found.message
        assert "0.01" in found.message

    def test_the_tolerance_admits_the_same_number_in_two_units_radiometer_family_only(
            self):
        """``1 MHz`` against ``1000 kHz`` -- radiometer family only (D-21):
        the homoscedastic route's only second spelling, ``celsius``, is
        measured to agree just as exactly and is not worth a document of its
        own -- a sigma is a scatter, not a temperature, and a celsius spelling
        for one is not a shape this suite ships a test for.

        **Kills** an exact ``==`` on the compared quantity, which unit
        conversion through float32 modifiers would occasionally miss even
        though this exact pair measures ``rel = 0.0`` (see ``_T6_RTOL``'s own
        comment)."""
        weighed = {"kind": "radiometer", "include_logdet": True,
                   "channel_width": {"value": 1000.0, "unit": "kHz"},
                   "integration_time": {"value": 2.0, "unit": "s"}}
        document = _pair(RADIOMETER_DRAWN, weighed)
        assert priced_findings(document) == ()


class TestTheHomoscedasticFamily:
    """Kills: scoping the whole check to the radiometer family, which §6's
    wording invites; an ``is``/``==`` on arrays, which raises or returns an
    array where a bool is wanted."""

    def test_the_agreeing_homoscedastic_pair_is_silent(self):
        """The base document itself, after D-10's fixture repair: both sides
        ``SIGMA_K`` = 0.05 K."""
        assert priced_findings(preflight_document()) == ()

    def test_the_homoscedastic_pair_is_compared(self):
        """A document that patches a disagreement in DELIBERATELY --
        ``model.noise.sigma: 0.5 K`` against the likelihood's ``SIGMA_K`` =
        0.05 K.  Measured accepted today (before this check exists).  Its
        partner above is the base document, which D-10's fixture repair made
        AGREE -- a test written on the base alone would be green against a
        check with no homoscedastic branch at all."""
        document = _pair(SIGMA_MISMATCHED_K, HOMOSCEDASTIC)
        found = priced_only(document, "C18")
        assert found.severity == "refuse"

    def test_a_broadcastable_homoscedastic_sigma_agrees(self):
        """A ``(1, n_freq)`` array, every entry the operator's own scalar.
        **Kills** an ``is``/``==`` comparison: ``==`` between a 0-d scalar
        and an ``(1, n_freq)`` array returns an ARRAY of booleans, and
        ``if`` on more than one element raises
        ``ValueError: The truth value of an array...`` -- exactly the
        failure ``jnp.allclose`` avoids."""
        document = _pair({"type": "NoiseOperator",
                          "sigma": {"value": HOMOSCEDASTIC["sigma"]["value"],
                                   "unit": "K"}},
                         SIGMA_BROADCAST_FREQ)
        assert priced_findings(document) == ()


class TestStandsDown:
    """Kills: the stand-down rule collapsing to ``twin == "full"`` alone
    (D-10); the stand-down rule replaced by a text-only reading that cannot
    see the built twin."""

    def test_stands_down_on_observed_from_file(self, tmp_path):
        """The data came from a file; the twin drew nothing; there is no
        second sigma to disagree with."""
        import numpy as np

        path = tmp_path / "d.npy"
        np.save(path, np.zeros((16, 8), dtype=np.float32))
        base = preflight_document()
        document = repatch(
            base,
            model={**base["model"], "noise": SIGMA_MISMATCHED_K},
            inference={**base["inference"],
                      "observed": {"file": {"format": "npy", "path": str(path)}}})
        assert priced_findings(document) == ()

    def test_stands_down_when_the_model_lights_no_noise(self):
        """The real base-minus-noise document (D-10/P10(d)): ``preflight_
        document()`` always lights ``noise`` (measured equal to
        ``STOCHASTIC_MODEL``), so "the base document" is
        ``repatch(preflight_document(), model=<BASE_MODEL minus noise>,
        inference=<inference minus twin>)`` -- and dropping
        ``inference.twin.without: [noise]`` at the SAME time is mandatory, or
        it is an ``AssemblyError`` (there is nothing left for ``without:`` to
        remove)."""
        base = preflight_document()
        model_without_noise = {key: value for key, value in base["model"].items()
                               if key != "noise"}
        inference_without_twin = {key: value for key, value in base["inference"].items()
                                  if key != "twin"}
        document = repatch(base, model=model_without_noise,
                           inference=inference_without_twin)
        assert "noise" not in load_document(document).twin.lit
        assert priced_findings(document) == ()

    def test_does_not_restate_the_family_mismatch(self):
        """A cross-family document -- ``model.noise`` draws with
        ``RadiometerNoiseOperator``, ``inference.noise`` weighs with
        ``homoscedastic`` -- is already refused by ``C18.kind`` (Task 2,
        pre-flight), which halts ``load_document`` before this check ever
        runs. **Kills** this check registering into the wrong pass (the
        shared-name trap ``postflight/__init__.py`` documents: importing
        ``preflight.register`` instead of ``postflight.register`` binds a
        check into the TEXT pass, where it would fire ALONGSIDE ``C18.kind``
        on this exact document) -- which would double-count and make
        ``raise_if_refused``'s tail wrong. ``Report.checks()`` is a
        frozenset and cannot see that: this counts ``report.findings``
        (D-10's own correction)."""
        base = preflight_document()
        document = repatch(base, model={**base["model"], "noise": RADIOMETER_DRAWN},
                           inference={**base["inference"], "noise": HOMOSCEDASTIC})
        found = text_findings(document)
        assert [one.check for one in found].count("C18") == 1
        with pytest.raises(ConfigError):
            load_document(document)


class TestATwinFitFitTwinCanStillBeTheGenerator:
    """MAJOR 5's own document, this check's version of it: with ZERO declared
    latents, ``refuse_stochastic_stages`` never runs -- it fires only while
    building a ``ParameterSpace`` over a declared latent -- so a document CAN
    spell ``observed.twin: fit`` with no ``inference.twin:`` repair at all,
    and its fit twin is then the SAME OBJECT as the full twin: still carrying
    the draw. **Kills** the stand-down rule collapsed to ``records[primary]
    ["twin"] == "full"`` alone (the plan's own weaker, earlier form, D-10) --
    under which this document's disagreement goes uncaught because nothing
    ever reads the fit twin."""

    def test_the_unrepaired_fit_twin_is_still_the_generator(self):
        base = preflight_document()
        inference = dict(base["inference"])
        inference.pop("twin", None)
        inference.pop("parameters", None)
        inference["observed"] = {"from": "simulation", "twin": "fit"}
        inference["noise"] = T6_RADIOMETER_TEN_FOLD
        document = repatch(base, model={**base["model"], "noise": RADIOMETER_DRAWN},
                           inference=inference)
        payload = priced_run(document)
        drawn = _t6_drawn(payload)
        assert drawn is not None
        found = priced_only(document, "C18")
        assert found.severity == "refuse"


class TestTheMessage:
    """Kills: a message naming only one path, which D-C17's own words
    forbid ("refuse a disagreement, naming both paths")."""

    def test_the_message_names_both_paths(self):
        document = _pair(RADIOMETER_DRAWN, T6_RADIOMETER_TEN_FOLD)
        found = priced_only(document, "C18")
        assert "model.noise" in found.message
        assert "inference.noise" in found.message
        assert "0.0007071067811865475" in found.message
        assert "7.071067811865475e-05" in found.message
        assert found.message.endswith("(check C18).")


class TestTheStandDownRuleAgreesWithTask2:
    """The advice-loop rule's cousin (D-4.2): one rule -- "did the twin that
    drew the data actually draw it" -- decided from two vantage points, the
    document's text (``preflight/gated.py::_t2c_generated``) and the built
    payload (:func:`~rheplicant.config.postflight.noise._t6_drawn`). If they
    ever disagree, one of them is wrong and nothing else would say so."""

    def test_the_built_rule_agrees_with_t2c_generated_on_every_shipped_builder(
            self):
        """Driven over all THIRTEEN ``*_document`` builders under
        ``tests/config/*_helpers.py`` -- the same set
        ``test_config_fixture_contract.py`` discovers and drives, so this
        cannot silently cover fewer than the real property walk does."""
        import inspect

        _FORWARD = {"kind": "forward"}
        rows = []
        for module, builders in _builders().items():
            for name, builder in builders.items():
                try:
                    inspect.signature(builder).bind()
                except TypeError:
                    document = builder(_FORWARD)
                else:
                    document = builder()
                text_says = _t2c_generated(document)
                payload = priced_run(document)
                built_says = _t6_drawn(payload) is not None
                rows.append((module, name, text_says, built_says))
        assert len(rows) == 13, rows
        disagreements = [row for row in rows if row[2] != row[3]]
        assert disagreements == [], disagreements


class TestTheAdviceLoop:
    """The advice-loop rule (§4, STANDING-RULES.md D): take a document this
    check refuses, apply the refusal's own advice LITERALLY, in BOTH
    directions it names, and assert the id is gone and ``load_document``
    succeeds."""

    def test_changing_the_operator_repairs_a_radiometer_disagreement(self):
        """"change model.noise's channel_width/integration_time" -- taken
        literally: move the OPERATOR to the likelihood's own numbers."""
        document = _pair(RADIOMETER_DRAWN, T6_RADIOMETER_TEN_FOLD)
        assert priced_only(document, "C18") is not None
        repaired_operator = {"type": "RadiometerNoiseOperator",
                             **{key: T6_RADIOMETER_TEN_FOLD[key]
                                for key in ("channel_width", "integration_time")}}
        repaired = _pair(repaired_operator, T6_RADIOMETER_TEN_FOLD)
        run = load_document(repaired)
        assert "C18" not in run.report.checks()

    def test_changing_the_likelihood_repairs_a_radiometer_disagreement(self):
        """"or inference.noise's" -- the other escape: move the LIKELIHOOD to
        the operator's own numbers."""
        document = _pair(RADIOMETER_DRAWN, T6_RADIOMETER_TEN_FOLD)
        assert priced_only(document, "C18") is not None
        repaired_likelihood = {"kind": "radiometer", "include_logdet": True,
                               **RADIOMETER_NODE}
        repaired = _pair(RADIOMETER_DRAWN, repaired_likelihood)
        run = load_document(repaired)
        assert "C18" not in run.report.checks()

    def test_changing_the_operator_repairs_a_homoscedastic_disagreement(self):
        """The homoscedastic family's own two escapes, the first: move the
        OPERATOR."""
        document = _pair(SIGMA_MISMATCHED_K, HOMOSCEDASTIC)
        assert priced_only(document, "C18") is not None
        repaired = _pair({"type": "NoiseOperator", "sigma": HOMOSCEDASTIC["sigma"]},
                         HOMOSCEDASTIC)
        run = load_document(repaired)
        assert "C18" not in run.report.checks()

    def test_changing_the_likelihood_repairs_a_homoscedastic_disagreement(self):
        """The homoscedastic family's second escape: move the LIKELIHOOD."""
        document = _pair(SIGMA_MISMATCHED_K, HOMOSCEDASTIC)
        assert priced_only(document, "C18") is not None
        repaired = _pair(SIGMA_MISMATCHED_K,
                         {"kind": "homoscedastic", "sigma": SIGMA_MISMATCHED_K["sigma"]})
        run = load_document(repaired)
        assert "C18" not in run.report.checks()


class TestTheRTOLMeasurement:
    """Anti-vacuity for :data:`_T6_RTOL`'s own comment: the mutation this
    task's brief calls out explicitly is ``_T6_RTOL = 10.0``, not ``0`` --
    the measured disagreement between two independently-built, agreeing
    objects is EXACTLY 0.0 on both routes, so 0 and 1e-9 are indistinguishable
    to every test in this module."""

    def test_the_constant_is_far_below_a_real_disagreement(self):
        assert _T6_RTOL < 1e-6

    def test_the_radiometer_route_measures_exactly_zero(self):
        """The measurement :data:`_T6_RTOL`'s own comment cites, re-run here
        so it cannot silently drift from the constant it justifies."""
        from rheplicant.radio.instrument.noise import RadiometerNoiseOperator

        op = RadiometerNoiseOperator(channel_width=1.0e6, integration_time=2.0)
        like = RadiometerNoise(channel_width=1.0e6, integration_time=2.0)
        assert op.fractional == like.fractional
