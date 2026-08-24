"""C18, the numeric half -- the sigma that drew the data and the sigma that
weighs it, made to agree.

The floor here is a normalized consumer, not another resolution producer.

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

import dataclasses
import re

import jax.numpy as jnp
import pytest

from rheplicant.config.document import load_document
from rheplicant.config.errors import ConfigError
from rheplicant.config.postflight.noise import (
    _T6_RTOL,
    _t6_drawn,
    _t6_drawn_in,
    _t6_generating_records,
    _t6_generating_twin,
    _t6_sigma_agreement,
)
from rheplicant.config.preflight.gated import _t2c_generated
from rheplicant.core.errors import AssemblyError
from rheplicant.inference import FlaggedNoise, HomoscedasticNoise, RadiometerNoise
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

    def test_stands_down_on_observed_from_file_with_no_twin_repair_at_all(
            self, tmp_path):
        """**Kills M24** (the ``from: simulation`` clause in
        :func:`~rheplicant.config.postflight.noise._t6_generating_twin`
        deleted): the shipped test above keeps ``inference.twin.without:
        [noise]`` (the base document's own default, ``exit_helpers._repaired``),
        so a mutant that deletes the ``from`` check still stands down there --
        the FIT twin it falls through to has ``noise`` repaired out, and
        ``_t6_drawn``'s ``except (KeyError, AssemblyError)`` masks the missing
        clause. This document drops ``inference.twin`` and ``inference.
        parameters`` entirely (zero latents, so nothing forces a repair) and
        keeps ``model.noise`` and ``inference.noise`` TEN-FOLD apart -- under
        the mutant, ``record.get("twin", "full")`` reads the file record's own
        explicit ``twin: None`` (not the default), which is not ``"full"``,
        so it falls through to ``inference.fit_twin`` -- the SAME object as
        ``run.twin`` here, still carrying ``noise`` -- and refuses. Measured
        directly against the mutant before this test was written."""
        import numpy as np

        path = tmp_path / "d.npy"
        np.save(path, np.zeros((16, 8), dtype=np.float32))
        base = preflight_document()
        inference = {key: value for key, value in base["inference"].items()
                    if key not in ("twin", "parameters")}
        inference["observed"] = {"file": {"format": "npy", "path": str(path)}}
        document = repatch(
            base, model={**base["model"], "noise": SIGMA_MISMATCHED_K},
            inference=inference)
        assert priced_findings(document) == ()

    def test_stands_down_when_inference_noise_is_none_even_though_model_noise_disagrees(
            self):
        """Contract step 3, as a document rather than a sentence -- **kills
        M22** (the ``kind: "none"`` stand-down in
        :func:`~rheplicant.config.postflight.noise._t6_sigma_agreement`
        deleted). ``model.noise`` is lit and set to a value that would refuse
        against the base document's own likelihood sigma, and
        ``inference.noise`` is absent (``kind: none``): there is nothing to
        weigh against, so this check has nothing to say -- Task 2's
        ``C18.kind`` is the one that speaks about an undeclared
        ``inference.noise`` on a fitting run, and it is not reached here
        because this document is ``kind: forward``
        (:data:`~tests.config.preflight_helpers.preflight_document`'s own
        default run)."""
        base = preflight_document()
        inference = {key: value for key, value in base["inference"].items()
                    if key != "noise"}
        document = repatch(base, model={**base["model"], "noise": SIGMA_MISMATCHED_K},
                           inference=inference)
        assert priced_findings(document) == ()

    def test_stands_down_when_inference_noise_is_none(self):
        """The base document's own ``model.noise`` (agreeing, not
        disagreeing) with ``inference.noise`` dropped -- the shape a
        fitting-run document with the SAME absence would ALSO trip
        ``C18.kind``'s WARN on, which this check's own docstring (contract
        step 3) says is not this check's sentence to add a second time.
        Kept as its own test because :meth:`test_stands_down_when_inference_
        noise_is_none_even_though_model_noise_disagrees` above changes TWO
        things at once (drops ``inference.noise`` AND disagrees
        ``model.noise``); this one changes only the absence."""
        base = preflight_document()
        inference = {key: value for key, value in base["inference"].items()
                    if key != "noise"}
        document = repatch(base, inference=inference)
        assert priced_findings(document) == ()

    def test_a_radiometer_draw_declines_a_homoscedastic_weigh_even_when_forced(
            self):
        """**Kills M31** (the radiometer-side family guard,
        ``weighed.kind not in _T6_RADIOMETER_KINDS``, deleted).
        ``preflight/gated.py::_sigma_families`` already REFUSES a
        cross-family document before ``load_document`` ever reaches this
        pass (:meth:`TestStandsDown.test_does_not_restate_the_family_mismatch`
        proves it), so no DOCUMENT can drive this guard through
        :func:`~tests.config.inflight_helpers.priced_findings` -- the payload
        is built from a real, same-family document and then has its
        ``inference.noise`` swapped after the fact, which is what this guard
        alone stands between a RadiometerNoiseOperator draw and
        :func:`~rheplicant.config.postflight.noise._t6_homoscedastic`'s
        ``.model.sigma`` read (a ``RadiometerNoise`` has no ``.sigma``)."""
        document = _pair(RADIOMETER_DRAWN, RADIOMETER)
        payload = priced_run(document)
        mismatched = payload.run.inference.noise._replace(
            kind="homoscedastic", model=HomoscedasticNoise(jnp.asarray(0.05)))
        swapped = dataclasses.replace(payload, run=payload.run._replace(
            inference=payload.run.inference._replace(noise=mismatched)))
        assert tuple(_t6_sigma_agreement(swapped)) == ()

    def test_a_homoscedastic_draw_declines_a_radiometer_weigh_even_when_forced(
            self):
        """**Kills M32** (the homoscedastic-side family guard,
        ``weighed.kind != "homoscedastic"``, deleted) -- the twin of the test
        above, forcing a ``NoiseOperator`` draw against a ``radiometer``
        weigh, which :func:`~rheplicant.config.postflight.noise._t6_radiometer`
        would read ``.fractional`` off a ``HomoscedasticNoise`` that has
        none."""
        document = _pair({"type": "NoiseOperator",
                          "sigma": {"value": HOMOSCEDASTIC["sigma"]["value"],
                                   "unit": "K"}},
                         HOMOSCEDASTIC)
        payload = priced_run(document)
        mismatched = payload.run.inference.noise._replace(
            kind="radiometer", model=RadiometerNoise(1.0e6, 2.0))
        swapped = dataclasses.replace(payload, run=payload.run._replace(
            inference=payload.run.inference._replace(noise=mismatched)))
        assert tuple(_t6_sigma_agreement(swapped)) == ()

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


#: An ``observation:`` patch declaring the ``aux.flags`` mask
#: ``inference.noise.flags: {from: observation}`` reads -- BLOCKER 1's
#: fixture.  ``zeros``, deliberately: this module's own comparison does not
#: care WHICH samples are flagged, only that the wrapper built around the
#: model can be unwrapped, so an all-``False`` mask is enough to exercise the
#: wrap without changing which of the compared numbers is finite.
def _flagged_observation():
    base = preflight_document()
    return {**base["observation"],
           "aux": {"flags": {"zeros": ["n_time", "n_freq"]}}}


class TestFlaggedNoiseIsUnwrapped:
    """BLOCKER 1: ``inference.noise.flags: {from: observation}`` wraps the
    built likelihood in :class:`~rheplicant.inference.noise.FlaggedNoise`
    (``config/sections/noise.py::_wrap_flags``, for BOTH ``homoscedastic``
    and ``radiometer`` -- ``radiometer_frozen``'s ``_KIND_KEYS`` does not even
    accept a ``flags:`` key, so it is out of scope here), and ``FlaggedNoise``
    carries none of ``.sigma``, ``.fractional`` or ``.floor``. Before this
    fix, reading any of them through the wrapper raised ``AttributeError``,
    which ``sweep`` turns into ``ConfigError: post-flight check 'C18' RAISED
    AttributeError: ...`` -- a CORRECT document (both families, agreeing
    sigmas) made unloadable. Four rows: agree/disagree crossed with
    radiometer/homoscedastic, so both the silent path (needs ``.sigma``/
    ``.fractional`` to compare) and the refusing path (ALSO needs
    ``.floor`` for the radiometer message) are exercised unwrapped."""

    def test_a_flagged_homoscedastic_pair_that_agrees_is_silent(self):
        document = _pair(
            {"type": "NoiseOperator", "sigma": {"value": HOMOSCEDASTIC["sigma"]["value"],
                                                 "unit": "K"}},
            {**HOMOSCEDASTIC, "flags": {"from": "observation"}},
            observation=_flagged_observation())
        assert priced_findings(document) == ()

    def test_a_flagged_homoscedastic_pair_that_disagrees_refuses(self):
        document = _pair(
            SIGMA_MISMATCHED_K,
            {**HOMOSCEDASTIC, "flags": {"from": "observation"}},
            observation=_flagged_observation())
        found = priced_only(document, "C18")
        assert found.severity == "refuse"

    def test_a_flagged_radiometer_pair_that_agrees_is_silent(self):
        document = _pair(
            RADIOMETER_DRAWN, {**RADIOMETER, "flags": {"from": "observation"}},
            observation=_flagged_observation())
        assert priced_findings(document) == ()

    def test_a_flagged_radiometer_pair_that_disagrees_refuses(self):
        document = _pair(
            RADIOMETER_DRAWN,
            {**T6_RADIOMETER_TEN_FOLD, "flags": {"from": "observation"}},
            observation=_flagged_observation())
        found = priced_only(document, "C18")
        assert found.severity == "refuse"

    def test_the_wrapped_models_realise_is_the_base_models_own_draw(self):
        """The docstring claim :func:`~rheplicant.config.postflight.noise.
        _t6_unwrapped` leans on, checked directly against the shipped class
        rather than taken on faith: ``FlaggedNoise.realise`` returns exactly
        ``self.base.realise(...)`` -- same array, not merely the same
        shape or dtype."""
        import jax

        base = HomoscedasticNoise(jnp.asarray(0.05))
        flagged = FlaggedNoise(base, jnp.zeros((2, 3), dtype=bool))
        prediction = jnp.ones((2, 3))
        key = jax.random.PRNGKey(0)
        assert jnp.array_equal(
            flagged.realise(prediction, key=key),
            base.realise(prediction, key=key))


class TestTheReplacedNoiseNodeIsNotBlamed:
    """BLOCKER 3: ``inference.twin.replace: {noise: ...}`` on a ``twin: fit``
    observation swaps the FIT twin's OWN ``noise`` node for the
    document-declared REPLACEMENT operator (``build_fit_twin``'s
    ``repaired.replace_node(node_id, ...)``, ``sections/twin.py``) -- the
    operator that actually drew the data is THAT replacement, never
    ``model.noise``.

    **Measured, before this fix**, on a document whose ``model.noise`` and
    ``inference.noise`` already AGREE (both 0.05 K) and whose ``replace:``
    supplies a DIFFERENT sigma (0.5 K):
    :func:`~rheplicant.config.postflight.noise._t6_generating_twin` read the
    fit twin regardless of the replacement, so
    :func:`~rheplicant.config.postflight.noise._t6_drawn` read the
    REPLACEMENT's own operator (sigma 0.5) off node ``noise`` -- not
    ``model.noise``'s (sigma 0.05, unread). The check then refused, and its
    message said ``"model.noise draws this document's data with sigma =
    0.5"`` -- a value ``model.noise:`` never declares in this document's
    text at all -- while naming ``inference.noise``'s own 0.05 as the
    disagreement. §3's advice ("change model.noise's ... or
    inference.noise's") is then a LIVE ADVICE LOOP: both sides already
    agree, so following either escape literally changes a value that was
    never the one that drew the data, and the refusal returns unchanged."""

    def _document(self):
        base = preflight_document()
        inference = dict(base["inference"])
        inference.pop("parameters", None)
        inference["twin"] = {"replace": {"noise": {
            "type": "NoiseOperator", "sigma": {"value": 0.5, "unit": "K"}}}}
        inference["observed"] = {"from": "simulation", "twin": "fit"}
        # model.noise and inference.noise AGREE (both SIGMA_K) -- the
        # replacement (0.5) is the only thing that disagrees with anything,
        # and it is not this check's node to read.
        return repatch(base, inference=inference)

    def test_stands_down_when_the_fit_twins_noise_node_was_replaced(self):
        document = self._document()
        assert priced_findings(document) == ()

    def test_the_generating_twin_reads_none_when_noise_was_replaced(self):
        """The mechanism directly, not just the outcome: :func:`_t6_
        generating_twin` returns ``None`` -- not the fit twin -- so
        :func:`_t6_drawn` never has a node to read at all."""
        payload = priced_run(self._document())
        assert _t6_generating_twin(payload) is None
        assert _t6_drawn(payload) is None

    def test_the_advice_loop_is_gone(self):
        """STANDING-RULES.md D, applied to the document BLOCKER 3 exists
        for: before the fix, this exact document refused, naming
        ``model.noise`` and advising a change to ``model.noise's ... or
        inference.noise's`` sigma -- and applying either literally still
        refused (both numbers already agreed; the REAL culprit, the
        replacement's own 0.5, was never named). After the fix,
        ``load_document`` already succeeds with no C18 finding at all, so
        there is no advice to follow -- the loop is closed by the check no
        longer speaking about a node it does not mean."""
        document = self._document()
        run = load_document(document)
        assert "C18" not in run.report.checks()

    def test_following_the_old_bogus_advice_literally_still_stands_down(self):
        """The stronger form of the proof above: apply the PRE-FIX
        refusal's own escape sentence LITERALLY -- "change model.noise's
        ... sigma" -- to a value that does not match ``inference.noise``
        either, and confirm the check still correctly stands down, because
        it is no longer reading ``model.noise`` for this document AT ALL.
        Before this fix, no value of ``model.noise.sigma`` could clear the
        refusal (measured in the review: 0.05, 0.5, 5.0 and 0.0001 all
        left ``findings=['C18']``) because the sentence pointed at a key
        the check was never actually reading."""
        base = preflight_document()
        inference = dict(base["inference"])
        inference.pop("parameters", None)
        inference["twin"] = {"replace": {"noise": {
            "type": "NoiseOperator", "sigma": {"value": 0.5, "unit": "K"}}}}
        inference["observed"] = {"from": "simulation", "twin": "fit"}
        document = repatch(base, model={**base["model"], "noise": {
            "type": "NoiseOperator", "sigma": {"value": 5.0, "unit": "K"}}},
            inference=inference)
        run = load_document(document)
        assert "C18" not in run.report.checks()


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

    def test_the_radiometer_message_is_pinned_whole(self):
        """**Kills M16/M17/M30**, whole-string equality rather than a
        substring search: the test above is satisfied by the ESCAPE
        sentence's own later repetition of both key names ("change
        model.noise's ... or inference.noise's"), so it cannot tell a
        renamed SUBJECT clause (M16/M17) from an intact one, and it checks
        both numbers are PRESENT without checking which clause each sits in,
        so it cannot tell a swapped drawn/weighed attribution (M30) from a
        correct one. STANDING-RULES.md C: pin by whole-string equality, not
        ``match=``."""
        document = _pair(RADIOMETER_DRAWN, T6_RADIOMETER_TEN_FOLD)
        found = priced_only(document, "C18")
        assert found.message == (
            "model.noise draws this document's data at a fractional scatter "
            "(1 / sqrt(channel_width * integration_time)) of "
            "0.0007071067811865475, and inference.noise (kind: radiometer) "
            "weighs it at a different fractional scatter of "
            "7.071067811865475e-05. The fit is weighted against a scatter "
            "its own data does not have, and it returns a finite, "
            "correctly-shaped answer whose error bars are wrong by whatever "
            "the two differ by. Make the two agree -- change model.noise's "
            "channel_width/integration_time, or inference.noise's -- so the "
            "same physical bandwidth and integration time reach both sides. "
            "(check C18)."
        )

    def test_the_homoscedastic_message_is_pinned_whole(self):
        """**Kills M16/M17/M38** for the homoscedastic family -- this route
        had NO message pin at all before this fix round
        (``test_the_homoscedastic_pair_is_compared`` only checks
        ``severity``), so a drawn number replaced by a placeholder (M38) or a
        renamed subject clause (M16/M17) was invisible to this file."""
        document = _pair(SIGMA_MISMATCHED_K, HOMOSCEDASTIC)
        found = priced_only(document, "C18")
        assert found.message == (
            "model.noise draws this document's data with sigma = "
            "Array(0.5, dtype=float32), and inference.noise (kind: "
            "homoscedastic) weighs it with a different sigma = "
            "Array(0.05, dtype=float32). The fit is weighted against a "
            "scatter its own data does not have, and it returns a finite, "
            "correctly-shaped answer whose error bars are wrong by whatever "
            "the two differ by. Make the two agree -- change model.noise"
            ".sigma, or inference.noise.sigma -- so both sides declare the "
            "same number (check C18)."
        )

    def test_the_frozen_radiometer_floor_clause_is_pinned_whole(self):
        """**Kills M36** (``NoiseBuild.frozen["floor"]`` never read for
        ``kind: radiometer_frozen`` -- ``_t6_radiometer_floor`` returns
        ``0.0`` unconditionally): the shipped floor tests
        (``test_a_disagreement_with_a_declared_floor_names_it``) only cover
        ``kind: radiometer``, whose floor comes from a different branch
        (``weighed.model.floor``) that M36 does not touch at all."""
        weighed = {**T6_FROZEN_HUNDRED_FOLD, "floor": FLOOR_DECLARED_K}
        document = _pair(RADIOMETER_DRAWN, weighed)
        found = priced_only(document, "C18")
        assert found.message == (
            "model.noise draws this document's data at a fractional scatter "
            "(1 / sqrt(channel_width * integration_time)) of "
            "0.0007071067811865475, and inference.noise (kind: "
            "radiometer_frozen) weighs it at a different fractional scatter "
            "of 7.071067811865475e-05. The fit is weighted against a "
            "scatter its own data does not have, and it returns a finite, "
            "correctly-shaped answer whose error bars are wrong by whatever "
            "the two differ by. Make the two agree -- change model.noise's "
            "channel_width/integration_time, or inference.noise's -- so the "
            "same physical bandwidth and integration time reach both sides. "
            "inference.noise.floor: 0.01 K is declared and takes no part in "
            "this comparison -- RadiometerNoiseOperator applies no floor, "
            "deliberately: a floor is a remedy for a reweighting iterate "
            "crossing zero, and a generator has no iterate. (check C18)."
        )


class TestTheExceptClauseIsExactlyTwoMembers:
    """**Kills M20/M21**: :func:`~rheplicant.config.postflight.noise._t6_drawn`
    catches exactly ``(KeyError, AssemblyError)`` around ``twin[_T6_NOISE_NODE]``
    -- its own docstring's claim that this "does not swallow an unrelated
    failure" is untested. No document in this suite makes ``__getitem__``
    raise anything but those two (the ``AmbiguousNodeError`` branch is
    "unreachable through ``__getitem__`` today", per that same docstring), so
    both directions are tested against a FAKE twin, with
    :func:`~rheplicant.config.postflight.noise._t6_generating_twin` patched
    to hand it back -- the same direct-call style this module's docstring
    already uses for :func:`~rheplicant.config.postflight.noise._t6_radiometer_floor`."""

    class _FakeTwin:
        def __init__(self, exc):
            self._exc = exc

        def __getitem__(self, node_id):
            raise self._exc

    def test_an_assemblyerror_subclass_stands_down_not_only_a_bare_one(
            self, monkeypatch):
        """**Kills M21** (``except (KeyError, AssemblyError)`` narrowed to
        ``except KeyError``): ``AmbiguousNodeError`` -- or any other
        ``AssemblyError`` subclass -- must stand down, not propagate."""
        class _SomeAssemblyError(AssemblyError):
            pass

        import rheplicant.config.postflight.noise as noise_mod

        monkeypatch.setattr(
            noise_mod, "_t6_generating_twin",
            lambda payload: self._FakeTwin(_SomeAssemblyError("boom")))
        assert noise_mod._t6_drawn(None) is None

    def test_an_unrelated_exception_propagates_rather_than_stands_down(
            self, monkeypatch):
        """**Kills M20** (``except (KeyError, AssemblyError)`` widened to
        ``except Exception``): a failure with nothing to do with a missing or
        ambiguous node must NOT be laundered into a silent stand-down --
        ``sweep`` is what turns an uncaught exception into a named "check
        'C18' RAISED" report, which is how an unrelated bug inside indexing
        would actually be noticed rather than hidden as "the document is
        fine"."""
        import rheplicant.config.postflight.noise as noise_mod

        monkeypatch.setattr(
            noise_mod, "_t6_generating_twin",
            lambda payload: self._FakeTwin(ValueError("unrelated")))
        with pytest.raises(ValueError):
            noise_mod._t6_drawn(None)


class TestTheStandDownRuleAgreesWithTask2:
    """The advice-loop rule's cousin (D-4.2): one rule -- "did the twin that
    drew the data actually draw it, still carrying the noise it would draw
    with" -- decided from two vantage points, the document's text
    (``preflight/gated.py::_t2c_generated``) and the built payload
    (:func:`~rheplicant.config.postflight.noise._t6_drawn`). If they ever
    disagree, one of them is wrong and nothing else would say so.

    **MAJOR 1's finding is real and is fixed here; the reviewer's PRESCRIBED
    comparator (``_t6_generating_twin`` in place of ``_t6_drawn``) is not,
    and this class does not adopt it -- measured, not assumed.**
    ``_t2c_generated``'s own docstring (``preflight/gated.py``) says its
    ``twin: fit`` branch is True only when ``inference.twin:`` has NOT
    ``_t2c_repaired`` the ``noise`` node out of the tree -- i.e. it already
    folds "does the generating twin still carry noise" into itself for the
    ``fit`` case, which is exactly what ``_t6_drawn`` computes on the built
    side (index the node, catch the exception if it is gone).
    ``_t6_generating_twin`` computes something narrower: "is there an
    assembly to read from at all", never checking whether ``without:``
    stripped the node. Swapping the comparator to ``_t6_generating_twin``
    was measured against ALL THIRTEEN shipped builders and against ELEVEN of
    them -- every ``conjugate_document``/``fanned_document``/``gcr_document``
    /``gls_document``/``gls_pair_document``/``two_latent_document``/
    ``wiener_document``/``joint_prior_document``/``npe_document``/
    ``nuts_document``/``trio_npe_document`` builder, every one of which is
    ``twin: fit`` with ``inference.twin.without: [noise]`` -- it introduces a
    BRAND NEW disagreement (``text=False`` against ``gentwin=True``) that did
    not exist before. ``_t6_drawn`` has ZERO disagreements against all
    thirteen, both before and after this fix round; the true bug MAJOR 1
    found is a narrower, single-document one, closed below without breaking
    the thirteen shipped documents that already worked."""

    def test_the_built_rule_agrees_with_t2c_generated_on_every_shipped_builder(
            self):
        """Driven over all THIRTEEN ``*_document`` builders under
        ``tests/config/*_helpers.py`` -- the same set
        ``test_config_fixture_contract.py`` discovers and drives, so this
        cannot silently cover fewer than the real property walk does.
        **Regression pin**: measured directly, comparing against
        ``_t6_generating_twin`` instead makes 11 of these 13 rows disagree
        (every ``twin: fit`` + ``without: [noise]`` builder) -- this is why
        the comparator here stays ``_t6_drawn``."""
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

    #: Non-builder documents MAJOR 1 names explicitly: shapes no shipped
    #: ``*_document`` factory produces, on which the rule must still agree.
    #: Each entry is a zero-argument document builder so pytest's own
    #: collection, not an eager module-level call, is what pays for building
    #: several extra documents.  ``_no_model_noise`` is deliberately absent
    #: from this dict -- it is the one genuine, EXPLAINED divergence, and it
    #: gets its own test below rather than a same-value assertion here.
    def _fit_without_uniform_sky(self):
        base = preflight_document()
        inference = dict(base["inference"])
        inference.pop("parameters", None)
        inference["twin"] = {"without": ["uniform_sky"]}
        inference["observed"] = {"from": "simulation", "twin": "fit"}
        return repatch(base, inference=inference)

    def _fit_zero_latents_unrepaired(self):
        base = preflight_document()
        inference = dict(base["inference"])
        inference.pop("parameters", None)
        inference.pop("twin", None)
        inference["observed"] = {"from": "simulation", "twin": "fit"}
        return repatch(base, inference=inference)

    def _fit_replace_noise(self):
        base = preflight_document()
        inference = dict(base["inference"])
        inference.pop("parameters", None)
        inference["twin"] = {"replace": {"noise": {
            "type": "NoiseOperator", "sigma": {"value": 0.5, "unit": "K"}}}}
        inference["observed"] = {"from": "simulation", "twin": "fit"}
        return repatch(base, inference=inference)

    def _multi_record_with_primary(self):
        base = preflight_document()
        inference = dict(base["inference"])
        inference["observed"] = {
            "primary": {"from": "simulation", "twin": "full", "at": {"g": 1.5}},
            "second": {"from": "simulation", "twin": "full", "at": {"g": 1.2}}}
        return repatch(base, inference=inference)

    def _multi_record_without_primary(self):
        base = preflight_document()
        inference = dict(base["inference"])
        inference["observed"] = {
            "alpha": {"from": "simulation", "twin": "full", "at": {"g": 1.5}},
            "beta": {"from": "simulation", "twin": "full", "at": {"g": 1.2}}}
        return repatch(base, inference=inference)

    def _no_observed(self):
        base = preflight_document()
        inference = {key: value for key, value in base["inference"].items()
                    if key != "observed"}
        return repatch(base, inference=inference)

    @pytest.mark.parametrize("label", [
        "fit_without_uniform_sky", "fit_zero_latents_unrepaired",
        "fit_replace_noise", "multi_record_with_primary",
        "multi_record_without_primary", "no_observed",
    ])
    def test_the_built_rule_agrees_on_documents_no_shipped_builder_makes(
            self, label):
        """Six of the seven non-builder rows MAJOR 1 names, all of which
        agree under ``_t6_drawn`` (``fit_replace_noise`` only after
        BLOCKER 3; before it, that row was ITS OWN disagreement -- ``text``
        already read the replacement as not-generating via
        ``_t2c_repaired``, while ``_t6_drawn`` still read ``model.noise``'s
        class off the unrepaired fit twin)."""
        document = getattr(self, f"_{label}")()
        payload = priced_run(document)
        assert _t2c_generated(document) == (_t6_drawn(payload) is not None)

    def test_the_built_rule_agrees_on_a_file_form_observation(self, tmp_path):
        """The seventh row, ``from: file``: needs a real path, so it is its
        own test rather than a ``parametrize`` row."""
        import numpy as np

        path = tmp_path / "d.npy"
        np.save(path, np.zeros((16, 8), dtype=np.float32))
        base = preflight_document()
        inference = {**base["inference"],
                    "observed": {"file": {"format": "npy", "path": str(path)}}}
        document = repatch(base, inference=inference)
        payload = priced_run(document)
        assert _t2c_generated(document) == (_t6_drawn(payload) is not None)

    def test_the_no_model_noise_divergence_is_real_and_does_not_matter(self):
        """The ONE genuine divergence between ``_t2c_generated`` and
        ``_t6_drawn``, measured rather than papered over: on a document with
        NO ``model.noise`` at all and ``observed.twin: full`` (the shipped
        default), ``_t2c_generated`` returns ``True`` unconditionally on the
        ``twin: full`` branch (it never inspects ``model:`` for a ``noise``
        node -- that is ``_t2c_drawn``'s separate job, called BEFORE
        ``_t2c_generated`` by ``_sigma_families``), while ``_t6_drawn``
        returns ``None`` (there is no node to index). **This does not make
        either check wrong**: ``_sigma_families`` never reaches
        ``_t2c_generated`` on this document at all (``_t2c_drawn(document)``
        already returns ``None`` and the caller stands down first), and
        ``_t6_sigma_agreement`` never reaches anything past ``_t6_drawn``
        either (its own ``if drawn is None: return ()`` is the first line
        after ``_t6_drawn`` is called) -- so the divergence lives entirely
        OUTSIDE the path either real check ever executes on this document,
        and both checks correctly produce nothing on it (see
        ``TestStandsDown.test_stands_down_when_the_model_lights_no_noise``).
        Calling the two helper functions in isolation, bypassing the
        type-recognition gate that always precedes them in production, is
        what manufactures the disagreement."""
        base = preflight_document()
        model_without_noise = {key: value for key, value in base["model"].items()
                               if key != "noise"}
        inference_without_twin = {key: value for key, value in base["inference"].items()
                                  if key != "twin"}
        document = repatch(base, model=model_without_noise,
                           inference=inference_without_twin)
        payload = priced_run(document)
        assert _t2c_generated(document) is True
        assert _t6_drawn(payload) is None
        # And yet the real checks agree perfectly on the OUTCOME:
        assert priced_findings(document) == ()


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

    def test_changing_the_operator_repairs_a_radiometer_frozen_disagreement(
            self):
        """MINOR: ``radiometer_frozen`` -- "the twin" this plan names --
        had no advice-loop test at all. The first escape: move the
        OPERATOR to the frozen kind's own numbers."""
        document = _pair(RADIOMETER_DRAWN, T6_FROZEN_HUNDRED_FOLD)
        assert priced_only(document, "C18") is not None
        repaired_operator = {"type": "RadiometerNoiseOperator",
                             **{key: T6_FROZEN_HUNDRED_FOLD[key]
                                for key in ("channel_width", "integration_time")}}
        repaired = _pair(repaired_operator, T6_FROZEN_HUNDRED_FOLD)
        run = load_document(repaired)
        assert "C18" not in run.report.checks()

    def test_changing_the_likelihood_repairs_a_radiometer_frozen_disagreement(
            self):
        """``radiometer_frozen``'s second escape: move the LIKELIHOOD to the
        operator's own numbers."""
        document = _pair(RADIOMETER_DRAWN, T6_FROZEN_HUNDRED_FOLD)
        assert priced_only(document, "C18") is not None
        repaired_likelihood = {**T6_FROZEN_HUNDRED_FOLD,
                               **{key: RADIOMETER_DRAWN[key]
                                  for key in ("channel_width", "integration_time")}}
        repaired = _pair(RADIOMETER_DRAWN, repaired_likelihood)
        run = load_document(repaired)
        assert "C18" not in run.report.checks()


class TestByObservationIsReachableAndCorrect:
    """MINOR: :attr:`~rheplicant.config.sections.noise.NoiseBuild.by_observation`
    (the per-observation frozen sigma ``radiometer_frozen`` with
    ``source: observed`` produces) is reachable on a two-record document, and
    this check is correct there because it reads ``weighed.frozen`` -- the
    channel width and integration time, which do not vary by observation --
    off the SAME primary :func:`~rheplicant.config.postflight.noise.
    _t6_generating_twin` reads, never off ``by_observation`` itself."""

    def test_by_observation_is_populated_and_c18_still_finds_the_primarys_disagreement(
            self):
        base = preflight_document()
        inference = dict(base["inference"])
        inference["noise"] = T6_FROZEN_HUNDRED_FOLD
        inference["observed"] = {
            "night": {"from": "simulation", "twin": "full", "at": {"g": 1.5}},
            "primary": {"from": "simulation", "twin": "full", "at": {"g": 1.2}},
        }
        document = repatch(base, model={**base["model"], "noise": RADIOMETER_DRAWN},
                           inference=inference)
        payload = priced_run(document)
        weighed = payload.run.inference.noise
        assert sorted((weighed.by_observation or {}).keys()) == ["night", "primary"]
        found = priced_only(document, "C18")
        assert found.severity == "refuse"


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


def _c18(document):
    """Every ``C18`` finding the post-flight pass made, in slot order.

    ``priced_only`` cannot be used below: the whole point of the widening is
    that one document now earns MORE than one, and ``_only`` refuses that.
    """
    return tuple(found for found in priced_findings(document)
                 if found.check == "C18")


class TestARecordWithoutAPrimaryIsStillChecked:
    """BLOCKER 2's other half: the numeric C18 widened to match ``C18.kind``.

    Two or more NAMED ``inference.observed:`` records with none of them
    literally called ``primary`` resolve no primary at all, and both C18
    slots used to stand down on that document exactly as they do on one with
    no ``observed:`` section. A1.2 closed it for the text half; the numeric
    half kept reading ``observed.primary`` alone, so the two vantage points
    diverged on this one shape -- recorded in both module docstrings as a
    follow-up rather than a defect, because a check that is silent is not a
    check that is wrong.

    Measured before this change, on the tenfold sigma disagreement below::

        single-record (``primary``)                       REFUSED
        named 'primary' (+ a second)                      REFUSED
        named 'alpha' ALONE (only record)                 REFUSED
        named 'alpha' + 'beta' (no 'primary')             NO FINDINGS

    The trigger is a record NAME, not a property of the document: the fourth
    row carries the same tenfold disagreement as the first three.
    """

    SIM = {"from": "simulation", "twin": "full", "at": {"g": 1.5}}

    def _observed(self, observed):
        """The tenfold sigma disagreement, under one ``observed:`` shape.

        Not through ``_pair``, which sets ``inference:`` itself and so cannot
        also be given an ``observed:`` block.
        """
        base = preflight_document()
        return repatch(
            base,
            model={**base["model"], "noise": SIGMA_MISMATCHED_K},
            inference={**base["inference"], "noise": HOMOSCEDASTIC,
                       "observed": observed},
        )

    def test_the_three_shapes_that_resolve_a_primary_still_earn_one_finding(self):
        """The regression half. Widening must not double-report a document
        that already worked, and must not change what it says."""
        for observed in ({"from": "simulation", "twin": "full"},
                         {"primary": self.SIM, "second": self.SIM},
                         {"alpha": self.SIM}):
            found = _c18(self._observed(observed))
            assert len(found) == 1, (observed, [f.message for f in found])
            assert "this document's data" in found[0].message

    def test_two_named_records_without_a_primary_are_each_checked(self):
        """The gap, closed: one finding per record, each naming its own."""
        found = _c18(
            self._observed({"alpha": self.SIM, "beta": self.SIM}))

        assert len(found) == 2, [f.message for f in found]
        assert {f.severity for f in found} == {"refuse"}
        named = sorted(re.search(r"observed record '(\w+)'", f.message).group(1)
                       for f in found)
        assert named == ["alpha", "beta"]

    def test_the_widened_message_is_pinned_whole(self):
        """Equality, not a substring -- and for a reason specific to here.

        ``TestNoMovedMessageWasReworded`` cannot see this rewording at all:
        it harvests literals from ``src/rheplicant/config/`` at ``be2027b``,
        and ``postflight/noise.py`` did not exist then, so the subject clause
        moving from a literal to ``_t6_subject(name)`` is outside its corpus.
        The pin below and the two whole-message pins in ``TestTheMessage``
        are the whole of what holds these two sentences.
        """
        found = _c18(
            self._observed({"alpha": self.SIM, "beta": self.SIM}))
        alpha = next(f for f in found if "'alpha'" in f.message)

        assert alpha.where == "model.noise"
        assert alpha.message == (
            "model.noise draws the data of observed record 'alpha' with "
            "sigma = Array(0.5, dtype=float32), and inference.noise (kind: "
            "homoscedastic) weighs it with a different sigma = "
            "Array(0.05, dtype=float32). The fit is weighted against a "
            "scatter its own data does not have, and it returns a finite, "
            "correctly-shaped answer whose error bars are wrong by whatever "
            "the two differ by. Make the two agree -- change model.noise"
            ".sigma, or inference.noise.sigma -- so both sides declare the "
            "same number (check C18)."
        )

    def test_a_sibling_whose_twin_no_longer_draws_is_not_checked(self):
        """``twin: fit`` against this fixture's ``without: [noise]`` repair.

        ``beta`` IS in :func:`_t6_generating_records` -- the record chose a
        twin and that twin exists -- and it is the node LOOKUP that drops it,
        which is the same division of labour ``_t2c_generated`` and
        ``_t6_drawn`` have at the single-primary level.
        """
        found = _c18(self._observed({
            "alpha": self.SIM,
            "beta": {"from": "simulation", "twin": "fit"},
        }))

        assert [f.message.count("'alpha'") for f in found] == [1], (
            [f.message for f in found]
        )

    def test_a_sibling_read_from_a_file_is_not_checked(self, tmp_path):
        """A file-form record carries no second sigma to disagree with, so it
        never enters :func:`_t6_generating_records` at all.

        The complement of the test above, and the one that kills a widening
        which merely reports every named record it finds: measured, dropping
        the ``twin is not None`` filter leaves the case above untouched (its
        twin is not ``None``) and makes THIS document earn a second finding.
        """
        import numpy as np

        path = tmp_path / "d.npy"
        np.save(path, np.zeros((16, 8), dtype=np.float32))
        document = self._observed({
            "alpha": self.SIM,
            "beta": {"file": {"format": "npy", "path": str(path)}},
        })

        assert [name for name, _ in
                _t6_generating_records(priced_run(document))] == ["alpha"]
        found = _c18(document)
        assert [f.message.count("'alpha'") for f in found] == [1], (
            [f.message for f in found]
        )

    def test_the_two_vantage_points_agree_on_WHICH_records_they_check(self):
        """``TestTheStandDownRuleAgreesWithTask2``, at the widened level.

        That class compares the two single-primary readers, and its own
        docstring records WHICH comparator makes them agree:
        ``_t2c_generated`` folds "does the generating twin still carry noise"
        into itself, so the built-side partner is ``_t6_drawn`` -- index the
        node, stand down if it is gone -- and not ``_t6_generating_twin``,
        which asks the narrower "is there an assembly to read at all".

        The same is true one level up, and this test found it out the hard
        way. Comparing the two ``_generating_records`` readers directly makes
        ``{'alpha': twin full, 'beta': twin fit}`` disagree on this fixture,
        whose ``inference.twin: {without: [noise]}`` repairs the fit twin:
        the text reader excludes ``beta`` and the built one does not, because
        stripping the node is something only the LOOKUP can see. Threading
        ``_t6_drawn_in`` is what makes this the widened cousin of that class
        rather than a differently-shaped question.
        """
        from rheplicant.config.preflight.gated import _t2c_generating_records

        for observed in ({"from": "simulation", "twin": "full"},
                         {"primary": self.SIM, "second": self.SIM},
                         {"alpha": self.SIM},
                         {"alpha": self.SIM, "beta": self.SIM},
                         {"alpha": self.SIM,
                          "beta": {"from": "simulation", "twin": "fit"}}):
            document = self._observed(observed)
            text = _t2c_generating_records(document)
            built = tuple(name for name, twin in _t6_generating_records(
                priced_run(document)) if _t6_drawn_in(twin) is not None)

            assert text == built, (observed, text, built)

    def test_both_checks_name_a_record_the_same_way(self):
        """``_t6_subject`` is spelled out rather than imported from
        ``_t2c_subject``, deliberately -- see its docstring. That leaves the
        two free to drift, so this is what notices."""
        from rheplicant.config.postflight.noise import _t6_subject
        from rheplicant.config.preflight.gated import _t2c_subject

        for name in (None, "alpha", "primary"):
            assert _t6_subject(name) == _t2c_subject(name), name
