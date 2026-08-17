"""C16 -- ADC saturation: the check, its escalator, and the probe it runs.

**P7's third test is MERGED, not written a third time.**  The task body names
three WARN-band tests: ``test_a_warn_needs_more_than_a_thousand_samples`` on
:data:`~tests.config.preflight_helpers.WIDE_GRID`,
``test_a_saturating_document_with_an_upstream_latent_refuses`` and
``test_a_saturating_document_with_no_upstream_latent_warns_below_the_
threshold``.  Measured (this module, ``test_a_saturating_document_with_no_
upstream_latent_warns_below_the_threshold``): with the base document's only
latent removed, ``WIDE_GRID`` at :data:`~tests.config.preflight_helpers.
T5_WIDE_WARN_SCALE` is EXACTLY the document
``test_a_warn_needs_more_than_a_thousand_samples`` would build under P7's own
replacement text.  Writing both would be one document under two names, so
only the second is here; its docstring says so again at the call site.

**``test_stands_down_on_a_many_or_composed_adc`` -- the MANY half is deleted,
the COMPOSED half is not.**  This module's earlier draft deleted the whole
test on the grounds that ``adc`` is ``many=False`` (D-20 / this task's brief
§4.1): ``adc``'s ``NodeSpec`` on ``RADIO_GRAPH`` (``radio/graph.py:157``)
never sets ``many=True``, and ``NodeSpec.many`` defaults to ``False``
(``core/graph.py:122``), so ``Assembly.__getitem__`` (``core/
graph.py:455-471``) can never raise ``AmbiguousNodeError`` for this node --
that branch fires only for a node holding SEVERAL instances, which a
``many=False`` node structurally cannot.  **That argument covers
``AmbiguousNodeError`` only.**  ``adc`` -- a legal, pre-flight-accepted
``compose: cascade`` -- resolves ``payload.run.twin["adc"]`` to a
``Pipeline``, not an ``ADCOperator``, and ``Pipeline`` carries no ``.scale``:
reading it in ``_saturation_stats`` raised ``AttributeError`` straight
through ``sweep``, which aborted the ENTIRE post-flight pass and hid every
other priced check's findings.  :func:`test_stands_down_on_a_composed_adc`
below is the test that draft's claim ("the only stand-down this node can
earn is the plain ``KeyError``") was wrong about; the check's own
``isinstance(node, ADCOperator)`` guard is what fixes it.  The plain
``KeyError`` an absent ``adc`` raises is a THIRD stand-down route, and that
is :func:`test_stands_down_without_an_adc` below.

**Boundary numbers are measured, not copied from the plan** -- see this
module's own assertions and
``digitising._t5_limit``/``digitising._saturation_stats``, which every
boundary test below calls directly rather than re-deriving the arithmetic a
second time.
"""

import warnings

import jax
import jax.numpy as jnp
import pytest

from rheplicant.config.document import load_document
from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import ConfigWarning
from rheplicant.config.postflight import digitising, priced
from rheplicant.core.state import State
from rheplicant.radio.instrument.adc import ADCOperator
from rheplicant.radio.instrument.gain import GainOperator
from tests.config.exit_helpers import HOMOSCEDASTIC, TRUTH_G
from tests.config.inflight_helpers import priced_findings, priced_only, priced_run
from tests.config.preflight_helpers import (
    ADC_SATURATING,
    ADC_UNSATURATED,
    T5_BINDING_LATENT,
    T5_BOUNDARY_SCALES,
    T5_LIKELIHOOD_NOISE,
    T5_MODEL_NOISE,
    T5_WIDE_WARN_SCALE,
    WIDE_GRID,
    preflight_document,
    t5_case,
    t5_model,
)


def test_t5_likelihood_noise_agrees_with_t5_model_noise():
    """MINOR 6 (Plan 3C fix round): ``T5_LIKELIHOOD_NOISE`` is DERIVED from
    ``T5_MODEL_NOISE``'s own ``sigma`` at import
    (``preflight_helpers.py:653-654``: ``{"kind": "homoscedastic", "sigma":
    dict(T5_MODEL_NOISE["sigma"])}``) so ``t5_case``'s two pinned sides can
    never independently drift -- but nothing asserted that the derivation
    actually held.  A future edit that replaced the ``dict(...)`` copy with a
    literal sigma of its own would silently reopen exactly the C18
    disagreement ``t5_case``'s own docstring says every C16 document is built
    to avoid.  This guards the DERIVATION, not a literal: it must still pass
    whatever number ``T5_MODEL_NOISE["sigma"]`` carries.
    """
    assert T5_LIKELIHOOD_NOISE["sigma"] == T5_MODEL_NOISE["sigma"]
    assert T5_LIKELIHOOD_NOISE["kind"] == "homoscedastic"


def _ids(document) -> frozenset[str]:
    """The post-flight check ids that fired -- the "and nothing else" reader."""
    return frozenset(found.check for found in priced_findings(document))


# --- stand-down --------------------------------------------------------------


def test_stands_down_without_an_adc():
    """No ``model.adc`` -> no C16, and no exception either.

    **Kills** a ``payload.run.twin["adc"]`` read with no presence check and
    no ``except KeyError`` in front of it: the base document lights no
    ``adc`` (measured; ``model:`` here is ``global_signal``/``uniform_sky``/
    ``gain``/``noise`` only), so an unguarded read raises ``KeyError`` and
    ``passes.sweep`` turns that into a hard ``ConfigError`` that would abort
    ``priced_findings`` here rather than let it return quietly.

    The ONE document in this module built by :func:`~tests.config.
    preflight_helpers.preflight_document` rather than :func:`~tests.config.
    preflight_helpers.t5_case`: with no ``adc`` there is no clip to make the
    base document's ``g`` non-affine and no pinned ``model.noise`` to
    disagree with the likelihood, so it needs neither half of ``t5_case``'s
    pin -- and driven through it the 0.5 K likelihood would land on the base
    model's own 0.05 K ``noise`` and earn a C18 refusal out of nowhere.
    """
    document = preflight_document()
    assert "C16" not in _ids(document)


#: A legal, pre-flight-accepted ``model.adc`` that composes two ``ADCOperator``
#: stages by cascade rather than lighting one directly.  ``adc``'s slot kind
#: on ``RADIO_GRAPH`` accepts a ``compose:`` block (measured: this document
#: loads and this test's ``priced_run`` runs the FULL pass, not just this
#: check), and ``payload.run.twin["adc"]`` resolves to a ``Pipeline``
#: (``core/pipeline.py``), not an ``ADCOperator`` -- the type C16's guard
#: checks for.
COMPOSED_ADC = {
    "compose": "cascade",
    "stages": [
        {"name": "first", "type": "ADCOperator",
         "scale": {"value": 1e6, "unit": "dimensionless"}, "n_bits": 12},
        {"name": "second", "type": "ADCOperator",
         "scale": {"value": 1.0, "unit": "dimensionless"}, "n_bits": 12},
    ],
}


def test_stands_down_on_a_composed_adc():
    """BLOCKER (adversarial review): a COMPOSED ``model.adc`` stands down
    instead of taking the whole post-flight pass down with it.

    **Kills** the ``isinstance(node, ADCOperator)`` guard being dropped from
    :func:`~rheplicant.config.postflight.digitising._adc_saturation`: without
    it, ``_saturation_stats`` reads ``node.scale`` off a ``Pipeline``, which
    has no such attribute, and the ``AttributeError`` propagates straight
    through ``passes.sweep`` as ``ConfigError: post-flight check 'C16' RAISED
    AttributeError: 'Pipeline' object has no attribute 'scale'`` -- which
    hides every OTHER priced check's findings on this document, not just
    C16's.  Without the guard, ``_ids`` below -- which runs the whole
    post-flight pass through ``priced_findings`` -- would itself raise
    ``ConfigError`` rather than return a frozenset, so this test is red on
    the whole pass, not just on a missing "C16".
    """
    document = t5_case(model=t5_model(COMPOSED_ADC))
    assert "C16" not in _ids(document)


# --- the two "nothing happened" documents ------------------------------------


def test_the_unsaturated_document_earns_nothing():
    """No latent at all, and no saturation -- C16 is silent.

    **Kills** a check that fires on every document regardless of what the
    probe measures.
    """
    document = t5_case(model=t5_model(ADC_UNSATURATED), inference=None)
    assert "C16" not in _ids(document)


def test_an_unsaturated_document_with_an_upstream_latent_is_silent():
    """The base document AS SHIPPED: latent ``g`` -> ``gain.gain`` (upstream
    of ``adc``), and no saturation.

    **Kills §6's literal reading** -- "refuse when a latent binds upstream
    of adc" taken WITHOUT the ``fraction > 0`` half of the conjunction.  The
    base document's only latent is upstream of ``adc`` (measured: ``gain``
    is one of the 27 of ``RADIO_GRAPH``'s 33 nodes ``digitising.
    _upstream_of_adc()`` returns), so a check that drops the fraction test
    refuses this document, which never saturates at all.
    """
    document = t5_case(model=t5_model(ADC_UNSATURATED))
    assert "C16" not in _ids(document)


# --- the >= boundary, on the base grid ---------------------------------------


@pytest.mark.parametrize("scale, expected_fraction, expected_n",
                         list(zip(T5_BOUNDARY_SCALES,
                                 (0.0, 0.0078125, 0.0078125),
                                 (128, 128, 128), strict=True)))
def test_the_saturation_fraction_uses_ge_not_gt(scale, expected_fraction,
                                                expected_n):
    """The three measured boundary cells, pinned exactly.

    **Kills ``>`` for ``>=``**: at the middle scale the achieved peak is
    EXACTLY the clip limit (2048.0 ``adc_count`` at ``n_bits: 12``), and a
    count that requires strictly-greater reads that cell as fraction 0
    instead of the measured 0.0078125 -- the outer two cells (fraction 0
    just below, 0.0078125 just above) are what proves the middle one is not
    an accident of rounding.
    """
    document = t5_case(model=t5_model({"scale": scale, "n_bits": 12}))
    payload = priced_run(document)
    node = payload.run.twin["adc"]
    peak, fraction, n = digitising._saturation_stats(payload.run.twin,
                                                      payload.run.state, node)
    assert fraction == expected_fraction
    assert n == expected_n


def test_the_refuse_threshold_is_not_tunable_to_let_128_samples_pass():
    """MAJOR (adversarial review): ``_T5_REFUSE_ABOVE`` cannot be raised to
    make the BASE grid's one-clipped-sample-in-128 fraction (0.0078125) land
    inside the WARN band.

    **Kills** ``_T5_REFUSE_ABOVE: 0.001 -> 0.01`` and ``-> 0.5``: at either
    value the base grid's ``1/128`` fraction sits AT OR BELOW the raised
    threshold, so this document -- no latent bound at all -- would earn a
    WARN instead of a REFUSE, and :data:`~tests.config.preflight_helpers.
    WIDE_GRID`'s entire reason to exist (the base grid cannot reach the WARN
    band at ANY scale, so a wider grid is needed to observe it at all)
    evaporates with nothing red to say so.  Same T5_BOUNDARY_SCALES middle
    cell as :func:`test_the_saturation_fraction_uses_ge_not_gt`, a document
    neither merged test built: no ``inference=None`` there, no ``WIDE_GRID``
    here.
    """
    document = t5_case(
        model=t5_model({"scale": T5_BOUNDARY_SCALES[1], "n_bits": 12}),
        inference=None)
    found = priced_only(document, "C16")
    assert found.severity == "refuse"


def test_a_negative_rail_clip_is_counted():
    """MAJOR (adversarial review): a NEGATIVE-going clip counts exactly like
    a positive one.

    **Kills** ``jnp.abs`` dropped from :func:`~rheplicant.config.postflight.
    digitising._saturation_probe`'s capture: every OTHER document in this
    module has a strictly positive base signal (the base document's gain is
    positive), so nothing else here reaches the clip's ``-limit`` rail at
    all.  ``gain: -1.1`` flips the sign of the whole downstream signal
    without changing its magnitude, so the achieved (unclipped) extremes are
    ``-2123.69`` / ``-1542.22`` rather than positive -- reachable, not
    equivalent: ``abs`` dropped reads every one of those as negative and
    never ``>= limit``, so the mutant reports fraction 0.0 on a document that
    genuinely clips.
    """
    document = t5_case(
        model={**t5_model({"scale": T5_BOUNDARY_SCALES[1], "n_bits": 12}),
              "gain": {"gain": {"value": -1.1, "unit": "dimensionless"}}})
    payload = priced_run(document)
    node = payload.run.twin["adc"]
    _peak, fraction, _n = digitising._saturation_stats(
        payload.run.twin, payload.run.state, node)
    assert fraction == 0.0234375
    assert priced_only(document, "C16").severity == "refuse"


# --- the escalator, on the wide grid where its band is reachable -------------


def test_a_saturating_document_with_an_upstream_latent_refuses():
    """``WIDE_GRID`` at the measured warn cell, WITH the base document's
    latent (``g`` -> ``gain.gain``, upstream of ``adc``) still bound.

    Fraction here (0.00048828125) sits inside the escalator's ``(0, 0.001]``
    band, where the escalator is the only thing that can move the severity
    -- see :data:`~tests.config.preflight_helpers.WIDE_GRID`'s docstring for
    why the base grid cannot reach this band at all.
    """
    document = t5_case(
        observation=WIDE_GRID,
        model=t5_model({"scale": T5_WIDE_WARN_SCALE, "n_bits": 12}))
    found = priced_only(document, "C16")
    assert found.severity == "refuse"
    assert "bound latent" in found.message


def test_a_saturating_document_with_no_upstream_latent_warns_below_the_threshold():
    """The SAME ``WIDE_GRID`` cell, with the latent removed entirely
    (``inference=None``) -- the escalator's other half.

    **This is also P7's replacement for
    ``test_a_warn_needs_more_than_a_thousand_samples``**: measured, that test
    written on ``WIDE_GRID`` with the base document's only latent removed is
    byte-for-byte this document and this assertion, so it is not written a
    second time -- see this module's docstring.

    Together with :func:`test_a_saturating_document_with_an_upstream_latent_
    refuses` this is the escalator's CONJUNCTION made visible: same fraction,
    same threshold, and severity moves only because a latent's ``into:``
    does or does not reach a node upstream of ``adc``.
    """
    document = t5_case(
        observation=WIDE_GRID,
        model=t5_model({"scale": T5_WIDE_WARN_SCALE, "n_bits": 12}),
        inference=None)
    found = priced_only(document, "C16")
    assert found.severity == "warn"
    assert "bound latent" not in found.message


def test_a_binding_counts_as_well_as_an_into():
    """The escalator's twin route: ``inference.bindings[].into`` rather than
    ``inference.parameters.g.into``.

    Same node (``gain.gain``), same measured cell as the two tests above,
    the latent reached the other of the two ways ``_t11_bindings`` walks.
    **Kills** an escalator that reads only ``inference.parameters.<n>.into``
    and misses ``bindings:`` -- STANDING-RULES.md's "hunt twins".
    """
    document = t5_case(
        observation=WIDE_GRID,
        model=t5_model({"scale": T5_WIDE_WARN_SCALE, "n_bits": 12}),
        inference=T5_BINDING_LATENT)
    found = priced_only(document, "C16")
    assert found.severity == "refuse"
    assert "bound latent" in found.message


#: The SAME latent as :data:`~tests.config.preflight_helpers.
#: T5_BINDING_LATENT`, reached by the OTHER spelling P19 forbids: a LIST
#: ``into:`` on a ``parameters.<n>`` entry, rather than a plain string.
#: ``preflight/model.py::_t11_bindings`` normalises both spellings of
#: ``into:`` -- a string or a list of strings -- so this is the twin route a
#: check reading only the string spelling would miss.
LIST_INTO = {
    "parameters": {"g": {"init": 1.0, "linear": True, "into": ["gain.gain"],
                         "prior": {"normal": {"loc": 1.0, "scale": 0.5}}}},
    "noise": HOMOSCEDASTIC,
    "observed": {"from": "simulation", "at": {"g": TRUTH_G}, "twin": "full"},
}


def test_a_list_form_into_reaches_the_escalator():
    """MAJOR (adversarial review): a LIST-form ``into:`` still reaches the
    escalator -- P19's hazard, alive behind the ``_t11_bindings`` reuse.

    **Kills** :func:`~rheplicant.config.postflight.digitising.
    _escalating_latents`'s body replaced with the naive ``parse_path(into)[0]``
    walk P19 forbids: that mutant is REACHABLE, not merely wrong in theory --
    it raises ``ConfigError: post-flight check 'C16' RAISED ConfigError: A
    path must be a string; got list (['gain.gain'])`` on THIS document,
    aborting the whole pass exactly like the BLOCKER above.  On HEAD, the
    list form works correctly because ``_t11_bindings`` is reused rather than
    re-walked.  Task 5 was explicitly blocked on P19; this is the regression
    test that keeps the next editor from reopening it with the suite still
    green.
    """
    document = t5_case(
        observation=WIDE_GRID,
        model=t5_model({"scale": T5_WIDE_WARN_SCALE, "n_bits": 12}),
        inference=LIST_INTO)
    found = priced_only(document, "C16")
    assert found.severity == "refuse"


# --- the message --------------------------------------------------------------


def test_the_message_names_scale_bits_peak_and_the_unit():
    """``adc.scale``, ``adc.n_bits``, the measured peak, and the literal
    string ``adc_count`` are all in the message.

    **Kills** a message reporting the peak in kelvin -- the number a reader
    of this layer has no calibration for; ``adc_count`` is the raw digitiser
    unit the clip limit and the peak share.
    """
    document = t5_case(model=t5_model(ADC_SATURATING))
    found = priced_only(document, "C16")
    assert "adc.scale" in found.message
    assert "adc.n_bits" in found.message
    assert "adc_count" in found.message

    payload = priced_run(document)
    node = payload.run.twin["adc"]
    peak, _fraction, _n = digitising._saturation_stats(
        payload.run.twin, payload.run.state, node)
    assert f"{peak:.6g}" in found.message
    assert f"of {_n} samples" in found.message
    assert "(check C16)" in found.message


def test_the_peak_is_the_maximum_magnitude_independently_measured():
    """MAJOR (adversarial review): ``peak`` is pinned against a SECOND,
    independent forward pass, not just against what ``_saturation_stats``
    itself returns.

    **Kills** ``jnp.max`` -> ``jnp.mean`` and ``jnp.max`` -> ``jnp.min``,
    both of which survive the whole module: every OTHER test that reads
    ``peak`` gets it BY CALLING ``_saturation_stats``, the function under
    test, which makes the assertion self-consistent under any mutation to
    that function's own reduction.  Here the peak is instead re-derived by
    swapping in a real ``ADCOperator`` at ``n_bits=40`` (clip limit
    ``2**39``, far outside anything this document's signal reaches, so
    nothing clips) and reading ``max(abs(out))`` directly off its output --
    a computation that does not go through ``_saturation_stats`` at all.
    """
    document = t5_case(model=t5_model({"scale": 1.0, "n_bits": 12}),
                                  inference=None)
    payload = priced_run(document)
    node = payload.run.twin["adc"]
    unclipped = payload.run.twin.replace_node(
        "adc", ADCOperator(scale=node.scale, n_bits=40))(payload.run.state).data
    peak, _fraction, _n = digitising._saturation_stats(
        payload.run.twin, payload.run.state, node)
    assert peak == float(jnp.max(jnp.abs(unclipped)))
    assert peak == pytest.approx(12.116166, abs=5e-6)


def test_where_is_model_adc():
    """MAJOR (adversarial review): ``found.where`` is pinned to the literal
    subject path.

    **Kills** ``where = "model.adc"`` -> ``where = "model"``: both are
    strings a substring assertion on the message would not catch, and no
    other test in this module reads ``found.where`` at all.
    """
    document = t5_case(model=t5_model(ADC_SATURATING))
    found = priced_only(document, "C16")
    assert found.where == "model.adc"


def test_the_message_reports_the_achieved_scale_and_bits():
    """MAJOR (adversarial review): the message's FACT-REPORTING half is
    pinned by exact substring, not just by the field NAME.

    **Kills** ``adc.scale=`` -> ``scale=`` and ``adc.n_bits=`` -> ``n_bits=``:
    both survive an ``assert "adc.scale" in found.message`` /
    ``assert "adc.n_bits" in found.message``, because the ADVICE tail
    ("Lower **adc.scale** or raise **adc.n_bits**...") still contains the
    unqualified word and satisfies the substring check even with the
    fact-reporting prefix mutated away.  Pinning the ``=`` and the achieved
    VALUE together is what the advice tail cannot also satisfy.
    """
    document = t5_case(model=t5_model(ADC_SATURATING))
    found = priced_only(document, "C16")
    assert "adc.scale=1e+06" in found.message
    assert "adc.n_bits=12" in found.message


def test_the_message_reports_the_measured_fraction():
    """MAJOR (adversarial review): the printed fraction is the ONE that was
    actually measured.

    **Kills** ``{fraction}`` -> ``{fraction * 2}`` in the message
    f-string: ``0.0488%`` is ``0.00048828125`` (this cell's measured
    fraction, ``1/2048``) formatted ``.4%``; doubled it would print
    ``0.0977%``, which no assertion in this module's other tests would
    notice since they check severity and the presence of ``"bound latent"``,
    never the number itself.
    """
    document = t5_case(
        observation=WIDE_GRID,
        model=t5_model({"scale": T5_WIDE_WARN_SCALE, "n_bits": 12}),
        inference=None)
    found = priced_only(document, "C16")
    assert "0.0488%" in found.message


def test_the_advice_names_a_target_and_applying_it_clears_the_refusal():
    """MINOR (adversarial review): the advice names a NUMBER, and that
    number, applied literally, clears the refusal -- STANDING-RULES.md
    section D, the advice-loop rule, applied to the target rather than to
    the two knob names.

    Not a wave-1 advice loop (both merged advice-loop tests already prove
    the two KNOBS clear a refusal): this instead pins that the TARGET named
    in the message is the one :func:`~rheplicant.config.postflight.
    digitising._t5_target_scale`/:func:`~rheplicant.config.postflight.
    digitising._t5_target_bits` compute, and that going one step past either
    target (a small margin below the scale target, one more than the bits
    target) actually silences C16 -- proving the named number is not
    decorative.
    """
    document = t5_case(model=t5_model(ADC_SATURATING))
    found = priced_only(document, "C16")
    payload = priced_run(document)
    node = payload.run.twin["adc"]
    peak, _fraction, _n = digitising._saturation_stats(
        payload.run.twin, payload.run.state, node)
    target_scale = digitising._t5_target_scale(peak, float(node.scale),
                                                digitising._t5_limit(node.n_bits))
    target_bits = digitising._t5_target_bits(peak, node.n_bits)
    assert f"below {target_scale:.6g}" in found.message
    assert f"at least {target_bits}" in found.message

    cleared_by_scale = t5_case(
        model=t5_model({"scale": target_scale * 0.99, "n_bits": 12}))
    assert "C16" not in _ids(cleared_by_scale)

    cleared_by_bits = t5_case(
        model=t5_model({"scale": 1e6, "n_bits": target_bits + 1}))
    assert "C16" not in _ids(cleared_by_bits)


# --- the probe leaves the model alone -----------------------------------------


def test_the_probe_does_not_change_the_model():
    """The twin's output is IDENTICAL with and without the probe swapped in.

    **Kills** a pass-through probe that drops the scale or the clip (returns
    ``state`` instead of ``super().__call__(state)``): every node downstream
    of ``adc`` would then see a different signal, and this check would be
    measuring a model nobody's forward pass actually runs.

    **Also asserts the captured value is a concrete array, not a tracer**
    (MINOR, adversarial review) -- cheap insurance for the day someone wraps
    this eager call in ``jit``: a tracer captured into ``captured`` would
    outlive the trace it came from, and ``_saturation_stats`` would then be
    reading a dead reference rather than a number.  This module's own
    docstring already says the eager-only premise out loud; this is what
    would go red the day it stops being true.
    """
    document = t5_case(model=t5_model(ADC_SATURATING))
    payload = priced_run(document)
    twin, state = payload.run.twin, payload.run.state
    node = twin["adc"]
    probe, captured = digitising._saturation_probe(node)

    original = twin(state).data
    probed = twin.replace_node("adc", probe)(state).data
    assert jnp.array_equal(original, probed)
    assert not isinstance(captured[0], jax.core.Tracer)


# --- the premise, pinned as a regression --------------------------------------


def test_the_dead_gradient_is_real():
    """``d(sum(prediction))/d(gain)`` is exactly 0.0 once the ADC saturates
    every sample, and non-zero when it saturates none.

    **This is a PACKAGE-LEVEL PREMISE regression, not a C16 exerciser**
    (confirmed, adversarial review's mutation sweep: this test is not among
    the tests that go red when C16's own check body -- ``_adc_saturation``,
    ``_saturation_stats``, the escalator -- is gutted).  It differentiates
    the FULL twin directly, through ``eqx.tree_at``-style node replacement,
    the way a real fit would differentiate through ``gain``, and calls no
    part of this module at all.  **Kills** the day someone "fixes" the clip
    into a soft one and C16's whole REASON TO EXIST (a saturated sample has
    no gradient, so a fit through it is silently blind) evaporates with the
    rest of this module's suite still green and none the wiser -- which is a
    real defect for this test to catch even though it is not exercising the
    check's own code path.
    """

    def gradient_at(scale: float) -> float:
        document = t5_case(model=t5_model({"scale": scale,
                                                      "n_bits": 12}))
        payload = priced_run(document)
        twin, state = payload.run.twin, payload.run.state
        current_gain = twin["gain"].gain

        def total(gain_value):
            varied = twin.replace_node("gain", GainOperator(gain=gain_value))
            return jnp.sum(varied(state).data)

        return float(jax.grad(total)(current_gain))

    assert gradient_at(1e6) == 0.0
    assert gradient_at(1.0) != 0.0


# --- the arithmetic ADCOperator itself owns -----------------------------------


@pytest.mark.parametrize("n_bits", list(range(1, 25)))
def test_t5_limit_matches_the_operators_own_clip_point(n_bits):
    """``_t5_limit`` agrees with what ``ADCOperator`` actually clips to.

    Driven against the OPERATOR's observed behaviour rather than against a
    second ``2.0 ** (n_bits - 1)`` literal written here: this is the "assert
    the two agree ... rather than pinning the arithmetic twice" rule from
    this task's contract, applied over the operator's whole legal range.
    """
    op = ADCOperator(scale=jnp.array(1.0), n_bits=n_bits)
    driven_far_outside = State(data=jnp.array([1e12, -1e12]))
    out = op(driven_far_outside)
    limit = digitising._t5_limit(n_bits)
    assert float(jnp.max(out.data)) == limit
    assert float(jnp.min(out.data)) == -limit


def test_upstream_of_adc_is_27_of_33_nodes():
    """The measurement this whole module's escalator design rests on.

    Pinned as a regression: if ``RADIO_GRAPH`` ever changes shape around
    ``adc``, this is what tells the next reader the escalator's own
    docstring claim ("27 of 33") needs re-measuring too.
    """
    from rheplicant.radio.graph import RADIO_GRAPH

    upstream = digitising._upstream_of_adc()
    assert len(upstream) == 27
    assert len(RADIO_GRAPH.nodes) == 33
    assert "gain" in upstream
    assert "adc" not in upstream


# --- the advice loop -----------------------------------------------------------


def test_advice_lowering_scale_clears_the_refusal():
    """C16's own advice, applied literally: lower ``adc.scale``.

    STANDING-RULES.md section D -- a refusal's advice is not sanity-checked
    by reading it, it is APPLIED, and the document is then asserted to load.
    """
    saturating = t5_case(model=t5_model(ADC_SATURATING))
    with pytest.raises(ConfigError):
        load_document(saturating)

    lowered = t5_case(model=t5_model({"scale": 1.0, "n_bits": 12}))
    run = load_document(lowered)
    assert "C16" not in {found.check for found in run.report.findings}


def test_advice_raising_n_bits_clears_the_refusal():
    """C16's other advice, applied literally: raise ``adc.n_bits``."""
    saturating = t5_case(model=t5_model(ADC_SATURATING))
    with pytest.raises(ConfigError):
        load_document(saturating)

    raised = t5_case(model=t5_model({"scale": 1e6, "n_bits": 32}))
    run = load_document(raised)
    assert "C16" not in {found.check for found in run.report.findings}


# --- every document here is one load_document accepts -------------------------
#
# The guard the wave-3/4 boundary defect needed and did not have.  Only the
# two advice-loop tests above call ``load_document``; every other test reaches
# the check through ``priced_only``/``priced_findings``/``_ids``, which go via
# ``_assemble`` and therefore OBSERVE a refusal rather than raise on it.  So
# this module stayed green for a whole wave while every document it built
# carried two refusals that have nothing to do with C16 -- measured on the
# merged tree, ``[('C12', 'refuse'), ('C18', 'refuse')]`` on all fifteen of the
# shapes below that declare a latent.  A C16 test whose document is refused for
# an unrelated reason is a test whose subject nobody can reach.


def _refusal_checks(document) -> frozenset[str]:
    """The check ids that would STOP ``load_document`` on ``document``.

    ``load_document`` raises on the first refusal, so a refused document's
    report is not reachable through it at all.  ``priced_run`` re-walks the
    same passes with neither raising hook -- it CARRIES the built report
    instead of raising it (``inflight_helpers``' own docstring) -- and
    ``priced`` runs the post-flight pass over the result, so the two together
    are exactly the findings ``load_document`` consults.

    Both halves are read, not just the post-flight one: C12, C16 and C18 all
    happen to live in the post-flight pass today (measured: ``run.report`` is
    empty on every shape below, before and after this fix), and a reader that
    took only ``priced`` would go blind the day a built-slot check refuses one
    of these documents -- which is the same shape of blindness this test
    exists to close.
    """
    payload = priced_run(document)
    return frozenset(found.check
                     for found in (*payload.run.report.refusals(),
                                   *priced(payload).refusals()))


#: Every document shape this module builds, and the refusals its report is
#: allowed to carry.  Pinned from a measurement rather than re-derived by
#: asking the check -- an expectation computed by running C16 would agree with
#: any C16 whatsoever.
#:
#: The three cells whose ``adc.scale`` this module COMPUTES rather than writes
#: (``_t5_target_scale``'s output in
#: :func:`test_the_advice_names_a_target_and_applying_it_clears_the_refusal`)
#: are not rows here -- they cannot be, since the scale is not known until the
#: check has run -- but they are the same shape as ``scale 1.0`` below with a
#: different number in it, and that test's own ``_ids`` reading covers them.
#:
#: **A new test that builds a shape not listed here should add a row.**  The
#: table is what makes "and nothing else" mean the whole document rather than
#: only the post-flight check under test.
_SHAPES: dict[str, tuple[dict, frozenset[str]]] = {
    "no adc at all": ({}, frozenset()),
    "composed adc": ({"model": t5_model(COMPOSED_ADC)}, frozenset()),
    "unsaturated, no inference": ({"model": t5_model(ADC_UNSATURATED),
                                   "inference": None}, frozenset()),
    "unsaturated, base latent": ({"model": t5_model(ADC_UNSATURATED)},
                                 frozenset()),
    "boundary below": ({"model": t5_model({"scale": T5_BOUNDARY_SCALES[0],
                                           "n_bits": 12})}, frozenset()),
    "boundary at": ({"model": t5_model({"scale": T5_BOUNDARY_SCALES[1],
                                        "n_bits": 12})}, frozenset({"C16"})),
    "boundary above": ({"model": t5_model({"scale": T5_BOUNDARY_SCALES[2],
                                           "n_bits": 12})}, frozenset({"C16"})),
    "boundary at, no inference": ({"model": t5_model(
        {"scale": T5_BOUNDARY_SCALES[1], "n_bits": 12}),
        "inference": None}, frozenset({"C16"})),
    "negative rail": ({"model": {**t5_model({"scale": T5_BOUNDARY_SCALES[1],
                                             "n_bits": 12}),
                                 "gain": {"gain": {"value": -1.1,
                                                   "unit": "dimensionless"}}}},
                      frozenset({"C16"})),
    "wide, base latent": ({"observation": WIDE_GRID,
                           "model": t5_model({"scale": T5_WIDE_WARN_SCALE,
                                              "n_bits": 12})},
                          frozenset({"C16"})),
    "wide, no inference (the WARN cell)": (
        {"observation": WIDE_GRID,
         "model": t5_model({"scale": T5_WIDE_WARN_SCALE, "n_bits": 12}),
         "inference": None}, frozenset()),
    "wide, bindings[] latent": ({"observation": WIDE_GRID,
                                 "model": t5_model(
                                     {"scale": T5_WIDE_WARN_SCALE,
                                      "n_bits": 12}),
                                 "inference": T5_BINDING_LATENT},
                                frozenset({"C16"})),
    "wide, list-form into": ({"observation": WIDE_GRID,
                              "model": t5_model({"scale": T5_WIDE_WARN_SCALE,
                                                 "n_bits": 12}),
                              "inference": LIST_INTO}, frozenset({"C16"})),
    "saturating": ({"model": t5_model(ADC_SATURATING)}, frozenset({"C16"})),
    "saturating, no inference": ({"model": t5_model(ADC_SATURATING),
                                  "inference": None}, frozenset({"C16"})),
    "scale 1.0, no inference": ({"model": t5_model({"scale": 1.0,
                                                    "n_bits": 12}),
                                 "inference": None}, frozenset()),
    "scale 1.0": ({"model": t5_model({"scale": 1.0, "n_bits": 12})},
                  frozenset()),
    "n_bits raised to 32": ({"model": t5_model({"scale": 1e6, "n_bits": 32})},
                            frozenset()),
}


@pytest.mark.parametrize("name", list(_SHAPES))
def test_no_document_here_carries_a_refusal_that_is_not_c16s(name):
    """The guard that would have caught the wave-3/4 defect AT THE BRANCH.

    Every shape this module builds is loaded, and its refusal set is asserted
    EQUAL to the C16 one under test -- empty for the cells that pass, and
    ``{"C16"}`` and nothing beside it for the cells that refuse.

    **Kills** :func:`~tests.config.preflight_helpers.t5_case`'s two pins,
    separately and measurably, and the two are told apart by WHICH refusal
    comes back.  Drop ``noise`` from its ``repatch`` and the twelve
    latent-carrying rows fail at ``frozenset({'C18'}) == frozenset()`` -- the
    base document's likelihood sigma (``HOMOSCEDASTIC``, 0.05 K) against
    ``t5_model``'s pinned ``model.noise`` (0.5 K), the disagreement C18 is
    for.  Drop ``checks`` and the same twelve fail at
    ``frozenset({'C12'}) == frozenset()`` -- ``check_linearity`` probes at
    1000x the latent's scale, where every one of these converters clips, so
    the prediction genuinely is not affine in ``g`` and the claim genuinely
    does not hold.  Measured, each mutant: **14 failed, 53 passed**; the four
    ``inference=None`` rows survive both (no latent to declare, no likelihood
    sigma to disagree), and the only two casualties outside this test are
    ``test_advice_lowering_scale_clears_the_refusal`` and
    ``test_advice_raising_n_bits_clears_the_refusal`` -- the two that broke
    when the wave merged, and the ONLY two of this module's forty-eight
    others that call ``load_document`` at all.  That is the blind spot stated
    as a number: forty-six tests could not see either refusal.

    **The WARNINGS are held to the same standard**, and not by a second
    column in the table: a document that loads still emits one
    ``ConfigWarning`` per warning finding, and an unrelated one riding along
    is the same defect one severity down.  The one cell that warns is
    ``WIDE_GRID`` with no latent -- the escalator's WARN half -- so its own
    C16 warning is admitted by name and anything else is not.
    """
    patch, expected = _SHAPES[name]
    # The empty patch is the no-adc row, and it is the ONE document t5_case
    # must not build -- see test_stands_down_without_an_adc.
    document = t5_case(**patch) if patch else preflight_document()
    assert _refusal_checks(document) == expected
    if expected:
        with pytest.raises(ConfigError):
            load_document(document)
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_document(document)
    assert [str(one.message) for one in caught
            if issubclass(one.category, ConfigWarning)
            and "(check C16)" not in str(one.message)] == []


def test_the_shape_table_holds_both_kinds_of_cell():
    """Anti-vacuity: the table above cannot be emptied, or made all-passing,
    with the parametrized test still green.

    Every assertion in :func:`test_no_document_here_carries_a_refusal_that_is_
    not_c16s` is of the form "this document carries what it should"; a table
    trimmed to its passing rows satisfies all of them while covering none of
    the documents the defect was actually on.
    """
    expectations = [expected for _patch, expected in _SHAPES.values()]
    assert expectations.count(frozenset({"C16"})) >= 8
    assert expectations.count(frozenset()) >= 6
