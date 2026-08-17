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

**``test_stands_down_on_a_many_or_composed_adc`` is DELETED, not written at
all** (D-20 / this task's brief §4.1).  ``adc``'s ``NodeSpec`` on
``RADIO_GRAPH`` (``radio/graph.py:157``) never sets ``many=True``, and
``NodeSpec.many`` defaults to ``False`` (``core/graph.py:122``), so
``Assembly.__getitem__`` (``core/graph.py:455-471``) can never raise
``AmbiguousNodeError`` for this node -- that branch fires only for a node
holding SEVERAL instances, which a ``many=False`` node structurally cannot.
The only stand-down this node can earn is the plain ``KeyError`` an absent
``adc`` raises, and that is :func:`test_stands_down_without_an_adc` below.

**Boundary numbers are measured, not copied from the plan** -- see this
module's own assertions and
``digitising._t5_limit``/``digitising._saturation_stats``, which every
boundary test below calls directly rather than re-deriving the arithmetic a
second time.
"""

import jax
import jax.numpy as jnp
import pytest

from rheplicant.config.document import load_document
from rheplicant.config.errors import ConfigError
from rheplicant.config.postflight import digitising
from rheplicant.core.state import State
from rheplicant.radio.instrument.adc import ADCOperator
from rheplicant.radio.instrument.gain import GainOperator
from tests.config.inflight_helpers import priced_findings, priced_only, priced_run
from tests.config.preflight_helpers import (
    ADC_SATURATING,
    ADC_UNSATURATED,
    T5_BINDING_LATENT,
    T5_BOUNDARY_SCALES,
    T5_WIDE_WARN_SCALE,
    WIDE_GRID,
    preflight_document,
)


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
    """
    document = preflight_document()
    assert "C16" not in _ids(document)


# --- the two "nothing happened" documents ------------------------------------


def test_the_unsaturated_document_earns_nothing():
    """No latent at all, and no saturation -- C16 is silent.

    **Kills** a check that fires on every document regardless of what the
    probe measures.
    """
    document = preflight_document(model={"adc": ADC_UNSATURATED}, inference=None)
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
    document = preflight_document(model={"adc": ADC_UNSATURATED})
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
    document = preflight_document(model={"adc": {"scale": scale, "n_bits": 12}})
    payload = priced_run(document)
    node = payload.run.twin["adc"]
    peak, fraction, n = digitising._saturation_stats(payload.run.twin,
                                                      payload.run.state, node)
    assert fraction == expected_fraction
    assert n == expected_n


# --- the escalator, on the wide grid where its band is reachable -------------


def test_a_saturating_document_with_an_upstream_latent_refuses():
    """``WIDE_GRID`` at the measured warn cell, WITH the base document's
    latent (``g`` -> ``gain.gain``, upstream of ``adc``) still bound.

    Fraction here (0.00048828125) sits inside the escalator's ``(0, 0.001]``
    band, where the escalator is the only thing that can move the severity
    -- see :data:`~tests.config.preflight_helpers.WIDE_GRID`'s docstring for
    why the base grid cannot reach this band at all.
    """
    document = preflight_document(
        observation=WIDE_GRID,
        model={"adc": {"scale": T5_WIDE_WARN_SCALE, "n_bits": 12}})
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
    document = preflight_document(
        observation=WIDE_GRID,
        model={"adc": {"scale": T5_WIDE_WARN_SCALE, "n_bits": 12}},
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
    document = preflight_document(
        observation=WIDE_GRID,
        model={"adc": {"scale": T5_WIDE_WARN_SCALE, "n_bits": 12}},
        inference=T5_BINDING_LATENT)
    found = priced_only(document, "C16")
    assert found.severity == "refuse"
    assert "bound latent" in found.message


# --- the message --------------------------------------------------------------


def test_the_message_names_scale_bits_peak_and_the_unit():
    """``adc.scale``, ``adc.n_bits``, the measured peak, and the literal
    string ``adc_count`` are all in the message.

    **Kills** a message reporting the peak in kelvin -- the number a reader
    of this layer has no calibration for; ``adc_count`` is the raw digitiser
    unit the clip limit and the peak share.
    """
    document = preflight_document(model={"adc": ADC_SATURATING})
    found = priced_only(document, "C16")
    assert "adc.scale" in found.message
    assert "adc.n_bits" in found.message
    assert "adc_count" in found.message

    payload = priced_run(document)
    node = payload.run.twin["adc"]
    peak, _fraction, _n = digitising._saturation_stats(
        payload.run.twin, payload.run.state, node)
    assert f"{peak:.6g}" in found.message


# --- the probe leaves the model alone -----------------------------------------


def test_the_probe_does_not_change_the_model():
    """The twin's output is IDENTICAL with and without the probe swapped in.

    **Kills** a pass-through probe that drops the scale or the clip (returns
    ``state`` instead of ``super().__call__(state)``): every node downstream
    of ``adc`` would then see a different signal, and this check would be
    measuring a model nobody's forward pass actually runs.
    """
    document = preflight_document(model={"adc": ADC_SATURATING})
    payload = priced_run(document)
    twin, state = payload.run.twin, payload.run.state
    node = twin["adc"]
    probe, _captured = digitising._saturation_probe(node)

    original = twin(state).data
    probed = twin.replace_node("adc", probe)(state).data
    assert jnp.array_equal(original, probed)


# --- the premise, pinned as a regression --------------------------------------


def test_the_dead_gradient_is_real():
    """``d(sum(prediction))/d(gain)`` is exactly 0.0 once the ADC saturates
    every sample, and non-zero when it saturates none.

    Independent of the check itself -- this differentiates the FULL twin
    directly, through ``eqx.tree_at``-style node replacement, the way a real
    fit would differentiate through ``gain``.  **Kills** the day someone
    "fixes" the clip into a soft one and this check's whole premise
    (a saturated sample has no gradient) evaporates with the rest of the
    suite green.
    """

    def gradient_at(scale: float) -> float:
        document = preflight_document(model={"adc": {"scale": scale,
                                                      "n_bits": 12}})
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
    saturating = preflight_document(model={"adc": ADC_SATURATING})
    with pytest.raises(ConfigError):
        load_document(saturating)

    lowered = preflight_document(model={"adc": {"scale": 1.0, "n_bits": 12}})
    run = load_document(lowered)
    assert "C16" not in {found.check for found in run.report.findings}


def test_advice_raising_n_bits_clears_the_refusal():
    """C16's other advice, applied literally: raise ``adc.n_bits``."""
    saturating = preflight_document(model={"adc": ADC_SATURATING})
    with pytest.raises(ConfigError):
        load_document(saturating)

    raised = preflight_document(model={"adc": {"scale": 1e6, "n_bits": 32}})
    run = load_document(raised)
    assert "C16" not in {found.check for found in run.report.findings}
