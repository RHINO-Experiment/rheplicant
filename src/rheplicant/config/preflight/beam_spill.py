"""Pre-flight: the horizon split asked of a projector that cannot supply it.

Check A50, two legs, and **both are inventions** -- neither has a sentence
anywhere in this package to hoist, which is the reason the row exists.

**Leg A is the silent one.**  ``horizon.mode: truncate_map`` cuts the beam at
the horizon; ``beam_spill: {from: projector}`` then asks the projector built
on that beam for ``horizon_fraction()``.  The two are the same physical
statement said twice, and saying it twice does not double the ground -- it
deletes it.  **Nothing is raised, anywhere.**  Every array comes back finite,
correctly shaped, in the right unit and with the right dtype; the only symptom
is a ground term that is not there.

**The four numbers leg A's message quotes, and the whole recipe that produces
them.**  They are RECORDED here rather than asserted by a test, and the recipe
is written out in full because a figure in a user-facing refusal that nobody
can re-derive is a figure that becomes a lie in silence.  Measured at
``ea4839b``::

    resources:
      beams:
        horn: {format: gaussian, sigma_deg: {value: 40, unit: deg},
               nside: 8, normalize: pixel_sum, frame: beam_local}
               # run twice: once as written, once with the line below added
               # horizon: {mode: truncate_map}
      projectors:
        p:    {engine: driftscan, beam: {ref: resources.beams.horn},
               lmax: 16, lat_deg: {value: 53.2, unit: deg},
               az_deg: {value: 0.0, unit: deg},
               el_deg: {value: 90.0, unit: deg},
               normalize_beam: true, horizon_mask: true,
               acknowledge_float32_sky: true}

through ``build_resources`` on a two-channel float64
:class:`~rheplicant.config.context.ResolutionContext`, then
``DriftScanProjector.horizon_fraction()`` on the built projector::

    horizon.mode absent          f_sky = 0.9510017    (1-f)*300 K = 14.699489
    horizon.mode truncate_map    f_sky = 0.99923813   (1-f)*300 K =  0.22855997

**Why it is not a test.**  The two builds cost **~9 s** together -- an
nside-8 map through ``map2alm`` at lmax 16, twice -- which is four times
§8's ~2 s per-test bound, and a guard over §8's bound is a stop-and-ask
rather than an implementer's call.  No fixture in
``test_preflight_beam_spill.py`` is gaussian, nside 8 or lmax 16 (its beam is
an nside-4 ``npy`` under an lmax-8 projector, which is what keeps that module
under half a second), so the recipe above is the only way back to these
numbers and it is deliberately complete: the beam WIDTH is the parameter an
earlier draft of this docstring left out, and without it the figures cannot be
reproduced at all.

**Leg B raises, late and in another vocabulary.**  A projector built with
``optimizations: [cache_beam_rotation]`` has ``beam_frame='reference'``, and
``DriftScanProjector.horizon_fraction()`` refuses that state -- the mask is
folded into the cached alms and the unmasked denominator is gone.  Measured:
the document dies inside ``build_model`` with a ``StateValidationError``,
after the beam has been read, analysed and rotated, and the message names
neither ``model.beam_spill`` nor the projector's ``optimizations:`` key.
``ConfigError`` and ``StateValidationError`` are SIBLINGS (§0.2 C-12), so a
caller catching the first does not see the second.

**Because both are invented, both messages are written in this layer's voice,
end with ``(check A50).``, and are pinned by equality on their whole text.**
There is no one-binding row for A50 in ``message_binding.py``: there is
nothing for it to be bound once against.

**Three things this check deliberately does NOT say.**

* It does not refuse ``horizon.mode`` on its own.  Three modes exist, the
  projector carries its own independent ``horizon_mask:``, and
  ``projector_mask`` beside ``beam_spill.from: projector`` is the CORRECT
  combination.  Only ``truncate_map`` double-counts.
* It follows ``model.beam_spill.projector.ref`` to
  ``resources.projectors.<p>.optimizations``, never to ``beam_frame``.  That
  key is in ``kinds/projectors.py::_NOT_WRITABLE`` and is already refused as a
  document key by a message that names the optimisation to ask for instead.
* It says nothing about ``{from: horizon_fraction, projector: {ref: ...}}`` in
  a value node.  That spelling has its own refusal
  (``kinds/projectors.py::_horizon_fraction``), and two sentences about one
  line is what §2.3 exists to stop.

**Engine gate: ``driftscan`` only.**  ``BeamSpillOperator.from_projector``
already refuses every other engine by name and by class -- measured,
``MatrixProjector does not expose horizon_fraction()`` and the same for
``GeneralPointingProjector`` -- and that sentence is better than anything this
layer could write about a projector kind that has no horizon cut at all.

**The route ``preflight/model.py::_nodes`` cannot see IS walked here**
(§0.3 E.10).  ``inference.twin.replace.beam_spill`` reaches the same
``build_node_operator`` -> ``_from_route`` -> ``from_projector`` path, so both
legs fire on it, and the ``where`` and the message name that path rather than
``model.beam_spill``.  Walking it is affordable precisely because A50 is
invented: a verbatim hoist would have carried the wrong section name into the
sentence.  ``preflight/model.py::_t4_entries`` supplies the other route a
single node has, ``compose: {stages: [...]}``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rheplicant.config.findings import Finding, refuse
from rheplicant.config.preflight import register
from rheplicant.config.preflight.document import _task3_over_layers, _task3_where
from rheplicant.config.preflight.model import _t4_entries
from rheplicant.config.resources import resolved_specs

#: The one node this module is about.  ``beam_spill`` is the trunk stage of
#: the astro branch and is NOT a ``many`` node (``radio/graph.py``), which is
#: what ``_t4_entries(..., many=False)`` below is told.
_B2_NODE = "beam_spill"

#: The optimisation that puts a projector in the reference frame, and the
#: prefix a projector reference has to start with to be one.
_B2_CACHED = "cache_beam_rotation"
_B2_PROJECTORS = "resources.projectors."
_B2_BEAMS = "resources.beams."


def _b2_entry(reference: Any, prefix: str) -> str | None:
    """``{ref: resources.<kind>.<name>...}`` -> the three-segment dotted name.

    Cut to three segments the way ``resources.py::_referenced_names`` cuts,
    so that a sub-value reference (``{ref: resources.beams.horn.maps}``)
    resolves to the ENTRY it reads out of.  ``None`` for anything that is not
    a mapping carrying a string ``ref:`` under ``prefix`` -- every one of
    those shapes has its own refusal in ``sections/model.py``, and this pass
    has nothing better to say.
    """
    if not isinstance(reference, Mapping):
        return None
    target = reference.get("ref")
    if not isinstance(target, str) or not target.startswith(prefix):
        return None
    parts = target.split(".")
    if len(parts) < 3:
        return None
    return ".".join(parts[:3])


def _b2_spill_entries(layer: Mapping[str, Any]) -> Iterable[tuple[str, Any]]:
    """``(document path, node spec)`` for every ``beam_spill`` in one layer.

    Both routes to ``build_node_operator``: the ``model:`` key and
    ``inference.twin.replace``.  ``_t4_entries`` expands the third,
    ``compose: {stages: [...]}``, where the stages are what reach the builder
    and the composing mapping never does.
    """
    model = layer.get("model")
    if isinstance(model, Mapping) and _B2_NODE in model:
        for path, spec in _t4_entries(_B2_NODE, model[_B2_NODE], many=False):
            yield f"model.{path}", spec
    inference = layer.get("inference")
    twin = inference.get("twin") if isinstance(inference, Mapping) else None
    replace = twin.get("replace") if isinstance(twin, Mapping) else None
    if isinstance(replace, Mapping) and _B2_NODE in replace:
        for path, spec in _t4_entries(_B2_NODE, replace[_B2_NODE], many=False):
            yield f"inference.twin.replace.{path}", spec


def _b2_from_projector(spec: Any) -> str | None:
    """The projector entry a ``beam_spill`` node reads f_sky off, if it does.

    ``python:`` is tested first because ``build_node_operator`` tests it
    first: ``{python: ..., from: projector}`` takes the hatch and never
    reaches ``_from_route``, so a check that read ``from:`` alone would speak
    about a route the document does not take.
    """
    if not isinstance(spec, Mapping) or "python" in spec:
        return None
    if spec.get("from") != "projector":
        return None
    return _b2_entry(spec.get("projector"), _B2_PROJECTORS)


def _b2_spill_in(layer: Mapping[str, Any]) -> Iterable[Finding]:
    """A50's two legs over one layer."""
    specs = resolved_specs(layer.get("resources"))
    for path, spec in _b2_spill_entries(layer):
        projector = _b2_from_projector(spec)
        if projector is None:
            continue
        entry = specs.get(projector)
        # A missing entry is a stand-down (`resolved_specs` DROPS what it
        # cannot resolve and `build_resources` stays the backstop), and so is
        # every engine but driftscan: `from_projector` refuses those by class
        # name, which is a better sentence than this one.
        if not isinstance(entry, Mapping) or entry.get("engine") != "driftscan":
            continue
        where = _task3_where(path)
        if _B2_CACHED in list(entry.get("optimizations") or []):
            # Leg B pre-empts leg A rather than joining it: on a projector
            # that is BOTH cached and built on a truncated beam, the call
            # raises before any fraction is computed, so the vanishing ground
            # term is not what the reader would meet. Two sentences about one
            # line is what S4 forbids.
            yield refuse("A50", where, _b2_leg_b(path, projector))
            continue
        beam = _b2_entry(entry.get("beam"), _B2_BEAMS)
        beam_entry = specs.get(beam) if beam is not None else None
        horizon = (beam_entry.get("horizon")
                   if isinstance(beam_entry, Mapping) else None)
        if isinstance(horizon, Mapping) and horizon.get("mode") == "truncate_map":
            yield refuse("A50", where, _b2_leg_a(path, projector, beam))


def _b2_leg_a(path: str, projector: str, beam: str) -> str:
    """Leg A's whole sentence: the truncated beam split a second time."""
    return (
        f"{path}: from: projector takes f_sky off {projector}, whose beam "
        f"{beam} is already cut at the horizon by horizon.mode: truncate_map. "
        "Cutting the beam and splitting it are two spellings of one physical "
        "statement, and writing both does not double the ground term -- it "
        "deletes it: horizon_fraction() over a truncated beam returns "
        "approximately 1.0, so (1 - f_sky) * t_ground goes to zero with "
        "nothing raised anywhere and every shape, unit and dtype still right. "
        "Measured on an nside-8 Gaussian horn at lmax 16: f_sky is 0.9510 "
        "untruncated and 0.99924 truncated, which turns a 14.7 K ground "
        f"contribution into 0.23 K. Either set {beam}'s horizon.mode to "
        "projector_mask and keep this node as it is, or keep truncate_map and "
        f"write sky_fraction: {{ref: {beam}.sky_fraction}} here instead of "
        "from: projector, which is the fraction the truncation already "
        "returned (check A50)."
    )


def _b2_leg_b(path: str, projector: str) -> str:
    """Leg B's whole sentence: f_sky asked of a reference-frame projector."""
    return (
        f"{path}: from: projector takes f_sky off {projector}, which declares "
        "optimizations: [cache_beam_rotation]. to_reference_frame() folds the "
        "horizon mask into that projector's cached alms, so the unmasked "
        "denominator horizon_fraction() divides by is gone and the call "
        "raises a StateValidationError -- from inside build_model, after the "
        "beam has been read, analysed and rotated, and in a class ConfigError "
        f"does not catch. Drop cache_beam_rotation from {projector}, or, if "
        "this run needs the cached rotation, declare a second driftscan "
        "projector over the same beam without it and point this node at that "
        "one (check A50)."
    )


@register("A50")
def _truncated_beam_spill(document: Mapping[str, Any]) -> Iterable[Finding]:
    """Check A50: ``beam_spill: {from: projector}`` the projector cannot answer.

    Leg A refuses a projector whose beam was cut with ``horizon.mode:
    truncate_map`` -- the ground term vanishes in silence.  Leg B refuses one
    built with ``optimizations: [cache_beam_rotation]`` -- the fraction cannot
    be read off it at all.  Both are ``engine: driftscan`` only, both walk
    ``model.beam_spill`` AND ``inference.twin.replace.beam_spill``, and both
    are walked once per layer so a ``variants:`` copy earns the same sentence
    with the variant that introduced it named in front.

    At most one finding per node: leg B pre-empts leg A, because on a
    projector that is both cached and truncated the call raises before any
    fraction is computed.
    """
    return _task3_over_layers(document, _b2_spill_in)
