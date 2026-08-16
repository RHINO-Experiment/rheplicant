"""Pre-flight: the two words ``inference.noise`` used to say three phases late.

``build_noise`` runs inside ``build_inference``, which is the LAST builder
``load_document`` calls -- after ``build_resources``, so both of these refusals
lost to a beam that does not exist.  Measured on the fixture document carrying
``preflight_helpers.UNREADABLE_BEAM``: a ``kind: radiometer`` with no
``include_logdet``, and a 1-D sigma with no ``axis:``, both came back as
*"No file at 'no_such_beam.npy'."*  Neither has anything to do with a file.

**Both rows are HOISTS, so the words do not move** (plan §2.3).  The sentences
live in ``sections/noise.py`` and this module imports the functions that
return them:

* **A26** -- :func:`~rheplicant.config.sections.noise._a26_sigma_axis_problem`,
  asked with the RANK the text declares rather than with a resolved shape.
* **A49**, both directions -- the required one is
  :func:`~rheplicant.config.sections.noise._a49_logdet_problem`; the refused
  one is ``check_unknown_keys`` carrying ``sections/noise.py``'s own
  ``_A49_HINTS``, caught and re-emitted the way
  ``preflight/document.py::_variant_text`` catches ``apply_variant``'s.
  Re-implementing that sweep here is what §2.5 forbids by name, and importing
  ``_KIND_KEYS`` is the same cross-module private read
  ``sections/noise.py`` already makes of ``observation._dimensioned``.

**What stays in ``build_noise``, deliberately.**  ``axis:`` on a sigma that is
not 1-D interpolates ``tuple(sigma.shape)`` -- the resolved extents, which are
the AXES slot's inputs and not a text pass's -- so the mirror leg is not
verbatim-hoistable and is not hoisted.  The 1-D expansion needs the array
itself.  Plan §0.2 C-4 rules on the third piece: ``check_noise_std_axis``
(``inference/noise.py:264``) is never called on this route at all -- measured,
``grep -rn "check_noise_std_axis(" src/rheplicant/config/`` has no hits -- so
what is hoisted is the refusal ``build_noise`` writes, not the package guard
§6 cites.

**Standing down is the whole of A26's correctness** (§3.2 (c)).  A sigma
written ``{ref: resources.arrays.s}`` or ``{file: {...}}`` carries no shape in
the text; this pass answers "I cannot tell" and says nothing, and
``build_noise`` still refuses it at P2 with the same sentence.  Refusing on "I
could not tell" refuses documents that build -- and that is not hypothetical:
the first version of this module read a draw's ``shape:`` as its rank and
refused ``{normal: {shape: [n_freq], loc: {ones: [n_time, n_freq]}}}``, which
LOADS, because ``draws.py`` broadcasts the operand against the draw.
:func:`_a26_operands_are_scalar` is that lesson in code.  Every false negative
is recorded in §7 by name.

**One precedence this hoist SWAPS, recorded rather than gated.**  At P2
``build_noise`` runs ``check_unknown_keys`` BEFORE its kind dispatch, so a
block that was wrong in A26's or A49's way *and* carried a typo'd key was told
about the typo first.  Now A26/A49 speak first and the key sweep arrives on a
second round trip.  It is a swap and not a loss -- the reader still meets both
-- and gating these two checks on the key sweep would give one typo two
voices, which is exactly what §0.3 E.4 ruling 6 avoids.

**``inference.twin.replace`` is NOT walked, and that is not a false
negative** (plan §0.3 E.10).  That ruling is about text checks that walk
``model:``: ``inference.twin.replace.<node>`` reaches the same
``build_node_operator`` path and is outside ``preflight/model.py::_nodes``.
Neither check here reads ``model:``.  Both read ``inference.noise``, which is
the likelihood's noise and not the graph node ``model.noise`` -- two different
things share the word, and ``sections/noise.py``'s own docstring opens by
saying so.  ``inference.twin.replace.noise`` replaces the graph NODE and can
neither carry ``include_logdet:`` nor set ``inference.noise.axis``;
``test_the_twin_replace_route_carries_no_inference_noise`` is that assertion
rather than this paragraph.

**Both checks walk ``variants:``**, through
``preflight/document.py::_task3_over_layers`` -- the same walker 3A's
``A1.horizon`` uses over ``resources.beams``.  A variant that patches
``inference.noise`` is a different document (§2.1) and the refusal it earns is
one nobody sees until that variant is selected.  :func:`_b6_is_layered`
skips the walk when no declared patch can reach ``inference:`` at all, which
is a COST gate with its own equivalence test and not a second layering rule;
see there for the measurement that made it worth writing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import Finding, refuse
from rheplicant.config.preflight import register
from rheplicant.config.preflight.document import _task3_over_layers
from rheplicant.config.resources import check_unknown_keys
from rheplicant.config.sections.noise import (
    _A49_HINTS,
    _KIND_KEYS,
    _a26_sigma_axis_problem,
    _a49_logdet_problem,
)
from rheplicant.config.values import VALUE_FORMS

#: Forms whose result is unconditionally 1-D, whatever their arguments say.
#: ``linspace``/``arange``/``modulo`` produce ``num`` samples along one axis
#: and ``num`` may be a shape SYMBOL -- which is exactly why A26 is answered
#: on rank and not on shape: no extent has to be resolved to know the answer.
#: ``from_grid`` hands back ``context.freq`` or ``context.time``, and an
#: observation grid is 1-D by construction (``observation.py``'s two grids).
_A26_ONE_D_FORMS = frozenset({"linspace", "arange", "modulo", "from_grid"})

#: Forms whose shape is the form key's own value: ``{zeros: [n_time, n_freq]}``.
_A26_SHAPE_FORMS = frozenset({"zeros", "ones"})

#: Forms whose shape sits under a nested ``shape:`` key.
_A26_NESTED_SHAPE_FORMS = frozenset({"full", "normal", "uniform"})

#: form -> the keys whose value is COMBINED with the drawn/filled array rather
#: than describing it.  ``draws.py`` builds a normal as
#: ``loc + scale * jax.random.normal(key, shape)`` -- plain arithmetic, which
#: BROADCASTS -- and ``_resolve_operand`` takes a whole value node for either
#: operand, so ``shape:`` is not the last word on the rank.  ``full``'s
#: ``value:`` is handed straight to ``jnp.full``.
_A26_OPERANDS = {"normal": ("loc", "scale"),
                 "uniform": ("low", "high"),
                 "full": ("value",)}


def _a26_literal_rank(literal: Any) -> int | None:
    """The rank ``jnp.asarray`` would give this written-out literal.

    ``{value: [0.1, 0.2]}`` is a 1-D sigma and A26's subject -- measured, it
    reaches ``build_noise`` with ``ndim == 1`` exactly as ``{list: [...]}``
    does.  The plan names neither; both are here because the naive reading
    ("only the array forms carry a shape") misses them and the document that
    exposes it is one line long.

    An EMPTY list answers None rather than a rank: ``{list: []}`` is the value
    grammar's own refusal (``refs.py`` calls a stack "a container, not a
    computation"), and A26 speaking first would pre-empt it.
    """
    rank = 0
    while isinstance(literal, (list, tuple)):
        if not literal:
            return None
        rank += 1
        literal = literal[0]
    if isinstance(literal, (bool, int, float, complex)):
        return rank
    return None


def _a26_shape_rank(spec: Any) -> int | None:
    """``len(shape)``, when the text wrote a shape at all.

    The shape POSITIONS are never resolved -- ``["n_time", "n_freq"]`` is rank
    2 whatever the grids turn out to be -- which is what keeps this inside
    §2.4's three sources.
    """
    return len(spec) if isinstance(spec, (list, tuple)) else None


def _a26_operands_are_scalar(form: str, spec: Mapping[str, Any]) -> bool:
    """Does every operand this form broadcasts against declare rank 0?

    **A ``shape:`` is not the last word on a draw's rank**, and reading it as
    one was a live REGRESSION: ``draws.py`` builds a normal as
    ``loc + scale * jax.random.normal(key, shape)`` -- plain arithmetic, which
    broadcasts -- and ``_resolve_operand`` (``draws.py:124-131``) accepts a
    whole value node for either operand.  Measured at ``ea4839b``,

        sigma: {normal: {shape: [n_freq], seed: {from: runtime.seeds.a},
                         loc: {ones: [n_time, n_freq]}}}

    LOADS, with a resolved sigma of shape ``(16, 8)``.  Reading ``shape:``
    alone gives text rank 1, so A26 refused a document the layer accepted --
    and applying A26's own remedy earned the mirror leg
    (*"says how to read a 1-D sigma; this one has shape (16, 8)"*), which is a
    fifth advice loop on top of the false positive.  §3.2 (c) is exactly about
    this: the text does not decide the rank, so the check stands down and
    ``build_noise`` keeps the last word.

    The operand is read with :func:`_a26_rank` itself rather than with an
    ``isinstance`` test, because ``_resolve_operand`` takes the whole value
    grammar: ``loc: 1.0``, ``loc: "1.0 K"`` and ``loc: {value: 1.0}`` are all
    rank 0 and leave ``shape:`` in charge, while ``loc: {ref: ...}`` answers
    None and is a stand-down for the same reason every other ``{ref:}`` is.

    ``{uniform: {..., low: <2-D>}}`` and ``{full: {shape: [...], value:
    [[...]]}}`` are treated the same way, and the reason is weaker but still
    good: measured, neither LOADS at ``ea4839b`` -- both die on a bare
    ``TypeError`` from jax, which is nobody's refusal -- so refusing them is
    not a regression, but A26's sentence would still be the wrong sentence and
    its remedy would still not work.
    """
    for key in _A26_OPERANDS[form]:
        if key in spec and _a26_rank(spec[key]) != 0:
            return False
    return True


def _a26_rank(node: Any) -> int | None:
    """The rank this value node's TEXT declares, or None for "cannot tell".

    None is a stand-down and never a refusal (§3.2 (c)).  It is the answer for
    ``{ref:}``, ``{file:}``, ``{from:}``, ``{python:}``, ``{basis_fit:}`` and
    ``{from_switch_order:}``, none of which say a shape out loud, and for any
    node the value grammar itself refuses -- zero form keys, several form
    keys, a bare list -- because each of those has a better sentence waiting.

    ``column:`` is the one modifier that changes rank (``modifiers.py``
    applies it last and it forces ``(n,)`` to ``(n, 1)``), so a node carrying
    a truthy one stands down: the value is 2-D, A26 is not its subject, and
    the mirror leg is ``build_noise``'s.  ``axis:`` INSIDE the node is not
    read here at all -- it is a modifier the grammar records and never
    applies, and reading it is the advice loop §7 records.

    **A ``shape:`` is not the last word**: a draw whose ``loc:``/``scale:``
    (or ``low:``/``high:``, or a ``full``'s ``value:``) is itself an array
    broadcasts up, so :func:`_a26_operands_are_scalar` gates that branch.
    Reading ``shape:`` alone refused a document that LOADS -- see there.
    """
    if isinstance(node, (bool, int, float, complex)):
        return 0
    if isinstance(node, str):
        return 0
    if not isinstance(node, Mapping):
        return None
    forms = [key for key in node if key in VALUE_FORMS]
    if len(forms) != 1 or node.get("column"):
        return None
    form = forms[0]
    if form in ("value", "list"):
        return _a26_literal_rank(node[form])
    if form in _A26_ONE_D_FORMS:
        return 1
    if form in _A26_SHAPE_FORMS:
        return _a26_shape_rank(node[form])
    if form in _A26_NESTED_SHAPE_FORMS:
        spec = node[form]
        if not isinstance(spec, Mapping):
            return None
        if not _a26_operands_are_scalar(form, spec):
            return None
        return _a26_shape_rank(spec.get("shape"))
    if form == "stack":
        entries = node[form]
        if not isinstance(entries, (list, tuple)) or not entries:
            return None
        inner = _a26_rank(entries[0])
        return None if inner is None else inner + 1
    return None


def _b6_is_layered(document: Mapping[str, Any]) -> bool:
    """Can any variant this document declares change ``inference.noise``?

    ``apply_variant`` is ``recursive_update(document, patch)`` and layering is
    ONE level deep by design (``layering.py``'s own docstring), so a patch
    whose top level names neither ``inference`` nor ``~inference`` leaves that
    section exactly as the base wrote it.  Both spellings, because ``~key`` is
    the delete form and a variant that DELETES ``inference:`` changes what
    these two checks read as surely as one that rewrites it.

    **This is a cost gate and nothing else**, and it is measured rather than
    assumed: ``test_the_layer_gate_changes_no_finding`` asserts that the
    findings are identical to the ungated walk's over a battery that includes
    both answers, and ``test_the_layer_gate_actually_skips_the_deepcopy``
    is its anti-vacuity partner.

    Why it exists: ``_task3_over_layers`` deep-copies the whole document once
    per variant per check that layers, and ``test_config_preflight.py``'s cold
    budget is the tightest guard in this suite.  Measured on its own 40-run,
    20-variant document -- whose variants patch ``model:`` alone -- these two
    checks ungated cost **31 ms against a 50 ms budget** where the same
    document without them costs 23 ms; gated they cost what the base walk
    costs and the twin is still guarded.
    """
    variants = document.get("variants")
    if not isinstance(variants, Mapping):
        return False
    return any(isinstance(patch, Mapping)
               and ("inference" in patch or "~inference" in patch)
               for patch in variants.values())


def _b6_over_layers(document, per_layer) -> Iterable[Finding]:
    """:func:`_task3_over_layers`, skipped when no variant can move the answer.

    3A's walker is CALLED and not re-implemented: when the gate says a variant
    could change ``inference.noise``, this is exactly ``A1.horizon``'s walk
    over exactly the same layers, prefixes and de-duplication.
    """
    if not _b6_is_layered(document):
        return per_layer(document)
    return _task3_over_layers(document, per_layer)


def _b6_noise_block(layer: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """``inference.noise``, when the layer holds one that is a mapping.

    ``_structural`` guarantees a top-level SECTION is present, never that it
    is a mapping, and ``inference.noise`` is two levels past where that
    guarantee reaches -- so both levels are tested.  A malformed one is
    ``build_noise``'s own refusal ("is a mapping with kind:") to say.
    """
    section = layer.get("inference")
    noise = section.get("noise") if isinstance(section, Mapping) else None
    return noise if isinstance(noise, Mapping) else None


def _a26_in(layer: Mapping[str, Any]) -> Iterable[Finding]:
    """A26 on one layer: a 1-D sigma with no ``axis:`` beside it."""
    noise = _b6_noise_block(layer)
    if noise is None or noise.get("kind") != "homoscedastic":
        return
    if "sigma" not in noise:
        # `build_noise` says "kind: homoscedastic requires sigma: -- a value
        # node", which names the fix.  A26 has no sigma to have a rank.
        return
    message = _a26_sigma_axis_problem(rank=_a26_rank(noise["sigma"]),
                                      axis=noise.get("axis", "none"))
    if message is not None:
        yield refuse("A26", "inference.noise.sigma", message)


@register("A26")
def _sigma_axis(document) -> Iterable[Finding]:
    """A26: a 1-D sigma with no ``axis:``, over every layer.

    ``where`` is ``inference.noise.sigma`` and not ``.axis``: the sigma is the
    key the user wrote, ``axis:`` is the key they did not, and a ``where``
    pointing at an absent key sends a reader looking for a line that is not
    there.  The message names ``axis:`` twice, so the fix is not lost.
    """
    return _b6_over_layers(document, _a26_in)


def _a49_in(layer: Mapping[str, Any]) -> Iterable[Finding]:
    """A49 on one layer, in both directions.

    The refused direction is ``check_unknown_keys``' sweep, run early and
    GATED on ``include_logdet`` being one of the unknown keys.  Without the
    gate this check would claim every stray key under ``inference.noise``
    under A49's id, which is a different check's subject wearing this one's
    name; with it, a document whose only fault is a typo'd ``flors:`` still
    hears ``build_noise`` say so at P2, in the same words.
    """
    noise = _b6_noise_block(layer)
    if noise is None:
        return
    kind = noise.get("kind")
    # `isinstance` FIRST, and that is not defensive typing -- it is §2.3's
    # TRAP measured on this very check.  `inference.noise.kind: [radiometer]`
    # is a document a user can write, and `kind not in _KIND_KEYS` on a list
    # raises `TypeError: unhashable type`, which the pass turns into "check
    # 'A49' RAISED" and which DISCARDS every other finding in the report.
    # `preflight/fitting.py::_decided` guards the same key the same way, and
    # `test_preflight_fitting.py`'s hostile battery is what found this one.
    if not isinstance(kind, str) or kind not in _KIND_KEYS:
        # `build_noise` (noise.py) names the four kinds.  A vocabulary
        # refusal from here would be a second voice for one typo.
        return
    if "include_logdet" in set(noise) - _KIND_KEYS[kind]:
        try:
            check_unknown_keys("inference.noise", dict(noise),
                               _KIND_KEYS[kind], label=f"kind: {kind}",
                               hints=_A49_HINTS)
        except ConfigError as exc:
            yield refuse("A49", "inference.noise.include_logdet", str(exc))
        return
    message = _a49_logdet_problem(kind, noise)
    if message is not None:
        yield refuse("A49", "inference.noise.include_logdet", message)


@register("A49")
def _logdet(document) -> Iterable[Finding]:
    """A49: ``include_logdet`` present exactly when the sigma needs it.

    ``where`` is ``inference.noise.include_logdet`` in BOTH directions -- the
    key is present in one and required-but-absent in the other, and it is the
    key at fault either way.  The refused direction's message opens with
    ``inference.noise:`` because it is ``check_unknown_keys``' sentence and
    that helper speaks for the whole entry; naming the section in the ``where``
    as well would point a reader at a block rather than at a line.
    """
    return _b6_over_layers(document, _a49_in)
