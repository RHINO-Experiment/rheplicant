"""``model:``, checked from the document's text (schema §6, A2-A8, A14, A30-A33).

Every check here is a second CALL SITE, never a second implementation: the
graph-shaped rules live in ``sections/compose.py`` and ``sections/model.py``
as module-level pure functions, this module hands them data read off the raw
document, and the build hands them the same data one phase later.  What moves
is the PHASE.  ``document.py`` builds the resources -- where a CST directory
is read and the spherical harmonic transform runs -- before it builds the
model, so every one of these refusals used to arrive after the beam was
analysed.

Three sources and no fourth (§2.4): the document mapping, ``RADIO_GRAPH``, and
operator classes resolved BY NAME off ``rheplicant.radio``.  No value node is
resolved, no operator is constructed, no file is read.  Re-measured with Task
11's two checks added (median of 2000 calls, in two separate processes so the
first-call figure is not warmed by the loop): the **eight** checks here answer
a **six**-node document -- ``preflight_helpers``' base model, which has four,
plus ``foregrounds:`` and ``bandpass:`` -- in **3.3e-05 s**; the first call is
**2.3e-04 s**.  §0.1's budget for the whole pass is 0.05 s, against
``load_document``'s 1.536 s on a toy beam, and the whole pass on a Task 11
document measures **4.2e-04 s**.  (Re-measured after A30 and A33 each grew a
:func:`_lit` call at the review: **3.1e-05 -> 3.3e-05 s** here and 3.5e-04 ->
4.2e-04 s for the pass.)

**That first call is not a deferred import**, which is what this file said
until Task 5 checked: pre-importing ``sections/switching``, ``core/fold`` and
``rheplicant.radio`` leaves it at 2.3e-04 s unchanged.  It is
``sections/model.operator_table()``, measured at 1.7e-04 s the first time and
2.3e-05 s every time after -- it is rebuilt per call rather than cached, which
is also most of the steady-state figure above.  Recorded rather than fixed:
the table is ``sections/model``'s and no task in this plan owns it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from _rheplicant_bootstrap.path_syntax import longest_legal_prefix
from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import Finding, refuse
from rheplicant.config.paths import parse_path
from rheplicant.config.preflight import register
from rheplicant.config.preflight.fitting import _latents

# Reached through this module by name elsewhere, and no longer used by this
# file's own code after the split, so `ruff --fix` removes a plain import.
from rheplicant.config.sections.model import (  # noqa: F401
    ambiguous_class_problem as ambiguous_class_problem,
)
from rheplicant.config.sections.model import (
    operator_table as operator_table,
)

from .model_radio import _a5_remedy as _a5_remedy
from .model_radio import _a14_cal_load_keys as _a14_cal_load_keys
from .model_radio import (
    _lit,
)
from .model_radio import _t4_switch_order as _t4_switch_order
from .model_radio import _t5_claims as _t5_claims
from .model_radio import _t5_downstream as _t5_downstream
from .model_radio import _t5_placement as _t5_placement
from .model_radio import _t5_radio_class as _t5_radio_class
from .model_radio import _two_at_one_node as _two_at_one_node
from .model_sources import _data_with_sources as _data_with_sources
from .model_sources import _no_source_and_no_data as _no_source_and_no_data
from .model_sources import _tone_placement as _tone_placement

# Re-exported so this module's importers keep working. `X as X` is the
# explicit re-export form: a plain import of a name this file does not
# itself use is F401, and `ruff --fix` deletes it whatever the comment
# on the line says.
from .model_stochastic import _A30_NOT_FITTING as _A30_NOT_FITTING
from .model_stochastic import (
    _A33_CONVENTION,
    _a33_convention,
)
from .model_stochastic import _a30_exits as _a30_exits
from .model_stochastic import _a30_placements as _a30_placements
from .model_stochastic import _a30_stochastic as _a30_stochastic
from .model_stochastic import _stochastic_in_fit_twin as _stochastic_in_fit_twin
from .model_stochastic import stochastic_nodes as stochastic_nodes
from .model_walk import _capability_level as _capability_level
from .model_walk import _double_count as _double_count
from .model_walk import _graph_shape as _graph_shape
from .model_walk import _t4_at_nodes as _t4_at_nodes
from .model_walk import _t4_entries as _t4_entries
from .model_walk import _t4_graph as _t4_graph

# ---------------------------------------------------------------------------
# Task 11 -- A30 (a stochastic stage the fit twin keeps) and A33 (a bandpass
# left free beside a gain).
# ---------------------------------------------------------------------------


def _t11_bindings(
    document: Mapping[str, Any],
) -> tuple[tuple[str, frozenset[str], tuple[str, ...], Any], ...]:
    """``(document path, latent NAMES, into-path HEADS, transform)`` per binding.

    BOTH spellings, because ``build_space`` walks two loops over one meaning:
    ``inference.parameters.<n>.into`` (``transforms.py::parse_transform``) and
    ``inference.bindings[i].into`` (``transforms.py``).  Both carry ``transform:``.
    A check that read one is 2C's shape 4 -- a hole closed on one route and
    left open on its twin -- in the one place this layer has an actual twin.

    The latent NAMES travel with the binding because A33's question is about
    two DIFFERENT parameters.  One latent written into both leaves (``into:
    [bandpass.bandpass, gain.gain]``, or two ``bindings`` entries naming the
    same latent) is one degree of freedom and no null direction at all, and a
    check that compared only path heads refuses it.

    **A binding whose latent names cannot be read is DROPPED**, not carried
    with a stand-in, and a name ``inference.parameters`` does not DECLARE is
    dropped by the same rule.  Its own refusal is more specific and arrives
    with the value the user wrote -- *"inference.bindings[0]: latents: is a
    non-empty list of latent names; got 7."* (``transforms.py::_merged_fan``), and
    *"inference.bindings[0]: 'ghost' is not a declared latent;
    inference.parameters declares ['g']."* -- and A33 answering first hands
    that reader a degeneracy lecture instead.  The membership half was
    measured LIVE: ``bindings: [{latents: ['ghost'], into:
    'bandpass.bandpass'}]`` earned A33 and was told to declare ``transform:
    unit_mean_bandpass``, which cannot help a latent that does not exist.
    This is ``_a23_prior_free``'s rule (``preflight/fitting_engines.py::_t7_engines``: *"``names``
    must already be names the document DECLARES WELL"*) applied on the side
    A33 left open -- A33's own docstring gates both path HEADS against
    ``_lit`` for exactly this shape and left the NAME ungated, which is 2C's
    "closed on one route, open on its twin" inside one function.
    ``inference.parameters.<n>`` entries need no filter: their name IS the
    declaration.  An
    earlier draft substituted the binding's ``where`` for its latent set, on
    the reasoning that no latent name can equal it; measured at ``36b7e54``
    that is true and beside the point, because it made A33 decide *"these are
    two different parameters"* from a value it had just failed to read, and
    it did so ASYMMETRICALLY -- the substitute counted as a difference on the
    gain side and as no difference on the bandpass side.

    ``inference.parameters.<n>`` entries are unaffected: their name is the
    mapping key, so it is always readable.

    ``parse_path`` RAISES on a malformed path (measured: ``''``, ``'a..b'``,
    ``None`` and ``['a']`` all give ``ConfigError``), and a check that raises
    aborts the pass and hides every later finding (§2.3's TRAP).  So an
    unparseable ``into:`` is skipped here and left to
    ``_selectors``/``parse_path`` at build time, which already names it.
    """
    section = document.get("inference")
    section = section if isinstance(section, Mapping) else {}
    declared = _latents(document)
    written: list[tuple[str, tuple[str, ...], Any, Any]] = [
        (f"inference.parameters.{name}", (name,), spec.get("into"), spec.get("transform"))
        for name, spec in declared.items()
        if spec.get("into") is not None
    ]
    bindings = section.get("bindings")
    if isinstance(bindings, (list, tuple)):
        for index, entry in enumerate(bindings):
            if not isinstance(entry, Mapping) or entry.get("into") is None:
                continue
            latents = entry.get("latents")
            latents = (latents,) if isinstance(latents, str) else latents
            names = (
                tuple(one for one in latents if isinstance(one, str) and one in declared)
                if isinstance(latents, (list, tuple))
                else ()
            )
            if not names:
                continue
            written.append(
                (f"inference.bindings[{index}]", names, entry.get("into"), entry.get("transform"))
            )

    out: list[tuple[str, frozenset[str], tuple[str, ...], Any]] = []
    for where, names, into, transform in written:
        paths = [into] if isinstance(into, str) else into
        if not isinstance(paths, (list, tuple)):
            continue
        heads: list[str] = []
        for path in paths:
            if not isinstance(path, str):
                continue
            try:
                head = parse_path(path)[0]
            except ConfigError:
                continue
            if isinstance(head, str):
                heads.append(head)
        out.append((where, frozenset(names), tuple(heads), transform))
    return tuple(out)


@register("A33")
def _bandpass_and_gain(document: Mapping[str, Any]) -> Iterable[Finding]:
    """Check A33: a latent free into ``bandpass`` beside a DIFFERENT one free
    into ``gain``, with no identifiability convention on the bandpass.

    Every declared latent is free -- ``_LATENT_KEYS`` (``parameters.py:
    27-29``) has no freeze -- so "both free" is "both bound".

    Pure text: the head of each ``into:`` path.  ``parse_path(path)[0]`` gives
    it (``config/paths.py::parse_path``) -- measured, ``parse_path('bandpass.taps[0]')``
    is ``('bandpass', 'taps', 0)``, so a deeper path still counts by its node.

    The SHAPE half of A33 -- that ``unit_mean_bandpass`` maps ``(n,)`` to
    ``(n-1,)``, so a latent bound through it is one channel shorter than the
    node it writes -- is check C17, shipped in place by Plan 3B inside
    ``config/sections/`` (``sections/transforms.py``, ``sections/observed.py``)
    rather than left for a later plan to build; this half is two path heads,
    two latent names and a transform.

    **Measured absent today** at ``0263e0f``: a document lighting ``bandpass``
    and ``gain`` with a free latent into each and no transform builds with no
    refusal, and no document in ``tests/config/`` or ``docs/`` binds a latent
    into a ``bandpass.*`` path at all -- so this check refuses nothing that
    exists and its own tests are the only coverage it will have.
    ``exit_helpers.py::GLS_PAIR`` calls its ``gain`` + ``global_signal.depth``
    pair *"the schema's A33 shape"*; that is an analogy about degeneracy and
    not an A33 document, because neither latent goes into ``bandpass``.

    **The MODEL is consulted as well as the bindings**, and that is a live
    correction rather than belt and braces.  An ``into:`` head is a node the
    user TYPED, not a node the document lights: measured at ``36b7e54``, a
    document whose model has no ``bandpass`` node at all earned A33 for
    ``into: bandpass.bandpass``, while the package's own sentence is *"Path
    'bandpass.bandpass' could not be walked against this twin: No node named
    'bandpass' in this assembly"*.  A typo'd head was answered with a
    degeneracy lecture and told to declare ``transform: unit_mean_bandpass``,
    which cannot help -- the path still resolves to nothing.  Both heads are
    gated, because a free ``gain`` the model does not light is the same
    mistake on the other side.
    """
    lit = _lit(document)
    if "bandpass" not in lit or "gain" not in lit:
        return ()
    bindings = _t11_bindings(document)
    on_bandpass = [one for one in bindings if "bandpass" in one[2]]
    if not on_bandpass:
        return ()
    on_bandpass_latents = frozenset().union(*(names for _, names, _, _ in on_bandpass))
    on_gain_latents = frozenset().union(
        *(names for _, names, heads, _ in bindings if "gain" in heads)
    )
    # The DIFFERENCE, not the presence: a latent written into both leaves is
    # one degree of freedom and the product IS constrained, so there is
    # nothing to trade and nothing to refuse.
    if not on_gain_latents - on_bandpass_latents:
        return ()
    verdicts = [_a33_convention(transform) for _, _, _, transform in on_bandpass]
    if any(verdict is not False for verdict in verdicts):
        return ()
    where = on_bandpass[0][0]
    return (
        refuse(
            "A33",
            longest_legal_prefix(f"{where}.transform"),
            (
                f"{where} is free into bandpass and this document also frees a "
                "latent into gain. The receiver's bandpass and the gain multiply the "
                "same prediction, so only their PRODUCT is constrained: the fit has "
                "one exactly null direction and returns a finite, correctly-shaped "
                "answer in which the two have traded an arbitrary constant. Declare "
                f"transform: {_A33_CONVENTION} on the bandpass binding -- it divides "
                "out the mean, which is the convention that makes the pair "
                "identifiable (check A33)."
            ),
        ),
    )
