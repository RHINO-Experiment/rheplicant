"""C15 -- the noise-wave per-channel rank, reported before the beam is read.

``radio/instrument/noise_wave.py``'s own docstring states the rule and the
measurement behind it: each switch position contributes exactly one equation
per frequency channel, so *while every temperature is free per channel* the
design matrix has rank ``min(n_src, k) * n_freq``, where ``k`` is the number
of free temperature families among ``t_unc, t_cos, t_sin, t_rx``.  A document
that frees them is a document whose switching cadence is already decided; this
check says what that cadence bought, in the layer's own voice, at the moment
the document is read rather than after a fit has returned a finite,
correctly-shaped, wholly prior-driven answer.

**Reported, never refused.**  Schema §6's C15 row says so and the package's
docstring says why: the rule is a design aid a user reads off, not a fault.  A
three-load four-family fit is *deficient by exactly ``n_freq``* and is still a
document somebody may mean to run -- with a prior carrying the difference.

**Why the AXES pass and not the text pass** (D-9).  ``n_freq`` is not
decidable from text.  The only text reader for it,
``preflight/values.py::_a41_scope``, answers ``None`` -- *"the document does
not say"* -- for a grid that is not a ``linspace``/``arange``/``modulo``/
``list``, for a symbolic ``num:``, and for **every ingested run**
(``observation.from_file`` and ``observation.freq`` are mutually refused).  A
C15 in pre-flight would therefore be silent by construction on a whole class
of documents, which is the false negative this plan exists to remove.
:class:`~rheplicant.config.inflight.Axes` carries ``context.shape_scope``,
which gives ``n_freq`` and ``n_source`` **unconditionally** and for both
routes -- an ingested run's frequency axis comes off the recording
(``sections/observation.py``) -- and the axes pass still runs in front of
``build_resources``, which is 90.9 % of ``load_document``'s wall time.

**``n_source`` is INHERITED, not re-derived.**
``ResolutionContext.shape_scope`` is already
``n_source_override or len(switch_order) or 1`` -- A15's rule with the
override honoured.  Writing ``len(order)`` here would report a rank of zero on
the correct one-load ``pointing.mode: none`` document, and writing the ``or 1``
again would be a second binding of a rule this layer already has one of.

**The measured trap: the basis detector must look at ``t_sys_extra``.**
:class:`~rheplicant.radio.t_sys.BasisTemperatureOperator`'s ``graph_node`` is
``t_sys_extra``, **not** ``noise_wave`` (measured; its fields are ``coeff``,
``time_basis``, ``freq_basis``).  A check that looked for the basis under
``model.noise_wave`` finds nothing and reports a number the package's own
docstring contradicts in *both* directions -- 12 coefficients identified
against a predicted 6, and rank 5 against a bound of 7.

**Two routes into the basis regime, and closing one is 3A's recorded twin
failure** (§2.6 item 10): a lit ``t_sys_extra`` of that type, **or** a latent
reaching a ``noise_wave.t_*`` leaf through a ``transform:`` -- which is itself
a twin, since ``inference.parameters.<n>.transform`` and
``inference.bindings[].transform`` are two spellings of one thing.

**Two routes into the leaf, likewise.**  ``build_space`` walks
``inference.parameters.<n>.into`` and ``inference.bindings[].into`` in two
loops over one meaning.  ``preflight/model.py::_t11_bindings`` closed exactly
this pair for A33 and is the shape copied here -- but it returns the path
**HEAD** and this check needs the **LEAF**, so the walk is reused and its
return is not.

**Nothing here may raise.**  ``passes.sweep`` turns any exception out of a
check into a ``ConfigError`` that aborts the whole pass and hides every
finding after it.  ``into:`` is legally a string OR a list of strings
(``sections/parameters.py::_names``) and ``paths.parse_path`` raises
``ConfigError`` on a non-``str`` and on a malformed path, so every call is
normalised and wrapped.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import Finding, report
from rheplicant.config.inflight import Axes, register_axes
from rheplicant.config.paths import parse_path

#: The four temperature families schema §6's ``k`` counts.  They are
#: ``NoiseWaveOperator``'s own leaves -- measured with ``dataclasses.fields``,
#: alongside ``gamma_src_re``, ``gamma_src_im``, ``gamma_rec_re``,
#: ``gamma_rec_im`` and ``switch_key``, which are couplings and a selector
#: rather than temperatures.  ``t_rx`` IS one of the four: its coupling is 1
#: rather than absent, so ``k`` is four when it is fitted and three when it is
#: taken as known.
_NOISE_WAVE_LEAVES: frozenset[str] = frozenset({"t_unc", "t_cos", "t_sin",
                                                "t_rx"})

#: The graph node ``NoiseWaveOperator`` declares, and therefore the head every
#: ``into:`` path into a temperature carries.
_NOISE_WAVE_NODE: str = "noise_wave"

#: The class whose presence means the temperatures have become coefficients of
#: a frequency basis.  Its ``graph_node`` is :data:`_T2C_BASIS_NODE` -- see the
#: module docstring's trap.
_T2C_BASIS_TYPE: str = "BasisTemperatureOperator"

#: Where :data:`_T2C_BASIS_TYPE` lands.  **Not** ``noise_wave``.
_T2C_BASIS_NODE: str = "t_sys_extra"

#: The OTHER spelling of the same operator (``sections/model.py``'s
#: ``t_sys_extra`` + ``from: basis`` route), which writes no ``type:`` at
#: all -- ``BasisTemperatureOperator.from_basis(basis, coeff)`` is built off
#: ``basis: {ref: resources.bases.<name>}`` and ``coeff:`` alone.  Measured on
#: a ``from: basis`` list-form document: ``declares_basis: False`` under the
#: original detector, and the document LOADS -- a wrong number, not silence.
_T2C_BASIS_ROUTE: str = "basis"

#: The transforms that leave the per-channel counting in force.  ``identity``
#: binds the leaf unchanged, so it ties no channels together; this layer
#: already reads the pair this way (``sections/inference.py::_derive_truth``).
_T2C_TRANSPARENT: frozenset[str] = frozenset({"identity"})


def _t2c_paths(into: Any) -> tuple[str, ...]:
    """The ``into:`` paths one entry declares, as strings.

    ``into:`` is legally a string OR a list of strings, and anything else is
    ``parse_latents``' own refusal at build time with the value the user wrote.
    Answering ``()`` here leaves that refusal to the place that says it best.
    """
    paths = [into] if isinstance(into, str) else into
    if not isinstance(paths, (list, tuple)):
        return ()
    return tuple(one for one in paths if isinstance(one, str))


def _t2c_leaf(path: str) -> str | None:
    """The noise-wave temperature ``path`` names, or ``None``.

    Both halves are gated: the HEAD must be the node
    ``NoiseWaveOperator`` declares, and the leaf must be one of the four.  A
    ``t_unc`` under some other node is not this operator's, and gating the
    leaf alone is the mistake A33's first version made on its own path heads.

    The leaf is the last STRING segment, so ``noise_wave.t_unc[0]`` still
    counts by its field -- ``parse_path`` returns ``('noise_wave', 't_unc',
    0)`` there, and an index is not a leaf name.
    """
    try:
        parsed = parse_path(path)
    except ConfigError:
        # `parse_path` raises on '', 'a..b' and every other malformed path,
        # and a check that raises aborts the pass.  `_selectors` names it at
        # build time with the value the user wrote.
        return None
    if not parsed or parsed[0] != _NOISE_WAVE_NODE:
        return None
    named = [one for one in parsed[1:] if isinstance(one, str)]
    if not named:
        return None
    return named[-1] if named[-1] in _NOISE_WAVE_LEAVES else None


def _t2c_where(where: str, fallback: str) -> str:
    """``where`` if this pass's guard can accept it, else ``fallback``.

    A latent's NAME is user text and reaches this module before
    ``parse_latents`` has looked at it -- the axes pass runs at
    ``document.py``'s hook, and ``build_inference`` is two builders later.
    Measured, ``parse_path('inference.parameters.a b')`` RAISES, and
    ``passes.check_where`` turns that into a ``ConfigError`` that aborts the
    whole axes pass and hides every finding after it.

    So an unusable path falls back to its PARENT, which is always legal and is
    still a line the reader can find.  ``preflight/gated.py::_t2_where`` closes
    the same hole on the ``inference.checks.<name>`` side; the two are separate
    because they are in different passes and neither package imports the
    other.
    """
    try:
        parse_path(where)
    except ConfigError:
        return fallback
    return where


def _t2c_routes(document: Mapping[str, Any]
                ) -> tuple[tuple[str, frozenset[str], Any], ...]:
    """``(document path, the noise-wave leaves it frees, transform)`` per entry.

    BOTH spellings, in document order -- ``inference.parameters.<n>.into``
    first, then ``inference.bindings[].into`` -- because ``build_space`` walks
    two loops over one meaning and a check that read one is 2C's shape 4 in
    the one place this layer has an actual twin.

    This is ``preflight/model.py::_t11_bindings``' walk with its return
    changed: that one answers with the path HEADS, which is A33's question,
    and this one needs the LEAF.

    **MINOR 1 (Plan 3C fix round): a ``bindings[]`` entry's ``latents:`` is
    now filtered exactly as ``_t11_bindings`` filters it.**  Before this fix
    this function counted every ``bindings[]`` entry whose ``into:`` reached a
    noise-wave leaf, whatever its ``latents:`` said -- ``_t11_bindings``
    deliberately DROPS an entry whose ``latents:`` is missing, non-string or
    names nothing ``inference.parameters`` declares, because a more specific
    refusal names that fault at build time and a check that counted it anyway
    reports a rank the document cannot reach.  Measured before this fix, on a
    single ``bindings[{'latents': ['ghost'], 'into': 'noise_wave.t_unc'}]``
    entry with no ``parameters:`` declaring ``ghost``: ``_t11_bindings`` (the
    A33 walk) answers ``[]`` while this function answered
    ``[('inference.bindings[0]', frozenset({'t_unc'}), None)]`` -- C15's ``k``
    counted a temperature no declared latent actually reaches.  Reusing
    ``_t11_bindings`` outright is not available here -- it returns the path
    HEAD (A33's question) and this function needs the LEAF -- so the same
    ``declared`` gate is applied to the same walk instead, at the same point
    :func:`~rheplicant.config.preflight.model._t11_bindings` applies it.
    """
    from rheplicant.config.preflight.fitting import _latents

    section = document.get("inference")
    section = section if isinstance(section, Mapping) else {}
    declared = _latents(document)
    written: list[tuple[str, Any, Any]] = []
    parameters = section.get("parameters")
    if isinstance(parameters, Mapping):
        written.extend(
            (_t2c_where(f"inference.parameters.{name}",
                        "inference.parameters"),
             spec.get("into"), spec.get("transform"))
            for name, spec in parameters.items()
            if isinstance(name, str) and isinstance(spec, Mapping))
    bindings = section.get("bindings")
    if isinstance(bindings, (list, tuple)):
        for index, entry in enumerate(bindings):
            if not isinstance(entry, Mapping) or entry.get("into") is None:
                continue
            latents = entry.get("latents")
            latents = (latents,) if isinstance(latents, str) else latents
            names = tuple(one for one in latents
                          if isinstance(one, str) and one in declared) \
                if isinstance(latents, (list, tuple)) else ()
            if not names:
                continue
            written.append((f"inference.bindings[{index}]",
                            entry.get("into"), entry.get("transform")))

    out: list[tuple[str, frozenset[str], Any]] = []
    for where, into, transform in written:
        leaves = frozenset(
            leaf for leaf in (_t2c_leaf(path) for path in _t2c_paths(into))
            if leaf is not None)
        if leaves:
            out.append((where, leaves, transform))
    return tuple(out)


def _t2c_declares_basis(document: Mapping[str, Any]) -> bool:
    """Does ``model.t_sys_extra`` light a :data:`_T2C_BASIS_TYPE`?

    ``t_sys_extra`` is a ``many`` node, so its spec is a LIST -- unlike
    ``cal_loads``, it is SUM-shaped rather than FAN-shaped, and
    ``many_shape_problem`` (``sections/compose.py``) refuses a mapping there
    outright: *"is a non-empty list (SUM); got dict"*.  **No mapping-shaped
    spelling of this branch is worth reading**, because none of them survive
    to reach a document ``load_document`` accepts -- an earlier version of
    this function read a bare mapping and a FAN mapping anyway, and MAJOR 4
    of the Task 2 fix round measured that the fixture built on the belief was
    itself A6-refused. Three entry shapes are read instead, all of them LIST
    entries: ``type: BasisTemperatureOperator``, the ``from: basis`` route
    (which writes no ``type:`` at all -- ``sections/model.py``'s ``t_sys_
    extra`` + ``from: basis`` builds ``BasisTemperatureOperator.from_basis``
    directly), and a ``python:`` relocation naming the class by its target's
    last segment.

    One level and no deeper, deliberately: a recursive walk over arbitrary
    document values would answer "basis" for a ``coeff:`` whose own mapping
    happened to carry the word, and the cost of a wrong ``True`` here is the
    check going silent about a number it could have given.
    """
    model = document.get("model")
    if not isinstance(model, Mapping):
        return False
    spec = model.get(_T2C_BASIS_NODE)
    if not isinstance(spec, (list, tuple)):
        return False
    for entry in spec:
        if not isinstance(entry, Mapping):
            continue
        if entry.get("type") == _T2C_BASIS_TYPE:
            return True
        if entry.get("from") == _T2C_BASIS_ROUTE:
            return True
        python = entry.get("python")
        if isinstance(python, str) and python.rsplit(
                ":", 1)[-1].rsplit(".", 1)[-1] == _T2C_BASIS_TYPE:
            return True
    return False


@register_axes("C15")
def _noise_wave_rank(facts: Axes) -> Iterable[Finding]:
    """Check C15: what a document's switching cadence identifies, per channel.

    ``facts`` is an :class:`~rheplicant.config.inflight.Axes`, **not** a
    document -- that is D-9's whole reason for this module's existence.

    Yields **at most one REPORT finding, never a refusal**, and nothing at all
    when no noise-wave temperature is free: a document that frees none has no
    cadence question to answer, and a check that reported ``min(n_source, 0) *
    n_freq`` on every document is a check nobody reads.
    """
    routes = _t2c_routes(facts.document)
    if not routes:
        return ()
    freed = frozenset().union(*(leaves for _, leaves, _ in routes))
    where = routes[0][0]
    family = sorted(_NOISE_WAVE_LEAVES)
    opening = (f"{where} frees {sorted(freed)} of the four noise-wave "
               f"temperatures {family}")

    transformed = any(
        transform is not None and transform not in _T2C_TRANSPARENT
        for _, _, transform in routes)
    if _t2c_declares_basis(facts.document) or transformed:
        route = ("a frequency basis" if _t2c_declares_basis(facts.document)
                 else "a transform:")
        return (report("C15", where, (
            f"{opening} through {route}, so the per-channel counting rule "
            "does not apply and no counting rule replaces it. A basis ties "
            "the channels together and the rule fails in BOTH directions: "
            "per-channel counting understates (two loads and a 3-coefficient "
            "basis identify all k * n_basis = 12 coefficients at k = 4, where "
            "min(n_source, k) * n_basis would say 6) and the bound "
            "rank <= min(n_source * n_freq, k * n_basis) overstates (one load "
            "whose Gamma is itself linear in frequency gives rank 5 against a "
            "bound of 7). Measure this parameterization with "
            "rheplicant.inference.identifiability instead (check C15).")),)

    scope = facts.context.shape_scope
    n_source, n_freq, k = scope.n_source, scope.n_freq, len(freed)
    return (report("C15", where, (
        f"{opening}. Each switch position contributes one equation per "
        "frequency channel, so while every temperature is free PER CHANNEL "
        "the design matrix has rank min(n_source, k) * n_freq = "
        f"min({n_source}, {k}) * {n_freq} = {min(n_source, k) * n_freq}. A "
        "four-family per-channel fit needs four distinct loads to be square; "
        "three loads leave it deficient by exactly n_freq, and sharing one "
        "Gamma across the cycle collapses every source onto the same row and "
        "drops the rank to n_freq whatever n_source is. Read a switching "
        "cadence off this number, and measure any other parameterization "
        "with rheplicant.inference.identifiability (check C15).")),)
