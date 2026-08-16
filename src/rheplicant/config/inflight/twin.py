"""B5 and C9 -- two things the assembled twin knows and nobody asks it.

**This slot saves nothing and this module must not pretend otherwise.**  The
built pass runs when ``load_document`` is ready to return, so the beam has
already been read, analysed and truncated by the time either check here
speaks.  Schema §6's preamble -- "all run before any file is read that is not
needed to decide them, and before any beam is analysed" -- is FALSE about
every row registered here, and ``test_inflight_twin.py`` ships the box that
says so: a document carrying a B5 violation AND an unreadable beam is refused
by **the beam**.  What this slot buys is that the checks run at all, in this
layer's voice, against the objects rather than detonating inside a fit.

**Why these two need the assembled twin.**

* **B5** compares the number of switch positions the fold actually produced
  against the number ``observation.switching.order`` declares.  Neither the
  document's text nor the resolved grids hold the first number: it is
  ``len(twin["receiver_input"].names)``, which is decided by how many live
  branches reached the selector, which is decided by which sources the model
  lights.
* **C9** compares a temperature leaf's resolved SHAPE against a square grid.
  ``{ones: [n_freq]}`` and ``{ones: [n_time]}`` are the same text on an
  8x8 run and different intentions; only the built leaf says which arrived.

**Both twins are walked, and that is a correction to the plan.**  §0.3 E.10
assumed ``Built.twin`` already carries ``inference.twin.replace``.  Measured,
it does not: ``Built.twin`` is the RAW twin (``model:`` as declared) and
``inference.fit_twin`` is the one ``replace:`` rebuilt, with
``inference.replaced`` naming which nodes moved.  Both objects are live --
``run_forward`` evaluates the raw twin and the fit evaluates the other -- so
:func:`_twins` walks both and :func:`_where` reports the document key that
built each one.  Operators the fit twin did not rebuild are the SAME objects
as the raw twin's (measured: ``fit["cal_loads_1"] is twin["cal_loads_1"]``),
which is what keeps one leaf from being reported twice.

**Routes.**  ``Built.document`` is the variant-applied mapping, so there is no
``variants:`` twin here.  Everything else each check walks is stated on the
check.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rheplicant.config.findings import Finding, refuse
from rheplicant.config.inflight import Built, register_built

#: The four noise-wave temperatures, spelled out because schema §6's C9 row
#: says "a ``noise_wave`` temperature" in the singular and the operator has
#: four of them.  ``gamma_src_re``/``gamma_src_im`` are NOT temperatures and
#: are not on this list: their first axis is the source index, so a 1-D one is
#: already refused by ``NoiseWaveOperator.__check_init__`` with its own
#: sentence.
NOISE_WAVE_TEMPERATURES: tuple[str, ...] = ("t_unc", "t_cos", "t_sin", "t_rx")

#: ``(field, legal shapes)`` per subject, and the two lists DIFFER.  Measured:
#: ``(n_time, n_freq)`` is REFUSED for ``CalLoadOperator.t_load``
#: ("a 2-D t_load is the per-SAMPLE form and must be exactly (n_time, 1)") and
#: LEGAL for every noise-wave temperature (``_check_temperature`` admits
#: "scalar, (n_freq,), (n_time, 1) and (n_time, n_freq)").  One message
#: quoting one list would therefore advise a shape the package refuses on half
#: its own subjects, which is the R4 loop this table exists to avoid.
_LEGAL_SHAPES: dict[str, str] = {
    "t_load": "scalar, (n_freq,) and (n_time, 1) -- and NOT (n_time, n_freq), "
              "which CalLoadOperator refuses by name because a load whose "
              "spectrum also moves is a different model than this one has",
    "temperature": "scalar, (n_freq,), (n_time, 1) and (n_time, n_freq)",
}

#: The tail both of C9's messages append.  Bound once and interpolated, so the
#: half of the sentence that is the same for both subjects cannot drift.
_C9_TAIL = (
    "config/modifiers.py ships the mechanism -- column: true forces (n,) to "
    "(n, 1) -- and until this check nothing ever DEMANDED it: measured on a "
    "square 8x8 grid, a bare 8-element temperature loads clean and is read as "
    "per-frequency, silently, whichever axis its author meant. Off a square "
    "grid the 1-D form is legal and documented and a length mismatch is "
    "already caught by name, so this fires only where the two axes are "
    "genuinely indistinguishable. Write column: true beside the value if it "
    "is per-SAMPLE, or leave it as it stands if per-CHANNEL is what was meant "
    "and say so by writing the length as n_freq"
)


def _twins(run: Built) -> list[tuple[Any, tuple[str, ...]]]:
    """``(twin, the nodes ``inference.twin.replace`` rebuilt in it)``, both of them.

    The raw twin first, then the fit twin when it is a different object.
    ``inference`` is never ``None`` (``build_inference`` returns one for a
    document with no ``inference:`` section at all), but ``fit_twin`` is read
    defensively anyway because a payload assembled by hand in a test is a
    supported caller.

    **The fit twin is a different object on almost every document**, because
    ``inference.twin.without:`` re-assembles -- the shipped fixture carries
    ``without: [noise]``.  What makes that cheap rather than duplicative is
    that a node ``replace:`` did NOT rebuild is the same operator object in
    both, so a check keyed on operator identity reports it once.
    """
    twins: list[tuple[Any, tuple[str, ...]]] = [(run.twin, ())]
    fit = getattr(run.inference, "fit_twin", None)
    if fit is not None and fit is not run.twin:
        replaced = getattr(run.inference, "replaced", ()) or ()
        twins.append((fit, tuple(replaced)))
    return twins


def _where(replaced: tuple[str, ...], node_id: str) -> str:
    """The document key that built this node, on this twin.

    ``model.<node>`` unless ``inference.twin.replace`` rebuilt it, which is the
    route §0.3 E.10 records ``preflight/model.py::_nodes`` cannot see.  It
    reaches ``build_node_operator`` down the same path and it is a real,
    separately-editable document line, so it gets its own ``where`` rather than
    being folded into ``model:``'s.
    """
    if node_id in replaced:
        return f"inference.twin.replace.{node_id}"
    return f"model.{node_id}"


def _placed(twin: Any, node_id: str) -> list[tuple[str, Any]]:
    """``(address, operator)`` for every instance sitting at ``node_id``.

    The address is the id ``Assembly.__getitem__`` reaches the operator by:
    the node id for a single instance, and the minted ``cal_loads_1`` /
    ``cal_loads_2`` ids when several sit on a ``many`` node -- asking for the
    bare id there raises ``AmbiguousNodeError`` by design.

    ``lit`` is consulted rather than catching ``KeyError``, so a node this
    document never lights costs an ``in`` and nothing else.  A twin that is
    not an ``Assembly`` at all (``model.kind: pipeline``) has no ``lit`` and
    stands every check in this module down.
    """
    if node_id not in (getattr(twin, "lit", None) or ()):
        return []
    siblings = dict(getattr(twin, "instances", ()) or ()).get(node_id)
    if siblings is None:
        return [(node_id, twin[node_id])]
    return [(address, twin[address]) for address in siblings]


# ---------------------------------------------------------------------------
# B5
# ---------------------------------------------------------------------------


def _b5_switch(twin: Any) -> tuple[int, str] | None:
    """``(switch positions, how the fold got them)``, or ``None`` to stand down.

    Three states, and the middle one is the reason this reads ``materialized``
    and ``skipped`` rather than reaching for ``twin["receiver_input"]`` and
    catching:

    * **materialized** -- two or more live branches reached the selector, so
      it really is a ``SelectOperator`` and ``names`` is its branch list;
    * **skipped** -- exactly one branch reached it, so the fold traversed the
      node as identity, there is no ``SelectOperator``, no ``names``, and
      **one** position;
    * neither -- nothing reached it, so this document has no switch to compare
      and the check has nothing to say.
    """
    if "receiver_input" in (getattr(twin, "materialized", None) or ()):
        names = list(getattr(twin["receiver_input"], "names", ()))
        return len(names), (
            f"twin['receiver_input'].names is {names} -- fold labels minted "
            "from graph node ids rather than the document's own, and names[0] "
            "moves with the model (it becomes 'astro_sum' the moment a second "
            "antenna-side node is lit), so a positional comparison cannot be "
            "written and a LENGTH comparison is the only one that exists")
    if "receiver_input" in (getattr(twin, "skipped", None) or ()):
        return 1, ("only one branch reaches receiver_input, so the fold "
                   "traversed the selector as identity: there is no switch at "
                   "all and every sample takes that one branch, whatever "
                   "coords.extra['receiver_input'] holds")
    return None


@register_built("B5")
def _switch_positions(run: Built) -> Iterable[Finding]:
    """B5: the switch the fold built has as many positions as the order declares.

    **The hole this closes is the DARK ANTENNA BRANCH and nothing else.**
    ``preflight/observing.py``'s A14 already refuses a document that declares
    an order and places no calibration load, and ``compose.py``'s
    ``cal_load_order_problem`` already refuses keys that are not
    ``switching.order[1:]`` in that order -- so on a synthetic run the only
    surviving shape is loads present with no source upstream of the switch.
    Measured: ``order: [antenna, ambient, hot]`` with the two loads lit and
    ``global_signal``/``uniform_sky`` removed gives
    ``names == ('cal_loads_1', 'cal_loads_2')`` against three declared
    positions, ``load_document`` accepts it, and the forward run reads
    position 0 ("antenna") as the ambient load, position 1 as the hot load and
    position 2 as zero.

    **The INGESTED route is a DECLARED FALSE NEGATIVE, and the reason is a
    measurement rather than an omission.**  ``preflight/observing.py`` stands
    A14 down entirely on ``observation.from_file``, so that route has no guard
    -- and this check cannot be it.  On an ingested run the antenna's
    contribution IS the recorded waterfall, not a model source, so the antenna
    branch is *structurally* dark: measured on
    ``test_preflight_ingest.py``'s canonical working document (three recorded
    switch labels, two ``from: thermistors`` loads, a ``gain``),
    ``names == ('cal_loads_1', 'cal_loads_2')`` against three declared
    positions -- and lighting a source to "fix" it is impossible, because an
    assembly with sources refuses a state that already carries data
    (``AssemblyError: This assembly contains source operators and generates
    its own data``, measured on that same twin).  So every ingested run with a
    cycle and a load would be refused, including the one the package ships as
    canonical, and the sentence would name an edit that does not exist (R4).
    Recorded in ``test_inflight_twin.py`` by name and left to the plan that
    can decide what an ingested switched run is supposed to assemble to.

    **§0.2 C-9: a LENGTH comparison, and the reason is measured.**  Schema §6
    says "positionally"; the assembled names are fold labels, ``names[0]``
    moves with the model, and no user label ever reaches them.

    **The first twin with a mismatch wins, and the two are not both
    reported.**  ``replace:`` cannot change how many branches reach the
    selector (``replace_node`` refuses a materialized junction by name), but
    ``inference.twin.without:`` CAN take the last antenna-side source out of
    the fit twin alone -- so the fit twin is walked and named separately, while
    a document whose raw twin is already wrong hears it once, about the
    ``model:`` line that is the cause.
    """
    order = getattr(run.context, "switch_order", ()) or ()
    # `isinstance` and not a bare truthiness test: a payload assembled by hand
    # is a supported caller (`test_config_inflight.py` assembles several) and
    # `len()` on an int raises INSIDE the check, which `sweep` reports as
    # "check 'B5' RAISED" -- the loss of every finding after it.
    if not isinstance(order, (tuple, list)) or not order:
        return
    if getattr(run.context, "ingest", None) is not None:
        return
    for twin, _replaced in _twins(run):
        switch = _b5_switch(twin)
        if switch is None:
            continue
        positions, how = switch
        if positions == len(order):
            continue
        raw = twin is run.twin
        where = "model.cal_loads" if raw else "inference.twin"
        which = "this run's twin" if raw else "the FIT twin"
        # The aside carries the fit twin's OWN edit, because on that route
        # `model:` is already right and the line to change is the one taking
        # the source back out again.
        aside = "" if raw else (
            " That is the twin inference.twin's without:/replace: rebuilt, "
            "and it is the one the fit evaluates -- so the edit is there: "
            "inference.twin.without: is taking the last antenna-side source "
            "out of it while model: keeps one.")
        yield refuse(
            "B5", where,
            f"observation.switching.order declares {len(order)} switch "
            f"positions {list(order)} and the receiver_input switch {which} "
            f"built has {positions}.{aside} "
            "The antenna branch is dark: the calibration "
            "loads are lit and nothing upstream of antenna_loss is, so the "
            "loads slide down into a cycle whose position 0 is the antenna. "
            "Every switch label is then off by one and the last position "
            "selects nothing -- and nothing raises: measured, such a document "
            "builds, its twin runs, and the data comes back finite, correctly "
            f"shaped and calibrated against the wrong load. {how}. Light a "
            "source on the antenna branch (global_signal, uniform_sky, "
            "foregrounds, point_sources, observed_astro_sky, ground_pickup, "
            "atmosphere or t_sys_extra): that is the fix, and it is the one "
            "that leaves a run which can still generate its own data. "
            "Dropping the cycle instead -- observation.switching: {mode: "
            "none}, and model.cal_loads with it -- silences this too, but on "
            "a model whose only sources ARE the loads it leaves a pure "
            "transform chain with nothing to transform, and the next "
            "simulation fails with 'This assembly is a pure transform chain'; "
            "it is a fix only for a run that has a source elsewhere. "
            "This check runs after build_resources, so it saves no beam: what "
            "it buys is the refusal instead of a plausible answer (check B5).")
        # ONE finding, not two. The fit twin inherits the raw twin's dark
        # branch, so reporting both would say the same thing twice about one
        # cause -- and `built_only` exists to catch exactly that.
        return


# ---------------------------------------------------------------------------
# C9
# ---------------------------------------------------------------------------


def _square(run: Built) -> int | None:
    """``n`` when ``n_time == n_freq``, else ``None``.

    The whole condition of C9's TRAP: off a square grid a 1-D temperature's
    length already says which axis it runs along, ``CalLoadOperator`` and
    ``_check_temperature`` both refuse a mismatch by name, and demanding
    ``column:`` there would refuse documents the package builds.
    """
    shapes = [getattr(getattr(run.context, axis, None), "shape", None)
              for axis in ("time", "freq")]
    if not all(isinstance(shape, tuple) and shape for shape in shapes):
        return None
    n_time, n_freq = int(shapes[0][0]), int(shapes[1][0])
    return n_time if n_time == n_freq else None


def _ambiguous(operator: Any, field: str, n: int) -> tuple[int, ...] | None:
    """The leaf's shape when it is the ambiguous ``(n,)``, else ``None``.

    Reads ``.shape`` off the built leaf rather than the document's text, which
    is the whole reason this row is in the built slot: ``{ones: [n_freq]}`` and
    ``{ones: [n_time]}`` resolve to the identical object on a square grid, and
    ``{ref:}`` and ``{file:}`` spellings carry no shape in the text at all.
    """
    shape = getattr(getattr(operator, field, None), "shape", None)
    if not isinstance(shape, tuple) or shape != (n,):
        return None
    return shape


def _cal_load_key(document: Mapping[str, Any], where: str, address: str,
                  addresses: list[str]) -> str:
    """``model.cal_loads.<label>`` when the label can be read, else the node.

    The twin's addresses are ``cal_loads_1``/``cal_loads_2`` -- minted ids, not
    document keys -- so the label is recovered by position from the FAN
    mapping, which ``compose.cal_load_order_problem`` has already pinned equal
    to ``switching.order[1:]`` in that order.  Everything is guarded rather
    than assumed: a document value may be a list, an int or ``None``, and a
    membership or index against one of those raises inside the check, which
    ``sweep`` turns into "check 'C9' RAISED" -- the loss of every later
    finding on that document, with every ``match=`` pin still passing.
    """
    if not where.startswith("model.") or not isinstance(document, Mapping):
        return where
    model = document.get("model")
    section = model.get("cal_loads") if isinstance(model, Mapping) else None
    if not isinstance(section, Mapping):
        return where
    labels = list(section)
    if len(labels) != len(addresses) or address not in addresses:
        return where
    return f"{where}.{labels[addresses.index(address)]}"


@register_built("C9")
def _square_grid_column(run: Built) -> Iterable[Finding]:
    """C9: a bare 1-D temperature on a grid whose two axes are the same length.

    Fires on ``CalLoadOperator.t_load`` and on all four ``NoiseWaveOperator``
    temperatures -- ``t_unc``, ``t_cos``, ``t_sin``, ``t_rx``, named rather
    than left to schema §6's singular -- and **only when ``n_time ==
    n_freq``**.

    **The legal-shape list differs per subject and the message says so.**
    ``(n_time, n_freq)`` is refused for ``t_load`` and legal for a noise-wave
    temperature, so a single quoted list would be advice the package refuses
    on half its own subjects (§0.3 E.6).

    Walks the raw twin and the fit twin (:func:`_twins`), so
    ``inference.twin.replace.cal_loads`` and
    ``inference.twin.replace.noise_wave`` are covered and reported at their own
    document keys.  Leaves the fit twin did not rebuild are the same objects as
    the raw twin's, so each one is reported once -- keyed on the LEAF's
    identity rather than the operator's, because ``replace:`` rebuilding a node
    does not necessarily rebuild every field of it.
    """
    n = _square(run)
    if n is None:
        return
    seen: set[int] = set()
    for twin, replaced in _twins(run):
        for node_id, fields, subject in (
            ("cal_loads", ("t_load",), "t_load"),
            ("noise_wave", NOISE_WAVE_TEMPERATURES, "temperature"),
        ):
            placed = _placed(twin, node_id)
            addresses = [address for address, _ in placed]
            for address, operator in placed:
                for field in fields:
                    leaf = getattr(operator, field, None)
                    if _ambiguous(operator, field, n) is None or id(leaf) in seen:
                        continue
                    seen.add(id(leaf))
                    where = _where(replaced, node_id)
                    if node_id == "cal_loads":
                        where = _cal_load_key(run.document, where, address,
                                              addresses)
                    yield refuse(
                        "C9", where,
                        f"{where}.{field}: this run's time and frequency axes "
                        f"are both {n} long, and this value is a bare "
                        f"({n},) array -- so nothing says whether it is one "
                        "temperature per SAMPLE or one per CHANNEL, and the "
                        "package reads a bare 1-D value as per-channel "
                        "whatever was meant. The legal shapes here are "
                        f"{_LEGAL_SHAPES[subject]}. {_C9_TAIL} (check C9).")
