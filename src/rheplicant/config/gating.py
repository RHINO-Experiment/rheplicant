"""The words an ``inference.checks:`` entry may say, and the one place a mode
and a ``report:`` are turned into a verdict.

Schema §4.7.8 asks one question of two orthogonal axes and the ambiguity has
been shipped in prose three times.  This module separates them:

* **``mode`` decides what a FAILURE produces** -- and, for ``skip``, that the
  check does not run at all.
* **``report:`` decides whether a check's NUMBERS are recorded when it
  PASSES.**

:func:`verdict` is the only place the cross-product of the two is decided.  A
check composes the sentence, because it holds the numbers and knows whether it
failed; this module owns the SEVERITY, because a severity chosen per check is
six chances to disagree about what ``mode: warn`` means.

**Four unrelated things in this layer are spelled "report", and conflating
any two of them is the likeliest way to ship something that reads right:**

============================================ ====================================
``inference.checks.<n>.report: true``        a DOCUMENT KEY: record this
                                             check's numbers.  :attr:`Gate.record`
``mode: report``                             a DOCUMENT MODE: do not gate on
                                             this check's failure.  In
                                             :data:`MODES`
``findings.REPORT``                          a SEVERITY, and its constructor
``ConfiguredRun.report``                     the whole ``Report`` a document
                                             earned
============================================ ====================================

**Six effective states, four of them writable.**  ``off`` (nobody asked for
this check) and ``auto_skip`` (it was asked for and is undefined here) are not
in :data:`MODES` and cannot be typed into a document.  ``off`` is deliberately
not spelled ``skip``: check A37 requires every written ``mode: skip`` to carry
its own ``reason:``, and a default-off check has no author to write one, so
collapsing the two would either force a fake reason into the record or force
A37 to exempt a case it cannot distinguish.  A37 reads the document's TEXT and
therefore never sees either of these two.

**Why :meth:`Gate.runs` is a positive test.**  ``state in ("refuse", "warn",
"report")`` and never ``state not in ("skip", ...)``: a state added later must
default to NOT running, because a check that is lost is silent and a refusal
that should not have happened is loud.

**What this module may import.**  ``errors``, ``findings`` and ``resources``,
and nothing else from this package -- in particular **not** ``sections/``.
``preflight/gated.py`` imports this module, and a transitive reach into
``sections/inference.py`` would drag ``jax.numpy`` and the builders behind the
pre-flight scope guard.  ``resources`` is on the list for
:func:`~rheplicant.config.resources.check_unknown_keys` alone (§2.5: one
unknown-key sweep, not a fourth hand-rolled one); it costs about half a second
and pulls no optional dependency, and ``preflight/noise.py`` already imports
it.  That helper RAISES, so :func:`check_gates` catches its ``ConfigError``
and hands the message on verbatim as a finding -- this module's whole contract
is that it raises nothing.

**The grammar moved here whole.**  ``sections/inference.py::_checks`` used to
raise seven sentences of its own; it now calls :func:`check_gates` and raises
the first finding's message, character for character.  That is the one legal
shape of a hoist: the section keeps refusing, and there is one binding for
each sentence rather than two that drift.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

from rheplicant.config import findings
from rheplicant.config.errors import ConfigError
from rheplicant.config.resources import check_unknown_keys

__all__ = [
    "AUTO_SKIP",
    "AUTO_SKIP_ID",
    "CHECK_ID",
    "CHECK_NAMES",
    "DEFAULT_MODE",
    "MODES",
    "OFF",
    "STATES",
    "Gate",
    "auto_skipped",
    "check_gates",
    "gates",
    "verdict",
]

#: The four words a document may write.  Moved here from
#: ``sections/inference.py``'s ``_MODES``, which is now an import of this: one
#: binding, because Plan 2D lost a task to one drafter's frozenset meeting
#: another's tuple.  **A TUPLE, and the order is load-bearing** -- the
#: mode-enum refusal interpolates ``list(MODES)`` and that sentence is pinned
#: by equality.
MODES: tuple[str, ...] = ("refuse", "warn", "report", "skip")

#: The three checks ``inference.checks`` knows.  Moved here from
#: ``sections/inference.py``'s ``_CHECK_NAMES``, likewise.  **A FROZENSET** --
#: it is used in ``set(spec) - allowed`` shapes and the refusal sorts it.
CHECK_NAMES: frozenset[str] = frozenset({"identifiability", "linearity", "prior_sensitivity"})

#: No user asked for this check.  NOT in :data:`MODES`, so it cannot be
#: written, so check A37 never has to exempt it.
OFF: str = "off"

#: The user asked for this check and it is undefined on this document; the
#: reason is generated rather than written.  NOT in :data:`MODES` either.
AUTO_SKIP: str = "auto_skip"

#: Every effective state, in the order §2.3's table gives them.  SIX: the four
#: writable ones, then the two that are not.
STATES: tuple[str, ...] = ("refuse", "warn", "report", "skip", OFF, AUTO_SKIP)

#: Schema §11.15's defaults.  ``linearity`` is the ONLY check on by default
#: and the cost table is why: it is ``len(scales) + 1`` forward passes per
#: linear claim, while ``identifiability`` is a ``jacfwd`` plus a dense SVD and
#: ``prior_sensitivity`` is that plus two Newton solves.  A dict over exactly
#: :data:`CHECK_NAMES`, so a fourth check name cannot arrive without a default
#: being chosen for it.
DEFAULT_MODE: dict[str, str] = {
    "linearity": "refuse",
    "identifiability": OFF,
    "prior_sensitivity": OFF,
}

#: check name -> the schema §6 id its findings carry.  C18 is the two-sigma
#: cross-check and belongs to no gate, so it is not here.
CHECK_ID: dict[str, str] = {
    "linearity": "C12",
    "identifiability": "C13",
    "prior_sensitivity": "C19",
}

#: The id an auto-skip reports under.  Never a registry slot: :func:`verdict`
#: binds it, so a user grepping the record for C14 finds every check that was
#: asked for and could not be decided.
AUTO_SKIP_ID: str = "C14"


class Gate(NamedTuple):
    """One check's effective configuration, defaults applied."""

    #: A member of :data:`CHECK_NAMES`.
    name: str
    #: A member of :data:`STATES`.
    state: str
    #: The document's ``report:``.  ALWAYS ``False`` for a non-running state:
    #: there are no numbers to record from a check that did not run.
    record: bool
    #: Written by the document for ``skip``, generated for ``auto_skip``,
    #: ``None`` otherwise.
    reason: str | None
    #: ``identifiability`` alone; ``None`` everywhere else.
    rtol: float | None

    def runs(self) -> bool:
        """Does this check get evaluated at all?

        A POSITIVE test over the three running states, never a negation of the
        non-running ones -- see this module's docstring.
        """
        return self.state in ("refuse", "warn", "report")

    def where(self) -> str:
        """The path the USER edits to change this gate.

        Never the subject's path: a finding's ``where`` is the line that caused
        the failure, and this is the line that decides what the failure costs.
        """
        return f"inference.checks.{self.name}"


def check_gates(section: Any) -> tuple[findings.Finding, ...]:
    """The ``inference.checks:`` GRAMMAR, as findings rather than as a raise.

    ``section`` is the RAW document value at ``inference.checks`` -- a mapping,
    or ``None``.  **Raises nothing**: a pre-flight check that raised would
    abort its whole pass and hide every finding after it.  The one helper here
    that does raise (``check_unknown_keys``) is caught and its message handed
    on verbatim.

    Decides, in this order, and the list is exhaustive:

    1. ``checks:`` is a mapping;
    2. every name is a check v1 knows;
    3. every entry is a mapping;
    4. the entry's key set, via ``check_unknown_keys``;
    5. ``mode`` is one of :data:`MODES`;
    6. ``mode: skip`` carries a string ``reason:`` (check A37);
    7. ``reason:`` appears only with ``mode: skip``;
    8. ``mode: skip`` is not asked to ``report:`` numbers it cannot produce.

    **At most ONE finding per entry**, in document order, so ``found[0]`` is
    byte-identical to the sentence ``sections/inference.py::_checks`` has
    raised since Plan 2B.  Item 8 is the only new one; the other seven moved
    here whole.

    Returns:
        The refusals, in document order.  Empty when the section is legal.
    """

    def decide(name: Any, spec: Any) -> findings.Finding | None:
        where = f"inference.checks.{name}"
        if name not in CHECK_NAMES:
            return findings.refuse(
                "A1", where, f"{where}: {name!r} is not a check; v1 knows {sorted(CHECK_NAMES)}."
            )
        if not isinstance(spec, Mapping):
            return findings.refuse("A1", where, f"{where}: is a mapping with mode:; got {spec!r}.")
        allowed = frozenset({"mode", "report", "reason"}) | (
            frozenset({"rtol"}) if name == "identifiability" else frozenset()
        )
        try:
            check_unknown_keys(where, dict(spec), allowed, label="a check:")
        except ConfigError as unknown:
            return findings.refuse("A1", where, str(unknown))
        mode = spec.get("mode")
        if mode not in MODES:
            return findings.refuse(
                "A1", where, f"{where}.mode: is one of {list(MODES)}; got {mode!r}."
            )
        reason = spec.get("reason")
        if mode == "skip" and not isinstance(reason, str):
            return findings.refuse(
                "A37",
                where,
                f"{where}: mode: skip carries its own reason: (check A37) -- "
                "three unrelated skips sharing one sentence was v0's mistake.",
            )
        if mode != "skip" and reason is not None:
            return findings.refuse("A1", where, f"{where}: reason: belongs to mode: skip alone.")
        if mode == "skip" and bool(spec.get("report", False)):
            return findings.refuse(
                "A1",
                where,
                f"{where}: mode: skip and report: true together ask to record "
                "the numbers of a check that will not run. Drop report:, or "
                "drop reason: and change mode: skip to mode: report so the "
                "check runs and has numbers to record (check A1).",
            )
        return None

    if section is None:
        return ()
    if not isinstance(section, Mapping):
        return (
            findings.refuse(
                "A1", "inference.checks", f"inference.checks: is a mapping; got {section!r}."
            ),
        )
    found = []
    for name, spec in section.items():
        one = decide(name, spec)
        if one is not None:
            found.append(one)
    return tuple(found)


def gates(section: Any, *, audit: Any = None) -> dict[str, Gate]:
    """The three gates, defaults applied, for a section that passed
    :func:`check_gates`.

    **CARDINALITY IS THREE, always**, keyed by :data:`CHECK_NAMES`, whatever
    the document says.  ``_checks(None)`` returning ``{}`` is what makes "what
    mode is linearity in" a question with no answer, and a caller that has to
    write ``gates.get("linearity")`` writes ``.get("linearity", <its own
    default>)`` instead -- and then there are two default tables and one of
    them is wrong.
    """
    declared = section if isinstance(section, Mapping) else {}
    out: dict[str, Gate] = {}
    for name in sorted(CHECK_NAMES):
        spec = declared.get(name)
        if not isinstance(spec, Mapping):
            spec = {}
        if "mode" in spec:
            state = spec["mode"]
        elif audit is None:
            state = DEFAULT_MODE[name]
        elif name == "linearity":
            state = audit.use_default("inference.checks.linearity.mode", DEFAULT_MODE[name])
        elif name == "prior_sensitivity":
            state = audit.use_default(
                "inference.checks.prior_sensitivity.mode", DEFAULT_MODE[name]
            )
        else:
            state = audit.use_default("inference.checks.identifiability.mode", DEFAULT_MODE[name])
        rtol = spec.get("rtol") if name == "identifiability" else None
        if audit is not None and name == "identifiability" and "rtol" not in spec:
            rtol = audit.use_default("inference.checks.identifiability.rtol", None)
        if "report" in spec:
            report = bool(spec["report"])
        elif audit is None:
            report = False
        elif name == "linearity":
            report = audit.use_default("inference.checks.linearity.report", False)
        elif name == "identifiability":
            report = audit.use_default("inference.checks.identifiability.report", False)
        else:
            report = audit.use_default("inference.checks.prior_sensitivity.report", False)
        gate = Gate(
            name=name,
            state=state,
            record=report,
            reason=spec.get("reason") if state == "skip" else None,
            rtol=float(rtol) if rtol is not None else None,
        )
        # `record` is `report:`, and `report:` governs the numbers of a check
        # that RAN.  Forced here rather than trusted from the document so that
        # a caller reading `gate.record` never has to ask `gate.runs()` first.
        out[name] = gate if gate.runs() else gate._replace(record=False)
        if audit is not None:
            effective = out[name]
            audit.gate(
                name,
                CHECK_ID[name],
                state,
                effective.state,
                effective.reason,
            )
    return out


def auto_skipped(gate: Gate, reason: str) -> Gate:
    """The same gate in the :data:`AUTO_SKIP` state, carrying ``reason``.

    Returns a NEW gate and mutates nothing -- ``Gate`` is a ``NamedTuple``, so
    the only way to mutate would be to rebind the mapping entry, and the
    mapping the pass hands round is shared by every check.

    ``record`` is forced ``False``: an auto-skip reports regardless (a check
    the user asked for and did not get must say so), and a ``True`` here would
    read as if ``report:`` had caused that.

    Raises:
        ValueError: ``reason`` is falsy (``""`` or ``None`` despite the
            annotation).  This function is called only from ``src/``, never
            from a document's own words, so an empty reason here is a bug in
            the CALLER that generated it -- never the document's fault.
            Left unchecked, :func:`verdict`'s ``gate.reason or ""`` fallback
            would turn it into a REPORT finding whose message is the empty
            string: an auto-skip saying nothing, which is the exact
            silent-loss failure mode this state exists to end, only shipped
            with a green check mark instead of an obvious gap.
    """
    if not reason:
        raise ValueError(
            "auto_skipped(gate, reason=...): reason must not be empty. An "
            "auto-skip with no reason is silent -- the failure mode this "
            'state exists to end -- and verdict()\'s `gate.reason or ""` '
            "fallback would otherwise turn it into a REPORT finding whose "
            "message says nothing."
        )
    return gate._replace(state=AUTO_SKIP, record=False, reason=reason)


def verdict(gate: Gate, *, failed: bool, where: str, message: str) -> findings.Finding | None:
    """The cross-product of a mode and a ``report:``, in one place.

    **AT MOST ONE finding, ever.**  A failure with ``report: true`` is ONE
    finding at the mode's severity whose message carries the numbers, not a
    refusal and a report: two would double-count in ``Report.checks()`` and
    make ``raise_if_refused``'s "N more refusals" tail wrong.

    Args:
        gate: the check's effective configuration.
        failed: did the check's own question come out badly?
        where: the SUBJECT's document path (``inference.parameters.g``,
            ``model.adc``) -- the line that caused this, not the gate's own.
            ``gate.where()`` is one hop away in the message.
        message: the sentence for THIS outcome, composed by the caller, which
            is the only place the numbers live.

    Returns:
        The one finding, or ``None`` when this outcome says nothing.  A gate
        that does not run and is not an auto-skip returns ``None`` whatever
        ``failed`` says: the caller should not have called, and raising would
        turn a caller's bug into the document's refusal.
    """
    if gate.state == AUTO_SKIP:
        # Ignores `failed`, `message` AND `gate.record`: the point of this
        # branch is that the check produced no answer at all, and silence
        # about a check the user asked for is the failure mode this state
        # exists to end.
        return findings.report(AUTO_SKIP_ID, where, gate.reason or "")
    if not gate.runs():
        return None
    check = CHECK_ID[gate.name]
    if failed:
        if gate.state == "refuse":
            return findings.refuse(check, where, message)
        if gate.state == "warn":
            return findings.warn(check, where, message)
        return findings.report(check, where, message)
    if gate.record:
        return findings.report(check, where, message)
    return None
