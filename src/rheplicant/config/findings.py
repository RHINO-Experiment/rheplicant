"""One check's verdict, and the ledger a whole pass hands back.

Until this module the config layer had exactly one way to say anything:
:class:`~rheplicant.config.errors.ConfigError`, raised at the first problem.
That is the right answer for a document that cannot be run and the wrong
answer to two other things schema §6 asks for.

**Collecting rather than raising.**  ``document._sweep`` raised at the first
problem, so a user with four errors paid four round trips and schema §10's
"Validate" button had nothing to populate.  Here a check RETURNS findings and
raises nothing, a pass collects them, and :meth:`Report.raise_if_refused`
turns the collection back into the one refusal type this layer has.

**The first refusal, verbatim.**  ``raise_if_refused`` raises a ``ConfigError``
whose message IS the first refusal's message, character for character, with a
one-line tail naming where the others are.  That is not politeness; it is what
lets a check MOVE out of a section and into the pre-flight pass without
rewriting the suite.  Measured with::

    python -c "import pathlib, re; \\
      pat = re.compile(r'pytest\\.raises\\(\\s*ConfigError\\s*,\\s*match=') ; \\
      print(sum(len(pat.findall(p.read_text())) \\
                for p in pathlib.Path('tests/config').rglob('*.py')))"

**442** assertions across 44 of ``tests/config``'s 57 modules pinned a refusal
message this way at ``be2027b``, the commit this file was written against.  Run
the same command at the commit that ADDS this file and it returns **445 across
45 of 58**: this module's own test file is the 58th, and three of its pins are
the extra three.  Both numbers are quoted because one of them is what the
command above prints today, and a docstring whose own command contradicts it is
worse than one with no command in it.

The count is not the load-bearing measurement.  This is: the same walk, asking
which of those patterns ends in ``$``, returns **0**, and asking which begin
with ``^`` returns **0** too.  No pinned pattern is anchored at either end, and
``pytest.raises(match=...)`` searches rather than fullmatches, so a tail
appended AFTER the first message cannot break one.  A prefix prepended BEFORE it
could not break one either, on today's patterns -- but it would move the first
line every user reads, so there is none.  **The zeros are what a later plan must
re-measure if it starts anchoring**; the 442 and the 444 will drift on their own
and mean nothing when they do.

**Three severities, not four.**  ``sections/inference.py:43`` parses
``_MODES = ("refuse", "warn", "report", "skip")`` and it is tempting to mirror
all four here.  ``skip`` is not a verdict.  It is a MODE a DOCUMENT declares
about a named check -- "do not run this one on me" -- and it carries its own
``reason:`` (``sections/inference.py:103-108``, which is check A37).  A check
that has been skipped emits nothing at all, so there is no finding whose
severity is ``"skip"``, and a fourth member here would invite one.
:meth:`Report.of` refuses the token by name for exactly that reason: the
person who reaches for it gets a sentence rather than an empty tuple.

**What consumes ``report``.**  Nothing, yet.  ``REPORT`` is the severity
``checks.<name>.mode: report`` (schema §4.7.8) will read, and that gating is
Plan 3C's.  It is bound here rather than later because ``SEVERITIES`` is what
:meth:`Report.of` validates against, and a two-member tuple that grows in a
later plan would make every ``of`` call site a place to re-check.
"""

from __future__ import annotations

import dataclasses
import warnings

from rheplicant.config.errors import ConfigError

__all__ = ["REFUSE", "REPORT", "SEVERITIES", "WARN", "ConfigWarning",
           "Finding", "Report", "refuse", "report", "warn"]

#: The document cannot be run as written.
REFUSE: str = "refuse"
#: The document will run, and is probably not what its author meant.
WARN: str = "warn"
#: Worth recording next to the run; not worth interrupting anyone over.
REPORT: str = "report"

#: The three verdicts a check may reach.  NOT ``sections/inference.py:43``'s
#: ``_MODES`` -- see this module's docstring for why ``skip`` is absent.
SEVERITIES: tuple[str, ...] = (REFUSE, WARN, REPORT)


class ConfigWarning(UserWarning):
    """A document this package will run, written in a way that is probably wrong.

    ``UserWarning`` rather than bare ``Warning``: the default filter shows
    ``UserWarning`` and the six ``pytest.warns(UserWarning, ...)`` sites this
    repository already has keep catching this layer's warnings.  A warning
    class nobody's filter admits is the failure this whole channel exists to
    avoid.
    """


@dataclasses.dataclass(frozen=True, slots=True)
class Finding:
    """One check's verdict about one place in the user's document."""

    #: The schema §6 id -- ``"A30"`` -- or ``""`` for a check with no id.
    check: str
    #: One of :data:`SEVERITIES`.
    severity: str
    #: The path into the USER'S DOCUMENT the reader must go and edit:
    #: ``"model.noise"``, ``"runs[2].blocks[0]"``, ``"inference.parameters.g"``.
    #: Never a path into ``src/``.  The pass validates the shape.
    where: str
    #: One complete sentence carrying the fix, in this layer's voice.  Ends
    #: with ``"(check A30)."`` when :attr:`check` is set.
    message: str

    def __post_init__(self) -> None:
        # A severity outside the three is a bug in a CHECK, never a fault of
        # the document, so it is a ValueError and not a ConfigError: a caller
        # who wrapped `load_document` in `except ConfigError` would otherwise
        # swallow it and report the document as at fault.  (ConfigError IS a
        # ValueError -- `config/errors.py` -- so the direction that matters is
        # the one this comment states, and the test for it asserts `not
        # isinstance(..., ConfigError)` rather than the exception type alone.)
        if self.severity not in SEVERITIES:
            raise ValueError(
                f"Finding.severity is one of {list(SEVERITIES)}; got "
                f"{self.severity!r}. Use refuse(), warn() or report()."
            )


@dataclasses.dataclass(frozen=True, slots=True)
class Report:
    """Everything a pass found, in the order the checks ran."""

    findings: tuple[Finding, ...] = ()

    def of(self, severity: str) -> tuple[Finding, ...]:
        """The findings at one severity, in declaration order."""
        if severity not in SEVERITIES:
            raise ValueError(
                f"{severity!r} is not one of {list(SEVERITIES)}. "
                "'skip' in particular is a MODE a document declares about a "
                "check (inference.checks.<name>.mode, schema §4.7.8), never a "
                "verdict a check reaches -- a skipped check emits nothing."
            )
        return tuple(one for one in self.findings if one.severity == severity)

    def refusals(self) -> tuple[Finding, ...]:
        """The findings that stop the document being run."""
        return self.of(REFUSE)

    def warnings(self) -> tuple[Finding, ...]:
        """The findings that do not.

        The name shadows the ``warnings`` MODULE inside this class's body and
        not inside its methods: a method body resolves ``warnings`` in the
        module globals, so :meth:`emit_warnings` below reaches the stdlib.
        Spelled out because it reads like a bug and is not.
        """
        return self.of(WARN)

    def checks(self) -> frozenset[str]:
        """The schema §6 ids that fired.  A finding with no id is not one."""
        return frozenset(one.check for one in self.findings if one.check)

    def raise_if_refused(self) -> None:
        """The first refusal, verbatim, as the ``ConfigError`` it always was.

        The tail is appended, never prepended, and the module docstring
        records the measurement that makes that safe.  The first REFUSAL and
        not the first finding: a warning collected ahead of it must not
        displace the sentence the user is shown.
        """
        refusals = self.refusals()
        if not refusals:
            return
        message = refusals[0].message
        rest = refusals[1:]
        if rest:
            message += (
                f"\n(This document has {len(rest)} more refusal"
                f"{'' if len(rest) == 1 else 's'}, at "
                f"{', '.join(one.where for one in rest)}.)"
            )
        raise ConfigError(message)

    def emit_warnings(self) -> None:
        """Say the warnings out loud, one ``warnings.warn`` each, as
        :class:`ConfigWarning`.

        ``stacklevel=3`` and not the ``2`` at ``sections/parameters.py:176``,
        because the frame counts differ and the target is the same.  That call
        sits one frame from its caller; this one sits two -- the chain this
        aims at is ``user -> load_document -> emit_warnings ->
        warnings.warn``, and Plan 3A's Task 2 is what puts ``load_document``
        in it, so until that lands the only caller is a test.

        The arithmetic itself was measured, on a probe of exactly that shape:
        ``1`` blames this module, ``2`` blames the direct caller
        (``load_document``), ``3`` blames the caller's caller -- the user's
        own ``load_document(...)`` line, which is the one worth pointing at.
        A caller who runs the pass by hand and calls this directly is one
        frame short and will see ITS caller blamed; that is the price of
        aiming at the shipped route, and it is named here rather than
        discovered.
        """
        for finding in self.warnings():
            warnings.warn(finding.message, ConfigWarning, stacklevel=3)

    def __bool__(self) -> bool:
        """True iff there is any finding AT ALL -- not "iff refused".

        ``if report:`` reads like "something is wrong" and a report of pure
        warnings is exactly that.  Callers deciding whether to RAISE ask
        :meth:`refusals`, and ``load_document`` calls
        :meth:`raise_if_refused` rather than testing this at all.
        """
        return bool(self.findings)


def refuse(check: str, where: str, message: str) -> Finding:
    """This document cannot be run as written."""
    return Finding(check=check, severity=REFUSE, where=where, message=message)


def warn(check: str, where: str, message: str) -> Finding:
    """This document will run and is probably not what its author meant."""
    return Finding(check=check, severity=WARN, where=where, message=message)


def report(check: str, where: str, message: str) -> Finding:
    """Worth recording beside the run; not worth interrupting anyone over."""
    return Finding(check=check, severity=REPORT, where=where, message=message)
