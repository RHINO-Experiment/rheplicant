"""runs: the exit list (schema §4.7.9).

A run entry names WHAT to do (``kind:``), against WHICH document layer
(``variant:``), on WHICH observed data (``on:``), and whether a refusal is
the point (``expect: refuse`` -- two of ``examples/gibbs_plan.py``'s exits
exist only to be refused, and this key turns that demonstration into a
checkable assertion).  Kind-specific keys travel untouched in ``options``;
each executor in ``sections/exits.py`` sweeps its own.

``runs:`` is read from the BASE document; a variant patching ``runs:``
changes what ``load_document(variant=...)`` accepts, never which runs
execute -- executor's decision, recorded in the 2B plan.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

from rheplicant.config.errors import ConfigError

__all__ = ["RunResult", "RunSpec", "parse_runs", "run_document"]

_RUN_KEYS = frozenset({"name", "kind", "variant", "on", "reuse", "expect"})
_KINDS = ("forward", "fisher", "optimize", "plan.estimate", "plan.sample",
          "conjugate.wiener", "conjugate.gcr", "conjugate.gls", "condition",
          "identifiability", "score_directions", "gradient", "mmodes",
          "predict", "nuts", "npe")
# Plan 2C's and Plan 2D's own deferral tuples are GONE rather than emptied:
# `predict` was 2C's last member and `npe` was 2D's, and an empty one would
# leave `if kind in ()` in `_one` below -- dead, green and forever, and read
# by the next author as a kind still owed.  Neither name is written here
# either, so that `grep -rn <that name> src` stays the check it was meant to
# be.  A deferral tuple a LATER plan adds must be named `_KINDS_*`: the
# disjointness guard discovers its tables by that prefix.
_KINDS_PLAN4 = ("compare", "benchmark")


class RunSpec(NamedTuple):
    """One parsed run entry."""

    name: str
    kind: str
    variant: str | None
    on: str
    expect: str
    options: dict[str, Any]
    reuse: str | None = None


class RunResult(NamedTuple):
    """One executed run: its product, or the refusal it expected.

    ``variant`` is the ``variant:`` the run was CONFIGURED on, so a later
    run reading this product can tell whether it is reading its own build.

    It is DEFAULTED so that a construction site which does not know about it
    still binds.  That buys the FOUR test sites, all of which pass four
    arguments -- three predate this commit and are unchanged, the fourth
    pins the default itself; it does not buy the two in ``src`` -- both sit
    in ``execute_run`` and both were edited to pass ``variant=run.variant``,
    because a field nothing populates is a field that reads ``None`` forever.
    The default is therefore a compatibility shim for callers outside this
    module, not a claim that this was a one-line change.
    """

    name: str
    kind: str
    product: Any
    error: Exception | None
    variant: str | None = None


def _one(index: int, entry: Any, several: bool) -> RunSpec:
    where = f"runs[{index}]"
    if not isinstance(entry, Mapping):
        raise ConfigError(f"{where}: is a mapping; got {entry!r}.")
    kind = entry.get("kind")
    if kind is None:
        raise ConfigError(f"{where}: kind: is required.")
    if kind in _KINDS_PLAN4:
        raise ConfigError(
            f"{where}: kind: {kind} arrives with Plan 4 (D-C16), with the "
            "outputs that make it reportable."
        )
    if kind not in _KINDS:
        raise ConfigError(
            f"{where}: kind: {kind!r} is not an exit; this layer runs "
            f"{list(_KINDS)}."
        )
    reuse = entry.get("reuse")
    if reuse is not None and not isinstance(reuse, str):
        raise ConfigError(f"{where}: reuse: is an earlier run's name; got "
                          f"{reuse!r}.")
    name = entry.get("name")
    if several and not isinstance(name, str):
        raise ConfigError(
            f"{where}: name: is required when there is more than one run."
        )
    if name is not None and not isinstance(name, str):
        raise ConfigError(f"{where}: name: is a string; got {name!r}.")
    variant = entry.get("variant")
    if variant is not None and not isinstance(variant, str):
        raise ConfigError(f"{where}: variant: is a name; got {variant!r}.")
    on = entry.get("on", "primary")
    if not isinstance(on, str):
        raise ConfigError(f"{where}: on: is an observed name; got {on!r}.")
    expect = entry.get("expect", "ok")
    if expect not in ("ok", "refuse"):
        raise ConfigError(f"{where}: expect: is ok or refuse; got "
                          f"{expect!r}.")
    options = {key: value for key, value in entry.items()
               if key not in _RUN_KEYS}
    return RunSpec(name=name if name is not None else kind, kind=kind,
                   variant=variant, on=on, expect=expect, options=options,
                   reuse=reuse)


def parse_runs(section: Any) -> tuple[RunSpec, ...]:
    """``runs:`` -> parsed entries, names resolved and unique."""
    if isinstance(section, Mapping):
        section = [section]
    if not isinstance(section, list) or not section:
        raise ConfigError(
            "runs: is a list of exits (or one exit mapping); got "
            f"{section!r}."
        )
    parsed = tuple(_one(index, entry, len(section) > 1)
                   for index, entry in enumerate(section))
    names = [run.name for run in parsed]
    for name in names:
        if names.count(name) > 1:
            raise ConfigError(f"runs: the name {name!r} appears twice.")
    return parsed


def run_document(document: Mapping, *,
                 base_dir: str | None = None) -> dict[str, RunResult]:
    """Execute every run a document declares, in order, by name.

    A compatibility wrapper over the orchestration (Plan 4A Task 10): the
    document is prepared through ``scope="all_layers"`` -- every declared
    variant validated, every schedule handler-parsed before the first
    executor -- the base schedule executes in declaration order, and the
    terminal error, if any, is re-raised exactly as the per-run loop used to
    raise it.  The return stays a plain ``dict`` keyed by run name.
    """
    from rheplicant.config.orchestration import execute_prepared, prepare_document

    if not isinstance(document, Mapping):
        raise ConfigError(
            f"A document is a mapping of sections; got "
            f"{type(document).__name__} ({document!r})."
        )
    prepared = prepare_document(document, scope="all_layers",
                                base_dir=base_dir)
    record = execute_prepared(prepared)
    if record.status != "ok":
        raise record.error
    return dict(record.results)
