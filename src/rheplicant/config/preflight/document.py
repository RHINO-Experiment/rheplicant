"""A1, A38 and A39 -- the document-shaped checks, decidable from text alone.

Three holes in A1's sweep, measured rather than inferred:

* ``runs[i]`` option keys.  ``runs.py:113-114`` puts every unrecognised key
  into ``RunSpec.options`` and ``exit_support._sweep`` sweeps them INSIDE the
  executor.  All sixteen registered executors do sweep -- so this is timing,
  not absence: measured, four ``forward`` runs with a typo on the last one
  execute r0, r1 and r2 before the refusal.  The fix reuses each executor's
  own allowed-key table (thirteen of the sixteen by importing the very object
  the executor sweeps with) and the executor's own ``_sweep``, so the two
  cannot drift; ``test_preflight_document.py`` reads the tables back out of
  the executors' source and compares.
* An unselected ``variants`` entry.  ``layering.py:41-85`` merges only the
  REQUESTED variant, so SEVEN document-grammar clauses never fire for the
  others: three of ``apply_variant``'s six (of the other three, two are about
  a name that is not declared, which is not a state a declared name can be in,
  and the third is a ``variants:`` section that is not a mapping, which
  :func:`_variant_text` refuses with a clause of its own) and four of
  ``_structural``'s five -- its ``schema_version`` VALUE
  clause is unreachable this way, because ``apply_variant`` refuses a patch
  that names the key at all.  Measured, a document whose unselected variant
  carries ``campaign:``, ``outputs:``, nested ``variants:``, a rewritten
  ``schema_version``, a non-mapping patch or a deleted required section loads
  clean -- six for six.
* ``resources.beams.<n>.horizon``'s two angles.  ``kinds/beams.py:482`` and
  ``:491`` are ``float(horizon.get("el_deg", 90.0))`` and
  ``float(horizon.get("apod_deg", 0.0))``: the keys are swept (``:204``) and
  the VALUES bypass the value grammar.  Measured, ``{value: 0.1, unit: rad}``
  arrives as a bare ``TypeError`` from inside the build, and under
  ``horizon.mode`` other than ``truncate_map`` it is never read at all.

**Every check here runs on the base document AND on each declared variant
merged over it.**  That is the 2C shape-4 lesson: a capability key, a run
option and a horizon angle all arrive through ``variants:`` as readily as
through the base, and a check reading ``document[...]`` closes one route and
leaves its twin open.  A finding the base document already produces is not
repeated per layer, and a finding a variant layer produces says which layer
in its own sentence -- ``Report.raise_if_refused`` quotes the MESSAGE
(``findings.py:170``), so a sentence that named only the inner path would
send a reader to grep a base document that does not contain it.

**Every import of a ``sections/`` module is function-local, deliberately.**
Two reasons, both measured: ``import rheplicant.config`` must not grow five
modules for a process that only reads a config (§0's invariant is about
optional *dependencies*, and this is the same argument one level down), and
this module is imported from ``preflight/__init__.py``'s foot while
``config/document.py`` is importing ``preflight`` -- a module-scope import of
``rheplicant.config.preflight._structural`` would be reading a
half-initialised module.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from _rheplicant_bootstrap.path_syntax import longest_legal_prefix
from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import Finding, refuse
from rheplicant.config.preflight import register

#: The eight keys schema §8 reserves at capability 3 or 4 -> (capability,
#: schema section).  Six are reachable from a registered check; the two
#: ``outputs.write`` rows are not, because ``_structural`` refuses
#: ``outputs:`` wholesale and raises before ``CHECKS`` runs.  They are carried
#: anyway so that Plan 4 inherits the capability and the section rather than
#: re-deriving them.
_CAPABILITY_KEYS: dict[str, tuple[str, str]] = {
    "campaign": ("capability 4 (streaming evidence)", "§8.2"),
    "inference.transitions": ("capability 4 (streaming evidence)", "§8.2"),
    "inference.parameters.<name>.scope":
        ("capability 4 (streaming evidence)", "§8.2"),
    "inference.parameters.<name>.support":
        ("capability 4 (streaming evidence)", "§8.2"),
    "inference.parameters.<name>.hyper":
        ("capability 4 (streaming evidence)", "§8.2"),
    "model.<node>.type: NeuralOperator":
        ("capability 3 (neural surrogates)", "§8.1"),
    "outputs.write.memory_archive":
        ("capability 4 (streaming evidence)", "§8.2"),
    "outputs.write.posterior_net":
        ("capability 3 (neural surrogates)", "§8.1"),
}

#: The two ``scope:`` names schema §8.2 reserves, and no others.  ``scope:
#: glboal`` is a typo, not a reservation, and telling its author to wait for
#: capability 4 sends them to the wrong fix; it falls through to
#: ``sections/parameters.py``'s own clause unchanged.
_TASK3_SCOPES_RESERVED = ("per_epoch", "linked")

#: run kind -> the option keys whose OWN executor raises a bespoke refusal
#: BEFORE its generic key sweep, so this pass must stand down on them.
#:
#: ``conjugate.py:580-582`` argues the ordering by name -- the bespoke refusal
#: runs "BEFORE the sweep on purpose: the sweep would fire first with the
#: generic 'does not take [...]' and the reader would fix the symptom by
#: deleting a key they had good reason to write."  Hoisting the generic sweep
#: in front of it inverts exactly that, in a phase the executor cannot reach.
#:
#: **The stand-down is for the whole RUN, not for the key alone, and that is
#: measured rather than chosen.**  Dropping only the spoken-for key and
#: sweeping the rest looks strictly better and is not:
#: ``test_config_exits_diagnostics.py:324`` declares ``prior_mean:`` AND
#: ``tol:`` on one ``condition`` run and pins that the CENTRE refusal is the
#: one heard -- "a message with no ``tol`` in it is proof of which check ran
#: first".  A P-1 that swept the rest of that run answers "does not take
#: ['tol']" one phase earlier, which is the exact inversion this table exists
#: to prevent.  Measured: the per-key form turns that test red and nothing in
#: this task's own module notices.
#:
#: A skip, not a rewording: the key is still refused, by the executor, with the
#: message the executor wrote.  What P-1 gives up is only earliness, and only
#: for a run carrying one of these five keys.
#: ``test_the_spoken_for_table_is_every_executor_that_raises_first`` derives
#: this table from the executors' own source and is what makes a sixth such
#: key a red test rather than a re-broken message.
_TASK3_SPOKEN_FOR: dict[str, frozenset[str]] = {
    "condition": frozenset({"prior_mean"}),
    "npe": frozenset({"seed"}),
    "plan.estimate": frozenset({"seed"}),
    "plan.sample": frozenset({"seed"}),
    "predict": frozenset({"from"}),
}


def _task3_allowed_run_options() -> dict[str, frozenset[str]]:
    """run kind -> the option keys its executor accepts.

    Thirteen of the sixteen entries ARE the executor module's own object --
    identity, not equality -- so for those drift is not possible, only
    deletion.  ``forward``, ``fisher`` and ``npe`` write their allowed set as a
    literal at the ``_sweep`` call site and have no name to import; those three
    are restated, and ``test_the_table_is_the_executors_own_allowed_sets``
    reads all sixteen back out of the executors' source and compares.
    """
    from rheplicant.config.sections.conjugate import (
        _CONDITION_KEYS,
        _GCR_KEYS,
        _GLS_KEYS,
        _WIENER_KEYS,
    )
    from rheplicant.config.sections.diagnostics import (
        _GRADIENT_KEYS,
        _IDENTIFIABILITY_KEYS,
        _MMODES_KEYS,
        _PREDICT_KEYS,
        _SCORE_KEYS,
    )
    from rheplicant.config.sections.exits import (
        _ESTIMATE_KEYS,
        _OPTIMIZE_KEYS,
        _SAMPLE_KEYS,
    )
    from rheplicant.config.sections.nuts import _NUTS_KEYS

    return {
        "forward": frozenset(),
        "fisher": frozenset({"space", "jitter"}),
        "npe": frozenset(),
        "optimize": _OPTIMIZE_KEYS,
        "plan.estimate": _ESTIMATE_KEYS,
        "plan.sample": _SAMPLE_KEYS,
        "conjugate.wiener": _WIENER_KEYS,
        "conjugate.gcr": _GCR_KEYS,
        "conjugate.gls": _GLS_KEYS,
        "condition": _CONDITION_KEYS,
        "identifiability": _IDENTIFIABILITY_KEYS,
        "score_directions": _SCORE_KEYS,
        "gradient": _GRADIENT_KEYS,
        "mmodes": _MMODES_KEYS,
        "predict": _PREDICT_KEYS,
        "nuts": _NUTS_KEYS,
    }


def _task3_run_options_in(layer) -> Iterable[Finding]:
    """The executor's own sweep, run one phase early, on one layer.

    A ``runs:`` section ``parse_runs`` cannot read yields nothing here: an
    unknown ``kind:`` is ``parse_runs``' refusal and ``run_document``'s to
    report, and a second voice for one typo is worse than a late one.
    """
    from rheplicant.config.sections.exit_support import _sweep
    from rheplicant.config.sections.runs import parse_runs

    allowed = _task3_allowed_run_options()
    try:
        runs = parse_runs(layer.get("runs"))
    except ConfigError:
        return
    for index, run in enumerate(runs):
        # Unreachable, and deliberately kept: `set(_KINDS) == set(EXECUTORS)`
        # is pinned both ways at `tests/config/test_config_exit_support.py:56`
        # and `:60`, and `parse_runs` refuses a kind outside `_KINDS`, so this
        # never fires today.  It is here so that a kind added to one table and
        # not to `_task3_allowed_run_options` is an unswept run rather than a
        # KeyError, which would abort the whole pass and hide every finding
        # after it (§2.3's TRAP).  Deleting it survives the suite; that is
        # what defensive means, not a defect to re-litigate.
        if run.kind not in allowed:
            continue
        if _TASK3_SPOKEN_FOR.get(run.kind, frozenset()) & set(run.options):
            continue
        try:
            _sweep(run, allowed[run.kind])
        except ConfigError as exc:
            yield refuse(
                "A1", f"runs[{index}]",
                f"{exc} This sweep is the executor's own, moved in front of "
                "the build: it used to run inside execute_run, so an "
                "unrecognised key on a later run cost every earlier run's "
                "execution first (check A1).")


@register("A1.runs")
def _run_option_keys(document) -> Iterable[Finding]:
    """A1: an option key no executor takes, before any run has executed."""
    return _task3_run_options_in(document)


def _task3_horizon_in(layer) -> Iterable[Finding]:
    """A1: a horizon angle that is not a plain number, on one layer."""
    resources = layer.get("resources")
    beams = resources.get("beams") if isinstance(resources, Mapping) else None
    if not isinstance(beams, Mapping):
        return
    for name, spec in beams.items():
        if not isinstance(spec, Mapping):
            continue
        horizon = spec.get("horizon")
        if not isinstance(horizon, Mapping):
            continue
        for key in ("el_deg", "apod_deg"):
            if key not in horizon:
                continue
            value = horizon[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                where = f"resources.beams.{name}.horizon.{key}"
                yield refuse(
                    "A1", longest_legal_prefix(where),
                    f"{where}: is a plain number of degrees; got {value!r}. "
                    "kinds/beams.py hands both horizon angles straight to "
                    "float(), so the value grammar never reaches here and a "
                    "value node arrives as a bare TypeError from inside the "
                    "build -- or, under horizon.mode other than truncate_map, "
                    "is never read at all (check A1).")


@register("A1.horizon")
def _task3_horizon_numbers(document) -> Iterable[Finding]:
    """A1: ``resources.beams.<name>.horizon``'s two angles, over every layer."""
    return _task3_horizon_in(document)


@register("A1.variants")
def _variant_text(document) -> Iterable[Finding]:
    """Turn this already-enumerated layer's structural error into A1.

    The pass driver calls this only after ``_structural`` rejected a variant
    layer.  Enumeration, origin tracking, and merge failure remain outside the
    check registry, so this function never applies or replays a variant.
    """
    from rheplicant.config.preflight import _structural

    try:
        _structural(document)
    except ConfigError as exc:
        if "campaign" in document:
            where = "campaign"
        else:
            allowed = {
                "schema_version", "runtime", "observation", "resources",
                "model", "inference", "runs", "campaign",
            }
            unknown = next((key for key in document if key not in allowed), None)
            if isinstance(unknown, str):
                # The raw key may not itself have a legal prefix.  The pass
                # driver attributes this variant-only finding first, then
                # cuts the complete ``variants.<name>.<key>`` path once.
                where = unknown
            else:
                missing = next(
                    (
                        section
                        for section in ("runtime", "observation", "model", "runs")
                        if section not in document
                    ),
                    "schema_version",
                )
                where = missing
        yield refuse(
            "A1",
            where,
            f"{exc} That is what selecting this variant would raise. An "
            "unselected variant is validated with every other layer, so the "
            "refusal is reported before any build (check A1).",
        )
        return


def _task3_targets(where: str, into: Any) -> tuple[str, ...]:
    """``into:``'s targets, counted the way ``parameters.py:114`` counts them.

    ``_names`` collapses ``"a"`` and ``["a"]`` to the same ``("a",)``, which is
    why §4.7.2's "req iff ``into`` is a list" is unimplementable without
    changing it and §2.6 item 5 decides for §2.2's "more than one entry".
    A malformed ``into:`` is ``parse_latents``' refusal, not this one's.
    """
    from rheplicant.config.sections.parameters import _names

    try:
        return _names(where, into, "into:")
    except ConfigError:
        return ()


def _task3_fan_one(where: str, spec: Mapping) -> Iterable[Finding]:
    """A38: one latent or one binding, in either spelling.

    The transform clause is load-bearing and is exactly the predicate
    ``sections/inference.py:177-178`` already uses: ``parse_transform``
    returns a non-``None`` canonical fan for every form except ``None`` and
    ``"identity"``, and ``_merged_fan(None, canonical)`` returns it -- so with
    a transform there is no guess left to refuse, and a literal
    ``len(into) > 1`` would refuse ``split_rows``, whose whole purpose is two
    targets.
    """
    targets = _task3_targets(where, spec.get("into"))
    if len(targets) < 2 or spec.get("fan") is not None:
        return
    transform = spec.get("transform")
    if transform is not None and transform != "identity":
        return
    yield refuse(
        "A38", longest_legal_prefix(where),
        f"{where}: into: names {len(targets)} targets {list(targets)} and "
        "fan: is absent. broadcast writes one produced value into every "
        "target and distribute writes the k-th into the k-th, and with fan: "
        "absent the only thing that decides is whether what the binding "
        "produced is a JAX array or a Python container -- measured on two "
        "scalar leaves, the same [2, 5] gives 4.0 one way and 10.0 the other "
        "(inference/parameters.py:299-303). Write fan: broadcast or fan: "
        "distribute (check A38).")


def _task3_fan_in(layer) -> Iterable[Finding]:
    """A38 over both spellings on one layer.

    ``transforms.py`` calls ``_merged_fan`` from TWO loops (``:356``, the
    ``parameters.into`` sugar, and ``:395``, the ``bindings[]`` longhand), so
    a check written over ``inference.parameters`` alone closes one route and
    leaves its twin open -- measured, the longhand builds too.
    """
    inference = layer.get("inference")
    if not isinstance(inference, Mapping):
        return
    parameters = inference.get("parameters")
    if isinstance(parameters, Mapping):
        for name, spec in parameters.items():
            if isinstance(spec, Mapping):
                yield from _task3_fan_one(f"inference.parameters.{name}", spec)
    bindings = inference.get("bindings")
    if isinstance(bindings, (list, tuple)):
        for index, entry in enumerate(bindings):
            if isinstance(entry, Mapping):
                yield from _task3_fan_one(f"inference.bindings[{index}]", entry)


@register("A38")
def _fan_present(document) -> Iterable[Finding]:
    """A38: ``fan:`` is required when ``into:`` names more than one target."""
    return _task3_fan_in(document)


def _task3_capability(where: str, key: str, got: str = "") -> Finding:
    """A39's one sentence: name the capability and the schema section.

    §2.3 designates these four re-voiced rather than moved, and it is the only
    such instance in Plan 3A: measured, a document declaring a capability-3/4
    key is told either "``inference:`` does not take ['transitions']", which
    reads as a typo, or "``outputs:`` is not read by this layer yet -- it
    arrives with Plan 4", which names the wrong reason entirely.  A39's whole
    content is *name the capability*.  The section that owns each key keeps
    its own message on its own path -- this module edits no section -- and
    ``test_the_sections_own_refusal_agrees_with_the_table`` holds the two to
    the same capability and the same schema section.
    """
    capability, section = _CAPABILITY_KEYS[key]
    return refuse("A39", longest_legal_prefix(where),
                  f"{where}: {got}is reserved with {capability}, schema "
                  f"{section}, and refused in v1 (check A39).")


def _task3_capability_in(layer) -> Iterable[Finding]:
    """A39 over one layer: the six capability keys a v1 document can reach."""
    inference = layer.get("inference")
    if isinstance(inference, Mapping):
        if "transitions" in inference:
            yield _task3_capability("inference.transitions",
                                    "inference.transitions")
        parameters = inference.get("parameters")
        if isinstance(parameters, Mapping):
            for name, spec in parameters.items():
                if not isinstance(spec, Mapping):
                    continue
                for key in ("support", "hyper"):
                    if key in spec:
                        yield _task3_capability(
                            f"inference.parameters.{name}.{key}",
                            f"inference.parameters.<name>.{key}")
                scope = spec.get("scope")
                if scope in _TASK3_SCOPES_RESERVED:
                    yield _task3_capability(
                        f"inference.parameters.{name}.scope",
                        "inference.parameters.<name>.scope", got=f"{scope!r} ")
    model = layer.get("model")
    if isinstance(model, Mapping):
        for node, spec in model.items():
            if isinstance(spec, Mapping) and spec.get("type") == "NeuralOperator":
                yield _task3_capability(f"model.{node}.type",
                                        "model.<node>.type: NeuralOperator",
                                        got="NeuralOperator ")


@register("A39")
def _capability_keys(document) -> Iterable[Finding]:
    """A39: every capability-3/4 key refused by its capability, not by luck."""
    return _task3_capability_in(document)
