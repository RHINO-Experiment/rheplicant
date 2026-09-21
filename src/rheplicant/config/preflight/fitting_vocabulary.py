"""The words these checks share, and the walks that produce them.

The engine names, the noise kinds, the block keys, and the walks that answer
"which runs", "which latents", "which kinds". The ruling on this file was that
the checks needing `runs:` and `inference:` together are "a join rather than a
subject"; this module is the join, stated once, and the passes beside it are
the subjects.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: ``engines.CONJUGATE``, ``engines.LOG_CONJUGATE`` and ``engines.GRADIENT``,
#: written out because this module may not import that package -- see the
#: module docstring for the guard that measures it.
_T7_CONJUGATE: str = "conjugate"

_T7_LOG_CONJUGATE: str = "log_conjugate"

_T7_GRADIENT: str = "gradient"

#: The engines a block may ask for, closed.  ``_BLOCK_KEYS``
#: (``exits.py::_parse_optimize``) accepts ANY string for ``engine:``, so today
#: ``engine: banana`` reaches the user as a ParameterSpaceError from
#: ``Block._check`` (``plan.py``) -- measured.
#:
#: A copy of ``ENGINES``, which is the thing this plan warns against: a closed
#: set written out stops being closed the day a third engine ships, and this
#: pass would then refuse a block the package accepts.  What stops that is not
#: the import -- which the guard above forbids -- but
#: ``test_the_engine_enum_is_the_packages_own``, which imports the engine names
#: and ``ENGINES`` in the TEST and asserts each against these names.
#:
#: That is not a hypothetical any more: a third engine DID ship
#: (``log_conjugate``, for a block that is conjugate once the data is taken to
#: logs), the test went red on the copy, and the copy was extended here.  The
#: guard is the reason that was a red test rather than a config layer quietly
#: refusing a partition ``auto_blocks`` had just produced.
_ENGINES: frozenset[str] = frozenset({_T7_CONJUGATE, _T7_LOG_CONJUGATE, _T7_GRADIENT})

#: The ``log_route_refusal`` reasons (``inference/loglinear.py``) that a
#: noise's TEXT can decide, spelled as ``LOG_ROUTE_REFUSALS`` spells them.
#: Written out because this module may not import that package; what keeps
#: the copy honest is ``tests/config/test_preflight_log_route.py``, which
#: builds the package's noise model for each row of a table and compares
#: :func:`_log_route_refusal_text` with ``log_route_refusal``, and checks each
#: word against ``LOG_ROUTE_REFUSALS``. ``noise_neither`` is the verdict T-002
#: G4 (A5-3) gave a ``RadiometerNoise`` with a declared floor.
#:
#: The package's third reason, ``fractional_too_large``, is NOT here: ``f =
#: 1 / sqrt(channel_width * integration_time)`` needs both values resolved,
#: and their default ``{from: observation}`` reads the observation's grid.
#: That document is still refused, by ``to_log_space`` at P3.
_T7_NOISE_ADDITIVE: str = "noise_additive"

_T7_NOISE_NEITHER: str = "noise_neither"

_LOG_ROUTE_REASONS: frozenset[str] = frozenset({_T7_NOISE_ADDITIVE, _T7_NOISE_NEITHER})

#: ``inference.noise.kind`` values whose sigma does not scale with the
#: prediction. ``homoscedastic`` is a ``HomoscedasticNoise``; a
#: ``radiometer_frozen`` sigma is DECIDED into an array once
#: (``sections/noise.py::freeze_sigma``) and a plan wraps a bare sigma as a
#: ``HomoscedasticNoise`` -- additive either way, and ``log_route_refusal``
#: says ``noise_additive`` for both.
_T7_ADDITIVE_KINDS: frozenset[str] = frozenset({"homoscedastic", "radiometer_frozen"})

#: The keys a block entry takes -- ``_BLOCK_KEYS`` (``exits.py::_parse_optimize``), copied
#: for the same reason :data:`_ENGINES` is: reaching it means importing
#: ``sections/exits``, which foot-imports ``conjugate``, ``diagnostics``,
#: ``npe`` and ``nuts``, and measured that adds ~30 ms to every ``import
#: rheplicant.config`` for one frozenset.  ``test_the_block_key_set_is_the_
#: packages_own`` imports ``exits._BLOCK_KEYS`` in the TEST and pins it, so a
#: fifth block key turns that red rather than leaving this check reading an
#: entry the grammar rejects.
_T7_BLOCK_KEYS: frozenset[str] = frozenset({"names", "steps", "engine", "learning_rate"})


def _latents(document: Mapping[str, Any]) -> dict[str, Any]:
    """``inference.parameters`` as the document writes it, or ``{}``.

    The KEYS are the space's latent names, in declaration order -- measured
    through ``load_document``: ``tuple(document["inference"]["parameters"]) ==
    tuple(built.inference.space.names)``.  ``fan:`` and ``transform:``
    describe the BINDING (``transforms.py::parse_transform`` appends one ``Bind``
    per entry and creates no latent) and ``hyper:`` is refused as capability 4
    (``sections/parameters.py::_names``), so nothing in v1 splits one
    declaration into two latents.  A16 rests entirely on that and
    ``test_the_space_names_are_the_declared_parameter_keys`` is what keeps it
    true.

    A latent whose body is not a mapping keeps its NAME and reads as ``{}``.
    Dropping it would make it look undeclared to A16, which would refuse a
    block that names it -- a wrong refusal caused by a malformed neighbour.
    ``sections/parameters.py::_names`` refuses the malformed body itself, at
    P2.
    """
    inference = document.get("inference")
    if not isinstance(inference, Mapping):
        return {}
    parameters = inference.get("parameters")
    if not isinstance(parameters, Mapping):
        return {}
    return {
        name: (body if isinstance(body, Mapping) else {})
        for name, body in parameters.items()
        if isinstance(name, str)
    }


def _runs(document: Mapping[str, Any]) -> tuple[dict, ...]:
    """``runs:`` as a tuple of mappings, INDEX-PRESERVING, with ``name`` filled.

    Three things this does, each of which a later task would otherwise redo:

    * a single mapping is wrapped in a list, the way ``parse_runs``
      (``runs.py::_one``) does, so ``runs: {kind: forward}`` is one run here
      too;
    * a malformed entry becomes ``{}`` rather than being dropped, so on the
      LIST form ``_runs(document)[i]`` is always ``runs[i]`` of the document
      and a ``Finding.where`` of ``runs[2]`` points where the user must type.
      Such an entry carries NO ``name`` -- read it with ``.get`` or behind a
      test on ``kind``;

      **On the single-mapping form the index is this function's, not the
      document's**, and the contract said otherwise until a reviewer read the
      two clauses together: ``runs: {kind: plan.sample, ...}`` yields findings
      at ``runs[0]``, which is not a path that document contains at all.  The
      READER is right and the sentence was wrong, so the sentence is what
      changed: dropping the wrap would take every fitting check off a document
      the package runs -- ``parse_runs`` wraps the same way -- to buy a
      ``where`` that is one index more literal.  ``runs[0]`` still names the
      only run there is, and the message itself opens ``runs['<name>']:``,
      which is what a reader actually navigates by.
      ``test_a_single_mapping_run_is_found_at_index_zero`` pins the shipped
      behaviour so the next reader inherits a decision rather than a
      discovery;
    * ``name`` is filled in by ``parse_runs``' own rule -- the entry's
      ``name`` when it has one, else the kind (``runs.py::_one``) -- because
      every refusal in this layer is prefixed ``runs['<name>']:`` and three
      tasks would otherwise each re-derive it.  A NEW dict is built; the
      caller's document is never mutated.

      **``runs.py::_one``'s test is ``is not None``, not "is a string"**, and
      the difference is one document: a non-string ``name:`` is REFUSED at
      ``runs.py::_one`` (*"name: is a string; got 7"*) rather than
      defaulted, so ``parse_runs`` has no such run to name.  This function
      does fill it -- ``name: 7`` is prefixed ``runs['<kind>']`` -- because
      ``parse_runs`` runs only on the ``run_document`` path (``runs.py::parse_runs``)
      and a ``load_document`` caller would otherwise get a check that
      declined on a document nothing else refuses.  The cost is a prefix
      naming a run the user did not write, on a document already refused for
      its ``name:``; recorded rather than repaired, because repairing it is
      a stand-down that loses the block checks on that document entirely.
    """
    section = document.get("runs")
    if isinstance(section, Mapping):
        section = [section]
    if not isinstance(section, list):
        return ()
    filled: list[dict] = []
    for entry in section:
        if not isinstance(entry, Mapping):
            filled.append({})
            continue
        kind = entry.get("kind")
        name = entry.get("name")
        if not isinstance(name, str):
            name = kind if isinstance(kind, str) else ""
        filled.append({**entry, "name": name})
    return tuple(filled)


def _kinds(document: Mapping[str, Any]) -> frozenset[str]:
    """Every ``runs[].kind`` the document declares, as a set.

    **It filters nothing, and the callers this docstring used to name do not
    exist.**  It said "A20's ``plan.*`` family, A21's ``fisher``, A30's 'is a
    fitting exit declared at all'"; measured, ``_prior_gates`` walks
    :func:`_runs` for A20 and A21 -- they need the run's index for their
    ``where`` -- so A30 was the only production caller, and it inherited two
    holes this function is entitled to have and A30 was not:

    * a kind ``sections/runs._KINDS`` does not contain is in here.  ``kind:
      banana`` is a member of this set, and A30 read it as a declared exit
      and made a claim about how it closes the fit twin;
    * a run declaring ``expect: refuse`` is in here.  That run is an
      assertion ABOUT a refusal, and a P-1 refusal cannot be captured
      (``_blocks`` and ``_prior_gates`` argue it at length), so a check that
      refuses on the mere PRESENCE of the kind destroys it.

    Neither is a defect here: the question this answers is "which kinds does
    the text declare", and both of those are declared.  They are defects in a
    caller that reads the answer as "which exits will this document reach",
    and the one caller that did now narrows it itself --
    ``preflight/model._a30_exits``, which intersects this with the closed
    enum and with the runs that are not expectations.  A check that needs the
    run's own index, options or ``expect:`` walks :func:`_runs` instead.
    """
    return frozenset(run["kind"] for run in _runs(document) if isinstance(run.get("kind"), str))


def _t7_names(entry: Mapping[str, Any]) -> tuple[str, ...] | None:
    """``names:`` when the grammar accepts it, and ``None`` when it does not.

    ``exits.py::_parse_optimize``'s own three tests -- a list, non-empty, all strings
    -- because everything else is that refusal's, in its own words: *"blocks[0]
    .names is a non-empty list of latent names"*.

    ``None`` and not ``()``, and the difference is the point.  The brief's
    spelling was ``for name in (entry.get("names") or ())``, which reads
    ``names: "d"`` as the one-name list ``['d']`` (a string is iterable) and
    **raises TypeError** on ``names: 5`` -- and inside the pass a TypeError
    becomes "check A16 RAISED" and discards every other finding in the
    report.
    """
    names = entry.get("names")
    if not isinstance(names, list) or not names:
        return None
    if not all(isinstance(name, str) for name in names):
        return None
    return tuple(names)


def _t7_entries(node: Any) -> tuple[Mapping[str, Any], ...] | None:
    """A ``blocks:`` list this check may read, or ``None`` to stand down.

    ``exits._blocks`` (``exits.py::_parse_optimize``) refuses **five** shapes before a
    ``Block`` is ever built, and all five are mirrored here: a ``blocks:``
    that is not a list and an empty one (``exits.py::_parse_optimize``), an entry that is not a
    mapping (``exits.py::_parse_optimize``), an entry carrying a key a block does not take
    (``exits.py::_parse_optimize``), and a malformed ``names:`` (``exits.py::_parse_optimize``).
    Each has a
    sentence naming the fault; a partition answer in front of one would say
    *"blocks: does not cover ['d', 'a', 'w']"* -- true, useless, and offering
    a fix ("add it to a block") that is not the fault.

    The unknown-key one was missed in the first draft of this function, whose
    docstring said "four shapes": measured, ``blocks: [{names: [d], step: 5}]``
    passed the four and reached the engine derivation, so a document whose
    real fault is a typo'd ``step:`` was answered with a coverage sentence.

    **All or nothing per list.**  One malformed entry makes the whole
    partition undecidable: the names it would have owned are unknown, so
    every OTHER entry's coverage answer would be computed against a set that
    is missing them.
    """
    if not isinstance(node, list) or not node:
        return None
    for entry in node:
        if not isinstance(entry, Mapping):
            return None
        if set(entry) - _T7_BLOCK_KEYS:
            return None
        if _t7_names(entry) is None:
            return None
    return tuple(node)
