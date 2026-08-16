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

import dataclasses
from collections.abc import Iterable, Mapping
from typing import Any

from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import Finding, refuse
from rheplicant.config.paths import parse_path
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


def _task3_where(label: str) -> str:
    """``label``, cut back to the longest prefix the path grammar can spell.

    A name the user chose is not required to be an identifier: ``variants:
    {unity-gain: ...}`` and ``parameters: {d-1: ...}`` both load today,
    measured, and neither ``apply_variant`` nor ``parse_latents`` validates a
    name.  ``Finding.where`` is held to ``config/paths.py``'s segment grammar
    by ``preflight._check_where``, which raises OUTSIDE the per-check ``try``
    -- so an un-spellable segment would kill the whole pass rather than report
    the violation.  The full path stays in the MESSAGE, which is what the
    reader is shown.
    """
    pieces = label.split(".")
    for stop in range(len(pieces), 1, -1):
        candidate = ".".join(pieces[:stop])
        try:
            parse_path(candidate)
        except ConfigError:
            continue
        return candidate
    return pieces[0]


#: The ONE-ENTRY memo behind :func:`_task3_layers`: ``(document, layers)`` for
#: the document most recently walked, or ``None`` before the first walk.
#:
#: **The document itself is held, not only its ``id()``, and that is a
#: correctness requirement rather than a convenience.**  ``id()`` in CPython is
#: a memory address, and an address is reused the moment its object is freed --
#: so a memo keyed on the bare integer would answer a hit for a DIFFERENT
#: document that happened to land where the first one used to be, and hand it
#: another document's layers.  Holding the reference is what makes the address
#: un-reusable for as long as the entry lives, and the hit path below asserts
#: ``is`` rather than ``==``: two equal documents are still two documents, and
#: either may be mutated after the other was walked.
#:
#: **What a live entry costs, stated as the measurement and not as an
#: impression.**  It is not "one document": it is the document PLUS one deep
#: merge of it per declared variant -- 42 340 bytes of document and 777 475
#: bytes of entry on the cold guard's own 21-variant row.  That is why
#: :func:`preflight` drops it on the way out as well as on the way in: a
#: process that loads one config would otherwise hold all of it for the rest of
#: its life, for a cache nothing will read again.  Inside the pass it is paid
#: once rather than once per check, which is the whole point.
#:
#: **No test can show this memo returning another document's layers if the
#: reference is dropped, and the honest reason is written here rather than
#: implied by a test that does not exist.**  Layer 0 IS ``("", document)``, so
#: the layers tuple holds the document too, and a single entry can only hold
#: one of them -- an id-only key would therefore be un-recyclable today by
#: accident.  It is an accident: a walk that ever handed out a COPY as layer 0,
#: or a memo that ever held more than one entry, loses it, and the failure
#: would be a silent wrong answer rather than an exception.  The reference is
#: explicit so that the guarantee stops depending on the shape of the value.
_TASK3_LAYER_MEMO: tuple[Mapping[str, Any], tuple[tuple[str, Mapping], ...]] | None = None


def _task3_forget_layers() -> None:
    """Drop the memo.  ``preflight`` calls this at the head AND the tail of
    every pass -- the head for correctness, the tail so nothing is retained.

    **Identity alone is not a safe key ACROSS passes, and that is measured
    rather than argued.**  Within one pass a document cannot change: every
    check reads text and layer 0 IS the caller's own document object, shared
    by every layering caller since the walk was written.  BETWEEN two passes
    it changes constantly -- ``tests/config/test_preflight_instrument.py:970``
    earns A13, writes the remedy the message advises straight into the
    document it already passed to :func:`preflight`, and requires the next pass
    to see it.  That is R4's "apply your own advice" shape, the suite is full
    of it, and a memo that outlived the pass would answer those documents from
    before their own fix -- silently, since layer 0 would show the mutation and
    only the merged variants would be stale.

    So the memo's lifetime is ONE pass, and the assumption it rests on is only
    the one the walk already rested on.

    The tail drop is a second reason rather than a second mechanism: the entry
    is the document plus one merge of it per variant, and a library that leaves
    that behind after answering a question is holding a cache with no reader.
    It also keeps ``inflight``'s two passes out of the question entirely --
    ``_assemble`` runs :func:`preflight` and then ``axes`` on the SAME document
    object, so an entry that outlived the pass would be inherited by the first
    axis check that ever walks layers.
    """
    global _TASK3_LAYER_MEMO

    _TASK3_LAYER_MEMO = None


def _task3_build_layers(
        document: Mapping[str, Any]) -> tuple[tuple[str, Mapping], ...]:
    """:func:`_task3_layers` without the memo -- the walk itself.

    Split out so that the memo is one readable branch rather than a flag
    threaded through the build, and so a test can drive the uncached walk
    directly.
    """
    from rheplicant.config.layering import apply_variant

    layers: list[tuple[str, Mapping]] = [("", document)]
    variants = document.get("variants")
    if not isinstance(variants, Mapping):
        return tuple(layers)
    for name in variants:
        try:
            layers.append((f"variants.{name}", apply_variant(document, name)))
        except ConfigError:
            continue
    return tuple(layers)


def _task3_layers(document: Mapping[str, Any]) -> tuple[tuple[str, Mapping], ...]:
    """``("", document)`` plus one ``("variants.<name>", merged)`` per variant.

    Layering is one level deep by design (``layering.py:9-12``), so the walk
    is one layer per DECLARED variant and never a pair of them merged.

    A variant ``apply_variant`` refuses is dropped rather than raised on:
    ``_variant_text`` is the check that reports it, and a walk that let the
    ``ConfigError`` out would abort the pass and hide every later finding
    (§2.3's TRAP).

    **Built ONCE per document and handed to every caller, because
    ``apply_variant`` is a deep merge of the whole document and every check
    that layers used to pay for its own copy of it.**  Measured on the cold
    guard's own child (40 ``plan.sample`` runs, 21 declared variants) at the
    wave-1 tip: **ten of the eleven merge sites ran** -- ``noise``'s walk is
    gated off on that document -- for 210 ``apply_variant`` calls and 45 ms
    against a 50 ms budget the pass then breached 3 runs in 5 under
    ``-n 16``.  With the
    memo it is 21 calls, one per declared variant, and the number no longer
    moves when a check that layers is added.

    **The assumption, stated rather than left implicit: a document is not
    mutated between two checks of one pass.**  That is already what the layer
    walk rests on -- layer 0 IS the caller's own document object, shared by
    every check since the walk was written -- and the memo extends the same
    assumption to the merged layers.  A document mutated between two calls
    WITHIN one pass gets the layers built for the first, and
    ``test_a_mutation_inside_one_pass_is_not_seen_by_the_memo`` pins that as
    documented behaviour rather than leaving it to be discovered.  Across
    passes there is no such limit: :func:`_task3_forget_layers` drops the entry
    at the head of every pass, for the reason written there.

    The layers are a ``tuple`` of ``(prefix, Mapping)`` and callers only read
    them, so nothing is copied on the way out.

    Reading the memo into a local before testing it is deliberate: the entry is
    an immutable tuple, so a concurrent pass on another document can cost this
    one a hit but can never hand it half of someone else's entry.
    """
    global _TASK3_LAYER_MEMO

    memo = _TASK3_LAYER_MEMO
    if memo is not None and memo[0] is document:
        return memo[1]
    layers = _task3_build_layers(document)
    _TASK3_LAYER_MEMO = (document, layers)
    return layers


def _task3_over_layers(document, per_layer) -> Iterable[Finding]:
    """``per_layer`` over every layer, with the base's own findings said once.

    A document with one A38 error and four variants would otherwise hand the
    user the same sentence five times, four of them blaming a variant that did
    not introduce it.

    **The key is the whole ``Finding``, not its ``where``.**  A variant that
    breaks the same rule DIFFERENTLY -- the base binds two targets, the
    variant rebinds the same latent to three -- has the same ``where`` and is
    a second violation; keying on ``where`` swallows it, which is a lost check
    rather than a duplicated sentence.
    ``test_a_variant_that_breaks_the_rule_DIFFERENTLY_is_still_reported``
    is that assertion.

    Keying on ``finding.message`` instead is, measured, an EQUIVALENT mutation
    TODAY: every message this module emits opens with the label its ``where``
    is derived from, so the message determines the finding (probed over 14
    findings on three documents built to collide -- two beams with the same
    two bad angles, two latents with the same two targets, two bindings alike
    -- zero collisions between unequal findings).  It is not equivalent in
    general, and the whole-``Finding`` key is what keeps it safe for a later
    check whose sentence does not carry its own path.
    """
    base: set[Finding] = set()
    for prefix, layer in _task3_layers(document):
        for finding in per_layer(layer):
            if not prefix:
                base.add(finding)
                yield finding
            elif finding not in base:
                yield dataclasses.replace(
                    finding,
                    where=_task3_where(f"{prefix}.{finding.where}"),
                    message=f"{prefix}: {finding.message}")


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
    return _task3_over_layers(document, _task3_run_options_in)


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
                    "A1", _task3_where(where),
                    f"{where}: is a plain number of degrees; got {value!r}. "
                    "kinds/beams.py hands both horizon angles straight to "
                    "float(), so the value grammar never reaches here and a "
                    "value node arrives as a bare TypeError from inside the "
                    "build -- or, under horizon.mode other than truncate_map, "
                    "is never read at all (check A1).")


@register("A1.horizon")
def _task3_horizon_numbers(document) -> Iterable[Finding]:
    """A1: ``resources.beams.<name>.horizon``'s two angles, over every layer."""
    return _task3_over_layers(document, _task3_horizon_in)


@register("A1.variants")
def _variant_text(document) -> Iterable[Finding]:
    """A1: the document-grammar refusals an unselected variant never earns.

    Each declared variant is merged with the layer's own ``apply_variant`` and
    the merge is handed to ``_structural``; the raised ``ConfigError`` becomes
    a finding.  It does NOT re-enter the registry per variant: P-1 is defined
    (§2.1) over one variant-applied document, a registered check that called
    ``preflight`` would walk over itself, and every base-document finding
    would be reported once per layer.  The model interior of an unselected
    variant therefore stays open here, and §6 records it.

    **The merge comes from :func:`_task3_layers`, which is the same merge.**
    This check used to call ``apply_variant`` itself, once per declared
    variant, on top of the once per variant every layering check was already
    paying -- so it was the tenth caller of a walk that had one answer.  A
    variant the merge REFUSES is not in the layers (``_task3_layers`` drops
    it), and that is exactly the variant this check exists to report: for
    those, and only those, ``apply_variant`` is called here to re-raise the
    ``ConfigError`` whose text becomes the finding.

    **A name that is not a string misses the lookup and re-merges, recorded
    rather than left to be discovered.**  ``merged`` is keyed by the layer
    prefix with ``variants.`` cut off, which is always a ``str``, while
    ``variants:`` may carry any YAML scalar as a key -- ``{1: {...}}`` is legal
    and loads.  Such a name falls through to the ``apply_variant`` branch and
    pays for one extra merge of its own layer, which is the pre-memo cost for
    that one variant.  The ANSWER is identical either way, because that branch
    is exactly the code this check used to run; only the cost differs, only on
    a document nobody has yet written.
    """
    from rheplicant.config.layering import apply_variant
    from rheplicant.config.preflight import _structural

    variants = document.get("variants")
    if variants is None:
        return
    if not isinstance(variants, Mapping):
        yield refuse(
            "A1", "variants",
            "variants: is a mapping of name -> patch; got "
            f"{type(variants).__name__} ({variants!r}). Nothing reads it "
            "until a variant is selected, so today this document loads and "
            "the error waits for the run that asks for one (check A1).")
        return
    merged = {prefix.removeprefix("variants."): layer
              for prefix, layer in _task3_layers(document) if prefix}
    for name in variants:
        try:
            _structural(merged[name] if name in merged
                        else apply_variant(document, name))
        except ConfigError as exc:
            yield refuse(
                "A1", _task3_where(f"variants.{name}"),
                f"variants.{name}: {exc} That is what selecting this variant "
                "would raise. An unselected variant is merged only when "
                "someone asks for it, so until now this document loaded and "
                "the refusal waited for the run that selects it (check A1).")


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
        "A38", _task3_where(where),
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
    return _task3_over_layers(document, _task3_fan_in)


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
    return refuse("A39", _task3_where(where),
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
    return _task3_over_layers(document, _task3_capability_in)
