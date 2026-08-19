"""What every exit executor shares: the sweep, the accessors, the registry.

An exit handler is FOUR bindings under one ``runs[].kind``: a ``parse`` that
freezes the entry's options into a :class:`ParsedOptions`, a ``pre_execute``
for the checks only a finished earlier run can answer, an ``execute`` that
produces the product, and the ordered ``deferred_checks`` a validating
caller reports rather than claims.  :func:`register` binds all four
atomically under one lock, and :func:`handler_for` projects the live tables
on every call, so which handler a kind has never depends on import order or
on a cached copy.

The leaf modules (``exits``, ``conjugate``, ``diagnostics``) import from here
and never from each other, so the registration is a one-way import.

TRANSITIONAL until Tasks 8-9: the eleven conjugate/diagnostics built-ins
still bind with ``parse`` omitted (Task 7 migrated the five base kinds to
explicit parsers).  Their parser is then :func:`_legacy_freeze_parse` -- the
same freeze/YAML-safe snapshot factory explicit parsers call -- and their
executor keeps today's ``(run, built, *, results=None)`` convention behind
an adapter that hands it the raw ``RunSpec``.  A registration WITH ``parse=``
stores its executor unwrapped, and that executor is called
``(parsed_run, configured_run, previous)`` positionally.
"""

from __future__ import annotations

import functools
import threading
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from _rheplicant_bootstrap.frozen import freeze, freeze_evidence
from rheplicant.config.errors import ConfigError
from rheplicant.config.sections.noise import decided_noise

if TYPE_CHECKING:
    from _rheplicant_bootstrap.types import JsonValue, LayerIdentity, TraceSink
    from _rheplicant_bootstrap.variants import LayerRef
    from rheplicant.config.document import ConfiguredRun
    from rheplicant.config.sections.runs import RunResult, RunSpec

__all__ = [
    "DEFERRED_CHECKS",
    "EXECUTORS",
    "PARSERS",
    "PRE_EXECUTORS",
    "ExitHandler",
    "ParsedOptions",
    "ParsedRun",
    "RunParseContext",
    "handler_for",
    "parse_run",
    "parsed_options",
    "register",
    "reuse_of",
]


@dataclass(frozen=True, slots=True)
class ParsedOptions:
    """One run entry's options, frozen twice: what execute reads, and the
    YAML-safe audit projection of it.

    The two views are detached from each other and from the caller's
    mapping: mutating the parsed document afterwards moves neither, and a
    hook that belongs in execution (a callable, a live object) never reaches
    the resolved view a serializer will read.
    """

    execution: Mapping[str, object]
    resolved: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class RunParseContext:
    """The cheapest context a kind's parser may read: position, layer, the
    common ``RunSpec``, and the configured build's static facts."""

    index: int
    layer: LayerRef
    spec: RunSpec
    configured_run: ConfiguredRun

@dataclass(frozen=True, slots=True)
class ParsedRun:
    """A ``RunSpec`` after its kind's parser: the spec, plus the two frozen
    option views the rest of the pipeline is allowed to read."""

    index: int
    layer: LayerRef
    spec: RunSpec
    parsed: ParsedOptions
    declaration_layer: LayerIdentity | None = None

    @property
    def audit_layer(self) -> LayerIdentity:
        """Layer whose schedule declared this run, distinct from its target."""
        return self.layer.identity if self.declaration_layer is None else self.declaration_layer

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def kind(self) -> str:
        return self.spec.kind

    @property
    def variant(self) -> str | None:
        return self.spec.variant

    @property
    def on(self) -> str:
        return self.spec.on

    @property
    def expect(self) -> str:
        return self.spec.expect

    @property
    def reuse(self) -> str | None:
        return self.spec.reuse

    @property
    def options(self) -> Mapping[str, object]:
        """The parsed execution view, never the raw ``spec.options``."""
        return self.parsed.execution


ParseExit = Callable[[Mapping[str, object], RunParseContext], ParsedOptions]
PreExecute = Callable[
    [ParsedRun, "ConfiguredRun", Mapping[str, "RunResult"]], None
]
ExecuteExit = Callable[
    [ParsedRun, "ConfiguredRun", Mapping[str, "RunResult"]], object
]


@dataclass(frozen=True, slots=True)
class ExitHandler:
    """One kind's four bindings, projected live out of the registry."""

    parse: ParseExit
    pre_execute: PreExecute
    execute: ExecuteExit
    deferred_checks: tuple[str, ...]


PARSERS: dict[str, ParseExit] = {}
PRE_EXECUTORS: dict[str, PreExecute] = {}
EXECUTORS: dict[str, ExecuteExit] = {}
DEFERRED_CHECKS: dict[str, tuple[str, ...]] = {}
_HANDLER_LOCK = threading.RLock()


def _require_yaml_safe(view: object, path: str) -> None:
    """Refuse a resolved-view leaf the audit record cannot hold.

    ``freeze_evidence`` has already refused sets, foreign objects, cycles
    and non-string keys by the time this walks the frozen view; what remains
    outside ``JsonValue`` is the binary family, which canonicalizes to
    ``bytes`` and would otherwise reach the resolved-YAML encoder.
    """
    if isinstance(view, Mapping):
        for key, child in view.items():
            _require_yaml_safe(child, f"{path}.{key}")
        return
    if type(view) is tuple:
        for index, child in enumerate(view):
            _require_yaml_safe(child, f"{path}[{index}]")
        return
    if view is None or isinstance(view, (bool, int, float, str)):
        return
    raise ConfigError(
        "runs[].options: the resolved audit view must be YAML-safe; "
        f"{path} is {type(view).__name__}."
    )


def parsed_options(
    execution: Mapping[str, object],
    *,
    resolved: Mapping[str, object],
) -> ParsedOptions:
    """Freeze one run entry's two option views, independently and atomically.

    The execution view uses the permissive recursive freeze: containers
    become read-only and leaf objects pass through, because a later explicit
    parser may legitimately place a hook there.  The resolved view uses the
    strict evidence freeze -- a detached copy with string keys, canonical
    scalars, and the depth/node/edge budgets -- and is then validated as
    YAML-safe BEFORE either view is returned, so a rejected entry hands back
    nothing at all.  There is no trace parameter: this factory never appends
    an event.
    """
    if not isinstance(execution, Mapping):
        raise ConfigError(
            "runs[].options: parsed execution options are a mapping; got "
            f"{type(execution).__name__}."
        )
    if not isinstance(resolved, Mapping):
        raise ConfigError(
            "runs[].options: resolved options are a mapping; got "
            f"{type(resolved).__name__}."
        )
    frozen_resolved = freeze_evidence(
        resolved, where="runs[].options resolved view"
    )
    _require_yaml_safe(frozen_resolved, "resolved")
    frozen_execution = freeze(execution)
    return ParsedOptions(execution=frozen_execution, resolved=frozen_resolved)


def _legacy_freeze_parse(
    options: Mapping[str, object], context: RunParseContext
) -> ParsedOptions:
    """The transitional parser the unmigrated built-ins sit on (Tasks 8-9).

    Both views are the entry's own options, frozen independently -- the same
    factory an explicit parser calls, so migrating a kind changes WHO calls
    it, never how the freeze behaves.
    """
    return parsed_options(options, resolved=options)


def _noop_pre_execute(
    parsed_run: ParsedRun,
    configured_run: ConfiguredRun,
    previous_results: Mapping[str, RunResult],
) -> None:
    """The default: every check this kind has is decidable at parse time."""
    return None


def _legacy_dispatch(
    parsed_run: ParsedRun,
    configured_run: ConfiguredRun,
    previous_results: Mapping[str, RunResult],
    _execute: Callable[..., Any],
) -> object:
    """The transitional legacy calling convention, as a closure-free body.

    Closure-free so that :func:`_adapt_legacy_executor` can rebind this code
    object onto the wrapped executor's OWN ``__globals__``.
    """
    return _execute(parsed_run.spec, configured_run, results=previous_results)


def _adapt_legacy_executor(execute: Callable[..., Any]) -> ExecuteExit:
    """Wrap a ``(run, built, *, results=None)`` executor as an ExecuteExit.

    The wrapper is a real function built from :func:`_legacy_dispatch`'s code
    object but rebound to ``execute``'s OWN ``__globals__``, and
    ``functools.update_wrapper`` then restores ``__module__``/``__name__`` and
    sets ``__wrapped__``.  That is what keeps the wrapped executor
    transparent to the two things that read it in place: the
    duplicate-registration message, which must name the claimant's module,
    and the preflight sweep census
    (``tests/config/test_preflight_document.py``), which parses
    ``inspect.getsource``'s unwrapped source and resolves helper calls
    through ``fn.__globals__`` -- a plain closure would report
    ``exit_support``'s globals and silently empty that census.
    """
    globals_ = getattr(execute, "__globals__", None)
    if type(globals_) is not dict:
        # A callable that is not a function has no module namespace; the
        # body below touches none, so ours is only a label.
        globals_ = globals()
    adapted = types.FunctionType(
        _legacy_dispatch.__code__,
        globals_,
        getattr(execute, "__name__", None) or "_legacy_dispatch",
        (execute,),
    )
    functools.update_wrapper(adapted, execute)
    return adapted


def register(
    kind: str,
    *,
    parse: ParseExit | None = None,
    pre_execute: PreExecute = _noop_pre_execute,
    deferred_checks: tuple[str, ...] = (),
) -> Callable[[ExecuteExit], ExecuteExit]:
    """Bind one exit handler to its ``runs[].kind`` -- atomically, all four.

    Registering the same kind twice is a programming error, not a
    configuration one -- and it is refused with a RAISE rather than the
    ``assert`` this used to carry, because ``python -O`` strips asserts and
    the shadowing is then completely silent.  Measured before the change::

        $ python -O -c "...register('_probe')(one); register('_probe')(two)..."
        under -O, second registration won: True

    -- so under ``-O`` which executor a document got depended on import
    order, with nothing said.  ``ConfigError`` and not ``RuntimeError``,
    following ``errors.py``'s "one refusal type for the whole config layer";
    the message is what carries "this is wiring, not your document", and it
    names both claimants so the second one can be found.

    That rule is stated layer-wide and, measured, is not yet applied
    layer-wide: this remains the only one of five registration decorators in
    ``config/`` to refuse a double registration at all.  The other four --
    ``files.register_reader``, ``values.register_form``,
    ``derive.register_derivation`` and ``resources.register_kind`` -- assign
    unconditionally, so the second registration wins in silence with no
    ``assert`` to strip.  They are outside Plan 3A's scope and are recorded on
    its residue ledger rather than fixed here.

    Every member is validated before ANY table is bound, the four
    assignments run under the one lock, and a failure mid-bind rolls all
    four back: a partial handler -- a parser with no executor, or the
    reverse -- is never observable.  Omitting ``parse`` is the transitional
    legacy shape: the parser is :func:`_legacy_freeze_parse` and the
    executor keeps its old calling convention behind an adapter.
    """

    def bind(execute: ExecuteExit) -> ExecuteExit:
        registries = (PARSERS, PRE_EXECUTORS, EXECUTORS, DEFERRED_CHECKS)
        if not callable(execute) or not callable(pre_execute):
            raise TypeError(
                "exit parser, pre-executor, and executor are callable"
            )
        chosen_parse = _legacy_freeze_parse if parse is None else parse
        if not callable(chosen_parse):
            raise TypeError(
                "exit parser, pre-executor, and executor are callable"
            )
        if isinstance(deferred_checks, (str, bytes)):
            raise ValueError(
                "deferred check names are unique non-empty strings"
            )
        checks = tuple(deferred_checks)
        if len(checks) != len(set(checks)) or not all(
                isinstance(check, str) and check for check in checks):
            raise ValueError(
                "deferred check names are unique non-empty strings"
            )
        stored_execute = (
            execute if parse is not None else _adapt_legacy_executor(execute)
        )
        with _HANDLER_LOCK:
            if any(kind in registry for registry in registries):
                incumbent = EXECUTORS[kind]
                raise ConfigError(
                    f"runs[].kind: {kind!r} is registered twice, by "
                    f"{incumbent.__module__} and by {execute.__module__}. "
                    "A kind has one executor, and which of the two you would "
                    "get depends on import order."
                )
            try:
                PARSERS[kind] = chosen_parse
                PRE_EXECUTORS[kind] = pre_execute
                EXECUTORS[kind] = stored_execute
                DEFERRED_CHECKS[kind] = checks
            except BaseException:
                for registry in registries:
                    registry.pop(kind, None)
                raise
        return execute

    return bind


def handler_for(kind: str) -> ExitHandler:
    """The complete live handler for ``kind``, assembled at call time."""
    with _HANDLER_LOCK:
        try:
            return ExitHandler(PARSERS[kind], PRE_EXECUTORS[kind],
                               EXECUTORS[kind], DEFERRED_CHECKS[kind])
        except KeyError:
            raise ConfigError(
                f"runs[].kind: {kind!r} is not registered; it takes "
                f"{sorted(EXECUTORS)}."
            ) from None


def parse_run(
    spec: RunSpec,
    configured: ConfiguredRun,
    *,
    index: int,
    layer: LayerRef,
    trace: TraceSink | None = None,
    declaration_layer: LayerIdentity | None = None,
) -> ParsedRun:
    """Parse one declared run through its current handler -> a ``ParsedRun``.

    Exactly one parser runs -- the one the registry holds NOW -- and the one
    parsed-run trace projection is appended only after that parser and its
    ``parsed_options`` complete successfully.  The row keys are the closed
    ``("descriptor", "resolved_options", "deferred_checks")`` of plan §1.4's
    ``parsed_run`` record, with the descriptor exactly
    ``{index, name, kind, variant}``.
    """
    handler = handler_for(spec.kind)
    context = RunParseContext(index=index, layer=layer, spec=spec,
                              configured_run=configured)
    parsed = handler.parse(spec.options, context)
    if not isinstance(parsed, ParsedOptions):
        raise TypeError(
            f"exit parser for {spec.kind!r} returned "
            f"{type(parsed).__name__}, not ParsedOptions."
        )
    run = ParsedRun(
        index=index,
        layer=layer,
        spec=spec,
        parsed=parsed,
        declaration_layer=declaration_layer,
    )
    if trace is not None:
        audit_row = {
            "descriptor": {"index": index, "name": spec.name,
                           "kind": spec.kind, "variant": spec.variant},
            "resolved_options": parsed.resolved,
            "deferred_checks": handler.deferred_checks,
        }
        trace.record_parsed_run(run.audit_layer, audit_row)
    return run


def _sweep(run: Any, allowed: frozenset[str]) -> None:
    unknown = sorted(set(run.options) - allowed)
    if unknown:
        raise ConfigError(
            f"runs[{run.name!r}]: kind: {run.kind} does not take {unknown}; "
            f"it takes {sorted(allowed)}."
        )


def _number(run: Any, key: str, value: Any, *, kind: type,
            minimum: float | None = None) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            f"runs[{run.name!r}]: {key}: is a number; got {value!r}."
        )
    if kind is int and not isinstance(value, int):
        # `int(2.5)` is 2, so a count declared 2.5 used to RUN as 2 -- the
        # document says one thing and the run does another, with nothing to
        # notice.  Two things in this repository already refuse the same
        # value: `transforms._whole`, shipped one task later in this very
        # plan, and the package itself (`n_steps must be a positive int`,
        # tests/inference/test_inference_construction_guards.py:191).  A
        # count that is not an integer is a typo, and the detectable reading
        # is the one this layer takes.
        raise ConfigError(
            f"runs[{run.name!r}]: {key}: is a whole number; got {value!r}. "
            f"It counts, so {kind(value)!r} and {value!r} are different runs "
            "and only one of them is what this document asked for."
        )
    if minimum is not None and not value >= minimum:
        raise ConfigError(
            f"runs[{run.name!r}]: {key}: must be >= {minimum:g}; got "
            f"{value!r}."
        )
    return kind(value)


#: Stands in for an argument a ``python:`` seam would pass, so a callable can
#: be probed without being run.  Its identity is all that matters.
_PROBE = object()


def _binds(fn: Any, *probes: Any) -> tuple[bool, Any]:
    """Does ``fn`` accept ``probes`` positionally? -> (verdict, signature).

    The one place this layer can tell a ``python:`` hook it cannot use from
    one it can, WITHOUT running it.  A contract check, not a restriction on
    the hatch (decision D-C11: recorded, not restricted) -- it forbids nothing
    a working hook can do, and asks only whether the callable accepts the
    arguments the seam is about to pass, which every hook that runs must.

    ``signature`` is None, and the verdict True, when ``inspect`` cannot
    describe the callable at all -- some C builtins, some jax wrappers.  The
    call is then its own check, and guessing there would refuse working code.

    Bind rather than count parameters: ``/``, ``*args``, defaults and
    keyword-only markers all behave, and none of them can be counted right.
    """
    import inspect

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return True, None
    try:
        signature.bind(*probes)
    except TypeError:
        return False, signature
    return True, signature


def _space(run: Any, built: Any) -> Any:
    space = built.inference.space
    if space is None:
        raise ConfigError(
            f"runs[{run.name!r}]: kind: {run.kind} fits latents, and this "
            "document declares no inference.parameters."
        )
    return space


def _on(run: Any, observed: Any) -> str:
    """The observation ``run.on`` names, resolved through ``"primary"``.

    ONE resolver, because two accessors reading ``on:`` two ways is exactly
    how a run comes to be weighed with one observation's sigma and compared
    against another's -- which is the bug the frozen sigma had until this
    task.  The refusal is worded once here rather than per accessor, so the
    two cannot drift.
    """
    name = run.on
    if name == "primary" and observed.primary is not None:
        name = observed.primary
    if name not in observed.entries:
        raise ConfigError(
            f"runs[{run.name!r}]: on: {run.on!r} names no observation; this "
            f"document declares {sorted(observed.entries)}."
        )
    return name


def _noise(run: Any, built: Any) -> Any:
    """The noise this run weighs with -- ITS observation's, not the primary's.

    UNCHANGED SIGNATURE: the fan is a behavioural change, not a new
    argument.  Measured, ``_noise(run, built)`` has SIX call sites in
    ``src`` -- ``exits.py:54`` and ``:229``, ``diagnostics.py:303``,
    ``conjugate.py:347``, and :func:`_decided_sigma`/:func:`_decided_model`
    here, through which every conjugate exit reaches its own -- and a new
    parameter would mean editing all six in a task that is about none of
    them, and every conjugate caller would need it threaded through as
    well.  ``forward``, ``identifiability``, ``score_directions``, ``mmodes``
    and ``predict`` never call it at all, so "one accessor per exit" was never
    true of this function.

    Only ``radiometer_frozen`` with ``source: observed`` fans at all, and
    ``by_observation`` is how this function knows: every other kind is one
    model or one array for the whole document, and ``source:
    prediction_at_init`` reads the TWIN, so it has nothing per-observation
    to fan.  When the mapping exists, so does ``inference.observed`` with a
    primary -- ``build_inference`` refuses the frozen build otherwise -- so
    the resolution below cannot meet a None.

    An ``on:`` the document does not declare is refused HERE, where a sigma
    would otherwise have to be chosen for it -- which means ONLY on the
    fanned kind.  Measured on a two-observation document, ``on: 'dusk'``
    reaches this function and returns a sigma under BOTH unfanned shapes:
    a model kind (``homoscedastic``), and ``radiometer_frozen`` with
    ``source: prediction_at_init``, which is frozen but not fanned because
    it reads the twin.  Neither resolves the name, so neither can reject it.
    ``_observed`` refuses the typo in all three cases, so only an exit that
    takes the sigma alone -- ``fisher`` -- can swallow one.  Catching it on
    every kind is a whole-document check over ``runs[].on`` against
    ``inference.observed``, which is Plan 3's static pass rather than this
    accessor's.
    """
    inference = built.inference
    if inference.noise.by_observation is not None:
        return inference.noise.by_observation[_on(run, inference.observed)]
    noise = decided_noise(inference.noise)
    if noise is None:
        raise ConfigError(
            f"runs[{run.name!r}]: kind: {run.kind} weighs residuals with "
            "inference.noise, and this document declares kind: none -- "
            "legal only for forward and optimize."
        )
    return noise


def _observed(run: Any, built: Any) -> Any:
    observed = built.inference.observed
    if observed is None:
        raise ConfigError(
            f"runs[{run.name!r}]: kind: {run.kind} compares against "
            "inference.observed, and this document declares none."
        )
    return observed.entries[_on(run, observed)]


def _passthrough(options: Mapping, keys: tuple[str, ...]) -> dict:
    return {key: options[key] for key in keys if key in options}


def _decided_sigma(run: Any, built: Any) -> Any:
    """The DECIDED sigma array the conjugate seam takes as ``noise_std=``.

    ``wiener_solve``, ``gcr_sample`` and ``condition_estimate`` compute
    ``1/sigma**2`` directly and refuse a NoiseModel outright
    (``linear.py:1031``).  A constant-sigma model is decided here -- its
    ``std`` ignores the prediction by contract
    (``depends_on_prediction`` is False), so evaluating it on the run's own
    grid gives the full-shaped array, which is also the one shape
    ``check_noise_std_axis`` never has to guess an axis for.  A
    prediction-dependent one cannot be decided at all, and that is check A27.

    Takes no ``observed``: the shape comes from ``built.state.coords``, so a
    document with no ``inference.observed`` still decides a sigma (which is
    what ``condition`` needs).
    """
    import jax.numpy as jnp

    from rheplicant.inference import NoiseModel

    decided = _noise(run, built)
    if not isinstance(decided, NoiseModel):
        return decided
    if decided.depends_on_prediction:
        raise ConfigError(
            f"runs[{run.name!r}]: kind: {run.kind} takes a DECIDED sigma "
            "array, and inference.noise.kind: "
            f"{built.inference.noise.kind} makes sigma a function of the "
            "prediction -- which a conjugate solve has not got, because the "
            "prediction is what it solves for (linear.py:1031, check A27). "
            "Two routes run this noise: kind: conjugate.gls iterates the "
            "covariance it implies, or inference.noise.kind: "
            "radiometer_frozen decides the sigma once and keeps this exit."
        )
    # Only the SHAPE is load-bearing.  A constant-sigma model's std() ignores
    # its argument's VALUES by contract and returns the dtype of its own
    # sigma, not the probe's (measured: a float32 sigma against a float64
    # prediction comes back float32).  So no dtype= here: passing
    # built.context.dtype would read as enforcing the document's dtype on the
    # result, which it does not do -- the document's dtype already reached
    # this sigma when build_noise resolved it.
    coords = built.state.coords
    shape = (int(coords.time.size), int(coords.freq.size))
    return decided.std(jnp.zeros(shape))


def _decided_model(run: Any, built: Any, *, wants: str, reads: str,
                   because: str, instead: str) -> Any:
    """The noise MODEL an exit that reads the noise as a rule needs (A28).

    The mirror of :func:`_decided_sigma`.  ``decided_noise`` returns either a
    NoiseModel or a frozen sigma array, and the two are not interchangeable
    at the conjugate seam: ``iterative_gls`` takes ``noise=`` (the RULE,
    ``gls.py:102-107``) where the three conjugate solves take ``noise_std=``
    (a decided array), and passing either one where the other belongs is a
    hard ParameterSpaceError in both directions.

    **All FOUR clauses are REQUIRED and keyword-only, and that is the fix
    rather than an ergonomic choice.**  Until Plan 3A this function wrote ONE
    sentence for BOTH callers -- ``conjugate.gls``'s -- so ``npe.py:477`` told
    a ``kind: npe`` run that it "solves for the covariance a
    PREDICTION-DEPENDENT sigma implies" and offered it ``kind:
    conjugate.wiener``.  Measured on
    ``posterior_helpers.npe_document(noise=FROZEN)``, both clauses were
    false: npe simulates a bank (``npe.py:475-480`` hands ``noise=`` to
    ``simulate_pairs``, which draws from it) and no conjugate exit produces
    an amortized posterior.  A third caller inheriting conjugate prose is the
    same defect a third time, and a REQUIRED argument is what stops it --
    there is no default left to inherit.

    **``reads`` and ``because`` are required for a second reason, and it is a
    defect this plan shipped and then had to take back.**  A two-clause form
    of this function templated *"so it reads inference.noise as a RULE"* and
    *"a decided array is not a rule"* as FIXED text -- which silently reworded
    ``conjugate.gls``'s sentence, whose ``be2027b`` text is *"so it reads
    inference.noise as a model"* and *"a decided array has no fixed point to
    iterate"*.  Plan §2.3 designates exactly four messages CORRECTED (A39's)
    and makes every other one a MOVE that keeps its words; A28's gls sentence
    was never the false one.  Whatever a caller and its sibling do not share
    belongs to the caller, so both fragments are clauses now and neither has a
    default a fifth caller could inherit.

    The sentence is
    ``kind: X <wants>, so it reads inference.noise as <reads>;
    inference.noise.kind: <k> decides its sigma into an array before any run
    sees it, and a decided array <because> (check A28). <instead>``.
    ``instead`` is a whole sentence of advice, ending in its own full stop.
    All four are supplied by the caller, which is the only place that knows
    what it does with the rule, and each caller binds its set ONCE at its own
    module scope (``conjugate._A28_GLS_CLAUSES``,
    ``conjugate._A28_GCR_CLAUSES``, ``npe._A28_NPE_CLAUSES``) so that the
    clause a reader is pinned against is the clause the call site spreads.
    """
    from rheplicant.inference import NoiseModel

    noise = _noise(run, built)
    if isinstance(noise, NoiseModel):
        return noise
    raise ConfigError(
        f"runs[{run.name!r}]: kind: {run.kind} {wants}, so it reads "
        f"inference.noise as {reads}; inference.noise.kind: "
        f"{built.inference.noise.kind} decides its sigma into an array "
        f"before any run sees it, and a decided array {because} "
        f"(check A28). {instead}"
    )


def reuse_of(run: Any, results: Mapping[str, Any] | None) -> Any:
    """The RunResult an exit's ``reuse:`` names, or a refusal saying why not.

    Runs execute in declaration order, so a reuse may only look backwards --
    naming a later run reads exactly like naming a missing one, and the
    message says so.
    """
    where = f"runs[{run.name!r}]"
    if run.reuse is None:
        raise ConfigError(
            f"{where}: kind: {run.kind} reads an earlier run's product, so "
            "reuse: <run name> is required."
        )
    results = results or {}
    if run.reuse not in results:
        raise ConfigError(
            f"{where}: reuse: {run.reuse!r} names no earlier run; runs "
            f"execute in declaration order and by now {sorted(results)} have "
            "run."
        )
    earlier = results[run.reuse]
    if earlier.error is not None:
        raise ConfigError(
            f"{where}: reuse: {run.reuse!r} refused ({earlier.error}), so it "
            "has no product to read."
        )
    return earlier
