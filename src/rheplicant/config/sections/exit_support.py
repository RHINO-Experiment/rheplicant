"""What every exit executor shares: the sweep, the accessors, the registry.

An executor is ``(run, built, *, results=None) -> product``.  It is registered
under its ``runs[].kind`` by the :func:`register` decorator, and
:data:`EXECUTORS` is the one table :func:`execute_run` dispatches through.
The leaf modules (``exits``, ``conjugate``, ``diagnostics``) import from here
and never from each other, so the registration is a one-way import.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from rheplicant.config.errors import ConfigError
from rheplicant.config.sections.noise import decided_noise

__all__ = ["EXECUTORS", "register", "reuse_of"]

EXECUTORS: dict[str, Callable[..., Any]] = {}


def register(kind: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Bind an executor to its ``runs[].kind``.

    Registering the same kind twice is a programming error, not a
    configuration one -- and it is refused with a RAISE rather than the
    ``assert`` this used to carry, because ``python -O`` strips asserts and
    the shadowing is then completely silent.  Measured before the change::

        $ python -O -c "...register('_probe')(one); register('_probe')(two)..."
        under -O, second registration won: True

    -- so under ``-O`` which executor a document got depended on import order,
    with nothing said.  ``ConfigError`` and not ``RuntimeError``, following
    ``errors.py``'s "one refusal type for the whole config layer"; the message
    is what carries "this is wiring, not your document", and it names both
    claimants so the second one can be found.

    That rule is stated layer-wide and, measured, is not yet applied
    layer-wide: this is the first of five registration decorators in
    ``config/`` to refuse a double registration at all.  The other four --
    ``files.register_reader``, ``values.register_form``,
    ``derive.register_derivation`` and ``resources.register_kind`` -- assign
    unconditionally, so the second registration wins in silence with no
    ``assert`` to strip.  They are outside Plan 3A's scope and are recorded on
    its residue ledger rather than fixed here.
    """

    def bind(fn: Callable[..., Any]) -> Callable[..., Any]:
        if kind in EXECUTORS:
            raise ConfigError(
                f"runs[].kind: {kind!r} is registered twice, by "
                f"{EXECUTORS[kind].__module__} and by {fn.__module__}. A kind "
                "has one executor, and which of the two you would get depends "
                "on import order."
            )
        EXECUTORS[kind] = fn
        return fn

    return bind


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
