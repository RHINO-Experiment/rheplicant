"""The adapter: a ``Pipeline`` and a ``ParameterSpace``, seen as a bayesmith ``Graph``.

Two things live here and nothing else does.

:func:`to_graph` builds the graph. It is the only place in this package that
knows how a declaration on this side spells itself on that one, so the mapping
exists once rather than once per exit.

:func:`translate` is the return path. bayesmith raises bayesmith's exceptions;
the names, classes and pinned sentences on this side are a keeping surface, so
every call across the seam is wrapped and every refusal comes back wearing this
package's own class. Wrapping is not string rewriting -- the affine refusal
carries its probe's numbers across as data, which is what
``bayesmith.AffinityRefused``'s payload exists for.

**Refusals that the graph seam would erase live BEFORE it**, in
:func:`_prevalidate`. A sigma vector whose axis the prediction cannot settle is
the founding case: once the noise model has been evaluated into a
``dist.Normal(mu, sigma)``, the ambiguity has already been resolved by
broadcasting and there is nothing left to refuse. The general rule -- if the
evidence for a refusal is destroyed by building the graph, refuse before
building it -- is why this module validates rather than translating alone.

**What this module does NOT do.** It does not choose an inference method, read
a ``SamplingPlan``, or partition anything: it hands back a ``Graph`` and stops.
And it does not widen the facade -- capabilities bayesmith has and this package
does not publish stay unreachable through here, because a keyword forwarded
"just in case" is a published surface nobody decided to publish.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, NoReturn

import jax
import jax.numpy as jnp

from rheplicant.core.errors import (
    LinearityRefused,
    ParameterSpaceError,
    StateValidationError,
)
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State
from rheplicant.inference.likelihood import check_observed_shape
from rheplicant.inference.noise import (
    FlaggedNoise,
    HomoscedasticNoise,
    NoiseModel,
    check_noise_std_axis,
)
from rheplicant.inference.parameters import Latent, ParameterSpace

#: The graph's name for the prediction. Double-underscored on both sides so it
#: reads as internal wherever it surfaces, and refused as a latent name below
#: so a declaration can never collide with it. bayesmith keys its posterior
#: draws by node name, and this node is not something a caller declared, so it
#: must not appear in that key space -- the facade drops it.
PREDICTION: str = "__mu__"

#: The graph's name for the observation node, for the same reasons.
OBSERVATION: str = "__data__"

#: The two names above, as the set a latent may not take.
INTERNAL_NAMES: frozenset[str] = frozenset({PREDICTION, OBSERVATION})


class SeamRefusal(ParameterSpaceError):
    """A bayesmith refusal with no counterpart class on this side.

    A SUBCLASS of :class:`ParameterSpaceError` for the reason
    :class:`~rheplicant.core.errors.LinearityRefused` is one: every
    ``except ParameterSpaceError`` already written keeps catching it, and the
    caller who wants to know it came from across the seam can ask.

    Carries the original as ``__cause__`` (the ``raise ... from`` is in
    :func:`translate`) and names the site, because "which call did this" is the
    first question and a translated message has, by construction, lost the
    bayesmith traceback's own answer to it.

    Attributes:
        site: the seam site the caller named -- a short phrase, not a
            traceback.
    """

    def __init__(self, *args: object, site: str) -> None:
        super().__init__(*args)
        self.site = str(site)

    def __reduce__(self):
        # Required keywords break Python's default `cls(*args)` rebuild, and
        # pytest-xdist serialises exceptions across workers -- so without this
        # a refusal reaches the report as a TypeError about a missing argument,
        # displacing the error it was reporting. The same repair
        # bayesmith.errors made for its three payload classes.
        return (_rebuild_seam_refusal, (self.args, self.site))


def _rebuild_seam_refusal(args: tuple, site: str) -> SeamRefusal:
    return SeamRefusal(*args, site=site)


@dataclass
class Seam:
    """What a :func:`translate` block hands back: the verdict it swallowed.

    ``blameless`` is ``None`` when the block ran to the end, and otherwise the
    ``NotGaussian`` / ``NotLogLinear`` instance that ended it. Those two are
    not errors -- they are the answer "no route of that kind exists here" to a
    question the caller asked, and this package's callers ask it in order to
    branch. So they are caught, and the block's remaining statements are
    skipped, which means a caller reads the verdict from HERE rather than from
    a value the block never got to assign::

        with translate("log space") as seam:
            route = bayesmith.log_space(graph)
        if seam.blameless is not None:
            ...            # no log route; `route` was never assigned

    Attributes:
        site: the site string this block was opened with.
        blameless: the swallowed verdict, or ``None``.
    """

    site: str
    blameless: Any = None

    @property
    def refused(self) -> bool:
        """Whether a blameless verdict ended the block."""
        return self.blameless is not None


def _bayesmith_errors() -> Any:
    """bayesmith's error module.

    Imported inside the function rather than at module scope so that importing
    this package does not import bayesmith, and so that a missing bayesmith
    fails at the call that needed it rather than at import of anything that
    merely sits beside it. ``bayesmith.errors`` is stdlib-only by that
    package's own contract, so this costs no jax import either.
    """
    import bayesmith.errors as errors

    return errors


@contextmanager
def translate(site: str) -> Iterator[Seam]:
    """Wrap one call across the seam; re-raise its refusals as this package's.

    Call it **outside** ``bayesmith.trace(...)``, around the bayesmith call
    itself. Inside a trace the graph is half-built, and an exception then
    reports a node the caller never wrote about a graph that does not exist
    yet; the site string could name only the trace.

    Three families, three fates:

    ``AffinityRefused``
        the declared affinity is false. Becomes
        :class:`~rheplicant.core.errors.LinearityRefused` **carrying the same
        numbers** -- ``errors``, ``rtol``, ``failed`` -- so a caller reads them
        as data on either side of the seam. ``weighted`` and ``weighted_rtol``
        come across too; they are bayesmith's second criterion and this
        package's own probe has no counterpart, which is precisely why the
        translated exception is where they become visible rather than a facade
        keyword.

    ``GraphError`` / ``TraceError`` / other ``StructureError``
        the graph is wrong, or a structural claim about it is. Becomes
        :class:`SeamRefusal`, a ``ParameterSpaceError``, with the original
        message quoted and the site named.

    ``NotGaussian`` / ``NotLogLinear``
        blameless verdicts: "there is no exact route of that kind here". These
        are **caught and not re-raised** -- they are answers, and a caller that
        asked in order to branch should not have to write ``except`` around a
        question. The block ends where the verdict was raised, and the verdict
        is on the yielded :class:`Seam`.

    Args:
        site: a short phrase naming the call, used in the translated message.

    Yields:
        A :class:`Seam` recording whether a blameless verdict ended the block.
    """
    errors = _bayesmith_errors()
    seam = Seam(site=site)
    try:
        yield seam
    except errors.AffinityRefused as refusal:
        raise LinearityRefused(
            f"{refusal}",
            errors=refusal.errors,
            rtol=refusal.rtol,
            failed=refusal.failed,
            weighted=refusal.weighted,
            weighted_rtol=refusal.weighted_rtol,
        ) from refusal
    except (errors.NotGaussian, errors.NotLogLinear) as verdict:
        seam.blameless = verdict
    except errors.BayesmithError as refusal:
        raise SeamRefusal(
            f"{site}: {refusal}",
            site=site,
        ) from refusal


# ---------------------------------------------------------------- building --


@dataclass(frozen=True)
class _Resolved:
    """What :func:`_prevalidate` settled, so :func:`to_graph` only assembles."""

    names: tuple[str, ...]
    linear_names: tuple[str, ...]
    priors: Mapping[str, Any]
    forward: Callable[[dict[str, jax.Array]], jax.Array]
    prediction_shape: tuple[int, ...]


def _refuse(message: str) -> NoReturn:
    """Raise, and be TYPED as raising.

    ``NoReturn`` and not ``None``: several callers below end with a call to
    this and nothing else, and a checker that thinks it can return would
    read those as falling off the end with an implicit ``None`` -- which is
    exactly what a resolved prior must never be.
    """
    raise ParameterSpaceError(message)


def _prior_of(
    latent: Latent, supplied: Mapping[str, Any], covered_by: str | None = None
) -> Any:
    """The one prior in force for ``latent`` -- declared or supplied, never both.

    ``priors=`` is the SINGLE entry for a prior that the declaration does not
    carry: the facade folds a call-site ``prior_mean=``/``prior_std=`` into it
    (see :func:`priors_from_keywords`), so there is one path and not two. A
    name appearing in both places is refused rather than resolved here,
    because reconciling two declarations is what that folding already did with
    the contradiction check attached, and doing it twice in two places is how
    the two readings drift.
    """
    name = latent.name
    if covered_by is not None:
        if name in supplied:
            _refuse(
                f"to_graph was given priors[{name!r}], and {name!r} is also covered by "
                f"this space's joint_prior over {covered_by}. That is two priors on one "
                "quantity: the posterior would be multiplied by both, which is a proper "
                "density and a plausible chain and not the model either declaration "
                "describes -- and no diagnostic reports a prior counted twice, because "
                "each one on its own is correct. Drop the priors= entry, or take the "
                "latent out of over=."
            )
        return _flat_prior_for(latent)
    if name in supplied:
        if latent.prior is not None:
            _refuse(
                f"to_graph was given priors[{name!r}] and latent {name!r} also declares "
                f"prior={type(latent.prior).__name__}(...). A graph node carries exactly "
                "one distribution, so one of these would silently win while the other "
                "read like it was in force. Reconcile them at the call site -- "
                "priors_from_keywords does that, with the contradiction check attached -- "
                "and pass to_graph the single survivor."
            )
        return supplied[name]
    if latent.prior is None:
        _refuse(
            f"Latent {name!r} declares no prior, and none was supplied for it. A free "
            "parameter is fine for the optimizers, but a graph node IS its distribution: "
            "there is no way to declare a node with no density, so a posterior over this "
            "space is not defined. Declare Latent(prior=...), or pass "
            f"priors={{{name!r}: ...}} for the run that needs one."
        )
    return latent.prior


def _refuse_unusable_priors(latents: tuple[Latent, ...], priors: Mapping[str, Any]) -> None:
    """Every supplied name must be a latent, and complex latents need ComplexNormal."""
    declared = {latent.name for latent in latents}
    unknown = sorted(name for name in priors if name not in declared)
    if unknown:
        _refuse(
            f"to_graph was given priors for {unknown}, which this space does not declare; "
            f"it declares {sorted(declared)}. The entry would be dropped in silence and "
            "the latent it names sampled from whatever its declaration happens to say."
        )


def _refuse_a_complex_latent_without_a_complex_prior(latent: Latent, prior: Any) -> None:
    """A complex latent's prior has to be one numpyro cannot supply.

    Measured while G9's minimal surface was built: no numpyro distribution
    samples complex, and a block reads its dtype off the prior's ``loc``. So
    ``bayesmith.ComplexNormal`` is the only declaration a complex latent has,
    and this refuses rather than PROMOTING a real ``Normal`` to one -- the
    promotion would have to choose what ``scale`` meant, and the convention on
    that side is that each half carries ``scale**2``, so promoting silently
    doubles the prior variance a reader thought they declared.
    """
    if not jnp.issubdtype(latent.init.dtype, jnp.complexfloating):
        return
    import bayesmith

    if isinstance(prior, bayesmith.ComplexNormal):
        return
    _refuse(
        f"Latent {latent.name!r} has dtype {latent.init.dtype}, so its prior must be a "
        f"bayesmith.ComplexNormal; it is a {type(prior).__name__}. No numpyro "
        "distribution samples complex values, which is why that class exists. It is NOT "
        "substituted for you here: ComplexNormal's two parts are independent and each "
        "carries scale**2, so reading a real Normal's scale as one would double the "
        "prior variance you declared without saying so."
    )


def _refuse_scopes_without_a_graph_spelling(latents: tuple[Latent, ...]) -> None:
    """``per_epoch`` and ``linked`` have no seam yet; say which, and where it goes."""
    for latent in latents:
        if latent.scope == "per_epoch":
            _refuse(
                f"Latent {latent.name!r} declares scope='per_epoch', which becomes a "
                "bayesmith Plate. The adapter does not emit plates yet -- that is the "
                "epoch wave -- and building the graph without one would marginalise a "
                "quantity declared to be re-drawn as though it were held fixed, which "
                "is a different model and a finite, wrong answer. Use the per-epoch "
                "exits that already exist, or wait for the plate seam."
            )
        if latent.scope == "linked":
            _refuse(
                f"Latent {latent.name!r} declares scope='linked', which needs a declared "
                "transition -- a chain, not a plate. bayesmith has no chain layer yet "
                "(it is the G3 gap), so there is nothing to map the transition onto and "
                "dropping it would turn a Markov chain into independent draws."
            )


def _observed_mask(noise: Any) -> Any:
    """``FlaggedNoise``'s flags as the graph's own declaration, or ``None``.

    THIS side spells an unobserved sample ``sigma = inf`` and every consumer
    here turns it into a clean zero weight. A graph node cannot do that,
    because ``Normal(mu, inf)`` has log-density ``-inf`` everywhere and takes
    the whole joint with it -- which is why this used to be a refusal naming
    the G1 gap.

    G1 landed (bayesmith 0.4.0) and the gap is closed the way the far side
    chose: the mask is DECLARED on the node rather than encoded in the scale,
    so ``log_joint``, ``to_numpyro`` and the exact solve all read one
    statement. The translation is one line, and its direction is worth
    stating: ``FlaggedNoise.flags`` is True where a sample was FLAGGED, while
    the graph's ``mask`` is True where a sample was TAKEN. They are negations
    of each other, and both are boolean arrays of the data's shape, so
    swapping them costs nothing and is caught by nothing -- which is why
    `tests/inference/test_graph_bridge.py` pins the polarity against the
    flagged CHANNEL rather than against a count.
    """
    if not isinstance(noise, FlaggedNoise):
        return None
    return jnp.logical_not(jnp.asarray(noise.flags))


def _refuse_a_noise_model_this_seam_cannot_read(noise: Any) -> None:
    """Say what a noise argument has to be, rather than failing where it is used."""
    if isinstance(noise, NoiseModel) and hasattr(noise, "depends_on_prediction"):
        return
    _refuse(
        f"to_graph takes a NoiseModel, not a {type(noise).__name__}. The observed node's "
        "distribution is built as Normal(mu, noise.std(mu)) and the graph is told whether "
        "that sigma moves with the prediction, so the seam needs both `std` and "
        "`depends_on_prediction`. Wrap a fixed sigma in HomoscedasticNoise(sigma=...) -- "
        "which is the same model, named."
    )


def _refuse_internal_names(latents: tuple[Latent, ...], reserved: frozenset[str]) -> None:
    """No latent may take a name this call gives one of the graph's own nodes.

    ``reserved`` rather than :data:`INTERNAL_NAMES`, because the two node names
    are no longer fixed: ``to_numpyro_model`` names them ``"prediction"`` and
    its ``obs_name`` so that the sites a caller reads out of ``get_samples()``
    keep the names this package has always given them (D26). A refusal keyed to
    the defaults would then guard the wrong two strings -- it would let a latent
    called ``prediction`` through and refuse one called ``__mu__``, which is the
    exact inverse of what that call needs.
    """
    for latent in latents:
        if latent.name in reserved:
            _refuse(
                f"Latent {latent.name!r} takes one of the adapter's internal node names "
                f"{sorted(reserved)}. Those name the prediction and the observation "
                "in the graph this builds, so the collision would be a duplicate node "
                "name -- or worse, a latent whose posterior draws came back keyed to the "
                "prediction. Rename the latent."
            )


def _flat_prior_for(latent: Latent) -> Any:
    """The improper density a joint-prior-covered latent declares on the graph.

    A covered latent still needs a NODE -- a sampler needs the coordinate, and
    a graph node IS its distribution -- but it must not carry a density, because
    the whole density over that block arrives once at the factor site. An
    improper flat site contributes exactly zero, so the block's log prior is
    the joint prior and nothing else.

    The spelling is the one the far side checks for **by type**: its
    ``_check_against`` refuses a covered latent whose density is not an
    ``ImproperUniform``, on the grounds that two priors on one quantity give a
    proper density and a plausible chain and no diagnostic that reports it.
    This is also the spelling ``to_numpyro_model`` has always used for the same
    latents, so the graph and the hand-written bridge declare the same thing.
    """
    import numpyro.distributions as distributions

    return distributions.ImproperUniform(
        distributions.constraints.real, (), event_shape=jnp.shape(latent.init)
    )


def _bayesmith_joint_prior(prior: Any) -> Any:
    """This package's ``JeffreysPrior`` as the graph's.

    Both are ``eqx.Module``s over the same two static fields, and the second is
    carried rather than dropped: ``rank_rtol=None`` means "the default" on both
    sides and the same default, but an EXPLICIT rank_rtol is a caller's
    decision about where a null eigenvalue starts, and losing it here would
    leave a rank verdict taken at a tolerance nobody asked for -- finite,
    plausible, and a different prior.
    """
    import bayesmith

    return bayesmith.JeffreysPrior(over=prior.over, rank_rtol=prior.rank_rtol)


def _refuse_a_scale_prior_the_seam_cannot_read(noise: Any, scale_prior: Any) -> None:
    """A declared scale latent replaces the noise model's sigma, so the noise
    model must have nothing else to say.

    ``scale_prior`` exists for one caller: ``to_numpyro_model`` given a
    distribution-valued ``noise_std``, which today becomes the sample site
    ``"noise_std"`` and then a
    :class:`~rheplicant.inference.noise.HomoscedasticNoise` around it. The
    sigma that noise model carries is therefore a PLACEHOLDER -- the value
    comes from the latent instead -- and the only thing read off it is its
    flags.

    So it has to be a homoscedastic model, optionally flagged, and nothing
    else. A radiometer's sigma tracks the prediction; silently replacing that
    with a constant latent would be a different model with the same shapes,
    the same dtypes and a perfectly healthy chain. The placeholder's
    unreachability is measured rather than asserted (see
    ``TestASampledScaleCrossesAsALatent`` in ``tests/inference/
    test_graph_bridge.py``), and this refusal is what keeps it measurable: it
    is only unreachable because nothing else could have contributed.
    """
    if scale_prior is None:
        return
    base = noise.base if isinstance(noise, FlaggedNoise) else noise
    if not isinstance(base, HomoscedasticNoise):
        _refuse(
            f"to_graph was given scale_prior={scale_prior[0]!r} together with "
            f"{type(base).__name__} noise. A declared scale latent REPLACES the noise "
            "model's sigma, so the only thing left to read off the noise model is its "
            "flags -- and a sigma that tracks the prediction has more to say than that. "
            "Replacing it with a constant latent would be a different model with the "
            "same shapes and a healthy chain. Pass a HomoscedasticNoise (optionally "
            "flagged), or drop scale_prior and let the noise model carry the sigma."
        )


def _prevalidate(
    space: ParameterSpace,
    pipeline: AbstractOperator,
    state_template: State,
    observed: Any,
    noise: Any,
    priors: Mapping[str, Any] | None,
    reserved: frozenset[str],
    scale_prior: tuple[str, Any] | None,
) -> _Resolved:
    """Everything that must be refused while the evidence for it still exists.

    Ordered so that the cheapest structural refusals come first and the ones
    needing a prediction shape come last -- ``jax.eval_shape`` is cheap, but it
    still runs the forward model abstractly, and a space with a name collision
    should not have to.
    """
    supplied: Mapping[str, Any] = {} if priors is None else dict(priors)
    latents = tuple(space.latents)

    _refuse_internal_names(latents, reserved)
    _refuse_scopes_without_a_graph_spelling(latents)
    _refuse_unusable_priors(latents, supplied)
    _refuse_a_noise_model_this_seam_cannot_read(noise)
    _refuse_a_scale_prior_the_seam_cannot_read(noise, scale_prior)

    joint = space.joint_prior
    resolved_priors = {}
    for latent in latents:
        covered = joint is not None and joint.covers(latent.name)
        prior = _prior_of(latent, supplied, covered_by=joint.label if covered else None)
        if not covered:
            # A covered latent's flat site has no loc to read a dtype off, and
            # a complex one is refused upstream by `check_identified` before a
            # graph is built at all -- so this check would be asking the flat
            # declaration a question it cannot answer.
            _refuse_a_complex_latent_without_a_complex_prior(latent, prior)
        resolved_priors[latent.name] = prior

    # `forward_fn` validates the space against the pipeline (aliased targets,
    # unbound latents, shapes that do not fit their leaf) before returning, so
    # those refusals arrive here already, wearing their own messages.
    forward, values0 = space.forward_fn(pipeline, state_template)
    prediction_shape = tuple(jax.eval_shape(forward, values0).shape)

    # The founding pre-validation: past this point sigma has been broadcast
    # into a distribution and the ambiguity is settled, wrongly and silently.
    check_noise_std_axis(noise, prediction_shape, "to_graph")
    check_observed_shape(prediction_shape, observed, predictor="this forward model")
    if jnp.iscomplexobj(observed):
        raise StateValidationError(
            "to_graph was given complex `observed`. The observation node is a real "
            "Normal, and a complex datum would have to be split into two real nodes with "
            "a covariance convention nobody here declared. A complex LATENT is supported "
            "(bayesmith.ComplexNormal); complex DATA is not."
        )

    return _Resolved(
        names=tuple(latent.name for latent in latents),
        linear_names=tuple(latent.name for latent in latents if latent.linear),
        priors=resolved_priors,
        forward=forward,
        prediction_shape=prediction_shape,
    )


def to_graph(
    space: ParameterSpace,
    pipeline: AbstractOperator,
    state_template: State,
    observed: Any,
    noise: Any,
    *,
    priors: Mapping[str, Any] | None = None,
    prediction_name: str = PREDICTION,
    observation_name: str = OBSERVATION,
    scale_prior: tuple[str, Any] | None = None,
) -> Any:
    """Build the bayesmith ``Graph`` this space, pipeline and data describe.

    The graph is three layers and always exactly three: one ``sample`` node per
    declared latent, one ``det`` node holding the whole forward model, and one
    ``observe`` node holding the likelihood. The forward model is opaque to
    bayesmith on purpose -- it is a pipeline, and the seam's job is to present
    it, not to decompose it. What bayesmith needs to know ABOUT it is declared:
    ``linear_in`` names the latents declared ``linear=True``, and that claim is
    checked on the far side before it is exploited.

    Args:
        space: the declaration. Validated against ``pipeline`` first.
        pipeline: the forward model.
        state_template: the state it is evaluated on. Closed over, fixed.
        observed: the data. Its shape must equal the prediction's exactly --
            broadcast-compatible is not enough, and the compatible cases are
            the dangerous ones.
        noise: a :class:`~rheplicant.inference.noise.NoiseModel`. Its ``std``
            is the observation node's sigma, so the mapping is the DENSITY
            side of the model. The generator side (``realise``) deliberately
            differs for the radiometer -- multiplicative, no floor, sign
            preserved -- and does not come through here.
        priors: ``{name: distribution}`` for latents whose declaration carries
            none. The single entry for a prior supplied at a call site; see
            :func:`priors_from_keywords`.

    Returns:
        A ``bayesmith.Graph``.

    Raises:
        ParameterSpaceError: for every declaration this seam cannot spell --
            an undeclared prior, a scope with no graph form, a joint prior, a
            name colliding with an internal node.
        StateValidationError: for a sigma vector whose axis the prediction
            cannot settle, and for complex data.
    """
    import bayesmith

    reserved = frozenset({prediction_name, observation_name}) | (
        frozenset() if scale_prior is None else frozenset({scale_prior[0]})
    )
    resolved = _prevalidate(
        space, pipeline, state_template, observed, noise, priors, reserved, scale_prior
    )
    forward = resolved.forward
    names = resolved.names
    priors_by_name = resolved.priors

    def model(data: Any) -> None:
        refs = [
            bayesmith.sample(name, _prior_factory(priors_by_name[name])) for name in names
        ]
        prediction = bayesmith.det(
            prediction_name,
            _prediction_fn(forward, names),
            *refs,
            linear_in=resolved.linear_names,
        )
        parents = [prediction]
        if scale_prior is not None:
            # The scale is a node, not a number. `observe` takes several
            # parents, so this needs nothing from the far side that 0.4.0 does
            # not already have -- measured before it was written (D27).
            parents.append(
                bayesmith.sample(scale_prior[0], _prior_factory(scale_prior[1]))
            )
        bayesmith.observe(
            observation_name,
            _observation_fn(noise, declared_scale=scale_prior is not None),
            *parents,
            obs=data,
            mask=_observed_mask(noise),
            depends_on_prediction=bool(noise.depends_on_prediction),
        )
        if space.joint_prior is not None:
            # Declared, not added by a consumer afterwards. A graph missing this
            # line builds, validates, samples and diagnoses cleanly, and is a
            # likelihood-only posterior -- which is exactly what the refusal
            # this replaced said it was preventing.
            bayesmith.joint_prior(_bayesmith_joint_prior(space.joint_prior))

    with translate("to_graph"):
        return bayesmith.trace(model, observed)


def graph_for_information(
    forward: Callable[[dict[str, jax.Array]], jax.Array],
    values: Mapping[str, jax.Array],
    noise: Any,
) -> Any:
    """A graph for a FISHER question, built from a bare forward callable.

    :func:`to_graph` takes a ``ParameterSpace`` and a pipeline. A Fisher block
    is asked of neither: ``JeffreysPrior.information`` is handed ``f(values) ->
    prediction`` and a values dict, because that is the shape its callers have
    -- a NumPyro model body holds a closure over the bound pipeline and nothing
    it could hand back. So this is the same three-layer construction reached
    from the other side, sharing this module's node functions so the
    observation node has one spelling and not two.

    **Two things are synthesised, and the legality of both is that the answer
    cannot reach them** -- the same argument D22 makes for the rank test, and
    like that one it is MEASURED rather than asserted (see
    ``TestTheSynthesisedInformationGraph`` in
    ``tests/inference/test_jeffreys_prior.py``):

    * the **data**, as zeros of the prediction's shape. A Fisher information is
      an EXPECTED information -- ``J^T N^-1 J`` plus the variance's own term --
      and no residual appears in either. The far side reads the noise from the
      graph at ``values`` and the design from the block; neither reads ``obs``.
    * every latent's **density**, as an improper flat one. The block's prior
      fields are deliberately empty on the far side, and the covered latents
      must be flat anyway or ``_check_against`` refuses them as two priors on
      one quantity. Declaring the uncovered ones flat as well keeps one rule
      here rather than two, and they are held fixed at ``values`` regardless.

    What is NOT synthesised is the noise: it is the caller's, because the whole
    subject of this prior is that the noise model chooses its shape.
    """
    import bayesmith
    import numpyro.distributions as distributions

    names = tuple(values)
    flat = distributions.ImproperUniform(distributions.constraints.real, (), ())
    prediction_shape = tuple(jax.eval_shape(forward, dict(values)).shape)
    check_noise_std_axis(noise, prediction_shape, "JeffreysPrior")

    def model(data: Any) -> None:
        refs = [bayesmith.sample(name, _prior_factory(flat)) for name in names]
        prediction = bayesmith.det(PREDICTION, _prediction_fn(forward, names), *refs)
        bayesmith.observe(
            OBSERVATION,
            _observation_fn(noise),
            prediction,
            obs=data,
            mask=_observed_mask(noise),
            depends_on_prediction=bool(noise.depends_on_prediction),
        )

    with translate("JeffreysPrior"):
        return bayesmith.trace(model, jnp.zeros(prediction_shape))


def _prior_factory(prior: Any) -> Callable[[], Any]:
    """``lambda: prior``, bound now.

    A closure written inline in the loop would capture the LOOP VARIABLE and
    every node would end up with the last latent's prior -- finite, correctly
    shaped, and a different model. Python's late binding is the whole reason
    this is a function.
    """
    return lambda: prior


def _prediction_fn(
    forward: Callable[[dict[str, jax.Array]], jax.Array], names: tuple[str, ...]
) -> Callable[..., jax.Array]:
    """The det node's ``fn``: positional parents in, prediction out.

    bayesmith calls ``fn`` with the parents' values in declaration order, and
    ``forward`` wants them keyed by name. Zipping is the whole adapter at this
    node -- ``names`` is the same tuple the ``sample`` nodes were declared in,
    so the two orders cannot drift.
    """

    def prediction(*values: jax.Array) -> jax.Array:
        return forward(dict(zip(names, values, strict=True)))

    return prediction


def _finite_sigma(noise: Any, prediction: jax.Array) -> jax.Array:
    """``noise.std``, with a flagged sample's ``inf`` replaced by the base's own.

    The scale a graph node declares has to be finite EVERYWHERE, and the far
    side checks that by name: ``check_gaussian`` refuses a non-finite sigma
    because "the sigma expression produced an infinity" and "this sample was
    flagged" need different fixes. The second is now said by the node's
    ``mask``, so the scale says only the first.

    The value put at a flagged sample is the BASE model's sigma, not a
    placeholder 1.0. It cannot reach any answer -- a masked sample contributes
    nothing to the quadratic form, the normaliser, the information or a draw,
    and the far side pins that a 1e9 change there moves nothing
    (`bayesmith/tests/exact/test_masking.py` -- the project is named in the
    path because it is not this repository's). Using the instrument's own number anyway means the
    graph reads as the model rather than as the encoding.
    """
    if isinstance(noise, FlaggedNoise):
        return noise.base.std(prediction)
    return noise.std(prediction)


def _observation_fn(noise: Any, *, declared_scale: bool = False) -> Callable[..., Any]:
    """The observed node's ``dist_fn``: ``Normal(mu, sigma)``.

    The sigma is asked of the noise model rather than re-derived from its
    fields, which is what keeps the radiometer's floor, its absolute value and
    its fractional level in one place. A second spelling here would be the copy
    that goes stale, and it would go stale silently: both readings are finite
    and correctly shaped.

    With ``declared_scale`` the sigma arrives as a second PARENT instead --
    a latent the graph declares (D27). The noise model is then read only for
    its flags, and ``_refuse_a_scale_prior_the_seam_cannot_read`` has already
    established that it has nothing else to say. The scale is broadcast to the
    prediction's shape rather than trusted to broadcast on its own: a scalar
    would, and so would a vector of the wrong length against the wrong axis.
    """
    import numpyro.distributions as distributions

    def observation(prediction: jax.Array) -> Any:
        return distributions.Normal(prediction, _finite_sigma(noise, prediction))

    def observation_with_scale(prediction: jax.Array, scale: jax.Array) -> Any:
        return distributions.Normal(
            prediction, jnp.broadcast_to(scale, jnp.shape(prediction))
        )

    return observation_with_scale if declared_scale else observation


# ------------------------------------------------- the single prior entry --


def priors_from_keywords(
    space: ParameterSpace,
    *,
    prior_mean: Any = None,
    prior_std: Any = None,
    caller: str,
) -> dict[str, Any]:
    """Fold call-site ``prior_mean=``/``prior_std=`` into ``priors=``.

    The facade's exits take those two keywords; a graph takes distributions.
    This is the ONE conversion, so the contradiction check that has always
    guarded those keywords -- a supplied value that disagrees with a declared
    prior is refused, never silently preferred -- guards the graph path too.

    Keywords may be a scalar (one value for every prior-free latent) or a dict
    keyed by latent name. A latent that declares its own prior is left alone
    and keeps it; the reconciliation refuses the case where both are given and
    they differ.

    Args:
        space: the declaration.
        prior_mean: ``loc`` for the synthesised priors.
        prior_std: ``scale`` for them.
        caller: the exit to name in a refusal.

    Returns:
        ``{name: Normal(loc, scale)}`` for exactly those latents that needed
        one. Latents whose declaration carries a prior are absent, so the
        result can be passed to :func:`to_graph` as-is.
    """
    import numpyro.distributions as distributions

    from rheplicant.inference.linear import _resolve_one_prior

    def per_latent(value: Any, name: str) -> Any:
        if isinstance(value, Mapping):
            return value.get(name)
        return value

    synthesised: dict[str, Any] = {}
    for latent in space.latents:
        name = latent.name
        mean, std = _resolve_one_prior(
            name,
            latent.prior,
            per_latent(prior_mean, name),
            per_latent(prior_std, name),
            caller,
        )
        if latent.prior is not None:
            continue
        if std is None:
            _refuse(
                f"{caller} has no prior for latent {name!r}: it declares none and no "
                "prior_std= was given for it. A graph node IS its distribution, so "
                "there is nothing to declare it with. Pass a large prior_std for an "
                "effectively flat prior, or declare Latent(prior=...)."
            )
        loc = jnp.zeros(latent.init.shape, latent.init.dtype) if mean is None else mean
        synthesised[name] = distributions.Normal(
            jnp.broadcast_to(jnp.asarray(loc), latent.init.shape),
            jnp.broadcast_to(jnp.asarray(std), latent.init.shape),
        )
    return synthesised
