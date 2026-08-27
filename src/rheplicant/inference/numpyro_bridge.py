"""Bridge a Pipeline/Assembly to a NumPyro probabilistic model.

Bayesian inference through the same seam as everything else (D7): a
:class:`~rheplicant.inference.parameters.ParameterSpace` says what is
inferred and how it enters the model, and this module turns it into priors
plus a likelihood::

    import numpyro.distributions as dist
    from rheplicant.inference import ParameterSpace, to_numpyro_model

    space = ParameterSpace.direct(
        "log_gain", init=0.0, into=lambda p: p["gain"].gain, fn=jnp.exp,
        prior=dist.Normal(0.0, 0.2),
    )
    model = to_numpyro_model(twin, state_template, space, noise_std=0.5)
    mcmc = numpyro.infer.MCMC(numpyro.infer.NUTS(model), num_warmup=500,
                              num_samples=500)
    mcmc.run(jax.random.key(0), observed=observation.data)

Sample sites are named by their **latents**, so NUTS explores — and the
samples come back keyed by — the coordinates the model was declared in.
``log_gain`` above is one site, even though its value reaches a pipeline leaf
called ``gain``, and it would remain one site if it drove five stages at once.

IMPORTANT — stochastic operators: in a Bayesian model the noise lives in the
LIKELIHOOD, not in the forward model. Build the pipeline you hand to
``to_numpyro_model`` *without* NoiseOperator/RFIOperator draws (the framework
already separates them as their own stages), or their fixed-key draws would
be treated as deterministic signal.

Posterior predictive / pushforward: :func:`predict_from_samples` runs the
pipeline over MCMC samples (pairs with
:mod:`rheplicant.inference.uncertainty`'s summaries).
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.errors import ParameterSpaceError, StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State
from rheplicant.inference.graph_bridge import to_graph
from rheplicant.inference.likelihood import check_observed_shape
from rheplicant.inference.noise import HomoscedasticNoise, check_noise_std_axis
from rheplicant.inference.parameters import ParameterSpace
from rheplicant.inference.uncertainty import as_noise_model

#: The deterministic site this package has always recorded the prediction at.
#: Not the graph's internal ``__mu__``: a caller is invited to read
#: ``mcmc.get_samples()["prediction"]`` by this module's own docstring and by
#: ``examples/tutorial_nuts.py``, so it is a public name and kept (D26).
PREDICTION_SITE: str = "prediction"


def _bayesmith_to_numpyro():
    """``bayesmith.to_numpyro``, imported at the call.

    Same reason ``graph_bridge`` imports its own: importing this package must
    not import bayesmith, and a missing one should fail at the call that needed
    it rather than at the import of anything sitting beside it.
    """
    import bayesmith

    return bayesmith.to_numpyro


def _require_numpyro():
    try:
        import numpyro  # noqa: F401
        import numpyro.distributions  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "This feature needs numpyro: pip install 'rheplicant[numpyro]'."
        ) from exc


def _require_priors(space: ParameterSpace) -> None:
    """Every latent must have a prior — a free parameter has no posterior.

    A latent the space's ``joint_prior`` covers HAS one; it is simply declared
    over the block rather than on the latent, and ``ParameterSpace`` already
    refuses a latent that carries both.
    """
    joint = space.joint_prior
    missing = [
        latent.name
        for latent in space.latents
        if latent.prior is None and not (joint is not None and joint.covers(latent.name))
    ]
    if missing:
        raise ParameterSpaceError(
            f"Latent(s) {missing} have no prior, so they cannot be sampled. A prior-free "
            "latent is a free parameter: fine for the calibrators, meaningless in a "
            "posterior. Give it a prior, or drop it from the space."
        )


def _refuse_a_latent_named_like_the_sampled_sigma(
    space: ParameterSpace, noise_std: Any
) -> None:
    """A sampled sigma takes the site ``"noise_std"``; a latent cannot have it too.

    The site below is named ``"noise_std"`` and belongs to no
    :class:`~rheplicant.inference.parameters.ParameterSpace` -- which is the
    whole subject of the refusal beneath this one. So a space that happens to
    declare a latent of that name collides with it.

    **It does not pass silently today**, and that is worth stating because the
    remedy here is not a repair but a re-wording: NumPyro raises a bare
    ``AssertionError`` reading "all sites must have unique names but got
    `noise_std` duplicated" (measured). That is loud, and it names neither the
    ``noise_std=`` argument that created the second site, nor the space that
    declared the first, nor this package's exception class -- and exception
    identity is a keeping surface. Same principle as the rest of this layer:
    a refusal the layer below would word badly lives above it.

    Only when the sigma is actually sampled. A latent named ``noise_std``
    beside a fixed sigma is an ordinary latent with an unfortunate name, and
    nothing collides.
    """
    import numpyro.distributions as distributions

    if not isinstance(noise_std, distributions.Distribution):
        return
    if not any(latent.name == "noise_std" for latent in space.latents):
        return
    raise ParameterSpaceError(
        "to_numpyro_model was given a "
        f"{type(noise_std).__name__} noise_std -- which becomes the sample site "
        "'noise_std' -- and this space also declares a latent called "
        "'noise_std'. Two sample sites cannot share a name: NumPyro refuses the "
        "model with a bare assertion naming neither the argument nor the space, "
        "so it is refused here instead. The two are different quantities -- the "
        "site is an inferred sigma that no ParameterSpace declares, and your "
        "latent is bound into the model -- so rename the latent, or pass a fixed "
        "noise_std (a scalar, an array, or a NoiseModel) and declare the sigma's "
        "prior on the latent instead."
    )


def _refuse_sampled_noise_std_under_a_joint_prior(
    space: ParameterSpace, noise_std: Any, allowed: bool
) -> None:
    """Refuse an inferred sigma alongside a joint prior, unless asked for.

    A ``noise_std`` that is a distribution becomes the sample site
    ``"noise_std"`` below. That site is in NO
    :class:`~rheplicant.inference.parameters.ParameterSpace`, so no ``over=``
    can name it and no declaration can say what its prior is. Meanwhile a
    :class:`~rheplicant.inference.priors.JeffreysPrior` over ``p`` latents is
    ``sqrt(det I)``, and ``I`` carries ``1/sigma^2`` in every entry — so the
    factor site multiplies the sigma posterior by ``sigma^-p`` whether or not
    anybody meant sigma to have a prior at all.

    Measured, in ``tests/inference/test_jeffreys_prior.py``: the factor site's
    derivative with respect to ``log sigma`` is **exactly -p**, pinned to 1e-9
    at ``p = 2`` and ``p = 3``. That is the ``sigma^-p``, and what it costs
    follows in closed form — the sigma posterior's mode moves from ``chi2/n`` to
    ``chi2/(n + p)``, which against a width of ``1/sqrt(2n)`` in ``log sigma``
    is ``p / sqrt(2n)`` of a standard deviation. Over 512 samples that is
    **0.062 sigma** at ``p = 2`` and **1.0 sigma** at ``p = 32``. A chain
    reports none of it: the sigma posterior stays proper, unimodal and merely
    narrower, and every convergence diagnostic is clean.
    """
    if allowed or space.joint_prior is None:
        return
    import numpyro.distributions as dist

    if not isinstance(noise_std, dist.Distribution):
        return
    count = len(space.joint_prior.over)
    raise ParameterSpaceError(
        f"to_numpyro_model was given a {type(noise_std).__name__} noise_std — which "
        "becomes the sample site 'noise_std', an INFERRED sigma — while this space "
        f"declares {type(space.joint_prior).__name__}(over="
        f"{list(space.joint_prior.over)}). That site belongs to no ParameterSpace, so "
        "no over= can name it and nothing declares its prior; but the joint prior is "
        "sqrt(det I) and every entry of I carries 1/sigma^2, so the factor site "
        f"multiplies the sigma posterior by sigma^-{count} — a prior on sigma that "
        "nobody wrote down. Measured: d(joint prior)/d log sigma is exactly -p. That "
        "moves the sigma posterior's mode from chi2/n to chi2/(n + p), which is "
        f"p/sqrt(2n) of a posterior width — for p = {count} over 512 samples, "
        f"{count / (2 * 512) ** 0.5:.3f} sigma; at p = 32, 1.0 sigma. The chain reports "
        "none of it: the sigma posterior stays proper, unimodal and merely narrower, "
        "and every diagnostic is clean. Pass a fixed noise_std (a scalar, an array, or a "
        "NoiseModel), or say allow_sampled_noise_std=True to take the sigma^-p tilt "
        "deliberately."
    )


def _refuse_a_joint_prior_in_single_precision(space: ParameterSpace) -> None:
    """A Jeffreys prior needs float64, and this is where that can still be said.

    ``JeffreysPrior.log_density`` is evaluated INSIDE the model body, at every
    leapfrog step, and the block's information matrix is refused by name when
    the ambient precision is single -- measured on an exactly degenerate block,
    a half-log-determinant of **-27.52** where the same block honestly gives
    **-338.05**, 310 nats, in a term NUTS exponentiates.

    The refusal has to be HERE and not there. ``jax_enable_x64`` is a
    tracing-time global and NumPyro traces ``model`` long after this function
    has returned, so the trick :func:`~rheplicant.inference.identifiability.
    identifiability` uses -- open x64 around its own arithmetic -- has nothing
    to open it around: a float32-traced model cannot contain a float64 factor.
    What arrives at trace time instead is a translated refusal quoting "a
    Jeffreys information matrix", which is true and names neither the
    ``joint_prior`` that caused it nor the model that declared it.

    This is D25 in the migration ledger. The route out is a float64 session,
    not a keyword: there is no reading of this prior at single precision that
    is worth having.
    """
    if space.joint_prior is None or jnp.result_type(float) == jnp.float64:
        return
    raise StateValidationError(
        f"to_numpyro_model was given a space declaring "
        f"{type(space.joint_prior).__name__}(over="
        f"{list(space.joint_prior.over)}), and the ambient precision is float32. "
        "The joint prior is sqrt(det I) over that block, evaluated at every "
        "leapfrog step, and its half-log-determinant is not supportable in "
        "single precision: measured on an exactly degenerate block, float32 "
        "gives -27.52 where the block honestly gives -338.05 -- a 310-nat error "
        "in a term the sampler exponentiates, with a proper posterior, a "
        "converged chain and clean diagnostics on the other side of it. "
        "Nothing downstream reports a log-prior that is wrong by 310 nats. "
        "Run this model in a float64 session (`jax.config.update("
        "'jax_enable_x64', True)` before the space is declared, so the "
        "model's own constants are built at the wider dtype), or drop "
        "inference.joint_prior and declare the block's latents with "
        "Latent(prior=...)."
    )


def to_numpyro_model(
    pipeline: AbstractOperator,
    state_template: State,
    space: ParameterSpace,
    noise_std: Any,
    flags: jax.Array | None = None,
    obs_name: str = "obs",
    *,
    allow_sampled_noise_std: bool = False,
):
    """Build a NumPyro model: priors -> bound pipeline -> Gaussian likelihood.

    Args:
        pipeline: the (deterministic) forward model.
        state_template: input state the model is evaluated on (closed over).
        space: what to infer and how it binds. Every latent needs a prior —
            either its own ``Latent(prior=...)`` or the space's ``joint_prior``.
            A declared
            :class:`~rheplicant.inference.priors.JeffreysPrior` is evaluated
            here and nowhere else: its latents get improper flat sample sites,
            its block is checked for rank once before any sample is drawn, and
            ``0.5 log det I`` is added at the ``"joint_prior"`` factor site with
            the same noise object the likelihood uses.
        noise_std: how noisy the data is. Three forms:

            * a scalar or array standard deviation;
            * a :class:`~rheplicant.inference.noise.NoiseModel` — in particular
              :class:`~rheplicant.inference.noise.RadiometerNoise`, whose sigma
              tracks the prediction and therefore the sampled parameters;
            * a NumPyro distribution, to infer a constant sigma (sampled at
              site ``"noise_std"``).
        flags: optional boolean mask (True = flagged); flagged samples are
            excluded from the likelihood (RFI flags -> noise covariance).
            Equivalent to wrapping ``noise_std`` in
            :class:`~rheplicant.inference.noise.FlaggedNoise`.
        obs_name: name of the observed sample site.
        allow_sampled_noise_std: take a sampled ``noise_std`` together with a
            declared ``joint_prior`` deliberately. Off by default, and the
            refusal explains what it costs: the ``"noise_std"`` site is in no
            ParameterSpace, so a Jeffreys prior over ``p`` latents tilts its
            posterior by ``sigma^-p`` with nothing reporting it — measured at
            about 1.0 sigma for ``p = 32``. Inert when no ``joint_prior`` is
            declared.

    Returns:
        A NumPyro model ``model(observed=None)`` — condition by passing
        ``observed=data``; run without it for prior-predictive checks. The
        noiseless prediction is recorded at the deterministic site
        ``"prediction"``. A conditioning ``observed`` whose shape is not
        exactly the prediction's is refused
        (:func:`~rheplicant.inference.likelihood.check_observed_shape`) while
        the model is traced, before any sample is drawn: broadcasting it would
        give NUTS a different posterior to explore, and it would explore it
        successfully.

    Note:
        **A prediction-dependent sigma brings its log-determinant with it, and
        that is the point of routing it through here.** ``Normal(loc,
        scale).log_prob`` contains ``-log scale``, so when ``scale`` is a
        function of the sampled parameters the term is part of the potential
        automatically — this is the full Gaussian density, not the generalized
        least squares that :func:`~rheplicant.inference.gls.iterative_gls`
        converges to (which freezes that dependence in order to keep each step
        linear). The two answers differ, in the direction
        :mod:`rheplicant.inference.noise` gives in closed form.

        An unobserved sample — infinite sigma, from
        :class:`~rheplicant.inference.noise.FlaggedNoise` or ``flags`` — is
        masked out rather than given an infinite scale, which would send the
        whole potential to ``-inf``. Masking is the limit that exists.

        **The model itself is bayesmith's.** This function declares the graph
        through :func:`~rheplicant.inference.graph_bridge.to_graph` and hands
        it to ``bayesmith.to_numpyro``; the sites, the mask and the joint-prior
        factor are emitted there, from the declaration. What stays here is what
        a graph cannot spell: the refusals above, the site NAMES (D26 — the
        graph calls its own nodes ``__mu__`` and ``__data__``, and this package
        has always called them ``"prediction"`` and ``obs_name``), and the
        meaning of ``observed=None``, which is the prior predictive here and
        "each node's declared data" there.
    """
    _require_numpyro()
    import numpyro.distributions as dist

    _require_priors(space)
    _refuse_a_joint_prior_in_single_precision(space)
    _refuse_a_latent_named_like_the_sampled_sigma(space, noise_std)
    _refuse_sampled_noise_std_under_a_joint_prior(
        space, noise_std, allow_sampled_noise_std
    )
    space.validate(pipeline)

    joint = space.joint_prior
    if joint is not None:
        # Once, here, and not inside the model body: a rank is a decision, and a
        # traced decision is one no branch can be taken on. This is the refusal;
        # the eigenvalue floor inside `log_density` is only the arithmetic that
        # keeps a degenerate block finite if one ever gets past it.
        joint.check_identified(
            space, pipeline, state_template, caller="to_numpyro_model"
        )

    # The prediction's shape, asked of the declaration rather than of a traced
    # array: `to_graph` needs data at BUILD time and this function is not given
    # any -- the caller passes it when the model is run. So the graph is built
    # against a placeholder of the right shape and every call conditions it
    # explicitly, which is also what makes `observed=None` mean the prior
    # predictive here rather than "use the declared data".
    forward, values0 = space.forward_fn(pipeline, state_template)
    prediction_shape = tuple(jax.eval_shape(forward, values0).shape)

    sampled_scale = isinstance(noise_std, dist.Distribution)
    noise = as_noise_model(
        # A placeholder sigma, and only its FLAGS are read: the value comes from
        # the declared scale latent instead. `to_graph` refuses this combination
        # for anything but a homoscedastic model, which is what makes the
        # placeholder unreachable rather than merely unread (D27).
        HomoscedasticNoise(jnp.ones(())) if sampled_scale else noise_std,
        flags,
    )
    # In front of the seam, with this function's name on it. `to_graph` runs the
    # same check and would name itself, which is true and unhelpful to someone
    # who called this. Measured 2026-08-27 (D28): this path used to ACCEPT an
    # ambiguous 1-D sigma on a square grid and broadcast it along the last axis
    # in silence, so the refusal is new here and is a fix, not a regression.
    check_noise_std_axis(noise, prediction_shape, "to_numpyro_model")

    graph = to_graph(
        space, pipeline, state_template, jnp.zeros(prediction_shape), noise,
        prediction_name=PREDICTION_SITE,
        observation_name=obs_name,
        scale_prior=("noise_std", noise_std) if sampled_scale else None,
    )
    declared = _bayesmith_to_numpyro()(graph)

    def model(observed: jax.Array | None = None):
        # `{}` and not `None`: the two packages spell the unconditioned model
        # differently on purpose. There, `None` means "each node's own declared
        # data"; here it has always meant the prior predictive, and that is a
        # keeping surface -- `mcmc.run(key, observed=data)` is how every caller
        # in this package conditions. The mapping is where the two meet.
        if observed is None:
            return declared({})
        # Trace time, not run time: both shapes are static, so this compiles
        # away entirely and NUTS refuses before it draws a single sample.
        check_observed_shape(prediction_shape, observed, predictor="this model")
        return declared({obs_name: observed})

    return model


def init_to_declared(space: ParameterSpace):
    """A NumPyro init strategy that starts where the ``ParameterSpace`` says.

    ``Latent(..., init=...)`` already states where the model starts, and the
    calibrators and :func:`~rheplicant.inference.linear.check_linearity` both
    use it. NUTS does not: a kernel built without an ``init_strategy`` uses
    NumPyro's default ``init_to_uniform``, which draws in the *unconstrained*
    space with no knowledge of the declaration. Pass this instead::

        kernel = numpyro.infer.NUTS(model, init_strategy=init_to_declared(space))

    **This is not a tuning knob.** On the ring toy of
    ``examples/tutorial_nuts.py`` — 1024 samples constraining three beam
    parameters — the default initialization gives ``r_hat = 840`` and an
    effective sample size of **2** out of 8000 draws, while the identical model
    started here gives ``r_hat = 1.002`` and ``n_eff = 1327``. Neither
    tightening the priors nor tripling the warmup moved those numbers at all;
    only the starting point did.

    The mechanism is ordinary and worth recognising: a posterior far narrower
    than its prior is a needle, ``init_to_uniform`` lands in the haystack, and
    warmup adapts a step size for wherever it landed. The declared ``init`` does
    not have to be good — the one in that tutorial is deliberately mis-set, some
    11000 nats below the peak — it only has to be somewhere a gradient can be
    followed.

    Args:
        space: the space the model was built from.

    Returns:
        A NumPyro init strategy, ready for ``NUTS(model, init_strategy=...)``.
    """
    _require_numpyro()
    import numpyro

    return numpyro.infer.init_to_value(values=space.initial_values())


def predict_from_samples(
    pipeline: AbstractOperator,
    state_template: State,
    space: ParameterSpace,
    samples: dict[str, jax.Array],
) -> jax.Array:
    """Posterior predictive: run the pipeline over MCMC samples.

    Args:
        pipeline, state_template, space: as given to :func:`to_numpyro_model`.
        samples: ``mcmc.get_samples()`` — latent name -> ``(n_samples, ...)``.

    Returns:
        ``(n_samples, *data.shape)`` noiseless predictions (add likelihood
        noise separately if you want the full predictive).
    """
    _require_numpyro()
    space.validate(pipeline)
    for name in space.names:
        if name not in samples:
            raise StateValidationError(
                f"samples is missing site {name!r}; available: {sorted(samples)}"
            )
        # Checking only the NAME lets a wrong-shaped stack broadcast into the
        # leaf and return a finite, correctly-shaped, wrong predictive.
        expected = space.latent(name).init.shape
        got = jnp.shape(samples[name])[1:]
        if got != expected:
            raise StateValidationError(
                f"samples[{name!r}] has per-sample shape {got}, but the latent is "
                f"{expected}. The leading axis must be the sample axis."
            )
    lengths = {jnp.shape(samples[name])[0] for name in space.names}
    if len(lengths) != 1:
        raise StateValidationError(
            f"samples have differing numbers of draws across sites: {sorted(lengths)}."
        )
    stacked = {name: samples[name] for name in space.names}

    def run(values: dict[str, jax.Array]) -> jax.Array:
        return space.bind(pipeline, values)(state_template).data

    return eqx.filter_vmap(run)(stacked)
