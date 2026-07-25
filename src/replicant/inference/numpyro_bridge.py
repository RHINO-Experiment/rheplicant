"""Bridge a Pipeline/Assembly to a NumPyro probabilistic model.

Bayesian calibration through the same seam as everything else (D7): priors
are attached to pipeline leaves *positionally* — a pytree with the
pipeline's structure holding a NumPyro distribution where a parameter is to
be inferred and ``None`` everywhere else::

    import numpyro.distributions as dist
    from replicant.inference import prior_template, set_prior, to_numpyro_model

    priors = prior_template(twin)
    priors = set_prior(priors, lambda p: p["gain"].gain, dist.Normal(1.0, 0.2))
    priors = set_prior(priors, lambda p: p["uniform_sky"].amplitude,
                       dist.LogNormal(jnp.log(100.0), 0.5))

    model = to_numpyro_model(twin, state_template, priors, noise_std=0.5)
    mcmc = numpyro.infer.MCMC(numpyro.infer.NUTS(model), num_warmup=500,
                              num_samples=500)
    mcmc.run(jax.random.key(0), observed=observation.data)

IMPORTANT — stochastic operators: in a Bayesian model the noise lives in the
LIKELIHOOD, not in the forward model. Build the pipeline you hand to
``to_numpyro_model`` *without* NoiseOperator/RFIOperator draws (the framework
already separates them as their own stages), or their fixed-key draws would
be treated as deterministic signal.

Posterior predictive / pushforward: :func:`predict_from_samples` runs the
pipeline over MCMC samples (pairs with
:mod:`replicant.inference.uncertainty`'s summaries).
"""

import re
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from replicant.core.errors import StateValidationError
from replicant.core.operator import AbstractOperator
from replicant.core.state import State


def _require_numpyro():
    try:
        import numpyro  # noqa: F401
        import numpyro.distributions  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "This feature needs numpyro: pip install 'replicant[numpyro]'."
        ) from exc


def _is_none(x: Any) -> bool:
    return x is None


def prior_template(pipeline: AbstractOperator) -> Any:
    """An all-``None`` pytree with the pipeline's structure, ready for priors."""
    return jax.tree.map(lambda _: None, pipeline)


def set_prior(priors: Any, where, distribution: Any) -> Any:
    """Attach ``distribution`` at the leaf selected by ``where`` (functional).

    ``where`` is the usual ``eqx.tree_at`` selector, e.g.
    ``lambda p: p["gain"].gain``.
    """
    return eqx.tree_at(where, priors, distribution, is_leaf=_is_none)


def _site_name(pipeline: AbstractOperator, path: tuple) -> str:
    """Semantic site name: composite stage/branch NAMES instead of raw indices.

    ``Assembly(...)["gain"].gain`` names its site ``"gain_gain"`` — stable
    across fold nesting, unlike raw structural paths.
    """
    from replicant.core.combinators import SelectOperator, SumOperator
    from replicant.core.graph import Assembly
    from replicant.core.pipeline import Pipeline

    obj: Any = pipeline
    parts: list[str] = []
    keys = list(path)
    i = 0
    while i < len(keys):
        key = keys[i]
        if isinstance(key, jax.tree_util.GetAttrKey):
            name = key.name
            child = getattr(obj, name)
            if isinstance(obj, Assembly) and name == "operator":
                obj = child  # transparent wrapper
                i += 1
                continue
            if (
                isinstance(obj, (Pipeline, SumOperator, SelectOperator))
                and name in ("stages", "branches")
                and i + 1 < len(keys)
                and isinstance(keys[i + 1], jax.tree_util.SequenceKey)
            ):
                index = keys[i + 1].idx
                parts.append(obj.names[index])
                obj = child[index]
                i += 2
                continue
            parts.append(name)
            obj = child
        elif isinstance(key, jax.tree_util.SequenceKey):
            parts.append(str(key.idx))
            obj = obj[key.idx]
        elif isinstance(key, jax.tree_util.DictKey):
            parts.append(str(key.key))
            obj = obj[key.key]
        else:  # pragma: no cover - defensive
            parts.append(str(key))
            i += 1
            continue
        i += 1
    # Sanitize each component INDIVIDUALLY, then join with "." — so a stage
    # named "a-b" ("a_b") can never collide with nested stages ("a", "b").
    return ".".join(re.sub(r"\W+", "_", part).strip("_") for part in parts)


def _prior_map(pipeline: AbstractOperator, priors: Any) -> dict[tuple, Any]:
    """Collect {tree path: distribution}, validating shapes against the leaves."""
    import numpyro.distributions as dist

    def is_dist(x: Any) -> bool:
        return isinstance(x, dist.Distribution)

    entries = jax.tree_util.tree_flatten_with_path(priors, is_leaf=is_dist)[0]
    prior_map = {path: d for path, d in entries if is_dist(d)}
    if not prior_map:
        raise StateValidationError(
            "priors contains no distributions; attach at least one with set_prior()."
        )

    leaves = dict(jax.tree_util.tree_flatten_with_path(pipeline)[0])
    for path, d in prior_map.items():
        if path not in leaves:
            raise StateValidationError(
                f"prior at {jax.tree_util.keystr(path)} does not correspond to a "
                "pipeline leaf — build priors with prior_template(pipeline)."
            )
        expected = jnp.shape(leaves[path])
        got = tuple(d.shape())
        if got != expected:
            raise StateValidationError(
                f"prior at {jax.tree_util.keystr(path)} has shape {got}, but the "
                f"pipeline leaf has shape {expected}."
            )
    return prior_map


def to_numpyro_model(
    pipeline: AbstractOperator,
    state_template: State,
    priors: Any,
    noise_std: Any,
    flags: jax.Array | None = None,
    obs_name: str = "obs",
):
    """Build a NumPyro model: priors -> forward pipeline -> Gaussian likelihood.

    Args:
        pipeline: the (deterministic) forward model.
        state_template: input state the model is evaluated on (closed over).
        priors: pytree from :func:`prior_template` + :func:`set_prior` —
            a distribution at every leaf to infer, ``None`` elsewhere
            (``None`` leaves stay fixed at the pipeline's values).
        noise_std: likelihood noise standard deviation — a scalar/array, or a
            NumPyro distribution to infer it (sampled at site ``"noise_std"``).
        flags: optional boolean mask (True = flagged); flagged samples are
            excluded from the likelihood (RFI flags -> noise covariance).
        obs_name: name of the observed sample site.

    Returns:
        A NumPyro model ``model(observed=None)`` — condition by passing
        ``observed=data``; run without it for prior-predictive checks. The
        noiseless prediction is recorded at the deterministic site
        ``"prediction"``.
    """
    _require_numpyro()
    import numpyro
    import numpyro.distributions as dist

    prior_map = _prior_map(pipeline, priors)
    site_names = [_site_name(pipeline, p) for p in prior_map]
    if len(set(site_names)) != len(site_names):  # pragma: no cover - defensive
        raise StateValidationError(f"Prior site names collide: {site_names}")

    def model(observed: jax.Array | None = None):
        def maybe_sample(path: tuple, leaf: Any):
            d = prior_map.get(path)
            if d is None:
                return leaf
            return numpyro.sample(_site_name(pipeline, path), d)

        sampled_pipeline = jax.tree_util.tree_map_with_path(maybe_sample, pipeline)
        prediction = sampled_pipeline(state_template).data
        numpyro.deterministic("prediction", prediction)

        scale = (
            numpyro.sample("noise_std", noise_std)
            if isinstance(noise_std, dist.Distribution)
            else noise_std
        )
        site = dist.Normal(prediction, scale)
        if flags is not None:
            with numpyro.handlers.mask(mask=~flags):
                numpyro.sample(obs_name, site, obs=observed)
        else:
            numpyro.sample(obs_name, site, obs=observed)

    return model


def predict_from_samples(
    pipeline: AbstractOperator,
    state_template: State,
    priors: Any,
    samples: dict[str, jax.Array],
) -> jax.Array:
    """Posterior predictive: run the pipeline over MCMC samples.

    Args:
        pipeline, state_template, priors: as given to :func:`to_numpyro_model`.
        samples: ``mcmc.get_samples()`` — site name -> ``(n_samples, ...)``.

    Returns:
        ``(n_samples, *data.shape)`` noiseless predictions (add likelihood
        noise separately if you want the full predictive).
    """
    _require_numpyro()
    prior_map = _prior_map(pipeline, priors)
    site_names = [_site_name(pipeline, p) for p in prior_map]
    if len(set(site_names)) != len(site_names):  # pragma: no cover - defensive
        raise StateValidationError(f"Prior site names collide: {site_names}")
    for name in site_names:
        if name not in samples:
            raise StateValidationError(
                f"samples is missing site {name!r}; available: {sorted(samples)}"
            )

    def stack_or_keep(path: tuple, leaf: Any):
        return samples[_site_name(pipeline, path)] if path in prior_map else leaf

    def axis_or_none(path: tuple, leaf: Any):
        return 0 if path in prior_map else None

    stacked = jax.tree_util.tree_map_with_path(stack_or_keep, pipeline)
    in_axes = jax.tree_util.tree_map_with_path(axis_or_none, pipeline)

    def run(pl: AbstractOperator) -> jax.Array:
        return pl(state_template).data

    return eqx.filter_vmap(run, in_axes=(in_axes,))(stacked)
