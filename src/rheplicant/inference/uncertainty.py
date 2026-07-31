"""Uncertainty propagation through differentiable forward models.

Two complementary routes, both riding on the framework's differentiability:

- **Linear (Fisher / delta-method)** — the domain-standard forecasting tool.
  :func:`fisher_information` builds ``F = J^T N^-1 J`` from the exact
  Jacobian of ``forward`` (``jax.jacfwd`` — no finite differences), so
  parameter forecasts and error bars are one linear solve away
  (:func:`parameter_covariance`), and :func:`propagate_covariance` pushes a
  parameter covariance to a per-sample prediction standard deviation
  (delta method). Exact for models linear in the parameters; a local
  approximation otherwise.

- **Monte Carlo pushforward** — :func:`push_forward` vmaps ``forward`` over
  a stack of parameter samples (e.g. a NumPyro posterior via
  :func:`~rheplicant.inference.numpyro_bridge.predict_from_samples`), giving the
  full predictive distribution with no linearity assumption.

A Laplace approximation is the composition of the two: MAP-fit with
:class:`~rheplicant.inference.calibrate.GradientCalibrator`, take
:func:`parameter_covariance` at the fit, sample from the Gaussian, and
:func:`push_forward`.

``forward`` here is the ``f(params) -> prediction`` callable produced by
:func:`~rheplicant.inference.forward.build_forward_fn` — uncertainty tooling
connects through the same seam as every other inference engine (D7).
"""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.noise import (
    FlaggedNoise,
    HomoscedasticNoise,
    NoiseModel,
    inverse_variance,
)


def _named_spans(
    params: Any,
) -> tuple[tuple[str, ...] | None, tuple[tuple[int, int], ...] | None, tuple[tuple, ...] | None]:
    """Map each name of a flat parameter dict to its span in the flat vector.

    Derived from the actual flattening rather than from an assumption about
    dict ordering. Returns ``(None, None, None)`` for anything that is not a
    flat ``{name: array}`` dict — the pytrees ``build_forward_fn`` produces,
    which have no names to give.
    """
    if not isinstance(params, dict):
        return None, None, None
    entries = jax.tree_util.tree_flatten_with_path(params)[0]
    names: list[str] = []
    spans: list[tuple[int, int]] = []
    shapes: list[tuple] = []
    offset = 0
    for path, leaf in entries:
        if len(path) != 1 or not isinstance(path[0], jax.tree_util.DictKey):
            return None, None, None  # nested — no flat naming to offer
        size = int(jnp.size(leaf))
        names.append(str(path[0].key))
        spans.append((offset, offset + size))
        shapes.append(jnp.shape(leaf))
        offset += size
    return tuple(names), tuple(spans), tuple(shapes)


class FlatMatrix(eqx.Module):
    """A matrix over a FLATTENED parameter vector, carrying its provenance.

    ``ravel_pytree`` ordering depends on the parameter pytree's structure, so
    a Fisher/covariance matrix is only meaningful together with the treedef
    it was flattened against. Carrying the structure lets
    :func:`propagate_covariance` reject a covariance built for a different
    parameterization instead of silently returning wrong numbers.

    When the parameters came from a
    :class:`~rheplicant.inference.parameters.ParameterSpace` — a flat
    ``{name: array}`` dict — the rows also carry their names, so error bars can
    be asked for by the name the model was declared in
    (``cov.sigma("fwhm_deg")``) rather than by position.

    Attributes:
        matrix: the ``(n_params, n_params)`` array.
        structure: treedef of the parameter pytree it was computed for.
        kind: ``"fisher"``, ``"covariance"``, or ``"matrix"``. Not decoration:
            ``sqrt(diag(F))`` is not an error bar, so :meth:`sigma` refuses to
            pretend otherwise on a Fisher matrix.
        names, spans, shapes: per-parameter name, ``(start, stop)`` span in the
            flat vector, and original shape. ``None`` for unnamed pytrees.
    """

    matrix: jax.Array
    structure: jax.tree_util.PyTreeDef = eqx.field(static=True)
    kind: str = eqx.field(static=True, default="matrix")
    names: tuple[str, ...] | None = eqx.field(static=True, default=None)
    spans: tuple[tuple[int, int], ...] | None = eqx.field(static=True, default=None)
    shapes: tuple[tuple, ...] | None = eqx.field(static=True, default=None)

    def _index(self, name: str) -> int:
        if self.names is None:
            raise StateValidationError(
                "This matrix's parameters are not named — it was built from a plain pytree. "
                "Build the forward function with ParameterSpace.forward_fn to get named rows."
            )
        if name not in self.names:
            raise StateValidationError(
                f"There is no parameter named {name!r}; available: {list(self.names)}."
            )
        return self.names.index(name)

    def span(self, name: str) -> tuple[int, int]:
        """``(start, stop)`` of one parameter within the flattened vector."""
        return self.spans[self._index(name)]

    def sigma(self, name: str) -> jax.Array:
        """Marginal standard deviation(s) of one named parameter.

        Only meaningful on a covariance: raises on a Fisher matrix rather than
        returning ``sqrt(diag(F))``, which looks like an error bar and is not
        one — inverting is exactly the step that couples the parameters.
        """
        if self.kind == "fisher":
            raise StateValidationError(
                "sqrt(diag(F)) of a Fisher matrix is not a standard deviation — it ignores "
                "every parameter degeneracy. Invert it with parameter_covariance() first."
            )
        index = self._index(name)
        start, stop = self.spans[index]
        return jnp.sqrt(jnp.diag(self.matrix)[start:stop]).reshape(self.shapes[index])

    def block(self, name: str, other: str | None = None) -> jax.Array:
        """The sub-matrix for one parameter, or the cross-block of two."""
        start, stop = self.span(name)
        other_start, other_stop = self.span(other) if other is not None else (start, stop)
        return self.matrix[start:stop, other_start:other_stop]


def _flat_forward(
    forward: Callable[[Any], jax.Array], params: Any
) -> tuple[Callable[[jax.Array], jax.Array], jax.Array, jax.Array]:
    """Flatten the parameter pytree: return (f_flat, x0, prediction0)."""
    complex_leaves = [
        path
        for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]
        if jnp.issubdtype(jnp.result_type(leaf), jnp.complexfloating)
    ]
    if complex_leaves:
        where = ", ".join(jax.tree_util.keystr(p) for p in complex_leaves)
        raise StateValidationError(
            f"Complex parameters ({where}) cannot go through jax.jacfwd, which is what the "
            "Fisher and delta-method routines are built on. A real prediction makes the map "
            "R-linear but not C-linear, so the Jacobian is only defined over the real degrees "
            "of freedom: split the latent into real and imaginary parts, or use "
            "rheplicant.inference.linear (linear_operator / wiener_solve / gcr_sample), which "
            "does that split internally."
        )
    x0, unravel = ravel_pytree(params)
    if x0.size == 0:
        raise StateValidationError(
            "params contains no trainable leaves — nothing to propagate."
        )

    def f_flat(x: jax.Array) -> jax.Array:
        return jnp.ravel(forward(unravel(x)))

    return f_flat, x0, forward(params)


def as_noise_model(noise_std: Any, flags: jax.Array | None = None) -> NoiseModel:
    """Normalize a ``noise_std`` argument into a :class:`NoiseModel`.

    A bare scalar or array becomes :class:`HomoscedasticNoise`; a noise model
    is passed through; ``flags`` wrap either in :class:`FlaggedNoise`. This is
    what lets every ``noise_std=`` argument in the package accept the seam
    without any of their signatures changing.

    The discrimination is by ``depends_on_prediction``, not by ``std``: jax and
    numpy arrays both *have* a ``.std`` method, so the protocol's data member
    is the only unambiguous marker.
    """
    noise = (
        noise_std
        if isinstance(noise_std, NoiseModel)
        else HomoscedasticNoise(jnp.asarray(noise_std))
    )
    return noise if flags is None else FlaggedNoise(noise, flags)


def _log_sigma_curvature(
    noise: NoiseModel,
    f_flat: Callable[[jax.Array], jax.Array],
    x0: jax.Array,
    prediction: jax.Array,
) -> jax.Array:
    """``(d log sigma/d theta)^T (d log sigma/d theta)`` over observed samples.

    The Gaussian information carried by the *variance* rather than the mean.
    Unobserved samples (infinite sigma) contribute a flat zero, which also
    keeps their derivative from being a NaN.
    """
    shape = jnp.shape(prediction)

    def log_sigma(x: jax.Array) -> jax.Array:
        sigma = noise.std(jnp.reshape(f_flat(x), shape))
        seen = jnp.isfinite(sigma)
        safe = jnp.where(seen, sigma, 1.0)
        return jnp.ravel(jnp.where(seen, jnp.log(safe), 0.0))

    jac = jax.jacfwd(log_sigma)(x0)  # (n_data, n_params)
    return jac.T @ jac


def fisher_information(
    forward: Callable[[Any], jax.Array],
    params: Any,
    noise_std: Any,
    flags: jax.Array | None = None,
) -> jax.Array:
    """Fisher information matrix at ``params``, for independent Gaussian noise.

    Args:
        forward: ``f(params) -> prediction``.
        params: where to evaluate.
        noise_std: standard deviation — scalar or broadcastable to the
            prediction — **or** a :class:`~rheplicant.inference.noise.NoiseModel`.
        flags: optional boolean mask; flagged samples carry zero weight, the
            same convention as
            :class:`~rheplicant.inference.likelihood.MaskedGaussianLikelihood`.

    Returns:
        A :class:`FlatMatrix` — the ``(n_params, n_params)`` Fisher matrix
        (``.matrix``) over the flattened parameter vector, tagged with the
        parameter structure it belongs to.

    Note:
        **When the noise depends on the parameters, ``J^T N^-1 J`` is not the
        Fisher matrix.** For ``d ~ N(mu(theta), Sigma(theta))`` the information
        has a second term from the covariance's own parameter dependence::

            F = J^T Sigma^-1 J  +  1/2 tr(Sigma^-1 dSigma Sigma^-1 dSigma)

        which for a diagonal covariance is ``2 (d log sigma/d theta)^T
        (d log sigma/d theta)``. It is included automatically whenever the
        noise model reports ``depends_on_prediction``, and omitted otherwise
        (where it is exactly zero). Under
        :class:`~rheplicant.inference.noise.RadiometerNoise` with fractional
        level ``f`` the correction is a clean factor: ``F = (1 + 2 f^2)
        J^T N^-1 J``. Reporting only the first term would forecast error bars
        that are too wide by ``sqrt(1 + 2 f^2)`` — a plausible number, and the
        wrong one.
    """
    f_flat, x0, prediction = _flat_forward(forward, params)
    jacobian = jax.jacfwd(f_flat)(x0)  # (n_data, n_params)
    noise = as_noise_model(noise_std, flags)
    weights = jnp.ravel(inverse_variance(noise, prediction))
    matrix = jacobian.T @ (weights[:, None] * jacobian)
    if noise.depends_on_prediction:
        matrix = matrix + 2.0 * _log_sigma_curvature(noise, f_flat, x0, prediction)
    names, spans, shapes = _named_spans(params)
    return FlatMatrix(
        matrix=matrix,
        structure=jax.tree_util.tree_structure(params),
        kind="fisher",
        names=names,
        spans=spans,
        shapes=shapes,
    )


def parameter_covariance(fisher: FlatMatrix, jitter: float = 0.0) -> FlatMatrix:
    """Invert a Fisher matrix into a parameter covariance (Cramer-Rao bound).

    Args:
        fisher: output of :func:`fisher_information`.
        jitter: optional Tikhonov term added to the diagonal for
            near-degenerate parameter combinations (prior-like regularizer).
    """
    n = fisher.matrix.shape[0]
    return FlatMatrix(
        matrix=jnp.linalg.inv(fisher.matrix + jitter * jnp.eye(n)),
        structure=fisher.structure,
        kind="covariance",
        names=fisher.names,
        spans=fisher.spans,
        shapes=fisher.shapes,
    )


def propagate_covariance(
    forward: Callable[[Any], jax.Array],
    params: Any,
    param_cov: jax.Array,
) -> jax.Array:
    """Delta-method prediction uncertainty: ``std = sqrt(diag(J Sigma J^T))``.

    Args:
        forward: ``f(params) -> prediction``.
        params: expansion point (pytree, same structure as the covariance's
            flattening).
        param_cov: covariance over the flattened parameter vector — a
            :class:`FlatMatrix` from :func:`parameter_covariance` (structure
            is verified against ``params``), or a raw ``(n_params, n_params)``
            array (external covariances; only the size can be checked — YOU
            must guarantee the flattening order matches).

    Returns:
        Per-sample prediction standard deviation, shaped like the prediction.
    """
    f_flat, x0, prediction = _flat_forward(forward, params)
    jacobian = jax.jacfwd(f_flat)(x0)
    if isinstance(param_cov, FlatMatrix):
        expected = jax.tree_util.tree_structure(params)
        if param_cov.structure != expected:
            raise StateValidationError(
                f"param_cov was computed for parameter structure "
                f"{param_cov.structure}, but params has structure {expected} — "
                "the flattened orderings differ and the numbers would be wrong."
            )
        # For a dict-based space the treedef encodes the KEY NAMES only, so two
        # spaces with the same latent names and different per-latent shapes pass
        # the structure check and produce finite, wrong error bars.
        names, _, shapes = _named_spans(params)
        if param_cov.shapes is not None and shapes is not None:
            if param_cov.names != names or param_cov.shapes != shapes:
                was = dict(zip(param_cov.names, param_cov.shapes, strict=True))
                now = dict(zip(names, shapes, strict=True))
                raise StateValidationError(
                    f"param_cov was computed for {was} but params is {now} — the "
                    "flattened orderings differ and the numbers would be wrong."
                )
        param_cov = param_cov.matrix
    if param_cov.shape != (x0.size, x0.size):
        raise StateValidationError(
            f"param_cov shape {param_cov.shape} does not match the flattened "
            f"parameter size {x0.size}."
        )
    variance = jnp.einsum("ip,pq,iq->i", jacobian, param_cov, jacobian)
    return jnp.sqrt(variance).reshape(prediction.shape)


def push_forward(
    forward: Callable[[Any], jax.Array],
    param_samples: Any,
) -> jax.Array:
    """Monte Carlo pushforward: run ``forward`` over stacked parameter samples.

    Args:
        forward: ``f(params) -> prediction``.
        param_samples: a params pytree whose every array leaf carries a
            leading sample axis of common length ``n_samples``.

    Returns:
        ``(n_samples, *prediction.shape)`` stacked predictions — summarize
        with e.g. ``mean(0)`` / ``std(0)`` / quantiles.
    """
    return jax.vmap(forward)(param_samples)
