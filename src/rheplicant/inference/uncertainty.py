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
    x0, unravel = ravel_pytree(params)
    if x0.size == 0:
        raise StateValidationError(
            "params contains no trainable leaves — nothing to propagate."
        )

    def f_flat(x: jax.Array) -> jax.Array:
        return jnp.ravel(forward(unravel(x)))

    return f_flat, x0, forward(params)


def _noise_weights(
    prediction: jax.Array, noise_std: jax.Array, flags: jax.Array | None
) -> jax.Array:
    """Per-sample inverse-variance weights, zeroed on flagged samples."""
    weights = jnp.broadcast_to(1.0 / jnp.asarray(noise_std) ** 2, prediction.shape)
    if flags is not None:
        if flags.shape != prediction.shape:
            raise StateValidationError(
                f"flags shape {flags.shape} does not match prediction shape "
                f"{prediction.shape}."
            )
        weights = jnp.where(flags, 0.0, weights)
    return jnp.ravel(weights)


def fisher_information(
    forward: Callable[[Any], jax.Array],
    params: Any,
    noise_std: jax.Array,
    flags: jax.Array | None = None,
) -> jax.Array:
    """Fisher information matrix ``F = J^T N^-1 J`` at ``params``.

    Assumes independent Gaussian noise with standard deviation ``noise_std``
    (scalar or broadcastable to the prediction shape); flagged samples
    (``flags`` True) carry zero weight — the same convention as
    :class:`~rheplicant.inference.likelihood.MaskedGaussianLikelihood`.

    Returns:
        A :class:`FlatMatrix` — the ``(n_params, n_params)`` Fisher matrix
        (``.matrix``) over the flattened parameter vector, tagged with the
        parameter structure it belongs to.
    """
    f_flat, x0, prediction = _flat_forward(forward, params)
    jacobian = jax.jacfwd(f_flat)(x0)  # (n_data, n_params)
    weights = _noise_weights(prediction, noise_std, flags)
    names, spans, shapes = _named_spans(params)
    return FlatMatrix(
        matrix=jacobian.T @ (weights[:, None] * jacobian),
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
