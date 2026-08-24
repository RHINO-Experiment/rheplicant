"""Spectral diagnostics for matrix-free symmetric operators.

An iterative solver can cheaply report ``‖M x - b‖``; what a caller actually
wants to know is ``‖x - x*‖``. The two differ by the condition number, so any
honest convergence guard over a matrix-free operator needs the ends of its
spectrum — and needs them without ever forming a matrix.

Everything here works on pytrees and takes the operator as a callable, so it
knows nothing about :mod:`rheplicant.inference.linear`'s blocks. That is
deliberate: it keeps the numerics separable from the model machinery, and the
dependency pointing one way.

**Why this lives in core.** Two layers need it and they may not see each other.
:mod:`rheplicant.inference.linear` guards its Wiener solves with it, and
:class:`~rheplicant.radio.filters.skyspace.SkySpaceFilter` guards its
map-making CG with it — but ``radio`` may not import ``inference`` (DESIGN.md's
hard rule; ``core.basis`` is here for the same reason, D28). The alternative was
a second power iteration in ``radio``, and a second copy of a subtle numeric is
the copy that goes stale.
"""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

#: Power-iteration steps for :func:`largest_eigenvalue`, for callers that
#: guard a solve with it. It typically settles within three; this leaves
#: margin at a fixed cost of ``POWER_ITERATIONS`` operator applications per
#: guarded solve. It was twice that while :func:`extreme_eigenvalues` was the
#: one callers used -- see that function's own docstring for why nothing here
#: uses it any more.
POWER_ITERATIONS: int = 12


def tree_norm(parts: Any) -> jax.Array:
    """Euclidean norm over a pytree, scaled so it survives float32.

    Squaring first overflows for entries beyond ~1.8e19, which turns the only
    convergence signal these solvers give into inf/inf = NaN exactly when the
    problem is badly scaled and the answer is most likely wrong.
    """
    leaves = [leaf for leaf in jax.tree.leaves(parts) if eqx.is_array(leaf)]
    if not leaves:  # pragma: no cover - defensive
        return jnp.array(0.0)
    biggest = jnp.max(jnp.stack([jnp.max(jnp.abs(leaf)) for leaf in leaves]))
    biggest = jnp.where(biggest > 0, biggest, 1.0)
    total = sum(jnp.sum((leaf / biggest) ** 2) for leaf in leaves)
    return biggest * jnp.sqrt(total)


def _scaled(parts: Any, factor: jax.Array) -> Any:
    return jax.tree.map(lambda leaf: leaf / factor, parts)


def _random_like(template: Any, key: jax.Array) -> Any:
    leaves, treedef = jax.tree.flatten(template)
    keys = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(
        treedef,
        [
            jax.random.normal(subkey, leaf.shape, dtype=leaf.dtype)
            for leaf, subkey in zip(leaves, keys, strict=True)
        ],
    )


def largest_eigenvalue(
    operator: Callable[[Any], Any],
    template: Any,
    key: jax.Array,
    iterations: int,
) -> jax.Array:
    """Top eigenvalue of a symmetric positive-definite operator, by power iteration.

    Each step costs one application of ``operator`` — for a normal operator
    that is the same JVP-plus-VJP a CG iteration costs, and no matrix is
    formed. The estimate approaches the true value from BELOW.

    Args:
        operator: the symmetric positive-definite map, pytree to pytree.
        template: a pytree of the operator's domain, used for shapes and dtypes.
        key: PRNG key for the starting vector.
        iterations: number of steps.
    """
    vector = _random_like(template, key)
    largest = tree_norm(vector)
    vector = _scaled(vector, largest)
    for _ in range(iterations):
        image = operator(vector)
        largest = tree_norm(image)
        vector = _scaled(image, jnp.where(largest > 0, largest, 1.0))
    return largest


def extreme_eigenvalues(
    operator: Callable[[Any], Any],
    template: Any,
    key: jax.Array,
    iterations: int,
) -> tuple[jax.Array, jax.Array]:
    """``(λ_max, λ_min)`` of a symmetric positive-definite operator.

    ``λ_min`` comes from a second power iteration on ``λ_max I - M``, whose top
    eigenvalue is ``λ_max - λ_min``. Measuring it beats bounding it: a caller
    who assumed the worst about ``λ_min`` would call every well-conditioned
    operator ill-conditioned by the whole dynamic range of the problem.

    The difference is taken between two numbers of size ``λ_max``, so it is
    cancellation-prone precisely when ``λ_min`` is tiny. Callers who hold an
    independent lower bound on ``λ_min`` — a prior's curvature, say — should
    floor the result with it; that is both rigorous and the scale at which the
    cancellation bites.
    """
    largest = largest_eigenvalue(operator, template, key, iterations)
    spread = largest_eigenvalue(
        lambda parts: jax.tree.map(
            lambda leaf, image: largest * leaf - image, parts, operator(parts)
        ),
        template,
        jax.random.fold_in(key, 1),
        iterations,
    )
    return largest, largest - spread
