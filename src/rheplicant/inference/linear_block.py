"""What a linear block IS: its domain, and the scales it is probed at.

The bottom of this family. Everything else here imports it and it imports
none of them, which is what lets the probe, the priors and the solve sit
beside each other rather than inside one file.
"""

import dataclasses
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from rheplicant.core.errors import ParameterSpaceError

DEFAULT_SCALES: tuple[float, ...] = (1e-3, 1.0, 1e3)

#: ``jax.ensure_compile_time_eval``, under the name of what it does at the one
#: place it is used: run this comparison **now**, on the constants in hand,
#: rather than emitting it into whatever trace happens to be open. See
#: :func:`_agrees`.
_RIGHT_NOW = jax.ensure_compile_time_eval


@dataclasses.dataclass(frozen=True)
class LinearBlock:
    """The affine action of one latent on the prediction: ``A x + offset``.

    Deliberately a plain dataclass rather than an ``eqx.Module``: this is a
    derived linear-algebra *handle*, not a differentiable model. ``forward``
    and ``adjoint`` are closures over a traced computation, so the block is
    something you build where you need it, not a pytree to carry around.

    A block may hold ONE latent or a GROUP of them, and the difference is
    carried by ``name``: a ``str`` for one, a ``tuple[str, ...]`` for a group.
    Everything else follows from what ``x`` then is. For one latent ``x`` is an
    array and ``shape``/``dtype``/``prior`` describe it directly; for a group
    ``x`` is a ``{name: array}`` dict and each of the three is a dict keyed the
    same way — the shape of a pytree being a pytree of shapes, which is the
    reading ``jax.eval_shape`` already uses.

    That is the whole of the generalization, and it is deliberately NOT a
    concatenation over real degrees of freedom. Every solve in this module runs
    on ``jax.tree.map`` and ``jax.scipy.sparse.linalg.cg``, both of which take
    pytrees; keeping the group a pytree means there is no offset arithmetic to
    invert, no ordering to state beyond JAX's own, and ``S`` is block-diagonal
    because each latent's variance sits on its own leaf rather than being
    spliced into a stacked vector at the right index.

    Attributes:
        name: the latent this block belongs to — or, for a group, the tuple of
            them in the caller's own order. :attr:`names` normalizes the two.
        shape: shape of ``x``; for a group, ``{name: shape}``.
        dtype: dtype of ``x``; for a group, ``{name: dtype}``.
        offset: ``prediction(0)`` — everything the other parameters contribute.
            For a group, everything OUTSIDE the group contributes.
        forward: ``x -> A x``, from ``jax.linearize``.
        adjoint: ``y -> Aᵀ y``, from ``jax.vjp``, shaped like ``x``.
        prior: the latent's declared prior, carried through from the
            :class:`~rheplicant.inference.parameters.Latent`. ``None`` for a
            prior-free latent, and for a block assembled by hand; for a group,
            ``{name: prior}`` with a ``None`` per prior-free member. It is what
            lets :func:`wiener_solve` and :func:`gcr_sample` read ``S`` off the
            declaration instead of making the caller hand-pass — and hand-sync
            — the same two numbers at every exit.

    Adjoint convention, which matters as soon as ``x`` is complex (sky
    ``alm`` coefficients are): ``adjoint`` is exactly ``jax.vjp``, and JAX
    returns the *conjugate* gradient for complex inputs. The identity that
    holds is therefore the one over the **real** inner product::

        Re sum(x * adjoint(y))  ==  sum(forward(x) * y)

    and NOT the sesquilinear ``sum(conj(x) * adjoint(y))``. The real pairing is
    the one a Gaussian likelihood forms, so this is the useful convention as
    well as the honest one; ``tests/inference/test_linear_blocks.py`` pins both
    halves so the distinction cannot rot into a silent factor.
    """

    name: str | tuple[str, ...]
    shape: tuple[int, ...] | dict[str, tuple[int, ...]]
    dtype: Any
    offset: jax.Array
    forward: Callable[[Any], jax.Array]
    adjoint: Callable[[jax.Array], Any]
    prior: Any = None

    @property
    def grouped(self) -> bool:
        """Whether this block holds several latents at once."""
        return isinstance(self.name, tuple)

    @property
    def names(self) -> tuple[str, ...]:
        """The latents in this block, in the caller's order — one, or several."""
        return self.name if isinstance(self.name, tuple) else (self.name,)

    def as_dict(self, x: Any) -> dict[str, Any]:
        """``x`` as the ``{name: array}`` mapping every consumer downstream reads.

        A solve returns this block's own domain — a bare array for a ``name=``
        block, a ``{name: array}`` dict for a ``names=`` group — and only the
        second is the shape anything else takes. ``space.forward_fn``'s
        ``forward``, :meth:`~rheplicant.inference.parameters.ParameterSpace.bind`,
        :func:`~rheplicant.inference.uncertainty.fisher_information`,
        :func:`~rheplicant.inference.identifiability.identifiability`'s ``at=``,
        :func:`linear_operator`'s ``at=`` and
        :func:`~rheplicant.inference.engines.conditional_potential` all index by
        latent name, and all six raise on the bare form — with six *different*
        exceptions, none of which names the actual mistake
        (``TypeError: JAX does not support string indexing; got idx='gain'`` is
        the friendliest of them, and it arrives from inside a trace).

        So this is the wrap, and it is deliberately **idempotent over the two
        spellings**: the same one call is correct whether the block was built
        with ``name=`` or with ``names=``, which is what lets calling code stop
        caring which it was. It returns a new dict; the block is untouched.

        Raises:
            ParameterSpaceError: for a group, if ``x`` is not a dict with one
                entry per member — that is someone else's solution, and
                wrapping it would put an array under a name it does not belong
                to.
        """
        if not self.grouped:
            return {self.name: x}
        if isinstance(x, dict) and set(x) == set(self.names):
            return dict(x)
        raise ParameterSpaceError(
            f"This block groups {list(self.names)}, so its solution is already a dict "
            f"with one entry per member and as_dict() has nothing to wrap; it was given "
            f"{type(x).__name__}"
            + (f" keyed by {sorted(x)}" if isinstance(x, dict) else "")
            + ". A bare array here is another block's answer, and wrapping it would file "
            "it under a name it does not belong to."
        )


def _domain_centre(block: LinearBlock, prior_mean: Any) -> Any:
    """``prior_mean`` laid out over the latent domain, zero where it is ``None``."""

    def one(shape, dtype, mean):
        if mean is None:
            return jnp.zeros(shape, dtype=dtype)
        return jnp.broadcast_to(jnp.asarray(mean, dtype=dtype), shape)

    if not block.grouped:
        return one(block.shape, block.dtype, prior_mean)
    return {
        member: one(block.shape[member], block.dtype[member], prior_mean[member])
        for member in block.names
    }
