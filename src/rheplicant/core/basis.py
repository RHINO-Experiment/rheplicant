"""Smooth (time, frequency) bases: the repair an identifiability refusal names.

:func:`~rheplicant.inference.identifiability.identifiability` refuses a model
whose Jacobian has a null space and tells the caller what to do about it —
"a smooth basis in place of one free parameter per cell is the usual repair".
This module is that basis, and nothing more. The framework needs no new
machinery to *use* it: a basis expansion is already a
:class:`~rheplicant.inference.parameters.Bind` with an ``fn``, over a latent
declared ``linear=True``::

    basis = SeparableBasis(
        time=basis_matrix("legendre", n=n_time, n_basis=3),
        freq=basis_matrix("legendre", n=n_freq, n_basis=4),
    )

    ParameterSpace(
        latents=[Latent("t_coeff", init=basis.fit(t_ant_guess), linear=True)],
        bindings=[Bind("t_coeff", into=lambda p: p["t_ant"].temperature,
                       fn=basis.expand)],
    )

``check_linearity`` verifies the ``linear=True`` claim against that binding,
:func:`~rheplicant.inference.linear.linear_operator` exports ``A`` and ``Aᵀ``
for it without forming a matrix, and both conjugate exits drive it. What was
missing was only the matrices, which are a **modelling choice** and not a
framework one — hence a small honest utility rather than a new abstraction.

**Why the basis has to reach the antenna temperature.** Measured on a known
5000 K calibration tone against a gain free per time sample (the numbers are
pinned in ``tests/radio/test_t_sys_basis.py``, at a generic coefficient point,
for ``n_time=7``, ``n_freq=5``)::

    free-per-cell T_ant,  tone ON  (5000 K)   n_par=42 rank=35 nullity=7
    free-per-cell T_ant,  tone OFF            n_par=42 rank=35 nullity=7
    (3,2)-basis T_ant,    tone ON  (5000 K)   n_par=13 rank=13 nullity=0
    (3,2)-basis T_ant,    tone OFF            n_par=13 rank=12 nullity=1

Against a free-per-cell antenna temperature the tone buys **exactly nothing** —
nullity stays at ``n_time`` either way, because the free cells absorb the whole
of ``g[t] x (tone profile)`` sample by sample, the tone's own channels included.
So smoothing the noise waves alone would leave the tone useless: the basis has
to reach ``T_ant`` too. See
:class:`~rheplicant.radio.t_sys.BasisTemperatureOperator`, which puts it there.

**And it is the FREQUENCY axis that does the work.** Same model, same tone,
varying which axis is restricted (also pinned)::

    n_k              n_j                nullity, tone ON   tone OFF
    3                2                  0                  1
    7 (complete)     2                  0                  7
    3                5 (complete)       1                  1
    7 (complete)     5 (complete)       7                  7

A basis complete in **frequency** makes the tone worth nothing whatever the
time axis does — the tone's profile is then inside the span and is reabsorbed,
nullity 1 with the tone and 1 without. A basis complete in **time** is fine as
long as frequency is restricted. Read that as: the tone is an argument for
frequency smoothness specifically, and ``n_j < n_freq`` is the condition it
needs. ``n_basis == n`` is therefore legal here rather than refused — it is a
perfectly well-conditioned, invertible matrix, and whether it costs anything is
a joint property of the model that only ``identifiability()`` can answer.

**Orientation, once, because it is the thing to get wrong.** A design matrix is
``(n, n_basis)`` — one row per sample, one column per basis function, the
orientation ``numpy.polynomial``'s ``*vander`` helpers use — and the expansion
is::

    T = time @ coeff @ freq.T          (n_time, n_freq)
        (n_time, n_k) (n_k, n_j) (n_j, n_freq)

With ``n_time == n_freq`` and ``n_k == n_j`` a swapped pair of design matrices
is shape-legal and returns the transpose of the intended field: finite,
correctly shaped, wrong. Nothing here can catch that case, for the same reason
:mod:`rheplicant.radio.instrument.noise_wave` cannot tell a per-time vector
from a spectrum when the two axes are the same length. On any other grid the
shape checks below do catch it, which is why the tests are non-square
throughout.

**Where this lives, and why not in** ``rheplicant.inference``. It reads like an
inference utility — ``identifiability()`` prescribes it, ``Bind`` consumes it —
and putting it there would have been a layering inversion. The design matrices
are held by an operator that sits ON the signal path
(:class:`~rheplicant.radio.t_sys.BasisTemperatureOperator`), so ``radio`` would
have had to import ``inference``, which nothing in this package does and which
the inference layer's own premise forbids: it "treats a Pipeline as data, never
lives inside it". ``core`` is the one layer both may depend on, and the fit is
honest rather than merely convenient — this module builds no
:class:`~rheplicant.core.state.State`, holds no radio physics, and imports
nothing but NumPy, JAX and :mod:`rheplicant.core.errors`. The axes are named
``time`` and ``freq`` because :class:`~rheplicant.core.coordinates.Coordinates`
names them that HERE, at the core layer, for every model this package carries.

Its refusals are :class:`~rheplicant.core.errors.StateValidationError` for the
same reason — that is core's error for "constructed with invalid contents", and
it is the one every operator in the package already raises.
``ParameterSpaceError`` would read well for the ``Bind`` route and is
deliberately not part of ``rheplicant.core``'s public surface.
"""

import dataclasses
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from rheplicant.core.errors import StateValidationError

#: The basis families :func:`basis_matrix` builds, in the order its error
#: message lists them.
#:
#: ``"legendre"`` and ``"polynomial"`` span EXACTLY the same functions and are
#: not interchangeable, because the choice is about conditioning and only about
#: conditioning. Measured ``cond(design)``::
#:
#:     n    n_basis    legendre   polynomial   fourier
#:     5    2          1.41       1.41         1.41
#:     9    6          3.43       41.2         1.41
#:     16   10         5.3        1.49e+03     1.41
#:     32   16         7.86       2.81e+05     1.41
#:
#: The raw monomials become nearly parallel as the degree grows —
#: ``x**14`` and ``x**16`` agree to a few percent over most of [-1, 1] — and
#: that condition number lands directly on ``kappa`` of the block's normal
#: operator, which is what
#: :func:`~rheplicant.inference.linear.wiener_solve`'s convergence guard bounds
#: the error by. So: ``"legendre"`` for a smooth quantity, and ``"polynomial"``
#: only when a caller needs the raw monomial coefficients themselves (comparing
#: against ``numpy.polyfit``, say). The ordering is pinned by
#: ``test_legendre_is_far_better_conditioned_than_the_same_span_in_monomials``.
#:
#: ``"fourier"`` is orthogonal on its own grid by construction, hence the flat
#: 1.41 — but it is a different SPAN, and a claim: that the quantity is periodic
#: on this axis (a sidereal cycle, a standing wave in a cable). On a
#: non-periodic axis its constant-plus-harmonics span forces the two ends to
#: agree, which is a statement about the physics and not a numerical detail.
BASIS_KINDS: tuple[str, ...] = ("legendre", "polynomial", "fourier")


def _grid(kind: str, n: int) -> np.ndarray:
    """The normalised coordinate each family is evaluated on.

    Two grids, because the two are not interchangeable. The polynomial families
    take ``linspace(-1, 1, n)``, the interval Legendre polynomials are
    orthogonal on. Fourier takes ``i/n`` — endpoint **excluded** — because that
    is the grid its harmonics are orthogonal on: on the endpoint-included grid
    the first and last sample sit one full period apart and carry the same
    phase, the harmonics acquire nonzero inner products, and the coefficients
    come back correlated in a way nobody declared. It is a one-character
    difference that changes no shape.
    """
    if kind == "fourier":
        return np.arange(n, dtype=np.float64) / n
    return np.linspace(-1.0, 1.0, n, dtype=np.float64)


def _fourier_columns(x: np.ndarray, n_basis: int) -> np.ndarray:
    """``[1, cos(2pi x), sin(2pi x), cos(4pi x), sin(4pi x), ...]``, truncated.

    An EVEN ``n_basis`` stops after a cosine, so the last harmonic is
    represented in one phase only. That is a legitimate truncation rather than
    a mistake — it is what "give me four Fourier terms" has to mean — but it is
    worth knowing that such a basis is not invariant under a shift of the axis,
    where an odd one is.
    """
    columns = [np.ones_like(x)]
    harmonic = 1
    while len(columns) < n_basis:
        columns.append(np.cos(2.0 * math.pi * harmonic * x))
        if len(columns) < n_basis:
            columns.append(np.sin(2.0 * math.pi * harmonic * x))
        harmonic += 1
    return np.stack(columns, axis=1)


def basis_matrix(kind: str, *, n: int, n_basis: int) -> jax.Array:
    """A ``(n, n_basis)`` design matrix over a normalised axis.

    Built in NumPy float64 and converted once, so the recurrence that generates
    the Legendre columns runs in double precision whatever
    ``jax_enable_x64`` says, and only the stored constant follows the caller's
    configuration.

    Args:
        kind: one of :data:`BASIS_KINDS`.
        n: number of samples on the axis — ``n_time`` or ``n_freq``.
        n_basis: number of basis functions, i.e. of coefficients on this axis.
            Keyword-only, along with ``n``, because ``basis_matrix("legendre",
            5, 7)`` reads plausibly in either order and one of the two orders is
            a silently over-complete basis.

    Returns:
        ``(n, n_basis)``: one row per sample, one column per function, with the
        constant function first for every kind — so coefficient ``[0, 0]`` of a
        :class:`SeparableBasis` is always the mean level.

    Raises:
        StateValidationError: on an unknown ``kind``, a non-positive ``n`` or
            ``n_basis``, or ``n_basis > n``. Not ``ParameterSpaceError`` --
            this module's docstring argues that one deliberately is not part of
            ``rheplicant.core``'s public surface, so a caller who wrote
            ``except ParameterSpaceError`` around this would catch nothing and
            could not import the name to try.
    """
    if kind not in BASIS_KINDS:
        raise StateValidationError(
            f"Unknown basis kind {kind!r}; this module builds {list(BASIS_KINDS)}. "
            "Each is a different claim about the quantity — a Fourier basis says it "
            "is periodic on this axis, a polynomial one that it is smooth — so the "
            "nearest-sounding name is not a safe guess. Build the design matrix "
            "yourself and hand it to SeparableBasis if you want a family that is not "
            "here; nothing below this line cares where the columns came from."
        )
    if n < 1:
        raise StateValidationError(
            f"basis_matrix needs n >= 1, got n={n}. An axis with no samples gives a "
            "design matrix with no rows, whose expansion is an empty array that "
            "broadcasts against nothing and fails much later, in an operator that "
            "cannot say what went wrong."
        )
    if n_basis < 1:
        raise StateValidationError(
            f"basis_matrix needs n_basis >= 1, got n_basis={n_basis}. A basis with no "
            "functions expands every coefficient to exactly zero, so the quantity it "
            "parameterizes is pinned at zero for the whole run while the fit converges "
            "and reports a healthy answer for every OTHER parameter. Pass n_basis=1 "
            "for a constant."
        )
    if n_basis > n:
        raise StateValidationError(
            f"basis_matrix was asked for n_basis={n_basis} functions on an axis of "
            f"n={n} samples. More coefficients than samples is a null space by "
            "counting alone — the last "
            f"{n_basis - n} of them are exactly unconstrained by this axis however "
            "much data there is, and a conjugate solve would return whatever the "
            "prior said along those directions while reporting a converged residual. "
            f"Use n_basis <= {n}, or add samples."
        )
    x = _grid(kind, n)
    if kind == "legendre":
        columns = np.polynomial.legendre.legvander(x, n_basis - 1)
    elif kind == "polynomial":
        columns = np.vander(x, n_basis, increasing=True)
    else:
        columns = _fourier_columns(x, n_basis)
    return jnp.asarray(columns)


def _check_design(axis: str, matrix: jax.Array) -> None:
    """Refuse a design matrix that cannot mean what its author meant.

    Called once per axis, so a message always says WHICH axis — with the two
    grids different lengths (the only configuration in which any of this is
    checkable at all) that is the difference between "you passed a vector" and
    "you passed the two matrices the wrong way round".

    The messages name "the {axis} design matrix" rather than a constructor
    argument, because there are two call sites and they spell it differently:
    :class:`SeparableBasis` takes ``time=``/``freq=`` and
    :class:`~rheplicant.radio.t_sys.BasisTemperatureOperator` takes
    ``time_basis=``/``freq_basis=``. A message naming one of them is wrong half
    the time it is read.
    """
    if matrix.ndim != 2:
        raise StateValidationError(
            f"The {axis} design matrix has shape {tuple(matrix.shape)}: a design matrix "
            f"is 2-D, (n_{axis}, n_basis), never {matrix.ndim}-D. A bare vector of "
            "length n would be read by the matrix product as a single ROW — one "
            "sample, n coefficients — and the expansion would come back the wrong "
            f"shape. Pass basis_matrix(kind, n=n_{axis}, n_basis=...)[:, None] if you "
            "really do mean one basis function."
        )
    if 0 in matrix.shape:
        raise StateValidationError(
            f"The {axis} design matrix has shape {tuple(matrix.shape)}, which is empty. "
            "An empty axis makes the expansion an array with a zero dimension: it "
            "still broadcasts, it still has the right number of dimensions, and every "
            "quantity on that axis is pinned to nothing at all."
        )
    if matrix.shape[1] > matrix.shape[0]:
        raise StateValidationError(
            f"The {axis} design matrix has shape {tuple(matrix.shape)}: "
            f"{matrix.shape[1]} basis functions over {matrix.shape[0]} samples. More "
            "coefficients than samples on one axis is a null space by counting alone, "
            "whatever the other axis does and however much data there is — the solve "
            "returns the prior along those directions and reports a converged "
            "residual. If the two design matrices are the right way round, drop "
            "columns; if they are not, the shapes here are the symptom."
        )


@dataclasses.dataclass(frozen=True, eq=False)
class SeparableBasis:
    """A separable expansion of a ``(n_time, n_freq)`` field: ``time @ C @ freq.T``.

    Deliberately a plain dataclass rather than an ``eqx.Module``, for the same
    reason :class:`~rheplicant.inference.linear.LinearBlock` is: this is a
    derived linear-algebra *handle*, something you build where you need it, not
    a pytree to carry through a model. An ``eqx.Module`` would be actively
    wrong here, and specifically at the pattern this class exists to serve —
    equinox wraps a Module's bound methods as pytrees, so ``Bind(...,
    fn=basis.expand)`` would put the design matrices in ``Bind``'s STATIC
    field, where an array is compared by ``__eq__`` and equinox warns about it.
    An operator that wants the matrices as differentiable leaves holds them
    directly; see :class:`~rheplicant.radio.t_sys.BasisTemperatureOperator`.

    ``eq=False`` is ordinary hygiene for a value object holding arrays, and it
    is worth stating exactly what it does and does not buy. A frozen dataclass
    defaults to ``eq=True``, which compares the fields elementwise: ``basis_a ==
    basis_b`` then RAISES ``ValueError: the truth value of an array with more
    than one element is ambiguous`` rather than answering, and ``__hash__``
    becomes ``None``. ``eq=False`` gives identity semantics for both, so the two
    ordinary things you can do to any Python object stay ordinary. It is NOT
    what makes ``fn=basis.expand`` safe as a static field — a bound method
    compares and hashes its ``__self__`` by POINTER, so that route works either
    way. Being an ``eqx.Module`` is the thing that would break it.

    Separable rather than general on purpose. A general basis over the joint
    grid would be ``(n_time * n_freq, n_coeff)`` and would say nothing about
    which axis a given function varies on; the separable form is the one whose
    smoothness claim decomposes — "``n_k`` functions in time, ``n_j`` in
    frequency" — and the frequency count is the one the calibration tone
    argument turns on (see the module docstring).

    Attributes:
        time: ``(n_time, n_k)`` design matrix on the time axis.
        freq: ``(n_freq, n_j)`` design matrix on the frequency axis.
    """

    time: jax.Array
    freq: jax.Array

    def __post_init__(self):
        object.__setattr__(self, "time", jnp.asarray(self.time))
        object.__setattr__(self, "freq", jnp.asarray(self.freq))
        _check_design("time", self.time)
        _check_design("freq", self.freq)

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_time, n_freq)`` — the grid :meth:`expand` lands on."""
        return (int(self.time.shape[0]), int(self.freq.shape[0]))

    @property
    def coeff_shape(self) -> tuple[int, int]:
        """``(n_k, n_j)`` — the shape of the coefficient this basis takes."""
        return (int(self.time.shape[1]), int(self.freq.shape[1]))

    def expand(self, coeff: Any) -> jax.Array:
        """``time @ coeff @ freq.T`` — the coefficients as a field on the grid.

        This is the function to hand to ``Bind(..., fn=...)``. It is affine in
        ``coeff`` (linear, in fact), which is the claim ``Latent(...,
        linear=True)`` makes and ``check_linearity`` checks.

        The result is ALWAYS 2-D and always the full grid, even for a
        one-coefficient basis on each axis. That matters where it is consumed:
        :mod:`rheplicant.radio.instrument.noise_wave` accepts ``()``,
        ``(n_freq,)``, ``(n_time, 1)`` and ``(n_time, n_freq)``, and states that
        a bare 1-D array is unresolvably ambiguous when ``n_time == n_freq``.
        An expansion never produces one.
        """
        got = tuple(jnp.shape(coeff))
        if got != self.coeff_shape:
            raise StateValidationError(
                f"This basis takes a coefficient of shape {self.coeff_shape} "
                f"(n_k, n_j), got {got}. Note that the transpose of one of these is "
                "the other whenever n_k == n_j, and then nothing here can tell them "
                "apart: the product would be legal and the expansion would come back "
                "as the transpose of the field you meant. The first axis of the "
                "coefficient indexes the TIME basis functions."
            )
        return self.time @ jnp.asarray(coeff) @ self.freq.T

    def fit(self, values: Any) -> jax.Array:
        """Least-squares coefficients for a field on this grid — the inverse of
        :meth:`expand`, where one exists.

        The natural way to build an ``init`` for the coefficient latent: hand it
        the field you would otherwise have declared per cell and it returns the
        closest thing this basis can say. That matters more than convenience —
        :func:`~rheplicant.inference.linear.check_linearity` takes its probe
        scales from ``max|init|``, so an all-zero init makes the probes absolute
        and never reaches the regime a 3000 K temperature actually lives in.

        A field OUTSIDE the span is projected onto it rather than refused: the
        residual is what the basis cannot represent, and if that residual
        matters the answer is a bigger basis, not an exception here. Compare
        ``values`` against ``expand(fit(values))`` to see it.

        Uses the pseudo-inverse of each axis independently, which is the exact
        least-squares solution because ``vec(A C Bᵀ) = (B ⊗ A) vec(C)`` and the
        pseudo-inverse of a Kronecker product is the Kronecker product of the
        pseudo-inverses.
        """
        got = tuple(jnp.shape(values))
        if got != self.shape:
            raise StateValidationError(
                f"This basis covers a grid of shape {self.shape} (n_time, n_freq), "
                f"got {got}. fit() solves for the coefficients of a field ON this "
                "grid; a field of another shape is a field of another model."
            )
        return jnp.linalg.pinv(self.time) @ jnp.asarray(values) @ jnp.linalg.pinv(self.freq).T


__all__ = ["BASIS_KINDS", "SeparableBasis", "basis_matrix"]
