"""How far an iterative optimiser is from the minimum, and when to ask.

Everything here takes a real-valued objective ``f`` over a pytree of arrays
and nothing else: no model, no data, no parameter names. It answers two
questions a block-coordinate or alternating optimiser cannot answer from its
own iterates.

**Where am I?** :func:`decrement` computes the Newton decrement
``lambda = sqrt(g^T H^-1 g)`` at a point, for ``g`` and ``H`` the gradient and
Hessian of ``f`` there. Near a minimum ``f`` is quadratic and
``f - f* = lambda**2 / 2``, so ``lambda`` is the distance to the minimum in
units of the curvature's own standard deviation -- which does not grow with
the number of terms in ``f`` and does not care which coordinates are slow.
A tolerance on the CHANGE of ``f`` between iterations answers neither: it is
relative to ``|f|``, and the contraction it can be extrapolated at is the
fastest mode still moving, so a slow mode hidden under a fast one passes.

**Can I trust the number?** The Hessian is never formed from the model --
every product is ``jax.jvp`` of ``jax.grad`` -- and the linear solve behind
``H^-1 g`` is inexact. :class:`Decrement` therefore carries an upper bound,
not only an estimate. With ``r = g - H x`` the true residual of the computed
``x`` and ``mu`` a LOWER bound on the smallest eigenvalue of ``H``,

    lambda  <=  (R + sqrt(R**2 + 4 g^T x)) / 2,      R = |r| / sqrt(mu)

by Cauchy-Schwarz in the ``H^-1`` inner product (``lambda**2 - g^T x = g^T
H^-1 r <= lambda R``), inflated by ``1 / sqrt(1 - eps kappa)`` for the
rounding of ``g`` and ``H`` themselves. :meth:`Decrement.certifies` reads
that bound, so an inexact solve can only make a verdict refuse.

**Everything turns on that lower bound.** Where the Hessian is formed (up to
:data:`DENSE_MAX` parameters) it is the scaled matrix's own smallest
eigenvalue, computed, which also decides positive definiteness. Above that
the solve is conjugate gradients, which can only sample the spectrum it has
reached, so the caller should pass ``floor=``: a VERIFIED lower bound, which
a model with proper Gaussian priors has, since ``H = J^T N^-1 J + P`` is
bounded below by the prior precision ``P`` whenever the map is linear and the
noise does not depend on it. Without one, a short Lanczos probe of ``H`` is
taken and the floor is the lower end of its Ritz interval -- an estimate, and
the reason a certificate there is best not relied on. A probe that cannot put
that interval above zero refuses outright, as the dense path does.

**Getting there.** :func:`polish` takes Newton steps -- a Steihaug-truncated
conjugate-gradient direction, an Armijo backtracking line search, and each
step kept only if it lowers ``f`` -- which is what removes a first-order
optimiser's step-size floor before the decrement is asked.

**When to ask.** A decrement costs a gradient and a solve, so
:func:`settled`, :func:`gap_step` and :func:`change_program` are the cheap
tests that pick the iterations worth spending one on. They schedule; none of
them certifies.

Limits. The decrement is a statement about the quadratic model of ``f`` at
the point, so where the curvature changes over one unit of distance it is
local. Above :data:`DENSE_MAX` parameters with no ``floor``, the bound rests
on a probe rather than a proof, and a direction of negative curvature that
neither the probe nor the Krylov space meets is not seen. In a precision
whose epsilon times ``kappa`` approaches one, the gradient and the Hessian
are rounding and nothing here recovers them; the status says so instead.
"""

import dataclasses
import math
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.flatten_util import ravel_pytree

# ------------------------------------------------------------ the decrement --

#: Relative residual at which the decrement's conjugate gradients stop, by
#: working precision (bytes). The attainable level is about ``eps * kappa``,
#: so a 4-byte float asks for less. What the stop costs the decrement is
#: bounded afterwards from the TRUE residual, not assumed from this number.
RTOL: dict[int, float] = {4: 1e-4, 8: 1e-8}

#: Conjugate-gradient iterations the decrement may take: ``4 n + 20`` for
#: ``n`` flattened parameters (exact arithmetic needs ``n``; the rest absorbs
#: lost orthogonality), capped here. An iteration that has not reached its
#: residual at the cap certifies nothing.
MAXITER: int = 1000

#: Parameters up to which the decrement forms the Hessian from ``n``
#: Hessian-vector products, :data:`BATCH` at a time so memory stays a few
#: forward passes, and solves it densely. Up to here that costs no more
#: products than conjugate gradients need on a correlated objective, and it
#: buys what no iteration can: the exact extreme eigenvalues, hence a
#: curvature floor the bound can divide by and a positive-definiteness test
#: that sees a direction the Krylov space never visits.
#:
#: Measured on a linear model of ``n`` coefficients over 1e4 data, one
#: decrement costs 27 ms at 256, 81 ms at 512, 319 ms at 1024 and 1.5 s at
#: 2048, the last dominated by the eigendecomposition's ``n**3``. 1024 is
#: where that is still a fraction of the sweeps it certifies; above it the
#: solve is conjugate gradients and the caller has to supply ``floor``, or
#: accept a probe.
DENSE_MAX: int = 1024
BATCH: int = 16

#: Lanczos steps the spectral probe takes when the iterative path is given no
#: ``floor``: enough for the extreme Ritz values of a clustered spectrum,
#: which is where a missed small eigenvalue would hurt, and one
#: Hessian-vector product each.
PROBE_STEPS: int = 32

#: What a decrement's solve did. ``UNREACHED``: the conjugate gradients did
#: not reach their residual within the cap. ``NONCONVEX``: a direction of
#: non-positive curvature, so the point is not near a minimum and the
#: decrement means nothing. Only :data:`CONVERGED` can certify.
CONVERGED, UNREACHED, NONCONVEX, NONFINITE = range(4)

#: Each status as a phrase a caller can put in a sentence about the solve.
STATUS_SAID: dict[int, str] = {
    CONVERGED: "reached its residual",
    UNREACHED: "did not reach its residual within its iteration cap",
    NONCONVEX: "met a direction of non-positive curvature, so the point is not "
               "near a minimum",
    NONFINITE: "was not finite",
}


@dataclasses.dataclass(frozen=True)
class Decrement:
    """One Newton decrement: what it measured, and how well.

    Attributes:
        estimate: ``sqrt(g^T x)`` for the computed ``x``, in units of the
            curvature's standard deviation -- the distance to the minimum of
            the objective's quadratic model, as the solve left it, and a
            lower bound on it.
        distance: the upper bound on that distance, ``inf`` when the residual
            or the curvature cannot bound it. This is what a verdict reads.
        lambda2, residual, reach, kappa: ``g^T x``; the true residual's size
            relative to ``g``; that residual divided by the square root of
            the curvature floor, which is the ``R`` of the bound; and the
            condition number the rounding allowance was computed from.
        products: Hessian-vector products spent, the residual's own and any
            spectral probe's included.
        dense: whether the Hessian was formed and solved directly, which is
            also whether the curvature floor was computed rather than
            supplied or probed.
        status: one of :data:`CONVERGED`, :data:`UNREACHED`,
            :data:`NONCONVEX`, :data:`NONFINITE`.
    """

    estimate: float
    distance: float
    lambda2: float
    residual: float
    reach: float
    kappa: float
    products: int
    dense: bool
    status: int

    def certifies(self, limit: float) -> bool:
        """Whether the point is within ``limit`` of the minimum, bound included."""
        return self.status == CONVERGED and self.distance <= limit


def decrement_program(
    objective: Callable[[Any], jax.Array],
    template: Any,
    *,
    floor: float | None = None,
    limit: float | None = None,
) -> Callable[[Any], tuple[jax.Array, ...]]:
    """``values -> (lambda2, rho, reach, products, status, kappa)``, jitted once.

    ``objective`` is a real function of a pytree shaped like ``template``; a
    complex leaf enters as its real and imaginary parts, so ``n`` counts real
    degrees of freedom. Build this once per shape and call it at each point:
    it is the expensive half, and :func:`decrement` is the reading of its
    output.

    Up to :data:`DENSE_MAX` parameters the Hessian is formed from ``n``
    products and solved by an eigendecomposition after scaling by its
    diagonal, and the curvature floor the bound needs is that scaled matrix's
    own smallest eigenvalue. Above it, ``H x = g`` is conjugate gradients
    from zero, unpreconditioned, and the floor is either ``floor`` -- a
    verified lower bound on the smallest eigenvalue of ``H``, which is what
    makes a certificate there sound -- or, failing that, a
    :data:`PROBE_STEPS`-step Lanczos probe of ``H``.

    ``limit`` is the distance the caller will certify. Given one, the
    iteration stops as soon as the verdict cannot change: when the bound is
    inside ``limit``, or when ``g^T x`` alone (which only grows) is already
    outside it. Without one it stops at a relative residual of
    :data:`RTOL`. Either way the residual returned is ``|g - H x|``
    recomputed with one more product, because a recursive residual drifts
    from the true one in finite precision, and it is the true one the bound
    needs.
    """
    flat0, unravel = ravel_pytree(template)
    complex_ = jnp.iscomplexobj(flat0)
    size = int(flat0.size)

    def to_real(flat):
        return jnp.concatenate([flat.real, flat.imag]) if complex_ else flat

    def from_real(vector):
        if complex_:
            return unravel(vector[:size] + 1j * vector[size:])
        return unravel(vector)

    def flattened(vector):
        return objective(from_real(vector))

    slope_of = jax.grad(flattened)
    n = 2 * size if complex_ else size
    rtol = RTOL.get(jnp.dtype(flat0.real.dtype).itemsize, 1e-8)
    dense = n <= DENSE_MAX

    @eqx.filter_jit
    def solved(values):
        vector = to_real(ravel_pytree(values)[0])
        slope = slope_of(vector)

        def curvature(direction):
            return jax.jvp(slope_of, (vector,), (direction,))[1]

        if dense:
            x, scale, products, status, bottom, kappa = _dense_solve(curvature, slope, n)
        else:
            x, scale, products, status, bottom, kappa = _iterative_solve(
                curvature, slope, n, rtol, floor, limit
            )
        remainder = slope - curvature(x)
        squared = jnp.sum(remainder * remainder / scale)
        target = jnp.sum(slope * slope / scale)
        rho = jnp.sqrt(squared / jnp.where(target > 0.0, target, 1.0))
        reach = jnp.sqrt(squared / jnp.where(bottom > 0.0, bottom, jnp.nan))
        lambda2 = jnp.sum(slope * x)
        status = jnp.where(
            jnp.all(jnp.isfinite(slope)) & jnp.isfinite(rho) & jnp.isfinite(lambda2),
            status, NONFINITE,
        )
        return lambda2, rho, reach, products + 1, status, kappa

    def program(values):
        return solved(values)

    # Which path the program took is a property of the shape, fixed when it was
    # built; a caller reads it to say why a certificate could not be had.
    program.dense = dense
    return program


def decrement(
    objective: Callable[[Any], jax.Array],
    at: Any,
    *,
    program: Callable[[Any], tuple[jax.Array, ...]] | None = None,
    floor: float | None = None,
    limit: float | None = None,
) -> Decrement:
    """The Newton decrement of ``objective`` at ``at``, with its error bound.

    ``program`` is a :func:`decrement_program` built earlier for this shape;
    without one this builds (and compiles) a program for this call alone, and
    ``floor`` and ``limit`` are passed to it.

    The bound is ``(R + sqrt(R**2 + 4 lambda2)) / 2`` for ``R`` the residual
    measured against the curvature floor, inflated by ``1 / sqrt(1 - eps
    kappa)``: a relative perturbation ``eps`` of ``H`` moves ``g^T H^-1 g`` by
    up to ``eps kappa`` of itself, and ``g`` and every Hessian product are
    rounded. That second term is negligible in an 8-byte float and refuses
    any Hessian with ``kappa`` near ``1 / eps`` in a 4-byte one, where the
    decrement's digits are rounding.
    """
    if program is None:
        program = decrement_program(objective, at, floor=floor, limit=limit)
    lambda2, rho, reach, products, status, kappa = program(at)
    eps = float(jnp.finfo(jnp.result_type(lambda2)).eps)
    lambda2, rho, reach, kappa = float(lambda2), float(rho), float(reach), float(kappa)
    squared = max(lambda2, 0.0)
    upper = 0.5 * (reach + math.sqrt(reach**2 + 4.0 * squared))
    inflation = 1.0 - eps * kappa
    if math.isfinite(upper) and inflation > 0.0:
        distance = upper / math.sqrt(inflation)
    else:
        distance = math.inf
    return Decrement(
        estimate=math.sqrt(squared) if math.isfinite(lambda2) else math.inf,
        distance=distance,
        lambda2=lambda2,
        residual=rho,
        reach=reach,
        kappa=kappa,
        products=int(products),
        dense=bool(getattr(program, "dense", False)),
        status=int(status),
    )


def _dense_solve(curvature: Callable, slope: jax.Array, n: int):
    """``H x = g`` from ``n`` Hessian columns, scaled by the diagonal.

    Returns the scaled matrix's smallest eigenvalue with the solve, so the
    caller's bound rests on a computed curvature floor rather than an
    estimate of one, and its condition number for the rounding allowance.

    Forming ``H`` costs ``n`` products and the eigendecomposition carries its
    own rounding, so a smallest eigenvalue below about ``n eps`` of the
    largest is indistinguishable from zero: the objective is flat along some
    direction, the minimum is not a point, and the decrement has nothing to
    measure.
    """
    unit = jnp.eye(n, dtype=slope.dtype)
    columns = lax.map(curvature, unit, batch_size=min(n, BATCH))
    hessian = 0.5 * (columns + columns.T)
    diagonal = jnp.diagonal(hessian)
    scale = jnp.where(diagonal > 0.0, diagonal, 1.0)
    root = jnp.sqrt(scale)
    eigenvalues, vectors = jnp.linalg.eigh(hessian / jnp.outer(root, root))
    lowest, highest = eigenvalues[0], eigenvalues[-1]
    eps = jnp.finfo(slope.dtype).eps
    convex = (lowest > n * eps * highest) & jnp.all(diagonal > 0.0)
    inverse = jnp.where(eigenvalues > 0.0, 1.0 / eigenvalues, 0.0)
    x = (vectors @ (inverse * (vectors.T @ (slope / root)))) / root
    status = jnp.where(convex, CONVERGED, NONCONVEX)
    kappa = highest / jnp.where(lowest > 0.0, lowest, jnp.nan)
    return x, scale, n, status, lowest, kappa


def _probe_spectrum(curvature: Callable, n: int, dtype: Any) -> tuple[jax.Array, jax.Array]:
    """Bounds on the extreme eigenvalues of ``H``, from a short Lanczos.

    :data:`PROBE_STEPS` steps from a fixed vector, without
    reorthogonalization, and one Hessian-vector product each. A Ritz value
    ``theta`` of the Lanczos matrix has an eigenvalue of ``H`` within
    ``beta |s|``, for ``beta`` the last off-diagonal and ``s`` the last
    component of ``theta``'s Ritz vector, so ``theta[0] - beta |s[0]|``
    bounds the spectrum BELOW and ``theta[-1] + beta |s[-1]|`` above. That is
    what makes a floor out of a probe: an eigenvalue the Krylov space has not
    reached shows up as a residual too large to bound anything with, not as a
    floor that is too high.

    Lost orthogonality duplicates converged Ritz values, which moves neither
    end of the interval outwards.
    """
    steps = min(PROBE_STEPS, n)
    start = jax.random.normal(jax.random.PRNGKey(0), (n,), dtype)
    start = start / jnp.linalg.norm(start)

    def step(carry, _):
        vector, previous, beta = carry
        product = curvature(vector)
        alpha = jnp.sum(vector * product)
        following = product - alpha * vector - beta * previous
        length = jnp.linalg.norm(following)
        safe = jnp.where(length > 0.0, length, 1.0)
        return (following / safe, vector, length), (alpha, length)

    zero = jnp.zeros((), dtype)
    _, (alphas, betas) = lax.scan(
        step, (start, jnp.zeros_like(start), zero), None, length=steps
    )
    off = betas[:-1]
    tridiagonal = jnp.diag(alphas) + jnp.diag(off, 1) + jnp.diag(off, -1)
    theta, vectors = jnp.linalg.eigh(tridiagonal)
    reach = betas[-1] * jnp.abs(vectors[-1, :])
    return theta[0] - reach[0], theta[-1] + reach[-1]


def _iterative_solve(curvature: Callable, slope: jax.Array, n: int, rtol: float,
                     floor: float | None, limit: float | None):
    """``H x = g`` by conjugate gradients from zero, against a curvature floor.

    The floor is the caller's when it has one and the Lanczos probe's
    otherwise (see :func:`decrement_program`). The condition number returned
    beside it is for the caller's rounding allowance only, and is the
    spectrum the arithmetic met -- the probe's interval, or the extreme
    Rayleigh quotients of the directions the iteration took. A floor is a
    claim about the objective; a condition number here is an observation
    about the run, and the bound is built from the first alone.

    With a ``limit`` the iteration stops on the verdict rather than on a
    fixed residual: a bound inside the limit certifies, and an estimate
    already outside it cannot be rescued by more iterations, since ``g^T x``
    only grows.
    """
    maxiter = min(4 * n + 20, MAXITER)
    eps = jnp.finfo(slope.dtype).eps
    if floor is None:
        least, largest = _probe_spectrum(curvature, n, slope.dtype)
        # The probe's interval, less a rounding allowance for the products it
        # was built from: a floor at or below zero is no floor at all.
        bottom = least - eps * jnp.abs(largest)
        probed = largest / jnp.where(least > 0.0, least, jnp.nan)
        convex, spent = bottom > 0.0, min(PROBE_STEPS, n)
    else:
        bottom = jnp.asarray(floor, slope.dtype)
        probed, convex, spent = jnp.asarray(jnp.nan, slope.dtype), jnp.asarray(True), 0
    target = jnp.sum(slope * slope)
    zeros = jnp.zeros(maxiter, slope.dtype)
    ceiling = jnp.asarray(jnp.inf if limit is None else limit, slope.dtype)

    def going(carry):
        return carry[0] < 0

    def step(carry):
        status, k, x, r, p, rr, seen, alphas = carry
        product = curvature(p)
        bend = jnp.sum(p * product)
        alpha = rr / jnp.where(bend > 0.0, bend, 1.0)
        x = x + alpha * p
        r = r - alpha * product
        following = jnp.sum(r * r)
        beta = following / jnp.where(rr > 0.0, rr, 1.0)
        # Rayleigh quotients of the directions taken: an estimate of the
        # spectrum the ARITHMETIC sees, which is what the caller's rounding
        # allowance is scaled by. The bound itself never reads them.
        quotient = bend / jnp.sum(p * p)
        seen = (jnp.maximum(seen[0], quotient), jnp.minimum(seen[1], quotient))
        lambda2 = jnp.sum(slope * x)
        reach = jnp.sqrt(following / jnp.where(bottom > 0.0, bottom, jnp.nan))
        bound = 0.5 * (reach + jnp.sqrt(reach**2 + 4.0 * jnp.maximum(lambda2, 0.0)))
        decided = (
            (following <= rtol**2 * target) if limit is None
            else (bound <= ceiling) | (lambda2 > ceiling**2)
        )
        status = jnp.where(
            ~(bend > 0.0), NONCONVEX,
            jnp.where(decided, CONVERGED,
                      jnp.where(k + 1 >= maxiter, UNREACHED, -1)))
        return (status, k + 1, x, r, r + beta * p, following, seen,
                alphas.at[k].set(alpha))

    start = (jnp.where(target > 0.0, jnp.where(convex, -1, NONCONVEX),
                       jnp.where(convex, CONVERGED, NONCONVEX)),
             jnp.asarray(0), jnp.zeros_like(slope), slope, slope, target,
             (jnp.asarray(0.0, slope.dtype), jnp.asarray(jnp.inf, slope.dtype)), zeros)
    status, k, x, _, _, _, seen, _ = lax.while_loop(going, step, start)
    spread = seen[0] / jnp.where(seen[1] > 0.0, seen[1], jnp.nan)
    kappa = jnp.where(jnp.isnan(probed), jnp.where(k > 0, spread, 1.0), probed)
    return x, jnp.ones_like(slope), k + spent, status, bottom, kappa


# ---------------------------------------------------------------- the polish --

#: Newton iterations :func:`polish` takes by default.
POLISH_ITERATIONS: int = 3

#: Conjugate-gradient iterations per Newton iteration, at most. A parameter
#: set with at most this many flattened elements gets a full Newton step (in
#: exact arithmetic CG terminates within ``n`` iterations on an
#: ``n``-dimensional positive-definite system); a larger, ill-conditioned one
#: may get a truncated step, which the acceptance test still keeps from
#: making anything worse.
POLISH_CG_MAXITER: int = 50

#: Relative residual at which a Newton step's conjugate gradients stop:
#: ``|H p + g| <= POLISH_CG_TOL |g|``. A step solved to this cuts a
#: quadratic's error by the same factor, so three iterations reach 1e-15.
POLISH_CG_TOL: float = 1e-5

#: Armijo's sufficient-decrease constant: a step ``t p`` is kept when
#: ``f(x + t p) - f(x) <= ARMIJO * t * g.p`` with ``g.p < 0``. The usual
#: 1e-4; it asks for a decrease the step's own slope predicts, not merely no
#: rise, so a jump the objective cannot resolve is refused.
ARMIJO: float = 1e-4

#: Step halvings the Armijo search may take before the Newton iteration keeps
#: its starting point. 2**-30 is below a 4-byte float's relative resolution.
ARMIJO_HALVINGS: int = 30


def _steihaug(curvature: Callable[[jax.Array], jax.Array], slope: jax.Array) -> jax.Array:
    """Newton direction by conjugate gradients, stopped at negative curvature.

    Solves ``H p = -g`` for ``H`` given as ``curvature(d) = H d``, and stops
    at the first direction with ``d.H d <= 0`` (Steihaug's truncation, with
    no trust region): the iterate reached so far is a descent direction, and
    at the first iteration that is ``-g`` itself. Plain CG does not stop
    there, and on an indefinite ``H`` it heads for whatever stationary point
    the quadratic model has -- measured on a bilinear product of two
    parameters, onto a saddle at ``f = 1494.7`` with the minimum at ``-57.0``.
    """
    scale = jnp.sqrt(jnp.sum(slope * slope))

    def going(carry):
        index, _, residual, _, _, done = carry
        return (~done) & (index < POLISH_CG_MAXITER) & (
            jnp.sqrt(jnp.sum(residual * residual)) > POLISH_CG_TOL * scale
        )

    def step(carry):
        index, point, residual, direction, squared, _ = carry
        product = curvature(direction)
        bend = jnp.sum(direction * product)
        negative = bend <= 0.0
        alpha = squared / jnp.where(negative, 1.0, bend)
        moved = point + alpha * direction
        following = residual - alpha * product
        renewed = jnp.sum(following * following)
        beta = renewed / jnp.where(squared > 0.0, squared, 1.0)
        # negative curvature: keep the point reached, or -g if none yet
        stopped = jnp.where(index == 0, direction, point)
        return (
            index + 1,
            jnp.where(negative, stopped, moved),
            jnp.where(negative, residual, following),
            jnp.where(negative, direction, following + beta * direction),
            jnp.where(negative, squared, renewed),
            negative,
        )

    start = (jnp.asarray(0), jnp.zeros_like(slope), -slope, -slope,
             jnp.sum(slope * slope), jnp.asarray(False))
    return lax.while_loop(going, step, start)[1]


def polish(
    objective: Callable[[Any], jax.Array],
    at: Any,
    *,
    iterations: int = POLISH_ITERATIONS,
) -> tuple[Any, jax.Array]:
    """Newton steps on ``objective``, each kept only if it helps.

    Returns ``(point, kept)``, ``kept`` counting the steps that were taken --
    zero means the point came back unchanged.

    A first-order optimiser restarted at each outer iteration has a precision
    floor set by its step size rather than by the problem: Adam with zeroed
    moments takes a first bias-corrected step of ``step_size *
    sign(gradient)`` however small the gradient is, and the resulting map has
    a fixed point a fraction of a step size from the optimum -- measured at
    2881 curvature standard deviations on a two-parameter power law fitted
    over 4096 channels, at every inner step count, shrinking with the step
    size and not with the iterations.

    A Newton step uses the curvature, so its length goes to zero with the
    gradient and the iterate lands on the optimum wherever the first-order
    steps left it. The direction is :func:`_steihaug`'s: conjugate gradients
    on Hessian-vector products, no Hessian formed, stopped at negative
    curvature. The step along it is backtracked (halved up to
    :data:`ARMIJO_HALVINGS` times) until Armijo's condition holds,
    ``f(x + t p) - f(x) <= ARMIJO * t * g.p`` with ``g.p < 0`` and the result
    finite; otherwise the iteration keeps its point. The decrease is compared
    as a DIFFERENCE, because ``f(x) + small`` rounds to ``f(x)`` when ``|f|``
    is large: measured, ``after <= before`` accepted a 4-byte-float jump from
    ``y = 3`` to ``y = -97.9`` on a potential offset by 1e7, where no step
    could be resolved at all.

    A complex parameter set is returned unchanged: the Hessian-vector product
    of a real function of complex arguments is not the operator this step
    needs.
    """
    flat0, unravel = ravel_pytree(at)
    if not jnp.issubdtype(flat0.dtype, jnp.floating):
        return at, jnp.asarray(0)

    def flattened(flat: jax.Array) -> jax.Array:
        return objective(unravel(flat))

    slope_of = jax.grad(flattened)

    def iterate(flat: jax.Array, _: Any) -> tuple[jax.Array, jax.Array]:
        slope = slope_of(flat)

        def curvature(direction: jax.Array) -> jax.Array:
            return jax.jvp(slope_of, (flat,), (direction,))[1]

        direction = _steihaug(curvature, slope)
        descent = jnp.sum(slope * direction)
        before = flattened(flat)

        def sufficient(length):
            after = flattened(flat + length * direction)
            return jnp.isfinite(after) & (after - before <= ARMIJO * length * descent)

        def shorter(carry):
            length, count, _ = carry
            return length * 0.5, count + 1, sufficient(length * 0.5)

        length, _, accepted = lax.while_loop(
            lambda carry: (~carry[2]) & (carry[1] < ARMIJO_HALVINGS),
            shorter,
            (jnp.asarray(1.0, flat.dtype), jnp.asarray(0), sufficient(1.0)),
        )
        keep = accepted & (descent < 0.0)
        return jnp.where(keep, flat + length * direction, flat), keep

    polished, kept = lax.scan(iterate, flat0, None, length=iterations)
    return unravel(polished), jnp.sum(kept)


# -------------------------------------------------------------- when to ask --

#: How many consecutive changes of the objective :func:`settled` needs.
SETTLED_CHANGES: int = 2

#: The floor under a relative tolerance on the objective, in units of its
#: machine epsilon: :func:`effective_tol` applies ``max(tol,
#: OBJECTIVE_FLOOR_EPS * eps)``.
#:
#: Without it a tolerance below the working precision's epsilon can be met
#: only if two consecutive evaluations agree to the last bit, which an
#: iterative inner solve does not deliver: on a bilinear model in a 4-byte
#: float the objective at its plateau moves by tens of ulps an iteration.
#: Replaying that trace against an 8-byte reference, a floor of 4 eps never
#: stops; 64 eps stops at iteration 94 and 0.079 of a curvature standard
#: deviation from the minimum; 256 eps at 89 and 0.13.
OBJECTIVE_FLOOR_EPS: int = 64

#: The resolution of a change of the objective, in units of machine epsilon:
#: ``RESOLUTION_EPS * eps * sqrt(sum (s0 + s1)**2)`` over its terms, where
#: ``s`` is each term's rounding magnitude (see :func:`change_program`).
#:
#: Measured against an exactly summed 8-byte reference over 60 random pairs
#: of nearby points on two collinear templates (32, 1e4 and 1e6 terms, in
#: both precisions), the largest error was 0.64 of the multiple-1 estimate;
#: 4 leaves a factor of six.
RESOLUTION_EPS: float = 4.0

#: The factor an inner solve's tolerance is multiplied by when the iteration
#: shows it is inexact, and where that stops, by working precision (bytes).
#: In a 4-byte float the floor is two machine epsilons, below which the
#: solve's residual is rounding; in an 8-byte one it is 1e-12, six digits
#: below the usual default.
SOLVE_TOL_STEP: float = 1e-2
SOLVE_TOL_FLOOR: dict[int, float] = {4: 2.4e-7, 8: 1e-12}


def settled(trace: list[float], tol: float) -> bool:
    """Whether the last :data:`SETTLED_CHANGES` changes of ``trace`` are within ``tol``.

    ``trace`` holds the objective after each iteration. Each change is
    measured in both directions, ``|f[k] - f[k-1]|``, against ``tol *
    max(|f[k]|, 1)``, so a rise larger than that is never settled; and two
    changes rather than one, so an iteration that happens to land at the same
    value from the other side of a minimum is not read as a fixed point.
    """
    if len(trace) < SETTLED_CHANGES + 1:
        return False
    return all(
        abs(trace[-k] - trace[-k - 1]) <= tol * max(abs(trace[-k]), 1.0)
        for k in range(1, SETTLED_CHANGES + 1)
    )


def effective_tol(tol: float, objective: jax.Array) -> float:
    """``max(tol, OBJECTIVE_FLOOR_EPS * eps)`` for the objective's dtype."""
    eps = float(jnp.finfo(jnp.asarray(objective).dtype).eps)
    return max(tol, OBJECTIVE_FLOOR_EPS * eps)


def solve_tol_floor(objective: jax.Array) -> float:
    """How far an inner solve's tolerance may be tightened, by precision.

    :data:`SOLVE_TOL_FLOOR` for the precisions it names, and two machine
    epsilons for any other -- a table lookup would raise on those, and a run
    should not fail because its dtype is unusual.
    """
    precision = jnp.finfo(jnp.result_type(objective))
    return SOLVE_TOL_FLOOR.get(precision.dtype.itemsize, 2.0 * float(precision.eps))


def change_program(resolution_eps: float = RESOLUTION_EPS) -> Callable:
    """``(terms0, scales0, terms1, scales1) -> (decrease, resolution)``, jitted.

    The change of an objective given as per-element TERMS, summed term by
    term, and the rounding that change carries: ``resolution_eps * eps *
    sqrt(sum (scale0 + scale1)**2)``, each term's two evaluations added as
    independent errors.

    Term by term because the difference of two totals is resolved only to
    ``eps |f|``, and ``|f|`` carries every constant the change does not:
    near convergence the two evaluations of one term are within a factor of
    two of each other, so their difference is exact in the working dtype
    (Sterbenz), and the sum of small differences carries almost none of the
    total's rounding.
    """

    @eqx.filter_jit
    def change(terms0, scales0, terms1, scales1):
        decrease = sum(jnp.sum(terms0[key] - terms1[key]) for key in terms1)
        spread = sum(jnp.sum((scales0[key] + scales1[key]) ** 2) for key in scales1)
        eps = jnp.finfo(jnp.result_type(*terms1.values())).eps
        return decrease, resolution_eps * eps * jnp.sqrt(spread)

    return change


@dataclasses.dataclass(frozen=True)
class GapState:
    """What :func:`gap_step` carries from one iteration to the next.

    Attributes:
        contraction: the last contraction estimated from a decrease the
            arithmetic resolved, or ``None`` when there is none (none yet, or
            a rise since).
        moved: whether any decrease so far exceeded its resolution.
        decrease, resolution: the previous iteration's decrease and its
            resolution.
    """

    contraction: float | None = None
    moved: bool = False
    decrease: float | None = None
    resolution: float | None = None


def gap_step(
    state: GapState, decrease: float, resolution: float, tol: float
) -> tuple[GapState, bool, float | None, float | None]:
    """One iteration of the gap PRE-SCREEN: ``(state, passed, gap, rho)``.

    ``decrease`` is ``D[k] = f[k-1] - f[k]`` and ``resolution`` bounds its
    rounding. The contraction is ``max(D[k], 0) / D[k-1]``, clipped to 1,
    re-estimated whenever ``D[k-1]`` exceeds its resolution (the last
    estimate is kept otherwise, and a run that never moved beyond the
    resolution uses 0); the gap is ``(max(D[k], 0) + resolution) / (1 - rho)``,
    the decrease plus its geometric tail, and the iteration passes when that
    is at most ``tol``. A rise the arithmetic resolves passes nothing and
    forgets the contraction: an exact conditional minimisation cannot raise
    the objective.

    This is a screen and not a certificate. The ratio reads the FASTEST mode
    still moving, so a slow mode with a gap left under a fast one's decreases
    passes it -- measured at 0.5 to 1.0 curvature standard deviations on two
    correlated pairs, and at 1.33 over random dense precisions. What
    certifies is :func:`decrement`, which an outer loop computes on an
    iteration that passes this screen, so that it pays one solve per
    candidate stop and not one per iteration.
    """
    if decrease < -resolution:
        return GapState(None, True, decrease, resolution), False, None, None
    contraction = state.contraction
    if state.decrease is not None and state.decrease > state.resolution:
        contraction = min(max(decrease, 0.0) / state.decrease, 1.0)
    moved = state.moved or decrease > resolution
    rho = contraction if moved else 0.0
    following = GapState(contraction, moved, decrease, resolution)
    if rho is None or rho >= 1.0:
        return following, False, None, rho
    gap = (max(decrease, 0.0) + resolution) / (1.0 - rho)
    return following, gap <= tol, gap, rho
