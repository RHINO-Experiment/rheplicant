"""Prior sensitivity: how far the declared prior moved the answer.

Every exit in this package reads ``Latent(prior=...)``. None of them says what
the prior *did*. That is a different question from "is the posterior right",
and it is the one a referee asks: the mode you report sits somewhere between
where the data put it and where the prior wanted it, and the only honest way to
quote it is with the distance stated.

**A chain cannot answer this about itself.** On the tour's nonlinear pair the
declared ``fg_beta ~ Normal(2.3, 0.3)`` moves the mode by **0.0069 sigma**. The
Monte Carlo standard error of a posterior mean from ``n_eff`` draws is
``1/sqrt(n_eff)`` sigma, so seeing 0.0069 sigma at all needs ``n_eff`` of order
2 x 10^4, and *measuring* it needs a second chain with the prior removed to
difference against — two chains whose noise adds in quadrature. A chain that
reaches ``n_eff = 500``, which is a healthy 4 x 1000 NUTS fit, carries an MCSE
of 0.045 sigma: six times the effect it would be looking for. Running longer is
not a small ask, it is a 10^4-fold one, and it answers a question two Newton
solves answer exactly.

Two routes, and they are both deterministic:

**The closed form.** Expand the log-posterior about the mode ``theta_hat``. The
mode the likelihood ALONE would choose is displaced from it by

.. math::

    \\Delta = H^{-1} P \\, (m - \\hat\\theta), \\qquad P = \\mathrm{diag}(s^{-2})

with ``H`` the LIKELIHOOD's curvature at the mode and ``(m, s)`` the declared
prior's location and scale. Not the posterior's ``H + P``, which is the matrix
already in hand for ``sigma_post``: putting that here is wrong by
``diag((H + P)^-1 P)``, the prior's share of the posterior precision, and that
is 6.9e-5 at the tour's declared beta but 5.9e-2 by ``s = 0.01`` — invisible
where the prior is weak and unbounded as it tightens. Per latent, in units of
that latent's own posterior width, the weak-prior form of the diagonal is
``sigma_post * |m - theta_hat| / s^2`` — the law
:attr:`PriorSensitivityReport.criterion_std` inverts — plus a cross term from
every OTHER latent's prior. The cross term is not decoration. On the tour's two
latents, measured:

============  ==========  ==========  ==========
latent        diagonal    cross       total
============  ==========  ==========  ==========
fg_log_amp    -2.66e-04   +2.74e-03   +2.47e-03
fg_beta       -7.03e-03   +1.04e-04   -6.92e-03
============  ==========  ==========  ==========

For ``fg_log_amp`` the cross term is **ten times** the latent's own pull and of
the opposite sign: a per-latent scalar formula would report that this prior
pushes the amplitude down, when what happens is that beta's prior drags it up.
The same reversal is what makes :meth:`PriorSensitivityReport.shift_at` come
back **positive** for beta at ``s = 3.0``, where beta's own prior has gone
loose enough for the amplitude's to take over.

**The refit.** Newton to the mode with the priors on, Newton again with them
off, difference the two. No expansion, no linearisation — the answer the closed
form is approximating. Measured against each other on the tour: 9.6e-6
(log-amp) and 2.1e-6 (beta) relative at the declared priors, which is the
model's nonlinearity over the displacement and nothing else, the derivation
being exact on a quadratic. :meth:`PriorSensitivityReport.shift_at`'s
counterfactual s-ladder is exact on one too, and is the same identity written
about the likelihood mode; what it cannot escape is the same nonlinearity,
which costs it 3.2e-4 by ``s = 0.025`` and 1.8e-3 at ``s = 0.01``, where the
shift has reached six sigma. Which is why both routes are reported and
:attr:`PriorSensitivityReport.verified` says whether they agreed.

**Which of the two is the approximate one depends on the model**, which is the
reason for shipping both rather than picking. On a latent the prediction is
*affine* in — a noise-wave temperature, say — the log-posterior is exactly
quadratic and the closed form is exact, while the refit has to recover a
5e-4 K displacement by differencing two modes of about 290 K each and loses
some six digits to cancellation before its linear solves' own roundoff. The
measured disagreement there is 2.4e-10, and it is the refit's floor, not the
closed form's error: ``290 * eps / 5e-4`` predicts 1.3e-10. On the tour's
nonlinear pair the roles reverse.

**The number to act on is** :attr:`PriorSensitivityReport.criterion_std`: the
prior width at which this latent's shift would reach
:data:`CRITERION_SHIFT`. Inverting the diagonal law gives
``sqrt(sigma_post * |m - theta_hat| / 0.1)``, and for the tour's beta that is
**0.0795** against a declared 0.3 — a factor of 3.8 of margin, which is the
statement "the prior is not driving this fit" with a number attached.

Three things this does NOT do, stated so they are not assumed.

It is **local**. Both routes expand about one mode; a multimodal posterior has
more than one, and the prior's job there may be to select between them, which
is not a displacement and is not measured here.

It reads the **declared** prior only. A ``linear=True`` latent whose prior
arrives as :func:`~rheplicant.inference.linear.wiener_solve`'s ``prior_std=``
keyword is refused by name rather than reported as prior-free — see
:func:`prior_sensitivity`'s ``Raises``.

It needs the **likelihood's** mode to exist. Along a direction the data cannot
see there is no such mode, only a ray, and the displacement from a ray is not a
number — so a rank-deficient selection is refused, and the rank comes from
:func:`~rheplicant.inference.identifiability.identifiability`, which already
knows how to say which latents the degeneracy mixes.
"""

import dataclasses
from collections.abc import Sequence
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from rheplicant.core.errors import ParameterSpaceError, StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State
from rheplicant.inference.identifiability import (
    _check_at,
    _check_differentiable,
    _flat_view,
    _in_float64,
    _resolve_names,
    identifiability,
)
from rheplicant.inference.likelihood import check_observed_shape
from rheplicant.inference.noise import NoiseModelLikelihood
from rheplicant.inference.parameters import ParameterSpace
from rheplicant.inference.uncertainty import as_noise_model

#: The shift :attr:`PriorSensitivityReport.criterion_std` solves for, in
#: posterior sigmas.
#:
#: Chosen against what a chain can see rather than against a convention. The
#: MCSE of a posterior mean from ``n_eff`` effective draws is ``1/sqrt(n_eff)``
#: sigma; a well-run 4 x 1000 NUTS fit lands at ``n_eff`` of a few hundred, so
#: its own noise is 0.04-0.06 sigma. A shift of 0.1 sigma is therefore the
#: smallest bias such a run could distinguish from its own scatter at about
#: 2 sigma of separation — the point below which "is the prior moving this?"
#: stops being answerable by sampling and starts needing this function.
#:
#: It is also small against the thing being biased: 0.1 sigma moves a 68%
#: interval's endpoints by 10% of the interval's half-width, which shifts the
#: reported central value without visibly changing the error bar. That is the
#: regime where a bias is easiest to publish by accident.
CRITERION_SHIFT: float = 0.1

#: Relative tolerance at which the closed form and the refit are called agreed.
#:
#: What it has to cover is the model's NONLINEARITY over the displacement, not
#: an error in the derivation: on an affine model the two routes agree to
#: 2.4e-10, which is the refit's own cancellation floor. Measured on the tour's
#: nonlinear pair, whose mode the priors move by 0.007 sigma: 9.6e-6 (log-amp)
#: and 2.1e-6 (beta). 3e-3 sits two and a half decades above that, deliberately
#: — a posterior that a prior moves by a whole sigma curves over that distance
#: far more than one moved by 0.007 — and a factor of 3 below the 1e-2 scale at
#: which a "0.1 sigma or not" verdict could flip. Pinned in
#: ``tests/inference/test_prior_sensitivity.py``.
VERIFY_RTOL: float = 3e-3

#: Absolute floor under :data:`VERIFY_RTOL`, in posterior sigmas.
#:
#: A relative comparison of two numbers that are both 1e-9 sigma reports the
#: Newton solver's own convergence floor and nothing else. One millionth of a
#: sigma is four decades below :data:`CRITERION_SHIFT` and below any MCSE a
#: chain could reach, so a disagreement smaller than this is not a
#: disagreement about anything.
VERIFY_ATOL: float = 1e-6

#: Newton steps allowed before the solve is called failed.
#:
#: The tour's MAP takes **7** from the declared init and its likelihood-only
#: refit **3** from the MAP, so this is 14x the measured need. It is a ceiling
#: on a loop that either converges quadratically or is not going to.
MAX_NEWTON_STEPS: int = 100

#: Convergence test: ``max(|dx| / (1 + |x|)) < NEWTON_TOL``.
#:
#: Mixed relative/absolute so a latent at 1e-8 and one at 1e8 are held to the
#: same standard. 1e-13 is two decades above float64 eps, which is where a
#: quadratically-converging step size stops shrinking and starts jittering.
NEWTON_TOL: float = 1e-13

#: Halvings of a Newton step allowed before the line search gives up.
#:
#: 2^-40 is 9e-13 of the full step: past that the step is doing nothing and the
#: problem is the direction, not its length.
MAX_BACKTRACKS: int = 40


def _newton(
    objective: Any, x0: jax.Array
) -> tuple[jax.Array, int, bool]:
    """Damped Newton on a scalar objective. Returns ``(x, steps, converged)``.

    Damped rather than plain because the second solve here runs with the priors
    REMOVED, which is exactly the configuration where a full Newton step can
    leave the basin — the priors were the term keeping the Hessian comfortably
    positive. The line search only ever shortens a step, never changes its
    direction, so a converged result is a stationary point of ``objective`` and
    not of some modified problem.

    Never raises: a solve that fails comes back with ``converged=False`` and
    the last iterate, because the caller has to decide whether a failed MAP
    (fatal) or a failed verification refit (reportable) is the situation.
    """
    value_and_grad = jax.value_and_grad(objective)
    hessian = jax.hessian(objective)
    x = x0
    for step in range(1, MAX_NEWTON_STEPS + 1):
        value, gradient = value_and_grad(x)
        direction = jnp.linalg.solve(hessian(x), gradient)
        if not bool(jnp.all(jnp.isfinite(direction))) or not bool(jnp.isfinite(value)):
            return x, step, False
        # `+ eps * |value|` rather than a strict decrease: at the mode the two
        # objective values differ only in their last bits, and a strict test
        # would backtrack 40 times on a converged solve and report failure.
        ceiling = float(value) + 1e-12 * abs(float(value))
        length = 1.0
        for _ in range(MAX_BACKTRACKS):
            trial = x - length * direction
            if float(objective(trial)) <= ceiling:
                break
            length *= 0.5
        else:
            return x, step, False
        moved = float(jnp.max(jnp.abs(length * direction) / (1.0 + jnp.abs(trial))))
        x = trial
        if moved < NEWTON_TOL:
            return x, step, True
    return x, MAX_NEWTON_STEPS, False


def _prior_moments(
    space: ParameterSpace,
    names: Sequence[str],
    shapes: Sequence[tuple[int, ...]],
) -> tuple[jax.Array, jax.Array]:
    """``(loc, scale)`` of every selected latent, flattened. Refuses, loudly.

    The two refusals here are the ones that keep this function from reporting a
    Gaussian displacement for something that has none. Both name the latent.
    """
    # Function-local, the same way `uncertainty._prior_precision` imports it:
    # `linear` owns what "a Gaussian prior ON THE LATENT" means — it is the
    # module that refuses LogNormal, which carries .loc, .scale AND a Normal
    # .base_dist while being Gaussian in log x, not in x.
    from rheplicant.inference.linear import _gaussian_parameters

    locations: list[jax.Array] = []
    scales: list[jax.Array] = []
    for name, shape in zip(names, shapes, strict=True):
        latent = space.latent(name)
        prior = latent.prior
        if prior is None and latent.linear:
            raise ParameterSpaceError(
                f"prior_sensitivity was asked about latent {name!r}, which is declared "
                "linear=True with no Latent(prior=...). A linear latent's prior reaches "
                "the solve as the prior_std=/prior_mean= arguments of wiener_solve or "
                "gcr_sample, and those are CALL-SITE values: they are supplied where the "
                "solve is called and nothing in this ParameterSpace records them. So the "
                "sensitivity reported here would be a sensitivity to a prior the solve "
                "never reads — finite, confidently shaped, and about a different "
                "posterior than the one you get. Move the prior into the declaration "
                "(Latent(..., prior=dist.Normal(m, s)), which wiener_solve reads and "
                f"reconciles against prior_std=), or drop {name!r} from names=."
            )
        if prior is None:
            raise ParameterSpaceError(
                f"prior_sensitivity was asked about latent {name!r}, which declares no "
                "prior at all. There is no prior for the answer to be sensitive TO, and "
                "a free parameter cannot be moved by one: what would come back is 0.0, "
                "reported in the same column as latents whose shift was measured, where "
                "it reads as 'this prior is harmless' rather than 'there is no prior'. "
                "Declare Latent(prior=dist.Normal(m, s)) if there is one, or drop "
                f"{name!r} from names=."
            )
        gaussian = _gaussian_parameters(prior)
        if gaussian is None:
            raise ParameterSpaceError(
                f"prior_sensitivity was asked about latent {name!r}, which declares a "
                f"{type(prior).__name__} prior. The shift this reports is the "
                "displacement a Gaussian's quadratic pull puts on the mode — "
                "sigma_post * |m - theta_hat| / s^2 — and a "
                f"{type(prior).__name__} has no quadratic form to put in it. Reading its "
                "midpoint and width as (m, s) would report a smooth pull for a prior "
                "that exerts none at all inside its support and an unbounded one outside "
                "it: a finite number, of the wrong sign as often as not, describing a "
                "bias that is not the bias this prior applies. Sample the space with "
                "to_numpyro_model + NUTS, which honours the prior as written, or exclude "
                f"{name!r} with names=."
            )
        loc, scale = gaussian
        locations.append(jnp.ravel(jnp.broadcast_to(jnp.asarray(loc, dtype=jnp.float64), shape)))
        scales.append(jnp.ravel(jnp.broadcast_to(jnp.asarray(scale, dtype=jnp.float64), shape)))
    return jnp.concatenate(locations), jnp.concatenate(scales)


def _refuse_rank_deficient(
    space: ParameterSpace,
    pipeline: AbstractOperator,
    state_template: State,
    names: Sequence[str],
    values: dict[str, jax.Array],
) -> None:
    """Refuse a selection whose Jacobian at the mode is rank-deficient.

    The rank verdict, the null direction and the participation shares all come
    from :func:`~rheplicant.inference.identifiability.identifiability`, which
    is where that measurement lives; this only decides that a shift cannot be
    reported without it.
    """
    report = identifiability(space, pipeline, state_template, names=names, at=values)
    if not report.nullity:
        return
    participation = report.participation(0)
    mixed = ", ".join(
        f"{name} {share:.2f}" for name, share in sorted(
            participation.items(), key=lambda item: -item[1]
        )
    )
    raise ParameterSpaceError(
        f"prior_sensitivity cannot report a prior shift for {list(names)}: at the mode, "
        f"identifiability() reports rank {report.rank} of {report.n_par}, so "
        f"{report.nullity} direction(s) of this selection are ones the data cannot see "
        f"at all. The first of them mixes {mixed} (participation, in column-normalised "
        "coordinates). The shift is defined as the displacement from the mode the "
        "LIKELIHOOD alone would choose, and along a null direction there is no such "
        "mode — it is a ray, and the distance from a ray is not a number. What would "
        "come back is nevertheless finite, because the declared priors make the "
        "posterior proper: it would be the prior reporting on itself, with a "
        "well-formed sigma and a plausible magnitude. Call identifiability() for the "
        "full direction, fix the parameterization, or restrict names= to a subset the "
        "data determines."
    )


@dataclasses.dataclass(frozen=True)
class PriorSensitivityReport:
    """What the declared priors did to the mode, per latent, in posterior sigmas.

    A plain frozen dataclass holding **numpy**, for the same reasons
    :class:`~rheplicant.inference.identifiability.IdentifiabilityReport` is
    one: this is a derived verdict rather than a differentiable model, its
    float64 contents would silently truncate the moment a default-precision JAX
    caller touched them, and ``verified`` is a decision, which is not something
    a traced program can branch on.

    Every array is flat over the SELECTED latents, in the order they were asked
    for — :attr:`names` and :attr:`spans` are the only coordinate system in the
    object. That order is the declaration order, **not** sorted order, and the
    difference is live: :func:`~rheplicant.inference.uncertainty.
    fisher_information` flattens its dict in sorted key order, so on a space
    declaring ``("fg_log_amp", "fg_beta")`` its rows come back the other way
    round. Borrowing a row from there by position would hand back one latent's
    width under another's name, with every shape agreeing.

    Attributes:
        names: the latents analysed, in the order the caller asked for.
        shapes: their shapes, in the same order.
        spans: ``(start, stop)`` of each latent in the flat vector.
        n_par: total number of real parameters.
        mode: ``theta_hat``, the MAP found by Newton on the exact log-posterior.
        prior_loc: the declared ``m``, broadcast per element.
        prior_std: the declared ``s``, broadcast per element.
        mean_offset: ``|m - theta_hat|``. A magnitude — the direction of the
            pull lives in :attr:`shift_sigma`'s sign, where a reader will look
            for it.
        sigma_post: ``sqrt(diag(Sigma))`` at the mode, from the exact Hessian of
            the negative log-posterior. Not
            :func:`~rheplicant.inference.uncertainty.fisher_information`'s
            expected information: the two differ by 3.3e-4 relative in the
            tour's beta block, and the observed curvature is the one the Newton
            refit walks on, so it is the one the two routes have to share if
            their agreement is to mean anything.
        shift_sigma: the closed form, signed, in units of :attr:`sigma_post`.
            Negative means the prior pulled the latent DOWN.
        shift_sigma_refit: the same displacement from an actual second Newton
            solve with the priors removed. All-NaN if that solve did not
            converge, in which case :attr:`refit_converged` is ``False``.
        verified: per element, whether the two routes agreed to
            :data:`VERIFY_RTOL` (with :data:`VERIFY_ATOL` underneath).
        criterion_std: the prior width at which this latent's shift would reach
            :data:`CRITERION_SHIFT`, from the diagonal law
            ``sqrt(sigma_post * mean_offset / 0.1)``. Compare it with
            :attr:`prior_std`: a declared width comfortably ABOVE it is the
            statement that the prior is not driving the fit. ``0.0`` when the
            prior mean sits exactly on the mode, which means no tightening of
            it moves anything.
        precision: the ``(n_par, n_par)`` posterior precision at the mode.
        newton_steps, refit_steps: what the two solves cost.
        refit_converged: whether the likelihood-only solve reached a mode.
    """

    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    spans: tuple[tuple[int, int], ...]
    n_par: int
    mode: np.ndarray
    prior_loc: np.ndarray
    prior_std: np.ndarray
    mean_offset: np.ndarray
    sigma_post: np.ndarray
    shift_sigma: np.ndarray
    shift_sigma_refit: np.ndarray
    verified: np.ndarray
    criterion_std: np.ndarray
    precision: np.ndarray
    newton_steps: int
    refit_steps: int
    refit_converged: bool

    #: Which flat arrays :meth:`for_latent` splits. A ``ClassVar`` and not a
    #: field: annotated without it, a dataclass would make this the tenth
    #: positional argument of the constructor and put the list of field names
    #: inside the report's own repr.
    _PER_ELEMENT: ClassVar[tuple[str, ...]] = (
        "mode",
        "prior_loc",
        "prior_std",
        "mean_offset",
        "sigma_post",
        "shift_sigma",
        "shift_sigma_refit",
        "verified",
        "criterion_std",
    )

    def _index(self, name: str) -> int:
        if name not in self.names:
            raise StateValidationError(
                f"This report says nothing about {name!r}; it covers "
                f"{list(self.names)}. Pass it in names= to have it analysed."
            )
        return self.names.index(name)

    def for_latent(self, name: str) -> dict[str, np.ndarray]:
        """Every per-element quantity for one latent, reshaped like the latent.

        The form a caller acts in: ``report.for_latent("fg_beta")["shift_sigma"]``
        is a number about beta, not an offset into a vector whose layout the
        caller has to have got right.
        """
        index = self._index(name)
        start, stop = self.spans[index]
        shape = self.shapes[index]
        return {
            field: np.asarray(getattr(self, field))[start:stop].reshape(shape)
            for field in self._PER_ELEMENT
        }

    def mode_of(self, name: str) -> np.ndarray:
        """``theta_hat`` for one latent, shaped like it."""
        return self.for_latent(name)["mode"]

    @property
    def worst(self) -> tuple[str, int, float]:
        """``(latent, index within it, signed shift)`` of the largest |shift|.

        What to print first. An anonymous argmax over the flat vector would
        name a position in a layout the caller did not choose.
        """
        flat = int(np.argmax(np.abs(self.shift_sigma)))
        for name, (start, stop) in zip(self.names, self.spans, strict=True):
            if start <= flat < stop:
                return name, flat - start, float(self.shift_sigma[flat])
        raise StateValidationError(  # pragma: no cover - spans tile [0, n_par)
            f"Inconsistent report: flat index {flat} falls in no latent's span "
            f"{list(self.spans)} over {self.n_par} parameters."
        )

    def shift_at(self, name: str, prior_std: Any) -> np.ndarray:
        """The shift this latent would suffer under a DIFFERENT prior width.

        The counterfactual :attr:`criterion_std` inverts, evaluated exactly
        rather than through the diagonal law: only ``name``'s entries of ``P``
        are replaced, and ``(H + P_s)^-1 P_s (I + H^-1 P_d)(m - theta_hat)`` is
        solved whole. Every other latent's prior stays as declared, cross terms
        included — which is why this can return a shift of the opposite sign to
        ``name``'s own pull once ``name``'s prior is loose enough for a
        neighbour's to dominate.

        **Anchoring.** That is not the expression :attr:`shift_sigma` uses, and
        the comment on the solve below derives why: the two are the same
        displacement written about different modes, and a counterfactual can
        only stand on the likelihood's. The ``(I + H^-1 P_d)`` factor is how it
        gets there without a second fit, and it makes the whole thing exact on
        a quadratic. Two consequences worth stating. At ``P_s = P_d`` it
        collapses to :attr:`shift_sigma` algebraically, so ``shift_at(name,
        declared_width)`` returns the reported shift to the last bit — if it
        did not, one of the two would be wrong. And what is left at tight
        widths is the model, not the method: against an actual re-run at the
        hypothesised width, 1.6e-5 at ``s = 0.1`` and 1.8e-3 at ``s = 0.01``,
        the latter being the tour's nonlinearity over six sigma of travel.

        Reported in the sigma the caller actually HAS — :attr:`sigma_post`, at
        the declared priors — and not in the sigma the counterfactual prior
        would produce. Dividing each row of a ladder by its own width would
        fold the prior's shrinking of the error bar into a number meant to
        report only the movement of the mode, and 'a 0.1 sigma shift' would
        then mean a different displacement in every row.

        Args:
            name: the latent to re-prior.
            prior_std: its hypothetical width — a scalar, or anything
                broadcastable to the latent's shape.

        Returns:
            The signed shift in posterior sigmas, shaped like the latent.

        Raises:
            StateValidationError: if ``name`` is not in this report, or the
                width is not positive and finite.
        """
        index = self._index(name)
        start, stop = self.spans[index]
        shape = self.shapes[index]
        try:
            replacement = np.broadcast_to(
                np.asarray(prior_std, dtype=np.float64), shape
            ).ravel()
        except (TypeError, ValueError) as error:
            raise StateValidationError(
                f"prior_std={prior_std!r} does not broadcast to {name!r}'s shape "
                f"{shape}, so there is no width to put on {stop - start} of its "
                "elements."
            ) from error
        if not np.all(np.isfinite(replacement)) or np.any(replacement <= 0.0):
            raise StateValidationError(
                f"shift_at({name!r}, {prior_std!r}) needs a positive, finite prior "
                "width: the shift goes as 1/s^2, so a zero or negative s is a "
                "division by zero or a NEGATIVE prior precision — an anti-prior that "
                "pushes the mode away from its own mean, and would come back as a "
                "finite shift with the sign reversed. For 'no prior at all', pass a "
                "width large against the posterior's, which is the limit s -> inf."
            )
        scales = self.prior_std.copy()
        scales[start:stop] = replacement
        # (H + P_s)^-1 P_s (I + H^-1 P_d) (m - theta_hat), and the shape of it
        # is the anchor rather than the algebra. Two identities hold exactly on
        # a quadratic:
        #
        #     theta_s - theta_L = H^-1 P_s (m - theta_s)          [anchored at theta_s]
        #                       = (H + P_s)^-1 P_s (m - theta_L)  [anchored at theta_L]
        #
        # `shift_sigma` reports the DECLARED prior, where theta_hat IS theta_s,
        # so it uses the first and is exact. A counterfactual has no theta_s --
        # finding it is the refit this method exists to avoid -- so it must use
        # the second, whose anchor theta_L it does not hold either. Standing
        # theta_hat in for theta_L raw costs diag((H + P)^-1 P), 6.9e-5 on the
        # tour's beta, and standing it in for theta_s instead (the first form,
        # evaluated where it does not apply) costs 6.43 % at s = 0.01.
        #
        # Neither substitution is necessary, because theta_L is not unknown:
        # the declared solve already displaced the mode by a measured amount,
        # and running that displacement backwards is exact --
        #
        #     m - theta_L = (I + H^-1 P_d) (m - theta_hat)
        #
        # -- which is the (I + H^-1 P_d) here. With it the counterfactual is
        # exact on a quadratic, and at s = s_declared the two P's coincide and
        # the whole thing collapses to H^-1 P_d (m - theta_hat): the ladder's
        # declared row IS `shift_sigma`, to roundoff, rather than 6.9e-5 away
        # from it. Measured against an actual re-run at the hypothesised width
        # on the tour's beta: 1.6e-5 at s = 0.1 and 1.8e-3 at s = 0.01, where
        # both routes are at the model's nonlinearity floor and `verified`
        # says so.
        diagonal = np.arange(self.n_par)
        likelihood_precision = self.precision.copy()
        likelihood_precision[diagonal, diagonal] -= 1.0 / self.prior_std**2
        offset = self.prior_loc - self.mode
        to_likelihood_mode = offset + np.linalg.solve(
            likelihood_precision, offset / self.prior_std**2
        )
        precision = likelihood_precision.copy()
        precision[diagonal, diagonal] += 1.0 / scales**2
        shift = np.linalg.solve(precision, to_likelihood_mode / scales**2)
        return (shift / self.sigma_post)[start:stop].reshape(shape)


def prior_sensitivity(
    space: ParameterSpace,
    pipeline: AbstractOperator,
    state_template: State,
    observed: jax.Array,
    noise_std: Any,
    flags: jax.Array | None = None,
    *,
    names: Sequence[str] | None = None,
    at: dict[str, jax.Array] | None = None,
) -> PriorSensitivityReport:
    """How far the declared priors moved the mode, in posterior sigmas.

    See the module docstring for why a NUTS run cannot be asked this and for
    the two routes taken instead. Both are deterministic; nothing here samples.

    The work is two Newton solves — one on the exact log-posterior, one on the
    likelihood alone — plus one dense Jacobian for the rank check. That is a
    design-time cost for tens to a few thousand parameters, the same envelope
    :func:`~rheplicant.inference.identifiability.identifiability` states, and
    for the same reason: a dense SVD is taken.

    Args:
        space: the parameter declaration. Every SELECTED latent must declare a
            Gaussian ``Latent(prior=...)``.
        pipeline: the forward model.
        state_template: the state it is evaluated on.
        observed: the data. Must match the prediction's shape exactly.
        noise_std: a scalar, an array, or a
            :class:`~rheplicant.inference.noise.NoiseModel` — the same seam
            every other exit takes.
        flags: optional boolean mask; ``True`` = not observed.
        names: which latents to analyse — a sequence, or a bare string for one.
            ``None`` means all of them, in declaration order. A subset asks the
            CONDITIONAL question, holding the rest at ``at``: the tour's beta
            has a marginal width of 2.499e-3 and a width of 2.302e-3 once the
            foreground amplitude is pinned, and those are different sigmas to
            report a shift in.
        at: values for the latents NOT selected, and the Newton starting point
            for those that are. Defaults to the space's declared initial values.

    Returns:
        A :class:`PriorSensitivityReport`. The two numbers to read first are
        ``shift_sigma`` and ``criterion_std``::

            report = prior_sensitivity(space, model, state, data, sigma)
            name, index, shift = report.worst
            criterion = report.for_latent(name)["criterion_std"].ravel()[index]
            print(f"{name}[{index}] moved {shift:+.4f} sigma by its prior; "
                  f"0.1 sigma would need s = {criterion:.3g}")

    Raises:
        ParameterSpaceError: if ``names``/``at`` name an undeclared latent;
            if ``observed`` does not match the prediction's shape; if a
            selected latent is complex or non-floating; if a selected latent
            declares no prior, declares one with no quadratic form (a Uniform,
            a LogNormal, a Half-Normal), or is ``linear=True`` with its prior
            living in a ``prior_std=`` call-site argument this function cannot
            see; or if the selection's Jacobian at the mode is rank-deficient.
        StateValidationError: if the Newton solve for the mode does not
            converge.

    Note:
        Runs with ``jax_enable_x64`` forced on and restores the caller's
        setting afterwards, including on the way out of an exception. The
        setting is process-global, so this is not thread-safe against other
        JAX work.
    """
    selected = _resolve_names(space, names)
    _check_at(space, at)
    _check_differentiable(space, selected)

    with _in_float64():
        forward, values0 = space.forward_fn(pipeline, state_template)
        values0 = {**values0, **(at or {})}
        data = jnp.asarray(observed, dtype=jnp.float64)
        check_observed_shape(
            jnp.shape(forward(values0)), data, predictor="this forward model"
        )
        noise = as_noise_model(
            noise_std,
            flags,
            prediction_shape=jnp.shape(data),
            caller="prior_sensitivity",
        )
        log_likelihood = NoiseModelLikelihood(noise)

        x0, shapes, spans = _flat_view(values0, selected)
        loc, scale = _prior_moments(space, selected, shapes)

        def unflatten(x: jax.Array) -> dict[str, jax.Array]:
            return {
                name: jnp.reshape(x[start:stop], shape)
                for name, shape, (start, stop) in zip(selected, shapes, spans, strict=True)
            }

        def neg_log_likelihood(x: jax.Array) -> jax.Array:
            return -log_likelihood(forward({**values0, **unflatten(x)}), data)

        def neg_log_posterior(x: jax.Array) -> jax.Array:
            return neg_log_likelihood(x) + 0.5 * jnp.sum(((x - loc) / scale) ** 2)

        mode, newton_steps, converged = _newton(neg_log_posterior, x0)
        if not converged:
            raise StateValidationError(
                f"prior_sensitivity could not find the mode: {newton_steps} damped "
                "Newton steps on the exact log-posterior did not converge to "
                f"max|dx|/(1+|x|) < {NEWTON_TOL:g}. Every number this reports is a "
                "displacement FROM that mode, so there is nothing to report — the "
                "closed form would expand about a point that is not stationary and "
                "come back finite. Three things do this: a starting point in another "
                "basin (pass at=), a model that computes its prediction in float32 "
                "(the Hessian is then noise at the 1e-7 level — identifiability() "
                "names that one), and a genuinely non-quadratic posterior, for which "
                "to_numpyro_model + NUTS is the right tool and this one is not."
            )

        precision = jax.hessian(neg_log_posterior)(mode)
        covariance = jnp.linalg.inv(precision)
        sigma_post = jnp.sqrt(jnp.diag(covariance))

        values_at_mode = {**values0, **unflatten(mode)}
        _refuse_rank_deficient(space, pipeline, state_template, selected, values_at_mode)

        offset = loc - mode
        # theta_hat - theta_L = H^-1 P (m - theta_hat), with H the LIKELIHOOD
        # curvature -- not the posterior's. Using (H + P)^-1 here is wrong by
        # exactly diag((H + P)^-1 P), the prior's share of the posterior
        # precision: 5.9e-2 at the tour's beta with s = 0.01, and it grows
        # without bound as the prior tightens, which is the regime the report
        # exists to describe.
        likelihood_precision = jax.hessian(neg_log_likelihood)(mode)
        shift_sigma = (
            jnp.linalg.solve(likelihood_precision, offset / scale**2) / sigma_post
        )

        likelihood_mode, refit_steps, refit_converged = _newton(neg_log_likelihood, mode)
        if refit_converged:
            shift_refit = (mode - likelihood_mode) / sigma_post
            agreed = jnp.abs(shift_sigma - shift_refit) <= (
                VERIFY_RTOL * jnp.abs(shift_refit) + VERIFY_ATOL
            )
        else:
            shift_refit = jnp.full_like(shift_sigma, jnp.nan)
            agreed = jnp.zeros_like(shift_sigma, dtype=bool)

        mean_offset = jnp.abs(offset)
        criterion = jnp.sqrt(sigma_post * mean_offset / CRITERION_SHIFT)

        # float64 explicitly rather than by inheritance: these arrays leave the
        # x64 context, and an array that arrived here as float32 would be a
        # report whose digits stop before the effect it is measuring does.
        def as_numpy(array: jax.Array) -> np.ndarray:
            return np.asarray(array, dtype=np.float64)

        return PriorSensitivityReport(
            names=selected,
            shapes=shapes,
            spans=spans,
            n_par=int(x0.size),
            mode=as_numpy(mode),
            prior_loc=as_numpy(loc),
            prior_std=as_numpy(scale),
            mean_offset=as_numpy(mean_offset),
            sigma_post=as_numpy(sigma_post),
            shift_sigma=as_numpy(shift_sigma),
            shift_sigma_refit=as_numpy(shift_refit),
            verified=np.asarray(agreed, dtype=bool),
            criterion_std=as_numpy(criterion),
            precision=as_numpy(precision),
            newton_steps=int(newton_steps),
            refit_steps=int(refit_steps),
            refit_converged=bool(refit_converged),
        )


__all__ = [
    "CRITERION_SHIFT",
    "MAX_BACKTRACKS",
    "MAX_NEWTON_STEPS",
    "NEWTON_TOL",
    "VERIFY_ATOL",
    "VERIFY_RTOL",
    "PriorSensitivityReport",
    "prior_sensitivity",
]
