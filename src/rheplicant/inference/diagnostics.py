"""What a campaign can say about its own trustworthiness, from stored terms alone.

A separate module from :mod:`rheplicant.inference.memory` for two reasons. The
practical one: ``memory.py`` is already near this project's 800-line ceiling.
The one that matters: a diagnostic is something you run **on** a memory, not
something a memory is, and the import graph should say so. Nothing here reads a
pipeline, a forward model or a byte of raw data -- only the fixed-size summaries
:func:`~rheplicant.inference.compress.compress_linear` stored while the data
still existed.

**The blindness is the design, not a gap.** Section 9's honest content is that a
deterministic error shared across every night splits into two halves, and only
one of them is visible from data at all:

* the **out-of-span** half leaves a residual, and is what
  :func:`coherent_mode` reports -- at sqrt(N), because a mean over N epochs is
  resolved at sqrt(N);
* the **in-span** half is absorbed into theta identically in every epoch. It
  leaves no residual, so it passes chi-square, split-half and leave-one-out, and
  it biases the answer without limit as the campaign grows.

That is why sections 9.4 and 9.5 are refusals based on what the analyst
*declares* rather than reports based on the numbers: there is no statistic to
improve.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from bayesmith.marginal.diagnostics import shrinkage_power as _far_shrinkage_power
from bayesmith.marginal.diagnostics import shrinkage_report as _far_shrinkage_report

from rheplicant.core.errors import StateValidationError

_PRIOR_REMEDY = (
    "pass prior_fisher over the same latents in the same column order -- "
    "section 2.2 says a single epoch legitimately constrains only a subspace, "
    "so the leave-one-out information is singular at small N without one."
)


@dataclass(frozen=True)
class EpochResidual:
    """One night's row of section 9.3's table.

    A plain frozen dataclass rather than an ``eqx.Module``: this is a report, it
    is never differentiated, never jitted and never stored, and making it a
    pytree would invite someone to put it somewhere that flattens.

    Attributes:
        epoch_id: the recording's data hash, as the term carries it.
        chi2: the epoch's residual chi-square, after its own best fit.
        dof: unflagged samples minus the rank of what the epoch fitted.
        reduced_chi2: ``chi2 / dof``, or ``nan`` when ``dof`` is zero -- an
            epoch whose design saturates its data has no residual to speak of,
            and a zero here would read as a perfect fit.
        templates: ``{name: projection}``, each a standard normal under the
            null.
    """

    epoch_id: str
    chi2: float
    dof: int
    reduced_chi2: float
    templates: dict[str, float]


def epoch_residuals(terms: Sequence[Any]) -> tuple[EpochResidual, ...]:
    """Section 9.3's per-epoch table, in the order the terms were given.

    Order is preserved rather than sorted by anything: for a
    :class:`~rheplicant.inference.chain.ChainMemory` the archive order **is**
    the campaign's time order, and a diagnostic that reordered it would make a
    drift look like scatter.
    """
    rows = []
    for term in terms:
        dof = int(term.residual_dof)
        chi2 = float(np.asarray(term.residual_chi2))
        projections = (
            {}
            if term.template_projections is None
            else {
                name: float(np.asarray(term.template_projections)[index])
                for index, name in enumerate(term.template_names)
            }
        )
        rows.append(
            EpochResidual(
                epoch_id=term.epoch_id,
                chi2=chi2,
                dof=dof,
                reduced_chi2=chi2 / dof if dof > 0 else float("nan"),
                templates=projections,
            )
        )
    return tuple(rows)


def _refuse_mixed_templates(terms: Sequence[Any]) -> tuple[str, ...]:
    """Every epoch must name the same templates, in the same order.

    The summary is read **by name across a whole campaign**, and the projections
    are stored as a bare array whose meaning is positional. Two nights that name
    ``("gain_ripple", "ground_pickup")`` and ``("ground_pickup", "gain_ripple")``
    have compatible shapes and incompatible contents, so averaging them
    positionally would report each template's mean as a mixture of both -- a
    finite, plausible number about nothing, which is precisely the failure this
    layer exists to refuse.
    """
    declared = {term.template_names for term in terms}
    if len(declared) > 1:
        raise ValueError(
            f"These epochs name different systematic templates: "
            f"{sorted(declared)}. The projections are stored positionally, so "
            "averaging across a campaign requires one list in one order. "
            "Recompress the odd epochs with the same templates= mapping, or "
            "diagnose the subsets separately."
        )
    return next(iter(declared)) if declared else ()


def coherent_mode(terms: Sequence[Any]) -> dict[str, Any]:
    """Is there a fault common to every night? -- section 9.3's whole question.

    A deterministic error shared across epochs -- one calibration solution, one
    beam model, one flag table -- contributes **no variance**. Split-half agrees
    to roundoff, leave-one-out returns the same scores, the posterior width is
    the same array element for element, and the answer is wrong. What it does
    move is a **mean**, and a mean over N epochs is resolved at sqrt(N), which is
    why this reports z-scores rather than magnitudes.

    Section 9's list of what a common mode passes includes "per-epoch
    chi-square", and measurement says otherwise: its mean is exactly what
    ``chi2_z`` below reports as a detection, and its scatter is inflated by
    noncentrality -- 5.5467 against ``sqrt(2 * 6) = 3.4641`` on this fixture,
    with no new randomness injected. The statistic whose scatter a shift really
    does leave alone is the named template projection, below.

    Two statistics, and neither is a substitute for the other. ``chi2_z`` needs
    no guess about what the fault looks like and therefore cannot say what it
    is; a named template says what it is and is silent when the guess was wrong
    -- measured on the repeated-design fixture, a template orthogonal to the
    true mode gives ``z = -1.01`` while ``chi2_z`` still gives ``+31.92``.

    ``scatter`` is reported beside every mean and is the part a reader should
    look at second: a mean-level fault leaves it at 1.0 -- measured 1.0020 on
    the clean campaign and 1.0020 on the biased one, the same number to four
    decimals -- while an under-estimated noise model raises both together.

    **What this cannot see.** The half of a coherent error lying inside the
    design's column space is absorbed into theta identically in every epoch. It
    leaves no residual, so it is invisible here, invisible to a held-out z, and
    invisible to split-half. On the fixture this docstring's numbers come from,
    that half has whitened norm ``2.3070`` against the visible half's ``2.0094``
    -- comparable, and one of them reportable. That is not a gap to be closed
    with a better statistic; it is why sections 9.4 and 9.5 are refusals based
    on what the analyst declares.

    Returns:
        ``{"n_epochs", "chi2_mean", "chi2_dof", "chi2_z", "templates"}``, where
        ``templates`` is ``{name: {"mean", "scatter", "z"}}``.

    Raises:
        ValueError: if the epochs do not all name the same templates, or if the
            campaign is empty.
    """
    terms = tuple(terms)
    if not terms:
        raise ValueError(
            "coherent_mode needs at least one epoch. An empty campaign has no "
            "mean to resolve, and returning zeros would read as a clean result."
        )
    names = _refuse_mixed_templates(terms)
    rows = epoch_residuals(terms)
    n = len(rows)

    chi2 = np.array([row.chi2 for row in rows])
    dofs = {row.dof for row in rows}
    # Every epoch's DOF, not the first one's: a campaign whose nights differ in
    # flagging has a different null per night, and the sum is what is chi-square
    # distributed. Var(sum of chi2_k) = 2 sum(k), so the z below is exact for a
    # ragged campaign and reduces to the textbook form for a uniform one.
    total_dof = sum(row.dof for row in rows)
    spread = np.sqrt(2.0 * total_dof) if total_dof > 0 else np.nan
    report: dict[str, Any] = {
        "n_epochs": n,
        "chi2_mean": float(chi2.mean()),
        "chi2_dof": sorted(dofs)[0] if len(dofs) == 1 else None,
        "chi2_z": float((chi2.sum() - total_dof) / spread),
        "templates": {},
    }
    for index, name in enumerate(names):
        values = np.array([row.templates[name] for row in rows])
        report["templates"][name] = {
            "mean": float(values.mean()),
            # Population spread, not the sample one: under the null each
            # projection is a standard normal in its own right, so the quantity
            # being checked against 1.0 is the second moment about the *known*
            # null mean of zero -- but the mean is reported separately and a
            # shifted-mean fault must not inflate this. `ddof=0` about the
            # sample mean is the compromise, and it is the same 1.0020 on
            # the clean campaign and on the biased one -- which is the
            # statement that the fault is a shift and not extra noise.
            "scatter": float(values.std()),
            # sqrt(N), which is the whole point: the mean does not shrink with
            # N and its uncertainty does.
            "z": float(values.mean() * np.sqrt(n)),
        }
        del index
    return report


@dataclass(frozen=True)
class HeldOut:
    """One night's row of section 9.1's table.

    A frozen dataclass for the same reason :class:`EpochResidual` is one: a
    report is never differentiated, never jitted and never stored.

    Attributes:
        epoch_id: the recording's identity, as the term carries it.
        chi2: ``m^T (I + V)^-1 m`` for the held-out residual ``m``.
        dof: rows of the epoch's factor -- the dimension of ``m``.
        z: ``(chi2 - dof) / sqrt(2 dof)``, a standard normal under the model.
    """

    epoch_id: str
    chi2: float
    dof: int
    z: float


def _campaign_arrays(
    terms: Sequence[Any],
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], int]:
    """``(factors, targets, width)`` for a campaign, with the latents checked.

    The columns of every stored factor are positional. Two terms declared over
    different latents -- or the same names in a different order -- have
    compatible shapes and incompatible contents, so summing their information
    would produce a finite, plausible posterior about nothing. That is the same
    failure :func:`_refuse_mixed_templates` refuses one level along, and it is
    refused here rather than in ``numpy``, which would only complain when the
    widths happened to differ too.
    """
    if not terms:
        raise ValueError(
            "held_out_z needs at least one epoch. An empty campaign has no night "
            "to hold out, and an empty result would read as 'no epoch was "
            "surprising'."
        )
    declared = {(term.info.names, term.info.shapes) for term in terms}
    if len(declared) > 1:
        raise ValueError(
            "These epochs are over different latents: "
            f"{sorted(names for names, _ in declared)}. A stored factor's columns "
            "are positional, so a campaign scored across two parameter sets would "
            "add one epoch's information to another epoch's columns -- finite, "
            "plausible and about nothing. Score the subsets separately."
        )
    factors = tuple(np.asarray(term.info.factor, dtype=float) for term in terms)
    targets = tuple(np.asarray(term.info.target, dtype=float) for term in terms)
    # An epoch's `dof` is its factor's row count, and `z` divides by
    # `sqrt(2 * dof)`. A night whose samples are all flagged compresses to zero
    # rows -- so does `SqrtInfo.null`, which the accumulator uses as padding --
    # and reached that division as a bare `ZeroDivisionError: float division by
    # zero`, one line below the NaN guard written for the same shape of problem
    # and unable to help because the division happens first. Refused here, with
    # the whole campaign in hand, so a long run with one dead night says which
    # night before it spends the other N-1 leave-one-out solves.
    empty = [term.epoch_id for term, row in zip(terms, factors, strict=True) if not row.shape[0]]
    if empty:
        raise ValueError(
            f"Epoch(s) {empty} have no rows in their stored factor, so they "
            "constrain nothing and have no residual to score: the held-out "
            "chi-square is on zero degrees of freedom and its standardisation "
            "divides by sqrt(2 * 0). That is a night whose samples were all "
            "flagged, or a null padding term that was never replaced. Drop "
            "those epochs from the campaign -- a score of 'unsurprising' for a "
            "night with no data is not a diagnostic result."
        )
    return factors, targets, factors[0].shape[1]


def held_out_z(
    terms: Sequence[Any], prior_fisher: Any, prior_mean: Any = None
) -> tuple[HeldOut, ...]:
    """Section 9.1: how surprising is each night to the rest of the campaign?

    For a linear-Gaussian model this is exact and needs no simulation. Write the
    leave-one-out posterior ``N(mu_{-e}, Sigma_{-e})`` and the epoch's own factor
    ``[R_e | z_e]``. Then ``z_e = R_e theta_true + eps`` with unit-covariance
    ``eps`` -- that is what the square-root form means -- and ``mu_{-e} -
    theta_true`` is independent of ``eps``, so

        ``m = R_e mu_{-e} - z_e  ~  N(0, I + R_e Sigma_{-e} R_e^T)``

    and ``m^T (I + V)^-1 m`` is chi-square on ``rank(R_e)`` degrees of freedom.
    The returned ``z`` is that, standardised.

    Computed from the archive rather than by downdating the accumulator: a QR
    accumulation cannot be un-summed stably. The campaign total is formed once
    in ``(F, b)`` form and one epoch's contribution subtracted per row, which is
    ``O(N)`` rather than ``O(N^2)`` and loses at most ``log10(N)`` digits of the
    sixteen float64 carries -- affordable because this is an offline diagnostic,
    and safe because the subtraction is of one PSD summand out of N, not of a
    triangular factor out of its own product.

    **What it can see, measured.** On a campaign whose nights genuinely differ,
    a single rogue epoch scores ``+72.96`` while the largest of the other 59
    scores ``4.24``; and a common-mode error over 300 varying nights lifts the
    campaign mean to ``+22.05`` sigma against ``+0.87`` for the clean run.

    **What it cannot see, measured.** If every night carries the same design --
    the realistic case, and the one section 1 describes -- a coherent error's
    in-span half shifts ``z_e`` and ``mu_{-e}`` by amounts that cancel in ``m``.
    The clean and the biased campaign then return the **same scores**: the
    largest disagreement over 640 epochs is ``4.9e-05``, and it shrinks as the
    prior's share of the posterior does -- ``1.9e-04`` at N = 160 against
    ``4.9e-05`` at N = 640 -- while the answer is wrong by ``52.6`` sigma. The
    spec promotes this diagnostic to primary; it is primary for a single rogue
    night and for a campaign whose nights genuinely differ, and it is blind to
    the fault section 12.11 is about. Read it beside :func:`coherent_mode`,
    never instead of it.

    Args:
        terms: the archive, in any order -- this statistic is exchangeable even
            when the campaign is not, because each epoch is scored against all
            the others.
        prior_fisher: ``F_prior`` over the same latents in the same column order.
            Required, not optional: ``Sigma_{-e}`` is singular at small N without
            it, and section 2.2 says a single epoch legitimately constrains only
            a subspace.
        prior_mean: the prior's mean, zero if absent.

    Returns:
        One :class:`HeldOut` per epoch, in the order the terms were given.

    Raises:
        ValueError: if the campaign is empty, if its epochs are over different
            latents, if ``prior_fisher`` or ``prior_mean`` is over a different
            number of columns, if the leave-one-out information is singular, or
            if any score comes out non-finite.
    """
    terms = tuple(terms)
    factors, targets, width = _campaign_arrays(terms)

    fisher = np.asarray(prior_fisher, dtype=float)
    if fisher.shape != (width, width):
        raise ValueError(
            f"prior_fisher has shape {fisher.shape}, but this campaign's epochs "
            f"are over {width} raveled values ({list(terms[0].info.names)}). The "
            "prior is added column for column, so a prior over a different "
            "parameter set is not a prior over these."
        )
    mean = np.zeros(width) if prior_mean is None else np.asarray(prior_mean, dtype=float)
    if mean.shape != (width,):
        raise ValueError(
            f"prior_mean has shape {mean.shape}, but this campaign's epochs are "
            f"over {width} raveled values ({list(terms[0].info.names)})."
        )

    total_fisher = fisher + sum(row.T @ row for row in factors)
    total_b = fisher @ mean + sum(
        row.T @ target for row, target in zip(factors, targets, strict=True)
    )

    rows: list[HeldOut] = []
    for term, row, target in zip(terms, factors, targets, strict=True):
        left_fisher = total_fisher - row.T @ row
        left_b = total_b - row.T @ target
        try:
            covariance = np.linalg.inv(np.linalg.cholesky(left_fisher)).T
        except np.linalg.LinAlgError as error:
            raise ValueError(
                f"The campaign with epoch {term.epoch_id!r} left out carries "
                "singular information, so there is no leave-one-out posterior to "
                f"score that epoch against. Remedy: {_PRIOR_REMEDY}"
            ) from error
        covariance = covariance @ covariance.T
        residual = row @ (covariance @ left_b) - target
        spread = np.eye(row.shape[0]) + row @ covariance @ row.T
        chi2 = float(residual @ np.linalg.solve(spread, residual))
        dof = int(row.shape[0])
        # `not (x >= 0.0)` rather than `x < 0.0`: NaN is False for both, and it
        # is NaN that has to be caught here. A NaN z sails through every
        # `z > threshold` an audit could write and reads as the quietest night.
        if not (chi2 >= 0.0) or not np.isfinite(chi2):
            raise ValueError(
                f"Epoch {term.epoch_id!r} scored {chi2}, which is not finite and "
                "non-negative. A stored factor or target carries NaN or inf, and "
                "NaN loses every comparison a campaign audit could make about it "
                "-- `z > threshold` is False for NaN, so the epoch would read as "
                "the quietest night of the run. Recompress that epoch."
            )
        rows.append(
            HeldOut(
                epoch_id=term.epoch_id,
                chi2=chi2,
                dof=dof,
                z=(chi2 - dof) / float(np.sqrt(2.0 * dof)),
            )
        )
    return tuple(rows)


def _shrinkage_table(sigmas: Mapping[int, Any]) -> tuple[np.ndarray, np.ndarray]:
    """``(log N, log sigma)`` as two flat arrays, with every trap refused first.

    Three of them, and each is here because the failure it prevents is a finite,
    plausible number rather than a crash:

    * fewer than two campaign sizes -- one point admits every slope, so a fitted
      power would be an invention;
    * a non-finite or non-positive sigma -- ``np.log`` maps those to ``nan`` and
      ``-inf``, and a ``nan`` power loses every comparison a caller could make
      about it, exactly as a ``nan`` z-score does;
    * a ragged table -- a pooled fit over two parameters at one size and three at
      another is not one fit, and its slope silently weights the sizes unequally.
    """
    sizes = sorted(sigmas)
    if len(sizes) < 2:
        raise ValueError(
            f"shrinkage_power needs at least two campaign sizes; got {sizes}. A "
            "single point admits every slope, so any number returned here would "
            "be the caller's assumption rather than a measurement."
        )
    bad_sizes = [n for n in sizes if not (int(n) > 0)]
    if bad_sizes:
        raise ValueError(
            f"These campaign sizes are not positive: {bad_sizes}. The fit is in "
            "log N, and log of a non-positive size is -inf or nan, which would "
            "make the returned power finite-looking or nan rather than refused."
        )
    columns = [np.atleast_1d(np.asarray(sigmas[n], dtype=float)) for n in sizes]
    widths = {column.shape for column in columns}
    if len(widths) > 1:
        raise ValueError(
            f"These campaign sizes report different numbers of widths: "
            f"{ {n: column.shape for n, column in zip(sizes, columns, strict=True)} }. "
            "A pooled power is one fit over the same latents at every size; a "
            "ragged table weights the sizes unequally without saying so. Fit the "
            "shared latents, or fit each size's set separately."
        )
    stacked = np.concatenate(columns)
    # `not (x > 0)` rather than `x <= 0`: NaN is False for both comparisons, and
    # NaN is the case that has to be caught. `np.isfinite` alone would let a
    # negative sigma through to `np.log`, which returns nan with a warning.
    if not np.all(np.isfinite(stacked)) or not np.all(stacked > 0.0):
        offenders = [
            (n, column.tolist())
            for n, column in zip(sizes, columns, strict=True)
            if not (np.all(np.isfinite(column)) and np.all(column > 0.0))
        ]
        raise ValueError(
            f"Every sigma must be finite and strictly positive; these are not: "
            f"{offenders}. A posterior width of zero, inf or nan is a broken "
            "covariance, not a very tight measurement -- check the campaign that "
            "produced it rather than fitting its logarithm."
        )
    log_size = np.repeat(np.log(np.asarray(sizes, dtype=float)), stacked.size // len(sizes))
    return log_size, np.log(stacked)


def shrinkage_power(sigmas: Mapping[int, Any]) -> float:
    """The fitted exponent of ``sigma_N ~ N^p``. **A sanity check, not a test.**

    Kept because section 9 says to keep it, and returned as a bare float only
    from here; :func:`shrinkage_report` is what a caller should print, because it
    carries the caveat in the same object as the number.

    For a Gaussian model ``sigma_N = (sum_e F_e + F_prior)^-1/2`` does not read
    the data. So this quantity is **data-independent**, ``p = -0.5`` holds by
    construction, and a uniform rescaling ``F_e -> (1+c) F_e`` moves it by
    exactly nothing -- verified in the spec for ``c`` in {0, +0.5, -0.3} and
    again in ``tests/evidence/test_coherent_bias.py``. Measured on the
    repeated-design fixture, the clean and the deliberately-biased campaign
    return ``-0.49991034`` and ``-0.49991034``, from per-N sigma arrays that are
    equal element for element.

    (The plan quoted ``-0.49989592``. That is the *first coordinate's own* slope;
    the second is ``-0.49992476``. A single float over a two-parameter campaign
    is the pooled fit, which is what this returns, and for the balanced grid a
    campaign audit actually has it equals the mean of the per-coordinate slopes.)

    v1's plan to prove a diagnostic works by injecting a shared systematic and
    watching this number was self-refuting, and that is the whole reason section
    9 exists in its present form.

    Args:
        sigmas: ``{campaign size: posterior widths}``. A scalar width is accepted
            and treated as a one-element array. Every size must report the same
            number of widths, in the same order.

    Returns:
        The ordinary-least-squares slope of ``log sigma`` on ``log N``, pooled
        over the widths. The intercept is free, so a uniform rescaling of every
        sigma cannot move the result.

    Raises:
        ValueError: for fewer than two campaign sizes, a non-positive size, a
            ragged table, or a sigma that is not finite and strictly positive.

    **Delegates to :func:`bayesmith.marginal.diagnostics.shrinkage_power`** as of
    the Wave D step-one batch (D61's ordering). Measured bitwise across the
    seam BEFORE the near-side body was removed -- ``|delta| = 0.0`` on an
    exact ``n**-0.5`` bank, an exact ``n**-1.0`` bank and a noisy one -- and
    that comparison cannot be re-taken now that this calls that.

    All four refusals live in the far side's ``_shrinkage_table`` with the
    same sentences, and its ``StructureError`` **is a** ``ValueError``
    (``ValueError`` is in its MRO), so this module's
    ``pytest.raises(ValueError, ...)`` pins hold through the seam unchanged.
    """
    return _far_shrinkage_power(sigmas)


def shrinkage_report(sigmas: Mapping[int, Any]) -> dict[str, Any]:
    """:func:`shrinkage_power` with its limits attached to it.

    ``detects_coherent_bias`` is ``False``, always, and it is a field rather than
    a sentence in a docstring because a number and its caveat travel together or
    they do not travel. A deterministic common-mode error -- one calibration
    solution, one beam model, one flag table applied to every night -- contributes
    **no variance** to the campaign's information: the posterior width is the
    same array element for element, split-half agrees to roundoff, leave-one-out
    returns the same scores, and the answer is wrong by 52.6 sigma at N = 640.
    Read :func:`coherent_mode` for a diagnostic that can fire.

    One measured refinement on section 9's own wording: the *per-epoch
    chi-square* scatter is **not** among the things a common mode leaves alone.
    It is inflated by noncentrality -- 5.5467 against ``sqrt(2 * 6) = 3.4641`` on
    the fixture -- without any new randomness being injected. What keeps its
    scatter exactly is the named template projection, 1.00200 clean and 1.00200
    biased, because there the fault is a pure additive shift.

    **Delegates to :func:`bayesmith.marginal.diagnostics.shrinkage_report`** as of
    the Wave D step-one batch (D61's ordering). Measured bitwise across the
    seam BEFORE the near-side body was removed -- ``|delta| = 0.0`` on an
    exact ``n**-0.5`` bank, an exact ``n**-1.0`` bank and a noisy one -- and
    that comparison cannot be re-taken now that this calls that.

    All four refusals live in the far side's ``_shrinkage_table`` with the
    same sentences, and its ``StructureError`` **is a** ``ValueError``
    (``ValueError`` is in its MRO), so this module's
    ``pytest.raises(ValueError, ...)`` pins hold through the seam unchanged.

    **The caveat is this package's own and is NOT taken from over there.**
    The far side's ends *"Use template_modes(), coherent_mode() and the
    systematic floor"*, and ``template_modes`` does not exist here -- advice
    naming a function the reader's package lacks is worse than none. This
    one also carries rheplicant's own measurement, the twelve digits on a
    campaign biased by 52.6 sigma, which is evidence rather than phrasing.
    """
    far = _far_shrinkage_report(sigmas)
    return {
        "power": far["power"],
        "n_values": far["n_values"],
        "detects_coherent_bias": far["detects_coherent_bias"],
        "caveat": (
            "sigma_N is data-independent for a Gaussian model, so this power is "
            "-0.5 by construction. A deterministic error shared across epochs "
            "contributes no variance, so it cannot move this number at all -- "
            "measured, the same value to twelve digits on a clean campaign and "
            "on one biased by 52.6 sigma. Use coherent_mode() and the systematic "
            "floor."
        ),
    }


def _prior_curvature(
    factorization: Any,
    at: Mapping[str, Any] | None,
    names: tuple[str, ...],
    spans: tuple[tuple[int, int], ...],
    width: int,
) -> np.ndarray:
    """``-d^2/dtheta^2 log pi(theta)``, block by block, in flatten order.

    Differentiated rather than read off a ``.scale`` attribute. A prior in this
    package is a duck type whose only guaranteed member is ``log_prob`` --
    :meth:`~rheplicant.inference.memory.BayesMemory.log_posterior` reads nothing
    else -- so a helper that reached for ``.scale`` would return **zero**
    information for anything that does not carry one. That failure is silent and
    it points the wrong way: less prior information means a wider sigma, and a
    wider sigma means the floor refusal fires later than it should.

    Evaluated at the latent's declared ``init`` unless ``at`` names a point,
    because ``init`` is the one value a
    :class:`~rheplicant.inference.parameters.ParameterSpace` guarantees is a
    legal value of the latent -- zeros would be outside the support of a
    log-normal or a gamma, and the resulting ``nan`` would read as a poisoned
    campaign. For the Gaussian priors this layer is exact for, the curvature
    does not depend on the point at all.

    Block diagonal by latent, and that is a property of the declaration rather
    than an approximation: ``global_priors`` is one prior per latent, so there
    is no cross term to lose. Within a latent the block is whatever the Hessian
    says, dense or not.
    """
    inits = {
        latent.name: latent.init
        for latent in factorization.space.latents
        if latent.scope == "global"
    }
    priors = factorization.global_priors
    curvature = np.zeros((width, width))
    for name, (start, stop) in zip(names, spans, strict=True):
        point = jnp.asarray(
            inits[name] if at is None or name not in at else at[name], dtype=float
        )
        shape = point.shape

        def density(flat: jax.Array, prior: Any = priors[name], shape: Any = shape):
            return jnp.sum(prior.log_prob(jnp.reshape(flat, shape)))

        block = -np.asarray(jax.hessian(density)(jnp.reshape(point, (-1,))))
        if not np.all(np.isfinite(block)):
            raise StateValidationError(
                f"The prior on {name!r} has non-finite curvature at the point it "
                "was differentiated at, so there is no posterior width to compare "
                "against a floor. That point is the latent's declared init unless "
                "at= names another; pass at= at a value inside the prior's "
                "support."
            )
        curvature[start:stop, start:stop] = block
    return curvature


def _refuse_bad_floors(floors: Mapping[str, Any], names: tuple[str, ...]) -> None:
    """A declared floor must name a latent this memory has, and be a real width.

    Both halves fail quietly otherwise, and in the same direction. A floor of
    ``nan`` makes ``sigma > floor`` False for every latent, so every campaign
    reads as breached -- the comparison guard below is written ``not (sigma >
    floor)`` precisely because NaN loses every comparison, and a NaN on the
    *floor* side turns that protection into a false alarm. A floor of ``inf``
    does the same thing without any NaN in sight. And a floor naming a latent
    the memory never accumulated would simply be dropped, which reads as "the
    campaign is above that floor".
    """
    unknown = sorted(name for name in floors if name not in names)
    if unknown:
        raise StateValidationError(
            f"The declared systematic floor names {unknown}, which this memory "
            f"does not accumulate; it accumulates {list(names)}. A floor is a "
            "prior width projected into the units of a latent the campaign "
            "actually carries, so a name it does not carry is a different model "
            "or a typo -- and dropped silently it would read as a campaign safely "
            "above that floor."
        )
    # `not (0 < value < inf)` rather than `value <= 0 or isinf(value)`: NaN is
    # False for every comparison in the chain, so this catches it with the same
    # expression that catches zero, negative and infinite.
    bad = sorted(name for name, value in floors.items() if not (0.0 < float(value) < math.inf))
    if bad:
        listed = ", ".join(f"{name}={float(floors[name])!r}" for name in bad)
        raise StateValidationError(
            f"These declared floor widths are not finite and strictly positive: "
            f"{listed}. A floor is a standard deviation in theta units. Zero or "
            "negative is not a width; inf puts every campaign below the floor, "
            "which reads as a detection; and nan loses every comparison, so it "
            "would breach every latent at once while the report showed nan."
        )


def _posterior_covariance(memory: Any, at: Mapping[str, Any] | None) -> tuple[
    np.ndarray, tuple[str, ...], tuple[tuple[int, int], ...]
]:
    """``(covariance, names, spans)`` -- ``(F_like + F_prior)^-1``, whole.

    The prior is added back deliberately.
    :meth:`~rheplicant.inference.memory.BayesMemory.fisher` excludes it, and
    says so, because it reports the *likelihood's* information; a floor is
    compared against the width a result is **quoted** with, which is the
    posterior's. Leaving the prior out makes every sigma larger than the quoted
    one, so the refusal would fire later than it should -- the silent direction.

    The **matrix** is returned rather than its diagonal, and that is the whole
    of this function's part in what
    :func:`_tightest_direction` fixes: a diagonal knows the width of each
    coordinate and nothing about the width of a combination, and the smallest
    width a correlated posterior has is never a coordinate's.
    """
    fisher = memory.fisher(at)
    names, spans = tuple(fisher.names), tuple(fisher.spans)
    total = np.asarray(fisher.matrix, dtype=float)
    width = total.shape[0]
    total = total + _prior_curvature(memory.factorization, at, names, spans, width)
    # Checked before the factorisation, not after. `np.linalg.cholesky` on a
    # matrix carrying NaN raises `LinAlgError`, which this function would then
    # report as "not positive definite" -- a confident, wrong diagnosis of a
    # poisoned stored factor. Propagating NaN instead is what the floor
    # comparison below is written to catch.
    if not np.all(np.isfinite(total)):
        return np.full((width, width), np.nan), names, spans
    try:
        lower = np.linalg.cholesky(total)
    except np.linalg.LinAlgError as error:
        raise StateValidationError(
            "The accumulated information plus the prior is not positive definite, "
            "so this campaign has no posterior width to compare against a floor. "
            "Every global latent needs a proper prior for that sum to be "
            "invertible at any N; check the declared priors before the epochs."
        ) from error
    inverse = np.linalg.inv(lower)
    return inverse.T @ inverse, names, spans


def _tightest_direction(block: np.ndarray) -> tuple[float, np.ndarray | None]:
    """``(width, unit direction)`` of the narrowest direction of one latent's block.

    **This is the correction to section 9.4's arithmetic, not to its argument.**
    The rationale was already right -- "the tightest is the first to go under,
    so it is the one a refusal must watch" -- and the code took ``np.min`` of
    the covariance's *diagonal*, which is the tightest **coordinate**. For any
    correlated posterior the smallest eigen-direction is below every diagonal
    entry, so the sentence was true of a quantity the function was not
    computing, and the gap is a basis rotation wide.

    Measured. On ``tests/evidence/campaign_bank.py``, whose posterior
    correlation is 0.5131, the tightest direction is 0.10419 against a tightest
    coordinate of 0.13425 at N = 1, and at a floor of 0.05 it crosses at N = 5
    where the tightest coordinate crosses at N = 8 and the loosest at N = 13:
    three nights of silence, and eight for a report keyed on the loosest. On a
    deliberately near-collinear design over 200 epochs the gap is the whole
    refusal: coordinate widths ``(1.402505, 1.403178)`` against a floor of 0.05,
    and a tightest direction of ``0.006335`` -- 7.9 times **below** it -- along
    ``(0.7073, 0.7069)``. That campaign quoted an error bar under its own
    declared systematic while ``below_floor`` said ``False``.

    The direction is returned because a bare number is half the value: "your
    error bar is too tight" is not actionable, "your error bar on
    ``0.707 x[0] + 0.707 x[1]`` is too tight" is. Its sign is fixed by making
    the largest-magnitude component positive, so the same posterior reports the
    same vector rather than one that flips with LAPACK's mood.

    ``None`` and a ``nan`` width for a block that is not finite: NaN cannot be
    eigendecomposed -- ``eigh`` raises ``LinAlgError`` -- and a poisoned
    campaign must report ``nan`` and be refused by the NaN-safe comparison in
    :func:`systematic_floor`, not raise a linear-algebra error here.
    """
    if not np.all(np.isfinite(block)):
        return float("nan"), None
    values, vectors = np.linalg.eigh(block)
    smallest = float(values[0])
    direction = np.asarray(vectors[:, 0], dtype=float)
    direction = direction * np.sign(direction[int(np.argmax(np.abs(direction)))])
    # A covariance is positive definite by construction here -- it is the
    # inverse of a Cholesky factor times its transpose -- so a non-positive
    # eigenvalue is roundoff on a block that constrains that direction not at
    # all. Zero is the honest width for it and it is below every legal floor,
    # which is the safe direction. `nan` is handled above, never here, because
    # `smallest > 0.0` is False for NaN and would silently become zero.
    return (math.sqrt(smallest) if smallest > 0.0 else 0.0), direction


def systematic_floor(
    memory: Any, floors: Mapping[str, Any], at: Mapping[str, Any] | None = None
) -> dict[str, dict[str, Any]]:
    """Section 9.4: has this campaign out-run its own calibration?

    The floor is the declared prior width of a shared calibration product --
    one solution, one beam model, one flag table serving every night --
    **projected into theta units by the analyst**. It is a declaration and not a
    measurement, and that is forced rather than lazy: the in-span half of a
    coherent error biases theta identically in every epoch and leaves no
    residual anywhere, so it passes per-epoch chi-square, split-half and
    leave-one-out, and no statistic computed from the stored terms can recover
    it. Measured on ``tests/evidence/campaign_bank.py``, the answer is wrong by
    52.6 sigma at N = 640 with every data-driven diagnostic clean.

    What the campaign *does* know is its own width, and that width falls as
    ``N^-1/2`` while the shared product's does not fall at all. So the whole
    content of this function is: when does one pass under the other.

    ``sigma`` is the width of the **tightest direction** of a latent's marginal
    posterior -- the square root of the smallest eigenvalue of its covariance
    block -- not the tightest coordinate, not the loosest and not the mean. A
    vector latent has a width in every direction and a floor is one number; the
    tightest is the first to go under, so it is the one a refusal must watch,
    and for a correlated posterior it is never a coordinate. Measured on the
    campaign fixture with a floor of 0.05 and a posterior correlation of 0.5131,
    the tightest direction crosses at N = 5, the tightest coordinate at N = 8
    and the loosest at N = 13. See :func:`_tightest_direction`, which also
    records the near-collinear campaign whose error bar sits 7.9 times under the
    floor with every coordinate width an order of magnitude above it.

    ``crossing_epoch`` is **computed from the observed width**, not quoted: the
    campaign's own ``sigma_N`` is extrapolated as ``sigma_N sqrt(N / N')`` and
    solved for ``sigma_{N'} = floor``, giving ``N' = ceil(N (sigma_N /
    floor)^2)``. That extrapolation ignores the prior's share, which shrinks as
    N grows, so it predicts the crossing marginally early; measured on the
    campaign fixture it is exact -- 8 predicted from N = 4 and from N = 16, 8
    observed -- because the prior is 0.25 against a per-epoch 42.5.

    Args:
        memory: a :class:`~rheplicant.inference.memory.BayesMemory`. Read for
            its Fisher, its factorization's priors and the campaign's length.
        floors: ``{global latent: declared width in that latent's units}``.
            Every entry must name a latent the memory accumulates and be finite
            and strictly positive.
        at: where to pull a reduced-basis term's coefficient-space information
            back into theta, and where to differentiate the priors. Defaults to
            each latent's declared ``init``.

    Returns:
        One entry per latent name, each a dict of ``"sigma"``, ``"floor"``,
        ``"below_floor"``, ``"crossing_epoch"`` and ``"direction"``. Spelled
        out rather than written as ``{name: {...}}``, because napoleon splits a
        Google-style ``Returns:`` block at its first colon to find a return
        TYPE -- and it does not exempt colons inside an inline literal, so that
        form had its opening backticks eaten into the rtype and left the
        closing pair orphaned. One Sphinx warning, and the paragraph rendered
        wrong. Keep this description colon-free.

        ``below_floor`` is the refusal's own comparison, computed here and
        nowhere else so that the NaN-safe form exists in one place;
        ``crossing_epoch`` is ``None`` when the width is not a finite positive
        number, because there is then no crossing to extrapolate;
        ``direction`` is the unit combination of that latent's raveled
        components whose width ``sigma`` is, and ``None`` for a poisoned block.

    Raises:
        StateValidationError: for an empty campaign, a floor naming a latent
            the memory does not accumulate, a floor that is not a finite
            positive width, a prior with non-finite curvature, or an
            information matrix that is not positive definite.
    """
    n_epochs = len(memory.archive)
    if n_epochs == 0:
        raise StateValidationError(
            "systematic_floor needs at least one epoch. The crossing epoch is "
            "N (sigma_N / floor)^2, which is zero for N = 0 -- an empty campaign "
            "would report that it passed under its systematic floor before it "
            "started."
        )
    covariance, names, spans = _posterior_covariance(memory, at)
    _refuse_bad_floors(floors, names)
    report: dict[str, dict[str, Any]] = {}
    for name, declared in floors.items():
        start, stop = spans[names.index(name)]
        sigma, direction = _tightest_direction(covariance[start:stop, start:stop])
        floor = float(declared)
        # `not (sigma > floor)`, never `sigma <= floor`: NaN is False for both
        # comparisons, so the second form waves a poisoned campaign through
        # while the same dict reports the nan. Third time this exact shape has
        # been needed in this subsystem. `inf` is handled by the same
        # expression and correctly: a campaign that constrains nothing is not
        # tighter than anything.
        report[name] = {
            "sigma": sigma,
            "floor": floor,
            "below_floor": not (sigma > floor),
            "crossing_epoch": (
                math.ceil(n_epochs * (sigma / floor) ** 2)
                if math.isfinite(sigma) and sigma > 0.0
                else None
            ),
            "direction": direction,
        }
    return report
