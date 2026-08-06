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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

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
    """
    log_size, log_sigma = _shrinkage_table(sigmas)
    centred = log_size - log_size.mean()
    return float(centred @ (log_sigma - log_sigma.mean()) / (centred @ centred))


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
    """
    power = shrinkage_power(sigmas)
    return {
        "power": power,
        "n_values": tuple(sorted(sigmas)),
        "detects_coherent_bias": False,
        "caveat": (
            "sigma_N is data-independent for a Gaussian model, so this power is "
            "-0.5 by construction. A deterministic error shared across epochs "
            "contributes no variance, so it cannot move this number at all -- "
            "measured, the same value to twelve digits on a clean campaign and "
            "on one biased by 52.6 sigma. Use coherent_mode() and the systematic "
            "floor."
        ),
    }
