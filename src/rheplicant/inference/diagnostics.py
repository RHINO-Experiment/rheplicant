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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


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
    beam model, one flag table -- contributes **no variance**. Per-epoch
    chi-square is right, split-half agrees, leave-one-out agrees, and the answer
    is wrong. What it does move is a **mean**, and a mean over N epochs is
    resolved at sqrt(N), which is why this reports z-scores rather than
    magnitudes.

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
