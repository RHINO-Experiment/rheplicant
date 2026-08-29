"""Four behaviours of the diagnostics layer that nothing was watching.

Each was found the same way as the ``compress_linear`` set: mutate the source,
run the whole ``tests/evidence`` directory, and see whether anything goes red.
All four came back green, so all four were free to change.

They matter beyond tidiness. Three of them are defended **in prose** -- by a
docstring sentence, a nine-line comment, or a number quoted in ``docs/``, all
written by someone who understood the choice and none of them checkable. The
fourth is an argument no caller ever passes.

============================================================  ================
mutation                                                      red before this
============================================================  ================
give every epoch the FIRST epoch's dof                        none
``scatter`` from the sample spread instead of the population  none
``reduced_chi2 = 0.0`` instead of ``nan`` when ``dof == 0``   none
``systematic_floor`` ignores its ``at=`` point entirely       none
============================================================  ================

The first is the sharpest. ``bayesmith``'s ``template_modes`` docstring names
"reading a row count off the first term" as the specific thing ``coherent_mode``
must not do, and near's ``coherent_mode`` carries a comment defending the
per-epoch sum -- while every campaign fixture in the suite gave every epoch the
same dof, so the two spellings could not be told apart.
"""

import numpy as np
import pytest

import jax.numpy as jnp

from rheplicant.inference import Bind, Latent, ParameterSpace
from rheplicant.inference.diagnostics import (
    EpochResidual,
    coherent_mode,
    epoch_residuals,
    systematic_floor,
)
from rheplicant.inference.factorize import Factorization
from rheplicant.inference.memory import BayesMemory
from tests.evidence import campaign_bank as camp


def _ragged(n_epochs=4, seed=11):
    """A campaign whose nights do NOT share a dof.

    Every fixture in this directory builds epochs of one length, which is what
    let the raggedness defect hide: with equal dof, summing per-epoch and
    reading the first epoch's row count give the same number.
    """
    collected = list(camp.terms(n_epochs, biased=False, seed=seed))
    import dataclasses  # noqa: PLC0415 -- local, this helper is the only user

    varied = []
    for index, term in enumerate(collected):
        # eqx.Module is frozen; rebuild with a different stored dof.
        varied.append(
            type(term)(
                **{
                    **{
                        f.name: getattr(term, f.name)
                        for f in dataclasses.fields(term)
                    },
                    "residual_dof": int(term.residual_dof) - index,
                }
            )
        )
    return tuple(varied)


def test_the_dof_is_summed_per_epoch_not_read_off_the_first():
    """A ragged campaign, which no other fixture here builds.

    ``coherent_mode``'s ``chi2_dof`` must be the sum of the epochs' own dof. If
    it reads the first epoch's row count and multiplies, a campaign whose nights
    saw different amounts of data reports a dof that belongs to no campaign.
    """
    terms = _ragged(4)
    stored = [int(t.residual_dof) for t in terms]
    assert len(set(stored)) > 1, "the fixture is not ragged, so this proves nothing"

    report = coherent_mode(terms)
    # `chi2_dof` is deliberately None for a ragged campaign -- it reports the
    # shared per-epoch dof and there is none. The sum is what reaches `chi2_z`,
    # and that is the number the raggedness defect would get wrong.
    assert report["chi2_dof"] is None
    chi2_sum = sum(float(r.chi2) for r in epoch_residuals(terms))
    total = sum(stored)
    expected = (chi2_sum - total) / np.sqrt(2.0 * total)
    assert report["chi2_z"] == pytest.approx(expected, rel=1e-12), (
        f"chi2_z is {report['chi2_z']!r}; summing the epochs' own dof ({stored}) "
        f"gives {expected!r}."
    )
    # And it is not the first-epoch-times-N spelling, which is the specific
    # thing bayesmith's template_modes docstring names as the defect.
    wrong_total = stored[0] * len(stored)
    wrong = (chi2_sum - wrong_total) / np.sqrt(2.0 * wrong_total)
    assert report["chi2_z"] != pytest.approx(wrong, rel=1e-6), (
        "the ragged and uniform spellings coincide on this fixture, so it "
        "cannot tell them apart"
    )


def test_every_epoch_keeps_its_own_dof_in_the_residual_rows():
    """The same fact one level down, where `epoch_residuals` reports it."""
    terms = _ragged(4)
    rows = epoch_residuals(terms)
    assert [r.dof for r in rows] == [int(t.residual_dof) for t in terms]


def test_the_template_scatter_is_the_population_spread():
    """``ddof=0``, defended by nine lines of comment and quoted in ``docs/``.

    The projections are already standardised, so their spread is a population
    quantity with a known value of 1 rather than an estimate of an unknown
    variance. At N = 3 the two spellings differ by 22 %, which is why this is
    pinned on a small campaign.
    """
    terms = camp.terms(3, biased=False)
    report = coherent_mode(terms)
    scatter = report["templates"]["gain_ripple"]["scatter"]
    projections = np.array(
        [float(np.asarray(t.template_projections)[0]) for t in terms]
    )
    assert scatter == pytest.approx(float(projections.std(ddof=0)), rel=1e-12)
    assert scatter != pytest.approx(float(projections.std(ddof=1)), rel=1e-6), (
        "the population and sample spreads coincide on this fixture, so it "
        "cannot tell the two apart -- use fewer epochs"
    )


def test_a_zero_dof_epoch_reports_nan_rather_than_a_perfect_fit():
    """``EpochResidual``'s own docstring: "a zero here would read as a perfect fit".

    An epoch whose design saturates its data has no residual to speak of.
    ``0.0`` is the most misleading value available, because it is what a
    flawless fit looks like.
    """
    row = EpochResidual(
        epoch_id="e0", chi2=0.0, dof=0, reduced_chi2=float("nan"), templates={}
    )
    assert np.isnan(row.reduced_chi2)

    terms = list(camp.terms(1, biased=False))
    import dataclasses  # noqa: PLC0415

    term = terms[0]
    saturated = type(term)(
        **{
            **{f.name: getattr(term, f.name) for f in dataclasses.fields(term)},
            "residual_dof": 0,
        }
    )
    produced = epoch_residuals((saturated,))[0]
    assert np.isnan(produced.reduced_chi2), (
        f"a zero-dof epoch reported reduced_chi2 = {produced.reduced_chi2!r}. "
        f"Zero is what a perfect fit looks like."
    )


class _Quartic:
    """A prior whose curvature depends on where you evaluate it.

    ``log pi = -x^4 / 4``, so ``-d^2/dx^2 log pi = 3 x^2``. Every prior in this
    directory's fixtures is Gaussian, and a Gaussian's curvature is
    ``1 / scale^2`` **wherever you stand** -- which is exactly why nothing could
    test the ``at=`` argument: on those fixtures ignoring it is an equivalent
    mutation, not a defect.

    Only ``log_prob`` is read downstream, so three lines is the whole prior.
    """

    def log_prob(self, x):
        return -0.25 * jnp.sum(x**4)


def _quartic_memory(n_epochs=4, init=1.0):
    latent = Latent(
        "x", init=jnp.full((camp.N_THETA,), init), prior=_Quartic()
    )
    space = ParameterSpace(
        latents=(latent,), bindings=(Bind("x", into=lambda p: p.x),)
    )
    memory = BayesMemory(Factorization(space))
    for term in camp.terms(n_epochs, biased=False):
        memory = memory.remember(term)
    return memory


def test_the_evaluation_point_reaches_the_reported_width():
    """``systematic_floor(..., at=)`` -- no caller in the repository passes it.

    The curvature here is ``3 x^2``, so standing at ``x = 3`` gives nine times
    the prior information of standing at ``x = 1``, a tighter posterior and a
    smaller reported sigma. An implementation that ignored ``at=`` would report
    the ``init`` answer twice.
    """
    memory = _quartic_memory(init=1.0)
    floors = {"x": 0.05}

    at_init = systematic_floor(memory, floors)
    at_three = systematic_floor(memory, floors, at={"x": jnp.full((camp.N_THETA,), 3.0)})

    sigma_init = at_init["x"]["sigma"]
    sigma_three = at_three["x"]["sigma"]
    assert sigma_three != pytest.approx(sigma_init, rel=1e-9), (
        f"systematic_floor reported sigma {sigma_three!r} at x=3 and "
        f"{sigma_init!r} at the latent's init. The prior's curvature here is "
        f"3 x^2, so the two cannot be the same number unless `at=` was ignored."
    )
    assert sigma_three < sigma_init, (
        "more prior curvature must give a tighter posterior, not a looser one"
    )
