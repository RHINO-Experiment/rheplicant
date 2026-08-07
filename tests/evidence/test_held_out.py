"""Section 9.1, and an honest statement of its reach.

The first tests are the diagnostic working. Two of the last are the diagnostic
NOT working, on the case section 12.11 is about, and they are the reason this
file exists rather than a paragraph in a docstring: a diagnostic whose blind spot
is only written down gets trusted anyway.

Every tolerance here is set from a measurement, printed once and then pinned, so
that a test cannot pass because its bound was generous.
"""

import numpy as np
import pytest

from rheplicant.inference.diagnostics import held_out_z
from tests.evidence import campaign_bank as camp


def _varying(n_epochs, biased, seed=17):
    """Nights whose design differs -- the case the held-out z can see."""
    import jax.numpy as jnp

    from rheplicant.inference.compress import compress_linear

    rng = np.random.default_rng(seed)
    out = []
    for e in range(n_epochs):
        design = rng.normal(size=(camp.N_SAMPLES, camp.N_THETA))
        data = (
            design @ camp.TRUTH
            + camp.SIGMA * rng.normal(size=camp.N_SAMPLES)
            + (camp.COMMON if biased else 0.0)
        )
        out.append(
            compress_linear(
                design={"x": jnp.asarray(design)},
                observed=jnp.asarray(data),
                noise_std=camp.SIGMA,
                shapes={"x": (camp.N_THETA,)},
                epoch_id=f"v{e}",
            )
        )
    return tuple(out)


def test_a_clean_campaign_gives_standard_normal_held_out_scores():
    """Measured: mean +0.050250 against a 4-sigma bound of 0.2309, scatter 1.0934."""
    scores = held_out_z(_varying(300, biased=False), camp.prior_fisher())
    values = np.array([row.z for row in scores])
    assert abs(values.mean()) < 4.0 / np.sqrt(300)
    assert values.std() == pytest.approx(1.0, rel=0.2)


def test_one_bad_night_shows_up_as_one_bad_score():
    """The diagnostic's day job: a single epoch that does not belong.

    Measured: the rogue scores +72.96 while the largest of the other 59 is
    4.2415 -- a separation of seventeen, not of a factor.
    """
    import jax.numpy as jnp

    from rheplicant.inference.compress import compress_linear

    collected = list(_varying(60, biased=False))
    collected[17] = compress_linear(
        design={"x": jnp.asarray(camp.DESIGN)},
        observed=jnp.asarray(camp.DESIGN @ camp.TRUTH + 6.0),
        noise_std=camp.SIGMA,
        shapes={"x": (camp.N_THETA,)},
        epoch_id="rogue",
    )
    scores = {
        row.epoch_id: row.z for row in held_out_z(tuple(collected), camp.prior_fisher())
    }
    assert scores["rogue"] > 10.0
    others = np.array([z for name, z in scores.items() if name != "rogue"])
    assert np.abs(others).max() < 6.0


def test_a_varying_design_campaign_does_show_a_coherent_mode():
    """Where the spec's promotion is right, and by how much.

    Measured at N = 300: the clean campaign's mean is +0.87 sigma, the biased
    one's is +22.05 sigma. This is the case the diagnostic ships for.
    """
    clean = np.array([r.z for r in held_out_z(_varying(300, False), camp.prior_fisher())])
    biased = np.array([r.z for r in held_out_z(_varying(300, True), camp.prior_fisher())])
    assert abs(clean.mean()) * np.sqrt(300) < 4.0
    assert biased.mean() * np.sqrt(300) > 8.0


def test_the_score_is_the_chi_square_of_the_held_out_residual():
    """Pinned against the closed form, so the standardisation cannot drift."""
    collected = _varying(20, biased=False)
    rows = held_out_z(collected, camp.prior_fisher())
    for row in rows[:3]:
        assert row.z == pytest.approx(
            (row.chi2 - row.dof) / np.sqrt(2 * row.dof), rel=1e-12
        )
        assert row.dof == int(np.asarray(collected[0].info.factor).shape[0])


def test_a_repeated_design_makes_the_held_out_score_blind():
    """Deviation 9, in executable form. This is the section 12.11 fixture.

    Every night has the same design, so the coherent in-span error shifts
    ``z_e`` and ``mu_{-e}`` by amounts that cancel in ``R mu_{-e} - z_e``. The
    two campaigns give the same held-out scores to within the prior's pull --
    measured, the largest disagreement over all 640 epochs is **4.93e-05**, and
    it falls as 1/N: 1.94e-04 at N = 160 against 4.93e-05 at N = 640, a ratio of
    3.95 for a factor-of-four change in N. The tolerance below is 5e-4, ten
    times the measured value and ten times tighter than the plan's 5e-3, so the
    assertion is about the measurement rather than about a generous bound.
    """
    clean = np.array([r.z for r in held_out_z(camp.terms(640, False), camp.prior_fisher())])
    biased = np.array([r.z for r in held_out_z(camp.terms(640, True), camp.prior_fisher())])
    np.testing.assert_allclose(biased, clean, atol=5e-4)


def test_the_blindness_tightens_as_the_priors_share_falls():
    """The mechanism, not just the fact: the residue is the prior's, so it is O(1/N).

    If the two campaigns agreed for some *other* reason -- a bug that returned a
    constant, say -- the disagreement would not scale. Measured: 1.94e-04 at
    N = 160 and 4.93e-05 at N = 640.
    """
    gaps = {}
    for n in (160, 640):
        clean = np.array([r.z for r in held_out_z(camp.terms(n, False), camp.prior_fisher())])
        biased = np.array([r.z for r in held_out_z(camp.terms(n, True), camp.prior_fisher())])
        gaps[n] = float(np.abs(biased - clean).max())
    assert gaps[160] / gaps[640] == pytest.approx(4.0, rel=0.25)


def test_and_the_answer_is_wrong_while_it_is_blind():
    """The other half of the same claim: blind AND wrong, not blind because right."""
    mean, covariance = camp.posterior(camp.terms(640, biased=True))
    width = np.sqrt(np.diag(covariance))
    ratio = np.abs(mean - camp.TRUTH) / width
    assert ratio.max() > 20.0  # measured 52.56838
    clean_mean, clean_cov = camp.posterior(camp.terms(640, biased=False))
    assert (
        np.abs(clean_mean - camp.TRUTH) / np.sqrt(np.diag(clean_cov))
    ).max() < 5.0  # measured 0.71746


def test_an_empty_campaign_is_refused_rather_than_scored_as_clean():
    """Returning ``()`` would read as 'no epoch was surprising'."""
    with pytest.raises(ValueError, match="at least one epoch"):
        held_out_z((), camp.prior_fisher())


def test_a_campaign_over_different_latents_is_refused():
    """The columns are positional, so two parameter sets are not one campaign."""
    import jax.numpy as jnp

    from rheplicant.inference.compress import compress_linear

    other = compress_linear(
        design={"y": jnp.asarray(camp.DESIGN)},
        observed=jnp.asarray(camp.DESIGN @ camp.TRUTH),
        noise_std=camp.SIGMA,
        shapes={"y": (camp.N_THETA,)},
        epoch_id="elsewhere",
    )
    with pytest.raises(ValueError, match="different latents"):
        held_out_z(camp.terms(2, biased=False) + (other,), camp.prior_fisher())
    # A guard that over-refuses is its own bug: the nearest legitimate case is
    # two batches of the same latents built from different seeds.
    same = camp.terms(2, biased=False) + camp.terms(2, biased=False, seed=99)
    assert len(held_out_z(same, camp.prior_fisher())) == 4


def test_a_prior_over_the_wrong_number_of_latents_is_refused():
    """``F_prior`` is added column-for-column; a 3x3 prior on 2 latents is not a prior."""
    with pytest.raises(ValueError, match="prior_fisher"):
        held_out_z(camp.terms(3, biased=False), np.eye(3))
    with pytest.raises(ValueError, match="prior_mean"):
        held_out_z(camp.terms(3, biased=False), camp.prior_fisher(), np.zeros(3))


def test_a_campaign_with_no_prior_is_refused_by_name_rather_than_by_linalg():
    """Why ``prior_fisher`` is required, made executable.

    Leave one epoch out of a one-epoch campaign and the remaining information is
    the prior alone. With no prior that is singular, and ``numpy`` says
    "Singular matrix", which names neither the epoch nor the remedy.
    """
    with pytest.raises(ValueError, match="prior_fisher"):
        held_out_z(camp.terms(1, biased=False), np.zeros((camp.N_THETA, camp.N_THETA)))
    # The nearest legitimate case: a prior weak enough to be almost nothing.
    rows = held_out_z(camp.terms(1, biased=False), np.eye(camp.N_THETA) * 1e-6)
    assert np.isfinite(rows[0].z)


def test_a_nan_score_is_refused_rather_than_returned():
    """NaN loses every comparison a caller could make about it.

    ``row.z > threshold`` is False for NaN, so one poisoned epoch walks through
    a campaign audit looking like the quietest night of the run.
    """
    import equinox as eqx
    import jax.numpy as jnp

    collected = list(camp.terms(3, biased=False))
    collected[1] = eqx.tree_at(
        lambda t: t.info.target,
        collected[1],
        jnp.asarray(collected[1].info.target).at[0].set(jnp.nan),
    )
    with pytest.raises(ValueError, match="not finite"):
        held_out_z(tuple(collected), camp.prior_fisher())


def test_an_epoch_with_no_rows_is_refused_by_name_rather_than_by_zero_division():
    """`dof` is the epoch's row count, and `z` divides by `sqrt(2 * dof)`.

    A fully flagged night compresses to a factor with zero rows -- so does
    `SqrtInfo.null`, which the accumulator uses as padding -- and it reached
    `(chi2 - dof) / sqrt(2 * dof)` with `dof = 0` and died on a bare
    `ZeroDivisionError: float division by zero`, one line below the NaN guard
    that was written for exactly this shape of problem and cannot help because
    the division happens first. Measured: it raised for a lone empty epoch and
    for an empty epoch sitting between two real ones, so a long campaign with
    one dead night lost every score it had already computed.
    """
    import jax.numpy as jnp

    from rheplicant.inference.sqrtinfo import SqrtInfo

    class _Empty:
        epoch_id = "all-flagged"
        info = SqrtInfo(
            factor=jnp.zeros((0, camp.N_THETA)),
            target=jnp.zeros((0,)),
            offset=jnp.zeros(()),
            names=("x",),
            shapes=((camp.N_THETA,),),
        )

    for collected in ((_Empty(),), (*camp.terms(2, biased=False)[:1], _Empty())):
        with pytest.raises(ValueError, match="no rows"):
            held_out_z(collected, camp.prior_fisher())


def test_a_campaign_of_ordinary_nights_is_still_scored():
    """Guard the guard: the row-count refusal must not catch a real campaign."""
    assert len(held_out_z(camp.terms(3, biased=False), camp.prior_fisher())) == 3
