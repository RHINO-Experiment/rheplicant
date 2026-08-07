"""Section 9.3: what one night's residual says, from ~100 stored bytes.

Every assertion is on an absolute number with a measured value, because the
failure this catches is a summary that is *plausible* -- a chi-square near its
DOF, a projection near zero -- rather than one that is right.
"""

import numpy as np
import pytest

from rheplicant.inference.diagnostics import coherent_mode, epoch_residuals
from tests.evidence import campaign_bank as camp


def test_the_fixture_carries_both_halves_of_the_common_mode():
    """Guard the fixture: with either half zero, half this file is vacuous."""
    parallel, perpendicular = camp.whitened_split()
    assert parallel == pytest.approx(2.30704, rel=1e-5)
    assert perpendicular == pytest.approx(2.00937, rel=1e-5)


def test_a_clean_epochs_chi_square_is_its_dof_on_average():
    collected = camp.terms(400, biased=False)
    chi2 = np.array([float(term.residual_chi2) for term in collected])
    dof = collected[0].residual_dof
    assert dof == camp.N_SAMPLES - camp.N_THETA
    assert chi2.mean() == pytest.approx(dof, abs=4 * np.sqrt(2 * dof / 400))
    assert chi2.std() == pytest.approx(np.sqrt(2 * dof), rel=0.15)


def test_the_stored_chi_square_is_the_offsets_own_quadratic_part():
    """Stored rather than recomputed, and the two must be the same number."""
    term = camp.terms(1, biased=False)[0]
    normalisation = -0.5 * camp.N_SAMPLES * float(np.log(2 * np.pi * camp.SIGMA**2))
    assert float(term.residual_chi2) == pytest.approx(
        -2.0 * (float(term.info.offset) - normalisation), rel=1e-10
    )


def test_a_clean_template_projection_is_standard_normal():
    collected = camp.terms(400, biased=False)
    projections = np.array([float(term.template_projections[0]) for term in collected])
    assert collected[0].template_names == ("gain_ripple",)
    assert projections.std() == pytest.approx(1.0, rel=0.12)
    assert abs(projections.mean()) < 4.0 / np.sqrt(400)


def test_a_common_mode_shows_as_a_nonzero_mean_at_root_n():
    """The whole of section 9.3, in the numbers it was written for."""
    report = coherent_mode(camp.terms(640, biased=True))
    assert report["chi2_z"] > 10.0
    assert report["templates"]["gain_ripple"]["z"] > 20.0
    # ...with CORRECT scatter, which is what says it is a mean-level fault and
    # not extra noise.
    assert report["templates"]["gain_ripple"]["scatter"] == pytest.approx(1.0, rel=0.15)


def test_the_same_report_stays_quiet_on_a_clean_campaign():
    """A guard that over-refuses is its own bug."""
    report = coherent_mode(camp.terms(640, biased=False))
    assert abs(report["chi2_z"]) < 4.0
    assert abs(report["templates"]["gain_ripple"]["z"]) < 4.0


def test_the_significance_grows_as_root_n_and_the_mean_does_not():
    """Section 7's arithmetic, one level along: a coherent fault does not average out."""
    z = {}
    means = {}
    for n in (40, 160, 640):
        report = coherent_mode(camp.terms(n, biased=True))
        z[n] = report["templates"]["gain_ripple"]["z"]
        means[n] = report["templates"]["gain_ripple"]["mean"]
    assert z[640] / z[160] == pytest.approx(2.0, rel=0.35)
    assert means[640] == pytest.approx(means[40], rel=0.35)


def test_a_template_that_misses_the_mode_stays_quiet_and_that_is_honest():
    """The analyst guessed wrong. The summary must not invent a detection."""
    orthogonal = np.linalg.svd(np.column_stack([camp.TEMPLATE, camp.DESIGN]))[0][:, -1]
    collected = camp.terms(640, biased=True, templates={"wrong_guess": orthogonal})
    report = coherent_mode(collected)
    assert abs(report["templates"]["wrong_guess"]["z"]) < 5.0
    # ...but the chi-square, which needs no guess, still fires.
    assert report["chi2_z"] > 10.0


def test_a_flagged_epoch_summarises_what_it_saw_rather_than_a_nan():
    """`0.0 * nan` is `nan`, and a summary is a mean over epochs.

    Renamed from the plan's ``..._contributes_no_summary_rather_than_a_nan``,
    which says the opposite of what happens and of what should: a flagged night
    contributes a perfectly good summary of its unflagged samples. What must not
    happen is a NaN, because one NaN epoch poisons the campaign mean that every
    z-score in :mod:`rheplicant.inference.diagnostics` is built on, and NaN
    loses every comparison a guard could make about it.

    The flagged sample here holds a NaN, which is the usual reason a sample is
    flagged. It reaches the template column as well as the residual, so the
    ``jnp.where`` in ``_residual_summary`` has to select rather than weight.
    """
    import jax.numpy as jnp

    from rheplicant.inference.compress import compress_linear

    sigma = jnp.array([camp.SIGMA] * 6 + [jnp.inf, jnp.inf])
    data = jnp.asarray(camp.DESIGN @ camp.TRUTH).at[6].set(jnp.nan)
    # The template is NaN where the data is, which is the realistic case: a
    # systematic template built from the same product that caused the flag. The
    # weight there is an exact 0.0, so multiplying instead of selecting turns
    # the whole projection into a NaN.
    template = jnp.asarray(camp.TEMPLATE).at[6].set(jnp.nan)
    term = compress_linear(
        design={"x": jnp.asarray(camp.DESIGN)},
        observed=data,
        noise_std=sigma,
        shapes={"x": (camp.N_THETA,)},
        epoch_id="flagged",
        templates={"gain_ripple": template},
    )
    assert term.residual_dof == 6 - camp.N_THETA
    assert np.isfinite(float(term.residual_chi2))
    assert np.isfinite(float(term.template_projections[0]))


def _with_template(template, sigma=camp.SIGMA):
    """One clean epoch of the fixture, compressed against ``template``."""
    import jax.numpy as jnp

    from rheplicant.inference.compress import compress_linear

    rng = np.random.default_rng(17)
    data = camp.DESIGN @ camp.TRUTH + camp.SIGMA * rng.normal(size=camp.N_SAMPLES)
    return compress_linear(
        design={"x": jnp.asarray(camp.DESIGN)},
        observed=jnp.asarray(data),
        noise_std=sigma,
        shapes={"x": (camp.N_THETA,)},
        epoch_id="e",
        templates={"gain_ripple": template},
    )


class TestATemplateIsModelInputAndIsCheckedLikeOne:
    """A template the epoch cannot project onto is refused, by name, at entry.

    ``_residual_summary`` selected on ``norm > 0.0``, and that test is **False
    for NaN**. So a NaN anywhere in an unflagged sample of a template took the
    same branch as a template lying entirely inside the design's column space,
    and both were reported as a projection of exactly ``0.0`` -- which is the
    value this summary's own docstring reserves for the genuine null. Measured
    on the expression, whitened by the fixture's sigma: healthy ``norm=1.3136``,
    ``norm>0=True``, projection ``-0.4957``; all-zero ``norm=0.0000``,
    ``norm>0=False``, projection ``+0.0000``; one NaN ``norm=nan``,
    ``norm>0=False``, projection ``+0.0000``. End to end over 200 epochs each
    carrying the fixture's genuine common mode, a single NaN on an unflagged
    sample of the template took ``gain_ripple`` from
    ``{'mean': 2.156, 'scatter': 1.062, 'z': 30.49}`` to
    ``{'mean': 0.0, 'scatter': 0.0, 'z': 0.0}`` while ``chi2_z`` stayed at
    18.542 -- a 30-sigma detection reported as silence, with the chi-square
    half of the same summary still shouting.

    Nothing downstream could have caught it: ``QuadraticLikelihood`` checks
    ``info.factor``, ``info.target`` and ``info.offset`` for dtype and never
    looks at ``template_projections``, and ``0.0`` passes every finiteness guard
    there is.

    ``inf`` was measured too, and on this fixture it is **not** the loud case it
    looks like: ``projector @ column`` is infinite as well, ``inf - inf`` is
    NaN, so ``norm=nan`` and an infinite template reports ``0.0`` by the same
    route. Both are refused by the same check.
    """

    def test_a_nan_on_an_unflagged_sample_is_refused_by_name(self):
        import jax.numpy as jnp

        from rheplicant.core.errors import StateValidationError

        template = jnp.asarray(camp.TEMPLATE).at[3].set(jnp.nan)
        with pytest.raises(StateValidationError, match="usable systematic template"):
            _with_template(template)
        with pytest.raises(StateValidationError, match="gain_ripple"):
            _with_template(template)

    def test_an_infinity_is_refused_by_the_same_check(self):
        import jax.numpy as jnp

        from rheplicant.core.errors import StateValidationError

        template = jnp.asarray(camp.TEMPLATE).at[3].set(jnp.inf)
        with pytest.raises(StateValidationError, match="usable systematic template"):
            _with_template(template)

    def test_a_non_finite_sample_that_is_flagged_is_still_admitted(self):
        """The guard must not over-refuse the case the file already pins.

        A template built from the same product that caused the flag is NaN
        exactly where the data is, and ``jnp.where(seen, ...)`` removes it
        before any arithmetic sees it. That epoch has a perfectly good summary
        of its unflagged samples, and refusing it would be a second bug.
        """
        import jax.numpy as jnp

        sigma = jnp.array([camp.SIGMA] * 6 + [jnp.inf, jnp.inf])
        template = jnp.asarray(camp.TEMPLATE).at[6].set(jnp.nan)
        term = _with_template(template, sigma=sigma)
        assert np.isfinite(float(term.template_projections[0]))

    def test_a_scalar_is_refused_rather_than_broadcast_into_a_projection(self):
        """Finding 7: ``templates=`` was never shape-checked at all.

        A scalar broadcast to a constant vector and came back as a projection
        that reads exactly like a measurement of the named systematic. On this
        epoch the real ``gain_ripple`` projects to ``-0.4957`` and the scalar
        ``1.0`` to ``-1.1330`` -- a *louder* detection of a template nobody
        supplied. ``3.0`` gives the same ``-1.1330``, because the projection is
        onto the direction and a constant vector has one direction whatever its
        length, so not even the magnitude betrays it.
        """
        from rheplicant.core.errors import StateValidationError

        with pytest.raises(StateValidationError, match="usable systematic template"):
            _with_template(1.0)

    def test_the_wrong_length_names_the_template_instead_of_the_broadcast(self):
        """It raised ``Incompatible shapes for broadcasting: shapes=[(8,), (5,), ()]``.

        Which names no template, no epoch, and no remedy.
        """
        import jax.numpy as jnp

        from rheplicant.core.errors import StateValidationError

        with pytest.raises(StateValidationError, match="gain_ripple.*8 samples"):
            _with_template(jnp.arange(5.0))


def test_the_summary_is_about_a_hundred_bytes_an_epoch():
    """Section 9.3's size claim, which is what makes it storable for 1000 nights."""
    term = camp.terms(1, biased=False)[0]
    payload = (
        np.asarray(term.residual_chi2).nbytes
        + np.asarray(term.template_projections).nbytes
        + 8
    )
    assert payload <= 128


def test_epoch_residuals_reports_one_row_per_epoch_in_order():
    collected = camp.terms(5, biased=False)
    rows = epoch_residuals(collected)
    assert tuple(row.epoch_id for row in rows) == tuple(f"n{e}" for e in range(5))
    assert all(row.dof == camp.N_SAMPLES - camp.N_THETA for row in rows)


def test_a_marginalised_nuisance_is_part_of_the_nights_own_best_fit():
    """The plan's version of ``_residual_summary`` got this wrong, measurably.

    It projected onto the *global* block alone. A nuisance integrated out at
    compression is part of what the night fitted, so excluding it leaves the
    nuisance's own contribution in the residual while the DOF is over-counted by
    its rank. Measured over 4000 clean epochs with a three-column nuisance:
    including it gives ``dof = 3`` and mean chi-square ``3.0000``; excluding it
    gives ``dof = 6`` against mean chi-square ``28.44``, which is a
    nine-sigma-per-epoch detection of nothing at all.

    400 epochs here rather than 4000, which is enough to separate 3 from 28 by a
    factor of ten and cheap enough to run every session.
    """
    import jax.numpy as jnp

    from rheplicant.inference.compress import compress_linear

    rng = np.random.default_rng(5)
    nuisance = rng.normal(size=(camp.N_SAMPLES, 3))
    collected = []
    for e in range(400):
        data = (
            camp.DESIGN @ camp.TRUTH
            + nuisance @ np.array([0.7, -1.1, 0.4])
            + camp.SIGMA * rng.normal(size=camp.N_SAMPLES)
        )
        collected.append(
            compress_linear(
                design={"x": jnp.asarray(camp.DESIGN)},
                observed=jnp.asarray(data),
                noise_std=camp.SIGMA,
                shapes={"x": (camp.N_THETA,)},
                epoch_id=f"p{e}",
                nuisance_design={"p": jnp.asarray(nuisance)},
                nuisance_prior_std={"p": 5.0},
                nuisance_shapes={"p": (3,)},
            )
        )
    assert collected[0].residual_dof == camp.N_SAMPLES - camp.N_THETA - 3
    chi2 = np.array([float(term.residual_chi2) for term in collected])
    assert chi2.mean() == pytest.approx(3.0, abs=4 * np.sqrt(6.0 / 400))


def test_a_campaign_that_names_different_templates_is_refused():
    """The projections are positional, so averaging them needs one order.

    Two nights naming ``("a", "b")`` and ``("b", "a")`` have compatible shapes
    and incompatible contents; averaging positionally reports each template's
    mean as a mixture of both, which is a finite, plausible number about
    nothing.
    """
    mixed = camp.terms(2, biased=False) + camp.terms(
        2, biased=False, seed=99, templates={"other": camp.TEMPLATE}
    )
    with pytest.raises(ValueError, match="different systematic templates"):
        coherent_mode(mixed)


def test_a_campaign_naming_the_same_templates_is_not_refused():
    """A guard that over-refuses is its own bug: the nearest legitimate case."""
    same = camp.terms(2, biased=False) + camp.terms(2, biased=False, seed=99)
    assert coherent_mode(same)["n_epochs"] == 4


def test_a_campaign_with_no_template_at_all_still_reports_its_chi_square():
    """The half that needs no guess about what the fault looks like."""
    report = coherent_mode(camp.terms(640, biased=True, templates={}))
    assert report["templates"] == {}
    assert report["chi2_z"] > 10.0


def test_an_empty_campaign_is_refused_rather_than_reported_as_clean():
    with pytest.raises(ValueError, match="at least one epoch"):
        coherent_mode(())


def test_a_term_whose_projection_count_disagrees_with_its_names_is_refused():
    """Read by name across a campaign, so a mismatch misattributes a ripple."""
    import jax.numpy as jnp

    from rheplicant.core.errors import StateValidationError
    from rheplicant.inference.compressed import QuadraticLikelihood
    from rheplicant.inference.sqrtinfo import SqrtInfo

    info = SqrtInfo(
        factor=jnp.zeros((1, 2)),
        target=jnp.zeros(1),
        offset=jnp.zeros(()),
        names=("x",),
        shapes=((2,),),
    )
    with pytest.raises(StateValidationError, match="another night's template"):
        QuadraticLikelihood(
            info=info,
            epoch_id="bad",
            n_observed=8,
            template_names=("a", "b"),
            template_projections=jnp.zeros(3),
        )
    # And the empty-array spelling of "no templates", which is a different
    # pytree from None and would misalign the archive rather than error.
    with pytest.raises(StateValidationError, match="reads every later leaf"):
        QuadraticLikelihood(
            info=info, epoch_id="bad", n_observed=8, template_projections=jnp.zeros(0)
        )
    # The nearest legitimate case: names and projections that do agree.
    assert QuadraticLikelihood(
        info=info,
        epoch_id="fine",
        n_observed=8,
        template_names=("a", "b"),
        template_projections=jnp.zeros(2),
    ).template_names == ("a", "b")


def test_the_inputs_a_term_records_are_sorted_and_hashable():
    """Section 9.5 compares these pairs for equality, so the order must be one."""
    import jax.numpy as jnp

    from rheplicant.inference.compress import compress_linear

    written = {"cal_solution": "sha256:0f17", "beam_model": "sha256:b3ee"}
    term = compress_linear(
        design={"x": jnp.asarray(camp.DESIGN)},
        observed=jnp.asarray(camp.DESIGN @ camp.TRUTH),
        noise_std=camp.SIGMA,
        shapes={"x": (camp.N_THETA,)},
        epoch_id="provenance",
        inputs=written,
    )
    assert term.inputs == (
        ("beam_model", "sha256:b3ee"),
        ("cal_solution", "sha256:0f17"),
    )
    assert hash(term.inputs)
    assert camp.terms(1, biased=False)[0].inputs == ()
