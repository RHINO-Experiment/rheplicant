import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.inference.compress import compress_linear


def _design(key, n_data, n_theta, n_phi):
    ka, kb = jax.random.split(key)
    return jax.random.normal(ka, (n_data, n_theta)), jax.random.normal(kb, (n_data, n_phi))


def _dense_marginal_log_likelihood(a_theta, a_phi, data, sigma, prior_std, x):
    """Oracle: the analytic N(A_theta x, sigma^2 I + A_phi S A_phi^T) density."""
    cov = np.diag(np.full(len(data), sigma**2)) + a_phi @ np.diag(
        np.full(a_phi.shape[1], prior_std**2)
    ) @ a_phi.T
    resid = np.asarray(data) - np.asarray(a_theta) @ np.asarray(x)
    sign, logdet = np.linalg.slogdet(cov)
    assert sign > 0
    return -0.5 * (
        resid @ np.linalg.solve(cov, resid) + logdet + len(data) * np.log(2 * np.pi)
    )


def test_with_no_nuisance_the_term_reproduces_the_gaussian_log_likelihood():
    key = jax.random.key(0)
    a_theta, _ = _design(key, n_data=40, n_theta=3, n_phi=1)
    truth = jnp.array([0.4, -1.1, 2.0])
    data = a_theta @ truth + 0.1 * jax.random.normal(jax.random.key(1), (40,))

    term = compress_linear(
        design={"x": a_theta}, observed=data, noise_std=0.1,
        shapes={"x": (3,)}, epoch_id="e0",
    )
    probe = {"x": jnp.array([0.1, 0.2, 0.3])}
    resid = data - a_theta @ probe["x"]
    expected = -0.5 * float(
        jnp.sum(resid**2) / 0.01 + 40 * jnp.log(2 * jnp.pi * 0.01)
    )
    assert float(term(probe)) == pytest.approx(expected, rel=1e-10)


def test_a_linear_gaussian_nuisance_is_marginalised_to_the_analytic_density():
    """The pin: the covariance must be N + A S A^T, log-det and prior norm included."""
    key = jax.random.key(2)
    a_theta, a_phi = _design(key, n_data=40, n_theta=2, n_phi=3)
    data = a_theta @ jnp.array([1.0, -0.5]) + 0.1 * jax.random.normal(
        jax.random.key(3), (40,)
    )

    term = compress_linear(
        design={"x": a_theta}, nuisance_design={"p": a_phi},
        nuisance_prior_std={"p": 0.7}, observed=data, noise_std=0.1,
        shapes={"x": (2,)}, nuisance_shapes={"p": (3,)}, epoch_id="e0",
    )
    probe = {"x": jnp.array([0.3, 0.9])}
    expected = _dense_marginal_log_likelihood(
        a_theta, a_phi, data, 0.1, 0.7, probe["x"]
    )
    assert float(term(probe)) == pytest.approx(expected, rel=1e-9)


def test_the_nuisance_priors_own_normalisation_is_in_the_offset():
    """Six nuisances at std=20 -- the `-sum(log(std))` term is worth 17.9744 nats.

    The Gaussian integral over the nuisance contributes ``+(n_phi/2) log(2 pi)``,
    which cancels the prior's own copy of that factor; the prior's
    ``-sum(log(std))`` has nothing to cancel against and must be carried. An
    implementation that keeps only the discarded block's ``-sum(log|diag|)``
    is wrong by exactly ``sum(log(std)) = n_phi * log(prior_std)``, which on
    this case is ``6 * log(20) = 17.974394``. Measured over 60 probe points in
    ``x``: the shipped term agrees with the dense oracle to a maximum relative
    error of 2.5e-11, and dropping the prior normalisation shifts it by
    17.974394 nats at every one of them. A constant never perturbs a gradient
    or a curvature, so only a comparison against an absolute density -- this
    one -- can see it.

    The tolerance below is absolute, not relative, and is set by the *oracle*:
    ``prior_std = 20`` gives the dense covariance a condition number of
    2.64e+06, so its ``slogdet`` plus ``solve`` reproduce a log-density of
    -8163.2755 to 1.3e-07 nats. That is five orders below the term being
    pinned, which is the point of choosing a large ``prior_std``.

    The case is chosen so the number is large and unmistakable. At
    ``prior_std = 1`` the term is ``log(1) = 0`` and a broken implementation
    passes; the plan's own probe table has exactly one row that is correct
    without the term, and it is the ``n_phi=1, prior_std=1`` row.
    """
    n_phi, prior_std = 6, 20.0
    key = jax.random.key(10)
    a_theta, a_phi = _design(key, n_data=40, n_theta=2, n_phi=n_phi)
    data = a_theta @ jnp.array([0.8, -1.4]) + 0.1 * jax.random.normal(
        jax.random.key(11), (40,)
    )

    term = compress_linear(
        design={"x": a_theta}, nuisance_design={"p": a_phi},
        nuisance_prior_std={"p": prior_std}, observed=data, noise_std=0.1,
        shapes={"x": (2,)}, nuisance_shapes={"p": (n_phi,)}, epoch_id="e0",
    )
    probe = {"x": jnp.array([0.3, 0.9])}
    expected = _dense_marginal_log_likelihood(
        a_theta, a_phi, data, 0.1, prior_std, probe["x"]
    )
    shipped = float(term(probe))
    assert shipped == pytest.approx(expected, rel=1e-9)

    # What the same arithmetic returns with the prior's normalisation dropped:
    # off by sum(log(std)) = 6 * log(20) = 17.974394, and by nothing else --
    # the leftover is 1.3e-07 nats, the oracle's roundoff, not a second bug.
    omission = n_phi * np.log(prior_std)
    assert omission == pytest.approx(17.974394, abs=1e-5)
    assert (shipped + omission) - expected == pytest.approx(omission, abs=1e-5)
    assert abs(shipped - expected) < 1e-5


def test_ignoring_the_nuisance_would_be_too_tight_which_this_measures():
    key = jax.random.key(4)
    a_theta, a_phi = _design(key, n_data=40, n_theta=2, n_phi=3)
    data = a_theta @ jnp.array([1.0, -0.5]) + 0.1 * jax.random.normal(
        jax.random.key(5), (40,)
    )
    common = dict(
        design={"x": a_theta}, observed=data, noise_std=0.1,
        shapes={"x": (2,)}, epoch_id="e0",
    )
    without = compress_linear(**common)
    with_nuisance = compress_linear(
        **common, nuisance_design={"p": a_phi}, nuisance_prior_std={"p": 0.7},
        nuisance_shapes={"p": (3,)},
    )
    tight = np.linalg.eigvalsh(np.asarray(without.info.fisher()))
    honest = np.linalg.eigvalsh(np.asarray(with_nuisance.info.fisher()))
    assert honest.min() < tight.min()


def test_flagged_samples_give_a_finite_term_not_minus_infinity():
    """sigma = inf must not take the normalisation with it."""
    from rheplicant.inference import FlaggedNoise, HomoscedasticNoise

    a_theta, _ = _design(jax.random.key(6), n_data=8, n_theta=2, n_phi=1)
    data = jax.random.normal(jax.random.key(7), (8,))
    flags = jnp.array([False, True, False, False, False, False, True, False])
    term = compress_linear(
        design={"x": a_theta}, observed=data,
        noise_std=FlaggedNoise(HomoscedasticNoise(jnp.array(0.1)), flags),
        shapes={"x": (2,)}, epoch_id="e0",
    )
    assert np.isfinite(float(term({"x": jnp.zeros(2)})))
    assert term.n_observed == 6


def test_a_fully_flagged_epoch_gives_the_null_term():
    from rheplicant.inference import FlaggedNoise, HomoscedasticNoise

    a_theta, _ = _design(jax.random.key(8), n_data=8, n_theta=2, n_phi=1)
    term = compress_linear(
        design={"x": a_theta}, observed=jax.random.normal(jax.random.key(9), (8,)),
        noise_std=FlaggedNoise(HomoscedasticNoise(jnp.array(0.1)), jnp.ones(8, bool)),
        shapes={"x": (2,)}, epoch_id="e0",
    )
    assert term.n_observed == 0
    assert float(term({"x": jnp.array([3.0, -2.0])})) == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(np.asarray(term.info.fisher()), 0.0)


def test_a_rank_deficient_epoch_is_representable_and_not_an_error():
    """One epoch that constrains one direction of two."""
    design = jnp.array([[1.0, 0.0]] * 10)
    term = compress_linear(
        design={"x": design}, observed=jnp.zeros(10), noise_std=0.1,
        shapes={"x": (2,)}, epoch_id="e0",
    )
    assert int(np.linalg.matrix_rank(np.asarray(term.info.fisher()), tol=1e-9)) == 1


class TestANaNAtAFlaggedSampleMustNotReachTheTerm:
    """The mask has to SELECT, because `0.0 * nan` is `nan`.

    A flagged sample is exactly where a bad value lives -- that is usually why
    it was flagged. Multiplying it by a zero weight propagates the very number
    the mask exists to discard, and the damage is asymmetric in the worst way:
    the poison reaches ``target`` and ``offset`` while ``factor`` stays finite,
    so curvature looks perfect. Measured on the earlier implementation, with
    one NaN at a flagged sample: ``log_likelihood`` nan, while ``audit()``
    reported ``fisher_lambda_min`` 94.06, ``fisher_condition`` 7.11 and
    ``all_exact`` True -- a healthy, well-conditioned campaign whose every
    density is NaN. Combining is a QR, so it is irreversible.
    """

    def _noise(self, flags):
        from rheplicant.inference import FlaggedNoise, HomoscedasticNoise

        return FlaggedNoise(HomoscedasticNoise(jnp.array(0.1)), flags)

    def _design_and_data(self):
        a_theta, _ = _design(jax.random.key(20), n_data=8, n_theta=2, n_phi=1)
        return a_theta, np.asarray(jax.random.normal(jax.random.key(21), (8,)))

    @pytest.mark.parametrize("poisoned", [np.nan, np.inf, -np.inf])
    def test_a_nonfinite_observation_at_a_flagged_sample_is_discarded(self, poisoned):
        a_theta, data = self._design_and_data()
        flags = jnp.array([False, True, False, False, False, False, True, False])
        clean = compress_linear(
            design={"x": a_theta}, observed=jnp.asarray(data),
            noise_std=self._noise(flags), shapes={"x": (2,)}, epoch_id="clean",
        )
        spiked = data.copy()
        spiked[1] = poisoned  # index 1 is flagged
        poisonedterm = compress_linear(
            design={"x": a_theta}, observed=jnp.asarray(spiked),
            noise_std=self._noise(flags), shapes={"x": (2,)}, epoch_id="spiked",
        )
        probe = {"x": jnp.array([0.3, -0.7])}
        # Not merely finite -- EQUAL. A flagged sample contributes nothing, so
        # its value cannot matter at all.
        assert float(poisonedterm(probe)) == pytest.approx(float(clean(probe)), abs=1e-12)
        assert poisonedterm.n_observed == clean.n_observed == 6

    def test_a_nan_in_a_design_row_at_a_flagged_sample_is_discarded(self):
        """The other half: the poison reaching `factor` defeats audit()'s own guard.

        ``audit()`` reports ``inf`` when the Fisher is singular via
        ``smallest <= 0``, and ``nan <= 0`` is False -- so a NaN curvature
        reports ``fisher_condition = nan`` and takes the "healthy" branch.
        """
        a_theta, data = self._design_and_data()
        flags = jnp.array([False, True, False, False, False, False, True, False])
        spiked = np.asarray(a_theta).copy()
        spiked[1, :] = np.nan  # flagged row
        term = compress_linear(
            design={"x": jnp.asarray(spiked)}, observed=jnp.asarray(data),
            noise_std=self._noise(flags), shapes={"x": (2,)}, epoch_id="e",
        )
        assert np.all(np.isfinite(np.asarray(term.info.fisher())))
        assert np.isfinite(float(term({"x": jnp.zeros(2)})))

    def test_a_nan_noise_std_is_refused_rather_than_read_as_a_flag(self):
        """`inf` means not observed; NaN means the noise model broke.

        ``jnp.isfinite`` cannot tell them apart, so an unguarded NaN sigma is
        silently reclassified as a flag: measured, eight samples in and one
        sigma NaN gave a perfectly finite term built on seven, with nothing
        anywhere reporting the loss.
        """
        from rheplicant.core.errors import StateValidationError

        a_theta, data = self._design_and_data()
        sigma = np.full(8, 0.1)
        sigma[3] = np.nan
        with pytest.raises(StateValidationError, match="NaN"):
            compress_linear(
                design={"x": a_theta}, observed=jnp.asarray(data),
                noise_std=jnp.asarray(sigma), shapes={"x": (2,)}, epoch_id="e",
            )


class TestWhatCompressionCanAndCannotBeTracedThrough:
    """`n_observed` is static provenance, so the flag pattern must be concrete.

    It is a Python ``int`` recorded on the term and written into the archive
    manifest -- not an array -- so under a trace it could only be invented. What
    decides the question is whether ``sigma`` is traced, not whether the data
    is: measured, ``observed`` is a tracer under jit, grad AND vmap, while
    ``sigma`` is a tracer only under jit. That is exactly the split between the
    transforms that cannot work and the two that can, so the guard tests sigma.
    """

    def _call(self, observed, noise_std=0.1):
        a_theta, _ = _design(jax.random.key(30), n_data=8, n_theta=2, n_phi=1)
        return compress_linear(
            design={"x": a_theta}, observed=observed, noise_std=noise_std,
            shapes={"x": (2,)}, epoch_id="e",
        )

    def test_jit_is_refused_by_name_rather_than_leaking_a_tracer_error(self):
        """Unguarded this was a raw TracerBoolConversionError from an unrelated line."""
        from rheplicant.core.errors import StateValidationError

        data = jax.random.normal(jax.random.key(31), (8,))
        with pytest.raises(StateValidationError, match="cannot run under jit"):
            jax.jit(lambda d: self._call(d).info.offset)(data)

    def test_traced_flags_are_refused_too(self):
        """A FlaggedNoise whose mask is traced is the same problem by another door."""
        from rheplicant.core.errors import StateValidationError
        from rheplicant.inference import FlaggedNoise, HomoscedasticNoise

        data = jax.random.normal(jax.random.key(32), (8,))

        def go(flags):
            noise = FlaggedNoise(HomoscedasticNoise(jnp.array(0.1)), flags)
            return self._call(data, noise_std=noise).info.offset

        with pytest.raises(StateValidationError, match="cannot run under jit"):
            jax.jit(go)(jnp.zeros(8, bool))

    def test_grad_through_the_data_still_works(self):
        """The transform the guard must NOT refuse, since it works today."""
        data = jax.random.normal(jax.random.key(33), (8,))
        probe = {"x": jnp.array([0.2, -0.4])}
        gradient = jax.grad(lambda d: self._call(d)(probe))(data)
        assert gradient.shape == (8,)
        assert np.all(np.isfinite(np.asarray(gradient)))

    def test_vmap_over_a_stack_of_epochs_still_works(self):
        data = jax.random.normal(jax.random.key(34), (3, 8))
        probe = {"x": jnp.array([0.2, -0.4])}
        values = jax.vmap(lambda d: self._call(d)(probe))(data)
        assert values.shape == (3,)
        assert np.all(np.isfinite(np.asarray(values)))
