"""``fisher_information`` must be able to read the declared prior (defect A5).

``Latent(prior=...)`` is the package's single statement of what a latent is a
priori, and every other exit reads it: ``to_numpyro_model`` samples it,
``wiener_solve`` and ``gcr_sample`` solve with it as ``S`` and refuse a
prior-free linear latent by name. The Fisher route was the one exit that never
saw the ``ParameterSpace`` at all, so the declaration could not reach it:

    declared Normal(10, 5.0)   -> sigma('amp') = 0.00568182
    declared Normal(10, 1e-06) -> sigma('amp') = 0.00568182

A 5,000,000x tightening of the prior moved the reported error bar by exactly
zero, and nothing in the output said the matrix was likelihood-only.

Two things are being pinned here. First that ``space=None`` still returns the
LIKELIHOOD Fisher, bit for bit — it is a public function and that is a real
quantity, not a bug. Second that ``space=`` adds the prior's own curvature,
relabels what it returns, and refuses the priors that have no curvature to add.
"""

import jax
import jax.numpy as jnp
import pytest

from rheplicant.core.errors import ParameterSpaceError, StateValidationError
from rheplicant.inference import Bind, Latent, ParameterSpace
from rheplicant.inference.uncertainty import (
    fisher_information,
    parameter_covariance,
)

pytest.importorskip("numpyro")
import numpyro.distributions as dist  # noqa: E402

N_DATA = 12
NOISE = 0.4


@pytest.fixture
def design():
    """A fixed, asymmetric design matrix over ``{'a_vec': (2,), 'z_scalar': ()}``.

    Not alphabetical in declaration order and not equal in size, so the
    name -> span mapping the prior term is placed through is doing real work.
    """
    return jax.random.normal(jax.random.key(3), (N_DATA, 3))


def make_forward(design):
    def forward(values):
        return design @ jnp.concatenate(
            [values["a_vec"], jnp.atleast_1d(values["z_scalar"])]
        )

    return forward


def make_space(vec_scale, scalar_scale):
    """A space whose two latents carry deliberately DIFFERENT prior widths."""
    return ParameterSpace(
        latents=[
            Latent(
                "a_vec",
                init=jnp.array([2.0, 3.0]),
                prior=dist.Normal(jnp.zeros(2), jnp.asarray(vec_scale)),
            ),
            Latent(
                "z_scalar",
                init=jnp.array(1.0),
                prior=dist.Normal(0.0, scalar_scale),
            ),
        ],
        bindings=[
            Bind("a_vec", into=lambda p: p["x"], fn=lambda v: v),
            Bind("z_scalar", into=lambda p: p["y"], fn=lambda v: v),
        ],
    )


VALUES = {"a_vec": jnp.array([2.0, 3.0]), "z_scalar": jnp.array(1.0)}


class TestSpaceIsOptional:
    """``space=None`` is today's behaviour and has to stay exactly that."""

    def test_without_a_space_the_matrix_is_the_likelihood_fisher(self, design):
        forward = make_forward(design)
        fisher = fisher_information(forward, VALUES, noise_std=NOISE)
        assert jnp.allclose(
            fisher.matrix, design.T @ design / NOISE**2, rtol=1e-5
        )

    def test_without_a_space_the_kind_is_unchanged(self, design):
        forward = make_forward(design)
        assert fisher_information(forward, VALUES, noise_std=NOISE).kind == "fisher"

    def test_a_declared_prior_alone_changes_nothing(self, design):
        """The defect, kept as a regression: passing no space must not start
        silently reading priors from somewhere else."""
        forward = make_forward(design)
        loose = fisher_information(forward, VALUES, noise_std=NOISE)
        assert jnp.allclose(loose.matrix, design.T @ design / NOISE**2, rtol=1e-5)


class TestPriorEntersTheMatrix:
    def test_the_prior_precision_lands_on_the_right_spans(self, design):
        """``F_post = F_like + diag(1/scale^2)``, exactly, for a linear model."""
        forward = make_forward(design)
        space = make_space(vec_scale=[0.5, 0.25], scalar_scale=2.0)
        posterior = fisher_information(
            forward, VALUES, noise_std=NOISE, space=space
        )
        expected = design.T @ design / NOISE**2 + jnp.diag(
            jnp.array([1 / 0.5**2, 1 / 0.25**2, 1 / 2.0**2])
        )
        assert jnp.allclose(posterior.matrix, expected, rtol=1e-5)

    def test_the_span_order_follows_the_flattening_not_the_declaration(self, design):
        """``a_vec`` is declared first and also sorts first; check that the
        placement is derived, by moving the WIDTHS and watching which rows
        respond rather than trusting either order."""
        forward = make_forward(design)
        base = fisher_information(forward, VALUES, noise_std=NOISE).matrix
        space = make_space(vec_scale=[0.5, 0.25], scalar_scale=2.0)
        added = (
            fisher_information(forward, VALUES, noise_std=NOISE, space=space).matrix
            - base
        )
        cov = parameter_covariance(
            fisher_information(forward, VALUES, noise_std=NOISE, space=space)
        )
        start, stop = cov.span("z_scalar")
        assert jnp.allclose(jnp.diag(added)[start:stop], 1 / 2.0**2, rtol=1e-6)

    def test_tightening_the_prior_tightens_the_error_bar(self, design):
        """The measurement from the defect report, as an assertion.

        **Float64, and the whole computation is inside the block.** The tight
        arm declares a prior four orders narrower than the data supports,
        which is the point being made -- and a posterior precision holding
        both is conditioned at ``kappa = 3.2e6``, past the float32 ceiling of
        2.90e+03 that ``parameter_covariance`` took from the far side (D29).
        The conditioning is not incidental to this test; it is the four orders
        the test is about, so widening the arithmetic is the honest fix rather
        than a way around a guard.

        The design is WIDENED rather than redrawn. It is data, and the
        condition number is a property of the matrix, not of how its entries
        were sampled -- so widening keeps one spelling of the fixture, where
        redrawing in float64 would make a second model that no assertion here
        compares against the first.
        """
        with jax.enable_x64(True):
            wide = jnp.asarray(design, jnp.float64)
            values = {k: jnp.asarray(v, jnp.float64) for k, v in VALUES.items()}
            forward = make_forward(wide)
            loose = parameter_covariance(
                fisher_information(
                    forward, values, noise_std=NOISE,
                    space=make_space(vec_scale=[5.0, 5.0], scalar_scale=5.0),
                )
            ).sigma("z_scalar")
            tight = parameter_covariance(
                fisher_information(
                    forward, values, noise_std=NOISE,
                    space=make_space(vec_scale=[5.0, 5.0], scalar_scale=1e-4),
                )
            ).sigma("z_scalar")
        assert float(tight) < float(loose)
        # A prior 4 orders tighter than the data pins the parameter to itself.
        assert float(tight) == pytest.approx(1e-4, rel=1e-2)
        # The block took: a float32 run of this reaches the SAME assertions by
        # the route the ceiling exists to refuse, and nothing above would say.
        assert tight.dtype == jnp.float64

    def test_a_prior_that_does_not_bind_barely_moves_anything(self, design):
        """The other direction: a wide prior must be nearly a no-op, or the
        term has been added on the wrong scale."""
        forward = make_forward(design)
        likelihood_only = parameter_covariance(
            fisher_information(forward, VALUES, noise_std=NOISE)
        ).sigma("z_scalar")
        with_wide_prior = parameter_covariance(
            fisher_information(
                forward, VALUES, noise_std=NOISE,
                space=make_space(vec_scale=[1e4, 1e4], scalar_scale=1e4),
            )
        ).sigma("z_scalar")
        assert jnp.allclose(likelihood_only, with_wide_prior, rtol=1e-3)

    def test_an_expanded_normal_is_recognised(self, design):
        """``Normal(...).expand([2])`` is still a Gaussian on the latent —
        the same unwrapping the conjugate exits already do."""
        forward = make_forward(design)
        space = ParameterSpace(
            latents=[
                Latent(
                    "a_vec",
                    init=jnp.array([2.0, 3.0]),
                    prior=dist.Normal(0.0, 0.5).expand([2]),
                ),
                Latent("z_scalar", init=jnp.array(1.0), prior=dist.Normal(0.0, 2.0)),
            ],
            bindings=[
                Bind("a_vec", into=lambda p: p["x"], fn=lambda v: v),
                Bind("z_scalar", into=lambda p: p["y"], fn=lambda v: v),
            ],
        )
        posterior = fisher_information(
            forward, VALUES, noise_std=NOISE, space=space
        )
        expected = design.T @ design / NOISE**2 + jnp.diag(
            jnp.array([1 / 0.5**2, 1 / 0.5**2, 1 / 2.0**2])
        )
        assert jnp.allclose(posterior.matrix, expected, rtol=1e-5)


class TestKindIsCarried:
    def test_a_space_relabels_the_matrix(self, design):
        forward = make_forward(design)
        space = make_space(vec_scale=[0.5, 0.25], scalar_scale=2.0)
        fisher = fisher_information(forward, VALUES, noise_std=NOISE, space=space)
        assert fisher.kind == "posterior_precision"

    def test_the_covariance_kind_follows_the_precision_it_came_from(self, design):
        forward = make_forward(design)
        space = make_space(vec_scale=[0.5, 0.25], scalar_scale=2.0)
        assert (
            parameter_covariance(
                fisher_information(forward, VALUES, noise_std=NOISE)
            ).kind
            == "covariance"
        )
        assert (
            parameter_covariance(
                fisher_information(forward, VALUES, noise_std=NOISE, space=space)
            ).kind
            == "posterior_covariance"
        )

    def test_sigma_refuses_a_likelihood_fisher(self, design):
        forward = make_forward(design)
        fisher = fisher_information(forward, VALUES, noise_std=NOISE)
        with pytest.raises(StateValidationError, match="parameter_covariance"):
            fisher.sigma("z_scalar")

    def test_sigma_refuses_a_posterior_precision_too(self, design):
        """The second branch of the same guard: sqrt(diag()) of a PRECISION is
        no more an error bar than sqrt(diag(F)) was."""
        forward = make_forward(design)
        space = make_space(vec_scale=[0.5, 0.25], scalar_scale=2.0)
        precision = fisher_information(
            forward, VALUES, noise_std=NOISE, space=space
        )
        with pytest.raises(StateValidationError, match="parameter_covariance"):
            precision.sigma("z_scalar")

    def test_sigma_works_on_both_covariance_kinds(self, design):
        forward = make_forward(design)
        space = make_space(vec_scale=[0.5, 0.25], scalar_scale=2.0)
        plain = parameter_covariance(
            fisher_information(forward, VALUES, noise_std=NOISE)
        )
        posterior = parameter_covariance(
            fisher_information(forward, VALUES, noise_std=NOISE, space=space)
        )
        assert plain.sigma("a_vec").shape == (2,)
        assert posterior.sigma("a_vec").shape == (2,)
        assert float(posterior.sigma("z_scalar")) < float(plain.sigma("z_scalar"))


class TestRefusals:
    def test_a_prior_with_no_quadratic_form_is_refused_by_name(self, design):
        """A Uniform has no second derivative to contribute; substituting its
        variance would report a posterior nobody declared."""
        forward = make_forward(design)
        space = ParameterSpace(
            latents=[
                Latent(
                    "a_vec",
                    init=jnp.array([2.0, 3.0]),
                    prior=dist.Normal(jnp.zeros(2), jnp.full(2, 0.5)),
                ),
                Latent(
                    "z_scalar",
                    init=jnp.array(1.0),
                    prior=dist.Uniform(0.0, 3.0),
                ),
            ],
            bindings=[
                Bind("a_vec", into=lambda p: p["x"], fn=lambda v: v),
                Bind("z_scalar", into=lambda p: p["y"], fn=lambda v: v),
            ],
        )
        with pytest.raises(ParameterSpaceError, match="z_scalar") as excinfo:
            fisher_information(forward, VALUES, noise_std=NOISE, space=space)
        assert "Uniform" in str(excinfo.value)

    def test_a_lognormal_is_refused_rather_than_duck_typed(self, design):
        """``LogNormal`` carries ``.loc``/``.scale`` and even a Normal
        ``base_dist`` while being Gaussian in log x, not in x."""
        forward = make_forward(design)
        space = ParameterSpace(
            latents=[
                Latent(
                    "a_vec",
                    init=jnp.array([2.0, 3.0]),
                    prior=dist.Normal(jnp.zeros(2), jnp.full(2, 0.5)),
                ),
                Latent("z_scalar", init=jnp.array(1.0), prior=dist.LogNormal(0.0, 1.0)),
            ],
            bindings=[
                Bind("a_vec", into=lambda p: p["x"], fn=lambda v: v),
                Bind("z_scalar", into=lambda p: p["y"], fn=lambda v: v),
            ],
        )
        with pytest.raises(ParameterSpaceError, match="LogNormal"):
            fisher_information(forward, VALUES, noise_std=NOISE, space=space)

    def test_a_prior_free_latent_is_refused_by_name(self, design):
        forward = make_forward(design)
        space = ParameterSpace(
            latents=[
                Latent(
                    "a_vec",
                    init=jnp.array([2.0, 3.0]),
                    prior=dist.Normal(jnp.zeros(2), jnp.full(2, 0.5)),
                ),
                Latent("z_scalar", init=jnp.array(1.0)),
            ],
            bindings=[
                Bind("a_vec", into=lambda p: p["x"], fn=lambda v: v),
                Bind("z_scalar", into=lambda p: p["y"], fn=lambda v: v),
            ],
        )
        with pytest.raises(ParameterSpaceError, match="z_scalar"):
            fisher_information(forward, VALUES, noise_std=NOISE, space=space)

    def test_unnamed_parameters_cannot_carry_a_space(self, design):
        """A plain array pytree has no names, so there is no span to add the
        prior at — and adding it by position would be a guess."""
        space = make_space(vec_scale=[0.5, 0.25], scalar_scale=2.0)
        with pytest.raises(StateValidationError, match="not named"):
            fisher_information(
                lambda theta: design @ theta,
                jnp.array([2.0, 3.0, 1.0]),
                noise_std=NOISE,
                space=space,
            )

    def test_params_that_do_not_match_the_space_are_refused(self, design):
        space = make_space(vec_scale=[0.5, 0.25], scalar_scale=2.0)
        with pytest.raises(ParameterSpaceError, match="do not match"):
            fisher_information(
                lambda v: design @ jnp.concatenate(
                    [v["a_vec"], jnp.atleast_1d(v["other"])]
                ),
                {"a_vec": jnp.array([2.0, 3.0]), "other": jnp.array(1.0)},
                noise_std=NOISE,
                space=space,
            )
