import jax
import jax.numpy as jnp
import numpy as np
import pytest

numpyro = pytest.importorskip("numpyro")
import numpyro.distributions as dist  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402

from rheplicant.core.errors import StateValidationError  # noqa: E402
from rheplicant.inference import Bind, Latent, ParameterSpace  # noqa: E402
from rheplicant.inference.compress import compress_linear  # noqa: E402
from rheplicant.inference.factorize import Factorization  # noqa: E402
from rheplicant.inference.memory import BayesMemory  # noqa: E402

SIGMA, PRIOR_STD, TRUTH = 0.2, 5.0, jnp.array([1.3, -0.7])


def _memory(n_epochs=8):
    latent = Latent(
        "x", init=jnp.zeros(2),
        prior=dist.Normal(0.0, PRIOR_STD).expand([2]).to_event(1), linear=True,
    )
    space = ParameterSpace(latents=(latent,), bindings=(Bind("x", into=lambda p: p.x),))
    memory = BayesMemory(Factorization(space))
    for epoch in range(n_epochs):
        kd, kn = jax.random.split(jax.random.key(epoch))
        design = jax.random.normal(kd, (30, 2))
        data = design @ TRUTH + SIGMA * jax.random.normal(kn, (30,))
        memory = memory.remember(
            compress_linear(
                design={"x": design}, observed=data, noise_std=SIGMA,
                shapes={"x": (2,)}, epoch_id=f"n{epoch}",
            )
        )
    return memory


def test_nuts_over_the_memory_recovers_the_analytic_posterior():
    memory = _memory()
    mcmc = MCMC(
        NUTS(memory.to_numpyro_model()), num_warmup=500, num_samples=2000,
        progress_bar=False,
    )
    mcmc.run(jax.random.key(0))
    draws = np.asarray(mcmc.get_samples()["x"])

    hessian = jax.hessian(memory.log_posterior)({"x": jnp.zeros(2)})["x"]["x"]
    covariance = np.linalg.inv(-np.asarray(hessian))
    grad = np.asarray(jax.grad(memory.log_posterior)({"x": jnp.zeros(2)})["x"])
    mean = covariance @ grad

    # Four Monte-Carlo standard errors per component, which differ between
    # components -- so the loop, rather than one call with an array `atol`.
    # numpy 2.5.1 builds assert_allclose's failure header eagerly as
    # f"...atol={atol:g}", so an array atol raises `TypeError: unsupported
    # format string passed to numpy.ndarray.__format__` even when the two
    # arrays are equal -- measured on an array compared against itself.
    tolerance = 4 * np.sqrt(np.diag(covariance) / 2000) + 1e-3
    for index, atol in enumerate(tolerance):
        np.testing.assert_allclose(
            draws.mean(axis=0)[index], mean[index], atol=float(atol),
            err_msg=f"posterior mean of x[{index}]",
        )
    np.testing.assert_allclose(
        draws.std(axis=0), np.sqrt(np.diag(covariance)), rtol=0.08
    )


def test_the_model_refuses_a_noise_argument():
    with pytest.raises(StateValidationError, match="already"):
        _memory().to_numpyro_model(noise_std=0.2)
