"""The scope partition, and the four declarations it refuses.

A `Factorization` is derived from a `ParameterSpace`, so what it has to get
right is not arithmetic but agreement: every latent that is integrated away
inside an epoch needs the prior it is integrated against, every latent that
chains across epochs needs the transition that chains it, and a campaign with
nothing global has nothing to accumulate. Each of those has a plausible wrong
answer that is silent -- a missing prior reads as "no prior yet", a missing
transition reads as independence -- so each is refused at construction rather
than checked later.
"""

import jax.numpy as jnp
import pytest

from rheplicant.core.errors import ParameterSpaceError
from rheplicant.inference import Bind, Latent, ParameterSpace
from rheplicant.inference.factorize import Factorization


class _Prior:
    """A stand-in for a numpyro distribution -- the package duck-types priors."""

    def __init__(self, tag="prior"):
        self.tag = tag


def _space(*latents):
    return ParameterSpace(
        latents=latents,
        bindings=tuple(
            Bind(latent.name, into=lambda p, n=latent.name: getattr(p, n))
            for latent in latents
        ),
    )


def test_partitions_latents_by_scope():
    space = _space(
        Latent("depth", init=-0.5, prior=_Prior()),
        Latent("log_gain", init=0.0, prior=_Prior(), scope="per_epoch"),
        Latent("drift", init=0.0, prior=_Prior(), scope="linked"),
    )
    fact = Factorization(space, linked={"drift": object()})
    assert fact.global_names == ("depth",)
    assert fact.per_epoch_names == ("log_gain",)
    assert fact.linked_names == ("drift",)


def test_global_shapes_follow_the_latents_init():
    space = _space(
        Latent("alms", init=jnp.zeros(4), prior=_Prior()),
        Latent("depth", init=-0.5, prior=_Prior()),
    )
    fact = Factorization(space)
    assert fact.global_shapes == ((4,), ())


def test_a_per_epoch_latent_without_a_prior_is_refused():
    space = _space(
        Latent("depth", init=-0.5, prior=_Prior()),
        Latent("log_gain", init=0.0, scope="per_epoch"),
    )
    with pytest.raises(ParameterSpaceError, match="integrated"):
        Factorization(space)


def test_a_linked_latent_without_a_transition_is_refused():
    space = _space(
        Latent("depth", init=-0.5, prior=_Prior()),
        Latent("drift", init=0.0, prior=_Prior(), scope="linked"),
    )
    with pytest.raises(ParameterSpaceError, match="transition"):
        Factorization(space)


def test_a_transition_for_a_latent_that_is_not_linked_is_refused():
    space = _space(Latent("depth", init=-0.5, prior=_Prior()))
    with pytest.raises(ParameterSpaceError, match="not declared"):
        Factorization(space, linked={"depth": object()})


def test_a_hyper_closure_naming_a_non_global_latent_is_refused():
    space = _space(
        Latent("depth", init=-0.5, prior=_Prior()),
        Latent("log_gain", init=0.0, prior=_Prior(), scope="per_epoch"),
        Latent("scatter", init=0.1, prior=_Prior(), scope="per_epoch"),
    )
    with pytest.raises(ParameterSpaceError, match="global"):
        Factorization(space, hyper={"log_gain": (("scatter",), lambda s: _Prior())})


def test_a_hyper_closure_for_a_global_latent_is_refused():
    space = _space(
        Latent("depth", init=-0.5, prior=_Prior()),
        Latent("scatter", init=0.1, prior=_Prior()),
    )
    with pytest.raises(ParameterSpaceError, match="per-epoch"):
        Factorization(space, hyper={"depth": (("scatter",), lambda s: _Prior())})


def test_a_space_with_no_global_latent_is_refused():
    space = _space(Latent("log_gain", init=0.0, prior=_Prior(), scope="per_epoch"))
    with pytest.raises(ParameterSpaceError, match="nothing to accumulate"):
        Factorization(space)


def test_global_priors_are_exposed_by_name():
    depth_prior = _Prior("depth")
    space = _space(Latent("depth", init=-0.5, prior=depth_prior))
    assert Factorization(space).global_priors == {"depth": depth_prior}


def test_a_global_latent_without_a_prior_is_refused():
    space = _space(Latent("depth", init=-0.5))
    with pytest.raises(ParameterSpaceError, match="no place in a posterior"):
        Factorization(space)
