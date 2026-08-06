"""What a transition promises, and what is checked before it is believed.

`process_std` and `initial_std` are strictly positive, checked once at
construction, and that check is not tidiness -- it is what makes every `zeta_e`
block constrained by construction, which is what lets the filter call the
traceable kernel instead of the checked `marginalise` once per epoch. Delete it
and the divergence it prevents comes back as a large finite plausible number
inside a `lax.scan`, where nothing tests for it.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.core.errors import ParameterSpaceError, StateValidationError
from rheplicant.inference.chain import (
    HyperTransition,
    LinearGaussianTransition,
    ornstein_uhlenbeck,
)
from rheplicant.inference.factorize import Factorization
from tests.evidence import chain_bank as bank


def test_a_transition_records_its_blocks_and_its_width():
    transition = LinearGaussianTransition(
        phi=0.8, process_std=0.5, initial_std=1.0, initial_mean=0.25
    )
    assert transition.width == 1
    assert transition.phi.shape == (1, 1)
    assert float(transition.initial_mean[0]) == 0.25
    assert transition.hyper == ()


def test_an_ornstein_uhlenbeck_is_stationary_at_its_own_sigma():
    """`var(zeta_{e+1}) = phi^2 var + Q` must return sigma^2 when it starts there."""
    transition = ornstein_uhlenbeck(tau=3.0, sigma=0.7)
    phi = float(transition.phi[0, 0])
    q = float(transition.process_std[0]) ** 2
    assert phi == pytest.approx(float(np.exp(-1 / 3.0)), rel=1e-12)
    assert phi**2 * 0.49 + q == pytest.approx(0.49, rel=1e-12)
    assert float(transition.initial_std[0]) == pytest.approx(0.7, rel=1e-12)


def test_a_wide_ornstein_uhlenbeck_is_a_diagonal_chain():
    transition = ornstein_uhlenbeck(tau=5.0, sigma=0.4, width=3)
    assert transition.width == 3
    np.testing.assert_allclose(
        np.asarray(transition.phi),
        float(np.exp(-0.2)) * np.eye(3),
        atol=1e-12,
    )


@pytest.mark.parametrize("bad", [0.0, -0.3])
def test_a_non_positive_process_std_is_refused_at_construction(bad):
    with pytest.raises(StateValidationError, match="strictly positive"):
        LinearGaussianTransition(phi=0.8, process_std=bad, initial_std=1.0)


def test_a_legitimately_tiny_process_std_is_accepted():
    """The nearest legitimate case: a chain that barely moves is still a chain."""
    transition = LinearGaussianTransition(phi=0.999, process_std=1e-9, initial_std=1.0)
    assert float(transition.process_std[0]) == 1e-9


def test_a_mismatched_phi_is_refused_rather_than_broadcast():
    with pytest.raises(StateValidationError, match="phi"):
        LinearGaussianTransition(
            phi=jnp.eye(2), process_std=jnp.ones(3), initial_std=jnp.ones(3)
        )


def test_a_hyper_transition_builds_a_fixed_one_from_named_globals():
    transition = HyperTransition(
        build=lambda values: ornstein_uhlenbeck(tau=values["tau_epochs"], sigma=0.5),
        hyper=("tau_epochs",),
        width=1,
    )
    resolved = transition.at({"tau_epochs": jnp.asarray(4.0), "unused": 1.0})
    assert isinstance(resolved, LinearGaussianTransition)
    assert float(resolved.phi[0, 0]) == pytest.approx(float(np.exp(-0.25)))


def test_a_hyper_transition_resolves_under_a_trace():
    """The whole reason it exists: the blocks are functions of theta."""
    transition = HyperTransition(
        build=lambda values: ornstein_uhlenbeck(tau=values["tau"], sigma=0.5),
        hyper=("tau",),
        width=1,
    )

    def phi_of(tau):
        return transition.at({"tau": tau}).phi[0, 0]

    assert float(jax.jit(phi_of)(4.0)) == pytest.approx(float(np.exp(-0.25)))
    assert float(jax.grad(phi_of)(4.0)) == pytest.approx(
        float(np.exp(-0.25)) / 16.0, rel=1e-9
    )


def test_a_fixed_transition_reports_itself_when_asked_to_resolve():
    transition = LinearGaussianTransition(phi=0.8, process_std=0.5, initial_std=1.0)
    assert transition.at({"anything": 1.0}) is transition


def test_a_hyper_transition_refuses_to_resolve_without_its_own_values():
    """The nearest legitimate mistake: calling the density with theta half-supplied."""
    transition = HyperTransition(
        build=lambda values: ornstein_uhlenbeck(tau=values["tau"], sigma=0.5),
        hyper=("tau",),
        width=1,
    )
    with pytest.raises(StateValidationError, match="no value was given"):
        transition.at({"something_else": jnp.asarray(1.0)})


def test_a_hyper_transition_that_changes_width_under_theta_is_refused():
    """The stored blocks were cut against the declared width and cannot be re-cut."""
    transition = HyperTransition(
        build=lambda values: ornstein_uhlenbeck(tau=values["tau"], sigma=0.5, width=2),
        hyper=("tau",),
        width=1,
    )
    with pytest.raises(StateValidationError, match="width"):
        transition.at({"tau": jnp.asarray(3.0)})


def test_a_transition_built_from_a_non_global_latent_is_refused():
    """Section 11's missing check, in the form that is writable."""
    transition = HyperTransition(
        build=lambda values: ornstein_uhlenbeck(tau=values["log_gain"], sigma=0.5),
        hyper=("log_gain",),
        width=1,
    )
    with pytest.raises(ParameterSpaceError, match="global"):
        bank.factorization(transition)


def test_a_transition_built_from_its_own_chain_is_refused():
    transition = HyperTransition(
        build=lambda values: ornstein_uhlenbeck(tau=values[bank.ZETA_NAME], sigma=0.5),
        hyper=(bank.ZETA_NAME,),
        width=1,
    )
    with pytest.raises(ParameterSpaceError, match="its own"):
        bank.factorization(transition)


def test_a_transition_built_from_a_declared_global_is_accepted():
    """The nearest legitimate case, which is the whole of `linked_hyper`."""
    transition = HyperTransition(
        build=lambda values: ornstein_uhlenbeck(tau=values["t_rx"], sigma=0.5),
        hyper=("t_rx",),
        width=1,
    )
    assert bank.factorization(transition).linked_names == (bank.ZETA_NAME,)


def test_a_fixed_transition_is_accepted_and_declares_nothing():
    """A fixed transition makes the linked check vacuous, which is the design."""
    transition = LinearGaussianTransition(
        phi=bank.PHI, process_std=bank.PROCESS_STD, initial_std=bank.INITIAL_STD
    )
    assert bank.factorization(transition).linked_names == (bank.ZETA_NAME,)


def test_a_transitions_blocks_stay_off_the_treedef():
    """`Factorization.linked` is dynamic now that a transition carries arrays.

    A static field goes into the treedef, where array `__eq__` decides equality;
    equinox warns "A JAX array is being set as static" and `memory.py` records
    that warning as what marks the wrong home for stored terms. Flipping the
    field back to `static=True` makes this test red -- verified by doing it.

    What it is NOT: measured with the field static, two Factorizations built from
    bit-identical transitions still compared treedef-equal at widths 1 and 3, so
    the harm is not a retrace storm. It is that the treedef's identity is then
    decided by the *values* of `phi`, `process_std` and `initial_std`, which is
    exactly what a `HyperTransition` makes vary.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        factorization = bank.factorization(
            LinearGaussianTransition(
                phi=bank.PHI,
                process_std=bank.PROCESS_STD,
                initial_std=bank.INITIAL_STD,
            )
        )
    assert [str(entry.message) for entry in caught if "static" in str(entry.message)] == []
    blocks = jax.tree_util.tree_leaves(factorization.linked[bank.ZETA_NAME])
    assert len(blocks) == 4, "phi, process_std, initial_std and initial_mean are leaves"


def test_represents_must_name_global_latents():
    space = bank.space()
    with pytest.raises(ParameterSpaceError, match="global"):
        Factorization(
            space,
            linked={
                bank.ZETA_NAME: LinearGaussianTransition(
                    phi=bank.PHI,
                    process_std=bank.PROCESS_STD,
                    initial_std=bank.INITIAL_STD,
                )
            },
            represents={"beam_map": ("no_such_latent",)},
        )


def test_represents_accepts_a_global_latent():
    factorization = Factorization(
        bank.space(),
        linked={
            bank.ZETA_NAME: LinearGaussianTransition(
                phi=bank.PHI,
                process_std=bank.PROCESS_STD,
                initial_std=bank.INITIAL_STD,
            )
        },
        represents={"beam_map": ("t_rx",)},
    )
    assert factorization.represents["beam_map"] == ("t_rx",)
