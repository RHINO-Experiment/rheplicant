"""Ordered accumulation, and the two refusals that keep the types apart.

A bag summed a chain's terms would treat six correlated nights as six
independent ones, which is condition C1b's failure exactly: one physical
fluctuation marginalised N times against independent priors, injecting
information that is not there. The posterior comes back narrower and centred,
so nothing downstream notices.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.chain import (
    ChainMemory,
    HyperTransition,
    LinearGaussianTransition,
    ornstein_uhlenbeck,
)
from rheplicant.inference.compress import compress_linear
from rheplicant.inference.factorize import Factorization
from rheplicant.inference.memory import BayesMemory
from rheplicant.inference.parameters import Bind, Latent, ParameterSpace
from tests.evidence import chain_bank as bank


def _transition(tau=None):
    if tau is not None:
        return ornstein_uhlenbeck(tau=tau, sigma=1.0)
    return LinearGaussianTransition(
        phi=bank.PHI,
        process_std=bank.PROCESS_STD,
        initial_std=bank.INITIAL_STD,
        initial_mean=bank.INITIAL_MEAN,
    )


def _terms(seed=0):
    A, C, d = bank.design(seed)
    return tuple(
        compress_linear(
            design={
                "t_rx": A[e][:, :1],
                "gain_slope": A[e][:, 1:],
                bank.ZETA_NAME: C[e],
            },
            observed=jnp.asarray(d[e]),
            noise_std=bank.SIGMA,
            shapes={"t_rx": (), "gain_slope": (), bank.ZETA_NAME: ()},
            epoch_id=f"e{e}",
        )
        for e in range(bank.N_EPOCHS)
    )


def _memory(transition=None, terms=None):
    memory = ChainMemory(bank.factorization(transition or _transition()))
    for term in terms or _terms():
        memory = memory.remember(term)
    return memory


def _values(probe):
    return {"t_rx": jnp.asarray(probe[0]), "gain_slope": jnp.asarray(probe[1])}


@pytest.mark.parametrize("probe", bank.PROBES)
def test_streaming_equals_the_batch_marginal(probe):
    """Validation item 12.7, at a fixed transition.

    Measured on the run that shipped this: 5.7e-14, 1.1e-13, 9.1e-13 and
    9.1e-13 nats from the dense oracle at the four probes, against a band of
    1e-9.
    """
    assert float(_memory().log_likelihood(_values(probe))) == pytest.approx(
        bank.oracle(probe), abs=1e-9
    )


@pytest.mark.parametrize("tau", [0.5, 3.0, 12.0])
def test_streaming_equals_batch_with_an_inferred_correlation_time(tau):
    """12.7 proper: the transition is resolved at evaluation, not at compression."""
    memory = _memory(_transition(tau=tau))
    transition = ornstein_uhlenbeck(tau=tau, sigma=1.0)
    expected = bank.oracle(
        bank.PROBES[1],
        phi=float(transition.phi[0, 0]),
        process_var=float(transition.process_std[0]) ** 2,
        initial_var=1.0,
    )
    assert float(memory.log_likelihood(_values(bank.PROBES[1]))) == pytest.approx(
        expected, abs=1e-8
    )


@pytest.mark.parametrize("tau", [0.5, 3.0, 12.0])
def test_a_hyper_chain_reads_the_correlation_time_off_the_sampled_value(tau):
    """The half a fixed OU cannot show: the SAME memory, three taus.

    ``test_streaming_equals_batch_with_an_inferred_correlation_time`` builds a
    new memory per tau, so a memory that had pinned the transition at
    ``remember`` time would still pass it. Here one memory holds a
    :class:`HyperTransition` and the correlation time arrives with the latent
    values, which is what a NUTS step actually does -- ``log_tau`` is read off
    ``values`` on every call, and each of the three answers is the dense
    oracle's for that tau.
    """
    memory = _memory(
        HyperTransition(
            build=lambda values: ornstein_uhlenbeck(
                tau=jnp.exp(values["gain_slope"]), sigma=1.0
            ),
            hyper=("gain_slope",),
            width=1,
        )
    )
    transition = ornstein_uhlenbeck(tau=tau, sigma=1.0)
    # `gain_slope` doubles as log(tau) here: the fixture declares two globals and
    # this needs one of them to reach the builder, which is what `hyper` names.
    probe = [bank.PROBES[1][0], float(np.log(tau))]
    expected = bank.oracle(
        probe,
        phi=float(transition.phi[0, 0]),
        process_var=float(transition.process_std[0]) ** 2,
        initial_var=1.0,
    )
    assert float(memory.log_likelihood(_values(probe))) == pytest.approx(
        expected, abs=1e-8
    )


def test_the_posterior_applies_the_prior_exactly_once():
    memory = _memory()
    values = _values(bank.PROBES[1])
    prior = sum(
        float(jnp.sum(p.log_prob(values[name])))
        for name, p in memory.factorization.global_priors.items()
    )
    assert float(memory.log_posterior(values)) - float(
        memory.log_likelihood(values)
    ) == pytest.approx(prior, abs=1e-12)


def test_a_chain_is_not_exchangeable_and_the_gap_is_large():
    """The claim `a chain is not a bag` makes, in nats.

    A bag's `remember` is order-invariant to roundoff -- Plan A pins that. Here
    the same two terms in the other order give a different likelihood, because
    epoch 3's drift is correlated with epoch 2's and not with epoch 5's.
    Measured on this fixture: 0.0752 nats, against a band of 1e-3 -- and
    against the recursion's own 9.1e-13 disagreement with the dense oracle,
    which is what makes 0.075 a signal rather than a small number.
    """
    terms = list(_terms())
    forward = _memory(terms=terms)
    swapped = _memory(terms=terms[:2] + [terms[3], terms[2]] + terms[4:])
    gap = abs(
        float(forward.log_likelihood(_values(bank.PROBES[1])))
        - float(swapped.log_likelihood(_values(bank.PROBES[1])))
    )
    assert gap > 1e-3, (
        "swapping two epochs changed nothing, so the chain is being accumulated "
        "as a bag -- check that ChainMemory appends rather than combining"
    )


def test_a_bag_refuses_a_term_carrying_a_linked_latent():
    memory = BayesMemory(bank.factorization(_transition()))
    with pytest.raises(StateValidationError, match="ChainMemory"):
        memory.remember(_terms()[0])


def test_a_bag_still_takes_a_term_over_its_own_globals():
    """The nearest legitimate case: the refusal must not swallow a plain term.

    A guard that over-refuses is its own bug, so the same factorization -- which
    does declare a linked latent -- accumulates a global-only epoch as before.
    """
    A, _, d = bank.design()
    term = compress_linear(
        design={"t_rx": A[0][:, :1], "gain_slope": A[0][:, 1:]},
        observed=jnp.asarray(d[0]),
        noise_std=bank.SIGMA,
        shapes={"t_rx": (), "gain_slope": ()},
        epoch_id="global_only",
    )
    memory = BayesMemory(bank.factorization(_transition())).remember(term)
    assert memory.archive == (term,)


def test_a_chain_refuses_a_term_that_carries_no_linked_latent():
    """The nearest legitimate mistake: a global-only epoch handed to the chain."""
    A, _, d = bank.design()
    term = compress_linear(
        design={"t_rx": A[0][:, :1], "gain_slope": A[0][:, 1:]},
        observed=jnp.asarray(d[0]),
        noise_std=bank.SIGMA,
        shapes={"t_rx": (), "gain_slope": ()},
        epoch_id="global_only",
    )
    memory = ChainMemory(bank.factorization(_transition()))
    with pytest.raises(StateValidationError, match=bank.ZETA_NAME):
        memory.remember(term)


def test_a_chain_refuses_a_repeated_epoch_and_takes_duplicate_true():
    memory = _memory()
    with pytest.raises(StateValidationError, match="already in this memory"):
        memory.remember(_terms()[0])
    assert len(memory.remember(_terms()[0], duplicate=True).epoch_ids) == (
        bank.N_EPOCHS + 1
    )


def test_a_chain_refuses_two_estimators():
    terms = list(_terms())
    A, C, d = bank.design()
    gls = compress_linear(
        design={
            "t_rx": A[0][:, :1],
            "gain_slope": A[0][:, 1:],
            bank.ZETA_NAME: C[0],
        },
        observed=jnp.asarray(d[0]),
        noise_std=bank.SIGMA,
        shapes={"t_rx": (), "gain_slope": (), bank.ZETA_NAME: ()},
        epoch_id="gls",
    )
    gls = type(gls)(
        info=gls.info,
        epoch_id="gls",
        n_observed=gls.n_observed,
        include_logdet=False,
    )
    with pytest.raises(StateValidationError, match="different estimators"):
        _memory(terms=terms).remember(gls)


def test_the_epoch_ids_are_ordered_and_the_stack_grows_by_one_a_night():
    memory = _memory()
    assert memory.epoch_ids == tuple(f"e{e}" for e in range(bank.N_EPOCHS))
    assert memory.stacked[0].shape == (bank.N_EPOCHS, bank.WIDTH, bank.WIDTH)


def test_the_stacked_blocks_are_the_fixtures_own_and_carry_its_offsets():
    """`remember` must pad by the accumulator's QR, not by `jnp.pad`.

    Zero-padding a `(rank, w)` factor to `(w, w)` gives the same factor and a
    *different offset* -- the fold's corner is the largest of the six constants,
    +45.95 nats over six epochs -- so this compares against `chain_bank.blocks`,
    which is built by `combine(null, info)`, offsets included.
    """
    factors, targets, offsets = _memory().stacked
    expected = bank.stacked()
    np.testing.assert_allclose(np.asarray(factors), np.asarray(expected[0]))
    np.testing.assert_allclose(np.asarray(targets), np.asarray(expected[1]))
    np.testing.assert_allclose(np.asarray(offsets), np.asarray(expected[2]))


def test_a_term_whose_columns_arrive_in_another_order_is_permuted_not_relabelled():
    """`compress_linear` names the columns in the caller's dict order, not ours.

    An epoch built with the drift's design block first is a quadratic form in a
    differently ordered vector: the same information, and unusable as it stands.
    It has to be permuted. Relabelling it instead -- writing the memory's names
    onto the stored factor -- is the silent version, and it comes back finite,
    well-conditioned, and about a different model. So the campaign built this
    way is compared against the dense oracle, which knows nothing about either
    ordering.
    """
    A, C, d = bank.design()
    terms = tuple(
        compress_linear(
            design={
                bank.ZETA_NAME: C[e],
                "gain_slope": A[e][:, 1:],
                "t_rx": A[e][:, :1],
            },
            observed=jnp.asarray(d[e]),
            noise_std=bank.SIGMA,
            shapes={bank.ZETA_NAME: (), "gain_slope": (), "t_rx": ()},
            epoch_id=f"e{e}",
        )
        for e in range(bank.N_EPOCHS)
    )
    assert terms[0].info.names == (bank.ZETA_NAME, "gain_slope", "t_rx")
    memory = _memory(terms=terms)
    assert memory.column_order == ("t_rx", "gain_slope", bank.ZETA_NAME)
    assert float(memory.log_likelihood(_values(bank.PROBES[1]))) == pytest.approx(
        bank.oracle(bank.PROBES[1]), abs=1e-9
    )


def test_a_memory_carrying_two_chains_is_refused():
    """Two chains are two memories: one joint transition is a different model."""
    space = bank.space()
    second = Latent(
        "gain_drift",
        init=jnp.asarray(0.0),
        prior=space.latents[-1].prior,
        scope="linked",
    )
    factorization = Factorization(
        ParameterSpace(
            latents=space.latents + (second,),
            bindings=space.bindings
            + (Bind("gain_drift", into=lambda p: p.gain_drift),),
        ),
        linked={bank.ZETA_NAME: _transition(), "gain_drift": _transition()},
    )
    with pytest.raises(StateValidationError, match="exactly one linked latent"):
        ChainMemory(factorization)


def test_a_jitted_chain_density_retraces_once_per_night_and_not_per_call():
    """Deviation 12, measured rather than asserted.

    The bag's accumulator has a fixed treedef and never retraces. A chain's
    stack has a growing leading axis, so it retraces once per `remember` -- and
    then not at all for the whole of a NUTS run, where N is fixed. Both halves
    are pinned, because only the second one matters and only the first one is
    surprising.
    """
    traces = []

    @jax.jit
    def density(stacked, values):
        traces.append(1)
        return stacked[2].sum() + values["t_rx"]

    memory = ChainMemory(bank.factorization(_transition()))
    for term in _terms():
        memory = memory.remember(term)
        density(memory.stacked, _values(bank.PROBES[0]))
    assert len(traces) == bank.N_EPOCHS
    for _ in range(5):
        density(memory.stacked, _values(bank.PROBES[1]))
    assert len(traces) == bank.N_EPOCHS


def test_the_fisher_is_the_marginals_and_is_named_in_flatten_order():
    """`gain_slope` sorts before `t_rx`; the columns were built the other way."""
    fisher = _memory().fisher(at=_values(bank.PROBES[1]))
    assert fisher.names == ("gain_slope", "t_rx")
    assert fisher.kind == "fisher"
    matrix = np.asarray(fisher.matrix)
    assert np.all(np.linalg.eigvalsh(matrix) > -1e-10)
    # ...and it is genuinely permuted: the two diagonal entries differ, so an
    # unpermuted matrix would be caught.
    assert abs(matrix[0, 0] - matrix[1, 1]) > 1e-6
