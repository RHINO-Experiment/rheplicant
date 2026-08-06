"""Section 9.5. Two nights that share a calibration solution are not independent.

Seven cases, and four of them must be ACCEPTED -- a guard that over-refuses is
its own bug, and every one of the four is a normal night in a real campaign: the
product was re-measured between nights, the product is modelled, the product
appears once, or the campaign declares no inputs at all.

The refusal is the mechanisation of section 1's "shared structure belongs in
theta". It is a guard rather than a report because the fault it names is the one
no report can see: measured on ``campaign_bank``, a deterministic error carried
by every night biases theta by 52.6 sigma at N = 640 while the per-epoch
chi-square, the split-half and the leave-one-out scores are all clean.

The plan matched these refusals on ``"conditionally independent"``. That phrase
is emitted by two other places -- ``BayesMemory._latents_ok``'s linked-latent
refusal and ``Factorization.represents``'s own docstring -- so a test matching it
would have passed against the wrong guard the moment this one was deleted and a
neighbouring one fired instead. ``"shares input product"`` is produced here and
nowhere else, which is what the sabotage below relies on.
"""

import jax.numpy as jnp
import pytest

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.compress import compress_linear
from rheplicant.inference.memory import BayesMemory
from tests.evidence import campaign_bank as camp


def _term(epoch_id, inputs=None):
    return compress_linear(
        design={"x": jnp.asarray(camp.DESIGN)},
        observed=jnp.asarray(camp.DESIGN @ camp.TRUTH),
        noise_std=camp.SIGMA,
        shapes={"x": (camp.N_THETA,)},
        epoch_id=epoch_id,
        inputs=inputs,
    )


def test_two_epochs_sharing_an_unmodelled_product_are_refused():
    memory = BayesMemory(camp.factorization())
    memory = memory.remember(_term("n0", {"beam_map": "sha:abc"}))
    with pytest.raises(StateValidationError, match="shares input product"):
        memory.remember(_term("n1", {"beam_map": "sha:abc"}))


def test_two_epochs_with_different_hashes_for_the_same_product_are_accepted():
    """The nearest legitimate case: the beam was re-measured between nights."""
    memory = BayesMemory(camp.factorization())
    memory = memory.remember(_term("n0", {"beam_map": "sha:abc"}))
    memory = memory.remember(_term("n1", {"beam_map": "sha:def"}))
    assert len(memory.archive) == 2


def test_two_epochs_sharing_a_represented_product_are_accepted():
    """The remedy the refusal names, exercised so the message is not a dead end."""
    memory = BayesMemory(camp.factorization(represents={"beam_map": ("x",)}))
    memory = memory.remember(_term("n0", {"beam_map": "sha:abc"}))
    memory = memory.remember(_term("n1", {"beam_map": "sha:abc"}))
    assert len(memory.archive) == 2


def test_a_campaign_that_declares_no_inputs_at_all_is_accepted():
    """The commonest legitimate case, and the one an over-eager guard breaks.

    ``inputs`` is optional and defaults to ``()``. Every epoch in every other
    test module in this directory is compressed without it, so a refusal keyed
    on "these two epochs carry the same inputs" rather than on a shared
    ``(product, hash)`` pair would refuse the whole existing suite.
    """
    memory = BayesMemory(camp.factorization())
    for epoch in range(4):
        memory = memory.remember(_term(f"n{epoch}"))
    assert len(memory.archive) == 4


def test_a_product_that_appears_once_is_accepted():
    """Declaring provenance must not itself be the offence.

    Two nights, two products, no pair shared -- which is what a campaign whose
    calibration is genuinely re-derived nightly looks like.
    """
    memory = BayesMemory(camp.factorization())
    memory = memory.remember(_term("n0", {"beam_map": "sha:abc"}))
    memory = memory.remember(_term("n1", {"flag_table": "sha:111"}))
    assert len(memory.archive) == 2


def test_a_deliberate_share_is_allowed_by_name():
    """D17's posture: legitimate double-counting is a choice made deliberately."""
    memory = BayesMemory(camp.factorization())
    memory = memory.remember(_term("n0", {"beam_map": "sha:abc"}))
    memory = memory.remember(
        _term("n1", {"beam_map": "sha:abc"}), shared_inputs=True
    )
    assert len(memory.archive) == 2


def test_the_message_names_the_product_and_both_epochs():
    memory = BayesMemory(camp.factorization())
    memory = memory.remember(_term("n0", {"flag_table": "sha:111"}))
    with pytest.raises(StateValidationError) as caught:
        memory.remember(_term("n1", {"flag_table": "sha:111"}))
    message = str(caught.value)
    assert "flag_table" in message and "n0" in message and "n1" in message


def test_the_clash_is_found_however_far_back_it_was():
    """Against the off-by-one that only compares with the previous epoch.

    A shared solution is shared with whichever night used it, and the campaign
    that motivates this is one solution serving hundreds. Checking only
    ``held[-1]`` passes every two-epoch test in this file.
    """
    memory = BayesMemory(camp.factorization())
    memory = memory.remember(_term("n0", {"beam_map": "sha:abc"}))
    for epoch in range(1, 5):
        memory = memory.remember(_term(f"n{epoch}", {"beam_map": f"sha:{epoch}"}))
    with pytest.raises(StateValidationError, match="shares input product"):
        memory.remember(_term("n9", {"beam_map": "sha:abc"}))


def test_a_chain_memory_refuses_the_same_way():
    """The rule is shared, not reimplemented -- see `reject_bad_term`."""
    from rheplicant.inference.chain import ChainMemory, LinearGaussianTransition
    from tests.evidence import chain_bank as bank

    transition = LinearGaussianTransition(
        phi=bank.PHI, process_std=bank.PROCESS_STD, initial_std=bank.INITIAL_STD
    )
    memory = ChainMemory(bank.factorization(transition))
    a, c, d = bank.design()

    def linked(epoch_id, inputs):
        return compress_linear(
            design={
                "t_rx": a[0][:, :1],
                "gain_slope": a[0][:, 1:],
                bank.ZETA_NAME: c[0],
            },
            observed=jnp.asarray(d[0]),
            noise_std=bank.SIGMA,
            shapes={"t_rx": (), "gain_slope": (), bank.ZETA_NAME: ()},
            epoch_id=epoch_id,
            inputs=inputs,
        )

    memory = memory.remember(linked("e0", {"beam_map": "sha:abc"}))
    with pytest.raises(StateValidationError, match="shares input product"):
        memory.remember(linked("e1", {"beam_map": "sha:abc"}))


def test_a_chain_memory_accepts_a_deliberate_share_too():
    """Both accumulators take the same escape hatch, or the rule has a way round it."""
    from rheplicant.inference.chain import ChainMemory, LinearGaussianTransition
    from tests.evidence import chain_bank as bank

    transition = LinearGaussianTransition(
        phi=bank.PHI, process_std=bank.PROCESS_STD, initial_std=bank.INITIAL_STD
    )
    memory = ChainMemory(bank.factorization(transition))
    a, c, d = bank.design()

    def linked(epoch_id, inputs):
        return compress_linear(
            design={
                "t_rx": a[0][:, :1],
                "gain_slope": a[0][:, 1:],
                bank.ZETA_NAME: c[0],
            },
            observed=jnp.asarray(d[0]),
            noise_std=bank.SIGMA,
            shapes={"t_rx": (), "gain_slope": (), bank.ZETA_NAME: ()},
            epoch_id=epoch_id,
            inputs=inputs,
        )

    memory = memory.remember(linked("e0", {"beam_map": "sha:abc"}))
    memory = memory.remember(linked("e1", {"beam_map": "sha:abc"}), shared_inputs=True)
    assert memory.epoch_ids == ("e0", "e1")


def test_provenance_survives_a_save_and_a_load(tmp_path):
    """Section 10: a static field equinox would take from the template.

    A reloaded memory that has forgotten which nights shared a solution will
    cheerfully sum them, with no error and no warning, and the raw data is gone.
    """
    from rheplicant.inference.archive import load_memory, save_memory

    memory = BayesMemory(camp.factorization())
    memory = memory.remember(_term("n0", {"beam_map": "sha:abc"}))
    memory = memory.remember(_term("n1", {"beam_map": "sha:def"}))
    save_memory(memory, tmp_path / "campaign.eqx")
    reloaded = load_memory(tmp_path / "campaign.eqx", camp.factorization())
    assert reloaded.archive[0].inputs == (("beam_map", "sha:abc"),)
    assert reloaded.archive[1].inputs == (("beam_map", "sha:def"),)
    with pytest.raises(StateValidationError, match="shares input product"):
        reloaded.remember(_term("n2", {"beam_map": "sha:abc"}))
