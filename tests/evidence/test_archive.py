import json

import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.archive import load_memory, save_memory
from rheplicant.inference.compressed import QuadraticLikelihood
from rheplicant.inference.sqrtinfo import SqrtInfo
from tests.evidence.test_memory import _factorization


def _memory_with_non_default_provenance():
    from rheplicant.inference.memory import BayesMemory

    memory = BayesMemory(_factorization())
    term = QuadraticLikelihood(
        info=SqrtInfo(
            factor=jnp.array([[1.5, 0.25], [0.0, 0.75]]),
            target=jnp.array([0.5, -0.25]),
            offset=jnp.array(-3.25),
            names=("depth", "width"), shapes=((), ()),
        ),
        epoch_id="night-042", n_observed=777,
        exact=False, support={"depth": (-2.0, 2.0), "width": (-1.0, 3.0)},
        include_logdet=False, noise_frozen_at="gls",
        # Format 3's five. Every one non-default, and the two static ones
        # deliberately unguessable from anything else in this term: a template
        # built from a convention rather than from the manifest would reproduce
        # `()` and `()` here without erroring.
        residual_chi2=jnp.array(7.5),
        template_projections=jnp.array([1.25, -0.5]),
        residual_dof=13,
        template_names=("gain_ripple", "ground_pickup"),
        inputs=(("beam_model", "sha256:b3ee"), ("cal_solution", "sha256:0f17")),
    )
    return memory.remember(term)


def test_every_static_field_survives_a_round_trip(tmp_path):
    """The pin for eqx.tree_serialise_leaves reverting statics to the template."""
    original = _memory_with_non_default_provenance()
    path = tmp_path / "campaign.rhep"
    save_memory(original, path)
    restored = load_memory(path, original.factorization)

    before, after = original.archive[0], restored.archive[0]
    assert after.epoch_id == before.epoch_id == "night-042"
    assert after.n_observed == before.n_observed == 777
    assert after.exact is False
    assert after.support == {"depth": (-2.0, 2.0), "width": (-1.0, 3.0)}
    assert after.include_logdet is False
    assert after.noise_frozen_at == "gls"
    assert after.prior_share == (0, 1)
    # Format 3. `residual_dof`, `template_names` and `inputs` are static, so
    # they come from the manifest; without their entries there they would come
    # back 0, () and () -- a campaign that had forgotten which nights shared a
    # calibration solution, and would cheerfully sum them.
    assert after.residual_dof == before.residual_dof == 13
    assert after.template_names == ("gain_ripple", "ground_pickup")
    assert after.inputs == (
        ("beam_model", "sha256:b3ee"),
        ("cal_solution", "sha256:0f17"),
    )
    assert float(after.residual_chi2) == 7.5
    np.testing.assert_array_equal(
        np.asarray(after.template_projections), np.asarray([1.25, -0.5])
    )
    np.testing.assert_array_equal(
        np.asarray(after.info.factor), np.asarray(before.info.factor)
    )
    assert after.info.factor.dtype == jnp.float64


def test_the_manifest_records_the_writers_x64_state(tmp_path):
    path = tmp_path / "campaign.rhep"
    save_memory(_memory_with_non_default_provenance(), path)
    manifest = json.loads((path.with_suffix(".json")).read_text())
    assert manifest["jax_enable_x64"] is True
    assert manifest["terms"][0]["epoch_id"] == "night-042"


def test_a_manifest_claiming_float32_is_refused(tmp_path):
    path = tmp_path / "campaign.rhep"
    memory = _memory_with_non_default_provenance()
    save_memory(memory, path)
    manifest_path = path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text())
    manifest["terms"][0]["dtype"] = "float32"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(StateValidationError, match="dtype"):
        load_memory(path, memory.factorization)


def test_a_manifest_from_another_format_version_is_refused(tmp_path):
    """The reader must not guess at a layout it does not know.

    Every other refusal in this file is about a mismatch the manifest can
    describe. This one is about a manifest whose *own* vocabulary may differ:
    a future writer that renames a key or stops writing one would leave this
    reader raising ``KeyError`` deep inside the template construction, which
    reads as a corrupt file rather than as a version it cannot handle. The
    version is therefore checked before anything else is read out of it.
    """
    path = tmp_path / "campaign.rhep"
    memory = _memory_with_non_default_provenance()
    save_memory(memory, path)
    manifest_path = path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["format_version"] == 3, "the version this code writes"
    manifest["format_version"] = 4
    del manifest["terms"][0]["noise_frozen_at"]  # what a later layout may drop
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(StateValidationError, match="format version"):
        load_memory(path, memory.factorization)


def test_a_memory_whose_latents_changed_is_refused(tmp_path):
    from rheplicant.inference import Bind, Latent, ParameterSpace
    from rheplicant.inference.factorize import Factorization
    from tests.evidence.test_memory import _Normal

    memory = _memory_with_non_default_provenance()
    path = tmp_path / "campaign.rhep"
    save_memory(memory, path)

    latents = (
        Latent("depth", init=-0.5, prior=_Normal(0.0, 1.0)),
        Latent("breadth", init=1.0, prior=_Normal(0.0, 2.0)),
    )
    other = Factorization(
        ParameterSpace(
            latents=latents,
            bindings=tuple(
                Bind(lat.name, into=lambda p, n=lat.name: getattr(p, n))
                for lat in latents
            ),
        )
    )
    with pytest.raises(StateValidationError, match="latent"):
        load_memory(path, other)


def test_the_restored_memory_gives_the_same_log_posterior(tmp_path):
    original = _memory_with_non_default_provenance()
    path = tmp_path / "campaign.rhep"
    save_memory(original, path)
    restored = load_memory(path, original.factorization)
    probe = {"depth": jnp.array(0.3), "width": jnp.array(-1.1)}
    assert float(restored.log_posterior(probe)) == pytest.approx(
        float(original.log_posterior(probe)), abs=1e-12
    )


class TestLoadReRunsTheRefusalsRememberEnforces:
    """The manifest is an editable text file, so it is a second way in.

    ``BayesMemory.remember`` refuses a repeated epoch, a mixed estimator, a
    tempered term and a shared input product. ``load_memory`` calls
    ``BayesMemory(archive=...)`` directly, which validates nothing -- so before
    this guard, a hand-edited manifest loaded silently,
    ``audit()["estimator"]`` reported one estimator for an archive holding two
    (it reads ``archive[0]``), and ``remember`` then admitted further terms of
    whichever estimator sat at index 0.

    The fourth is section 9.5's, and it lives in
    :class:`TestTheProvenanceRuleIsReRunOnLoadToo` below because it needs a
    two-term archive whose binary is as real as its manifest.
    """

    def _saved(self, tmp_path, mutate):
        path = tmp_path / "campaign.rhep"
        save_memory(_memory_with_non_default_provenance(), path)
        manifest_path = path.with_suffix(".json")
        manifest = json.loads(manifest_path.read_text())
        mutate(manifest)
        manifest_path.write_text(json.dumps(manifest))
        return path

    def test_a_repeated_epoch_id_is_refused(self, tmp_path):
        def duplicate(manifest):
            manifest["terms"].append(dict(manifest["terms"][0]))

        path = self._saved(tmp_path, duplicate)
        with pytest.raises(StateValidationError, match="more than once"):
            load_memory(path, _factorization())

    def test_a_mixed_estimator_archive_is_refused(self, tmp_path):
        def mix(manifest):
            other = dict(manifest["terms"][0])
            other["epoch_id"] = "night-043"
            other["include_logdet"] = not other["include_logdet"]
            manifest["terms"].append(other)

        path = self._saved(tmp_path, mix)
        with pytest.raises(StateValidationError, match="mixes estimators"):
            load_memory(path, _factorization())

    def test_a_tempered_term_is_refused(self, tmp_path):
        def temper(manifest):
            manifest["terms"][0]["prior_share"] = [1, 300]

        path = self._saved(tmp_path, temper)
        with pytest.raises(StateValidationError, match="prior_share"):
            load_memory(path, _factorization())

    def test_the_refusal_happens_before_any_array_is_read(self, tmp_path):
        """Deleting the binary must not change which error comes out.

        The manifest is checked first precisely so that a bad archive is
        refused on its own description rather than on whatever the reader
        happens to trip over in the payload.
        """

        def duplicate(manifest):
            manifest["terms"].append(dict(manifest["terms"][0]))

        path = self._saved(tmp_path, duplicate)
        path.unlink()
        with pytest.raises(StateValidationError, match="more than once"):
            load_memory(path, _factorization())


class TestTheProvenanceRuleIsReRunOnLoadToo:
    """Section 9.5, through the door this module's own docstring documents.

    ``remember`` refuses two epochs that share an input-product hash unless the
    product is represented among the global latents, because a shared
    calibration solution is a shared error with no variance at all: per-epoch
    chi-square is right, split-half agrees, leave-one-out agrees, and the answer
    is wrong. ``load_memory`` re-ran three of ``remember``'s refusals and not
    that one.

    Reproduced with a **one-character** manifest edit and no ``shared_inputs=``
    anywhere: written, the two terms read
    ``[[['beam_map', 'sha:abc']], [['beam_map', 'sha:def']]]``; edited to
    ``[[['beam_map', 'sha:abc']], [['beam_map', 'sha:abc']]]``, ``load_memory``
    ACCEPTED both, while ``remember`` on the same pair said "Epoch 'n1' shares
    input product 'beam_map' (hash 'sha:abc') with ['n0']" and the *duplicate*
    rule fired on the very same edited file. Concatenating two runs' manifests
    reaches the same state with no editing at all, which is why this is not a
    tampering guard.

    Every archive here is genuinely two terms in the binary as well as in the
    manifest, so an accepted case really loads rather than dying on a short
    file, and a refused case is refused on the archive's meaning rather than on
    its damage.
    """

    def _pair(self, tmp_path, second_hash, share=False):
        """A real two-night archive; ``share`` retypes the second hash on disk."""
        from rheplicant.inference.memory import BayesMemory

        memory = BayesMemory(_factorization())
        for epoch_id, digest in (("n0", "sha:abc"), ("n1", second_hash)):
            memory = memory.remember(
                QuadraticLikelihood(
                    info=SqrtInfo(
                        factor=jnp.array([[1.5, 0.25], [0.0, 0.75]]),
                        target=jnp.array([0.5, -0.25]),
                        offset=jnp.array(-3.25),
                        names=("depth", "width"), shapes=((), ()),
                    ),
                    epoch_id=epoch_id, n_observed=64,
                    inputs=(("beam_map", digest),),
                )
            )
        path = tmp_path / "campaign.rhep"
        save_memory(memory, path)
        if share:
            manifest_path = path.with_suffix(".json")
            manifest = json.loads(manifest_path.read_text())
            manifest["terms"][1]["inputs"] = manifest["terms"][0]["inputs"]
            manifest_path.write_text(json.dumps(manifest))
        return path

    def _represents(self, product):
        from rheplicant.inference.factorize import Factorization

        return Factorization(_factorization().space, represents={product: ("depth",)})

    def test_a_shared_unmodelled_product_is_refused(self, tmp_path):
        path = self._pair(tmp_path, "sha:def", share=True)
        with pytest.raises(StateValidationError, match="share input product"):
            load_memory(path, _factorization())

    def test_the_message_names_the_product_the_hash_and_both_epochs(self, tmp_path):
        path = self._pair(tmp_path, "sha:def", share=True)
        with pytest.raises(StateValidationError) as caught:
            load_memory(path, _factorization())
        message = str(caught.value)
        assert "beam_map" in message
        assert "sha:abc" in message
        assert "n0" in message and "n1" in message

    def test_a_re_measured_product_still_loads(self, tmp_path):
        """The nearest legitimate case, and it must stay legitimate.

        Same product, different hash: the beam was re-measured between nights,
        the two errors are independent draws, and summing them is exactly
        right. Matching on the product NAME alone would refuse the normal
        campaign, so the comparison is on the ``(product, hash)`` pair.
        """
        path = self._pair(tmp_path, "sha:def")
        restored = load_memory(path, _factorization())
        assert [term.inputs for term in restored.archive] == [
            (("beam_map", "sha:abc"),),
            (("beam_map", "sha:def"),),
        ]

    def test_a_shared_product_that_is_modelled_still_loads(self, tmp_path):
        """The remedy the message names, exercised so it is not a dead end.

        A product carried as a global latent is integrated with the rest of
        theta, so sharing its hash is no longer a claim of independence about
        something unmodelled. The refusal has to read the *supplied*
        factorization to know that, which is why it takes ``represents`` rather
        than deciding on the manifest alone.
        """
        path = self._pair(tmp_path, "sha:def", share=True)
        restored = load_memory(path, self._represents("beam_map"))
        assert len(restored.archive) == 2

    def test_representing_a_different_product_does_not_excuse_this_one(self, tmp_path):
        """`represents` is read by key, so a near-miss must not open the gate."""
        path = self._pair(tmp_path, "sha:def", share=True)
        with pytest.raises(StateValidationError, match="share input product"):
            load_memory(path, self._represents("cal_solution"))


class TestTheManifestIsWrittenLastBecauseItIsTheCommit:
    """Its presence is what says a readable archive exists.

    Written first, a failing ``tree_serialise_leaves`` left a manifest
    describing a file that was never created, and ``load_memory`` died on a raw
    ``FileNotFoundError`` from equinox rather than on anything this module says.
    """

    def test_a_failed_serialisation_leaves_no_manifest_behind(self, tmp_path, monkeypatch):
        from rheplicant.inference import archive as archive_module

        def explode(path, tree):
            raise OSError("simulated: disk full")

        # Patched through `archive_module.eqx` rather than by importing equinox
        # here, so the substitution is the name this module actually calls --
        # patching a separately-imported alias would leave save_memory bound to
        # the real function and the test would pass while proving nothing.
        monkeypatch.setattr(archive_module.eqx, "tree_serialise_leaves", explode)
        path = tmp_path / "campaign.rhep"
        with pytest.raises(OSError, match="disk full"):
            save_memory(_memory_with_non_default_provenance(), path)
        assert not path.with_suffix(".json").exists(), (
            "a manifest survived a failed save, so it describes an archive that "
            "does not exist"
        )

    def test_a_binary_without_its_manifest_is_refused_by_name(self, tmp_path):
        path = tmp_path / "campaign.rhep"
        save_memory(_memory_with_non_default_provenance(), path)
        path.with_suffix(".json").unlink()
        with pytest.raises(StateValidationError, match="No manifest"):
            load_memory(path, _factorization())

    def test_a_complete_save_still_round_trips(self, tmp_path):
        """Guard the guard: reordering the writes must not break the happy path."""
        original = _memory_with_non_default_provenance()
        path = tmp_path / "campaign.rhep"
        save_memory(original, path)
        assert path.exists() and path.with_suffix(".json").exists()
        restored = load_memory(path, original.factorization)
        assert restored.archive[0].noise_frozen_at == "gls"


class TestTheFormatDescribesABagAndSaysSoWhenHandedSomethingElse:
    """`save_memory` refused a foreign *term* by name and a foreign *memory* by crash.

    Handed a `ChainMemory` it built the manifest as far as
    `int(memory.accumulated.factor.shape[0])` and raised
    `AttributeError: 'ChainMemory' object has no attribute 'accumulated'` --
    which names an implementation detail of the thing it was not given, says
    nothing about what this format is, and offers no remedy. Every other
    unsupported input on this path gets a sentence.

    A chain is not a bag with a longer archive: `accumulated_rows` is a bag's
    running QR, which a chain deliberately does not have, and the transition a
    chain reads is a live object that may hold a Python callable
    (`HyperTransition`) with no textual form for a manifest to record. So this
    is a refusal rather than a to-do: `load_memory` returns a `BayesMemory`,
    and there is no reading of this format under which it could return the
    other one.
    """

    def _chain(self):
        import jax.numpy as jnp

        from rheplicant.inference.chain import ChainMemory, LinearGaussianTransition
        from rheplicant.inference.compress import compress_linear
        from tests.evidence import chain_bank as bank

        transition = LinearGaussianTransition(
            phi=bank.PHI,
            process_std=bank.PROCESS_STD,
            initial_std=bank.INITIAL_STD,
            initial_mean=bank.INITIAL_MEAN,
        )
        design, drift, data = bank.design(0)
        memory = ChainMemory(bank.factorization(transition))
        for epoch in range(2):
            memory = memory.remember(
                compress_linear(
                    design={
                        "t_rx": design[epoch][:, :1],
                        "gain_slope": design[epoch][:, 1:],
                        bank.ZETA_NAME: drift[epoch],
                    },
                    observed=jnp.asarray(data[epoch]),
                    noise_std=bank.SIGMA,
                    shapes={"t_rx": (), "gain_slope": (), bank.ZETA_NAME: ()},
                    epoch_id=f"e{epoch}",
                )
            )
        return memory

    def test_a_chain_memory_is_refused_by_name(self, tmp_path):
        with pytest.raises(StateValidationError, match="ChainMemory") as caught:
            save_memory(self._chain(), tmp_path / "chain.rhep")
        message = str(caught.value)
        assert "BayesMemory" in message
        # A remedy, as the foreign-term refusal has one.
        assert "process that built it" in message

    def test_nothing_is_written_when_the_memory_is_refused(self, tmp_path):
        """The refusal is before the binary, as the foreign-term one is."""
        path = tmp_path / "chain.rhep"
        with pytest.raises(StateValidationError):
            save_memory(self._chain(), path)
        assert not path.exists()
        assert not path.with_suffix(".json").exists()

    def test_a_bag_still_saves(self, tmp_path):
        """Guard the guard."""
        path = tmp_path / "bag.rhep"
        save_memory(_memory_with_non_default_provenance(), path)
        assert path.exists() and path.with_suffix(".json").exists()
