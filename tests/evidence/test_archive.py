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
    assert manifest["format_version"] == 2, "the version this code writes"
    manifest["format_version"] = 3
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

    ``BayesMemory.remember`` refuses a repeated epoch, a mixed estimator and a
    tempered term. ``load_memory`` calls ``BayesMemory(archive=...)`` directly,
    which validates nothing -- so before this guard, a hand-edited manifest
    loaded silently, ``audit()["estimator"]`` reported one estimator for an
    archive holding two (it reads ``archive[0]``), and ``remember`` then
    admitted further terms of whichever estimator sat at index 0.
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
